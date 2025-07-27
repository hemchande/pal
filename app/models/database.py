from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import logging

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/wildfire_system")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.wildfire_system

# Collections
fire_incidents_collection = db.fire_incidents
weather_data_collection = db.weather_data
human_activity_collection = db.human_activity
risk_assessments_collection = db.risk_assessments
resources_collection = db.resources
alerts_collection = db.alerts
recommendations_collection = db.recommendations

# Create indexes for better performance
async def create_indexes():
    """Create MongoDB indexes for optimal query performance"""
    try:
        # Fire incidents indexes
        await fire_incidents_collection.create_index([("incident_id", ASCENDING)], unique=True)
        await fire_incidents_collection.create_index([("location", "2dsphere")])
        await fire_incidents_collection.create_index([("status", ASCENDING)])
        await fire_incidents_collection.create_index([("discovery_date", DESCENDING)])
        
        # Weather data indexes
        await weather_data_collection.create_index([("location", "2dsphere")])
        await weather_data_collection.create_index([("timestamp", DESCENDING)])
        
        # Human activity indexes
        await human_activity_collection.create_index([("location", "2dsphere")])
        await human_activity_collection.create_index([("timestamp", DESCENDING)])
        await human_activity_collection.create_index([("activity_type", ASCENDING)])
        
        # Risk assessments indexes
        await risk_assessments_collection.create_index([("location", "2dsphere")])
        await risk_assessments_collection.create_index([("timestamp", DESCENDING)])
        await risk_assessments_collection.create_index([("risk_score", DESCENDING)])
        
        # Resources indexes
        await resources_collection.create_index([("location", "2dsphere")])
        await resources_collection.create_index([("resource_type", ASCENDING)])
        await resources_collection.create_index([("status", ASCENDING)])
        
        # Alerts indexes
        await alerts_collection.create_index([("location", "2dsphere")])
        await alerts_collection.create_index([("created_at", DESCENDING)])
        await alerts_collection.create_index([("status", ASCENDING)])
        
        # Recommendations indexes
        await recommendations_collection.create_index([("created_at", DESCENDING)])
        await recommendations_collection.create_index([("priority", DESCENDING)])
        await recommendations_collection.create_index([("target_audience", ASCENDING)])
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

# Pydantic Models
class Location(BaseModel):
    type: str = "Point"
    coordinates: List[float]  # [longitude, latitude]

class FireStatus(str, Enum):
    ACTIVE = "active"
    CONTAINED = "contained"
    OUT = "out"

class FireIncident(BaseModel):
    incident_id: str
    location: Location
    discovery_date: datetime
    containment_date: Optional[datetime] = None
    acres_burned: Optional[float] = None
    cause: Optional[str] = None
    status: FireStatus
    severity_level: int = Field(ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class WeatherData(BaseModel):
    location: Location
    timestamp: datetime
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    precipitation: Optional[float] = None
    pressure: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class HumanActivity(BaseModel):
    location: Location
    timestamp: datetime
    activity_type: str  # hiking, camping, construction, etc.
    intensity: float = Field(ge=0.0, le=1.0)
    source: str  # safegraph, google_trends, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RiskAssessment(BaseModel):
    location: Location
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_factors: Optional[Dict[str, Any]] = None
    predicted_spread_direction: Optional[str] = None
    predicted_spread_speed: Optional[float] = None
    population_at_risk: Optional[int] = None
    evacuation_time_estimate: Optional[int] = None  # minutes
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    DEPLOYED = "deployed"
    MAINTENANCE = "maintenance"

class Resource(BaseModel):
    resource_type: str  # fire_truck, helicopter, crew, etc.
    location: Location
    status: ResourceStatus
    capacity: Optional[float] = None
    current_location: Optional[str] = None
    estimated_arrival_time: Optional[int] = None  # minutes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AlertType(str, Enum):
    EVACUATION = "evacuation"
    WARNING = "warning"
    INFO = "info"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class Alert(BaseModel):
    alert_type: AlertType
    location: Location
    radius_km: float
    message: str
    severity: AlertSeverity
    target_audience: str  # public, firefighters, officials
    status: AlertStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

class RecommendationType(str, Enum):
    DEPLOYMENT = "deployment"
    EVACUATION = "evacuation"
    PREVENTION = "prevention"

class RecommendationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"

class Recommendation(BaseModel):
    recommendation_type: RecommendationType
    target_audience: str  # firefighters, officials, public
    priority: int = Field(ge=1, le=5)
    description: str
    reasoning: Optional[str] = None
    estimated_impact: Optional[str] = None
    implementation_time: Optional[int] = None  # minutes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: RecommendationStatus = RecommendationStatus.PENDING

# Database operations
async def get_database():
    """Get database instance"""
    return db

async def close_database():
    """Close database connection"""
    client.close()

# Helper functions for geospatial queries
def create_location(lat: float, lon: float) -> Location:
    """Create a Location object from lat/lon coordinates"""
    return Location(coordinates=[lon, lat])  # MongoDB uses [longitude, latitude]

def create_geospatial_query(lat: float, lon: float, max_distance_meters: float = 10000):
    """Create a geospatial query for finding documents within a radius"""
    return {
        "location": {
            "$near": {
                "$geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "$maxDistance": max_distance_meters
            }
        }
    }

def create_geospatial_query_box(min_lat: float, max_lat: float, min_lon: float, max_lon: float):
    """Create a geospatial query for finding documents within a bounding box"""
    return {
        "location": {
            "$geoWithin": {
                "$box": [
                    [min_lon, min_lat],
                    [max_lon, max_lat]
                ]
            }
        }
    } 