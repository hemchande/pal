from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta

from app.models.database import create_indexes, get_database, close_database
from app.data.ingestion import DataIngestionService
from app.data.fusion import DataFusionService
from app.ml.analysis import WildfireAnalysisEngine
from app.api.advanced_analysis import router as advanced_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wildfire Mitigation System",
    description="Real-time wildfire risk monitoring and response system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include advanced analysis routes
app.include_router(advanced_router)

# Pydantic models for API requests/responses
class RegionRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: Optional[float] = 10.0

class AlertRequest(BaseModel):
    alert_type: str
    latitude: float
    longitude: float
    radius_km: float
    message: str
    severity: str
    target_audience: str

# Global services
ingestion_service = DataIngestionService()
fusion_service = DataFusionService()
analysis_engine = WildfireAnalysisEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize database and create indexes on startup"""
    try:
        await create_indexes()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        await close_database()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "Wildfire Mitigation System API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = await get_database()
        # Test database connection
        await db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/ingest/data")
async def ingest_data(background_tasks: BackgroundTasks):
    """Trigger data ingestion cycle"""
    try:
        # Define target locations (major California cities for demo)
        target_locations = [
            {"lat": 37.7749, "lon": -122.4194},  # San Francisco
            {"lat": 34.0522, "lon": -118.2437},  # Los Angeles
            {"lat": 32.7157, "lon": -117.1611},  # San Diego
            {"lat": 38.5816, "lon": -121.4944},  # Sacramento
        ]
        
        # Run ingestion in background
        background_tasks.add_task(ingestion_service.ingest_all_data, target_locations)
        
        return {
            "message": "Data ingestion started",
            "locations": len(target_locations),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting data ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fires/active")
async def get_active_fires():
    """Get all active fire incidents"""
    try:
        fires = await ingestion_service.get_active_fires()
        return {
            "fires": fires,
            "count": len(fires),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active fires: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/{latitude}/{longitude}")
async def get_weather_data(latitude: float, longitude: float, hours_back: int = 24):
    """Get weather data for a specific location"""
    try:
        weather_data = await ingestion_service.get_recent_weather_data(latitude, longitude, hours_back)
        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "weather_data": weather_data,
            "count": len(weather_data),
            "hours_back": hours_back,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/activity/{latitude}/{longitude}")
async def get_human_activity(latitude: float, longitude: float, radius_km: float = 10):
    """Get human activity data for a region"""
    try:
        activity_data = await ingestion_service.get_human_activity_in_region(latitude, longitude, radius_km)
        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius_km,
            "activity_data": activity_data,
            "count": len(activity_data),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting human activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/region")
async def analyze_region(region: RegionRequest, background_tasks: BackgroundTasks):
    """Analyze wildfire risk for a region"""
    try:
        target_region = {
            "min_lat": region.min_lat,
            "max_lat": region.max_lat,
            "min_lon": region.min_lon,
            "max_lon": region.max_lon
        }
        
        # Run analysis in background
        background_tasks.add_task(analysis_engine.analyze_region, target_region)
        
        return {
            "message": "Region analysis started",
            "region": target_region,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting region analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/{latitude}/{longitude}")
async def get_risk_assessment(latitude: float, longitude: float):
    """Get risk assessment for a specific location"""
    try:
        from app.models.database import risk_assessments_collection, create_geospatial_query
        
        # Query recent risk assessments for the location
        query = create_geospatial_query(latitude, longitude, max_distance_meters=5000)
        query["timestamp"] = {"$gte": datetime.utcnow() - timedelta(hours=6)}
        
        risk_assessments = await risk_assessments_collection.find(query).sort("timestamp", -1).limit(1).to_list(length=1)
        
        if risk_assessments:
            return {
                "location": {"latitude": latitude, "longitude": longitude},
                "risk_assessment": risk_assessments[0],
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Generate on-demand risk assessment
            h3_index = fusion_service.lat_lon_to_h3(latitude, longitude)
            weather_features = await fusion_service.get_weather_features(h3_index)
            fire_features = await fusion_service.get_fire_features(h3_index)
            activity_features = await fusion_service.get_human_activity_features(h3_index)
            
            combined_features = {**weather_features, **fire_features, **activity_features}
            risk_score = analysis_engine.risk_model.predict_risk(combined_features)
            
            return {
                "location": {"latitude": latitude, "longitude": longitude},
                "risk_assessment": {
                    "risk_score": risk_score,
                    "weather_features": weather_features,
                    "fire_features": fire_features,
                    "activity_features": activity_features,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{target_audience}")
async def get_recommendations(target_audience: str, latitude: Optional[float] = None, longitude: Optional[float] = None):
    """Get recommendations for a specific audience"""
    try:
        from app.models.database import recommendations_collection
        
        query = {"target_audience": target_audience}
        if latitude and longitude:
            # Add geospatial filter if coordinates provided
            from app.models.database import create_geospatial_query
            geo_query = create_geospatial_query(latitude, longitude, max_distance_meters=50000)
            query.update(geo_query)
        
        recommendations = await recommendations_collection.find(query).sort("priority", -1).limit(10).to_list(length=10)
        
        return {
            "target_audience": target_audience,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts")
async def create_alert(alert: AlertRequest):
    """Create a new alert"""
    try:
        from app.models.database import Alert, alerts_collection, AlertType, AlertSeverity, AlertStatus
        
        alert_record = Alert(
            alert_type=AlertType(alert.alert_type),
            location=create_location(alert.latitude, alert.longitude),
            radius_km=alert.radius_km,
            message=alert.message,
            severity=AlertSeverity(alert.severity),
            target_audience=alert.target_audience,
            status=AlertStatus.ACTIVE
        )
        
        await alerts_collection.insert_one(alert_record.dict())
        
        return {
            "message": "Alert created successfully",
            "alert": alert_record.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/active")
async def get_active_alerts(latitude: Optional[float] = None, longitude: Optional[float] = None):
    """Get active alerts"""
    try:
        from app.models.database import alerts_collection, AlertStatus
        
        query = {"status": AlertStatus.ACTIVE}
        if latitude and longitude:
            from app.models.database import create_geospatial_query
            geo_query = create_geospatial_query(latitude, longitude, max_distance_meters=50000)
            query.update(geo_query)
        
        alerts = await alerts_collection.find(query).sort("created_at", -1).to_list(length=50)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get summary data for dashboard"""
    try:
        from app.models.database import (
            fire_incidents_collection, risk_assessments_collection,
            alerts_collection, recommendations_collection, FireStatus, AlertStatus
        )
        
        # Get counts
        active_fires = await fire_incidents_collection.count_documents({"status": FireStatus.ACTIVE})
        high_risk_areas = await risk_assessments_collection.count_documents({"risk_score": {"$gte": 0.7}})
        active_alerts = await alerts_collection.count_documents({"status": AlertStatus.ACTIVE})
        pending_recommendations = await recommendations_collection.count_documents({"status": "pending"})
        
        # Get recent high-risk assessments
        recent_risks = await risk_assessments_collection.find(
            {"risk_score": {"$gte": 0.5}}
        ).sort("timestamp", -1).limit(5).to_list(length=5)
        
        return {
            "summary": {
                "active_fires": active_fires,
                "high_risk_areas": high_risk_areas,
                "active_alerts": active_alerts,
                "pending_recommendations": pending_recommendations
            },
            "recent_high_risks": recent_risks,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 