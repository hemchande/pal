from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from datetime import datetime

from app.ml.gnn_fire_prediction import FirePredictionService
from app.analysis.human_activity_alignment import HumanActivityAlignmentService
from app.services.safe_haven_finder import SafeHavenFinder
from app.services.fire_monitoring import FireMonitoringService
from app.data.ingestion import DataIngestionService
from app.data.fusion import DataFusionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced", tags=["Advanced Analysis"])

# Pydantic models for requests
class AdvancedAnalysisRequest(BaseModel):
    region: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    current_location: Optional[Tuple[float, float]] = None
    analysis_types: List[str] = ["gnn_prediction", "activity_alignment", "safe_havens", "fire_monitoring"]
    time_horizon_hours: int = 24

class SafeHavenRequest(BaseModel):
    current_location: Tuple[float, float]
    radius_km: float = 50
    include_places: bool = True

# Initialize services
gnn_service = FirePredictionService()
alignment_service = HumanActivityAlignmentService()
safe_haven_service = SafeHavenFinder()
monitoring_service = FireMonitoringService()
ingestion_service = DataIngestionService()
fusion_service = DataFusionService()

@router.post("/comprehensive-analysis")
async def comprehensive_analysis(request: AdvancedAnalysisRequest, background_tasks: BackgroundTasks):
    """Run comprehensive analysis including GNN predictions, activity alignment, and monitoring"""
    try:
        logger.info("Starting comprehensive wildfire analysis...")
        
        results = {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'region': request.region,
            'analysis_types': request.analysis_types,
            'results': {}
        }
        
        # Run requested analyses
        if "gnn_prediction" in request.analysis_types:
            background_tasks.add_task(
                run_gnn_prediction, request.region, request.time_horizon_hours
            )
            results['results']['gnn_prediction'] = {
                'status': 'started',
                'message': 'GNN prediction analysis started in background'
            }
        
        if "activity_alignment" in request.analysis_types:
            background_tasks.add_task(run_activity_alignment)
            results['results']['activity_alignment'] = {
                'status': 'started',
                'message': 'Activity alignment analysis started in background'
            }
        
        if "fire_monitoring" in request.analysis_types:
            background_tasks.add_task(run_fire_monitoring)
            results['results']['fire_monitoring'] = {
                'status': 'started',
                'message': 'Fire monitoring started in background'
            }
        
        if "safe_havens" in request.analysis_types and request.current_location:
            background_tasks.add_task(
                run_safe_haven_analysis, request.current_location
            )
            results['results']['safe_havens'] = {
                'status': 'started',
                'message': 'Safe haven analysis started in background'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gnn-prediction/{region_id}")
async def get_gnn_prediction(region_id: str, time_horizon_hours: int = 24):
    """Get GNN-based fire spread predictions for a region"""
    try:
        # Define region based on region_id (simplified)
        regions = {
            'california': {
                "min_lat": 32.5121,
                "max_lat": 42.0095,
                "min_lon": -124.4096,
                "max_lon": -114.1312
            },
            'bay_area': {
                "min_lat": 37.0,
                "max_lat": 38.5,
                "min_lon": -123.0,
                "max_lon": -121.5
            }
        }
        
        region = regions.get(region_id, regions['bay_area'])
        
        # Get unified grid data
        grid_data = await fusion_service.create_unified_grid(region)
        
        # Run GNN prediction
        predictions = gnn_service.predict_fire_spread(grid_data, time_horizon_hours)
        
        return {
            'region_id': region_id,
            'region': region,
            'time_horizon_hours': time_horizon_hours,
            'predictions': predictions,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in GNN prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity-alignment")
async def get_activity_alignment():
    """Get human activity alignment with active fires"""
    try:
        # Get active fires
        fires = await ingestion_service.get_active_fires()
        
        # Get human activity data for California cities
        california_cities = [
            {"lat": 37.7749, "lon": -122.4194},  # San Francisco
            {"lat": 34.0522, "lon": -118.2437},  # Los Angeles
            {"lat": 32.7157, "lon": -117.1611},  # San Diego
            {"lat": 38.5816, "lon": -121.4944},  # Sacramento
        ]
        
        all_activities = []
        for city in california_cities:
            activities = await ingestion_service.get_human_activity_in_region(
                city["lat"], city["lon"], radius_km=20
            )
            all_activities.extend(activities)
        
        # Analyze alignment
        analysis = await alignment_service.analyze_activity_alignment(fires, all_activities)
        
        return {
            'fires_analyzed': len(fires),
            'activities_analyzed': len(all_activities),
            'alignment_analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in activity alignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safe-havens")
async def find_safe_havens(request: SafeHavenRequest):
    """Find safe havens and evacuation routes"""
    try:
        # Get active fires
        fires = await ingestion_service.get_active_fires()
        
        # Find safe havens
        safe_havens = await safe_haven_service.find_safe_havens(
            request.current_location, fires, request.radius_km
        )
        
        return {
            'current_location': request.current_location,
            'radius_km': request.radius_km,
            'safe_havens': safe_havens,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error finding safe havens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fire-monitoring")
async def get_fire_monitoring():
    """Get comprehensive fire monitoring results"""
    try:
        # Monitor fire progression
        monitoring_results = await monitoring_service.monitor_fire_progression()
        
        # Get status summary
        status_summary = await monitoring_service.get_fire_status_summary()
        
        return {
            'monitoring_results': monitoring_results,
            'status_summary': status_summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in fire monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/heatmap-data")
async def get_heatmap_data():
    """Get heatmap data for visualization"""
    try:
        # Get active fires
        fires = await ingestion_service.get_active_fires()
        
        # Get human activity data
        california_cities = [
            {"lat": 37.7749, "lon": -122.4194},  # San Francisco
            {"lat": 34.0522, "lon": -118.2437},  # Los Angeles
            {"lat": 32.7157, "lon": -117.1611},  # San Diego
            {"lat": 38.5816, "lon": -121.4944},  # Sacramento
        ]
        
        all_activities = []
        for city in california_cities:
            activities = await ingestion_service.get_human_activity_in_region(
                city["lat"], city["lon"], radius_km=20
            )
            all_activities.extend(activities)
        
        # Create heatmap data
        heatmap_data = alignment_service._create_activity_heatmap(activities, fires)
        
        return {
            'heatmap_data': heatmap_data,
            'fire_count': len(fires),
            'activity_count': len(all_activities),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evacuation-priorities")
async def get_evacuation_priorities():
    """Get prioritized evacuation recommendations"""
    try:
        # Get activity alignment analysis
        fires = await ingestion_service.get_active_fires()
        
        california_cities = [
            {"lat": 37.7749, "lon": -122.4194},  # San Francisco
            {"lat": 34.0522, "lon": -118.2437},  # Los Angeles
            {"lat": 32.7157, "lon": -117.1611},  # San Diego
            {"lat": 38.5816, "lon": -121.4944},  # Sacramento
        ]
        
        all_activities = []
        for city in california_cities:
            activities = await ingestion_service.get_human_activity_in_region(
                city["lat"], city["lon"], radius_km=20
            )
            all_activities.extend(activities)
        
        analysis = await alignment_service.analyze_activity_alignment(fires, all_activities)
        
        # Extract evacuation priorities
        evacuation_priorities = analysis.get('risk_assessments', {}).get('evacuation_priorities', [])
        
        return {
            'evacuation_priorities': evacuation_priorities,
            'total_priorities': len(evacuation_priorities),
            'critical_areas': len([p for p in evacuation_priorities if p.get('evacuation_urgency') == 'immediate']),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evacuation priorities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def run_gnn_prediction(region: Dict[str, float], time_horizon_hours: int):
    """Background task for GNN prediction"""
    try:
        grid_data = await fusion_service.create_unified_grid(region)
        predictions = gnn_service.predict_fire_spread(grid_data, time_horizon_hours)
        logger.info(f"GNN prediction completed. {len(predictions.get('high_risk_areas', []))} high-risk areas identified.")
    except Exception as e:
        logger.error(f"Error in background GNN prediction: {e}")

async def run_activity_alignment():
    """Background task for activity alignment"""
    try:
        fires = await ingestion_service.get_active_fires()
        
        california_cities = [
            {"lat": 37.7749, "lon": -122.4194},
            {"lat": 34.0522, "lon": -118.2437},
            {"lat": 32.7157, "lon": -117.1611},
            {"lat": 38.5816, "lon": -121.4944},
        ]
        
        all_activities = []
        for city in california_cities:
            activities = await ingestion_service.get_human_activity_in_region(
                city["lat"], city["lon"], radius_km=20
            )
            all_activities.extend(activities)
        
        analysis = await alignment_service.analyze_activity_alignment(fires, all_activities)
        logger.info(f"Activity alignment completed. {analysis.get('summary', {}).get('total_overlaps', 0)} overlaps found.")
    except Exception as e:
        logger.error(f"Error in background activity alignment: {e}")

async def run_fire_monitoring():
    """Background task for fire monitoring"""
    try:
        monitoring_results = await monitoring_service.monitor_fire_progression()
        logger.info(f"Fire monitoring completed. {monitoring_results.get('summary', {}).get('total_alerts', 0)} alerts generated.")
    except Exception as e:
        logger.error(f"Error in background fire monitoring: {e}")

async def run_safe_haven_analysis(current_location: Tuple[float, float]):
    """Background task for safe haven analysis"""
    try:
        fires = await ingestion_service.get_active_fires()
        safe_havens = await safe_haven_service.find_safe_havens(current_location, fires, radius_km=50)
        logger.info(f"Safe haven analysis completed. {len(safe_havens.get('safe_havens', []))} safe locations found.")
    except Exception as e:
        logger.error(f"Error in background safe haven analysis: {e}") 