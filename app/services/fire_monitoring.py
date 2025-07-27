import numpy as np
import pandas as pd
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
from app.models.database import fire_incidents_collection, alerts_collection, create_location, AlertType, AlertSeverity, AlertStatus
from app.ml.gnn_fire_prediction import FirePredictionService

logger = logging.getLogger(__name__)

class FireMonitoringService:
    def __init__(self):
        self.prediction_service = FirePredictionService()
        self.fire_history = {}  # Track fire progression over time
        self.alert_thresholds = {
            'size_increase': 0.2,  # 20% size increase triggers alert
            'spread_speed': 5.0,   # 5 km/h spread speed triggers alert
            'direction_change': 30, # 30 degree direction change triggers alert
            'severity_increase': 1  # 1 level severity increase triggers alert
        }
    
    async def monitor_fire_progression(self) -> Dict[str, Any]:
        """Monitor all active fires for progression and changes"""
        try:
            logger.info("Starting fire progression monitoring...")
            
            # Get all active fires
            active_fires = await self._get_active_fires()
            
            monitoring_results = {
                'fires_monitored': len(active_fires),
                'progression_alerts': [],
                'containment_updates': [],
                'spread_warnings': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for fire in active_fires:
                fire_id = fire.get('incident_id', 'unknown')
                
                # Analyze fire progression
                progression_analysis = await self._analyze_fire_progression(fire)
                
                # Check for significant changes
                alerts = await self._check_for_alerts(fire, progression_analysis)
                monitoring_results['progression_alerts'].extend(alerts)
                
                # Update fire history
                self._update_fire_history(fire_id, progression_analysis)
                
                # Check for containment progress
                containment_update = await self._check_containment_progress(fire)
                if containment_update:
                    monitoring_results['containment_updates'].append(containment_update)
                
                # Check for spread warnings
                spread_warning = await self._check_spread_warnings(fire, progression_analysis)
                if spread_warning:
                    monitoring_results['spread_warnings'].append(spread_warning)
            
            # Generate summary report
            monitoring_results['summary'] = self._generate_monitoring_summary(monitoring_results)
            
            logger.info(f"Fire monitoring complete. {len(monitoring_results['progression_alerts'])} alerts generated.")
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring fire progression: {e}")
            return {}
    
    async def _get_active_fires(self) -> List[Dict[str, Any]]:
        """Get all active fires from database"""
        try:
            cursor = fire_incidents_collection.find({"status": "active"})
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"Error getting active fires: {e}")
            return []
    
    async def _analyze_fire_progression(self, fire: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze progression of a specific fire"""
        try:
            fire_id = fire.get('incident_id', 'unknown')
            current_time = datetime.utcnow()
            
            # Get historical data for this fire
            fire_history = self.fire_history.get(fire_id, [])
            
            # Current fire state
            current_state = {
                'timestamp': current_time,
                'acres_burned': fire.get('acres_burned', 0),
                'severity_level': fire.get('severity_level', 1),
                'location': fire['location'],
                'status': fire.get('status', 'active')
            }
            
            # Calculate progression metrics
            progression_metrics = self._calculate_progression_metrics(fire_history, current_state)
            
            # Predict future spread
            spread_prediction = await self._predict_fire_spread(fire)
            
            return {
                'fire_id': fire_id,
                'current_state': current_state,
                'progression_metrics': progression_metrics,
                'spread_prediction': spread_prediction,
                'analysis_timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fire progression: {e}")
            return {}
    
    def _calculate_progression_metrics(self, fire_history: List[Dict[str, Any]], 
                                     current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate progression metrics from historical data"""
        if not fire_history:
            return {
                'size_change_rate': 0.0,
                'spread_speed': 0.0,
                'direction_change': 0.0,
                'severity_change': 0,
                'containment_progress': 0.0
            }
        
        # Get most recent historical state
        last_state = fire_history[-1]
        
        # Calculate size change rate (acres per hour)
        time_diff = (current_state['timestamp'] - last_state['timestamp']).total_seconds() / 3600
        if time_diff > 0:
            size_change = current_state['acres_burned'] - last_state['acres_burned']
            size_change_rate = size_change / time_diff
        else:
            size_change_rate = 0.0
        
        # Calculate spread speed (simplified - would need more sophisticated analysis)
        spread_speed = size_change_rate * 0.1  # Rough conversion from acres to km/h
        
        # Calculate direction change (simplified)
        direction_change = 0.0  # Would need centroid analysis for actual direction
        
        # Calculate severity change
        severity_change = current_state['severity_level'] - last_state['severity_level']
        
        # Calculate containment progress (if status changed)
        containment_progress = 0.0
        if current_state['status'] == 'contained' and last_state['status'] == 'active':
            containment_progress = 100.0
        
        return {
            'size_change_rate': size_change_rate,
            'spread_speed': spread_speed,
            'direction_change': direction_change,
            'severity_change': severity_change,
            'containment_progress': containment_progress
        }
    
    async def _predict_fire_spread(self, fire: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future fire spread using GNN model"""
        try:
            # Create a region around the fire for prediction
            fire_lat = fire['location']['coordinates'][1]
            fire_lon = fire['location']['coordinates'][0]
            
            # Define prediction region (20km radius)
            region_data = self._create_prediction_region(fire_lat, fire_lon, radius_km=20)
            
            # Use GNN prediction service
            predictions = self.prediction_service.predict_fire_spread(region_data, time_horizon_hours=24)
            
            return {
                'predicted_spread': predictions.get('predictions', []),
                'high_risk_areas': predictions.get('high_risk_areas', []),
                'model_confidence': predictions.get('model_confidence', 0.0),
                'prediction_horizon': 24
            }
            
        except Exception as e:
            logger.error(f"Error predicting fire spread: {e}")
            return {}
    
    def _create_prediction_region(self, center_lat: float, center_lon: float, 
                                radius_km: float) -> List[Dict[str, Any]]:
        """Create a region for fire spread prediction"""
        region_data = []
        
        # Create a grid of points around the fire
        lat_step = radius_km / 111.0 / 10  # Convert to degrees
        lon_step = radius_km / (111.0 * np.cos(np.radians(center_lat))) / 10
        
        for i in range(-10, 11):
            for j in range(-10, 11):
                lat = center_lat + (i * lat_step)
                lon = center_lon + (j * lon_step)
                
                # Create mock data for each point
                region_data.append({
                    'h3_index': f"test_{i}_{j}",
                    'latitude': lat,
                    'longitude': lon,
                    'weather': {
                        'avg_temperature': 25.0,
                        'avg_humidity': 40.0,
                        'avg_wind_speed': 15.0,
                        'wind_direction': 'northwest'
                    },
                    'fire': {
                        'active_fire_count': 1 if i == 0 and j == 0 else 0,
                        'total_acres_burning': 100.0 if i == 0 and j == 0 else 0.0
                    },
                    'human_activity': {
                        'total_activity_intensity': 0.3,
                        'hiking_intensity': 0.2,
                        'camping_intensity': 0.1
                    }
                })
        
        return region_data
    
    async def _check_for_alerts(self, fire: Dict[str, Any], 
                              progression_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for conditions that require alerts"""
        alerts = []
        fire_id = fire.get('incident_id', 'unknown')
        
        try:
            metrics = progression_analysis.get('progression_metrics', {})
            
            # Check for significant size increase
            if metrics.get('size_change_rate', 0) > self.alert_thresholds['size_increase']:
                alert = await self._create_alert(
                    fire_id, 'size_increase',
                    f"Fire {fire_id} showing significant size increase: {metrics['size_change_rate']:.2f} acres/hour",
                    'high'
                )
                alerts.append(alert)
            
            # Check for high spread speed
            if metrics.get('spread_speed', 0) > self.alert_thresholds['spread_speed']:
                alert = await self._create_alert(
                    fire_id, 'spread_speed',
                    f"Fire {fire_id} spreading rapidly: {metrics['spread_speed']:.2f} km/h",
                    'critical'
                )
                alerts.append(alert)
            
            # Check for severity increase
            if metrics.get('severity_change', 0) >= self.alert_thresholds['severity_increase']:
                alert = await self._create_alert(
                    fire_id, 'severity_increase',
                    f"Fire {fire_id} severity increased by {metrics['severity_change']} levels",
                    'high'
                )
                alerts.append(alert)
            
            # Check for direction change
            if abs(metrics.get('direction_change', 0)) > self.alert_thresholds['direction_change']:
                alert = await self._create_alert(
                    fire_id, 'direction_change',
                    f"Fire {fire_id} direction changed by {metrics['direction_change']:.1f} degrees",
                    'medium'
                )
                alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Error checking for alerts: {e}")
        
        return alerts
    
    async def _create_alert(self, fire_id: str, alert_type: str, message: str, 
                          severity: str) -> Dict[str, Any]:
        """Create and store an alert"""
        try:
            from app.models.database import Alert, AlertType, AlertSeverity, AlertStatus
            
            alert_record = Alert(
                alert_type=AlertType.WARNING,
                location=create_location(37.7749, -122.4194),  # Default location
                radius_km=10.0,
                message=message,
                severity=AlertSeverity(severity),
                target_audience="firefighters",
                status=AlertStatus.ACTIVE
            )
            
            await alerts_collection.insert_one(alert_record.dict())
            
            return {
                'fire_id': fire_id,
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return {}
    
    async def _check_containment_progress(self, fire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for containment progress"""
        try:
            fire_id = fire.get('incident_id', 'unknown')
            
            # Check if fire status changed to contained
            if fire.get('status') == 'contained':
                return {
                    'fire_id': fire_id,
                    'event_type': 'containment',
                    'message': f"Fire {fire_id} has been contained",
                    'timestamp': datetime.utcnow().isoformat(),
                    'acres_burned': fire.get('acres_burned', 0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking containment progress: {e}")
            return None
    
    async def _check_spread_warnings(self, fire: Dict[str, Any], 
                                   progression_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for spread warnings to neighboring areas"""
        try:
            fire_id = fire.get('incident_id', 'unknown')
            spread_prediction = progression_analysis.get('spread_prediction', {})
            
            high_risk_areas = spread_prediction.get('high_risk_areas', [])
            
            if high_risk_areas:
                # Identify areas at risk of encroachment
                at_risk_areas = []
                for area in high_risk_areas:
                    if area.get('spread_probability', 0) > 0.8:
                        at_risk_areas.append({
                            'location': [area.get('latitude', 0), area.get('longitude', 0)],
                            'risk_level': area.get('risk_level', 'unknown'),
                            'time_to_spread': area.get('time_to_spread', 0)
                        })
                
                if at_risk_areas:
                    return {
                        'fire_id': fire_id,
                        'event_type': 'spread_warning',
                        'message': f"Fire {fire_id} may encroach on {len(at_risk_areas)} areas",
                        'at_risk_areas': at_risk_areas,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking spread warnings: {e}")
            return None
    
    def _update_fire_history(self, fire_id: str, progression_analysis: Dict[str, Any]):
        """Update fire history with current analysis"""
        if fire_id not in self.fire_history:
            self.fire_history[fire_id] = []
        
        # Keep only last 24 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.fire_history[fire_id] = [
            entry for entry in self.fire_history[fire_id]
            if entry['timestamp'] > cutoff_time
        ]
        
        # Add current state
        current_state = progression_analysis.get('current_state', {})
        if current_state:
            self.fire_history[fire_id].append(current_state)
    
    def _generate_monitoring_summary(self, monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of monitoring results"""
        return {
            'total_alerts': len(monitoring_results['progression_alerts']),
            'containment_updates': len(monitoring_results['containment_updates']),
            'spread_warnings': len(monitoring_results['spread_warnings']),
            'critical_alerts': len([a for a in monitoring_results['progression_alerts'] 
                                  if a.get('severity') == 'critical']),
            'high_priority_alerts': len([a for a in monitoring_results['progression_alerts'] 
                                       if a.get('severity') in ['critical', 'high']])
        }
    
    async def get_fire_status_summary(self) -> Dict[str, Any]:
        """Get summary of all fire statuses"""
        try:
            active_fires = await self._get_active_fires()
            
            summary = {
                'total_active_fires': len(active_fires),
                'total_acres_burning': sum(fire.get('acres_burned', 0) for fire in active_fires),
                'fires_by_severity': {},
                'recent_containments': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Group fires by severity
            for fire in active_fires:
                severity = fire.get('severity_level', 1)
                if severity not in summary['fires_by_severity']:
                    summary['fires_by_severity'][severity] = 0
                summary['fires_by_severity'][severity] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting fire status summary: {e}")
            return {}

# Example usage
async def monitor_california_fires():
    """Monitor all fires in California"""
    monitoring_service = FireMonitoringService()
    
    # Monitor fire progression
    monitoring_results = await monitoring_service.monitor_fire_progression()
    
    # Get status summary
    status_summary = await monitoring_service.get_fire_status_summary()
    
    logger.info(f"Fire monitoring complete. {monitoring_results.get('summary', {}).get('total_alerts', 0)} alerts generated.")
    
    return {
        'monitoring_results': monitoring_results,
        'status_summary': status_summary
    }

if __name__ == "__main__":
    asyncio.run(monitor_california_fires()) 