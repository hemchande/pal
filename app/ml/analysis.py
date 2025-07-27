import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import os
import json
import openai
from app.models.database import create_location, RiskAssessment, Recommendation, RecommendationType, RecommendationStatus
from app.data.fusion import DataFusionService

logger = logging.getLogger(__name__)

class WildfireRiskModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, combined_features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for the ML model"""
        feature_names = [
            'avg_temperature', 'avg_humidity', 'avg_wind_speed', 'max_wind_speed',
            'total_precipitation', 'active_fire_count', 'total_acres_burning',
            'max_severity_level', 'avg_severity_level', 'fire_age_hours',
            'total_activity_intensity', 'avg_activity_intensity', 'activity_count',
            'hiking_intensity', 'camping_intensity', 'construction_intensity'
        ]
        
        features = []
        for feature in feature_names:
            value = combined_features.get(feature, 0.0)
            # Handle None values
            if value is None:
                value = 0.0
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the risk model with historical data"""
        try:
            X = []
            y = []
            
            for data_point in training_data:
                features = self.prepare_features(data_point["combined_features"])
                X.append(features.flatten())
                
                # Use actual risk score if available, otherwise create synthetic one
                risk_score = data_point.get("risk_score", self._calculate_synthetic_risk(data_point))
                y.append(risk_score)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.is_trained = True
            logger.info(f"Model trained successfully. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def predict_risk(self, combined_features: Dict[str, Any]) -> float:
        """Predict wildfire risk score (0-1)"""
        try:
            if not self.is_trained:
                # Use simple heuristic if model not trained
                return self._calculate_synthetic_risk(combined_features)
            
            features = self.prepare_features(combined_features)
            features_scaled = self.scaler.transform(features)
            risk_score = self.model.predict(features_scaled)[0]
            
            # Ensure risk score is between 0 and 1
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return self._calculate_synthetic_risk(combined_features)
    
    def _calculate_synthetic_risk(self, features: Dict[str, Any]) -> float:
        """Calculate synthetic risk score using heuristics"""
        risk_factors = []
        
        # Weather factors
        if features.get("avg_temperature", 0) > 30:
            risk_factors.append(0.3)
        if features.get("avg_humidity", 100) < 30:
            risk_factors.append(0.2)
        if features.get("avg_wind_speed", 0) > 20:
            risk_factors.append(0.25)
        if features.get("total_precipitation", 0) < 5:
            risk_factors.append(0.15)
        
        # Fire factors
        if features.get("active_fire_count", 0) > 0:
            risk_factors.append(0.4)
        if features.get("max_severity_level", 0) >= 4:
            risk_factors.append(0.3)
        
        # Human activity factors
        if features.get("total_activity_intensity", 0) > 0.5:
            risk_factors.append(0.2)
        if features.get("hiking_intensity", 0) > 0.7:
            risk_factors.append(0.15)
        
        # Calculate weighted risk score
        if risk_factors:
            base_risk = sum(risk_factors) / len(risk_factors)
            # Add some randomness for realistic variation
            noise = np.random.normal(0, 0.1)
            return max(0.0, min(1.0, base_risk + noise))
        else:
            return 0.1  # Low base risk

class LLMAnalyzer:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            logger.warning("OpenAI API key not found. LLM features will be disabled.")
    
    async def analyze_risk_factors(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze risk factors and provide reasoning"""
        if not self.openai_api_key:
            return self._generate_mock_analysis(risk_data)
        
        try:
            prompt = self._create_risk_analysis_prompt(risk_data)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a wildfire risk assessment expert. Analyze the provided data and give concise, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            return self._parse_llm_response(analysis, risk_data)
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._generate_mock_analysis(risk_data)
    
    async def generate_recommendations(self, risk_data: Dict[str, Any], target_audience: str) -> List[Dict[str, Any]]:
        """Generate strategic recommendations using LLM"""
        if not self.openai_api_key:
            return self._generate_mock_recommendations(risk_data, target_audience)
        
        try:
            prompt = self._create_recommendation_prompt(risk_data, target_audience)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a wildfire response strategist. Generate specific, actionable recommendations based on the risk assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            recommendations_text = response.choices[0].message.content
            return self._parse_recommendations(recommendations_text, target_audience)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._generate_mock_recommendations(risk_data, target_audience)
    
    def _create_risk_analysis_prompt(self, risk_data: Dict[str, Any]) -> str:
        """Create prompt for risk analysis"""
        weather = risk_data.get("weather", {})
        fire = risk_data.get("fire", {})
        activity = risk_data.get("human_activity", {})
        
        prompt = f"""
        Analyze the following wildfire risk data and provide insights:
        
        Weather Conditions:
        - Temperature: {weather.get('avg_temperature', 'N/A')}°C
        - Humidity: {weather.get('avg_humidity', 'N/A')}%
        - Wind Speed: {weather.get('avg_wind_speed', 'N/A')} mph
        - Wind Direction: {weather.get('wind_direction', 'N/A')}
        - Precipitation: {weather.get('total_precipitation', 'N/A')} mm
        
        Fire Status:
        - Active Fires: {fire.get('active_fire_count', 0)}
        - Acres Burning: {fire.get('total_acres_burning', 0)}
        - Max Severity: {fire.get('max_severity_level', 0)}/5
        - Fire Age: {fire.get('fire_age_hours', 0)} hours
        
        Human Activity:
        - Total Activity: {activity.get('total_activity_intensity', 0)}
        - Hiking Intensity: {activity.get('hiking_intensity', 0)}
        - Camping Intensity: {activity.get('camping_intensity', 0)}
        
        Risk Score: {risk_data.get('risk_score', 0):.2f}
        
        Provide:
        1. Key risk factors contributing to this score
        2. Potential fire spread direction and speed
        3. Population at risk estimate
        4. Evacuation time estimate
        """
        return prompt
    
    def _create_recommendation_prompt(self, risk_data: Dict[str, Any], target_audience: str) -> str:
        """Create prompt for recommendations"""
        risk_score = risk_data.get('risk_score', 0)
        
        prompt = f"""
        Generate specific recommendations for {target_audience} based on wildfire risk score: {risk_score:.2f}
        
        Context:
        - Location: {risk_data.get('latitude', 'N/A')}, {risk_data.get('longitude', 'N/A')}
        - Risk Level: {'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'Low'}
        
        For {target_audience}, provide 3-5 specific, actionable recommendations including:
        1. Immediate actions (next 1-2 hours)
        2. Short-term strategies (next 24 hours)
        3. Resource allocation priorities
        4. Communication strategies
        
        Format each recommendation with:
        - Priority (1-5)
        - Description
        - Reasoning
        - Estimated impact
        - Implementation time (minutes)
        """
        return prompt
    
    def _parse_llm_response(self, response: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Extract key information from response
            lines = response.split('\n')
            analysis = {
                "risk_factors": [],
                "predicted_spread_direction": "unknown",
                "predicted_spread_speed": 0.0,
                "population_at_risk": 0,
                "evacuation_time_estimate": 30
            }
            
            for line in lines:
                line = line.strip().lower()
                if "wind" in line and "direction" in line:
                    # Extract wind direction
                    if "north" in line:
                        analysis["predicted_spread_direction"] = "north"
                    elif "south" in line:
                        analysis["predicted_spread_direction"] = "south"
                    elif "east" in line:
                        analysis["predicted_spread_direction"] = "east"
                    elif "west" in line:
                        analysis["predicted_spread_direction"] = "west"
                
                if "speed" in line:
                    # Extract speed estimate
                    words = line.split()
                    for i, word in enumerate(words):
                        if word.isdigit() and i < len(words) - 1:
                            if "mph" in words[i+1] or "km/h" in words[i+1]:
                                analysis["predicted_spread_speed"] = float(word)
                                break
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._generate_mock_analysis(risk_data)
    
    def _parse_recommendations(self, response: str, target_audience: str) -> List[Dict[str, Any]]:
        """Parse LLM recommendations into structured format"""
        try:
            recommendations = []
            lines = response.split('\n')
            
            current_rec = {}
            for line in lines:
                line = line.strip()
                if line.startswith('Priority:'):
                    if current_rec:
                        recommendations.append(current_rec)
                    current_rec = {"target_audience": target_audience}
                    current_rec["priority"] = int(line.split(':')[1].strip())
                elif line.startswith('Description:'):
                    current_rec["description"] = line.split(':', 1)[1].strip()
                elif line.startswith('Reasoning:'):
                    current_rec["reasoning"] = line.split(':', 1)[1].strip()
                elif line.startswith('Impact:'):
                    current_rec["estimated_impact"] = line.split(':', 1)[1].strip()
                elif line.startswith('Time:'):
                    time_str = line.split(':', 1)[1].strip()
                    current_rec["implementation_time"] = int(time_str.split()[0]) if time_str.split()[0].isdigit() else 30
            
            if current_rec:
                recommendations.append(current_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error parsing recommendations: {e}")
            return self._generate_mock_recommendations({}, target_audience)
    
    def _generate_mock_analysis(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock analysis when LLM is not available"""
        risk_score = risk_data.get('risk_score', 0)
        
        return {
            "risk_factors": ["High temperature", "Low humidity", "Strong winds"] if risk_score > 0.5 else ["Moderate conditions"],
            "predicted_spread_direction": "northwest" if risk_score > 0.7 else "north",
            "predicted_spread_speed": 15.0 if risk_score > 0.7 else 5.0,
            "population_at_risk": int(risk_score * 1000),
            "evacuation_time_estimate": int(30 + risk_score * 60)
        }
    
    def _generate_mock_recommendations(self, risk_data: Dict[str, Any], target_audience: str) -> List[Dict[str, Any]]:
        """Generate mock recommendations when LLM is not available"""
        risk_score = risk_data.get('risk_score', 0)
        
        if target_audience == "firefighters":
            return [
                {
                    "target_audience": "firefighters",
                    "priority": 5,
                    "description": "Deploy additional firefighting units to high-risk areas",
                    "reasoning": f"Risk score of {risk_score:.2f} indicates elevated fire danger",
                    "estimated_impact": "Reduce response time by 40%",
                    "implementation_time": 30
                }
            ]
        elif target_audience == "officials":
            return [
                {
                    "target_audience": "officials",
                    "priority": 4,
                    "description": "Issue public safety alerts for affected regions",
                    "reasoning": "Proactive communication can prevent casualties",
                    "estimated_impact": "Reach 10,000+ residents",
                    "implementation_time": 15
                }
            ]
        else:  # public
            return [
                {
                    "target_audience": "public",
                    "priority": 3,
                    "description": "Avoid outdoor activities in high-risk areas",
                    "reasoning": "Current conditions favor rapid fire spread",
                    "estimated_impact": "Reduce ignition sources",
                    "implementation_time": 5
                }
            ]

class WildfireAnalysisEngine:
    def __init__(self):
        self.risk_model = WildfireRiskModel()
        self.llm_analyzer = LLMAnalyzer()
        self.fusion_service = DataFusionService()
    
    async def analyze_region(self, target_region: Dict[str, float]) -> List[Dict[str, Any]]:
        """Complete analysis of a region"""
        try:
            # Create unified grid
            grid = await self.fusion_service.create_unified_grid(target_region)
            
            results = []
            for cell in grid:
                # Predict risk
                risk_score = self.risk_model.predict_risk(cell["combined_features"])
                
                # LLM analysis
                analysis_data = {
                    **cell,
                    "risk_score": risk_score
                }
                
                llm_analysis = await self.llm_analyzer.analyze_risk_factors(analysis_data)
                
                # Store risk assessment
                risk_assessment = {
                    "latitude": cell["latitude"],
                    "longitude": cell["longitude"],
                    "timestamp": cell["timestamp"],
                    "risk_score": risk_score,
                    "risk_factors": llm_analysis,
                    "predicted_spread_direction": llm_analysis.get("predicted_spread_direction"),
                    "predicted_spread_speed": llm_analysis.get("predicted_spread_speed"),
                    "population_at_risk": llm_analysis.get("population_at_risk"),
                    "evacuation_time_estimate": llm_analysis.get("evacuation_time_estimate")
                }
                
                await self.fusion_service.store_risk_assessment(risk_assessment)
                
                # Generate recommendations for different audiences
                recommendations = []
                for audience in ["firefighters", "officials", "public"]:
                    audience_recs = await self.llm_analyzer.generate_recommendations(analysis_data, audience)
                    recommendations.extend(audience_recs)
                
                results.append({
                    "cell": cell,
                    "risk_assessment": risk_assessment,
                    "recommendations": recommendations
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing region: {e}")
            return []

# Example usage
async def analyze_california():
    """Analyze California for wildfire risks"""
    engine = WildfireAnalysisEngine()
    
    # California bounding box
    california_region = {
        "min_lat": 32.5121,
        "max_lat": 42.0095,
        "min_lon": -124.4096,
        "max_lon": -114.1312
    }
    
    results = await engine.analyze_region(california_region)
    logger.info(f"Analysis complete. Processed {len(results)} grid cells")
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(analyze_california()) 