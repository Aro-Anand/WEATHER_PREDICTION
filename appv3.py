import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os
from dotenv import load_dotenv
import json
import re
from patterns.patterns import regional_patterns
from geopy.geocoders import Nominatim
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import Google Generative AI
import google.generativeai as genai

st.set_page_config(
    page_title="Advanced Weather Prediction System",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Regional weather patterns and characteristics (Extended for all major Indian cities)
REGIONAL_PATTERNS = regional_patterns
def get_api_key(key_name, service_name):
    key = os.getenv(key_name)
    if not key:
        st.warning(f"⚠️ {service_name} API key not found. Please set {key_name} in your .env file.")
    return key
class OpenWeatherMapAPI:
    """OpenWeatherMap API integration for real-time weather data"""
    
    def __init__(self):
        self.api_key = get_api_key('OPENWEATHER_API_KEY', 'OpenWeatherMap')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.geocoder = Nominatim(user_agent="weather_app")
    
    def get_coordinates(self, location):
        """Get coordinates for a location using geocoding"""
        try:
            # First check if location is in our regional patterns
            for city, data in REGIONAL_PATTERNS.items():
                if city.lower() in location.lower():
                    return data['coordinates']['lat'], data['coordinates']['lon']
            
            # Use geocoding for other locations
            location_obj = self.geocoder.geocode(f"{location}, India")
            if location_obj:
                return location_obj.latitude, location_obj.longitude
            return None, None
        except:
            return None, None
    
    def get_current_weather(self, location):
        """Get current weather for a location"""
        if not self.api_key:
            return None

        lat, lon = self.get_coordinates(location)
        if not lat or not lon:
            return None

        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }

        return self._make_api_request(url, params)

    def _make_api_request(self, url, params):
        """Make API request with error handling"""
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None

    def get_5day_forecast(self, location):
        """Get 5-day weather forecast for a location"""
        lat, lon = self.get_coordinates(location)
        if not lat or not lon:
            return None
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    
    def parse_current_weather(self, data):
        """Parse current weather data"""
        if not data:
            return None
        
        return {
            'location': data['name'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'visibility': data.get('visibility', 0) / 1000,  # Convert to km
            'datetime': datetime.fromtimestamp(data['dt'])
        }
    
    def parse_forecast(self, data):
        """Parse 5-day forecast data"""
        if not data:
            return None
        
        forecast_list = []
        for item in data['list']:
            forecast_list.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'description': item['weather'][0]['description'],
                'rainfall': item.get('rain', {}).get('3h', 0)
            })
        
        return pd.DataFrame(forecast_list)

class AdvancedWeatherPredictor:
    """Advanced weather prediction with multiple approaches"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.regional_models = {}
        self.api = OpenWeatherMapAPI()
        
    def generate_synthetic_indian_weather_data(self, location, years=5):
        """Generate synthetic weather data based on Indian weather patterns"""
        
        # Find the closest regional pattern
        region_key = self._find_closest_region(location)
        if not region_key:
            region_key = 'Delhi'  # Default
        
        pattern = REGIONAL_PATTERNS[region_key]
        
        # Generate comprehensive historical data
        start_date = datetime.now() - timedelta(days=365 * years)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        np.random.seed(42)
        data = []
        
        for date in dates:
            month = date.month
            day_of_year = date.dayofyear
            
            # Get seasonal parameters
            season_data = self._get_seasonal_data(month, pattern)
            
            # Temperature with realistic variation
            base_temp = season_data['avg_temp']
            daily_variation = 8 * np.sin(2 * np.pi * (date.hour if hasattr(date, 'hour') else 14) / 24)
            yearly_variation = 3 * np.sin(2 * np.pi * day_of_year / 365)
            random_variation = np.random.normal(0, 2)
            
            temperature = base_temp + daily_variation + yearly_variation + random_variation
            
            # Humidity with monsoon influence
            base_humidity = season_data['humidity']
            if month in pattern['monsoon_months']:
                humidity = base_humidity + np.random.normal(5, 8)
            else:
                humidity = base_humidity + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 100)
            
            # Atmospheric pressure
            base_pressure = 1013.25
            if month in pattern['monsoon_months']:
                base_pressure -= 8  # Lower pressure during monsoon
            pressure = base_pressure + np.random.normal(0, 5)
            
            # Wind speed
            base_wind = 3
            if month in pattern['monsoon_months']:
                base_wind += 6
            if month in pattern['cyclone_season']:
                base_wind += 10
            wind_speed = max(0, base_wind + np.random.normal(0, 3))
            
            # Rainfall
            if month in pattern['monsoon_months']:
                rainfall = max(0, np.random.exponential(8))
            elif month in pattern['cyclone_season']:
                rainfall = max(0, np.random.exponential(4))
            else:
                rainfall = max(0, np.random.exponential(0.8))
            
            data.append({
                'date': date,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'rainfall': rainfall,
                'month': month,
                'day_of_year': day_of_year,
                'season': self._get_season_name(month, pattern)
            })
        
        return pd.DataFrame(data)
    
    def _find_closest_region(self, location):
        """Find the closest regional pattern for a location"""
        location_lower = location.lower()
        
        # Direct match
        for region in REGIONAL_PATTERNS.keys():
            if region.lower() in location_lower:
                return region
        
        # State-based matching
        for region, data in REGIONAL_PATTERNS.items():
            if data['state'].lower() in location_lower:
                return region
        
        return None
    
    def _get_seasonal_data(self, month, pattern):
        """Get seasonal weather data for a month"""
        # Summer months
        if month in [3, 4, 5]:
            return {'avg_temp': pattern['avg_temp_range'][1] - 5, 'humidity': 55}
        # Monsoon months
        elif month in pattern['monsoon_months']:
            return {'avg_temp': pattern['avg_temp_range'][0] + 8, 'humidity': 80}
        # Winter months
        elif month in [12, 1, 2]:
            return {'avg_temp': pattern['avg_temp_range'][0] + 3, 'humidity': 60}
        # Post-monsoon
        else:
            return {'avg_temp': pattern['avg_temp_range'][0] + 6, 'humidity': 70}
    
    def _get_season_name(self, month, pattern):
        """Get season name for a month"""
        if month in [3, 4, 5]:
            return 'summer'
        elif month in pattern['monsoon_months']:
            return 'monsoon'
        elif month in [12, 1, 2]:
            return 'winter'
        else:
            return 'post_monsoon'
    
    def train_ensemble_models(self, location, df):
        """Train ensemble of models for weather prediction"""
        
        # Feature engineering
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['temp_lag_1'] = df['temperature'].shift(1)
        df['temp_lag_7'] = df['temperature'].shift(7)
        df['humidity_lag_1'] = df['humidity'].shift(1)
        df['rainfall_lag_1'] = df['rainfall'].shift(1)
        
        # Rolling averages
        df['temp_rolling_7'] = df['temperature'].rolling(window=7).mean()
        df['humidity_rolling_7'] = df['humidity'].rolling(window=7).mean()
        
        # Remove NaN values
        df = df.dropna()
        
        # Features for prediction
        feature_cols = [
            'day_of_year', 'month', 'day_of_week', 'humidity', 'pressure',
            'wind_speed', 'rainfall', 'temp_lag_1', 'temp_lag_7', 'humidity_lag_1',
            'rainfall_lag_1', 'temp_rolling_7', 'humidity_rolling_7'
        ]
        
        X = df[feature_cols]
        
        # Multiple target variables
        y_temp = df['temperature']
        y_humidity = df['humidity']
        y_rainfall = df['rainfall']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models for each target
        models = {}
        
        # Random Forest (Good for non-linear relationships)
        models['rf_temp'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['rf_temp'].fit(X_scaled, y_temp)
        
        models['rf_humidity'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['rf_humidity'].fit(X_scaled, y_humidity)
        
        models['rf_rainfall'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['rf_rainfall'].fit(X_scaled, y_rainfall)
        
        # Gradient Boosting (Good for sequential patterns)
        models['gb_temp'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        models['gb_temp'].fit(X_scaled, y_temp)
        
        models['gb_humidity'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        models['gb_humidity'].fit(X_scaled, y_humidity)
        
        models['gb_rainfall'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        models['gb_rainfall'].fit(X_scaled, y_rainfall)
        
        # Store models
        self.regional_models[location] = {
            'models': models,
            'scaler': scaler,
            'features': feature_cols,
            'last_data': df.tail(30)  # Keep last 30 days for prediction
        }
    
    def train_prophet_model(self, location, df):
        """Train Prophet model for time series forecasting"""
        
        # Prepare data for Prophet
        prophet_data = df[['date', 'temperature']].rename(columns={'date': 'ds', 'temperature': 'y'})
        
        # Create and train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_data)
        
        # Store model
        if location not in self.regional_models:
            self.regional_models[location] = {}
        self.regional_models[location]['prophet'] = model
    
    def predict_extended_weather(self, location, days_ahead):
        """Predict weather for extended periods using ensemble approach"""
        
        # Get real-time data for the last few days
        current_weather = self.api.get_current_weather(location)
        forecast_5day = self.api.get_5day_forecast(location)
        
        predictions = []
        
        # For 1-5 days, use OpenWeatherMap API
        if days_ahead <= 5:
            if forecast_5day:
                forecast_df = self.api.parse_forecast(forecast_5day)
                return forecast_df.head(days_ahead * 8)  # 8 forecasts per day (3-hour intervals)
        
        # For longer periods, use trained models
        if location in self.regional_models:
            models_data = self.regional_models[location]
            
            # Use Prophet for long-term trends
            if 'prophet' in models_data:
                prophet_model = models_data['prophet']
                
                # Create future dates
                future_dates = pd.date_range(
                    start=datetime.now().date(),
                    periods=days_ahead,
                    freq='D'
                )
                
                future_df = pd.DataFrame({'ds': future_dates})
                prophet_forecast = prophet_model.predict(future_df)
                
                # Combine with ensemble models for other variables
                for i, date in enumerate(future_dates):
                    # Get Prophet temperature prediction
                    temp_prophet = prophet_forecast.iloc[i]['yhat']
                    
                    # Use ensemble models for other variables
                    base_prediction = {
                        'date': date,
                        'temperature': temp_prophet,
                        'humidity': self._predict_humidity(location, date, temp_prophet),
                        'rainfall': self._predict_rainfall(location, date),
                        'pressure': self._predict_pressure(location, date),
                        'wind_speed': self._predict_wind_speed(location, date),
                        'method': 'Prophet + Ensemble Models'
                    }
                    
                    predictions.append(base_prediction)
        
        else:
            # Fallback to statistical methods
            predictions = self._statistical_prediction(location, days_ahead)
        
        return pd.DataFrame(predictions)
    
    def _predict_humidity(self, location, date, temperature):
        """Predict humidity based on temperature and seasonal patterns"""
        region_key = self._find_closest_region(location)
        if region_key:
            pattern = REGIONAL_PATTERNS[region_key]
            season_data = self._get_seasonal_data(date.month, pattern)
            base_humidity = season_data['humidity']
            
            # Adjust for temperature (higher temp = lower humidity generally)
            temp_adjustment = (temperature - 25) * -1.5
            return np.clip(base_humidity + temp_adjustment + np.random.normal(0, 5), 20, 100)
        return 60  # Default
    
    def _predict_rainfall(self, location, date):
        """Predict rainfall based on seasonal patterns"""
        region_key = self._find_closest_region(location)
        if region_key:
            pattern = REGIONAL_PATTERNS[region_key]
            if date.month in pattern['monsoon_months']:
                return max(0, np.random.exponential(6))
            elif date.month in pattern['cyclone_season']:
                return max(0, np.random.exponential(3))
            else:
                return max(0, np.random.exponential(0.5))
        return 0
    
    def _predict_pressure(self, location, date):
        """Predict atmospheric pressure"""
        base_pressure = 1013.25
        region_key = self._find_closest_region(location)
        if region_key:
            pattern = REGIONAL_PATTERNS[region_key]
            if date.month in pattern['monsoon_months']:
                base_pressure -= 8
        return base_pressure + np.random.normal(0, 3)
    
    def _predict_wind_speed(self, location, date):
        """Predict wind speed based on seasonal patterns"""
        base_wind = 4
        region_key = self._find_closest_region(location)
        if region_key:
            pattern = REGIONAL_PATTERNS[region_key]
            if date.month in pattern['monsoon_months']:
                base_wind += 5
            if date.month in pattern['cyclone_season']:
                base_wind += 8
        return max(0, base_wind + np.random.normal(0, 2))
    
    def _statistical_prediction(self, location, days_ahead):
        """Fallback statistical prediction method"""
        predictions = []
        
        # Generate synthetic data for the location
        synthetic_data = self.generate_synthetic_indian_weather_data(location, years=3)
        
        # Use simple seasonal averaging
        current_date = datetime.now()
        
        for i in range(days_ahead):
            future_date = current_date + timedelta(days=i)
            
            # Find historical data for similar dates
            similar_dates = synthetic_data[
                (synthetic_data['date'].dt.month == future_date.month) &
                (synthetic_data['date'].dt.day == future_date.day)
            ]
            
            if not similar_dates.empty:
                prediction = {
                    'date': future_date,
                    'temperature': similar_dates['temperature'].mean(),
                    'humidity': similar_dates['humidity'].mean(),
                    'rainfall': similar_dates['rainfall'].mean(),
                    'pressure': similar_dates['pressure'].mean(),
                    'wind_speed': similar_dates['wind_speed'].mean(),
                    'method': 'Statistical Averaging'
                }
            else:
                # Use regional averages
                region_key = self._find_closest_region(location)
                if region_key:
                    pattern = REGIONAL_PATTERNS[region_key]
                    season_data = self._get_seasonal_data(future_date.month, pattern)
                    prediction = {
                        'date': future_date,
                        'temperature': season_data['avg_temp'],
                        'humidity': season_data['humidity'],
                        'rainfall': 0,
                        'pressure': 1013,
                        'wind_speed': 5,
                        'method': 'Regional Pattern Matching'
                    }
                else:
                    prediction = {
                        'date': future_date,
                        'temperature': 25,
                        'humidity': 60,
                        'rainfall': 0,
                        'pressure': 1013,
                        'wind_speed': 5,
                        'method': 'Default Values'
                    }
            
            predictions.append(prediction)
        
        return predictions

class WeatherChatbot:
    """Conversational weather assistant using Google Gemini"""
    
    def __init__(self):
        self.gemini_api_key = get_api_key('GOOGLE_API_KEY', 'Google Gemini')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
        
        self.weather_predictor = AdvancedWeatherPredictor()
        self.context = []
        
        # Enhanced location mappings for better recognition
        self.location_aliases = {
            'bombay': 'Mumbai',
            'calcutta': 'Kolkata',
            'madras': 'Chennai',
            'bangalore': 'Bengaluru',
            'mysore': 'Mysuru',
            'poona': 'Pune',
            'kanpur': 'Kanpur',
            'allahabad': 'Prayagraj',
            'baroda': 'Vadodara',
            'trivandrum': 'Thiruvananthapuram'
        }
        
        # Weather-related keywords for better query understanding
        self.weather_keywords = {
            'temperature': ['temp', 'temperature', 'hot', 'cold', 'warm', 'cool', 'degree'],
            'rain': ['rain', 'rainfall', 'precipitation', 'shower', 'drizzle', 'downpour'],
            'humidity': ['humidity', 'moisture', 'humid', 'muggy', 'damp'],
            'wind': ['wind', 'breeze', 'gust', 'windy', 'storm'],
            'pressure': ['pressure', 'barometric'],
            'cyclone': ['cyclone', 'hurricane', 'typhoon', 'storm', 'cyclonic'],
            'flood': ['flood', 'flooding', 'inundation', 'waterlogging'],
            'drought': ['drought', 'dry', 'arid', 'water shortage'],
            'monsoon': ['monsoon', 'monsoons', 'rainy season']
        }
    
    def preprocess_query_with_gemini(self, query):
        """Use Gemini to correct spelling mistakes and extract intent"""
        if not self.model:
            return query, self.extract_location_and_timeframe(query)
        
        preprocessing_prompt = f"""
        You are a weather query preprocessor for India. Your task is to:
        1. Correct any spelling mistakes in the query
        2. Identify the location (city/state) mentioned
        3. Identify the time frame requested
        4. Determine if this is about severe weather (cyclone, flood, drought, etc.)
        5. Standardize the query for better processing
        
        Original query: "{query}"
        
        Please respond in this JSON format:
        {{
            "corrected_query": "corrected version of the query",
            "location": "identified location or null",
            "timeframe_days": number_of_days_or_null,
            "weather_type": "normal/cyclone/flood/drought/severe",
            "intent": "current_weather/forecast/historical/severe_weather_info"
        }}
        
        Common Indian city name corrections:
        - Bombay → Mumbai
        - Calcutta → Kolkata  
        - Madras → Chennai
        - Bangalore → Bengaluru
        """
        
        try:
            response = self.model.generate_content(preprocessing_prompt)
            # Extract JSON from response
            response_text = response.text
            
            # Try to extract JSON
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
                
                corrected_query = parsed_response.get('corrected_query', query)
                location = parsed_response.get('location')
                timeframe = parsed_response.get('timeframe_days', 1)
                weather_type = parsed_response.get('weather_type', 'normal')
                intent = parsed_response.get('intent', 'current_weather')
                
                # Apply location aliases
                if location:
                    location = self.location_aliases.get(location.lower(), location)
                
                return corrected_query, (location or 'Delhi', timeframe, weather_type, intent)
            
        except Exception as e:
            st.warning(f"Query preprocessing failed: {e}")
        
        # Fallback to original method
        location, timeframe = self.extract_location_and_timeframe(query)
        return query, (location, timeframe, 'normal', 'current_weather')
    
    def extract_location_and_timeframe(self, query):
        """Enhanced location and timeframe extraction"""
        query_lower = query.lower()
        
        # Extract locations with improved matching
        locations = []
        
        # Check for states and cities in REGIONAL_PATTERNS
        for location in REGIONAL_PATTERNS.keys():
            if location.lower() in query_lower:
                locations.append(location)
        
        # Check aliases
        for alias, real_name in self.location_aliases.items():
            if alias in query_lower:
                locations.append(real_name)
        
        # Enhanced timeframe extraction
        timeframes = []

        # More comprehensive time patterns
        time_patterns = {
            r'today': 0,
            r'tomorrow': 1,
            r'day after tomorrow': 2,
            r'next week|1 week|7 days': 7,
            r'next month|1 month|30 days': 30,
            r'next (\d+) days?': None,
            r'after (\d+) days?': None,
            r'(\d+) days? ahead': None,
            r'in (\d+) days?': None,
            r'for (\d+) days?': None,
            r'this weekend': 2,
            r'this week': 7,
            r'next weekend': 9,
        }
        
        for pattern, days in time_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                if days is None:
                    for match in matches:
                        if isinstance(match, str) and match.isdigit():
                            timeframes.append(int(match))
                        elif isinstance(match, tuple):
                            for m in match:
                                if m.isdigit():
                                    timeframes.append(int(m))
                else:
                    timeframes.append(days)
        
        # Default values
        location = locations[0] if locations else 'Delhi'
        timeframe = timeframes[0] if timeframes else 1
        
        return location, timeframe
    
    def explain_methodology(self, days_ahead):
        """Explain the methodology used for predictions"""
        methodology = "\n### 📚 Methodology\n\n"
        
        if days_ahead <= 5:
            methodology += "🌐 **Data Source**: OpenWeatherMap API (Real-time forecasts)\n"
        else:
            methodology += "🤖 **Data Source**: AI Models + Historical Patterns\n"
            methodology += "📊 **Models Used**: Prophet Time Series + Ensemble Methods\n"
        
        methodology += "🇮🇳 **Regional Adaptation**: Indian weather patterns and seasonal variations\n"
        methodology += "⚠️ **Disclaimer**: Predictions are estimates. Always check official meteorological sources for critical decisions.\n"
        
        return methodology

    def handle_severe_weather_query(self, query, location, weather_type):
        """Handle queries about cyclones, floods, droughts, etc."""
        
        if not self.model:
            return self.generate_fallback_severe_weather_response(location, weather_type)
        
        # Get current weather context
        current_weather = self.weather_predictor.api.get_current_weather(location)
        current_data = self.weather_predictor.api.parse_current_weather(current_weather)
        
        # Get regional patterns
        region_key = self.weather_predictor._find_closest_region(location)
        regional_pattern = REGIONAL_PATTERNS.get(region_key, {})
        
        severe_weather_prompt = f"""
        You are a severe weather expert for India. The user is asking about {weather_type} conditions.
        
        User Query: {query}
        Location: {location}
        Weather Type: {weather_type}
        
        Current Weather Data: {current_data}
        Regional Pattern: {regional_pattern}
        
        Please provide a comprehensive response covering:
        1. Current risk assessment for {weather_type} in {location}
        2. Seasonal patterns and typical occurrence times
        3. Historical context and frequency
        4. Warning signs and preparedness measures
        5. Current weather conditions that might contribute to {weather_type}
        6. Safety recommendations and emergency contacts
        
        Make the response informative, accurate, and helpful for someone in {location}.
        """
        
        try:
            response = self.model.generate_content(severe_weather_prompt)
            return response.text
        except Exception as e:
            return self.generate_fallback_severe_weather_response(location, weather_type)
    
    def generate_fallback_severe_weather_response(self, location, weather_type):
        """Fallback response for severe weather queries"""
        
        region_key = self.weather_predictor._find_closest_region(location)
        regional_pattern = REGIONAL_PATTERNS.get(region_key, {})
        
        response = f"## {weather_type.title()} Information for {location}\n\n"
        
        if weather_type == 'cyclone':
            response += "🌀 **Cyclone Information:**\n"
            response += f"• **Peak Season**: {regional_pattern.get('cyclone_season', 'April-June, October-December')}\n"
            response += "• **Warning Signs**: Sudden pressure drop, increased wind speed, heavy rainfall\n"
            response += "• **Preparedness**: Stay indoors, stock essential supplies, monitor weather updates\n"
            response += "• **Emergency**: Contact local disaster management authorities\n\n"
            
        elif weather_type == 'flood':
            response += "🌊 **Flood Information:**\n"
            response += f"• **Risk Period**: During monsoon season ({regional_pattern.get('monsoon_months', 'June-September')})\n"
            response += "• **Warning Signs**: Continuous heavy rainfall, river level rise, waterlogging\n"
            response += "• **Safety**: Move to higher ground, avoid walking/driving through flooded areas\n"
            response += "• **Emergency**: Contact local flood control authorities\n\n"
            
        elif weather_type == 'drought':
            response += "🏜️ **Drought Information:**\n"
            response += "• **Risk Period**: Extended dry periods, delayed monsoon\n"
            response += "• **Indicators**: Low rainfall, high temperatures, water scarcity\n"
            response += "• **Conservation**: Water rationing, crop management, livestock care\n"
            response += "• **Support**: Contact agricultural extension services\n\n"
        
        response += "⚠️ **Important**: For real-time alerts and official updates, monitor:\n"
        response += "• India Meteorological Department (IMD)\n"
        response += "• National Disaster Management Authority (NDMA)\n"
        response += "• Local disaster management authorities\n"
        
        return response
    
    def get_weather_response(self, query):
        """Enhanced weather response with improved query processing"""
        
        # Preprocess query with Gemini
        corrected_query, (location, days_ahead, weather_type, intent) = self.preprocess_query_with_gemini(query)
        
        # Handle severe weather queries
        if weather_type in ['cyclone', 'flood', 'drought', 'severe']:
            severe_response = self.handle_severe_weather_query(corrected_query, location, weather_type)
            return severe_response, None, pd.DataFrame()
        
        # Get current weather with error handling
        try:
            current_weather = self.weather_predictor.api.get_current_weather(location)
            current_data = self.weather_predictor.api.parse_current_weather(current_weather)
            
            if not current_data:
                # Try alternative location search
                alternative_location = self.find_alternative_location(location)
                if alternative_location:
                    current_weather = self.weather_predictor.api.get_current_weather(alternative_location)
                    current_data = self.weather_predictor.api.parse_current_weather(current_weather)
                    location = alternative_location
                    
        except Exception as e:
            st.warning(f"Error fetching current weather: {e}")
            current_data = None
        
        # Get predictions with error handling
        try:
            predictions = self.weather_predictor.predict_extended_weather(location, days_ahead)
        except Exception as e:
            st.warning(f"Error generating predictions: {e}")
            predictions = pd.DataFrame()
        
        # Generate enhanced response
        response = self.generate_enhanced_response(corrected_query, location, days_ahead, current_data, predictions, intent)
        
        return response, current_data, predictions
    
    def find_alternative_location(self, location):
        """Find alternative location if original fails"""
        
        # Check if location is a state, find major city
        state_cities = {
            'maharashtra': 'Mumbai',
            'karnataka': 'Bengaluru',
            'tamil nadu': 'Chennai',
            'west bengal': 'Kolkata',
            'rajasthan': 'Jaipur',
            'gujarat': 'Ahmedabad',
            'punjab': 'Chandigarh',
            'haryana': 'Gurgaon',
            'uttar pradesh': 'Lucknow',
            'madhya pradesh': 'Bhopal',
            'bihar': 'Patna',
            'odisha': 'Bhubaneswar',
            'jharkhand': 'Ranchi',
            'chhattisgarh': 'Raipur',
            'kerala': 'Kochi',
            'andhra pradesh': 'Hyderabad',
            'telangana': 'Hyderabad',
            'assam': 'Guwahati',
            'himachal pradesh': 'Shimla',
            'uttarakhand': 'Dehradun'
        }
        
        location_lower = location.lower()
        for state, city in state_cities.items():
            if state in location_lower:
                return city
        
        # If location contains common words, extract city name
        words = location.split()
        for word in words:
            if word in REGIONAL_PATTERNS:
                return word
        
        return None
    
    def generate_enhanced_response(self, query, location, days_ahead, current_data, predictions, intent):
        """Generate enhanced response with better context"""
        
        if not self.model:
            return self.generate_fallback_response(location, days_ahead, current_data, predictions)
        
        # Prepare enhanced context
        context = f"""
        User Query: {query}
        Location: {location}
        Forecast Period: {days_ahead} days
        Query Intent: {intent}
        
        Current Weather Data:
        {current_data if current_data else "Current weather data not available"}
        
        Predictions:
        {predictions.to_dict('records') if not predictions.empty else "Predictions not available"}
        
        Regional Pattern: {REGIONAL_PATTERNS.get(location, 'Pattern not found')}
        
        Additional Context:
        - Current date: {datetime.now().strftime('%Y-%m-%d')}
        - Season: {self.get_current_season()}
        """
        
        enhanced_prompt = f"""
        You are an expert Indian meteorologist providing weather information. Based on the data provided, give a comprehensive and conversational response.
        
        {context}
        
        Guidelines for response:
        1. Start with a direct answer to the user's question
        2. Provide current weather conditions if available
        3. Give detailed forecast for the requested period
        4. Include relevant seasonal context and regional patterns
        5. Add practical advice (clothing, travel, outdoor activities)
        6. Mention any weather advisories or warnings
        7. Be conversational and friendly
        8. Use appropriate weather emojis
        9. If data is limited, explain the methodology used
        
        Format the response in markdown with clear sections.
        """
        
        try:
            response = self.model.generate_content(enhanced_prompt)
            return response.text
        except Exception as e:
            st.warning(f"Enhanced response generation failed: {e}")
            return self.generate_fallback_response(location, days_ahead, current_data, predictions)
    
    def get_current_season(self):
        """Determine current season in India"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"
    
    def generate_fallback_response(self, location, days_ahead, current_data, predictions):
        """Enhanced fallback response"""
        
        response = f"## 🌤️ Weather Report for {location}\n\n"
        
        # Current weather with better formatting
        if current_data:
            response += f"### 📍 Current Conditions\n"
            response += f"🌡️ **Temperature**: {current_data['temperature']:.1f}°C (Feels like {current_data['feels_like']:.1f}°C)\n"
            response += f"💧 **Humidity**: {current_data['humidity']}%\n"
            response += f"🌪️ **Wind**: {current_data['wind_speed']} m/s\n"
            response += f"📊 **Pressure**: {current_data['pressure']} hPa\n"
            response += f"👁️ **Visibility**: {current_data.get('visibility', 'N/A')} km\n"
            response += f"🌤️ **Conditions**: {current_data['description'].title()}\n\n"
        else:
            response += "### ⚠️ Current weather data not available\n\n"
        
        # Enhanced forecast
        if not predictions.empty:
            response += f"### 📅 {days_ahead}-Day Forecast\n\n"
            
            for i, row in predictions.head(days_ahead).iterrows():
                date = row['date']
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                
                # Determine day name
                day_name = date.strftime('%A')
                date_str = date.strftime('%B %d, %Y')
                
                response += f"**{day_name}, {date_str}**\n"
                response += f"   🌡️ Temperature: {row.get('temperature', 'N/A'):.1f}°C\n"
                response += f"   💧 Humidity: {row.get('humidity', 'N/A'):.1f}%\n"
                
                if 'rainfall' in row and row['rainfall'] > 0:
                    response += f"   🌧️ Rainfall: {row['rainfall']:.1f}mm\n"
                
                response += f"   🌪️ Wind: {row.get('wind_speed', 'N/A'):.1f} m/s\n"
                
                # Add weather advice
                temp = row.get('temperature', 25)
                if temp > 35:
                    response += f"   ⚠️ **Hot day**: Stay hydrated, avoid outdoor activities\n"
                elif temp < 15:
                    response += f"   🧥 **Cool day**: Wear warm clothing\n"
                
                if 'rainfall' in row and row['rainfall'] > 10:
                    response += f"   ☔ **Rainy day**: Carry umbrella, avoid unnecessary travel\n"
                
                response += "\n"
        else:
            response += "### ⚠️ Extended forecast not available\n"
            response += "Forecast data could not be generated. Please try again later.\n\n"
        
        # Add methodology explanation
        response += self.explain_methodology(days_ahead)
        
        return response

def create_weather_visualizations(current_data, predictions):
    """Create weather visualizations"""
    
    figures = []
    
    # Temperature trend
    if not predictions.empty and 'temperature' in predictions.columns:
        fig_temp = px.line(
            predictions.head(10), 
            x='date', 
            y='temperature',
            title='Temperature Forecast',
            labels={'temperature': 'Temperature (°C)', 'date': 'Date'}
        )
        fig_temp.update_layout(height=400)
        figures.append(('Temperature Trend', fig_temp))
    
    # Humidity and Rainfall
    if not predictions.empty:
        fig_multi = go.Figure()
        
        if 'humidity' in predictions.columns:
            fig_multi.add_trace(go.Scatter(
                x=predictions['date'].head(10),
                y=predictions['humidity'].head(10),
                mode='lines+markers',
                name='Humidity (%)',
                yaxis='y'
            ))
        
        if 'rainfall' in predictions.columns:
            fig_multi.add_trace(go.Scatter(
                x=predictions['date'].head(10),
                y=predictions['rainfall'].head(10),
                mode='lines+markers',
                name='Rainfall (mm)',
                yaxis='y2'
            ))
        
        fig_multi.update_layout(
            title='Humidity and Rainfall Forecast',
            xaxis_title='Date',
            yaxis=dict(title='Humidity (%)', side='left'),
            yaxis2=dict(title='Rainfall (mm)', side='right', overlaying='y'),
            height=400
        )
        figures.append(('Humidity & Rainfall', fig_multi))
    
    # Current weather gauge (if available)
    if current_data:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_data['temperature'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Current Temperature in {current_data['location']}"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 35], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        figures.append(('Current Temperature', fig_gauge))
    
    return figures

def main():
    """Enhanced main Streamlit application"""
    
    st.title("🌪️ Advanced Weather Prediction System")
    st.subheader("Conversational Weather Assistant for India")
    
    # Initialize chatbot with error handling
    try:
        chatbot = WeatherChatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check API status
        if chatbot.weather_predictor.api.api_key:
            st.success("✅ OpenWeatherMap API: Connected")
        else:
            st.error("❌ OpenWeatherMap API: Not configured")
            st.info("Please add OPENWEATHER_API_KEY to your .env file")
        
        if chatbot.model:
            st.success("✅ Gemini AI: Connected")
        else:
            st.warning("⚠️ Gemini AI: Not configured")
            st.info("Please add GOOGLE_API_KEY to your .env file for enhanced features")
        
        st.markdown("---")
        st.header("📍 Popular Locations")
        
        # Enhanced location buttons
        popular_cities = list(REGIONAL_PATTERNS.keys())[:12]
        cols = st.columns(2)
        
        for i, city in enumerate(popular_cities):
            col = cols[i % 2]
            with col:
                if st.button(f"📍 {city}", key=f"loc_{city}"):
                    st.session_state.location_query = f"What's the weather in {city}?"
        
        st.markdown("---")
        st.header("⚠️ Severe Weather")
        
        severe_weather_queries = [
            "Cyclone risk in Chennai",
            "Flood alert in Mumbai",
            "Drought conditions in Rajasthan"
        ]
        
        for query in severe_weather_queries:
            if st.button(query, key=f"severe_{query}"):
                st.session_state.location_query = query
    
    # Enhanced main interface
    st.header("💬 Ask about Weather")
    
    # Enhanced example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **🌤️ Regular Weather:**
        - "What's the weather in Mumbai tomorrow?"
        - "Will it rain in Delhi next week?"
        - "Temperature forecast for Bangalore next 5 days"
        
        **🌀 Severe Weather:**
        - "Cyclone risk in Chennai this season"
        - "Flood conditions in Kerala"
        - "Drought situation in Karnataka"
        
        **🗓️ Time-based:**
        - "Weather this weekend in Goa"
        - "Next month's weather in Pune"
        - "Weather for the next 10 days in Kolkata"
        """)
    
    # Enhanced input handling
    user_query = st.text_input(
        "Ask me about weather anywhere in India:",
        placeholder="e.g., What's the weather in Mumbai tomorrow? (supports spell correction)",
        value=st.session_state.get('location_query', '')
    )
    
    # Clear previous query
    if 'location_query' in st.session_state:
        del st.session_state.location_query
    
    # Enhanced query processing
    if user_query:
        with st.spinner("🔍 Analyzing weather patterns and correcting query..."):
            try:
                # Get response with enhanced processing
                response, current_data, predictions = chatbot.get_weather_response(user_query)
                
                # Display response
                st.markdown("## 🤖 Weather Assistant Response")
                st.markdown(response)
                
                # Enhanced visualizations
                if current_data or not predictions.empty:
                    st.markdown("## 📊 Interactive Weather Visualizations")
                    
                    figures = create_weather_visualizations(current_data, predictions)
                    
                    if figures:
                        tab_names = [name for name, _ in figures]
                        tabs = st.tabs(tab_names)
                        
                        for i, (name, fig) in enumerate(figures):
                            with tabs[i]:
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Visualization data not available")
                
                # Enhanced data display
                with st.expander("📋 Detailed Forecast Data"):
                    if not predictions.empty:
                        # Format the dataframe for better display
                        display_df = predictions.copy()
                        if 'date' in display_df.columns:
                            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                        
                        # Round numeric columns
                        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
                        display_df[numeric_cols] = display_df[numeric_cols].round(2)
                        
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No detailed forecast data available")
                
                # Current weather details
                if current_data:
                    with st.expander("🌤️ Current Weather Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Temperature", f"{current_data['temperature']:.1f}°C", 
                                     f"Feels like {current_data['feels_like']:.1f}°C")
                            st.metric("Humidity", f"{current_data['humidity']}%")
                            st.metric("Wind Speed", f"{current_data['wind_speed']} m/s")
                        
                        with col2:
                            st.metric("Pressure", f"{current_data['pressure']} hPa")
                            st.metric("Visibility", f"{current_data.get('visibility', 'N/A')} km")
                            st.write(f"**Description:** {current_data['description'].title()}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("💡 **Troubleshooting tips:**")
                st.info("- Check if the location name is spelled correctly")
                st.info("- Try using a major city name instead of a small town")
                st.info("- Ensure your API keys are properly configured")
                st.info("- Try rephrasing your question")
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    ### 🌟 Advanced Features
    
    **🤖 AI-Powered:**
    - Automatic spelling correction using Google Gemini
    - Natural language understanding
    - Contextual weather responses
    
    **🌍 Comprehensive Coverage:**
    - All major Indian cities and states
    - Regional weather pattern analysis
    - Seasonal and monsoon considerations
    
    **⚡ Multiple Data Sources:**
    - Real-time data from OpenWeatherMap API
    - Historical pattern analysis
    - Advanced ML models for extended forecasts
    
    **🚨 Severe Weather Support:**
    - Cyclone tracking and alerts
    - Flood risk assessment
    - Drought condition monitoring
    """)
    
    # Add usage statistics
    if st.button("📊 Show System Info"):
        st.info(f"""
        **System Configuration:**
        - Supported Locations: {len(REGIONAL_PATTERNS)} major cities
        - API Status: {'✅ Connected' if chatbot.weather_predictor.api.api_key else '❌ Not configured'}
        - AI Enhancement: {'✅ Enabled' if chatbot.model else '❌ Disabled'}
        - Current Season: {chatbot.get_current_season()}
        """)

if __name__ == "__main__":
    main()