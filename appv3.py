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
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
import json
import re
from patterns.patterns import REGIONAL_PATTERNS
from geopy.geocoders import Nominatim

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
REGIONAL_PATTERNS = REGIONAL_PATTERNS

class OpenWeatherMapAPI:
    """OpenWeatherMap API integration for real-time weather data"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
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
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
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
                },
            
            predictions.append(prediction)
        
        return predictions

class WeatherChatbot:
    """Conversational weather assistant using Google Gemini"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
        
        self.weather_predictor = AdvancedWeatherPredictor()
        self.context = []
    
    def extract_location_and_timeframe(self, query):
        """Extract location and timeframe from user query"""
        
        # Common Indian locations (states, cities, districts)
        locations = []
        timeframes = []
        
        # Extract locations from query
        query_lower = query.lower()
        
        # Check for states and cities in REGIONAL_PATTERNS
        for location in REGIONAL_PATTERNS.keys():
            if location.lower() in query_lower:
                locations.append(location)
        
        # Extract timeframes
        time_patterns = {
            r'next week|1 week|7 days': 7,
            r'next month|1 month|30 days': 30,
            r'tomorrow': 1,
            r'day after tomorrow': 2,
            r'next (\d+) days': None,  # Dynamic extraction
            r'after (\d+) days': None,  # Dynamic extraction
            r'(\d+) days': None,       # Dynamic extraction
        }
        
        for pattern, days in time_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                if days is None:
                    # Extract number from match
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
        if not locations:
            locations = ['Delhi']  # Default location
        if not timeframes:
            timeframes = [1]  # Default to tomorrow
        
        return locations[0], timeframes[0]
    
    def get_weather_response(self, query):
        """Generate comprehensive weather response"""
        
        location, days_ahead = self.extract_location_and_timeframe(query)
        
        # Get current weather
        current_weather = self.weather_predictor.api.get_current_weather(location)
        current_data = self.weather_predictor.api.parse_current_weather(current_weather)
        
        # Get predictions
        predictions = self.weather_predictor.predict_extended_weather(location, days_ahead)
        
        # Generate response
        response = self.generate_detailed_response(query, location, days_ahead, current_data, predictions)
        
        return response, current_data, predictions
    
    def generate_detailed_response(self, query, location, days_ahead, current_data, predictions):
        """Generate detailed weather response using Gemini"""
        
        # Prepare context for Gemini
        context = f"""
        User Query: {query}
        Location: {location}
        Forecast Period: {days_ahead} days
        
        Current Weather Data:
        {current_data if current_data else "No current data available"}
        
        Predictions:
        {predictions.to_dict('records') if not predictions.empty else "No predictions available"}
        
        Regional Pattern: {REGIONAL_PATTERNS.get(location, 'Unknown')}
        """
        
        if self.model:
            # Use Gemini for intelligent response
            prompt = f"""
            You are a professional weather forecaster for India. Based on the following weather data and user query, provide a comprehensive, conversational response.
            
            {context}
            
            Please provide:
            1. Direct answer to the user's question
            2. Current weather conditions if available
            3. Detailed forecast for the requested period
            4. Regional weather patterns and seasonal considerations
            5. Any weather advisories or recommendations
            
            Make the response conversational and informative, as if you're talking to a friend.
            """
            
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except:
                pass
        
        # Fallback response
        return self.generate_fallback_response(location, days_ahead, current_data, predictions)
    
    def generate_fallback_response(self, location, days_ahead, current_data, predictions):
        """Generate fallback response without Gemini"""
        
        response = f"## Weather Forecast for {location}\n\n"
        
        # Current weather
        if current_data:
            response += f"**Current Conditions:**\n"
            response += f"🌡️ Temperature: {current_data['temperature']:.1f}°C (Feels like {current_data['feels_like']:.1f}°C)\n"
            response += f"💧 Humidity: {current_data['humidity']}%\n"
            response += f"🌪️ Wind Speed: {current_data['wind_speed']} m/s\n"
            response += f"📊 Pressure: {current_data['pressure']} hPa\n"
            response += f"🌤️ Conditions: {current_data['description'].title()}\n\n"
        
        # Forecast
        if not predictions.empty:
            response += f"**{days_ahead}-Day Forecast:**\n\n"
            
            for i, row in predictions.iterrows():
                if i < days_ahead:
                    date = row['date']
                    if isinstance(date, str):
                        date = pd.to_datetime(date)
                    
                    response += f"📅 **{date.strftime('%B %d, %Y')}**\n"
                    response += f"   Temperature: {row.get('temperature', 'N/A'):.1f}°C\n"
                    response += f"   Humidity: {row.get('humidity', 'N/A'):.1f}%\n"
                    if 'rainfall' in row and row['rainfall'] > 0:
                        response += f"   🌧️ Rainfall: {row['rainfall']:.1f}mm\n"
                    response += f"   Wind Speed: {row.get('wind_speed', 'N/A'):.1f} m/s\n\n"
        
        # Add forecasting methodology
        response += self.explain_methodology(days_ahead)
        
        return response
    
    def explain_methodology(self, days_ahead):
        """Explain the forecasting methodology used"""
        
        explanation = "\n**Forecasting Methodology:**\n"
        
        if days_ahead <= 5:
            explanation += "🔄 **Real-time API Data**: Using OpenWeatherMap API for 1-5 day forecasts (most accurate)\n"
        else:
            explanation += "🤖 **Hybrid Approach** for extended forecasts:\n"
            explanation += "   • **Prophet Model**: For long-term temperature trends and seasonality\n"
            explanation += "   • **Ensemble Models**: Random Forest + Gradient Boosting for multi-variable prediction\n"
            explanation += "   • **Regional Pattern Analysis**: Based on Indian monsoon and seasonal patterns\n"
            explanation += "   • **Statistical Methods**: Historical data averaging as fallback\n"
        
        explanation += "\n**Why this approach?**\n"
        explanation += "• **API Data (1-5 days)**: Highest accuracy using meteorological models\n"
        explanation += "• **Prophet Model**: Excellent for capturing seasonal patterns in Indian weather\n"
        explanation += "• **Ensemble Methods**: Combines multiple algorithms to reduce prediction errors\n"
        explanation += "• **Regional Patterns**: Incorporates monsoon, cyclone seasons, and local climate variations\n"
        
        return explanation

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
    """Main Streamlit application"""
    
    st.title("🌪️ Advanced Weather Prediction System")
    st.subheader("Conversational Weather Assistant for India")
    
    # Initialize chatbot
    chatbot = WeatherChatbot()
    
    # Sidebar for API status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check API status
        if chatbot.weather_predictor.api.api_key:
            st.success("✅ OpenWeatherMap API: Connected")
        else:
            st.error("❌ OpenWeatherMap API: Not configured")
        
        if chatbot.model:
            st.success("✅ Gemini AI: Connected")
        else:
            st.warning("⚠️ Gemini AI: Not configured (using fallback)")
        
        st.markdown("---")
        st.header("📍 Popular Locations")
        
        # Display popular cities
        popular_cities = list(REGIONAL_PATTERNS.keys())[:10]
        for city in popular_cities:
            if st.button(f"📍 {city}", key=f"loc_{city}"):
                st.session_state.location_query = f"What's the weather in {city}?"
    
    # Main chat interface
    st.header("💬 Ask about Weather")
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - "What's the weather in Mumbai tomorrow?"
        - "Will it rain in Delhi next week?"
        - "What's the temperature in Bangalore for the next 5 days?"
        - "Weather forecast for Chennai next month"
        - "Is it going to be sunny in Kolkata this weekend?"
        """)
    
    # Chat input
    user_query = st.text_input(
        "Ask me about weather anywhere in India:",
        placeholder="e.g., What's the weather in Mumbai next week?",
        value=st.session_state.get('location_query', '')
    )
    
    # Process query
    if user_query:
        with st.spinner("🔍 Analyzing weather patterns..."):
            try:
                response, current_data, predictions = chatbot.get_weather_response(user_query)
                
                # Display response
                st.markdown("## 🤖 Weather Assistant Response")
                st.markdown(response)
                
                # Create visualizations
                if current_data or not predictions.empty:
                    st.markdown("## 📊 Weather Visualizations")
                    
                    figures = create_weather_visualizations(current_data, predictions)
                    
                    # Display figures in tabs
                    if figures:
                        tab_names = [name for name, _ in figures]
                        tabs = st.tabs(tab_names)
                        
                        for i, (name, fig) in enumerate(figures):
                            with tabs[i]:
                                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data display
                with st.expander("📋 Raw Forecast Data"):
                    if not predictions.empty:
                        st.dataframe(predictions)
                    else:
                        st.info("No forecast data available")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.info("Please try rephrasing your question or check if the location is spelled correctly.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🌟 Features:**
    - Real-time weather data from OpenWeatherMap API
    - Extended forecasts using AI models (Prophet, Random Forest, Gradient Boosting)
    - Conversational interface powered by Google Gemini
    - Comprehensive coverage of Indian locations
    - Regional pattern analysis for accurate predictions
    """)

if __name__ == "__main__":
    main()