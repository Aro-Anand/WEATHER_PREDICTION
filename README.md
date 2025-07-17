# 🌪️ Advanced Weather Prediction System

A powerful, intelligent, and region-aware **weather prediction app** for India, built using **Streamlit**, **Machine Learning**, **Time Series Forecasting**, and **LLMs** like **Google Gemini**.

---

## 🔍 Overview

This application serves as a **conversational weather assistant** for Indian users. It combines **real-time weather data** from the OpenWeatherMap API, **long-term weather forecasting models** (Prophet, Random Forest, Gradient Boosting), and **AI-driven conversation generation** using **Google Gemini**.

---

## ✨ Key Features

- **🌤️ Real-time Weather** – Current conditions from OpenWeatherMap API
- **📅 Short & Long-Term Forecasts** – Up to 30-day forecasts using ML models
- **🧠 AI-Powered Chat** – Google Gemini interprets natural language queries
- **📈 Visual Forecasts** – Beautiful Plotly charts for temperature, humidity, rainfall
- **📍 Regional Pattern Support** – Indian climate-specific seasonal modeling
- **🧪 Synthetic Data Generator** – Generates weather data for training models
- **📊 Ensemble Models** – Random Forest and Gradient Boosting for accuracy
- **📦 Modular Design** – Easily extendable architecture

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Models:** Prophet, Random Forest, Gradient Boosting
- **LLMs:** Google Gemini 1.5 Flash
- **Visualization:** Plotly
- **Data:** OpenWeatherMap API + Synthetic Generation
- **Other Libraries:** `scikit-learn`, `pandas`, `numpy`, `statsmodels`, `geopy`, `dotenv`, `tensorflow`

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-weather-predictor.git
cd advanced-weather-predictor

2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file in the root directory and add your API keys:

env
Copy
Edit
OPENWEATHER_API_KEY=your_openweathermap_api_key
GOOGLE_API_KEY=your_gemini_api_key
4. Start the Streamlit App
bash
Copy
Edit
streamlit run app.py
💬 Example Queries
You can ask the assistant queries like:

"What's the weather in Mumbai tomorrow?"

"Will it rain in Delhi next week?"

"Show me the 5-day forecast for Chennai"

"What’s the temperature in Bangalore next month?"

📂 Project Structure
bash
Copy
Edit
📦 advanced-weather-predictor/
├── app.py                        # Main Streamlit app
├── patterns/
│   └── patterns.py               # Indian regional climate definitions
├── .env                          # API keys (not committed)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
🌐 Data Sources
Weather Data: OpenWeatherMap

Location Geocoding: [Geopy + Nominatim]

AI Responses: Google Gemini

🧠 Forecasting Methodology
1-5 Days: OpenWeatherMap real-time forecasts

6-30 Days:

📈 Prophet for temperature trends

🌲 Random Forest & Gradient Boosting for humidity, rainfall, wind

🌦️ Regional pattern matching for fallback scenarios

🧪 Synthetic Data Generation
If historical data is unavailable, the app generates synthetic weather data using realistic Indian monsoon, cyclone, and seasonal patterns.

🛡️ Security & Privacy
API keys are secured using environment variables (.env)

No user data is stored or shared