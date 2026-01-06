# House Price Predictor

A full-stack machine learning application for predicting house prices. Built with React, FastAPI, and scikit-learn.

![House Price Predictor](https://via.placeholder.com/800x400?text=House+Price+Predictor)

## 🚀 Features

- **ML-Powered Predictions**: Gradient Boosting model trained on the Ames Housing dataset
- **Modern UI**: Clean, responsive React interface with Material-UI
- **RESTful API**: FastAPI backend with automatic documentation
- **Prediction History**: All predictions are saved and can be reviewed
- **Model Management**: Train new models and switch between versions
- **Real-time Stats**: Dashboard with prediction statistics

## 📁 Project Structure

```
aiproject/
├── frontend/               # React + Vite + TypeScript
│   ├── src/
│   │   ├── api/           # API client
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   ├── App.tsx        # Main app component
│   │   ├── main.tsx       # Entry point
│   │   └── theme.ts       # MUI theme
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                # FastAPI + SQLAlchemy
│   ├── routes/            # API endpoints
│   │   ├── predictions.py
│   │   ├── models.py
│   │   └── health.py
│   ├── main.py            # FastAPI app
│   ├── config.py          # Settings
│   ├── database.py        # DB connection
│   ├── models.py          # SQLAlchemy models
│   ├── schemas.py         # Pydantic schemas
│   ├── ml_service.py      # ML integration
│   └── requirements.txt
│
├── ml/                     # Machine Learning
│   ├── models/            # Saved models
│   ├── config.py          # ML configuration
│   ├── pipeline.py        # ML pipeline
│   └── train.py           # Training script
│
└── data/                   # Dataset
    ├── train.csv
    ├── test.csv
    └── data_description.txt
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Clone and Setup

```bash
cd princoun
```

### 2. Backend Setup

```bash
# Create virtual environment
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the ML Model

```bash
# From project root
cd ml
python train.py
```

This will train the Gradient Boosting model and save it to `ml/models/`.

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5173

## 🧪 Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "OverallQual": 7,
      "OverallCond": 5,
      "YearBuilt": 2005,
      "TotalBsmtSF": 1000,
      "GrLivArea": 1500,
      "FullBath": 2,
      "HalfBath": 1,
      "BedroomAbvGr": 3,
      "TotRmsAbvGrd": 7,
      "GarageCars": 2,
      "GarageArea": 500,
      "MSZoning": "RL",
      "Neighborhood": "CollgCr",
      "BldgType": "1Fam",
      "HouseStyle": "2Story",
      "CentralAir": "Y",
      "KitchenQual": "Gd"
    }
  }'
```

### Get Predictions History

```bash
curl http://localhost:8000/predictions
```

### Train New Model

```bash
curl -X POST http://localhost:8000/models/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "my_model", "description": "Test model"}'
```

## 🏗️ Architecture

### Frontend

- **React 18** with TypeScript for type safety
- **Material-UI (MUI)** for modern, responsive design
- **React Router** for client-side routing
- **Axios** for API communication
- **Recharts** for data visualization

### Backend

- **FastAPI** for high-performance API
- **SQLAlchemy** ORM with SQLite (default) or PostgreSQL
- **Pydantic** for strict data validation
- **CORS** configured for React frontend

### Machine Learning

- **scikit-learn** Pipeline with:
  - SimpleImputer for missing values
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  - GradientBoostingRegressor for predictions

### Database Schema

**predictions**

- id (Primary Key)
- input_features (JSON)
- predicted_price (Float)
- model_version (String)
- created_at (DateTime)

**ml_models**

- id (Primary Key)
- name (String)
- version (String, Unique)
- model_path (String)
- metrics (JSON)
- is_active (Boolean)
- created_at (DateTime)

## 📊 Model Performance

The Gradient Boosting model achieves:

- **RMSE**: ~$25,000-30,000
- **R² Score**: ~0.88-0.91
- Cross-validated on 5 folds

## 🔧 Configuration

### Backend (.env)

```env
APP_NAME=House Price Predictor
APP_VERSION=1.0.0
DEBUG=True

# SQLite (default, no setup needed)
DATABASE_URL=sqlite:///./house_prices.db

# PostgreSQL (optional)
# DATABASE_URL=postgresql://user:password@localhost:5432/house_prices
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
```

## 📱 Pages

1. **Home** - Dashboard with stats and quick actions
2. **Predict** - Form to enter house features and get predictions
3. **History** - View all past predictions
4. **Models** - Manage ML models, train new ones, view metrics

## 🎨 Design Decisions

- **Modern SaaS-like UI** with gradient accents
- **Responsive design** works on mobile and desktop
- **Loading states** for all async operations
- **Error handling** with user-friendly messages
- **Feature-based folder structure** for scalability

## 🔐 Security Notes

- CORS is configured for localhost development
- No authentication implemented (add for production)
- Input validation via Pydantic schemas

## 📝 License

Kerkeni Amir 

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
