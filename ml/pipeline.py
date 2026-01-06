"""ML Pipeline for House Price Prediction"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
from pathlib import Path
import json

from .config import (
    TRAIN_DATA_PATH, MODELS_DIR, RANDOM_STATE, TEST_SIZE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, SIMPLE_FEATURES
)


class HousePriceModel:
    """House Price Prediction Model with scikit-learn Pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.metrics = {}
        self.version = None
        self.feature_names = []
        
    def _create_pipeline(self) -> Pipeline:
        """Create the preprocessing and model pipeline"""
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='drop'
        )
        
        # Full pipeline with Gradient Boosting model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=RANDOM_STATE
            ))
        ])
        
        return pipeline
    
    def load_data(self) -> pd.DataFrame:
        """Load training data"""
        df = pd.read_csv(TRAIN_DATA_PATH)
        return df
    
    def train(self) -> dict:
        """Train the model and return metrics"""
        print("Loading data...")
        df = self.load_data()
        
        # Prepare features and target
        X = df.drop(columns=['Id', TARGET])
        y = df[TARGET]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        print("Creating pipeline...")
        self.pipeline = self._create_pipeline()
        
        print("Training model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error'
        )
        cv_rmse = -cv_scores.mean()
        
        self.metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'cv_rmse': float(cv_rmse),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"CV RMSE: ${cv_rmse:,.2f}")
        
        return self.metrics
    
    def save(self, model_name: str = None) -> dict:
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Generate version
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name is None:
            model_name = f"house_price_model_{self.version}"
        
        model_path = MODELS_DIR / f"{model_name}.joblib"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        # Save model
        joblib.dump(self.pipeline, model_path)
        
        # Save metadata
        metadata = {
            'version': self.version,
            'model_name': model_name,
            'model_path': str(model_path),
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'created_at': datetime.now().isoformat(),
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'simple_features': SIMPLE_FEATURES
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        
        return metadata
    
    def load(self, model_path: str = None) -> None:
        """Load a trained model"""
        if model_path is None:
            # Load latest model
            model_files = list(MODELS_DIR.glob("house_price_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Train a model first.")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        else:
            model_path = Path(model_path)
        
        self.pipeline = joblib.load(model_path)
        
        # Load metadata
        metadata_path = model_path.with_suffix('.json').parent / (model_path.stem + '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.version = metadata.get('version')
                self.metrics = metadata.get('metrics', {})
                self.feature_names = metadata.get('feature_names', [])
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, features: dict) -> float:
        """Make a prediction for a single house"""
        if self.pipeline is None:
            self.load()
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns exist
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        
        return float(prediction)
    
    def predict_batch(self, features_list: list) -> list:
        """Make predictions for multiple houses"""
        if self.pipeline is None:
            self.load()
        
        df = pd.DataFrame(features_list)
        
        # Ensure all required columns exist
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        
        predictions = self.pipeline.predict(df)
        
        return predictions.tolist()
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the model"""
        if self.pipeline is None:
            self.load()
        
        regressor = self.pipeline.named_steps['regressor']
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Numeric features
        feature_names.extend(NUMERIC_FEATURES)
        
        # Categorical features (after one-hot encoding)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names.extend(cat_features)
        
        importances = regressor.feature_importances_
        
        # Sort by importance
        importance_dict = dict(zip(feature_names[:len(importances)], importances.tolist()))
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])  # Top 20 features
        
        return sorted_importance


# Singleton instance
_model_instance = None

def get_model() -> HousePriceModel:
    """Get or create the model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = HousePriceModel()
    return _model_instance
