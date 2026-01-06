"""
Preprocessing Pipeline for Ames Housing Dataset
Clean, modular preprocessing with imputation, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Tuple, Optional


# Features to use for modeling (selected based on correlation and domain knowledge)
NUMERIC_FEATURES = [
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'TotalBsmtSF', 'GrLivArea', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'LotArea', 'LotFrontage'
]

CATEGORICAL_FEATURES = [
    'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'HeatingQC', 'CentralAir', 'KitchenQual', 'GarageType',
    'SaleType', 'SaleCondition'
]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from the dataframe."""
    
    def __init__(self, features: List[str]):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Handle missing columns gracefully
        available_features = [f for f in self.features if f in X.columns]
        return X[available_features].copy()


def get_numeric_features() -> List[str]:
    """Get list of numeric feature names."""
    return NUMERIC_FEATURES.copy()


def get_categorical_features() -> List[str]:
    """Get list of categorical feature names."""
    return CATEGORICAL_FEATURES.copy()


def create_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline with imputation, encoding, and optional scaling.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        scale_numeric: Whether to apply StandardScaler to numeric features
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES.copy()
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES.copy()
    
    # Numeric pipeline: Impute with median, optionally scale
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    
    numeric_pipeline = Pipeline(steps=numeric_steps)
    
    # Categorical pipeline: Impute with constant, one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop',  # Drop columns not specified
        verbose_feature_names_out=False
    )
    
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'SalePrice',
    log_transform_target: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        log_transform_target: Whether to apply log1p to target
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Separate features and target
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col].copy()
    
    # Log transform target for better distribution
    if log_transform_target:
        y = np.log1p(y)
    
    return X, y


def inverse_transform_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Convert log-transformed predictions back to original scale."""
    return np.expm1(y_pred)


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / "data" / "train.csv"
    df = pd.read_csv(data_path)
    
    print("Testing preprocessing pipeline...")
    print(f"Original shape: {df.shape}")
    
    X, y = prepare_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Processed shape: {X_processed.shape}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print("\n✓ Preprocessing pipeline works correctly!")
