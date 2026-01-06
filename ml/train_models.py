"""
Model Training Script for Ames Housing Dataset
Trains 3 models with 5-fold CV, selects best, saves pipeline and metrics.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import (
    create_preprocessing_pipeline,
    prepare_data,
    inverse_transform_predictions,
    get_numeric_features,
    get_categorical_features,
)

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_models() -> dict:
    """
    Define the 3 models to compare.
    
    Returns:
        Dictionary of model name -> model instance
    """
    return {
        'Ridge': Ridge(alpha=10.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
    }


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> dict:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Scikit-learn model or pipeline
        X: Features
        y: Target (log-transformed)
        cv: Number of folds
    
    Returns:
        Dictionary with RMSE, MAE, R2 scores
    """
    scoring = {
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate metrics (note: MSE is negated by sklearn)
    rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
    mae_scores = -cv_results['test_neg_mae']
    r2_scores = cv_results['test_r2']
    
    return {
        'rmse_mean': float(rmse_scores.mean()),
        'rmse_std': float(rmse_scores.std()),
        'mae_mean': float(mae_scores.mean()),
        'mae_std': float(mae_scores.std()),
        'r2_mean': float(r2_scores.mean()),
        'r2_std': float(r2_scores.std()),
        'rmse_folds': rmse_scores.tolist(),
        'mae_folds': mae_scores.tolist(),
        'r2_folds': r2_scores.tolist()
    }


def plot_cv_comparison(results: dict, save_path: Path):
    """Plot comparison of model CV results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.6
    
    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # RMSE
    ax = axes[0]
    rmse_means = [results[m]['rmse_mean'] for m in models]
    rmse_stds = [results[m]['rmse_std'] for m in models]
    bars = ax.bar(x, rmse_means, width, yerr=rmse_stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('RMSE (log scale)', fontsize=11)
    ax.set_title('RMSE Comparison (5-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    for bar, val in zip(bars, rmse_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE
    ax = axes[1]
    mae_means = [results[m]['mae_mean'] for m in models]
    mae_stds = [results[m]['mae_std'] for m in models]
    bars = ax.bar(x, mae_means, width, yerr=mae_stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('MAE (log scale)', fontsize=11)
    ax.set_title('MAE Comparison (5-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # R2
    ax = axes[2]
    r2_means = [results[m]['r2_mean'] for m in models]
    r2_stds = [results[m]['r2_std'] for m in models]
    bars = ax.bar(x, r2_means, width, yerr=r2_stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('R² Comparison (5-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path):
    """Plot residual analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=30, c='#3498db')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values (log scale)', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title(f'{model_name}: Residuals vs Predicted', fontsize=12, fontweight='bold')
    
    # Residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=50, edgecolor='white', color='#2ecc71', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(x=residuals.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {residuals.mean():.4f}')
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path):
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, c='#3498db')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Actual Values (log scale)', fontsize=11)
    ax.set_ylabel('Predicted Values (log scale)', fontsize=11)
    ax.set_title(f'{model_name}: Predicted vs Actual\nRMSE={rmse:.4f}, R²={r2:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    save_path: Path,
    cv: int = 5
):
    """Plot learning curve for the model."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Convert to RMSE
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)
    
    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    test_mean = test_rmse.mean(axis=1)
    test_std = test_rmse.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                    alpha=0.2, color='#3498db')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, 
                    alpha=0.2, color='#e74c3c')
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#3498db', linewidth=2, 
            label='Training Score')
    ax.plot(train_sizes_abs, test_mean, 'o-', color='#e74c3c', linewidth=2, 
            label='Validation Score')
    
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('RMSE (log scale)', fontsize=11)
    ax.set_title(f'{model_name}: Learning Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def train_and_evaluate():
    """Main training and evaluation function."""
    print("=" * 60)
    print("AMES HOUSING - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset shape: {df.shape}")
    
    # Prepare data
    print("\n2. Preparing data...")
    X, y = prepare_data(df, target_col='SalePrice', log_transform_target=True)
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Target (log1p) range: {y.min():.2f} - {y.max():.2f}")
    
    # Create preprocessing pipeline
    print("\n3. Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(scale_numeric=True)
    
    # Get models
    models = get_models()
    print(f"\n4. Training {len(models)} models with 5-fold CV...")
    
    # Evaluate each model
    results = {}
    pipelines = {}
    
    for name, model in models.items():
        print(f"\n   Training: {name}")
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Cross-validation
        cv_results = evaluate_model_cv(pipeline, X, y, cv=5)
        results[name] = cv_results
        pipelines[name] = pipeline
        
        print(f"   RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
        print(f"   MAE:  {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
        print(f"   R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    # Select best model (lowest RMSE)
    print("\n5. Selecting best model...")
    best_model_name = min(results, key=lambda x: results[x]['rmse_mean'])
    best_pipeline = pipelines[best_model_name]
    best_results = results[best_model_name]
    
    print(f"   Best Model: {best_model_name}")
    print(f"   RMSE: {best_results['rmse_mean']:.4f}")
    print(f"   R²: {best_results['r2_mean']:.4f}")
    
    # Fit best model on full data
    print("\n6. Fitting best model on full dataset...")
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    
    # Generate plots
    print("\n7. Generating plots...")
    
    # CV comparison
    plot_cv_comparison(results, FIGURES_DIR / 'cv_comparison.png')
    
    # Residuals
    plot_residuals(y.values, y_pred, best_model_name, FIGURES_DIR / 'residuals.png')
    
    # Predicted vs Actual
    plot_predicted_vs_actual(y.values, y_pred, best_model_name, FIGURES_DIR / 'predicted_vs_actual.png')
    
    # Learning curve
    plot_learning_curve(best_pipeline, X, y, best_model_name, FIGURES_DIR / 'learning_curve.png')
    
    # Save best pipeline
    print("\n8. Saving best model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_pipeline_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(best_pipeline, model_path)
    print(f"   Model saved: {model_path}")
    
    # Save metrics
    metrics = {
        'best_model': best_model_name,
        'timestamp': timestamp,
        'dataset': {
            'instances': len(df),
            'features': len(X.columns),
            'numeric_features': get_numeric_features(),
            'categorical_features': get_categorical_features()
        },
        'cv_results': results,
        'best_model_metrics': {
            'rmse': best_results['rmse_mean'],
            'rmse_std': best_results['rmse_std'],
            'mae': best_results['mae_mean'],
            'mae_std': best_results['mae_std'],
            'r2': best_results['r2_mean'],
            'r2_std': best_results['r2_std']
        },
        'model_path': str(model_path)
    }
    
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved: {metrics_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"  RMSE (CV): {best_results['rmse_mean']:.4f} ± {best_results['rmse_std']:.4f}")
    print(f"  MAE (CV):  {best_results['mae_mean']:.4f} ± {best_results['mae_std']:.4f}")
    print(f"  R² (CV):   {best_results['r2_mean']:.4f} ± {best_results['r2_std']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nAll CV Results:")
    for name, res in results.items():
        marker = "★" if name == best_model_name else " "
        print(f"  {marker} {name}: RMSE={res['rmse_mean']:.4f}, R²={res['r2_mean']:.4f}")
    
    return best_pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
