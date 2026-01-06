"""Training script for House Price Prediction Model"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.pipeline import HousePriceModel


def main():
    """Train and save the model"""
    print("=" * 50)
    print("House Price Prediction - Model Training")
    print("=" * 50)
    print()
    
    # Create model instance
    model = HousePriceModel()
    
    # Train
    metrics = model.train()
    
    print()
    print("-" * 50)
    print("Training completed successfully!")
    print("-" * 50)
    print()
    
    # Save model
    metadata = model.save()
    
    print()
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Version: {metadata['version']}")
    print(f"Path: {metadata['model_path']}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print()
    
    # Test prediction
    print("Testing prediction...")
    test_house = {
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2005,
        'TotalBsmtSF': 1000,
        'GrLivArea': 1500,
        'FullBath': 2,
        'HalfBath': 1,
        'BedroomAbvGr': 3,
        'TotRmsAbvGrd': 7,
        'GarageCars': 2,
        'GarageArea': 500,
        'MSZoning': 'RL',
        'Neighborhood': 'CollgCr',
        'BldgType': '1Fam',
        'HouseStyle': '2Story',
        'CentralAir': 'Y',
        'KitchenQual': 'Gd'
    }
    
    prediction = model.predict(test_house)
    print(f"Test prediction: ${prediction:,.2f}")
    print()
    
    return metadata


if __name__ == "__main__":
    main()
