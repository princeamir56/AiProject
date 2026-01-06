"""ML Configuration"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml" / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Training data path
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature configuration
NUMERIC_FEATURES = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold'
]

CATEGORICAL_FEATURES = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition'
]

# Simplified features for the prediction form (user-friendly subset)
SIMPLE_FEATURES = {
    'numeric': [
        'OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF',
        'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'TotRmsAbvGrd', 'GarageCars', 'GarageArea'
    ],
    'categorical': [
        'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
        'CentralAir', 'KitchenQual'
    ]
}

TARGET = 'SalePrice'
