"""ML Service - Bridge between FastAPI and ML module"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import HousePriceModel, get_model
from ml.config import MODELS_DIR, SIMPLE_FEATURES


class MLService:
    """Service for ML operations"""
    
    def __init__(self):
        self._model: Optional[HousePriceModel] = None
        self._model_loaded = False
    
    @property
    def model(self) -> HousePriceModel:
        """Get or load the model"""
        if not self._model_loaded:
            self._model = get_model()
            try:
                self._model.load()
                self._model_loaded = True
            except FileNotFoundError:
                # Model not trained yet
                pass
        return self._model
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        try:
            _ = self.model
            return self._model_loaded
        except Exception:
            return False
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Make a prediction"""
        if not self._model_loaded:
            try:
                self.model.load()
                self._model_loaded = True
            except FileNotFoundError:
                raise ValueError("No trained model available. Please train a model first.")
        
        return self.model.predict(features)
    
    def train(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Train a new model"""
        model = HousePriceModel()
        metrics = model.train()
        metadata = model.save(model_name)
        
        # Update the loaded model
        self._model = model
        self._model_loaded = True
        
        return {
            'metrics': metrics,
            'metadata': metadata
        }
    
    def get_model_version(self) -> Optional[str]:
        """Get current model version"""
        if self._model_loaded and self._model:
            return self._model.version
        return None
    
    def get_model_metrics(self) -> Dict[str, float]:
        """Get current model metrics"""
        if self._model_loaded and self._model:
            return self._model.metrics
        return {}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        for metadata_file in MODELS_DIR.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    models.append({
                        'name': metadata.get('model_name'),
                        'version': metadata.get('version'),
                        'metrics': metadata.get('metrics'),
                        'created_at': metadata.get('created_at'),
                        'path': metadata.get('model_path')
                    })
            except Exception:
                continue
        
        # Sort by creation date (newest first), handle None values
        models.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        
        return models
    
    def load_model_by_version(self, version: str) -> bool:
        """Load a specific model version"""
        model_path = MODELS_DIR / f"house_price_model_{version}.joblib"
        
        if not model_path.exists():
            return False
        
        self._model = HousePriceModel()
        self._model.load(str(model_path))
        self._model_loaded = True
        
        return True
    
    def get_dropdown_options(self) -> Dict[str, List[Dict[str, str]]]:
        """Get dropdown options for the form"""
        return {
            'MSZoning': [
                {'value': 'A', 'label': 'Agriculture'},
                {'value': 'C', 'label': 'Commercial'},
                {'value': 'FV', 'label': 'Floating Village Residential'},
                {'value': 'I', 'label': 'Industrial'},
                {'value': 'RH', 'label': 'Residential High Density'},
                {'value': 'RL', 'label': 'Residential Low Density'},
                {'value': 'RP', 'label': 'Residential Low Density Park'},
                {'value': 'RM', 'label': 'Residential Medium Density'}
            ],
            'Neighborhood': [
                {'value': 'Blmngtn', 'label': 'Bloomington Heights'},
                {'value': 'Blueste', 'label': 'Bluestem'},
                {'value': 'BrDale', 'label': 'Briardale'},
                {'value': 'BrkSide', 'label': 'Brookside'},
                {'value': 'ClearCr', 'label': 'Clear Creek'},
                {'value': 'CollgCr', 'label': 'College Creek'},
                {'value': 'Crawfor', 'label': 'Crawford'},
                {'value': 'Edwards', 'label': 'Edwards'},
                {'value': 'Gilbert', 'label': 'Gilbert'},
                {'value': 'IDOTRR', 'label': 'Iowa DOT and Rail Road'},
                {'value': 'MeadowV', 'label': 'Meadow Village'},
                {'value': 'Mitchel', 'label': 'Mitchell'},
                {'value': 'NAmes', 'label': 'North Ames'},
                {'value': 'NoRidge', 'label': 'Northridge'},
                {'value': 'NPkVill', 'label': 'Northpark Villa'},
                {'value': 'NridgHt', 'label': 'Northridge Heights'},
                {'value': 'NWAmes', 'label': 'Northwest Ames'},
                {'value': 'OldTown', 'label': 'Old Town'},
                {'value': 'SWISU', 'label': 'South & West of ISU'},
                {'value': 'Sawyer', 'label': 'Sawyer'},
                {'value': 'SawyerW', 'label': 'Sawyer West'},
                {'value': 'Somerst', 'label': 'Somerset'},
                {'value': 'StoneBr', 'label': 'Stone Brook'},
                {'value': 'Timber', 'label': 'Timberland'},
                {'value': 'Veenker', 'label': 'Veenker'}
            ],
            'BldgType': [
                {'value': '1Fam', 'label': 'Single-family Detached'},
                {'value': '2FmCon', 'label': 'Two-family Conversion'},
                {'value': 'Duplx', 'label': 'Duplex'},
                {'value': 'TwnhsE', 'label': 'Townhouse End Unit'},
                {'value': 'TwnhsI', 'label': 'Townhouse Inside Unit'}
            ],
            'HouseStyle': [
                {'value': '1Story', 'label': 'One Story'},
                {'value': '1.5Fin', 'label': 'One and Half Story, Finished'},
                {'value': '1.5Unf', 'label': 'One and Half Story, Unfinished'},
                {'value': '2Story', 'label': 'Two Story'},
                {'value': '2.5Fin', 'label': 'Two and Half Story, Finished'},
                {'value': '2.5Unf', 'label': 'Two and Half Story, Unfinished'},
                {'value': 'SFoyer', 'label': 'Split Foyer'},
                {'value': 'SLvl', 'label': 'Split Level'}
            ],
            'KitchenQual': [
                {'value': 'Ex', 'label': 'Excellent'},
                {'value': 'Gd', 'label': 'Good'},
                {'value': 'TA', 'label': 'Typical/Average'},
                {'value': 'Fa', 'label': 'Fair'},
                {'value': 'Po', 'label': 'Poor'}
            ]
        }


# Singleton instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get or create the ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
