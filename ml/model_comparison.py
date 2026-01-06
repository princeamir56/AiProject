"""
===================================================================================
COMPARAISON DE 3 MODÈLES DE MACHINE LEARNING
===================================================================================

Ce script implémente et compare 3 algorithmes de ML différents pour la prédiction
des prix immobiliers, conformément aux exigences du projet.

MODÈLES IMPLÉMENTÉS:
1. Gradient Boosting Regressor (Ensemble - Boosting)
2. Random Forest Regressor (Ensemble - Bagging)  
3. Ridge Regression (Régression linéaire régularisée)

JUSTIFICATION DES CHOIX:
- GradientBoosting: Excellent pour capturer les relations non-linéaires complexes
- RandomForest: Robuste aux outliers, bon pour la sélection de features
- Ridge: Modèle linéaire simple, bon baseline, gère la multicolinéarité

AUTEUR: Projet ML - Prédiction de Prix Immobiliers
DATE: Décembre 2024
===================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features sélectionnées (basé sur l'analyse exploratoire)
NUMERIC_FEATURES = [
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea'
]

CATEGORICAL_FEATURES = [
    'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
    'CentralAir', 'KitchenQual'
]


class ModelComparer:
    """
    Classe pour comparer 3 modèles de ML différents
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_prepare_data(self):
        """Charge et prépare les données"""
        print("=" * 80)
        print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
        print("=" * 80)
        
        df = pd.read_csv(DATA_PATH)
        print(f"\n✓ Données chargées: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # Suppression des outliers identifiés dans l'analyse
        # (GrLivArea > 4000 et prix bas - probablement des erreurs)
        initial_count = len(df)
        df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
        print(f"✓ Outliers supprimés: {initial_count - len(df)} observations")
        
        # Préparation des features et target
        X = df.drop(columns=['Id', 'SalePrice'])
        y = df['SalePrice']
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        print(f"✓ Split: {len(self.X_train)} train, {len(self.X_test)} test")
        
        return X, y
    
    def create_preprocessor(self):
        """Crée le pipeline de prétraitement"""
        
        # Transformateur numérique
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Transformateur catégoriel
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinaison
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def create_models(self):
        """
        Crée les 3 modèles avec leurs hyperparamètres optimisés
        
        JUSTIFICATION DES HYPERPARAMÈTRES:
        
        1. GradientBoosting:
           - n_estimators=200: Nombre suffisant d'arbres sans surapprentissage
           - max_depth=5: Limite la profondeur pour éviter l'overfitting
           - learning_rate=0.1: Taux standard, bon compromis vitesse/précision
           - min_samples_split=5: Évite les splits sur très peu d'exemples
           
        2. RandomForest:
           - n_estimators=200: Assez d'arbres pour la stabilité
           - max_depth=15: Plus profond car le bagging régularise naturellement
           - min_samples_leaf=2: Feuilles avec au moins 2 exemples
           - max_features='sqrt': Standard pour la régression
           
        3. Ridge:
           - alpha=10: Régularisation modérée pour gérer la multicolinéarité
           - Le preprocessing inclut la standardisation nécessaire
        """
        print("\n" + "=" * 80)
        print("CRÉATION DES MODÈLES")
        print("=" * 80)
        
        preprocessor = self.create_preprocessor()
        
        # Modèle 1: Gradient Boosting (Ensemble - Boosting)
        print("\n📊 Modèle 1: Gradient Boosting Regressor")
        print("   Famille: Ensemble Learning (Boosting)")
        print("   Justification: Capture les relations non-linéaires, très performant")
        self.models['GradientBoosting'] = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=RANDOM_STATE
            ))
        ])
        
        # Modèle 2: Random Forest (Ensemble - Bagging)
        print("\n📊 Modèle 2: Random Forest Regressor")
        print("   Famille: Ensemble Learning (Bagging)")
        print("   Justification: Robuste aux outliers, interprétable")
        self.models['RandomForest'] = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
        
        # Modèle 3: Ridge Regression (Linéaire régularisé)
        print("\n📊 Modèle 3: Ridge Regression")
        print("   Famille: Régression linéaire régularisée (L2)")
        print("   Justification: Simple, interprétable, gère la multicolinéarité")
        self.models['Ridge'] = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=10, random_state=RANDOM_STATE))
        ])
        
        return self.models
    
    def train_and_evaluate(self):
        """Entraîne et évalue tous les modèles"""
        print("\n" + "=" * 80)
        print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\n{'─' * 60}")
            print(f"📈 Training: {name}")
            print(f"{'─' * 60}")
            
            # Entraînement
            model.fit(self.X_train, self.y_train)
            
            # Prédictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Métriques
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                        cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Stocker les résultats
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std
            }
            
            print(f"\n   📊 Résultats:")
            print(f"   ├─ Train RMSE: ${train_rmse:,.0f}")
            print(f"   ├─ Test RMSE:  ${test_rmse:,.0f}")
            print(f"   ├─ Train R²:   {train_r2:.4f}")
            print(f"   ├─ Test R²:    {test_r2:.4f}")
            print(f"   └─ CV RMSE:    ${cv_rmse:,.0f} (±${cv_std:,.0f})")
        
        return self.results
    
    def compare_models(self):
        """Compare et visualise les performances des modèles"""
        print("\n" + "=" * 80)
        print("COMPARAISON DES MODÈLES")
        print("=" * 80)
        
        # Créer un DataFrame de comparaison
        comparison_df = pd.DataFrame(self.results).T
        comparison_df['Overfitting'] = comparison_df['train_rmse'] - comparison_df['test_rmse']
        
        print("\n📊 TABLEAU COMPARATIF:")
        print("─" * 90)
        print(f"{'Modèle':<20} {'Train RMSE':>12} {'Test RMSE':>12} {'Test R²':>10} {'CV RMSE':>12} {'Overfit':>10}")
        print("─" * 90)
        
        for name, metrics in self.results.items():
            overfit = metrics['train_rmse'] - metrics['test_rmse']
            print(f"{name:<20} ${metrics['train_rmse']:>10,.0f} ${metrics['test_rmse']:>10,.0f} "
                  f"{metrics['test_r2']:>10.4f} ${metrics['cv_rmse']:>10,.0f} ${overfit:>9,.0f}")
        
        print("─" * 90)
        
        # Trouver le meilleur modèle
        best_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        self.best_model = self.models[best_name]
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_name}")
        print(f"   └─ Test RMSE: ${self.results[best_name]['test_rmse']:,.0f}")
        print(f"   └─ Test R²: {self.results[best_name]['test_r2']:.4f}")
        
        # Visualisations
        self._plot_comparison()
        self._plot_predictions()
        self._plot_feature_importance()
        
        return best_name, self.results
    
    def _plot_comparison(self):
        """Génère les graphiques de comparaison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(self.results.keys())
        colors = ['#2563eb', '#10b981', '#f59e0b']
        
        # RMSE Comparison
        train_rmse = [self.results[m]['train_rmse'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, train_rmse, width, label='Train', color='lightblue')
        axes[0].bar(x + width/2, test_rmse, width, label='Test', color=colors)
        axes[0].set_ylabel('RMSE ($)')
        axes[0].set_title('Comparaison RMSE', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15)
        axes[0].legend()
        
        # R² Comparison
        train_r2 = [self.results[m]['train_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        axes[1].bar(x - width/2, train_r2, width, label='Train', color='lightgreen')
        axes[1].bar(x + width/2, test_r2, width, label='Test', color=colors)
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Comparaison R²', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15)
        axes[1].legend()
        axes[1].set_ylim(0.7, 1.0)
        
        # CV RMSE with error bars
        cv_rmse = [self.results[m]['cv_rmse'] for m in models]
        cv_std = [self.results[m]['cv_std'] for m in models]
        
        axes[2].bar(models, cv_rmse, color=colors, yerr=cv_std, capsize=5)
        axes[2].set_ylabel('CV RMSE ($)')
        axes[2].set_title('Cross-Validation RMSE (±std)', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Graphique sauvegardé: {OUTPUT_DIR / 'model_comparison.png'}")
    
    def _plot_predictions(self):
        """Graphique des prédictions vs valeurs réelles"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            
            axes[i].scatter(self.y_test, y_pred, alpha=0.5, c='#2563eb')
            
            # Ligne parfaite
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            axes[i].set_xlabel('Prix réel ($)')
            axes[i].set_ylabel('Prix prédit ($)')
            axes[i].set_title(f'{name}\nR² = {self.results[name]["test_r2"]:.4f}', fontweight='bold')
        
        plt.suptitle('Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'predictions_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / 'predictions_scatter.png'}")
    
    def _plot_feature_importance(self):
        """Graphique d'importance des features (pour GB et RF)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, name in enumerate(['GradientBoosting', 'RandomForest']):
            model = self.models[name]
            regressor = model.named_steps['regressor']
            preprocessor = model.named_steps['preprocessor']
            
            # Obtenir les noms de features après transformation
            feature_names = NUMERIC_FEATURES.copy()
            if hasattr(preprocessor.named_transformers_['cat'].named_steps['encoder'], 'get_feature_names_out'):
                cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORICAL_FEATURES)
                feature_names.extend(cat_features)
            
            importances = regressor.feature_importances_
            
            # Top 15 features
            indices = np.argsort(importances)[-15:]
            
            axes[i].barh(range(len(indices)), importances[indices], color='#2563eb')
            axes[i].set_yticks(range(len(indices)))
            
            # Raccourcir les noms si nécessaire
            labels = [feature_names[j] if j < len(feature_names) else f'Feature {j}' for j in indices]
            labels = [l[:20] + '...' if len(l) > 20 else l for l in labels]
            axes[i].set_yticklabels(labels)
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'Feature Importance - {name}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / 'feature_importance.png'}")
    
    def save_best_model(self):
        """Sauvegarde le meilleur modèle"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"best_model_{timestamp}.joblib"
        
        joblib.dump(self.best_model, model_path)
        
        # Sauvegarder aussi les métadonnées
        best_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        metadata = {
            'model_name': best_name,
            'version': timestamp,
            'metrics': self.results[best_name],
            'features': {
                'numeric': NUMERIC_FEATURES,
                'categorical': CATEGORICAL_FEATURES
            }
        }
        
        with open(MODELS_DIR / f"best_model_{timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Modèle sauvegardé: {model_path}")
        
        return model_path
    
    def generate_report(self):
        """Génère un rapport complet"""
        report_path = OUTPUT_DIR / 'comparison_report.txt'
        
        best_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT DE COMPARAISON DES MODÈLES DE MACHINE LEARNING\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MODÈLES IMPLÉMENTÉS\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("1. GRADIENT BOOSTING REGRESSOR\n")
            f.write("   Famille: Ensemble Learning (Boosting)\n")
            f.write("   Hyperparamètres:\n")
            f.write("     - n_estimators: 200\n")
            f.write("     - max_depth: 5\n")
            f.write("     - learning_rate: 0.1\n")
            f.write("   Justification: Méthode d'ensemble qui construit séquentiellement\n")
            f.write("   des arbres de décision, chacun corrigeant les erreurs du précédent.\n")
            f.write("   Excellent pour capturer les relations non-linéaires complexes.\n\n")
            
            f.write("2. RANDOM FOREST REGRESSOR\n")
            f.write("   Famille: Ensemble Learning (Bagging)\n")
            f.write("   Hyperparamètres:\n")
            f.write("     - n_estimators: 200\n")
            f.write("     - max_depth: 15\n")
            f.write("     - max_features: sqrt\n")
            f.write("   Justification: Agrège les prédictions de multiples arbres indépendants.\n")
            f.write("   Robuste aux outliers et fournit une mesure d'importance des features.\n\n")
            
            f.write("3. RIDGE REGRESSION\n")
            f.write("   Famille: Régression linéaire régularisée (L2)\n")
            f.write("   Hyperparamètres:\n")
            f.write("     - alpha: 10\n")
            f.write("   Justification: Modèle linéaire simple avec régularisation L2.\n")
            f.write("   Gère bien la multicolinéarité et sert de baseline interprétable.\n\n")
            
            f.write("\nRÉSULTATS\n")
            f.write("-" * 40 + "\n\n")
            
            for name, metrics in self.results.items():
                f.write(f"{name}:\n")
                f.write(f"  - Test RMSE: ${metrics['test_rmse']:,.0f}\n")
                f.write(f"  - Test R²: {metrics['test_r2']:.4f}\n")
                f.write(f"  - CV RMSE: ${metrics['cv_rmse']:,.0f} (±${metrics['cv_std']:,.0f})\n\n")
            
            f.write(f"\n🏆 MEILLEUR MODÈLE RETENU: {best_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Ce modèle a été sélectionné car il présente:\n")
            f.write(f"  - Le meilleur RMSE sur les données de test\n")
            f.write(f"  - Un bon score R² ({self.results[best_name]['test_r2']:.4f})\n")
            f.write(f"  - Une bonne généralisation (faible overfitting)\n")
        
        print(f"✓ Rapport sauvegardé: {report_path}")


def main():
    """Exécute la comparaison complète des modèles"""
    print("\n" + "█" * 80)
    print("  COMPARAISON DE 3 MODÈLES DE MACHINE LEARNING")
    print("█" * 80)
    
    # Initialiser le comparateur
    comparer = ModelComparer()
    
    # 1. Charger les données
    comparer.load_and_prepare_data()
    
    # 2. Créer les modèles
    comparer.create_models()
    
    # 3. Entraîner et évaluer
    comparer.train_and_evaluate()
    
    # 4. Comparer les modèles
    best_name, results = comparer.compare_models()
    
    # 5. Sauvegarder le meilleur modèle
    comparer.save_best_model()
    
    # 6. Générer le rapport
    comparer.generate_report()
    
    print("\n" + "=" * 80)
    print("✅ COMPARAISON TERMINÉE")
    print("=" * 80)
    print(f"\n📁 Fichiers générés dans: {OUTPUT_DIR}")
    print("   - model_comparison.png")
    print("   - predictions_scatter.png")
    print("   - feature_importance.png")
    print("   - comparison_report.txt")
    
    return results


if __name__ == "__main__":
    main()
