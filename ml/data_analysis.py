"""
===================================================================================
ANALYSE EXPLORATOIRE DES DONNÉES - AMES HOUSING DATASET
===================================================================================

Ce script effectue une analyse complète du jeu de données Ames Housing pour la 
prédiction des prix immobiliers.

OBJECTIFS:
1. Description du dataset (instances, attributs, types)
2. Analyse de la variable cible (SalePrice)
3. Analyse des corrélations
4. Détection des valeurs manquantes et aberrantes
5. Sélection des attributs pertinents

AUTEUR: Projet ML - Prédiction de Prix Immobiliers
DATE: Décembre 2024
===================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Chemins
DATA_PATH = Path(__file__).parent.parent / "data" / "train.csv"
OUTPUT_DIR = Path(__file__).parent / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Charge le jeu de données"""
    print("=" * 80)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Données chargées depuis: {DATA_PATH}")
    print(f"✓ Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    return df


def describe_dataset(df):
    """
    Section B.2: Description du dataset
    - Nombre d'instances et d'attributs
    - Types de variables
    - Variable cible
    """
    print("\n" + "=" * 80)
    print("B.2 - DESCRIPTION DU DATASET")
    print("=" * 80)
    
    print("\n📊 INFORMATIONS GÉNÉRALES:")
    print("-" * 40)
    print(f"  • Nombre d'instances (lignes): {df.shape[0]}")
    print(f"  • Nombre d'attributs (colonnes): {df.shape[1]}")
    print(f"  • Variable cible: SalePrice (prix de vente)")
    
    # Types de variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📈 TYPES DE VARIABLES:")
    print("-" * 40)
    print(f"  • Variables numériques: {len(numeric_cols)}")
    print(f"  • Variables catégorielles: {len(categorical_cols)}")
    
    print(f"\n  Variables numériques principales:")
    for col in numeric_cols[:10]:
        print(f"    - {col}: {df[col].dtype}")
    
    print(f"\n  Variables catégorielles principales:")
    for col in categorical_cols[:10]:
        print(f"    - {col}: {df[col].nunique()} valeurs uniques")
    
    # Statistiques descriptives de la variable cible
    print(f"\n🎯 VARIABLE CIBLE - SalePrice:")
    print("-" * 40)
    print(f"  • Minimum: ${df['SalePrice'].min():,.0f}")
    print(f"  • Maximum: ${df['SalePrice'].max():,.0f}")
    print(f"  • Moyenne: ${df['SalePrice'].mean():,.0f}")
    print(f"  • Médiane: ${df['SalePrice'].median():,.0f}")
    print(f"  • Écart-type: ${df['SalePrice'].std():,.0f}")
    
    return numeric_cols, categorical_cols


def analyze_target_distribution(df):
    """Analyse la distribution de la variable cible"""
    print("\n" + "=" * 80)
    print("DISTRIBUTION DE LA VARIABLE CIBLE")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution originale
    axes[0].hist(df['SalePrice'], bins=50, color='#2563eb', edgecolor='white', alpha=0.7)
    axes[0].set_title('Distribution de SalePrice', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Prix de vente ($)')
    axes[0].set_ylabel('Fréquence')
    axes[0].axvline(df['SalePrice'].mean(), color='red', linestyle='--', label=f'Moyenne: ${df["SalePrice"].mean():,.0f}')
    axes[0].axvline(df['SalePrice'].median(), color='green', linestyle='--', label=f'Médiane: ${df["SalePrice"].median():,.0f}')
    axes[0].legend()
    
    # Distribution log-transformée
    axes[1].hist(np.log1p(df['SalePrice']), bins=50, color='#10b981', edgecolor='white', alpha=0.7)
    axes[1].set_title('Distribution de log(SalePrice)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('log(Prix de vente)')
    axes[1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Graphique sauvegardé: {OUTPUT_DIR / 'target_distribution.png'}")
    
    # Test de normalité
    from scipy import stats
    _, p_value = stats.normaltest(df['SalePrice'])
    _, p_value_log = stats.normaltest(np.log1p(df['SalePrice']))
    
    print(f"\n📊 Test de normalité (D'Agostino):")
    print(f"  • SalePrice: p-value = {p_value:.2e} {'(non normal)' if p_value < 0.05 else '(normal)'}")
    print(f"  • log(SalePrice): p-value = {p_value_log:.2e} {'(non normal)' if p_value_log < 0.05 else '(normal)'}")


def analyze_missing_values(df):
    """
    Section B.4: Nettoyage - Valeurs manquantes
    """
    print("\n" + "=" * 80)
    print("B.4 - ANALYSE DES VALEURS MANQUANTES")
    print("=" * 80)
    
    # Calcul des valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Valeurs Manquantes': missing.values,
        'Pourcentage (%)': missing_pct.values
    })
    missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(
        by='Pourcentage (%)', ascending=False
    )
    
    print(f"\n📊 Colonnes avec valeurs manquantes: {len(missing_df)}")
    print("\n Top 15 colonnes avec le plus de valeurs manquantes:")
    print("-" * 60)
    
    for _, row in missing_df.head(15).iterrows():
        bar = "█" * int(row['Pourcentage (%)'] / 5)
        print(f"  {row['Colonne']:<15} | {row['Valeurs Manquantes']:>4} | {row['Pourcentage (%)']:>5.1f}% {bar}")
    
    # Visualisation
    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Reds(missing_df['Pourcentage (%)'].head(20) / 100)
        bars = ax.barh(missing_df['Colonne'].head(20), missing_df['Pourcentage (%)'].head(20), color=colors)
        ax.set_xlabel('Pourcentage de valeurs manquantes (%)')
        ax.set_title('Top 20 colonnes avec valeurs manquantes', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, pct in zip(bars, missing_df['Pourcentage (%)'].head(20)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Graphique sauvegardé: {OUTPUT_DIR / 'missing_values.png'}")
    
    return missing_df


def analyze_correlations(df, numeric_cols):
    """
    Section B.3: Analyse des corrélations pour la sélection des attributs
    """
    print("\n" + "=" * 80)
    print("B.3 - ANALYSE DES CORRÉLATIONS")
    print("=" * 80)
    
    # Corrélation avec la variable cible
    correlations = df[numeric_cols].corrwith(df['SalePrice']).sort_values(ascending=False)
    
    print("\n📊 Top 15 variables les plus corrélées avec SalePrice:")
    print("-" * 60)
    
    for col, corr in correlations.head(16).items():
        if col != 'SalePrice':
            bar = "█" * int(abs(corr) * 20)
            sign = "+" if corr > 0 else "-"
            print(f"  {col:<20} | {sign}{abs(corr):.3f} {bar}")
    
    # Heatmap des corrélations
    top_features = correlations.head(12).index.tolist()
    correlation_matrix = df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Matrice de corrélation - Top 12 Features', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Graphique sauvegardé: {OUTPUT_DIR / 'correlation_matrix.png'}")
    
    # Variables retenues
    selected_numeric = correlations[abs(correlations) > 0.3].index.tolist()
    selected_numeric = [c for c in selected_numeric if c != 'SalePrice']
    
    print(f"\n✓ Variables numériques retenues (|corr| > 0.3): {len(selected_numeric)}")
    for col in selected_numeric:
        print(f"    - {col}: {correlations[col]:.3f}")
    
    return correlations, selected_numeric


def analyze_outliers(df):
    """
    Section B.4: Détection des valeurs aberrantes
    """
    print("\n" + "=" * 80)
    print("B.4 - DÉTECTION DES VALEURS ABERRANTES")
    print("=" * 80)
    
    key_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']
    
    fig, axes = plt.subplots(1, len(key_features), figsize=(16, 4))
    
    outlier_counts = {}
    
    for i, col in enumerate(key_features):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_counts[col] = len(outliers)
        
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(f'{col}\n({len(outliers)} outliers)', fontsize=10)
        axes[i].set_ylabel('Valeur')
    
    plt.suptitle('Boxplots des variables clés', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outliers_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Valeurs aberrantes détectées (méthode IQR):")
    print("-" * 40)
    for col, count in outlier_counts.items():
        print(f"  • {col}: {count} outliers ({count/len(df)*100:.1f}%)")
    
    # Scatter plot GrLivArea vs SalePrice
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.5, c='#2563eb')
    ax.set_xlabel('Surface habitable (GrLivArea) en sq ft')
    ax.set_ylabel('Prix de vente ($)')
    ax.set_title('GrLivArea vs SalePrice - Détection visuelle des outliers', fontsize=14, fontweight='bold')
    
    # Identifier les outliers potentiels (grandes surfaces, prix bas)
    outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
    ax.scatter(outliers['GrLivArea'], outliers['SalePrice'], c='red', s=100, label='Outliers potentiels')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_outliers.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Graphiques sauvegardés dans: {OUTPUT_DIR}")
    print(f"\n⚠️  Recommandation: {len(outliers)} observations avec GrLivArea > 4000 et prix < $300k")
    print("   Ces points pourraient être supprimés pour améliorer le modèle.")
    
    return outlier_counts


def analyze_categorical(df, categorical_cols):
    """Analyse des variables catégorielles"""
    print("\n" + "=" * 80)
    print("ANALYSE DES VARIABLES CATÉGORIELLES")
    print("=" * 80)
    
    # Variables catégorielles les plus importantes
    important_cats = ['Neighborhood', 'MSZoning', 'BldgType', 'HouseStyle', 'CentralAir', 'KitchenQual']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(important_cats):
        if col in df.columns:
            means = df.groupby(col)['SalePrice'].mean().sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(means)))
            axes[i].barh(means.index, means.values, color=colors)
            axes[i].set_xlabel('Prix moyen ($)')
            axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
            axes[i].invert_yaxis()
    
    plt.suptitle('Prix moyen par catégorie', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'categorical_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Graphique sauvegardé: {OUTPUT_DIR / 'categorical_analysis.png'}")
    
    # Afficher les catégories avec le plus grand impact
    print("\n📊 Impact des variables catégorielles sur le prix:")
    for col in important_cats:
        if col in df.columns:
            means = df.groupby(col)['SalePrice'].mean()
            print(f"\n  {col} ({df[col].nunique()} catégories):")
            print(f"    - Prix min moyen: ${means.min():,.0f} ({means.idxmin()})")
            print(f"    - Prix max moyen: ${means.max():,.0f} ({means.idxmax()})")


def feature_selection_summary(df, selected_numeric):
    """
    Section B.3: Résumé de la sélection des attributs
    """
    print("\n" + "=" * 80)
    print("B.3 - RÉSUMÉ DE LA SÉLECTION DES ATTRIBUTS")
    print("=" * 80)
    
    # Features numériques sélectionnées
    numeric_selected = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
        'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF'
    ]
    
    # Features catégorielles sélectionnées
    categorical_selected = [
        'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
        'CentralAir', 'KitchenQual', 'GarageFinish', 'Foundation'
    ]
    
    print("\n📌 VARIABLES NUMÉRIQUES SÉLECTIONNÉES:")
    print("-" * 50)
    for feat in numeric_selected:
        if feat in df.columns:
            corr = df[feat].corr(df['SalePrice'])
            print(f"  ✓ {feat:<20} (corr: {corr:+.3f})")
    
    print("\n📌 VARIABLES CATÉGORIELLES SÉLECTIONNÉES:")
    print("-" * 50)
    for feat in categorical_selected:
        if feat in df.columns:
            n_unique = df[feat].nunique()
            print(f"  ✓ {feat:<20} ({n_unique} catégories)")
    
    print("\n📌 VARIABLES EXCLUES:")
    print("-" * 50)
    excluded = {
        'PoolQC': 'Trop de valeurs manquantes (99.5%)',
        'MiscFeature': 'Trop de valeurs manquantes (96.3%)',
        'Alley': 'Trop de valeurs manquantes (93.8%)',
        'Fence': 'Trop de valeurs manquantes (80.8%)',
        'FireplaceQu': 'Trop de valeurs manquantes (47.3%)',
        'Id': 'Identifiant, non informatif',
        'Utilities': 'Quasi-constante (99.9% = AllPub)',
        'Street': 'Quasi-constante (99.6% = Pave)',
    }
    for feat, reason in excluded.items():
        print(f"  ✗ {feat:<20} → {reason}")
    
    return numeric_selected, categorical_selected


def generate_report(df):
    """Génère un rapport texte complet"""
    report_path = OUTPUT_DIR / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'ANALYSE EXPLORATOIRE - AMES HOUSING DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("JUSTIFICATION DU CHOIX DU DATASET\n")
        f.write("-" * 40 + "\n")
        f.write("""
Le jeu de données Ames Housing (Kaggle) a été choisi pour les raisons suivantes:

1. PERTINENCE: Ce dataset est parfaitement adapté à un problème de régression
   pour la prédiction de prix immobiliers, correspondant à l'objectif du projet.

2. RICHESSE: Avec 81 variables décrivant divers aspects des propriétés 
   (surface, qualité, localisation, etc.), il permet une modélisation complète.

3. TAILLE: 1460 observations offrent suffisamment de données pour 
   l'entraînement et la validation des modèles.

4. QUALITÉ: Dataset bien documenté, largement utilisé dans la communauté ML,
   avec un bon équilibre entre variables numériques et catégorielles.

5. SOURCE FIABLE: Provenant de Kaggle, une source reconnue pour les datasets
   de machine learning.
""")
        
        f.write("\n\nDESCRIPTION DU DATASET\n")
        f.write("-" * 40 + "\n")
        f.write(f"  • Nombre d'instances: {df.shape[0]}\n")
        f.write(f"  • Nombre d'attributs: {df.shape[1]}\n")
        f.write(f"  • Variable cible: SalePrice (numérique continue)\n")
        f.write(f"  • Variables numériques: {len(df.select_dtypes(include=[np.number]).columns)}\n")
        f.write(f"  • Variables catégorielles: {len(df.select_dtypes(include=['object']).columns)}\n")
        
        f.write("\n\nSTATISTIQUES DE LA VARIABLE CIBLE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  • Minimum: ${df['SalePrice'].min():,.0f}\n")
        f.write(f"  • Maximum: ${df['SalePrice'].max():,.0f}\n")
        f.write(f"  • Moyenne: ${df['SalePrice'].mean():,.0f}\n")
        f.write(f"  • Médiane: ${df['SalePrice'].median():,.0f}\n")
        
    print(f"\n✓ Rapport sauvegardé: {report_path}")


def main():
    """Exécute l'analyse complète"""
    print("\n" + "█" * 80)
    print("  ANALYSE EXPLORATOIRE DES DONNÉES - AMES HOUSING DATASET")
    print("█" * 80)
    
    # 1. Charger les données
    df = load_data()
    
    # 2. Description du dataset
    numeric_cols, categorical_cols = describe_dataset(df)
    
    # 3. Distribution de la variable cible
    analyze_target_distribution(df)
    
    # 4. Valeurs manquantes
    missing_df = analyze_missing_values(df)
    
    # 5. Corrélations
    correlations, selected_numeric = analyze_correlations(df, numeric_cols)
    
    # 6. Valeurs aberrantes
    outlier_counts = analyze_outliers(df)
    
    # 7. Variables catégorielles
    analyze_categorical(df, categorical_cols)
    
    # 8. Sélection des attributs
    numeric_selected, categorical_selected = feature_selection_summary(df, selected_numeric)
    
    # 9. Générer le rapport
    generate_report(df)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)
    print(f"\n📁 Fichiers générés dans: {OUTPUT_DIR}")
    print("   - target_distribution.png")
    print("   - missing_values.png")
    print("   - correlation_matrix.png")
    print("   - outliers_boxplots.png")
    print("   - scatter_outliers.png")
    print("   - categorical_analysis.png")
    print("   - analysis_report.txt")
    
    return df


if __name__ == "__main__":
    main()
