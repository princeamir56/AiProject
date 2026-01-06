import {
    Analytics as AnalyticsIcon,
    BarChart as BarChartIcon,
    Dataset as DatasetIcon,
    Description as DescriptionIcon,
    GitHub as GitHubIcon,
    Home as HomeIcon,
    Numbers as NumbersIcon,
    TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import {
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Divider,
    Grid,
    Link,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography,
} from '@mui/material';

// Dataset statistics
const datasetStats = {
  name: 'Ames Housing Dataset',
  source: 'Kaggle',
  sourceUrl: 'https://www.kaggle.com/c/house-prices-advanced-regression-techniques',
  instances: 1460,
  attributes: 81,
  target: 'SalePrice',
  targetType: 'Numérique Continue',
};

// Top features by correlation
const topFeatures = [
  { name: 'OverallQual', correlation: 0.791, description: 'Qualité générale de la maison' },
  { name: 'GrLivArea', correlation: 0.709, description: 'Surface habitable en sq ft' },
  { name: 'GarageCars', correlation: 0.640, description: 'Capacité du garage en voitures' },
  { name: 'GarageArea', correlation: 0.623, description: 'Surface du garage en sq ft' },
  { name: 'TotalBsmtSF', correlation: 0.614, description: 'Surface totale du sous-sol' },
  { name: 'FullBath', correlation: 0.561, description: 'Nombre de salles de bain complètes' },
  { name: 'TotRmsAbvGrd', correlation: 0.534, description: 'Nombre total de pièces (hors SdB)' },
  { name: 'YearBuilt', correlation: 0.523, description: 'Année de construction' },
];

// Target variable statistics
const targetStats = {
  min: 34900,
  max: 755000,
  mean: 180921,
  median: 163000,
  std: 79443,
};

export default function DatasetPage() {
  return (
    <Box className="animate-fade-in">
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box
          sx={{
            width: 56,
            height: 56,
            borderRadius: 3,
            background: 'linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <DatasetIcon sx={{ color: 'white', fontSize: 32 }} />
        </Box>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            À Propos du Dataset
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Informations détaillées sur le jeu de données utilisé
          </Typography>
        </Box>
      </Box>

      {/* Main Info Card */}
      <Card 
        sx={{ 
          mb: 4, 
          background: 'linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)',
          color: 'white',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            right: -50,
            top: -50,
            width: 200,
            height: 200,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.1)',
          }}
        />
        <CardContent sx={{ p: 4, position: 'relative' }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
                {datasetStats.name}
              </Typography>
              <Typography variant="body1" sx={{ opacity: 0.9, mb: 3 }}>
                Ce dataset contient des informations détaillées sur les ventes immobilières 
                à Ames, Iowa (USA). Il est largement utilisé pour l'apprentissage du machine 
                learning appliqué à la régression et la prédiction de prix.
              </Typography>
              <Button
                variant="contained"
                startIcon={<GitHubIcon />}
                component={Link}
                href={datasetStats.sourceUrl}
                target="_blank"
                sx={{ 
                  bgcolor: 'white', 
                  color: 'primary.main',
                  '&:hover': { bgcolor: 'grey.100' }
                }}
              >
                Voir sur Kaggle
              </Button>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h2" sx={{ fontWeight: 700 }}>
                  {datasetStats.instances.toLocaleString()}
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.8 }}>
                  Propriétés analysées
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={6} sm={3}>
          <Card className="card-hover">
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <NumbersIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {datasetStats.instances}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Instances (lignes)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card className="card-hover">
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <BarChartIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {datasetStats.attributes}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Attributs (colonnes)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card className="card-hover">
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <TrendingUpIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                38
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Var. Numériques
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card className="card-hover">
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <DescriptionIcon sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                43
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Var. Catégorielles
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Target Variable */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <AnalyticsIcon color="primary" />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Variable Cible: SalePrice
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Le prix de vente des propriétés en dollars américains. 
                C'est la variable que notre modèle cherche à prédire.
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary">Minimum</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      ${targetStats.min.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Maximum</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      ${targetStats.max.toLocaleString()}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary">Moyenne</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
                      ${targetStats.mean.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Médiane</Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                      ${targetStats.median.toLocaleString()}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Justification */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <HomeIcon color="primary" />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Justification du Choix
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                  <Chip label="1" size="small" color="primary" />
                  <Box>
                    <Typography variant="subtitle2">Pertinence</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Parfaitement adapté à un problème de régression pour la prédiction de prix.
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                  <Chip label="2" size="small" color="primary" />
                  <Box>
                    <Typography variant="subtitle2">Richesse</Typography>
                    <Typography variant="body2" color="text.secondary">
                      81 variables décrivant tous les aspects des propriétés.
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                  <Chip label="3" size="small" color="primary" />
                  <Box>
                    <Typography variant="subtitle2">Qualité</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Dataset bien documenté, largement utilisé dans la communauté ML.
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                  <Chip label="4" size="small" color="primary" />
                  <Box>
                    <Typography variant="subtitle2">Source Fiable</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Provenant de Kaggle, référence pour les datasets ML.
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Top Features Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Top Features par Corrélation avec SalePrice
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.50' }}>
                      <TableCell sx={{ fontWeight: 600 }}>Rang</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Variable</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Corrélation</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {topFeatures.map((feature, index) => (
                      <TableRow key={feature.name} hover>
                        <TableCell>
                          <Chip 
                            label={index + 1} 
                            size="small" 
                            color={index < 3 ? 'primary' : 'default'}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {feature.name}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box
                              sx={{
                                width: 80,
                                height: 8,
                                bgcolor: 'grey.200',
                                borderRadius: 4,
                                overflow: 'hidden',
                              }}
                            >
                              <Box
                                sx={{
                                  width: `${feature.correlation * 100}%`,
                                  height: '100%',
                                  bgcolor: feature.correlation > 0.6 ? 'success.main' : 'primary.main',
                                  borderRadius: 4,
                                }}
                              />
                            </Box>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {feature.correlation.toFixed(3)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {feature.description}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Cleaning Info */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Nettoyage et Préparation des Données
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                    <Typography variant="subtitle2" color="primary" sx={{ mb: 1 }}>
                      Valeurs Manquantes
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • Imputation par la médiane (variables numériques)
                      <br />• Imputation par "missing" (variables catégorielles)
                      <br />• Suppression des colonnes avec &gt;80% manquants
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                    <Typography variant="subtitle2" color="primary" sx={{ mb: 1 }}>
                      Valeurs Aberrantes
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • Détection via méthode IQR
                      <br />• Suppression de 2 outliers (GrLivArea &gt; 4000)
                      <br />• Validation visuelle par scatter plots
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                    <Typography variant="subtitle2" color="primary" sx={{ mb: 1 }}>
                      Transformations
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      • StandardScaler pour normalisation
                      <br />• OneHotEncoder pour variables catégorielles
                      <br />• Pipeline sklearn pour automatisation
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
