import {
    Analytics as AnalyticsIcon,
    Assessment as AssessmentIcon,
    BarChart as BarChartIcon,
    ShowChart as ChartIcon,
    CheckCircle as CheckCircleIcon,
    CompareArrows as CompareIcon,
    DataObject as DataIcon,
    Timeline as TimelineIcon,
    TrendingUp as TrendingUpIcon,
    EmojiEvents as TrophyIcon,
} from '@mui/icons-material';
import {
    Box,
    Card,
    CardContent,
    Chip,
    Grid,
    LinearProgress,
    Paper,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tabs,
    Typography,
} from '@mui/material';
import { useState } from 'react';

// Model comparison data
const modelResults = [
  {
    name: 'HistGradientBoosting',
    rmse: 0.1375,
    rmseStd: 0.0126,
    mae: 0.0951,
    maeStd: 0.0053,
    r2: 0.8811,
    r2Std: 0.0130,
    isBest: true,
  },
  {
    name: 'RandomForest',
    rmse: 0.1451,
    rmseStd: 0.0096,
    mae: 0.0988,
    maeStd: 0.0031,
    r2: 0.8674,
    r2Std: 0.0112,
    isBest: false,
  },
  {
    name: 'Ridge',
    rmse: 0.1445,
    rmseStd: 0.0243,
    mae: 0.0950,
    maeStd: 0.0027,
    r2: 0.8650,
    r2Std: 0.0484,
    isBest: false,
  },
];

// EDA statistics
const edaStats = {
  instances: 1460,
  features: 81,
  numericFeatures: 38,
  categoricalFeatures: 43,
  missingFeatures: 19,
  targetMin: 34900,
  targetMax: 755000,
  targetMean: 180921,
  targetMedian: 163000,
};

// Top correlations
const topCorrelations = [
  { feature: 'OverallQual', correlation: 0.791 },
  { feature: 'GrLivArea', correlation: 0.709 },
  { feature: 'GarageCars', correlation: 0.640 },
  { feature: 'GarageArea', correlation: 0.623 },
  { feature: 'TotalBsmtSF', correlation: 0.614 },
  { feature: 'FullBath', correlation: 0.561 },
  { feature: 'TotRmsAbvGrd', correlation: 0.534 },
  { feature: 'YearBuilt', correlation: 0.523 },
];

// Figures
const figures = [
  // EDA Figures
  {
    id: 'saleprice_distribution',
    title: 'Distribution de SalePrice',
    description: 'Distribution avant/après transformation log1p avec Q-Q plots',
    src: '/figures/saleprice_distribution.png',
    category: 'eda',
  },
  {
    id: 'missing_values',
    title: 'Valeurs Manquantes',
    description: 'Top features avec des valeurs manquantes',
    src: '/figures/missing_values.png',
    category: 'eda',
  },
  {
    id: 'correlation_heatmap',
    title: 'Heatmap des Corrélations',
    description: 'Matrice de corrélation des top features',
    src: '/figures/correlation_heatmap.png',
    category: 'eda',
  },
  {
    id: 'scatter_plots',
    title: 'Scatter Plots',
    description: 'Top 6 features numériques vs SalePrice',
    src: '/figures/scatter_plots.png',
    category: 'eda',
  },
  {
    id: 'boxplots',
    title: 'Boxplots',
    description: 'Distribution de SalePrice par catégories',
    src: '/figures/boxplots.png',
    category: 'eda',
  },
  // Model Figures
  {
    id: 'cv_comparison',
    title: 'Cross-Validation Comparison',
    description: 'RMSE, MAE, and R² scores for all 3 models with 5-fold CV',
    src: '/figures/cv_comparison.png',
    category: 'models',
  },
  {
    id: 'predicted_vs_actual',
    title: 'Predicted vs Actual',
    description: 'Scatter plot comparing predictions to actual values',
    src: '/figures/predicted_vs_actual.png',
    category: 'models',
  },
  {
    id: 'residuals',
    title: 'Residual Analysis',
    description: 'Residuals distribution and residuals vs predicted values',
    src: '/figures/residuals.png',
    category: 'models',
  },
  {
    id: 'learning_curve',
    title: 'Learning Curve',
    description: 'Training and validation performance vs training set size',
    src: '/figures/learning_curve.png',
    category: 'models',
  },
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

export default function AnalysisPage() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const bestModel = modelResults.find((m) => m.isBest);

  return (
    <Box className="animate-fade-in">
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box
          sx={{
            width: 56,
            height: 56,
            borderRadius: 3,
            background: 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <AnalyticsIcon sx={{ color: 'white', fontSize: 32 }} />
        </Box>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Analyse ML & Résultats
          </Typography>
          <Typography variant="body1" color="text.secondary">
            EDA, comparaison des modèles et visualisations
          </Typography>
        </Box>
      </Box>

      {/* Best Model Summary */}
      {bestModel && (
        <Card
          sx={{
            mb: 4,
            background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
            color: 'white',
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <TrophyIcon sx={{ fontSize: 40 }} />
                  <Box>
                    <Typography variant="overline" sx={{ opacity: 0.9 }}>
                      Meilleur Modèle
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {bestModel.name}
                    </Typography>
                  </Box>
                </Box>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Sélectionné via 5-Fold Cross-Validation avec le meilleur score RMSE
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        {bestModel.r2.toFixed(2)}
                      </Typography>
                      <Typography variant="caption" sx={{ opacity: 0.9 }}>
                        R² Score
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        {bestModel.rmse.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" sx={{ opacity: 0.9 }}>
                        RMSE (log)
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        {bestModel.mae.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" sx={{ opacity: 0.9 }}>
                        MAE (log)
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} variant="fullWidth">
          <Tab icon={<BarChartIcon />} label="EDA" />
          <Tab icon={<CompareIcon />} label="Modèles" />
          <Tab icon={<ChartIcon />} label="Visualisations" />
        </Tabs>
      </Paper>

      {/* Tab 1: EDA */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {/* Dataset Stats */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <DataIcon color="primary" />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Statistiques du Dataset
                  </Typography>
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'primary.50', textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" sx={{ fontWeight: 700 }}>
                        {edaStats.instances.toLocaleString()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Instances
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'secondary.50', textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary" sx={{ fontWeight: 700 }}>
                        {edaStats.features}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Features
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'success.50', textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main" sx={{ fontWeight: 700 }}>
                        {edaStats.numericFeatures}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Numériques
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'warning.50', textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main" sx={{ fontWeight: 700 }}>
                        {edaStats.categoricalFeatures}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Catégorielles
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Target Variable */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <TrendingUpIcon color="success" />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Variable Cible: SalePrice
                  </Typography>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Min</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      ${edaStats.targetMin.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Max</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      ${edaStats.targetMax.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Moyenne</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
                      ${edaStats.targetMean.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Médiane</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      ${edaStats.targetMedian.toLocaleString()}
                    </Typography>
                  </Box>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  La transformation log1p est appliquée pour normaliser la distribution
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Top Correlations */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <TimelineIcon color="primary" />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Top Corrélations avec SalePrice
                  </Typography>
                </Box>
                <Grid container spacing={2}>
                  {topCorrelations.map((item) => (
                    <Grid item xs={12} sm={6} md={3} key={item.feature}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {item.feature}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ flexGrow: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={item.correlation * 100}
                              sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: 'grey.200',
                                '& .MuiLinearProgress-bar': {
                                  borderRadius: 4,
                                  bgcolor: item.correlation > 0.7 ? 'success.main' : 'primary.main',
                                },
                              }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 45 }}>
                            {item.correlation.toFixed(3)}
                          </Typography>
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 2: Models Comparison */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          {/* Model Cards */}
          {modelResults.map((model) => (
            <Grid item xs={12} md={4} key={model.name}>
              <Card
                sx={{
                  height: '100%',
                  border: model.isBest ? '2px solid' : 'none',
                  borderColor: 'success.main',
                  position: 'relative',
                }}
              >
                {model.isBest && (
                  <Chip
                    icon={<CheckCircleIcon />}
                    label="Meilleur"
                    color="success"
                    size="small"
                    sx={{ position: 'absolute', top: 12, right: 12 }}
                  />
                )}
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                    {model.name}
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        RMSE (log)
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {model.rmse.toFixed(4)} ± {model.rmseStd.toFixed(4)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        MAE (log)
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {model.mae.toFixed(4)} ± {model.maeStd.toFixed(4)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">
                        R² Score
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ fontWeight: 600, color: model.isBest ? 'success.main' : 'inherit' }}
                      >
                        {model.r2.toFixed(4)} ± {model.r2Std.toFixed(4)}
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      R² Progress
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={model.r2 * 100}
                      sx={{
                        height: 10,
                        borderRadius: 5,
                        mt: 0.5,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 5,
                          bgcolor: model.isBest ? 'success.main' : 'primary.main',
                        },
                      }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}

          {/* Comparison Table */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <AssessmentIcon color="primary" />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Tableau Comparatif (5-Fold CV)
                  </Typography>
                </Box>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 600 }}>Modèle</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          RMSE
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          MAE
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          R²
                        </TableCell>
                        <TableCell align="center" sx={{ fontWeight: 600 }}>
                          Status
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {modelResults.map((model) => (
                        <TableRow
                          key={model.name}
                          sx={{ bgcolor: model.isBest ? 'success.50' : 'inherit' }}
                        >
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {model.isBest && <TrophyIcon color="success" fontSize="small" />}
                              <Typography sx={{ fontWeight: model.isBest ? 600 : 400 }}>
                                {model.name}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            {model.rmse.toFixed(4)} ± {model.rmseStd.toFixed(4)}
                          </TableCell>
                          <TableCell align="right">
                            {model.mae.toFixed(4)} ± {model.maeStd.toFixed(4)}
                          </TableCell>
                          <TableCell align="right">
                            {model.r2.toFixed(4)} ± {model.r2Std.toFixed(4)}
                          </TableCell>
                          <TableCell align="center">
                            {model.isBest ? (
                              <Chip label="Sélectionné" color="success" size="small" />
                            ) : (
                              <Chip label="Candidat" variant="outlined" size="small" />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 3: Visualizations */}
      <TabPanel value={tabValue} index={2}>
        {/* EDA Figures Section */}
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <DataIcon color="primary" />
          Analyse Exploratoire (EDA)
        </Typography>
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {figures.filter(f => f.category === 'eda').map((figure) => (
            <Grid item xs={12} md={6} key={figure.id}>
              <Card className="card-hover">
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                    {figure.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {figure.description}
                  </Typography>
                  <Box
                    component="img"
                    src={figure.src}
                    alt={figure.title}
                    sx={{
                      width: '100%',
                      height: 'auto',
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Model Figures Section */}
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <AssessmentIcon color="secondary" />
          Évaluation des Modèles
        </Typography>
        <Grid container spacing={3}>
          {figures.filter(f => f.category === 'models').map((figure) => (
            <Grid item xs={12} md={6} key={figure.id}>
              <Card className="card-hover">
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                    {figure.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {figure.description}
                  </Typography>
                  <Box
                    component="img"
                    src={figure.src}
                    alt={figure.title}
                    sx={{
                      width: '100%',
                      height: 'auto',
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>
    </Box>
  );
}
