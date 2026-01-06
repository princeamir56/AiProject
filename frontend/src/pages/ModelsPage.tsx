import {
    CheckCircle as ActiveIcon,
    ModelTraining as ModelIcon,
    Speed as SpeedIcon,
    PlayArrow as TrainIcon,
} from '@mui/icons-material';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Grid,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { apiClient, MLModelInfo } from '../api';
import { ErrorMessage, LoadingSpinner, StatCard } from '../components';

export default function ModelsPage() {
  const [models, setModels] = useState<MLModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [featureImportance, setFeatureImportance] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainDialogOpen, setTrainDialogOpen] = useState(false);
  const [modelName, setModelName] = useState('');
  const [modelDescription, setModelDescription] = useState('');

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsData, importanceData] = await Promise.all([
        apiClient.getModels(),
        apiClient.getFeatureImportance().catch(() => ({ feature_importance: {} })),
      ]);
      setModels(modelsData.models);
      setActiveModel(modelsData.active_model);
      setFeatureImportance(importanceData.feature_importance);
    } catch (err) {
      setError('Failed to fetch models data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleTrain = async () => {
    setTrainDialogOpen(false);
    setTraining(true);
    setError(null);
    
    try {
      await apiClient.trainModel(
        modelName || undefined,
        modelDescription || undefined
      );
      setModelName('');
      setModelDescription('');
      fetchData();
    } catch (err) {
      setError('Failed to train model');
      console.error(err);
    } finally {
      setTraining(false);
    }
  };

  const handleActivate = async (version: string) => {
    try {
      await apiClient.activateModel(version);
      fetchData();
    } catch (err) {
      console.error('Failed to activate model:', err);
    }
  };



  const formatMetricValue = (value: number | undefined) => {
    if (value === undefined) return 'N/A';
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  // Prepare chart data
  const chartData = Object.entries(featureImportance)
    .slice(0, 10)
    .map(([name, value]) => ({
      name: name.length > 15 ? name.substring(0, 15) + '...' : name,
      importance: (value * 100).toFixed(2),
    }));

  if (loading) {
    return <LoadingSpinner message="Loading models..." />;
  }

  const currentModel = models.find((m) => m.is_active);

  return (
    <Box>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <ModelIcon sx={{ fontSize: 32, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>
              ML Models
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Manage and train machine learning models
            </Typography>
          </Box>
        </Box>
        <Button
          variant="contained"
          startIcon={<TrainIcon />}
          onClick={() => setTrainDialogOpen(true)}
          disabled={training}
        >
          Train New Model
        </Button>
      </Box>

      {training && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography>Training in progress...</Typography>
            <LinearProgress sx={{ flex: 1 }} />
          </Box>
        </Alert>
      )}

      {error && <ErrorMessage message={error} onRetry={fetchData} />}

      {/* Current Model Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Active Model"
            value={activeModel || 'None'}
            icon={<ActiveIcon />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="RMSE"
            value={currentModel?.metrics?.rmse ? `$${formatMetricValue(currentModel.metrics.rmse)}` : 'N/A'}
            subtitle="Root Mean Square Error"
            icon={<SpeedIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="R² Score"
            value={formatMetricValue(currentModel?.metrics?.r2)}
            subtitle="Coefficient of Determination"
            icon={<SpeedIcon />}
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Models"
            value={models.length}
            icon={<ModelIcon />}
            color="info"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Feature Importance Chart */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Top 10 Feature Importance
              </Typography>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" unit="%" />
                    <YAxis dataKey="name" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 6 }}>
                  <Typography color="text.secondary">
                    No feature importance data available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Models List */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Model Versions
              </Typography>
              
              {models.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 6 }}>
                  <ModelIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
                  <Typography color="text.secondary">
                    No models trained yet
                  </Typography>
                  <Typography variant="body2" color="text.disabled">
                    Train your first model to get started
                  </Typography>
                </Box>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Version</TableCell>
                        <TableCell>RMSE</TableCell>
                        <TableCell>R²</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {models.map((model) => (
                        <TableRow key={model.version}>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {model.version}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {model.metrics?.rmse ? `$${formatMetricValue(model.metrics.rmse)}` : 'N/A'}
                          </TableCell>
                          <TableCell>
                            {formatMetricValue(model.metrics?.r2)}
                          </TableCell>
                          <TableCell>
                            {model.is_active ? (
                              <Chip
                                label="Active"
                                size="small"
                                color="success"
                                icon={<ActiveIcon />}
                              />
                            ) : (
                              <Chip label="Inactive" size="small" variant="outlined" />
                            )}
                          </TableCell>
                          <TableCell>
                            {!model.is_active && (
                              <Button
                                size="small"
                                onClick={() => handleActivate(model.version)}
                              >
                                Activate
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Train Dialog */}
      <Dialog
        open={trainDialogOpen}
        onClose={() => setTrainDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Train New Model</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            This will train a new model using the current dataset. The process may take a few minutes.
          </Typography>
          <TextField
            fullWidth
            label="Model Name (optional)"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="Leave empty for auto-generated name"
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Description (optional)"
            value={modelDescription}
            onChange={(e) => setModelDescription(e.target.value)}
            multiline
            rows={2}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleTrain}>
            Start Training
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
