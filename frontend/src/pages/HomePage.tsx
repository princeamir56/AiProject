import {
    ArrowForward as ArrowForwardIcon,
    Calculate as CalculateIcon,
    CheckCircle as CheckIcon,
    Speed as SpeedIcon,
    Storage as StorageIcon,
    TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import {
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Grid,
    Stack,
    Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient, HealthResponse, StatsResponse } from '../api';
import { ErrorMessage, LoadingSpinner, StatCard } from '../components';

export default function HomePage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsData, healthData] = await Promise.all([
        apiClient.getStats(),
        apiClient.getHealth(),
      ]);
      setStats(statsData);
      setHealth(healthData);
    } catch (err) {
      setError('Failed to connect to the backend. Please ensure the API is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Connecting to backend..." />;
  }

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(price);
  };

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          mb: 5,
          p: 4,
          borderRadius: 4,
          background: 'linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)',
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
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
        <Box
          sx={{
            position: 'absolute',
            right: 50,
            bottom: -30,
            width: 120,
            height: 120,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.05)',
          }}
        />
        
        <Typography variant="h3" sx={{ fontWeight: 700, mb: 2 }}>
          House Price Predictor
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 400, mb: 3, opacity: 0.9, maxWidth: 600 }}>
          Predict house prices instantly using our advanced machine learning model trained on real estate data.
        </Typography>
        
        <Stack direction="row" spacing={2}>
          <Button
            component={Link}
            to="/predict"
            variant="contained"
            size="large"
            endIcon={<ArrowForwardIcon />}
            sx={{
              bgcolor: 'white',
              color: 'primary.main',
              '&:hover': {
                bgcolor: 'grey.100',
              },
            }}
          >
            Start Predicting
          </Button>
          <Button
            component={Link}
            to="/models"
            variant="outlined"
            size="large"
            sx={{
              borderColor: 'rgba(255,255,255,0.5)',
              color: 'white',
              '&:hover': {
                borderColor: 'white',
                bgcolor: 'rgba(255,255,255,0.1)',
              },
            }}
          >
            View Models
          </Button>
        </Stack>
      </Box>

      {error && <ErrorMessage message={error} onRetry={fetchData} />}

      {/* Status Chips */}
      <Stack direction="row" spacing={2} sx={{ mb: 4 }}>
        <Chip
          icon={<CheckIcon />}
          label={`API: ${health?.status || 'Unknown'}`}
          color={health?.status === 'healthy' ? 'success' : 'error'}
          variant="outlined"
        />
        <Chip
          icon={<StorageIcon />}
          label={`Database: ${health?.database || 'Unknown'}`}
          color={health?.database === 'connected' ? 'success' : 'error'}
          variant="outlined"
        />
        <Chip
          icon={<SpeedIcon />}
          label={`Model: ${health?.model_loaded ? 'Loaded' : 'Not Loaded'}`}
          color={health?.model_loaded ? 'success' : 'warning'}
          variant="outlined"
        />
      </Stack>

      {/* Stats Grid */}
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
        Dashboard Overview
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 5 }}>
        <Grid item xs={12} sm={6} lg={3}>
          <StatCard
            title="Total Predictions"
            value={stats?.total_predictions || 0}
            subtitle="All time"
            icon={<CalculateIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatCard
            title="Predictions Today"
            value={stats?.predictions_today || 0}
            subtitle="Last 24 hours"
            icon={<TrendingUpIcon />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatCard
            title="Average Price"
            value={formatPrice(stats?.avg_predicted_price || null)}
            subtitle="Predicted"
            icon={<StorageIcon />}
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatCard
            title="Active Model"
            value={stats?.active_model_version || 'None'}
            subtitle="Version"
            icon={<SpeedIcon />}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Features Section */}
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
        Features
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card className="card-hover" sx={{ height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box
                sx={{
                  width: 56,
                  height: 56,
                  borderRadius: 3,
                  background: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}
              >
                <CalculateIcon sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                Instant Predictions
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Get house price predictions in seconds using our ML model trained on thousands of real estate transactions.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card className="card-hover" sx={{ height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box
                sx={{
                  width: 56,
                  height: 56,
                  borderRadius: 3,
                  background: 'linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}
              >
                <StorageIcon sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                Prediction History
              </Typography>
              <Typography variant="body2" color="text.secondary">
                All your predictions are saved and can be accessed anytime. Track and compare different property valuations.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card className="card-hover" sx={{ height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box
                sx={{
                  width: 56,
                  height: 56,
                  borderRadius: 3,
                  background: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}
              >
                <SpeedIcon sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                Model Management
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Train new models, view performance metrics, and switch between different model versions easily.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
