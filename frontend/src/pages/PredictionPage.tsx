import {
    Calculate as CalculateIcon,
    Home as HomeIcon,
    Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Collapse,
    Divider,
    Grid,
    InputAdornment,
    MenuItem,
    Paper,
    Slider,
    TextField,
    Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { apiClient, DropdownOptions, HouseFeatures, PredictionResponse } from '../api';
import { LoadingSpinner } from '../components';

const defaultFeatures: HouseFeatures = {
  OverallQual: 5,
  OverallCond: 5,
  YearBuilt: 2000,
  TotalBsmtSF: 1000,
  GrLivArea: 1500,
  FullBath: 2,
  HalfBath: 0,
  BedroomAbvGr: 3,
  TotRmsAbvGrd: 6,
  GarageCars: 2,
  GarageArea: 400,
  MSZoning: 'RL',
  Neighborhood: 'NAmes',
  BldgType: '1Fam',
  HouseStyle: '1Story',
  CentralAir: 'Y',
  KitchenQual: 'TA',
};

export default function PredictionPage() {
  const [features, setFeatures] = useState<HouseFeatures>(defaultFeatures);
  const [options, setOptions] = useState<DropdownOptions | null>(null);
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const data = await apiClient.getOptions();
        setOptions(data);
      } catch (err) {
        console.error('Failed to fetch options:', err);
      } finally {
        setOptionsLoading(false);
      }
    };
    fetchOptions();
  }, []);

  const handleInputChange = (field: keyof HouseFeatures, value: string | number) => {
    setFeatures((prev) => ({ ...prev, [field]: value }));
    setPrediction(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiClient.createPrediction({
        features,
        session_id: `session_${Date.now()}`,
      });
      setPrediction(result);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to get prediction: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFeatures(defaultFeatures);
    setPrediction(null);
    setError(null);
  };

  if (optionsLoading) {
    return <LoadingSpinner message="Loading form options..." />;
  }

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          Predict House Price
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Enter the property details below to get an instant price prediction
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Form Section */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent sx={{ p: 4 }}>
              {/* Quality & Condition */}
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                <HomeIcon color="primary" /> Quality & Condition
              </Typography>
              
              <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Overall Quality (1-10): {features.OverallQual}
                  </Typography>
                  <Slider
                    value={features.OverallQual}
                    onChange={(_, value) => handleInputChange('OverallQual', value as number)}
                    min={1}
                    max={10}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Overall Condition (1-10): {features.OverallCond}
                  </Typography>
                  <Slider
                    value={features.OverallCond}
                    onChange={(_, value) => handleInputChange('OverallCond', value as number)}
                    min={1}
                    max={10}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Size & Area */}
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Size & Area
              </Typography>
              
              <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Year Built"
                    type="number"
                    value={features.YearBuilt}
                    onChange={(e) => handleInputChange('YearBuilt', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 1800, max: 2025 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Living Area"
                    type="number"
                    value={features.GrLivArea}
                    onChange={(e) => handleInputChange('GrLivArea', parseInt(e.target.value) || 0)}
                    InputProps={{
                      endAdornment: <InputAdornment position="end">sq ft</InputAdornment>,
                    }}
                    inputProps={{ min: 0 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Basement Area"
                    type="number"
                    value={features.TotalBsmtSF}
                    onChange={(e) => handleInputChange('TotalBsmtSF', parseInt(e.target.value) || 0)}
                    InputProps={{
                      endAdornment: <InputAdornment position="end">sq ft</InputAdornment>,
                    }}
                    inputProps={{ min: 0 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Garage Area"
                    type="number"
                    value={features.GarageArea}
                    onChange={(e) => handleInputChange('GarageArea', parseInt(e.target.value) || 0)}
                    InputProps={{
                      endAdornment: <InputAdornment position="end">sq ft</InputAdornment>,
                    }}
                    inputProps={{ min: 0 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Garage Cars"
                    type="number"
                    value={features.GarageCars}
                    onChange={(e) => handleInputChange('GarageCars', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 10 }}
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Rooms */}
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Rooms
              </Typography>
              
              <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={6} sm={3}>
                  <TextField
                    fullWidth
                    label="Bedrooms"
                    type="number"
                    value={features.BedroomAbvGr}
                    onChange={(e) => handleInputChange('BedroomAbvGr', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 20 }}
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <TextField
                    fullWidth
                    label="Full Baths"
                    type="number"
                    value={features.FullBath}
                    onChange={(e) => handleInputChange('FullBath', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 10 }}
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <TextField
                    fullWidth
                    label="Half Baths"
                    type="number"
                    value={features.HalfBath}
                    onChange={(e) => handleInputChange('HalfBath', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 10 }}
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <TextField
                    fullWidth
                    label="Total Rooms"
                    type="number"
                    value={features.TotRmsAbvGrd}
                    onChange={(e) => handleInputChange('TotRmsAbvGrd', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 30 }}
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Location & Type */}
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Location & Type
              </Typography>
              
              <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="Zoning"
                    value={features.MSZoning}
                    onChange={(e) => handleInputChange('MSZoning', e.target.value)}
                  >
                    {options?.MSZoning.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="Neighborhood"
                    value={features.Neighborhood}
                    onChange={(e) => handleInputChange('Neighborhood', e.target.value)}
                  >
                    {options?.Neighborhood.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="Building Type"
                    value={features.BldgType}
                    onChange={(e) => handleInputChange('BldgType', e.target.value)}
                  >
                    {options?.BldgType.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="House Style"
                    value={features.HouseStyle}
                    onChange={(e) => handleInputChange('HouseStyle', e.target.value)}
                  >
                    {options?.HouseStyle.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="Kitchen Quality"
                    value={features.KitchenQual}
                    onChange={(e) => handleInputChange('KitchenQual', e.target.value)}
                  >
                    {options?.KitchenQual.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    select
                    fullWidth
                    label="Central Air"
                    value={features.CentralAir}
                    onChange={(e) => handleInputChange('CentralAir', e.target.value)}
                  >
                    <MenuItem value="Y">Yes</MenuItem>
                    <MenuItem value="N">No</MenuItem>
                  </TextField>
                </Grid>
              </Grid>

              {/* Actions */}
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={handleReset}
                >
                  Reset
                </Button>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<CalculateIcon />}
                  onClick={handleSubmit}
                  disabled={loading}
                >
                  {loading ? 'Predicting...' : 'Predict Price'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Result Section */}
        <Grid item xs={12} lg={4}>
          <Box sx={{ position: 'sticky', top: 24 }}>
            <Card
              sx={{
                background: prediction
                  ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                  : 'linear-gradient(135deg, #64748b 0%, #475569 100%)',
                color: 'white',
                mb: 3,
              }}
            >
              <CardContent sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="overline" sx={{ opacity: 0.8 }}>
                  Predicted Price
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 700, my: 2 }}>
                  {prediction ? prediction.formatted_price : '$---,---'}
                </Typography>
                {prediction && (
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    Model: v{prediction.model_version}
                  </Typography>
                )}
              </CardContent>
            </Card>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <Collapse in={!!prediction}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Prediction Details
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">ID</Typography>
                    <Typography variant="body2">#{prediction?.id}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Living Area</Typography>
                    <Typography variant="body2">{features.GrLivArea} sq ft</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Bedrooms</Typography>
                    <Typography variant="body2">{features.BedroomAbvGr}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Bathrooms</Typography>
                    <Typography variant="body2">{features.FullBath + features.HalfBath * 0.5}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Quality</Typography>
                    <Typography variant="body2">{features.OverallQual}/10</Typography>
                  </Box>
                </Box>
              </Paper>
            </Collapse>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
