import {
    Delete as DeleteIcon,
    History as HistoryIcon,
    Visibility as ViewIcon,
} from '@mui/icons-material';
import {
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    IconButton,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TablePagination,
    TableRow,
    Tooltip,
    Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { apiClient, PredictionHistory } from '../api';
import { ErrorMessage, LoadingSpinner } from '../components';

export default function HistoryPage() {
  const [predictions, setPredictions] = useState<PredictionHistory[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionHistory | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getPredictions(page + 1, rowsPerPage);
      setPredictions(data.predictions);
      setTotal(data.total);
    } catch (err) {
      setError('Failed to fetch predictions history');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [page, rowsPerPage]);

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Are you sure you want to delete this prediction?')) return;
    
    try {
      await apiClient.deletePrediction(id);
      fetchPredictions();
    } catch (err) {
      console.error('Failed to delete prediction:', err);
    }
  };

  const handleViewDetails = (prediction: PredictionHistory) => {
    setSelectedPrediction(prediction);
    setDetailsOpen(true);
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(price);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getFeatureValue = (features: Record<string, unknown>, key: string): string => {
    const value = features[key];
    if (value === undefined || value === null) return 'N/A';
    if (typeof value === 'number') return value.toLocaleString();
    return String(value);
  };

  if (loading && predictions.length === 0) {
    return <LoadingSpinner message="Loading predictions history..." />;
  }

  return (
    <Box>
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <HistoryIcon sx={{ fontSize: 32, color: 'primary.main' }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Prediction History
          </Typography>
          <Typography variant="body1" color="text.secondary">
            View and manage all your past predictions
          </Typography>
        </Box>
      </Box>

      {error && <ErrorMessage message={error} onRetry={fetchPredictions} />}

      <Card>
        <CardContent sx={{ p: 0 }}>
          {predictions.length === 0 ? (
            <Box sx={{ p: 6, textAlign: 'center' }}>
              <HistoryIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                No predictions yet
              </Typography>
              <Typography variant="body2" color="text.disabled">
                Start by making your first prediction
              </Typography>
            </Box>
          ) : (
            <>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.50' }}>
                      <TableCell sx={{ fontWeight: 600 }}>ID</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Predicted Price</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Living Area</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Bedrooms</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Model</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
                      <TableCell sx={{ fontWeight: 600 }} align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((prediction) => (
                      <TableRow
                        key={prediction.id}
                        hover
                        sx={{ '&:last-child td': { border: 0 } }}
                      >
                        <TableCell>
                          <Chip
                            label={`#${prediction.id}`}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body1" sx={{ fontWeight: 600, color: 'success.main' }}>
                            {formatPrice(prediction.predicted_price)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {getFeatureValue(prediction.input_features, 'GrLivArea')} sq ft
                        </TableCell>
                        <TableCell>
                          {getFeatureValue(prediction.input_features, 'BedroomAbvGr')}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={prediction.model_version || 'N/A'}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {formatDate(prediction.created_at)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleViewDetails(prediction)}
                            >
                              <ViewIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDelete(prediction.id)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <TablePagination
                component="div"
                count={total}
                page={page}
                onPageChange={handleChangePage}
                rowsPerPage={rowsPerPage}
                onRowsPerPageChange={handleChangeRowsPerPage}
                rowsPerPageOptions={[5, 10, 25, 50]}
              />
            </>
          )}
        </CardContent>
      </Card>

      {/* Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Prediction Details #{selectedPrediction?.id}
        </DialogTitle>
        <DialogContent dividers>
          {selectedPrediction && (
            <Box>
              <Paper
                sx={{
                  p: 3,
                  mb: 3,
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: 'white',
                  textAlign: 'center',
                }}
              >
                <Typography variant="overline">Predicted Price</Typography>
                <Typography variant="h4" sx={{ fontWeight: 700 }}>
                  {formatPrice(selectedPrediction.predicted_price)}
                </Typography>
              </Paper>
              
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2 }}>
                Input Features
              </Typography>
              
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: 2,
                }}
              >
                {Object.entries(selectedPrediction.input_features).map(([key, value]) => (
                  <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      {key}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {String(value)}
                    </Typography>
                  </Box>
                ))}
              </Box>
              
              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                <Typography variant="body2" color="text.secondary">
                  Created: {formatDate(selectedPrediction.created_at)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Model Version: {selectedPrediction.model_version || 'N/A'}
                </Typography>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
