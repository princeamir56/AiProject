import { Box } from '@mui/material';
import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import AnalysisPage from './pages/AnalysisPage';
import DatasetPage from './pages/DatasetPage';
import HistoryPage from './pages/HistoryPage';
import HomePage from './pages/HomePage';
import ModelsPage from './pages/ModelsPage';
import PredictionPage from './pages/PredictionPage';

function App() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<PredictionPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
        </Routes>
      </Layout>
    </Box>
  );
}

export default App;
