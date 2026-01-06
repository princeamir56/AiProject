import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface HouseFeatures {
  OverallQual: number;
  OverallCond: number;
  YearBuilt: number;
  TotalBsmtSF: number;
  GrLivArea: number;
  FullBath: number;
  HalfBath: number;
  BedroomAbvGr: number;
  TotRmsAbvGrd: number;
  GarageCars: number;
  GarageArea: number;
  MSZoning: string;
  Neighborhood: string;
  BldgType: string;
  HouseStyle: string;
  CentralAir: string;
  KitchenQual: string;
}

export interface PredictionRequest {
  features: HouseFeatures;
  session_id?: string;
}

export interface PredictionResponse {
  id: number;
  predicted_price: number;
  formatted_price: string;
  model_version: string | null;
  input_features: Record<string, unknown>;
  created_at: string;
}

export interface PredictionHistory {
  id: number;
  predicted_price: number;
  input_features: Record<string, unknown>;
  model_version: string | null;
  created_at: string;
}

export interface PredictionListResponse {
  predictions: PredictionHistory[];
  total: number;
  page: number;
  per_page: number;
}

export interface MLModelInfo {
  id: number;
  name: string;
  version: string;
  metrics: Record<string, number> | null;
  is_active: boolean;
  created_at: string;
}

export interface MLModelListResponse {
  models: MLModelInfo[];
  active_model: string | null;
}

export interface TrainResponse {
  success: boolean;
  message: string;
  model_info: MLModelInfo | null;
  metrics: Record<string, number> | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  database: string;
  model_loaded: boolean;
}

export interface StatsResponse {
  total_predictions: number;
  avg_predicted_price: number | null;
  min_predicted_price: number | null;
  max_predicted_price: number | null;
  predictions_today: number;
  active_model_version: string | null;
}

export interface DropdownOption {
  value: string;
  label: string;
}

export interface DropdownOptions {
  MSZoning: DropdownOption[];
  Neighborhood: DropdownOption[];
  BldgType: DropdownOption[];
  HouseStyle: DropdownOption[];
  KitchenQual: DropdownOption[];
}

// API functions
export const apiClient = {
  // Health & Stats
  getHealth: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  getStats: async (): Promise<StatsResponse> => {
    const response = await api.get<StatsResponse>('/stats');
    return response.data;
  },

  getOptions: async (): Promise<DropdownOptions> => {
    const response = await api.get<DropdownOptions>('/options');
    return response.data;
  },

  // Predictions
  createPrediction: async (request: PredictionRequest): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>('/predictions', request);
    return response.data;
  },

  getPredictions: async (page = 1, perPage = 10): Promise<PredictionListResponse> => {
    const response = await api.get<PredictionListResponse>('/predictions', {
      params: { page, per_page: perPage },
    });
    return response.data;
  },

  getPrediction: async (id: number): Promise<PredictionResponse> => {
    const response = await api.get<PredictionResponse>(`/predictions/${id}`);
    return response.data;
  },

  deletePrediction: async (id: number): Promise<void> => {
    await api.delete(`/predictions/${id}`);
  },

  // Models
  getModels: async (): Promise<MLModelListResponse> => {
    const response = await api.get<MLModelListResponse>('/models');
    return response.data;
  },

  trainModel: async (modelName?: string, description?: string): Promise<TrainResponse> => {
    const response = await api.post<TrainResponse>('/models/train', {
      model_name: modelName,
      description,
    });
    return response.data;
  },

  activateModel: async (version: string): Promise<{ message: string }> => {
    const response = await api.post<{ message: string }>(`/models/${version}/activate`);
    return response.data;
  },

  getModelMetrics: async (): Promise<{ version: string | null; metrics: Record<string, number>; is_loaded: boolean }> => {
    const response = await api.get('/models/metrics');
    return response.data;
  },

  getFeatureImportance: async (): Promise<{ feature_importance: Record<string, number> }> => {
    const response = await api.get('/models/feature-importance');
    return response.data;
  },
};

export default apiClient;
