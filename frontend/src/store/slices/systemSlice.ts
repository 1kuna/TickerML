import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { SystemMetrics, ServiceStatus } from '../../types';
import { apiService } from '../../services/api';

interface SystemState {
  metrics: SystemMetrics | null;
  services: ServiceStatus[];
  kafka: { running: boolean; status: string } | null;
  alerts: any[];
  isLoading: boolean;
  error: string | null;
}

const initialState: SystemState = {
  metrics: null,
  services: [],
  kafka: null,
  alerts: [],
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchSystemStatus = createAsyncThunk(
  'system/fetchStatus',
  async () => {
    const response = await apiService.getSystemStatus();
    return response;
  }
);

export const fetchSystemMetrics = createAsyncThunk(
  'system/fetchMetrics',
  async () => {
    const response = await apiService.getSystemMetrics();
    return response;
  }
);

export const startService = createAsyncThunk(
  'system/startService',
  async (serviceName: string) => {
    await apiService.startService(serviceName);
    return serviceName;
  }
);

export const stopService = createAsyncThunk(
  'system/stopService',
  async (serviceName: string) => {
    await apiService.stopService(serviceName);
    return serviceName;
  }
);

export const restartService = createAsyncThunk(
  'system/restartService',
  async (serviceName: string) => {
    await apiService.restartService(serviceName);
    return serviceName;
  }
);

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.metrics = action.payload;
    },
    setServices: (state, action: PayloadAction<ServiceStatus[]>) => {
      state.services = action.payload;
    },
    updateServiceStatus: (state, action: PayloadAction<{ name: string; status: ServiceStatus }>) => {
      const index = state.services.findIndex(s => s.name === action.payload.name);
      if (index !== -1) {
        state.services[index] = action.payload.status;
      } else {
        state.services.push(action.payload.status);
      }
    },
    setKafka: (state, action: PayloadAction<{ running: boolean; status: string }>) => {
      state.kafka = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  setMetrics,
  setServices,
  updateServiceStatus,
  setKafka,
  setLoading,
  setError,
} = systemSlice.actions;

export default systemSlice.reducer;