import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SystemMetrics, ServiceStatus } from '@/types';

interface SystemState {
  metrics: SystemMetrics | null;
  services: Record<string, ServiceStatus>;
  kafka: { running: boolean; status: string } | null;
  isLoading: boolean;
  error: string | null;
}

const initialState: SystemState = {
  metrics: null,
  services: {},
  kafka: null,
  isLoading: false,
  error: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.metrics = action.payload;
    },
    setServices: (state, action: PayloadAction<Record<string, ServiceStatus>>) => {
      state.services = action.payload;
    },
    updateServiceStatus: (state, action: PayloadAction<{ key: string; status: ServiceStatus }>) => {
      state.services[action.payload.key] = action.payload.status;
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