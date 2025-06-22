import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { PortfolioSnapshot, Position, PerformanceMetrics } from '../../types';
import { apiService } from '../../services/api';

interface PortfolioState {
  snapshot: PortfolioSnapshot | null;
  positions: Position[];
  performance: PerformanceMetrics | null;
  history: any[];
  summary: PortfolioSnapshot | null;
  tradeHistory: any[];
  isLoading: boolean;
  error: string | null;
}

const initialState: PortfolioState = {
  snapshot: null,
  positions: [],
  performance: null,
  history: [],
  summary: null,
  tradeHistory: [],
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchPortfolioSummary = createAsyncThunk(
  'portfolio/fetchSummary',
  async () => {
    const response = await apiService.getPortfolioSnapshot();
    return response;
  }
);

export const fetchPositions = createAsyncThunk(
  'portfolio/fetchPositions',
  async () => {
    const response = await apiService.getPositions();
    return response;
  }
);

export const fetchTradeHistory = createAsyncThunk(
  'portfolio/fetchTradeHistory',
  async () => {
    const response = await apiService.getTradeHistory();
    return response;
  }
);

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    setSnapshot: (state, action: PayloadAction<PortfolioSnapshot>) => {
      state.snapshot = action.payload;
    },
    setPositions: (state, action: PayloadAction<Position[]>) => {
      state.positions = action.payload;
    },
    setPerformance: (state, action: PayloadAction<PerformanceMetrics>) => {
      state.performance = action.payload;
    },
    setHistory: (state, action: PayloadAction<any[]>) => {
      state.history = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPortfolioSummary.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchPortfolioSummary.fulfilled, (state, action) => {
        state.isLoading = false;
        state.summary = action.payload;
      })
      .addCase(fetchPortfolioSummary.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to fetch portfolio summary';
      })
      .addCase(fetchPositions.fulfilled, (state, action) => {
        state.positions = action.payload;
      })
      .addCase(fetchTradeHistory.fulfilled, (state, action) => {
        state.tradeHistory = action.payload;
      });
  },
});

export const {
  setSnapshot,
  setPositions,
  setPerformance,
  setHistory,
  setLoading,
  setError,
} = portfolioSlice.actions;

export default portfolioSlice.reducer;