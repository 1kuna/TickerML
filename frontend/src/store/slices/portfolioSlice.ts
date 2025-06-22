import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { PortfolioSnapshot, Position, PerformanceMetrics } from '@/types';

interface PortfolioState {
  snapshot: PortfolioSnapshot | null;
  positions: Position[];
  performance: PerformanceMetrics | null;
  history: any[];
  isLoading: boolean;
  error: string | null;
}

const initialState: PortfolioState = {
  snapshot: null,
  positions: [],
  performance: null,
  history: [],
  isLoading: false,
  error: null,
};

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