import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { TradingStatus, Position } from '../../types';
import { apiService } from '../../services/api';

interface TradingState {
  status: TradingStatus | null;
  positions: Position[];
  isTrading: boolean;
  isLoading: boolean;
  error: string | null;
}

const initialState: TradingState = {
  status: null,
  positions: [],
  isTrading: false,
  isLoading: false,
  error: null,
};

// Async thunks
export const placePaperOrder = createAsyncThunk(
  'trading/placePaperOrder',
  async (orderData: any) => {
    const response = await apiService.placePaperOrder(orderData);
    return response;
  }
);

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    setStatus: (state, action: PayloadAction<TradingStatus>) => {
      state.status = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const { setStatus, setLoading, setError } = tradingSlice.actions;
export default tradingSlice.reducer;