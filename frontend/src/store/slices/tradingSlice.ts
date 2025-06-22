import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { TradingStatus } from '@/types';

interface TradingState {
  status: TradingStatus | null;
  isLoading: boolean;
  error: string | null;
}

const initialState: TradingState = {
  status: null,
  isLoading: false,
  error: null,
};

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