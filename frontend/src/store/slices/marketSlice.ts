import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { MarketData, OrderBook, Trade } from '../../types';

interface MarketState {
  prices: MarketData[];
  orderBooks: Record<string, OrderBook>;
  recentTrades: Record<string, Trade[]>;
  currentSymbol: string;
  currentExchange: string;
  orderBook: OrderBook | null;
  isConnected: boolean;
  isLoading: boolean;
  lastUpdate: string | null;
  error: string | null;
}

const initialState: MarketState = {
  prices: [],
  orderBooks: {},
  recentTrades: {},
  currentSymbol: 'BTC/USDT',
  currentExchange: 'binance',
  orderBook: null,
  isConnected: false,
  isLoading: false,
  lastUpdate: null,
  error: null,
};

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    setMarketPrices: (state, action: PayloadAction<MarketData[]>) => {
      state.prices = action.payload;
      state.lastUpdate = new Date().toISOString();
    },
    updateMarketPrice: (state, action: PayloadAction<MarketData>) => {
      const index = state.prices.findIndex(
        p => p.symbol === action.payload.symbol && p.exchange === action.payload.exchange
      );
      if (index !== -1) {
        state.prices[index] = action.payload;
      } else {
        state.prices.push(action.payload);
      }
      state.lastUpdate = new Date().toISOString();
    },
    setOrderBook: (state, action: PayloadAction<{ key: string; data: OrderBook }>) => {
      state.orderBooks[action.payload.key] = action.payload.data;
    },
    setRecentTrades: (state, action: PayloadAction<{ key: string; trades: Trade[] }>) => {
      state.recentTrades[action.payload.key] = action.payload.trades;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setCurrentSymbol: (state, action: PayloadAction<string>) => {
      state.currentSymbol = action.payload;
    },
    setCurrentExchange: (state, action: PayloadAction<string>) => {
      state.currentExchange = action.payload;
    },
  },
});

export const {
  setMarketPrices,
  updateMarketPrice,
  setOrderBook,
  setRecentTrades,
  setLoading,
  setError,
  setCurrentSymbol,
  setCurrentExchange,
} = marketSlice.actions;

export default marketSlice.reducer;