import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { User, LoginCredentials, AuthTokens } from '../../types';
import { apiService } from '../../services/api';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  tokens: AuthTokens | null;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  tokens: null,
};

// Async thunks
export const loginAsync = createAsyncThunk(
  'auth/login',
  async (credentials: LoginCredentials, { rejectWithValue }) => {
    try {
      const tokens = await apiService.login(credentials);
      
      // Store tokens
      localStorage.setItem('access_token', tokens.access_token);
      localStorage.setItem('refresh_token', tokens.refresh_token);
      
      // Get user info
      const user = await apiService.getCurrentUser();
      
      return { user, tokens };
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Login failed';
      return rejectWithValue(message);
    }
  }
);

export const getCurrentUserAsync = createAsyncThunk(
  'auth/getCurrentUser',
  async (_, { rejectWithValue }) => {
    try {
      const user = await apiService.getCurrentUser();
      return user;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to get user';
      return rejectWithValue(message);
    }
  }
);

export const logoutAsync = createAsyncThunk(
  'auth/logout',
  async (_, { rejectWithValue }) => {
    try {
      await apiService.logout();
      return null;
    } catch (error: any) {
      // Still clear local state even if API call fails
      return null;
    }
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setTokens: (state, action: PayloadAction<AuthTokens>) => {
      state.tokens = action.payload;
      state.isAuthenticated = true;
    },
    clearAuth: (state) => {
      state.user = null;
      state.isAuthenticated = false;
      state.tokens = null;
      state.error = null;
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    },
  },
  extraReducers: (builder) => {
    // Login
    builder
      .addCase(loginAsync.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginAsync.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.tokens = action.payload.tokens;
        state.isAuthenticated = true;
        state.error = null;
      })
      .addCase(loginAsync.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.isAuthenticated = false;
        state.user = null;
        state.tokens = null;
      });

    // Get current user
    builder
      .addCase(getCurrentUserAsync.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(getCurrentUserAsync.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
        state.error = null;
      })
      .addCase(getCurrentUserAsync.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.isAuthenticated = false;
        state.user = null;
      });

    // Logout
    builder
      .addCase(logoutAsync.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logoutAsync.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.tokens = null;
        state.error = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      })
      .addCase(logoutAsync.rejected, (state) => {
        state.isLoading = false;
        // Still clear auth state on logout error
        state.user = null;
        state.isAuthenticated = false;
        state.tokens = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      });
  },
});

export const { clearError, setTokens, clearAuth } = authSlice.actions;
export default authSlice.reducer;