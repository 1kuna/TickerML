import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  ApiResponse, 
  AuthTokens, 
  LoginCredentials, 
  User,
  MarketData,
  OHLCV,
  OrderBook,
  Trade,
  PortfolioSnapshot,
  PerformanceMetrics,
  Position,
  TradingStatus,
  OrderRequest,
  SystemMetrics,
  ServiceStatus,
  Model,
  TrainingJob,
  ConfigFile,
  LogEntry,
  LogStatistics
} from '../types';

class ApiService {
  private api: AxiosInstance;
  
  constructor() {
    this.api = axios.create({
      baseURL: '/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor to handle token refresh
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          const refreshToken = localStorage.getItem('refresh_token');
          if (refreshToken) {
            try {
              const response = await this.refreshToken(refreshToken);
              localStorage.setItem('access_token', response.access_token);
              
              // Retry original request
              originalRequest.headers.Authorization = `Bearer ${response.access_token}`;
              return this.api(originalRequest);
            } catch (refreshError) {
              // Refresh failed, redirect to login
              this.logout();
              window.location.href = '/login';
            }
          } else {
            // No refresh token, redirect to login
            this.logout();
            window.location.href = '/login';
          }
        }
        
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(credentials: LoginCredentials): Promise<AuthTokens> {
    const response: AxiosResponse<AuthTokens> = await this.api.post('/auth/login', credentials);
    return response.data;
  }

  async refreshToken(refreshToken: string): Promise<{ access_token: string; token_type: string }> {
    const response = await this.api.post('/auth/refresh', { refresh_token: refreshToken });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response: AxiosResponse<User> = await this.api.get('/auth/me');
    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await this.api.post('/auth/logout');
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    }
  }

  // Market Data
  async getMarketPrices(params?: { symbols?: string; exchange?: string }): Promise<MarketData[]> {
    const response: AxiosResponse<MarketData[]> = await this.api.get('/market/prices', { params });
    return response.data;
  }

  async getOHLCVData(
    symbol: string,
    params?: {
      exchange?: string;
      interval?: string;
      start_time?: string;
      end_time?: string;
      limit?: number;
    }
  ): Promise<OHLCV[]> {
    const response: AxiosResponse<OHLCV[]> = await this.api.get(`/market/ohlcv/${symbol}`, { params });
    return response.data;
  }

  async getOrderBook(symbol: string, exchange: string, depth: number = 20): Promise<OrderBook> {
    const response: AxiosResponse<OrderBook> = await this.api.get(
      `/market/orderbook/${symbol}`,
      { params: { exchange, depth } }
    );
    return response.data;
  }

  async getRecentTrades(
    symbol: string,
    params?: { exchange?: string; limit?: number }
  ): Promise<Trade[]> {
    const response: AxiosResponse<Trade[]> = await this.api.get(`/market/trades/${symbol}`, { params });
    return response.data;
  }

  async getMarketSummary(): Promise<any> {
    const response = await this.api.get('/market/summary');
    return response.data;
  }

  async getExchanges(): Promise<any[]> {
    const response = await this.api.get('/market/exchanges');
    return response.data;
  }

  // Portfolio
  async getPortfolioSnapshot(): Promise<PortfolioSnapshot> {
    const response: AxiosResponse<PortfolioSnapshot> = await this.api.get('/portfolio/snapshot');
    return response.data;
  }

  async getPortfolioHistory(hours: number = 24): Promise<any> {
    const response = await this.api.get('/portfolio/history', { params: { hours } });
    return response.data;
  }

  async getPerformanceMetrics(days: number = 30): Promise<PerformanceMetrics> {
    const response: AxiosResponse<PerformanceMetrics> = await this.api.get('/portfolio/performance', { params: { days } });
    return response.data;
  }

  async getPositions(): Promise<Position[]> {
    const response: AxiosResponse<Position[]> = await this.api.get('/portfolio/positions');
    return response.data;
  }

  async getRiskMetrics(): Promise<any> {
    const response = await this.api.get('/portfolio/risk');
    return response.data;
  }

  async getDailySummary(date?: string): Promise<any> {
    const response = await this.api.get('/portfolio/daily-summary', { params: { date } });
    return response.data;
  }

  // Trading
  async getTradingStatus(): Promise<TradingStatus> {
    const response: AxiosResponse<TradingStatus> = await this.api.get('/trading/status');
    return response.data;
  }

  async startTrading(config: any): Promise<any> {
    const response = await this.api.post('/trading/start', config);
    return response.data;
  }

  async stopTrading(): Promise<any> {
    const response = await this.api.post('/trading/stop');
    return response.data;
  }

  async pauseTrading(): Promise<any> {
    const response = await this.api.post('/trading/pause');
    return response.data;
  }

  async resumeTrading(): Promise<any> {
    const response = await this.api.post('/trading/resume');
    return response.data;
  }

  async placeOrder(order: OrderRequest): Promise<any> {
    const response = await this.api.post('/trading/orders', order);
    return response.data;
  }

  async getTradingSettings(): Promise<any> {
    const response = await this.api.get('/trading/settings');
    return response.data;
  }

  async updateTradingSettings(settings: any): Promise<any> {
    const response = await this.api.put('/trading/settings', settings);
    return response.data;
  }

  async getTradeHistory(limit: number = 50, offset: number = 0): Promise<any[]> {
    const response = await this.api.get('/trading/trades/history', { params: { limit, offset } });
    return response.data;
  }

  // System
  async getSystemStatus(): Promise<{
    system_metrics: SystemMetrics;
    services: Record<string, ServiceStatus>;
    kafka: { running: boolean; status: string };
  }> {
    const response = await this.api.get('/system/status');
    return response.data;
  }

  async startService(serviceName: string): Promise<any> {
    const response = await this.api.post(`/system/services/${serviceName}/start`);
    return response.data;
  }

  async stopService(serviceName: string): Promise<any> {
    const response = await this.api.post(`/system/services/${serviceName}/stop`);
    return response.data;
  }

  async restartService(serviceName: string): Promise<any> {
    const response = await this.api.post(`/system/services/${serviceName}/restart`);
    return response.data;
  }

  async startAllServices(): Promise<any> {
    const response = await this.api.post('/system/services/all/start');
    return response.data;
  }

  async stopAllServices(): Promise<any> {
    const response = await this.api.post('/system/services/all/stop');
    return response.data;
  }

  async getServices(): Promise<any[]> {
    const response = await this.api.get('/system/services');
    return response.data;
  }

  async getSystemServices(): Promise<any[]> {
    const response = await this.api.get('/system/services');
    return response.data;
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await this.api.get('/system/metrics');
    return response.data;
  }

  async placePaperOrder(orderData: any): Promise<any> {
    const response = await this.api.post('/trading/paper/order', orderData);
    return response.data;
  }

  async updateServiceConfig(serviceName: string, config: any): Promise<any> {
    const response = await this.api.put(`/system/services/${serviceName}/config`, config);
    return response.data;
  }

  // Data Collection
  async getCollectors(): Promise<any> {
    const response = await this.api.get('/data/collectors');
    return response.data;
  }

  async configureCollector(collectorType: string, config: any): Promise<any> {
    const response = await this.api.post(`/data/collectors/${collectorType}/configure`, config);
    return response.data;
  }

  async getDataStatistics(hours: number = 24): Promise<any> {
    const response = await this.api.get('/data/statistics', { params: { hours } });
    return response.data;
  }

  async getDataQuality(): Promise<any[]> {
    const response = await this.api.get('/data/quality');
    return response.data;
  }

  // Models
  async getModels(): Promise<Model[]> {
    const response: AxiosResponse<Model[]> = await this.api.get('/models/');
    return response.data;
  }

  async getModel(modelId: string): Promise<Model> {
    const response: AxiosResponse<Model> = await this.api.get(`/models/${modelId}`);
    return response.data;
  }

  async startTraining(config: any): Promise<any> {
    const response = await this.api.post('/models/train', config);
    return response.data;
  }

  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    const response: AxiosResponse<TrainingJob> = await this.api.get(`/models/training/${jobId}`);
    return response.data;
  }

  async cancelTraining(jobId: string): Promise<any> {
    const response = await this.api.delete(`/models/training/${jobId}`);
    return response.data;
  }

  async deployModel(modelId: string, config: any): Promise<any> {
    const response = await this.api.post(`/models/${modelId}/deploy`, config);
    return response.data;
  }

  async getModelMetrics(modelId: string): Promise<any> {
    const response = await this.api.get(`/models/${modelId}/metrics`);
    return response.data;
  }

  async deleteModel(modelId: string): Promise<any> {
    const response = await this.api.delete(`/models/${modelId}`);
    return response.data;
  }

  async getTrainingJobs(): Promise<TrainingJob[]> {
    const response: AxiosResponse<TrainingJob[]> = await this.api.get('/models/training/jobs');
    return response.data;
  }

  // Configuration
  async getConfigFiles(): Promise<ConfigFile[]> {
    const response: AxiosResponse<ConfigFile[]> = await this.api.get('/config/files');
    return response.data;
  }

  async getConfig(fileName: string): Promise<any> {
    const response = await this.api.get(`/config/${fileName}`);
    return response.data;
  }

  async updateConfig(fileName: string, config: any, backup: boolean = true): Promise<any> {
    const response = await this.api.put(`/config/${fileName}`, {
      file_name: fileName,
      content: config,
      backup
    });
    return response.data;
  }

  async getConfigHistory(fileName: string): Promise<any[]> {
    const response = await this.api.get(`/config/${fileName}/history`);
    return response.data;
  }

  async validateConfig(fileName: string, config: any): Promise<any> {
    const response = await this.api.post(`/config/validate/${fileName}`, config);
    return response.data;
  }

  async getConfigSchema(fileName: string): Promise<any> {
    const response = await this.api.get(`/config/schema/${fileName}`);
    return response.data;
  }

  // Logs
  async getLogs(params?: {
    service?: string;
    level?: string;
    lines?: number;
    search?: string;
    since?: string;
  }): Promise<LogEntry[]> {
    const response: AxiosResponse<LogEntry[]> = await this.api.get('/logs/', { params });
    return response.data;
  }

  async getLogStatistics(hours: number = 24): Promise<LogStatistics> {
    const response: AxiosResponse<LogStatistics> = await this.api.get('/logs/statistics', { params: { hours } });
    return response.data;
  }

  async searchLogs(params: {
    pattern: string;
    service?: string;
    level?: string;
    max_results?: number;
  }): Promise<string[]> {
    const response: AxiosResponse<string[]> = await this.api.get('/logs/search', { params });
    return response.data;
  }

  async tailServiceLogs(service: string, lines: number = 50): Promise<string[]> {
    const response: AxiosResponse<string[]> = await this.api.get(`/logs/tail/${service}`, { params: { lines } });
    return response.data;
  }

  async getLogHealth(): Promise<any> {
    const response = await this.api.get('/logs/health');
    return response.data;
  }

  // Health Check
  async getHealth(): Promise<any> {
    const response = await this.api.get('/health', { baseURL: '/api' });
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;