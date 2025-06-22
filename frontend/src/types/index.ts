// API Response Types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
  status?: string;
}

// Authentication Types
export interface User {
  username: string;
  role: 'admin' | 'trader' | 'viewer';
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

// Market Data Types
export interface MarketData {
  symbol: string;
  exchange: string;
  price: number;
  volume_24h: number;
  change_24h: number;
  change_percent_24h: number;
  high_24h: number;
  low_24h: number;
  last_updated: string;
}

export interface OHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBookEntry {
  price: number;
  quantity: number;
  count?: number;
}

export interface OrderBook {
  symbol: string;
  exchange: string;
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  mid_price: number;
  spread: number;
  spread_percent: number;
  last_updated: string;
}

export interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  exchange: string;
  price: number;
  quantity: number;
  side: 'buy' | 'sell';
  is_maker?: boolean;
}

// Portfolio Types
export interface PortfolioSnapshot {
  timestamp: string;
  cash_balance: number;
  total_value: number;
  positions_value: number;
  daily_pnl: number;
  daily_return: number;
  total_return: number;
  max_drawdown: number;
  positions_count: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  position_value: number;
  weight_percent: number;
}

export interface PerformanceMetrics {
  total_return: number;
  total_return_percent: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  best_trade: number;
  worst_trade: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_trade_duration_hours: number;
  risk_adjusted_return: number;
}

// Trading Types
export interface TradingStatus {
  status: 'active' | 'paused' | 'stopped';
  portfolio: {
    cash_balance: number;
    total_value: number;
    daily_pnl: number;
    max_drawdown: number;
    positions_count: number;
    positions_value: number;
  };
  today: {
    trades_count: number;
    pnl: number;
  };
  risk_utilization: {
    position_count: string;
    capital_deployed: string;
    drawdown: string;
  };
}

export interface OrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
}

// System Types
export interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_available_gb: number;
  memory_total_gb: number;
  disk_percent: number;
  disk_free_gb: number;
  gpu_usage?: number;
  gpu_memory_mb?: number;
  network_bytes_sent: number;
  network_bytes_recv: number;
  timestamp: string;
}

export interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  pid?: number;
  cpu_percent?: number;
  memory_mb?: number;
  uptime_seconds?: number;
  last_error?: string;
}

// Model Types
export interface Model {
  id: string;
  name: string;
  type: 'decision_transformer' | 'price_prediction' | 'volatility_prediction' | 'multi_task';
  version: string;
  created_at: string;
  accuracy?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  status: 'training' | 'active' | 'inactive' | 'failed' | 'deployed';
  file_size_mb?: number;
  description?: string;
  metrics?: Record<string, any>;
}

export interface TrainingJob {
  job_id: string;
  model_type: Model['type'];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  validation_loss?: number;
  started_at: string;
  estimated_completion?: string;
  error_message?: string;
  metrics?: Record<string, any>;
}

// Configuration Types
export interface ConfigFile {
  name: string;
  path: string;
  last_modified: string;
  size_bytes: number;
  description: string;
  editable: boolean;
}

// Log Types
export interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  service: string;
  message: string;
  file?: string;
  line_number?: number;
  exception?: string;
}

export interface LogStatistics {
  total_entries: number;
  entries_by_level: Record<string, number>;
  entries_by_service: Record<string, number>;
  error_rate: number;
  warning_rate: number;
  most_common_errors: Array<{ message: string; count: number }>;
  time_range: { start: string; end: string };
}

// WebSocket Types
export interface WebSocketMessage<T = any> {
  type: string;
  data: T;
  timestamp: string;
}

export interface MarketUpdate {
  type: 'market_update';
  data: MarketData[];
  timestamp: string;
}

export interface PortfolioUpdate {
  type: 'portfolio_update';
  data: {
    timestamp: number;
    cash_balance: number;
    total_value: number;
    daily_pnl: number;
  };
  timestamp: string;
}

// UI State Types
export interface DashboardState {
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark';
  activeMenuItem: string;
  notifications: Notification[];
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}

// Chart Types
export interface ChartData {
  timestamps: number[];
  values: number[];
}

export interface CandlestickData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

// Filter and Pagination Types
export interface PaginationParams {
  page: number;
  pageSize: number;
}

export interface FilterParams {
  search?: string;
  dateRange?: [string, string];
  status?: string;
  level?: string;
}