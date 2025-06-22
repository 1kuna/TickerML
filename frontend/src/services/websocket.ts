import { io, Socket } from 'socket.io-client';
import { useEffect, useCallback, useRef } from 'react';
import { WebSocketMessage, MarketUpdate, PortfolioUpdate } from '@/types';

export type WebSocketEventHandler<T = any> = (data: T) => void;

interface ThrottleConfig {
  limit: number;
  windowMs: number;
}

interface ConnectionPool {
  market: Socket | null;
  portfolio: Socket | null;
  trading: Socket | null;
  system: Socket | null;
}

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<string, Set<WebSocketEventHandler>> = new Map();
  
  // Performance optimization features
  private connectionPool: ConnectionPool = {
    market: null,
    portfolio: null,
    trading: null,
    system: null,
  };
  
  private throttleConfigs: Map<string, ThrottleConfig> = new Map([
    ['market_update', { limit: 10, windowMs: 1000 }], // Max 10 updates per second
    ['orderbook_update', { limit: 5, windowMs: 1000 }], // Max 5 orderbook updates per second
    ['trade_update', { limit: 20, windowMs: 1000 }], // Max 20 trade updates per second
    ['portfolio_update', { limit: 2, windowMs: 1000 }], // Max 2 portfolio updates per second
  ]);
  
  private throttleTrackers: Map<string, number[]> = new Map();
  private lastEventData: Map<string, any> = new Map();
  private batchBuffer: Map<string, any[]> = new Map();
  private batchTimeout: Map<string, NodeJS.Timeout> = new Map();

  connect(url: string = ''): void {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }

    const token = localStorage.getItem('access_token');
    
    this.socket = io(url, {
      auth: {
        token: token ? `Bearer ${token}` : undefined,
      },
      transports: ['websocket'],
      upgrade: true,
      rememberUpgrade: true,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.handleReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.emit('connection', { status: 'error', error: error.message });
      this.handleReconnect();
    });

    // Market data events
    this.socket.on('market_update', (data: MarketUpdate) => {
      this.emit('market_update', data);
    });

    // Portfolio events
    this.socket.on('portfolio_update', (data: PortfolioUpdate) => {
      this.emit('portfolio_update', data);
    });

    // Trading events
    this.socket.on('trade_executed', (data: any) => {
      this.emit('trade_executed', data);
    });

    this.socket.on('order_update', (data: any) => {
      this.emit('order_update', data);
    });

    // System events
    this.socket.on('system_alert', (data: any) => {
      this.emit('system_alert', data);
    });

    this.socket.on('service_status', (data: any) => {
      this.emit('service_status', data);
    });

    // Model training events
    this.socket.on('training_progress', (data: any) => {
      this.emit('training_progress', data);
    });

    // Log events
    this.socket.on('log_entry', (data: any) => {
      this.emit('log_entry', data);
    });

    // Generic message handler
    this.socket.onAny((event: string, data: any) => {
      console.log('WebSocket event:', event, data);
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.eventHandlers.clear();
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('connection', { 
        status: 'failed', 
        error: 'Max reconnection attempts reached' 
      });
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  // Throttling mechanism
  private isThrottled(event: string): boolean {
    const config = this.throttleConfigs.get(event);
    if (!config) return false;

    const now = Date.now();
    let tracker = this.throttleTrackers.get(event) || [];
    
    // Remove old entries outside the window
    tracker = tracker.filter(timestamp => now - timestamp < config.windowMs);
    
    if (tracker.length >= config.limit) {
      return true;
    }

    tracker.push(now);
    this.throttleTrackers.set(event, tracker);
    return false;
  }

  // Data deduplication
  private isDuplicateData(event: string, data: any): boolean {
    const lastData = this.lastEventData.get(event);
    if (!lastData) {
      this.lastEventData.set(event, data);
      return false;
    }

    // Simple deep comparison for basic data types
    if (JSON.stringify(lastData) === JSON.stringify(data)) {
      return true;
    }

    this.lastEventData.set(event, data);
    return false;
  }

  // Batching mechanism for high-frequency events
  private addToBatch(event: string, data: any): void {
    if (!this.batchBuffer.has(event)) {
      this.batchBuffer.set(event, []);
    }

    this.batchBuffer.get(event)!.push(data);

    // Clear existing timeout and set new one
    const existingTimeout = this.batchTimeout.get(event);
    if (existingTimeout) {
      clearTimeout(existingTimeout);
    }

    const timeout = setTimeout(() => {
      this.flushBatch(event);
    }, 100); // Batch for 100ms

    this.batchTimeout.set(event, timeout);
  }

  private flushBatch(event: string): void {
    const batch = this.batchBuffer.get(event);
    if (batch && batch.length > 0) {
      this.emit(`${event}_batch`, batch);
      this.batchBuffer.set(event, []);
    }
    this.batchTimeout.delete(event);
  }

  // Subscription methods
  subscribeToMarketData(symbols: string[]): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', {
        channel: 'market',
        symbols: symbols,
      });
    }
  }

  subscribeToPortfolio(): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', {
        channel: 'portfolio',
      });
    }
  }

  subscribeToTrades(): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', {
        channel: 'trades',
      });
    }
  }

  subscribeToOrderBook(symbol: string, exchange: string): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', {
        channel: 'orderbook',
        symbol: symbol,
        exchange: exchange,
      });
    }
  }

  subscribeToAlerts(): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', {
        channel: 'alerts',
      });
    }
  }

  unsubscribeFromChannel(channel: string): void {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { channel });
    }
  }

  // Event handling
  on<T = any>(event: string, handler: WebSocketEventHandler<T>): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  off<T = any>(event: string, handler: WebSocketEventHandler<T>): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(event);
      }
    }
  }

  private emit<T = any>(event: string, data: T): void {
    // Apply throttling for high-frequency events
    if (this.isThrottled(event)) {
      return;
    }

    // Skip duplicate data
    if (this.isDuplicateData(event, data)) {
      return;
    }

    // Use batching for certain high-frequency events
    const batchableEvents = ['market_update', 'orderbook_update', 'trade_update'];
    if (batchableEvents.includes(event)) {
      this.addToBatch(event, data);
      return;
    }

    // Emit event normally
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  // Send messages
  send(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  // Status
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  getConnectionState(): string {
    if (!this.socket) return 'disconnected';
    return this.socket.connected ? 'connected' : 'connecting';
  }

  // Utility methods for specific subscriptions
  startMarketDataStream(symbols: string[] = ['BTCUSD', 'ETHUSD']): void {
    this.subscribeToMarketData(symbols);
  }

  startPortfolioStream(): void {
    this.subscribeToPortfolio();
  }

  startTradingStream(): void {
    this.subscribeToTrades();
  }

  startSystemAlerts(): void {
    this.subscribeToAlerts();
  }

  // Ping to keep connection alive
  ping(): void {
    if (this.socket?.connected) {
      this.socket.emit('ping');
    }
  }

  // Start keep-alive ping
  startKeepAlive(intervalMs: number = 30000): void {
    setInterval(() => {
      this.ping();
    }, intervalMs);
  }

  // Connection pooling methods
  getSpecializedConnection(type: keyof ConnectionPool): Socket | null {
    return this.connectionPool[type];
  }

  createSpecializedConnection(type: keyof ConnectionPool, url: string = ''): void {
    if (this.connectionPool[type]?.connected) {
      return;
    }

    const token = localStorage.getItem('access_token');
    const socket = io(`${url}/${type}`, {
      auth: { token: token ? `Bearer ${token}` : undefined },
      transports: ['websocket'],
    });

    socket.on('connect', () => {
      console.log(`Specialized ${type} WebSocket connected`);
    });

    socket.on('disconnect', () => {
      console.log(`Specialized ${type} WebSocket disconnected`);
      this.connectionPool[type] = null;
    });

    this.connectionPool[type] = socket;
  }

  // Performance monitoring
  getPerformanceMetrics(): object {
    return {
      throttleTrackers: Object.fromEntries(this.throttleTrackers),
      batchBufferSizes: Object.fromEntries(
        Array.from(this.batchBuffer.entries()).map(([key, batch]) => [key, batch.length])
      ),
      connectionStates: Object.fromEntries(
        Object.entries(this.connectionPool).map(([key, socket]) => [
          key,
          socket?.connected ? 'connected' : 'disconnected'
        ])
      ),
      mainConnectionState: this.getConnectionState(),
    };
  }

  // Clean up resources
  cleanup(): void {
    // Clear all timeouts
    this.batchTimeout.forEach(timeout => clearTimeout(timeout));
    this.batchTimeout.clear();
    
    // Clear data caches
    this.lastEventData.clear();
    this.throttleTrackers.clear();
    this.batchBuffer.clear();
    
    // Close specialized connections
    Object.values(this.connectionPool).forEach(socket => {
      if (socket?.connected) {
        socket.disconnect();
      }
    });
    
    // Close main connection
    this.disconnect();
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// React hooks for WebSocket integration

export const useWebSocket = () => {
  return websocketService;
};

// Enhanced hook for performance monitoring
export const useWebSocketWithMetrics = () => {
  const metricsRef = useRef<any>(null);
  
  const getMetrics = useCallback(() => {
    metricsRef.current = websocketService.getPerformanceMetrics();
    return metricsRef.current;
  }, []);

  const cleanup = useCallback(() => {
    websocketService.cleanup();
  }, []);

  return {
    websocket: websocketService,
    getMetrics,
    cleanup,
  };
};

// Hook for throttled event listening
export const useThrottledWebSocketEvent = <T = any>(
  event: string,
  handler: WebSocketEventHandler<T>,
  throttleMs: number = 1000
) => {
  const lastCallRef = useRef<number>(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const throttledHandler = useCallback((data: T) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCallRef.current;

    if (timeSinceLastCall >= throttleMs) {
      lastCallRef.current = now;
      handler(data);
    } else {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      timeoutRef.current = setTimeout(() => {
        lastCallRef.current = Date.now();
        handler(data);
      }, throttleMs - timeSinceLastCall);
    }
  }, [handler, throttleMs]);

  useEffect(() => {
    websocketService.on(event, throttledHandler);
    return () => {
      websocketService.off(event, throttledHandler);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [event, throttledHandler]);
};

export default websocketService;