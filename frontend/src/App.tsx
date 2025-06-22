import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ConfigProvider, theme } from 'antd';
import { store } from './store';
import { useAppDispatch, useAppSelector } from './store/hooks';
import { getCurrentUserAsync } from './store/slices/authSlice';
import { websocketService } from './services/websocket';

// Components
import DashboardLayout from './components/Layout/DashboardLayout';
import LoginPage from './components/Auth/LoginPage';
import LoadingScreen from './components/Common/LoadingScreen';

// Pages
import DashboardOverview from './components/Dashboard/DashboardOverview';
import TradingDashboard from './components/Trading/TradingDashboard';
import PortfolioDashboard from './components/Portfolio/PortfolioDashboard';
import DataCollectionDashboard from './components/DataCollection/DataCollectionDashboard';
import ModelsDashboard from './components/Models/ModelsDashboard';
import SystemDashboard from './components/System/SystemDashboard';
import ConfigurationDashboard from './components/Configuration/ConfigurationDashboard';
import LogsDashboard from './components/Logs/LogsDashboard';

import './App.css';
import './styles/themes.css';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAppSelector((state) => state.auth);
  
  if (isLoading) {
    return <LoadingScreen />;
  }
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

const AppContent: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isAuthenticated, isLoading } = useAppSelector((state) => state.auth);
  const { theme: uiTheme } = useAppSelector((state) => state.ui);

  useEffect(() => {
    // Check for existing token on app start
    const token = localStorage.getItem('access_token');
    if (token && !isAuthenticated) {
      dispatch(getCurrentUserAsync());
    }
  }, [dispatch, isAuthenticated]);

  useEffect(() => {
    // Initialize WebSocket connection when authenticated
    if (isAuthenticated) {
      websocketService.connect();
      websocketService.startKeepAlive();
      
      // Subscribe to essential channels
      websocketService.startMarketDataStream();
      websocketService.startPortfolioStream();
      websocketService.startSystemAlerts();
      
      return () => {
        websocketService.disconnect();
      };
    }
  }, [isAuthenticated]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <div className={uiTheme === 'dark' ? 'dark-theme' : ''}>
      <ConfigProvider
        theme={{
          algorithm: uiTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
          token: {
            colorPrimary: '#2563eb',
            colorSuccess: '#10b981',
            colorWarning: '#f59e0b',
            colorError: '#ef4444',
            colorInfo: '#3b82f6',
            borderRadius: 12,
            borderRadiusSM: 8,
            borderRadiusLG: 16,
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            boxShadowSecondary: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            fontSize: 14,
            fontSizeHeading1: 32,
            fontSizeHeading2: 24,
            fontSizeHeading3: 20,
            colorBgContainer: '#ffffff',
            colorBgElevated: '#ffffff',
            colorBorder: '#e2e8f0',
            colorBorderSecondary: '#f1f5f9',
            colorText: '#1e293b',
            colorTextSecondary: '#64748b',
            colorTextTertiary: '#94a3b8',
            wireframe: false,
          },
          components: {
            Card: {
              borderRadiusLG: 12,
              boxShadowTertiary: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            },
            Button: {
              borderRadius: 8,
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            },
            Table: {
              borderRadiusLG: 8,
            },
            Layout: {
              siderBg: uiTheme === 'dark' ? '#1f1f1f' : '#f8fafc',
              headerBg: uiTheme === 'dark' ? '#1f1f1f' : '#ffffff',
            },
          },
        }}
      >
      <Router>
        <Routes>
          <Route 
            path="/login" 
            element={
              isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
            } 
          />
          
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <DashboardLayout>
                  <Routes>
                    <Route path="/" element={<DashboardOverview />} />
                    <Route path="/trading" element={<TradingDashboard />} />
                    <Route path="/portfolio" element={<PortfolioDashboard />} />
                    <Route path="/data" element={<DataCollectionDashboard />} />
                    <Route path="/models" element={<ModelsDashboard />} />
                    <Route path="/system" element={<SystemDashboard />} />
                    <Route path="/config" element={<ConfigurationDashboard />} />
                    <Route path="/logs" element={<LogsDashboard />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                  </Routes>
                </DashboardLayout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
      </ConfigProvider>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <AppContent />
    </Provider>
  );
};

export default App;