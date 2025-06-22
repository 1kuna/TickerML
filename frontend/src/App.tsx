import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ConfigProvider, theme } from 'antd';
import { store } from '@/store';
import { useAppDispatch, useAppSelector } from '@/store';
import { getCurrentUserAsync } from '@/store/slices/authSlice';
import { websocketService } from '@/services/websocket';

// Components
import DashboardLayout from '@/components/Layout/DashboardLayout';
import LoginPage from '@/components/Auth/LoginPage';
import LoadingScreen from '@/components/Common/LoadingScreen';

// Pages
import DashboardOverview from '@/components/Dashboard/DashboardOverview';
import TradingDashboard from '@/components/Trading/TradingDashboard';
import PortfolioDashboard from '@/components/Portfolio/PortfolioDashboard';
import DataCollectionDashboard from '@/components/DataCollection/DataCollectionDashboard';
import ModelsDashboard from '@/components/Models/ModelsDashboard';
import SystemDashboard from '@/components/System/SystemDashboard';
import ConfigurationDashboard from '@/components/Configuration/ConfigurationDashboard';
import LogsDashboard from '@/components/Logs/LogsDashboard';

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
            colorPrimary: '#1890ff',
            borderRadius: 6,
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