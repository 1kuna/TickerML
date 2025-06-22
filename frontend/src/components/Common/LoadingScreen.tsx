import React from 'react';
import { Spin } from 'antd';

const LoadingScreen: React.FC = () => {
  return (
    <div 
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        flexDirection: 'column',
        background: '#001529',
        color: '#fff'
      }}
    >
      <Spin size="large" />
      <div style={{ marginTop: 16, fontSize: 16 }}>
        Loading TickerML Dashboard...
      </div>
    </div>
  );
};

export default LoadingScreen;