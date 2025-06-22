import React, { useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, Table, Tag, Typography } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, DollarOutlined, TrophyOutlined, WarningOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/store';
import apiService from '@/services/api';

const { Title } = Typography;

const DashboardOverview: React.FC = () => {
  const dispatch = useAppDispatch();
  const { snapshot } = useAppSelector((state) => state.portfolio);
  const { status } = useAppSelector((state) => state.trading);
  const { metrics, services } = useAppSelector((state) => state.system);

  useEffect(() => {
    // Load initial data
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      // This will be implemented with Redux actions later
      console.log('Loading dashboard data...');
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };

  // Mock data for demonstration
  const portfolioValue = 12453.67;
  const dailyPnL = 234.56;
  const dailyReturn = 1.9;
  const positions = [
    { symbol: 'BTC/USD', value: 7500, pnl: 156.78, pnlPercent: 2.1, status: 'profit' },
    { symbol: 'ETH/USD', value: 3200, pnl: -45.23, pnlPercent: -1.4, status: 'loss' },
  ];

  const recentTrades = [
    { time: '09:32', action: 'Buy', symbol: 'BTC/USD', amount: 0.05, price: 43567, status: 'completed' },
    { time: '09:28', action: 'Sell', symbol: 'ETH/USD', amount: 1.2, price: 3234, status: 'completed' },
    { time: '09:25', action: 'Buy', symbol: 'BTC/USD', amount: 0.03, price: 43445, status: 'completed' },
  ];

  const systemHealth = [
    { service: 'Data Collector', status: 'running', cpu: 45, memory: 62 },
    { service: 'Paper Trader', status: 'running', cpu: 23, memory: 41 },
    { service: 'Kafka', status: 'running', cpu: 34, memory: 55 },
    { service: 'Model Inference', status: 'stopped', cpu: 0, memory: 0 },
  ];

  const tradeColumns = [
    { title: 'Time', dataIndex: 'time', key: 'time' },
    { title: 'Action', dataIndex: 'action', key: 'action', 
      render: (action: string) => (
        <Tag color={action === 'Buy' ? 'green' : 'red'}>{action}</Tag>
      )
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
    { title: 'Amount', dataIndex: 'amount', key: 'amount' },
    { title: 'Price', dataIndex: 'price', key: 'price', 
      render: (price: number) => `$${price.toLocaleString()}`
    },
    { title: 'Status', dataIndex: 'status', key: 'status',
      render: (status: string) => (
        <Tag color={status === 'completed' ? 'green' : 'orange'}>{status}</Tag>
      )
    },
  ];

  const serviceColumns = [
    { title: 'Service', dataIndex: 'service', key: 'service' },
    { title: 'Status', dataIndex: 'status', key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'green' : 'red'}>{status}</Tag>
      )
    },
    { title: 'CPU %', dataIndex: 'cpu', key: 'cpu',
      render: (cpu: number) => `${cpu}%`
    },
    { title: 'Memory %', dataIndex: 'memory', key: 'memory',
      render: (memory: number) => `${memory}%`
    },
  ];

  return (
    <div>
      <Title level={2}>Dashboard Overview</Title>
      
      {/* Portfolio Summary */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Portfolio Value"
              value={portfolioValue}
              precision={2}
              prefix={<DollarOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Today's P&L"
              value={dailyPnL}
              precision={2}
              prefix={<DollarOutlined />}
              suffix={
                dailyReturn >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />
              }
              valueStyle={{ color: dailyReturn >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Daily Return"
              value={dailyReturn}
              precision={1}
              suffix="%"
              valueStyle={{ color: dailyReturn >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Open Positions"
              value={positions.length}
              prefix={<TrophyOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Current Positions */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="Current Positions" size="small">
            {positions.map((pos, index) => (
              <div key={index} style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '8px 0',
                borderBottom: index < positions.length - 1 ? '1px solid #f0f0f0' : 'none'
              }}>
                <div>
                  <strong>{pos.symbol}</strong>
                  <div style={{ color: '#666', fontSize: '12px' }}>
                    ${pos.value.toLocaleString()}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ 
                    color: pos.status === 'profit' ? '#3f8600' : '#cf1322',
                    fontWeight: 'bold'
                  }}>
                    ${pos.pnl.toFixed(2)}
                  </div>
                  <div style={{ 
                    color: pos.status === 'profit' ? '#3f8600' : '#cf1322',
                    fontSize: '12px'
                  }}>
                    {pos.pnlPercent > 0 ? '+' : ''}{pos.pnlPercent}%
                  </div>
                </div>
              </div>
            ))}
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="System Health" size="small">
            <Table
              dataSource={systemHealth}
              columns={serviceColumns}
              size="small"
              pagination={false}
              rowKey="service"
            />
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Recent Trades" size="small">
            <Table
              dataSource={recentTrades}
              columns={tradeColumns}
              size="small"
              pagination={false}
              rowKey={(record, index) => index || 0}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DashboardOverview;