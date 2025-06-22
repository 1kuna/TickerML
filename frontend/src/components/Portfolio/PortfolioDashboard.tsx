import React, { useState, useEffect, useRef } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Button,
  Typography, 
  Statistic, 
  Tag,
  Progress,
  Select,
  DatePicker,
  Space,
  Alert,
  Tooltip as AntTooltip
} from 'antd';
import { 
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  PieChartOutlined,
  BarChartOutlined,
  ReloadOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { LineChart, Line, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { useAppSelector, useAppDispatch } from '../../store/hooks';
import { fetchPortfolioSummary, fetchPositions, fetchTradeHistory } from '../../store/slices/portfolioSlice';
import { Position, Trade, PortfolioSnapshot } from '../../types';
import { apiService } from '../../services/api';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

const PortfolioDashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { 
    summary, 
    positions, 
    tradeHistory, 
    performance, 
    isLoading 
  } = useAppSelector(state => state.portfolio);
  
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [allocationData, setAllocationData] = useState<any[]>([]);
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);

  // Load portfolio data on mount
  useEffect(() => {
    dispatch(fetchPortfolioSummary());
    dispatch(fetchPositions());
    dispatch(fetchTradeHistory());
  }, [dispatch]);

  // Format performance data for chart
  useEffect(() => {
    if (performance?.equity_curve) {
      const chartData = performance.equity_curve.map((point: any) => ({
        date: dayjs(point.timestamp).format('MM-DD HH:mm'),
        value: point.total_value,
        pnl: point.unrealized_pnl,
      }));
      setPerformanceData(chartData);
    }
  }, [performance]);

  // Format allocation data for pie chart
  useEffect(() => {
    if (positions?.length) {
      const totalValue = positions.reduce((sum, pos) => sum + pos.market_value, 0);
      const chartData = positions
        .filter(pos => pos.market_value > 0)
        .map((pos: Position) => ({
          symbol: pos.symbol,
          value: pos.market_value,
          percentage: ((pos.market_value / totalValue) * 100).toFixed(1),
        }));
      
      // Add cash allocation
      const cashValue = summary?.cash_balance || 0;
      if (cashValue > 0) {
        chartData.push({
          symbol: 'Cash',
          value: cashValue,
          percentage: ((cashValue / (totalValue + cashValue)) * 100).toFixed(1),
        });
      }
      
      setAllocationData(chartData);
    }
  }, [positions, summary]);

  const handleTimeRangeChange = (range: string) => {
    setTimeRange(range);
    // Fetch new performance data based on range
    // Implementation would call API with date range
  };

  const handleDateRangeChange = (dates: any) => {
    setDateRange(dates);
    if (dates && dates.length === 2) {
      // Fetch performance data for custom date range
    }
  };

  // Position columns
  const positionColumns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => <Text strong>{symbol}</Text>,
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (quantity: number) => quantity.toFixed(4),
    },
    {
      title: 'Entry Price',
      dataIndex: 'entry_price',
      key: 'entry_price',
      render: (price: number) => `$${price.toFixed(2)}`,
    },
    {
      title: 'Current Price',
      dataIndex: 'current_price',
      key: 'current_price',
      render: (price: number) => `$${price.toFixed(2)}`,
    },
    {
      title: 'Market Value',
      dataIndex: 'market_value',
      key: 'market_value',
      render: (value: number) => `$${value.toFixed(2)}`,
    },
    {
      title: 'P&L',
      dataIndex: 'unrealized_pnl',
      key: 'unrealized_pnl',
      render: (pnl: number, record: Position) => {
        const pnlPercent = record.entry_price ? 
          ((record.current_price - record.entry_price) / record.entry_price) * 100 : 0;
        return (
          <div>
            <div className={pnl >= 0 ? 'profit-color' : 'loss-color'}>
              ${pnl.toFixed(2)}
            </div>
            <div className={pnlPercent >= 0 ? 'profit-color' : 'loss-color'}>
              ({pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%)
            </div>
          </div>
        );
      },
    },
    {
      title: 'Weight',
      dataIndex: 'weight',
      key: 'weight',
      render: (weight: number) => (
        <div>
          <Progress 
            percent={weight * 100} 
            size="small" 
            showInfo={false}
            strokeColor={weight > 0.25 ? '#ff4d4f' : '#52c41a'}
          />
          <Text>{(weight * 100).toFixed(1)}%</Text>
        </div>
      ),
    },
  ];

  // Trade history columns
  const tradeColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => dayjs(timestamp).format('MM-DD HH:mm:ss'),
    },
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
    },
    {
      title: 'Side',
      dataIndex: 'side',
      key: 'side',
      render: (side: string) => (
        <Tag color={side === 'buy' ? 'green' : 'red'}>
          {side.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (quantity: number) => quantity.toFixed(4),
    },
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `$${price.toFixed(2)}`,
    },
    {
      title: 'Total',
      dataIndex: 'total_value',
      key: 'total_value',
      render: (value: number) => `$${value.toFixed(2)}`,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={
          status === 'filled' ? 'green' : 
          status === 'pending' ? 'orange' : 
          status === 'cancelled' ? 'red' : 'default'
        }>
          {status.toUpperCase()}
        </Tag>
      ),
    },
  ];

  // Performance chart config
  const performanceConfig = {
    data: performanceData,
    xField: 'date',
    yField: 'value',
    height: 300,
    smooth: true,
    color: '#1890ff',
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: 'Portfolio Value',
        value: `$${datum.value?.toFixed(2)}`,
      }),
    },
  };

  // Allocation chart config
  const allocationConfig = {
    data: allocationData,
    dataKey: 'value',
    angleField: 'value',
    colorField: 'symbol',
    height: 300,
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}%',
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: datum.symbol,
        value: `$${datum.value?.toFixed(2)} (${datum.percentage}%)`,
      }),
    },
  };

  const totalPnl = summary?.total_pnl || 0;
  const totalPnlPercent = summary?.total_value && summary?.cash_balance ? 
    ((summary.total_value - 10000) / 10000) * 100 : 0; // Assuming $10k starting balance
  const maxDrawdown = performance?.max_drawdown || 0;
  const sharpeRatio = performance?.sharpe_ratio || 0;

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>Portfolio Dashboard</Title>
          </Col>
          <Col>
            <Select value={timeRange} onChange={handleTimeRangeChange} style={{ width: 120 }}>
              <Option value="1d">1 Day</Option>
              <Option value="7d">1 Week</Option>
              <Option value="30d">1 Month</Option>
              <Option value="90d">3 Months</Option>
              <Option value="1y">1 Year</Option>
            </Select>
          </Col>
          <Col>
            <RangePicker 
              value={dateRange}
              onChange={handleDateRangeChange}
              format="YYYY-MM-DD"
            />
          </Col>
          <Col>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={() => {
                dispatch(fetchPortfolioSummary());
                dispatch(fetchPositions());
                dispatch(fetchTradeHistory());
              }}
            >
              Refresh
            </Button>
          </Col>
        </Row>
      </div>

      {/* Portfolio Summary */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Value"
              value={summary?.total_value || 0}
              precision={2}
              prefix={<DollarOutlined />}
              valueStyle={{ fontSize: '24px' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total P&L"
              value={totalPnl}
              precision={2}
              prefix={totalPnl >= 0 ? <RiseOutlined /> : <FallOutlined />}
              suffix={`(${totalPnlPercent >= 0 ? '+' : ''}${totalPnlPercent.toFixed(2)}%)`}
              valueStyle={{ 
                color: totalPnl >= 0 ? '#3f8600' : '#cf1322',
                fontSize: '20px' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Cash Balance"
              value={summary?.cash_balance || 0}
              precision={2}
              prefix={<DollarOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Positions"
              value={positions?.length || 0}
              suffix="symbols"
              prefix={<PieChartOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Risk Metrics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Max Drawdown"
              value={Math.abs(maxDrawdown)}
              precision={2}
              suffix="%"
              prefix={<WarningOutlined />}
              valueStyle={{ 
                color: Math.abs(maxDrawdown) > 10 ? '#cf1322' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Sharpe Ratio"
              value={sharpeRatio}
              precision={2}
              prefix={<BarChartOutlined />}
              valueStyle={{ 
                color: sharpeRatio > 1 ? '#3f8600' : sharpeRatio > 0 ? '#faad14' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Win Rate"
              value={performance?.win_rate || 0}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: (performance?.win_rate || 0) > 50 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg Trade"
              value={performance?.avg_trade_pnl || 0}
              precision={2}
              prefix={<DollarOutlined />}
              valueStyle={{ 
                color: (performance?.avg_trade_pnl || 0) >= 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Risk Alerts */}
      {maxDrawdown < -20 && (
        <Alert
          message="High Drawdown Alert"
          description={`Current drawdown of ${Math.abs(maxDrawdown).toFixed(2)}% exceeds recommended limits.`}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={16}>
        {/* Performance Chart */}
        <Col span={16}>
          <Card title="Portfolio Performance" loading={isLoading}>
            <Line {...performanceConfig} />
          </Card>
        </Col>

        {/* Asset Allocation */}
        <Col span={8}>
          <Card title="Asset Allocation" loading={isLoading}>
            <Pie {...allocationConfig} />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        {/* Current Positions */}
        <Col span={24}>
          <Card title="Current Positions" loading={isLoading}>
            <Table
              dataSource={positions || []}
              columns={positionColumns}
              pagination={false}
              scroll={{ x: 800 }}
              rowKey="symbol"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        {/* Recent Trades */}
        <Col span={24}>
          <Card title="Recent Trades" loading={isLoading}>
            <Table
              dataSource={tradeHistory?.slice(0, 50) || []}
              columns={tradeColumns}
              pagination={{
                pageSize: 20,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `Total ${total} trades`,
              }}
              scroll={{ x: 800 }}
              rowKey="id"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PortfolioDashboard;