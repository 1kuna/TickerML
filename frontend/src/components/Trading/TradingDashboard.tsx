import React, { useState, useEffect, useRef } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Select, 
  Button, 
  Table, 
  Input, 
  Form, 
  Typography, 
  Statistic, 
  Tag,
  Space,
  Alert,
  Spin
} from 'antd';
import { 
  CaretUpOutlined, 
  CaretDownOutlined, 
  ReloadOutlined,
  DollarOutlined,
  RiseOutlined
} from '@ant-design/icons';
import { createChart, ColorType, IChartApi } from 'lightweight-charts';
import { useAppSelector, useAppDispatch } from '../../store/hooks';
import { setCurrentSymbol, setCurrentExchange } from '../../store/slices/marketSlice';
import { placePaperOrder } from '../../store/slices/tradingSlice';
import { OrderType, OrderSide, OrderBook, Trade, OHLCV } from '../../types';
import { apiService } from '../../services/api';
import { websocketService } from '../../services/websocket';

const { Title, Text } = Typography;
const { Option } = Select;

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const TradingDashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { currentSymbol, currentExchange, orderBook, recentTrades, isConnected } = useAppSelector(state => state.market);
  const { positions, isTrading } = useAppSelector(state => state.trading);
  
  const [form] = Form.useForm();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [orderType, setOrderType] = useState<OrderType>('market');
  const [orderSide, setOrderSide] = useState<OrderSide>('buy');
  const [loading, setLoading] = useState(false);

  // Available symbols and exchanges
  const symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/USD'];
  const exchanges = ['binance_us', 'coinbase', 'kraken', 'kucoin'];

  // Initialize chart
  useEffect(() => {
    if (chartContainerRef.current) {
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: '#1f1f1f' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: '#2B2B43' },
          horzLines: { color: '#2B2B43' },
        },
        width: chartContainerRef.current.clientWidth,
        height: 400,
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
        },
      });

      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      });

      chartRef.current = chart;

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };

      window.addEventListener('resize', handleResize);
      
      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
      };
    }
  }, []);

  // Load historical data
  useEffect(() => {
    const loadChartData = async () => {
      if (!currentSymbol || !currentExchange) return;
      
      setLoading(true);
      try {
        const ohlcvData = await apiService.getOHLCVData(
          currentSymbol,
          currentExchange,
          '1h',
          new Date(Date.now() - 24 * 60 * 60 * 1000), // Last 24 hours
          new Date(),
          100
        );

        const formattedData: ChartData[] = ohlcvData.map((item: OHLCV) => ({
          time: Math.floor(new Date(item.timestamp).getTime() / 1000),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume
        }));

        setChartData(formattedData);

        // Update chart
        if (chartRef.current) {
          const candlestickSeries = chartRef.current.addCandlestickSeries();
          const volumeSeries = chartRef.current.addHistogramSeries();
          
          candlestickSeries.setData(formattedData);
          volumeSeries.setData(formattedData.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open ? '#26a69a' : '#ef5350'
          })));
        }
      } catch (error) {
        console.error('Failed to load chart data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadChartData();
  }, [currentSymbol, currentExchange]);

  // Subscribe to real-time updates
  useEffect(() => {
    if (currentSymbol && currentExchange) {
      websocketService.subscribeToOrderBook(currentSymbol, currentExchange);
      websocketService.subscribeToTrades(currentSymbol, currentExchange);
    }

    return () => {
      if (currentSymbol && currentExchange) {
        websocketService.unsubscribeFromOrderBook(currentSymbol, currentExchange);
        websocketService.unsubscribeFromTrades(currentSymbol, currentExchange);
      }
    };
  }, [currentSymbol, currentExchange]);

  const handleSymbolChange = (symbol: string) => {
    dispatch(setCurrentSymbol(symbol));
  };

  const handleExchangeChange = (exchange: string) => {
    dispatch(setCurrentExchange(exchange));
  };

  const handlePlaceOrder = async (values: any) => {
    try {
      await dispatch(placePaperOrder({
        symbol: currentSymbol!,
        exchange: currentExchange!,
        side: orderSide,
        type: orderType,
        quantity: parseFloat(values.quantity),
        price: orderType === 'limit' ? parseFloat(values.price) : undefined,
      })).unwrap();
      
      form.resetFields();
    } catch (error) {
      console.error('Failed to place order:', error);
    }
  };

  // Order book columns
  const orderBookColumns = [
    {
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number, record: any) => (
        <Text className={record.side === 'buy' ? 'profit-color' : 'loss-color'}>
          ${price.toFixed(2)}
        </Text>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (quantity: number) => quantity.toFixed(4),
    },
    {
      title: 'Total',
      dataIndex: 'total',
      key: 'total',
      render: (total: number) => `$${total.toFixed(2)}`,
    },
  ];

  // Recent trades columns
  const tradesColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleTimeString(),
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
      title: 'Price',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `$${price.toFixed(2)}`,
    },
    {
      title: 'Size',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (quantity: number) => quantity.toFixed(4),
    },
  ];

  const currentPrice = orderBook?.bids?.[0]?.price || 0;
  const priceChange = chartData.length > 1 ? 
    ((chartData[chartData.length - 1].close - chartData[chartData.length - 2].close) / chartData[chartData.length - 2].close) * 100 : 0;

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>Trading Dashboard</Title>
          </Col>
          <Col>
            <Select
              style={{ width: 150 }}
              value={currentSymbol}
              onChange={handleSymbolChange}
              placeholder="Select Symbol"
            >
              {symbols.map(symbol => (
                <Option key={symbol} value={symbol}>{symbol}</Option>
              ))}
            </Select>
          </Col>
          <Col>
            <Select
              style={{ width: 150 }}
              value={currentExchange}
              onChange={handleExchangeChange}
              placeholder="Select Exchange"
            >
              {exchanges.map(exchange => (
                <Option key={exchange} value={exchange}>{exchange}</Option>
              ))}
            </Select>
          </Col>
          <Col>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={() => window.location.reload()}
            >
              Refresh
            </Button>
          </Col>
          <Col flex={1}>
            <div style={{ textAlign: 'right' }}>
              {!isConnected && (
                <Alert 
                  message="WebSocket Disconnected" 
                  type="warning" 
                  showIcon 
                  style={{ display: 'inline-block' }}
                />
              )}
            </div>
          </Col>
        </Row>
      </div>

      {/* Price Summary */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Current Price"
              value={currentPrice}
              precision={2}
              prefix={<DollarOutlined />}
              valueStyle={{ color: priceChange >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="24h Change"
              value={priceChange}
              precision={2}
              suffix="%"
              prefix={priceChange >= 0 ? <CaretUpOutlined /> : <CaretDownOutlined />}
              valueStyle={{ color: priceChange >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="24h Volume"
              value={chartData.reduce((sum, d) => sum + d.volume, 0)}
              precision={0}
              prefix={<RiseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Spread"
              value={orderBook ? ((orderBook.asks[0]?.price || 0) - (orderBook.bids[0]?.price || 0)) : 0}
              precision={2}
              prefix={<DollarOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        {/* Chart */}
        <Col span={16}>
          <Card title="Price Chart" loading={loading}>
            <div ref={chartContainerRef} className="chart-container" />
          </Card>
        </Col>

        {/* Order Form */}
        <Col span={8}>
          <Card title="Place Order">
            <Form form={form} layout="vertical" onFinish={handlePlaceOrder}>
              <Form.Item>
                <Space>
                  <Button 
                    type={orderSide === 'buy' ? 'primary' : 'default'}
                    onClick={() => setOrderSide('buy')}
                    style={{ background: orderSide === 'buy' ? '#52c41a' : undefined }}
                  >
                    Buy
                  </Button>
                  <Button 
                    type={orderSide === 'sell' ? 'primary' : 'default'}
                    onClick={() => setOrderSide('sell')}
                    style={{ background: orderSide === 'sell' ? '#ff4d4f' : undefined }}
                  >
                    Sell
                  </Button>
                </Space>
              </Form.Item>

              <Form.Item>
                <Select value={orderType} onChange={setOrderType}>
                  <Option value="market">Market</Option>
                  <Option value="limit">Limit</Option>
                </Select>
              </Form.Item>

              {orderType === 'limit' && (
                <Form.Item
                  label="Price"
                  name="price"
                  rules={[{ required: true, message: 'Please enter price' }]}
                >
                  <Input 
                    type="number" 
                    step="0.01"
                    placeholder="0.00"
                    addonBefore="$"
                  />
                </Form.Item>
              )}

              <Form.Item
                label="Quantity"
                name="quantity"
                rules={[{ required: true, message: 'Please enter quantity' }]}
              >
                <Input 
                  type="number" 
                  step="0.0001"
                  placeholder="0.0000"
                />
              </Form.Item>

              <Form.Item>
                <Button 
                  type="primary" 
                  htmlType="submit"
                  loading={isTrading}
                  block
                  style={{ 
                    background: orderSide === 'buy' ? '#52c41a' : '#ff4d4f',
                    borderColor: orderSide === 'buy' ? '#52c41a' : '#ff4d4f'
                  }}
                >
                  {orderSide === 'buy' ? 'Buy' : 'Sell'} {currentSymbol}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        {/* Order Book */}
        <Col span={12}>
          <Card title="Order Book" className="orderbook-container">
            <div style={{ height: 300, overflowY: 'auto' }}>
              <Table
                dataSource={orderBook?.asks?.slice(0, 10).reverse() || []}
                columns={orderBookColumns}
                pagination={false}
                size="small"
                rowClassName="ask-row"
              />
              <div style={{ textAlign: 'center', padding: '8px 0', background: '#f5f5f5' }}>
                <Text strong>${currentPrice.toFixed(2)}</Text>
              </div>
              <Table
                dataSource={orderBook?.bids?.slice(0, 10) || []}
                columns={orderBookColumns}
                pagination={false}
                size="small"
                rowClassName="bid-row"
              />
            </div>
          </Card>
        </Col>

        {/* Recent Trades */}
        <Col span={12}>
          <Card title="Recent Trades">
            <Table
              dataSource={(recentTrades && currentSymbol ? recentTrades[`${currentSymbol}-${currentExchange}`] || [] : []).slice(0, 20)}
              columns={tradesColumns}
              pagination={false}
              size="small"
              scroll={{ y: 300 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TradingDashboard;