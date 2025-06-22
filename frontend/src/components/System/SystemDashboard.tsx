import React, { useState, useEffect } from 'react';
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
  Switch,
  Modal,
  Form,
  Input,
  Space,
  Alert,
  Tooltip,
  Divider
} from 'antd';
import { 
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  StopOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  MonitorOutlined,
  AlertOutlined
} from '@ant-design/icons';
import { Line, Gauge } from '@ant-design/charts';
import { useAppSelector, useAppDispatch } from '../../store/hooks';
import { 
  fetchSystemStatus, 
  fetchSystemMetrics, 
  restartService, 
  stopService, 
  startService 
} from '../../store/slices/systemSlice';
import { SystemService, SystemMetrics, AlertRule } from '../../types';
import { apiService } from '../../services/api';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { confirm } = Modal;

const SystemDashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { 
    services, 
    metrics, 
    alerts, 
    isLoading 
  } = useAppSelector(state => state.system);
  
  const [metricsData, setMetricsData] = useState<any[]>([]);
  const [selectedService, setSelectedService] = useState<SystemService | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();

  // Load system data on mount
  useEffect(() => {
    dispatch(fetchSystemStatus());
    dispatch(fetchSystemMetrics());
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      dispatch(fetchSystemStatus());
      dispatch(fetchSystemMetrics());
    }, 30000);

    return () => clearInterval(interval);
  }, [dispatch]);

  // Format metrics data for charts
  useEffect(() => {
    if (metrics?.cpu_usage_history) {
      const chartData = metrics.cpu_usage_history.map((point: any, index: number) => ({
        time: dayjs().subtract(metrics.cpu_usage_history.length - index, 'minute').format('HH:mm'),
        cpu: point.cpu,
        memory: point.memory,
        disk: point.disk,
      }));
      setMetricsData(chartData);
    }
  }, [metrics]);

  const handleServiceAction = async (service: SystemService, action: 'start' | 'stop' | 'restart') => {
    const actionName = action.charAt(0).toUpperCase() + action.slice(1);
    
    confirm({
      title: `${actionName} ${service.name}?`,
      content: `Are you sure you want to ${action} the ${service.name} service?`,
      onOk: async () => {
        try {
          switch (action) {
            case 'start':
              await dispatch(startService(service.name)).unwrap();
              break;
            case 'stop':
              await dispatch(stopService(service.name)).unwrap();
              break;
            case 'restart':
              await dispatch(restartService(service.name)).unwrap();
              break;
          }
          dispatch(fetchSystemStatus());
        } catch (error) {
          console.error(`Failed to ${action} service:`, error);
        }
      },
    });
  };

  const handleConfigureService = (service: SystemService) => {
    setSelectedService(service);
    form.setFieldsValue({
      auto_restart: service.auto_restart,
      max_restarts: service.max_restarts,
      restart_delay: service.restart_delay,
    });
    setIsModalVisible(true);
  };

  const handleSaveConfiguration = async (values: any) => {
    if (!selectedService) return;
    
    try {
      await apiService.updateServiceConfig(selectedService.name, values);
      setIsModalVisible(false);
      dispatch(fetchSystemStatus());
    } catch (error) {
      console.error('Failed to update service configuration:', error);
    }
  };

  // Service status columns
  const serviceColumns = [
    {
      title: 'Service',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: SystemService) => (
        <Space>
          <Text strong>{name}</Text>
          <Tag color={
            record.status === 'running' ? 'green' : 
            record.status === 'stopped' ? 'red' : 
            record.status === 'starting' ? 'orange' : 'default'
          }>
            {record.status.toUpperCase()}
          </Tag>
        </Space>
      ),
    },
    {
      title: 'Health',
      dataIndex: 'health',
      key: 'health',
      render: (health: string) => {
        const icon = health === 'healthy' ? <CheckCircleOutlined /> :
                    health === 'degraded' ? <ExclamationCircleOutlined /> :
                    <CloseCircleOutlined />;
        const color = health === 'healthy' ? 'green' :
                     health === 'degraded' ? 'orange' : 'red';
        return <Tag icon={icon} color={color}>{health.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'CPU',
      dataIndex: 'cpu_usage',
      key: 'cpu_usage',
      render: (cpu: number) => (
        <Progress 
          percent={cpu} 
          size="small" 
          strokeColor={cpu > 80 ? '#ff4d4f' : cpu > 60 ? '#faad14' : '#52c41a'}
        />
      ),
    },
    {
      title: 'Memory',
      dataIndex: 'memory_usage',
      key: 'memory_usage',
      render: (memory: number) => (
        <Progress 
          percent={memory} 
          size="small" 
          strokeColor={memory > 80 ? '#ff4d4f' : memory > 60 ? '#faad14' : '#52c41a'}
        />
      ),
    },
    {
      title: 'Uptime',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => {
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        return `${hours}h ${minutes}m`;
      },
    },
    {
      title: 'Restarts',
      dataIndex: 'restart_count',
      key: 'restart_count',
      render: (count: number) => (
        <Text className={count > 5 ? 'loss-color' : 'neutral-color'}>
          {count}
        </Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: SystemService) => (
        <Space>
          {record.status === 'stopped' ? (
            <Button 
              icon={<PlayCircleOutlined />} 
              size="small"
              type="primary"
              onClick={() => handleServiceAction(record, 'start')}
            >
              Start
            </Button>
          ) : (
            <Button 
              icon={<PauseCircleOutlined />} 
              size="small"
              onClick={() => handleServiceAction(record, 'stop')}
            >
              Stop
            </Button>
          )}
          <Button 
            icon={<ReloadOutlined />} 
            size="small"
            onClick={() => handleServiceAction(record, 'restart')}
            disabled={record.status === 'stopped'}
          >
            Restart
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            size="small"
            onClick={() => handleConfigureService(record)}
          >
            Config
          </Button>
        </Space>
      ),
    },
  ];

  // Alert columns
  const alertColumns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => dayjs(timestamp).format('MM-DD HH:mm:ss'),
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={
          severity === 'critical' ? 'red' :
          severity === 'warning' ? 'orange' :
          severity === 'info' ? 'blue' : 'default'
        }>
          {severity.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Service',
      dataIndex: 'service',
      key: 'service',
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'resolved' ? 'green' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
  ];

  // Metrics chart config
  const metricsConfig = {
    data: metricsData,
    xField: 'time',
    yField: 'cpu',
    seriesField: 'type',
    height: 250,
    smooth: true,
    color: ['#1890ff', '#52c41a', '#faad14'],
    point: {
      size: 2,
      shape: 'circle',
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: 'CPU Usage',
        value: `${datum.cpu?.toFixed(1)}%`,
      }),
    },
  };

  // System health gauges
  const cpuGaugeConfig = {
    percent: (metrics?.cpu_usage || 0) / 100,
    range: { color: '#1890ff' },
    indicator: {
      pointer: { style: { stroke: '#D0D0D0' } },
      pin: { style: { stroke: '#D0D0D0' } },
    },
    statistic: {
      content: {
        style: { fontSize: '16px', lineHeight: '16px' },
        formatter: () => `${(metrics?.cpu_usage || 0).toFixed(1)}%`,
      },
    },
  };

  const memoryGaugeConfig = {
    ...cpuGaugeConfig,
    percent: (metrics?.memory_usage || 0) / 100,
    range: { color: '#52c41a' },
    statistic: {
      content: {
        style: { fontSize: '16px', lineHeight: '16px' },
        formatter: () => `${(metrics?.memory_usage || 0).toFixed(1)}%`,
      },
    },
  };

  const diskGaugeConfig = {
    ...cpuGaugeConfig,
    percent: (metrics?.disk_usage || 0) / 100,
    range: { color: '#faad14' },
    statistic: {
      content: {
        style: { fontSize: '16px', lineHeight: '16px' },
        formatter: () => `${(metrics?.disk_usage || 0).toFixed(1)}%`,
      },
    },
  };

  const criticalAlerts = alerts?.filter(alert => alert.severity === 'critical' && alert.status === 'active') || [];

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>System Control Panel</Title>
          </Col>
          <Col>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={() => {
                dispatch(fetchSystemStatus());
                dispatch(fetchSystemMetrics());
              }}
            >
              Refresh
            </Button>
          </Col>
        </Row>
      </div>

      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <Alert
          message={`${criticalAlerts.length} Critical System Alert${criticalAlerts.length > 1 ? 's' : ''}`}
          description={criticalAlerts.map(alert => alert.message).join(', ')}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* System Overview */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card title="CPU Usage">
            <Gauge {...cpuGaugeConfig} height={120} />
          </Card>
        </Col>
        <Col span={6}>
          <Card title="Memory Usage">
            <Gauge {...memoryGaugeConfig} height={120} />
          </Card>
        </Col>
        <Col span={6}>
          <Card title="Disk Usage">
            <Gauge {...diskGaugeConfig} height={120} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Services"
              value={services?.filter(s => s.status === 'running').length || 0}
              suffix={`/ ${services?.length || 0}`}
              prefix={<CloudServerOutlined />}
              valueStyle={{ 
                color: services?.every(s => s.status === 'running') ? '#3f8600' : '#cf1322'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* System Metrics Chart */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={24}>
          <Card title="System Metrics (Last Hour)" loading={isLoading}>
            <Line {...metricsConfig} />
          </Card>
        </Col>
      </Row>

      {/* Service Management */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={24}>
          <Card title="Service Management" loading={isLoading}>
            <Table
              dataSource={services || []}
              columns={serviceColumns}
              pagination={false}
              rowKey="name"
              scroll={{ x: 800 }}
            />
          </Card>
        </Col>
      </Row>

      {/* System Alerts */}
      <Row gutter={16}>
        <Col span={24}>
          <Card title="System Alerts" loading={isLoading}>
            <Table
              dataSource={alerts?.slice(0, 20) || []}
              columns={alertColumns}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `Total ${total} alerts`,
              }}
              rowKey="id"
              scroll={{ x: 600 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Service Configuration Modal */}
      <Modal
        title={`Configure ${selectedService?.name}`}
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleSaveConfiguration}>
          <Form.Item
            label="Auto Restart"
            name="auto_restart"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            label="Max Restarts"
            name="max_restarts"
            rules={[{ required: true, message: 'Please enter max restarts' }]}
          >
            <Input type="number" min={0} max={10} />
          </Form.Item>

          <Form.Item
            label="Restart Delay (seconds)"
            name="restart_delay"
            rules={[{ required: true, message: 'Please enter restart delay' }]}
          >
            <Input type="number" min={1} max={300} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Save Configuration
              </Button>
              <Button onClick={() => setIsModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default SystemDashboard;