import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Typography, 
  Card, 
  Input, 
  Select, 
  Button, 
  Space, 
  Tag, 
  Row, 
  Col,
  Switch,
  Tooltip,
  Badge,
  Alert,
  AutoComplete,
  DatePicker,
  Slider,
  Popover
} from 'antd';
import {
  SearchOutlined,
  ClearOutlined,
  DownloadOutlined,
  ReloadOutlined,
  FilterOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  BugOutlined
} from '@ant-design/icons';
import { useAppSelector } from '../../store/hooks';
import { apiService } from '../../services/api';
import { useThrottledWebSocketEvent } from '../../services/websocket';
import dayjs from 'dayjs';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Search } = Input;
const { RangePicker } = DatePicker;

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  service: string;
  message: string;
  filename?: string;
  line_number?: number;
  function_name?: string;
  traceback?: string;
  metadata?: Record<string, any>;
}

interface LogFilter {
  level: string[];
  service: string[];
  search: string;
  timeRange: [dayjs.Dayjs, dayjs.Dayjs] | null;
  maxLines: number;
}

const LogsDashboard: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRealTime, setIsRealTime] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [errorCount, setErrorCount] = useState(0);
  const [warningCount, setWarningCount] = useState(0);
  
  const [filter, setFilter] = useState<LogFilter>({
    level: [],
    service: [],
    search: '',
    timeRange: null,
    maxLines: 1000
  });

  const logContainerRef = useRef<HTMLDivElement>(null);
  const [services, setServices] = useState<string[]>([]);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  // Load initial logs and services
  useEffect(() => {
    loadLogs();
    loadServices();
  }, []);

  // Real-time log updates via WebSocket
  useThrottledWebSocketEvent('log_entry', (newLog: LogEntry) => {
    if (!isPaused && isRealTime) {
      setLogs(prevLogs => {
        const updated = [newLog, ...prevLogs].slice(0, filter.maxLines);
        return updated;
      });
      
      // Update counters
      if (newLog.level === 'ERROR' || newLog.level === 'CRITICAL') {
        setErrorCount(prev => prev + 1);
      } else if (newLog.level === 'WARNING') {
        setWarningCount(prev => prev + 1);
      }
    }
  }, 500);

  const loadLogs = async () => {
    setIsLoading(true);
    try {
      const params: any = {
        limit: filter.maxLines,
      };
      
      if (filter.level.length > 0) {
        params.levels = filter.level.join(',');
      }
      
      if (filter.service.length > 0) {
        params.services = filter.service.join(',');
      }
      
      if (filter.search) {
        params.search = filter.search;
      }
      
      if (filter.timeRange) {
        params.start_time = filter.timeRange[0].toISOString();
        params.end_time = filter.timeRange[1].toISOString();
      }

      const response = await apiService.getLogs(params);
      setLogs(response || []);
      
      // Count errors and warnings
      const errors = response?.filter((log: LogEntry) => 
        log.level === 'ERROR' || log.level === 'CRITICAL'
      ).length || 0;
      const warnings = response?.filter((log: LogEntry) => 
        log.level === 'WARNING'
      ).length || 0;
      
      setErrorCount(errors);
      setWarningCount(warnings);
      
    } catch (error) {
      console.error('Failed to load logs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadServices = async () => {
    try {
      const response = await apiService.getSystemServices();
      const serviceNames = response.map((service: any) => service.name);
      setServices(serviceNames);
    } catch (error) {
      console.error('Failed to load services:', error);
    }
  };

  // Filter logs based on current filter settings
  useEffect(() => {
    let filtered = [...logs];

    // Level filter
    if (filter.level.length > 0) {
      filtered = filtered.filter(log => filter.level.includes(log.level));
    }

    // Service filter
    if (filter.service.length > 0) {
      filtered = filtered.filter(log => filter.service.includes(log.service));
    }

    // Search filter
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(searchLower) ||
        log.service.toLowerCase().includes(searchLower) ||
        (log.filename && log.filename.toLowerCase().includes(searchLower))
      );
    }

    // Time range filter
    if (filter.timeRange) {
      filtered = filtered.filter(log => {
        const logTime = dayjs(log.timestamp);
        return logTime.isAfter(filter.timeRange![0]) && logTime.isBefore(filter.timeRange![1]);
      });
    }

    setFilteredLogs(filtered);
  }, [logs, filter]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = 0; // Scroll to top since we're adding new logs at the beginning
    }
  }, [filteredLogs, autoScroll]);

  const handleSearch = (value: string) => {
    setFilter(prev => ({ ...prev, search: value }));
    if (value && !searchHistory.includes(value)) {
      setSearchHistory(prev => [value, ...prev.slice(0, 9)]); // Keep last 10 searches
    }
  };

  const handleFilterChange = (key: keyof LogFilter, value: any) => {
    setFilter(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilter({
      level: [],
      service: [],
      search: '',
      timeRange: null,
      maxLines: 1000
    });
  };

  const exportLogs = () => {
    const logText = filteredLogs.map(log => {
      const timestamp = showTimestamps ? `[${log.timestamp}] ` : '';
      const level = `[${log.level}] `;
      const service = `[${log.service}] `;
      return `${timestamp}${level}${service}${log.message}`;
    }).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tickerml-logs-${dayjs().format('YYYY-MM-DD-HH-mm-ss')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'DEBUG': return '#8c8c8c';
      case 'INFO': return '#1890ff';
      case 'WARNING': return '#faad14';
      case 'ERROR': return '#ff4d4f';
      case 'CRITICAL': return '#a8071a';
      default: return '#000000';
    }
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'ERROR':
      case 'CRITICAL':
        return <CloseCircleOutlined />;
      case 'WARNING':
        return <ExclamationCircleOutlined />;
      case 'DEBUG':
        return <BugOutlined />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  const renderLogLine = (log: LogEntry, index: number) => (
    <div
      key={log.id || index}
      className={`log-line log-level-${log.level.toLowerCase()}`}
      style={{
        padding: '4px 8px',
        borderBottom: '1px solid #f0f0f0',
        fontFamily: 'Monaco, Consolas, "Courier New", monospace',
        fontSize: '12px',
        lineHeight: '1.4',
        backgroundColor: index % 2 === 0 ? '#fafafa' : '#ffffff'
      }}
    >
      <Space size="small">
        {showTimestamps && (
          <Text style={{ color: '#8c8c8c', minWidth: '140px' }}>
            {dayjs(log.timestamp).format('MM-DD HH:mm:ss.SSS')}
          </Text>
        )}
        <Tag 
          color={getLevelColor(log.level)} 
          icon={getLevelIcon(log.level)}
          style={{ minWidth: '70px', textAlign: 'center' }}
        >
          {log.level}
        </Tag>
        <Tag color="blue" style={{ minWidth: '100px', textAlign: 'center' }}>
          {log.service}
        </Tag>
        <Text style={{ flex: 1 }}>{log.message}</Text>
        {log.filename && (
          <Text style={{ color: '#8c8c8c', fontSize: '11px' }}>
            {log.filename}:{log.line_number}
          </Text>
        )}
      </Space>
      {log.traceback && (
        <div style={{ marginTop: '4px', marginLeft: '160px' }}>
          <Text code style={{ fontSize: '11px', whiteSpace: 'pre-wrap' }}>
            {log.traceback}
          </Text>
        </div>
      )}
    </div>
  );

  const filterPopoverContent = (
    <div style={{ width: 300 }}>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <div>
          <Text strong>Log Levels:</Text>
          <Select
            mode="multiple"
            style={{ width: '100%', marginTop: 4 }}
            placeholder="Select levels"
            value={filter.level}
            onChange={(value) => handleFilterChange('level', value)}
          >
            <Option value="DEBUG">DEBUG</Option>
            <Option value="INFO">INFO</Option>
            <Option value="WARNING">WARNING</Option>
            <Option value="ERROR">ERROR</Option>
            <Option value="CRITICAL">CRITICAL</Option>
          </Select>
        </div>

        <div>
          <Text strong>Services:</Text>
          <Select
            mode="multiple"
            style={{ width: '100%', marginTop: 4 }}
            placeholder="Select services"
            value={filter.service}
            onChange={(value) => handleFilterChange('service', value)}
          >
            {services.map(service => (
              <Option key={service} value={service}>{service}</Option>
            ))}
          </Select>
        </div>

        <div>
          <Text strong>Time Range:</Text>
          <RangePicker
            style={{ width: '100%', marginTop: 4 }}
            showTime
            value={filter.timeRange}
            onChange={(value) => handleFilterChange('timeRange', value)}
          />
        </div>

        <div>
          <Text strong>Max Lines: {filter.maxLines}</Text>
          <Slider
            min={100}
            max={5000}
            step={100}
            value={filter.maxLines}
            onChange={(value) => handleFilterChange('maxLines', value)}
            style={{ marginTop: 8 }}
          />
        </div>
      </Space>
    </div>
  );

  return (
    <div>
      <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>System Logs</Title>
        </Col>
        <Col flex={1}>
          {(errorCount > 0 || warningCount > 0) && (
            <Space>
              {errorCount > 0 && (
                <Badge count={errorCount} style={{ backgroundColor: '#ff4d4f' }}>
                  <Tag color="red">Errors</Tag>
                </Badge>
              )}
              {warningCount > 0 && (
                <Badge count={warningCount} style={{ backgroundColor: '#faad14' }}>
                  <Tag color="orange">Warnings</Tag>
                </Badge>
              )}
            </Space>
          )}
        </Col>
      </Row>

      {/* Controls */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={8}>
            <AutoComplete
              options={searchHistory.map(item => ({ value: item }))}
              style={{ width: '100%' }}
            >
              <Search
                placeholder="Search logs..."
                value={filter.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                onSearch={handleSearch}
                enterButton={<SearchOutlined />}
              />
            </AutoComplete>
          </Col>
          
          <Col>
            <Space>
              <Popover content={filterPopoverContent} title="Advanced Filters" trigger="click">
                <Button icon={<FilterOutlined />}>
                  Filters
                  {(filter.level.length > 0 || filter.service.length > 0 || filter.timeRange) && (
                    <Badge dot style={{ marginLeft: 4 }} />
                  )}
                </Button>
              </Popover>
              
              <Button icon={<ClearOutlined />} onClick={clearFilters}>
                Clear
              </Button>
              
              <Button icon={<ReloadOutlined />} onClick={loadLogs} loading={isLoading}>
                Refresh
              </Button>
              
              <Button icon={<DownloadOutlined />} onClick={exportLogs}>
                Export
              </Button>
            </Space>
          </Col>

          <Col>
            <Space>
              <Tooltip title="Real-time updates">
                <Switch
                  checked={isRealTime}
                  onChange={setIsRealTime}
                  checkedChildren="Real-time"
                  unCheckedChildren="Static"
                />
              </Tooltip>
              
              <Tooltip title="Pause/Resume">
                <Button
                  icon={isPaused ? <PlayCircleOutlined /> : <PauseCircleOutlined />}
                  onClick={() => setIsPaused(!isPaused)}
                  type={isPaused ? "primary" : "default"}
                />
              </Tooltip>
              
              <Tooltip title="Auto-scroll">
                <Switch
                  checked={autoScroll}
                  onChange={setAutoScroll}
                  checkedChildren="Auto"
                  unCheckedChildren="Manual"
                  size="small"
                />
              </Tooltip>
              
              <Tooltip title="Show timestamps">
                <Switch
                  checked={showTimestamps}
                  onChange={setShowTimestamps}
                  checkedChildren="Time"
                  unCheckedChildren="No Time"
                  size="small"
                />
              </Tooltip>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Log Viewer */}
      <Card 
        title={`Logs (${filteredLogs.length} entries)`}
        loading={isLoading}
        bodyStyle={{ padding: 0 }}
      >
        <div
          ref={logContainerRef}
          style={{
            height: '600px',
            overflowY: 'auto',
            backgroundColor: '#fafafa',
            border: '1px solid #d9d9d9'
          }}
        >
          {filteredLogs.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <Text type="secondary">No logs found matching current filters</Text>
            </div>
          ) : (
            filteredLogs.map((log, index) => renderLogLine(log, index))
          )}
        </div>
      </Card>
    </div>
  );
};

export default LogsDashboard;