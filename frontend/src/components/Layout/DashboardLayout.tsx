import React from 'react';
import { Layout, Menu, Button, Avatar, Dropdown, Badge, Space, Switch, Tooltip } from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  RobotOutlined,
  SettingOutlined,
  LogoutOutlined,
  UserOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PieChartOutlined,
  MonitorOutlined,
  FileTextOutlined,
  ControlOutlined,
  BulbOutlined,
  BulbFilled,
} from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '@/store';
import { logoutAsync } from '@/store/slices/authSlice';
import { toggleSidebar, setActiveMenuItem, toggleTheme } from '@/store/slices/uiSlice';

const { Header, Sider, Content } = Layout;

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  
  const { user } = useAppSelector((state) => state.auth);
  const { sidebarCollapsed, notifications, theme } = useAppSelector((state) => state.ui);

  const menuItems: MenuProps['items'] = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Overview',
    },
    {
      key: '/trading',
      icon: <LineChartOutlined />,
      label: 'Trading',
    },
    {
      key: '/portfolio',
      icon: <PieChartOutlined />,
      label: 'Portfolio',
    },
    {
      key: '/data',
      icon: <DatabaseOutlined />,
      label: 'Data Collection',
    },
    {
      key: '/models',
      icon: <RobotOutlined />,
      label: 'Models',
    },
    {
      key: '/system',
      icon: <MonitorOutlined />,
      label: 'System',
    },
    {
      key: '/config',
      icon: <ControlOutlined />,
      label: 'Configuration',
    },
    {
      key: '/logs',
      icon: <FileTextOutlined />,
      label: 'Logs',
    },
  ];

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
    dispatch(setActiveMenuItem(key));
  };

  const handleLogout = () => {
    dispatch(logoutAsync());
  };

  const handleThemeToggle = () => {
    dispatch(toggleTheme());
  };

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      onClick: handleLogout,
    },
  ];

  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={sidebarCollapsed}
        theme="dark"
        width={256}
        style={{
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100,
        }}
      >
        <div 
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #303030',
          }}
        >
          <div style={{ color: '#1890ff', fontSize: sidebarCollapsed ? '16px' : '20px', fontWeight: 'bold' }}>
            {sidebarCollapsed ? 'TML' : 'TickerML'}
          </div>
        </div>
        
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ marginTop: 8 }}
        />
      </Sider>

      <Layout style={{ marginLeft: sidebarCollapsed ? 80 : 256, transition: 'margin-left 0.2s' }}>
        <Header 
          style={{
            padding: '0 24px',
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'fixed',
            top: 0,
            right: 0,
            left: sidebarCollapsed ? 80 : 256,
            zIndex: 99,
            transition: 'left 0.2s',
          }}
        >
          <Button
            type="text"
            icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => dispatch(toggleSidebar())}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />

          <Space size="middle">
            <Tooltip title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
              <Button
                type="text"
                icon={theme === 'light' ? <BulbOutlined /> : <BulbFilled />}
                onClick={handleThemeToggle}
                style={{ fontSize: '16px' }}
              />
            </Tooltip>

            <Badge count={unreadNotifications} size="small">
              <Button
                type="text"
                icon={<BellOutlined />}
                style={{ fontSize: '16px' }}
              />
            </Badge>

            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <Space style={{ cursor: 'pointer' }}>
                <Avatar icon={<UserOutlined />} />
                <span>{user?.username}</span>
              </Space>
            </Dropdown>
          </Space>
        </Header>

        <Content
          style={{
            margin: '88px 24px 24px 24px',
            padding: 24,
            minHeight: 'calc(100vh - 112px)',
            background: '#f5f5f5',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default DashboardLayout;