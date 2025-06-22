import React, { useState } from 'react';
import { Form, Input, Button, Card, message, Alert } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { loginAsync, clearError } from '../../store/slices/authSlice';
import { LoginCredentials } from '../../types';

const LoginPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isLoading, error } = useAppSelector((state) => state.auth);
  const [form] = Form.useForm();

  const handleSubmit = async (values: LoginCredentials) => {
    try {
      await dispatch(loginAsync(values)).unwrap();
      message.success('Login successful!');
    } catch (error) {
      // Error is handled by Redux
    }
  };

  const handleFormChange = () => {
    if (error) {
      dispatch(clearError());
    }
  };

  return (
    <div 
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #001529 0%, #002766 100%)',
        padding: '20px'
      }}
    >
      <Card 
        style={{ 
          width: 400, 
          boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
          borderRadius: '8px'
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <h1 style={{ color: '#1890ff', margin: 0, fontSize: '28px' }}>
            TickerML
          </h1>
          <p style={{ color: '#666', marginTop: 8 }}>
            Professional Trading Dashboard
          </p>
        </div>

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          onChange={handleFormChange}
        >
          <Form.Item
            name="username"
            label="Username"
            rules={[{ required: true, message: 'Please enter your username' }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="Username"
              size="large"
            />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password"
            rules={[{ required: true, message: 'Please enter your password' }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
              size="large"
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Button
              type="primary"
              htmlType="submit"
              loading={isLoading}
              size="large"
              block
            >
              Sign In
            </Button>
          </Form.Item>
        </Form>

        <div style={{ marginTop: 24, padding: '16px', background: '#f5f5f5', borderRadius: '4px' }}>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>Demo Accounts:</h4>
          <div style={{ fontSize: '12px', color: '#666' }}>
            <div><strong>Admin:</strong> admin / admin123</div>
            <div><strong>Trader:</strong> trader / trader123</div>
            <div><strong>Viewer:</strong> viewer / viewer123</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;