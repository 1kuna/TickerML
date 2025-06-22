import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const ConfigurationDashboard: React.FC = () => {
  return (
    <div>
      <Title level={2}>Configuration</Title>
      <Card>
        <p>Configuration dashboard will include:</p>
        <ul>
          <li>System configuration editor</li>
          <li>Risk parameters</li>
          <li>Exchange settings</li>
          <li>Model parameters</li>
        </ul>
      </Card>
    </div>
  );
};

export default ConfigurationDashboard;