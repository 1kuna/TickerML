import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const DataCollectionDashboard: React.FC = () => {
  return (
    <div>
      <Title level={2}>Data Collection</Title>
      <Card>
        <p>Data collection dashboard will include:</p>
        <ul>
          <li>Exchange connection status</li>
          <li>Data quality metrics</li>
          <li>Collection configuration</li>
          <li>Stream monitoring</li>
        </ul>
      </Card>
    </div>
  );
};

export default DataCollectionDashboard;