import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const ModelsDashboard: React.FC = () => {
  return (
    <div>
      <Title level={2}>Models Dashboard</Title>
      <Card>
        <p>Models dashboard will include:</p>
        <ul>
          <li>Model training interface</li>
          <li>Performance metrics</li>
          <li>Model deployment</li>
          <li>Training progress</li>
        </ul>
      </Card>
    </div>
  );
};

export default ModelsDashboard;