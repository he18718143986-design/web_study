// frontend/src/components/AggregatorPanel.tsx
import React from "react";
import { Card, List } from "antd";

interface Props {
  report: any;
}

const AggregatorPanel: React.FC<Props> = ({ report }) => {
  if (!report) return null;
  return (
    <Card title="仲裁结果">
      <h4>Confirmed</h4>
      <List dataSource={report.confirmed ?? []} renderItem={(i: any) => <List.Item>{i.point_text ?? i}</List.Item>} />
      <h4>Contradictions</h4>
      <List dataSource={report.contradictions ?? []} renderItem={(i: any) => <List.Item>{i.cluster_id ?? JSON.stringify(i)}</List.Item>} />
      <h4>Followups</h4>
      <List dataSource={report.followups ?? []} renderItem={(i: any) => <List.Item>{i.question ?? i}</List.Item>} />
      <div style={{ marginTop: 12 }}>Recommendation: {report.recommendation}</div>
    </Card>
  );
};

export default AggregatorPanel;
