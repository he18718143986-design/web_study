// frontend/src/components/ResultsGrid.tsx
import React from "react";
import { Card, List } from "antd";

interface SummaryPoint { id?: string; text: string; confidence?: string }
interface ResponseItem {
  model_id: string;
  parsed?: { summary_points?: SummaryPoint[] };
  raw?: string;
  parse_error?: string;
  error?: string;
  meta?: any;
}

interface Props {
  results: ResponseItem[];
}

const ResultsGrid: React.FC<Props> = ({ results }) => {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      {results.map((r) => (
        <Card key={r.model_id} title={`${r.model_id}`}>
          {r.error ? (
            <div style={{ color: "red", padding: "8px", background: "#fff1f0", border: "1px solid #ffccc7", borderRadius: "4px" }}>
              <strong>错误:</strong> {r.error}
              {r.meta?.timeout && <div style={{ marginTop: 4, fontSize: "12px", color: "#999" }}>⏱️ 请求超时</div>}
            </div>
          ) : r.parse_error ? (
            <div style={{ color: "orange", padding: "8px", background: "#fffbe6", border: "1px solid #ffe58f", borderRadius: "4px" }}>
              <strong>解析错误:</strong> {r.parse_error}
              {r.raw && <div style={{ marginTop: 8, fontSize: "12px", color: "#888" }}>原始响应: {r.raw.substring(0, 200)}...</div>}
            </div>
          ) : r.parsed?.summary_points && r.parsed.summary_points.length > 0 ? (
            <>
              <List
                size="small"
                dataSource={r.parsed.summary_points}
                renderItem={(item) => <List.Item>{item.text} ({item.confidence})</List.Item>}
              />
              {r.raw && <div style={{ marginTop: 8, fontSize: "12px", color: "#888" }}>原始响应: {r.raw.substring(0, 200)}...</div>}
            </>
          ) : (
            <div style={{ color: "#999", fontStyle: "italic" }}>暂无响应数据</div>
          )}
          {r.meta && (
            <div style={{ marginTop: 8, fontSize: "11px", color: "#bbb" }}>
              延迟: {r.meta.latency_s}s | 时间戳: {r.meta.timestamp ? new Date(r.meta.timestamp).toLocaleTimeString() : "N/A"}
            </div>
          )}
        </Card>
      ))}
    </div>
  );
};

export default ResultsGrid;
