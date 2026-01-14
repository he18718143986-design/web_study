// frontend/src/components/QuestionPanel.tsx
import React, { useState } from "react";
import { Button, Input, Space } from "antd";

interface Props {
  onSubmit: (question: string, models?: string[]) => void;
  loading?: boolean;
}

const QuestionPanel: React.FC<Props> = ({ onSubmit, loading }) => {
  const [q, setQ] = useState("");

  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <Input.TextArea rows={4} value={q} onChange={(e) => setQ(e.target.value)} placeholder="请输入问题" />
      <Button type="primary" onClick={() => onSubmit(q, ["mock","hf"])} loading={loading}>
        提交
      </Button>
    </Space>
  );
};

export default QuestionPanel;
