// frontend/src/App.tsx
import { useState } from "react";
import { Layout, Typography, Divider, Spin, message } from "antd";
import { runQuery } from "./api/llm";
import QuestionPanel from "./components/QuestionPanel";
import ResultsGrid from "./components/ResultsGrid";
import AggregatorPanel from "./components/AggregatorPanel";

const { Header, Content } = Layout;
const { Title } = Typography;

export default function App() {
  const [loading, setLoading] = useState(false);
  const [sessionResult, setSessionResult] = useState<any | null>(null);

  const handleSubmit = async (question: string, models: string[] = ["mock"]) => {
    if (!question.trim()) {
      message.warning("请输入问题");
      return;
    }
    setLoading(true);
    try {
      const data = await runQuery({
        question,
        models,
        max_rounds: 1,
      });
      // data 结构按后端 openapi：包含 session_id, rounds[], final_report 等
      // 取首轮 aggregator_report（或 adjust 根据你的后端）
      const firstRound = data.rounds?.[0];
      const aggregatorReport = firstRound?.report ?? data.aggregator_report ?? data.final_report;
      setSessionResult({
        question,
        responses: firstRound?.multi?.responses ?? data.initial_responses ?? [],
        aggregator: aggregatorReport ?? null,
        session_id: data.session_id ?? null,
      });
    } catch (err: any) {
      console.error(err);
      if (err.code === "ECONNABORTED") {
        message.error("请求超时，LLM API 调用可能需要更长时间。请检查后端服务是否正常运行，或稍后重试。");
      } else if (err.response) {
        message.error(`后端错误: ${err.response.status} - ${err.response.data?.detail || err.message}`);
      } else if (err.request) {
        message.error("无法连接到后端服务，请确保后端服务正在运行（http://localhost:8000）");
      } else {
        message.error(`请求失败: ${err.message || "未知错误"}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ background: "#001529" }}>
        <Title style={{ color: "#fff", margin: 0 }} level={3}>
          Multi-LLM Arbiter (MVP)
        </Title>
      </Header>

      <Content style={{ padding: 24 }}>
        <QuestionPanel onSubmit={handleSubmit} loading={loading} />

        <Divider />

        {loading && <Spin />}

        {sessionResult && (
          <>
            <ResultsGrid results={sessionResult.responses} />
            <Divider />
            <AggregatorPanel report={sessionResult.aggregator} />
          </>
        )}
      </Content>
    </Layout>
  );
}
