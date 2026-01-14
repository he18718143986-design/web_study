// frontend/src/api/llm.ts
import axios from 'axios';

const api = axios.create({
  baseURL: '/', // 使用 vite proxy -> 前端可保持 '/'，proxy 会转发到 http://localhost:8000
  timeout: 120000, // 增加到 120 秒，因为 LLM API 调用可能需要较长时间
});

export interface QueryRequest {
  question: string;
  models?: string[];
  prompt_id?: string;
  prompt_version?: string;
  max_rounds?: number;
}

export async function runQuery(payload: QueryRequest) {
  const res = await api.post('/v1/query', payload);
  return res.data;
}
