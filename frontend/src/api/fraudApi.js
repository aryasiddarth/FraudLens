import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export const getModelInfo = async () => {
  const { data } = await api.get('/model/info');
  return data;
};

export const predictFraud = async (transaction) => {
  const { data } = await api.post('/predict', transaction);
  return data;
};

export const predictBatch = async (transactions) => {
  const { data } = await api.post('/predict/batch', { transactions });
  return data;
};

export default api;
