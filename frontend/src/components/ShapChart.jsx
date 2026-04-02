import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import './shap-chart.css';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const item = payload[0].payload;
    const dir = item.value > 0 ? '↑ Increases fraud risk' : '↓ Decreases fraud risk';
    const col = item.value > 0 ? '#ef4444' : '#10b981';
    return (
      <div className="shap-tooltip">
        <div className="shap-tooltip-feature mono">{item.feature}</div>
        <div className="shap-tooltip-val" style={{ color: col }}>
          SHAP: {item.value.toFixed(4)}
        </div>
        <div className="shap-tooltip-dir" style={{ color: col }}>{dir}</div>
      </div>
    );
  }
  return null;
};

export default function ShapChart({ topFeatures }) {
  if (!topFeatures || topFeatures.length === 0) return null;

  const data = topFeatures.map(f => ({
    feature: f.feature,
    value: parseFloat(f.value.toFixed(4)),
    abs: Math.abs(f.value),
    direction: f.direction,
  })).sort((a, b) => a.value - b.value);

  const maxAbs = Math.max(...data.map(d => d.abs));

  return (
    <div className="shap-wrapper glass-card">
      <div className="shap-header">
        <div>
          <div className="shap-title">SHAP Feature Attribution</div>
          <div className="shap-subtitle">Why did the model make this prediction?</div>
        </div>
        <div className="shap-legend">
          <span className="shap-legend-item fraud">↑ Fraud signal</span>
          <span className="shap-legend-item legit">↓ Legit signal</span>
        </div>
      </div>

      <div className="shap-chart-area">
        <ResponsiveContainer width="100%" height={data.length * 42 + 40}>
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 10, right: 20, left: 50, bottom: 10 }}
          >
            <CartesianGrid horizontal={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis
              type="number"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              domain={[-maxAbs * 1.2, maxAbs * 1.2]}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fill: '#e2e8f0', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
              tickLine={false}
              axisLine={false}
              width={50}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={28}>
              {data.map((entry, i) => (
                <Cell
                  key={`cell-${i}`}
                  fill={entry.value > 0 ? '#ef4444' : '#10b981'}
                  fillOpacity={0.7 + (entry.abs / maxAbs) * 0.3}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="shap-footer">
        <span>🔍</span>
        <span>Features with <strong style={{color:'#ef4444'}}>positive SHAP</strong> push toward fraud; <strong style={{color:'#10b981'}}>negative</strong> push toward legitimate.</span>
      </div>
    </div>
  );
}
