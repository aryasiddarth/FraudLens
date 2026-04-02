import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './prediction-result.css';

function GaugeArc({ probability }) {
  const r = 80;
  const cx = 110, cy = 110;
  const startAngle = 180;
  const endAngle = 0;
  const totalAngle = 180;
  const valueAngle = startAngle - probability * totalAngle;

  const toRad = d => d * Math.PI / 180;
  const arcPath = (a1, a2) => {
    const x1 = cx + r * Math.cos(toRad(a1));
    const y1 = cy - r * Math.sin(toRad(a1));
    const x2 = cx + r * Math.cos(toRad(a2));
    const y2 = cy - r * Math.sin(toRad(a2));
    const large = Math.abs(a1 - a2) > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 0 ${x2} ${y2}`;
  };

  const needleX = cx + r * 0.85 * Math.cos(toRad(valueAngle));
  const needleY = cy - r * 0.85 * Math.sin(toRad(valueAngle));

  const color = probability < 0.3 ? '#10b981' : probability < 0.6 ? '#f59e0b' : probability < 0.85 ? '#ef4444' : '#dc2626';

  return (
    <svg viewBox="0 0 220 130" width="220" height="130">
      {/* Track */}
      <path d={arcPath(180, 0)} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="14" strokeLinecap="round" />
      {/* Safe zone */}
      <path d={arcPath(180, 126)} fill="none" stroke="rgba(16,185,129,0.3)" strokeWidth="14" />
      {/* Medium zone */}
      <path d={arcPath(126, 72)} fill="none" stroke="rgba(245,158,11,0.3)" strokeWidth="14" />
      {/* Danger zone */}
      <path d={arcPath(72, 27)} fill="none" stroke="rgba(239,68,68,0.3)" strokeWidth="14" />
      {/* Critical zone */}
      <path d={arcPath(27, 0)} fill="none" stroke="rgba(220,38,38,0.4)" strokeWidth="14" />
      {/* Fill */}
      <path d={arcPath(180, valueAngle)} fill="none" stroke={color} strokeWidth="14"
        strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
      {/* Needle */}
      <line x1={cx} y1={cy} x2={needleX} y2={needleY}
        stroke={color} strokeWidth="2.5" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
      <circle cx={cx} cy={cy} r="5" fill={color} />
      {/* Labels */}
      <text x="28" y="118" fill="rgba(255,255,255,0.4)" fontSize="9" textAnchor="middle">0%</text>
      <text x="192" y="118" fill="rgba(255,255,255,0.4)" fontSize="9" textAnchor="middle">100%</text>
    </svg>
  );
}

const getRiskConfig = (riskLevel) => {
  switch (riskLevel) {
    case 'LOW':      return { color: '#10b981', glow: 'rgba(16,185,129,0.3)', label: 'LOW RISK',      icon: '✓', bgClass: 'result-legit' };
    case 'MEDIUM':   return { color: '#f59e0b', glow: 'rgba(245,158,11,0.3)',  label: 'MEDIUM RISK',   icon: '⚠', bgClass: 'result-medium' };
    case 'HIGH':     return { color: '#ef4444', glow: 'rgba(239,68,68,0.3)',   label: 'HIGH RISK',     icon: '⚠', bgClass: 'result-fraud' };
    case 'CRITICAL': return { color: '#dc2626', glow: 'rgba(220,38,38,0.4)',   label: 'CRITICAL RISK', icon: '✕', bgClass: 'result-critical' };
    default:         return { color: '#94a3b8', glow: 'transparent', label: riskLevel, icon: '?', bgClass: '' };
  }
};

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { is_fraud, fraud_probability, risk_level, threshold_used } = result;
  const cfg = getRiskConfig(risk_level);
  const pct = (fraud_probability * 100).toFixed(2);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={fraud_probability}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className={`result-card glass-card ${cfg.bgClass}`}
        style={{ '--result-color': cfg.color, '--result-glow': cfg.glow }}
      >
        {/* Verdict Banner */}
        <div className="result-banner">
          <motion.div
            className="result-icon"
            initial={{ scale: 0.5, rotate: -20 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.1, type: 'spring', stiffness: 300 }}
          >
            {cfg.icon}
          </motion.div>
          <div>
            <div className="result-verdict">{is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE TRANSACTION'}</div>
            <div className="result-subtext">{is_fraud ? 'This transaction has been flagged as suspicious.' : 'No fraud indicators detected.'}</div>
          </div>
          <div className="result-pulse">
            <span className={`pulse-dot ${is_fraud ? 'red' : 'green'}`} />
          </div>
        </div>

        {/* Gauge */}
        <div className="result-gauge-row">
          <div className="result-gauge">
            <GaugeArc probability={fraud_probability} />
            <div className="result-pct" style={{ color: cfg.color }}>
              {pct}<span style={{ fontSize: '0.9rem' }}>%</span>
            </div>
            <div className="result-pct-label">Fraud Probability</div>
          </div>

          <div className="result-details">
            <div className="result-detail-item">
              <span className="result-detail-label">Risk Level</span>
              <span className="badge" style={{
                background: `${cfg.glow}`, color: cfg.color,
                border: `1px solid ${cfg.color}40`
              }}>
                {cfg.label}
              </span>
            </div>
            <div className="result-detail-item">
              <span className="result-detail-label">Decision Threshold</span>
              <span className="mono" style={{ color: 'var(--text-primary)', fontSize: '1rem', fontWeight: 600 }}>
                {(threshold_used * 100).toFixed(1)}%
              </span>
            </div>
            <div className="result-detail-item">
              <span className="result-detail-label">Confidence</span>
              <div className="w-full" style={{ marginTop: 4 }}>
                <div className="progress-bar">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ delay: 0.3, duration: 0.8, ease: 'easeOut' }}
                    style={{
                      background: `linear-gradient(90deg, ${cfg.color}99, ${cfg.color})`
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
