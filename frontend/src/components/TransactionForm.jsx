import { useState } from 'react';
import './transaction-form.css';

const V_FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const ALL_FEATURES = ['Time', ...V_FEATURES, 'Amount'];

const FRAUD_EXAMPLE = {
  Time: 406, V1: -2.3122265423263, V2: 1.95199201064158, V3: -1.60985073229769,
  V4: 3.9979055875468, V5: -0.522187864667764, V6: -1.42654531920595,
  V7: -2.53738730624579, V8: 1.39165724829804, V9: -2.77008927719433,
  V10: -2.77227214465915, V11: 3.20203320709635, V12: -2.89990738849473,
  V13: -0.595221881324605, V14: -4.28925378244217, V15: 0.389724120274487,
  V16: -1.14074717980657, V17: -2.83005567450437, V18: -0.0168224681808257,
  V19: 0.416955705037907, V20: 0.126910559061474, V21: 0.517232370861711,
  V22: -0.0350493686052974, V23: -0.465211076986171, V24: 0.320198197514045,
  V25: 0.0445191674731724, V26: 0.177839798284401, V27: 0.261145002567677,
  V28: -0.143275874698919, Amount: 239.93
};

const LEGIT_EXAMPLE = {
  Time: 0, V1: -1.3598071336738, V2: -0.0727811733098497, V3: 2.53634673796914,
  V4: 1.37815522427443, V5: -0.338320769942518, V6: 0.462387777762292,
  V7: 0.239598554061257, V8: 0.0986979012610507, V9: 0.363786969611213,
  V10: 0.0907941719789513, V11: -0.551599533260813, V12: -0.617800855762348,
  V13: -0.991389847235408, V14: -0.311169353699879, V15: 1.46817697209427,
  V16: -0.470400525259478, V17: 0.207971241929242, V18: 0.0257905801985591,
  V19: 0.403992960255733, V20: 0.251412098239705, V21: -0.018306777944153,
  V22: 0.277837575558899, V23: -0.110473910188767, V24: 0.0669280749146731,
  V25: 0.128539358273528, V26: -0.189114843888824, V27: 0.133558376740387,
  V28: -0.0210530534538215, Amount: 149.62
};

const initialValues = () => Object.fromEntries(ALL_FEATURES.map(f => [f, '']));

export default function TransactionForm({ onSubmit, loading }) {
  const [values, setValues] = useState(initialValues());
  const [focusedField, setFocusedField] = useState(null);

  const handleChange = (field, val) => setValues(prev => ({ ...prev, [field]: val }));

  const handleLoad = (example) => {
    const str = Object.fromEntries(Object.entries(example).map(([k, v]) => [k, String(v)]));
    setValues(str);
  };

  const handleClear = () => setValues(initialValues());

  const handleSubmit = (e) => {
    e.preventDefault();
    const parsed = {};
    for (const [k, v] of Object.entries(values)) {
      const num = parseFloat(v);
      if (isNaN(num)) { alert(`"${k}" must be a valid number.`); return; }
      parsed[k] = num;
    }
    onSubmit(parsed);
  };

  const isComplete = ALL_FEATURES.every(f => values[f] !== '');

  return (
    <div className="tx-form-wrapper">
      {/* Quick Load */}
      <div className="tx-quick-load">
        <span className="section-title">Transaction Analyzer</span>
        <div className="tx-quick-btns">
          <button type="button" className="btn btn-ghost" style={{fontSize:'0.8rem',padding:'7px 16px'}} onClick={() => handleLoad(LEGIT_EXAMPLE)}>
            ✓ Load Legitimate
          </button>
          <button type="button" className="btn btn-danger" style={{fontSize:'0.8rem',padding:'7px 16px'}} onClick={() => handleLoad(FRAUD_EXAMPLE)}>
            ⚠ Load Fraud Sample
          </button>
          <button type="button" className="btn btn-ghost" style={{fontSize:'0.8rem',padding:'7px 16px'}} onClick={handleClear}>
            ✕ Clear
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Special Fields */}
        <div className="tx-special-row">
          <div className="form-group">
            <label className="form-label">Time (seconds)</label>
            <input
              className={`form-input ${focusedField === 'Time' ? 'focused' : ''}`}
              type="number" step="any" placeholder="0.0"
              value={values.Time}
              onChange={e => handleChange('Time', e.target.value)}
              onFocus={() => setFocusedField('Time')}
              onBlur={() => setFocusedField(null)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Amount (USD)</label>
            <input
              className={`form-input ${focusedField === 'Amount' ? 'focused' : ''}`}
              type="number" step="any" min="0" placeholder="0.00"
              value={values.Amount}
              onChange={e => handleChange('Amount', e.target.value)}
              onFocus={() => setFocusedField('Amount')}
              onBlur={() => setFocusedField(null)}
            />
          </div>
        </div>

        {/* PCA Features Grid */}
        <div className="tx-pca-label">
          <span className="form-label">PCA Features (V1 – V28)</span>
          <span className="tx-pca-note">Pre-transformed; enter raw values from dataset</span>
        </div>
        <div className="tx-grid">
          {V_FEATURES.map(feat => (
            <div key={feat} className="form-group">
              <label className="form-label tx-v-label">{feat}</label>
              <input
                className={`form-input tx-v-input ${focusedField === feat ? 'focused' : ''}`}
                type="number" step="any" placeholder="0.0"
                value={values[feat]}
                onChange={e => handleChange(feat, e.target.value)}
                onFocus={() => setFocusedField(feat)}
                onBlur={() => setFocusedField(null)}
              />
            </div>
          ))}
        </div>

        {/* Progress */}
        <div className="tx-progress">
          <div className="flex justify-between mb-1" style={{fontSize:'0.75rem',color:'var(--text-muted)'}}>
            <span>Fields filled</span>
            <span>{ALL_FEATURES.filter(f => values[f] !== '').length} / {ALL_FEATURES.length}</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${(ALL_FEATURES.filter(f => values[f] !== '').length / ALL_FEATURES.length) * 100}%`,
                background: isComplete ? 'linear-gradient(90deg, #10b981, #34d399)' : 'linear-gradient(90deg, #2563eb, #60a5fa)'
              }}
            />
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary w-full"
          style={{marginTop:'20px', padding:'14px', fontSize:'1rem', justifyContent:'center'}}
          disabled={loading || !isComplete}
          id="analyze-btn"
        >
          {loading ? (
            <><div className="spinner" /> Analyzing Transaction...</>
          ) : (
            <><span>🔍</span> Analyze Transaction</>
          )}
        </button>
      </form>
    </div>
  );
}
