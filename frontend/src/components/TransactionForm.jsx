import { useState } from 'react';
import './transaction-form.css';

const SELECT_FIELDS = [
  {
    key: 'NAME_CONTRACT_TYPE',
    label: 'Contract Type',
    options: ['Cash loans', 'Revolving loans'],
  },
  {
    key: 'CODE_GENDER',
    label: 'Gender',
    options: ['F', 'M'],
  },
  {
    key: 'FLAG_OWN_CAR',
    label: 'Owns Car',
    options: ['Y', 'N'],
  },
  {
    key: 'FLAG_OWN_REALTY',
    label: 'Owns Realty',
    options: ['Y', 'N'],
  },
  {
    key: 'NAME_INCOME_TYPE',
    label: 'Income Type',
    options: ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed', 'Student'],
  },
  {
    key: 'NAME_EDUCATION_TYPE',
    label: 'Education',
    options: [
      'Secondary / secondary special',
      'Higher education',
      'Incomplete higher',
      'Lower secondary',
      'Academic degree',
    ],
  },
];

const NUMBER_FIELDS = [
  { key: 'CNT_CHILDREN', label: 'Children Count', min: 0, placeholder: '0' },
  { key: 'AMT_INCOME_TOTAL', label: 'Total Income', min: 0, placeholder: '180000' },
  { key: 'AMT_CREDIT', label: 'Credit Amount', min: 0, placeholder: '450000' },
  { key: 'AMT_ANNUITY', label: 'Annuity Amount', min: 0, placeholder: '25000' },
  { key: 'AMT_GOODS_PRICE', label: 'Goods Price', min: 0, placeholder: '405000' },
  { key: 'DAYS_BIRTH', label: 'Days Birth (negative)', placeholder: '-14000' },
  { key: 'DAYS_EMPLOYED', label: 'Days Employed (negative)', placeholder: '-2000' },
  { key: 'EXT_SOURCE_1', label: 'External Score 1', min: 0, max: 1, step: 'any', placeholder: '0.45' },
  { key: 'EXT_SOURCE_2', label: 'External Score 2', min: 0, max: 1, step: 'any', placeholder: '0.62' },
  { key: 'EXT_SOURCE_3', label: 'External Score 3', min: 0, max: 1, step: 'any', placeholder: '0.51' },
];

const ALL_FEATURES = [...SELECT_FIELDS.map(f => f.key), ...NUMBER_FIELDS.map(f => f.key)];

const FRAUD_EXAMPLE = {
  NAME_CONTRACT_TYPE: 'Cash loans',
  CODE_GENDER: 'M',
  FLAG_OWN_CAR: 'N',
  FLAG_OWN_REALTY: 'N',
  NAME_INCOME_TYPE: 'Unemployed',
  NAME_EDUCATION_TYPE: 'Lower secondary',
  CNT_CHILDREN: 3,
  AMT_INCOME_TOTAL: 90000,
  AMT_CREDIT: 900000,
  AMT_ANNUITY: 52000,
  AMT_GOODS_PRICE: 900000,
  DAYS_BIRTH: -9000,
  DAYS_EMPLOYED: -100,
  EXT_SOURCE_1: 0.08,
  EXT_SOURCE_2: 0.12,
  EXT_SOURCE_3: 0.11,
};

const LEGIT_EXAMPLE = {
  NAME_CONTRACT_TYPE: 'Cash loans',
  CODE_GENDER: 'F',
  FLAG_OWN_CAR: 'N',
  FLAG_OWN_REALTY: 'Y',
  NAME_INCOME_TYPE: 'Working',
  NAME_EDUCATION_TYPE: 'Higher education',
  CNT_CHILDREN: 0,
  AMT_INCOME_TOTAL: 202500,
  AMT_CREDIT: 406597.5,
  AMT_ANNUITY: 24700.5,
  AMT_GOODS_PRICE: 351000,
  DAYS_BIRTH: -9461,
  DAYS_EMPLOYED: -637,
  EXT_SOURCE_1: 0.41,
  EXT_SOURCE_2: 0.62,
  EXT_SOURCE_3: 0.51,
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
    for (const field of SELECT_FIELDS) {
      const val = values[field.key];
      if (!val) { alert(`"${field.label}" is required.`); return; }
      parsed[field.key] = val;
    }
    for (const field of NUMBER_FIELDS) {
      const val = values[field.key];
      const num = parseFloat(val);
      if (isNaN(num)) { alert(`"${field.label}" must be a valid number.`); return; }
      parsed[field.key] = num;
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
        <div className="tx-pca-label">
          <span className="form-label">Applicant Profile</span>
          <span className="tx-pca-note">Home Credit-style inputs for loan default risk scoring</span>
        </div>
        <div className="tx-grid">
          {SELECT_FIELDS.map(field => (
            <div key={field.key} className="form-group">
              <label className="form-label tx-v-label">{field.label}</label>
              <select
                className={`form-input tx-v-input ${focusedField === field.key ? 'focused' : ''}`}
                value={values[field.key]}
                onChange={e => handleChange(field.key, e.target.value)}
                onFocus={() => setFocusedField(field.key)}
                onBlur={() => setFocusedField(null)}
              >
                <option value="">Select...</option>
                {field.options.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
          ))}
        </div>

        <div className="tx-pca-label">
          <span className="form-label">Numeric Indicators</span>
          <span className="tx-pca-note">Enter values from the applicant record</span>
        </div>
        <div className="tx-grid">
          {NUMBER_FIELDS.map(field => (
            <div key={field.key} className="form-group">
              <label className="form-label tx-v-label">{field.label}</label>
              <input
                className={`form-input tx-v-input ${focusedField === field.key ? 'focused' : ''}`}
                type="number"
                step={field.step || 'any'}
                min={field.min}
                max={field.max}
                placeholder={field.placeholder}
                value={values[field.key]}
                onChange={e => handleChange(field.key, e.target.value)}
                onFocus={() => setFocusedField(field.key)}
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
