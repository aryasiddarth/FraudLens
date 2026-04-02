# FraudLens
FraudLens is a machine learning based risk detection system designed to identify high-risk loan applications in imbalanced real-world financial datasets. The system focuses on detecting likely defaults while minimizing false positives, helping teams prioritize risky cases without over-flagging legitimate applicants.

## Dataset configuration
- Training now reads dataset path from `FRAUDLENS_DATASET_PATH`.
- If not set, it defaults to `application_data.csv` (if present), otherwise `creditcard.csv`.
- Current frontend/backend analyzer is aligned to Home Credit-style `application_data.csv` features.

Example:
`$env:FRAUDLENS_DATASET_PATH="C:\Users\aryas\Desktop\FraudLens\application_data.csv"; python ml/train.py`
