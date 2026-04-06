# 💳 Credit Risk Explainer

A minimal **Explainable Credit Risk** app using scikit-learn + SHAP + Streamlit.

## Stack
| Layer | Library |
|---|---|
| Data | Pandas, NumPy |
| Model | scikit-learn (Random Forestxd) |
| Explainability | SHAP (TreeExplainer) |
| UI | Streamlit |

## Dataset
German Credit Dataset (UCI / Statlog) — 1 000 applicants, 20 features, binary target (good/bad credit).  
Downloaded automatically on first run. Falls back to synthetic data if unreachable.

## Quick Start

```bash
# 1. Create & activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Browser opens at **http://localhost:8501**

## Features
- **Sidebar sliders/dropdowns** — adjust any applicant attribute in real time
- **Instant prediction** — default probability + risk label
- **Per-prediction SHAP waterfall** — which features pushed the score up/down
- **Global feature importance** — overall model behaviour
- **Dataset preview** — inspect raw data

## Project Structure
```
credit_risk_app/
├── app.py           # single-file Streamlit application
├── requirements.txt
└── README.md
```