# 🏠 Property Lead Scoring AI

A machine learning system that automatically classifies property buyers as 🟢 Green, 🟡 Yellow, or 🔴 Red — helping sales teams focus on the highest-value leads.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates 5,000 synthetic Malaysian buyer leads |
| `train_model.py` | Trains the Random Forest model (90%+ accuracy) |
| `app.py` | Streamlit web dashboard for live lead scoring |
| `Lead_Scoring_Notebook.ipynb` | Jupyter notebook combining all steps |

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib
```

### Step 2 — Generate the dataset
```bash
python generate_data.py
```

### Step 3 — Train the model
```bash
python train_model.py
```

### Step 4 — Launch the web app
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🧠 Model Details

- **Algorithm:** Random Forest (300 trees)
- **Accuracy:** ~90.7% on test set
- **Features:** 16 (financial, project fit, engagement, intent)
- **Output:** Green (2) / Yellow (1) / Red (0) + confidence scores

## 🔍 Top Features (by importance)
1. CCRIS/CTOS Credit Issue
2. Monthly Income
3. Response Count
4. Debt-to-Income Ratio
5. Showroom Visit

---

## 💡 Next Steps (to impress clients)
- Connect to real CRM data (replace CSV with API calls)
- Add WhatsApp message sentiment scoring
- Deploy to cloud (Streamlit Cloud is free)
- Add weekly model retraining pipeline