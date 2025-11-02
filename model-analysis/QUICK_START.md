# ⚡ Quick Start Guide

Get up and running in **3 simple steps**!

---

## 🚀 Option 1: Automated (Recommended)

```bash
bash run_analysis.sh
```

That's it! The script will:
- ✅ Check dependencies
- ✅ Run the analysis
- ✅ Launch the dashboard

---

## 🔧 Option 2: Manual

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Analysis
```bash
jupyter notebook analysis.ipynb
```
Then: **Kernel → Restart & Run All**

### Step 3: Launch Dashboard
```bash
streamlit run app.py
```

---

## 📊 What You'll Get

The dashboard opens at `http://localhost:8501` with:

1. **📖 The Story** - Project overview
2. **🔍 Data Source** - Dataset explanation
3. **📊 Exploratory Analysis** - Interactive visualizations
   - Including **Provider Impact Analysis** 🆕
4. **🎯 Attack Methods** - Deep dive into attacks
5. **🏢 Provider Analysis** - Radar plots by provider
6. **🔮 Forecasting Models** - ML model comparison
7. **📈 Radar Plots Gallery** - All models visualized
8. **🎲 Make Predictions** - Forecast future AI safety

---

## ⚠️ Important Notes

### Must Run Analysis First!
The notebook **must** be run before the dashboard:
- Generates `data/` folder with analysis results
- Trains ML models
- Creates visualizations

### Files Generated
After running the notebook, you'll have:
```
data/
├── analysis_data.pkl      # Complete analysis
├── all_models.pkl          # All 5 ML models
├── best_model.pkl          # Best model
├── scaler.pkl              # Feature scaler
└── predictions.pkl         # Predictions
```

---

## 🐛 Troubleshooting

### "Model results not available"
→ Run the analysis notebook first!

### "Module not found"
```bash
pip install -r requirements.txt
```

### "streamlit: command not found"
```bash
python3 -m streamlit run app.py
```

### Cache issues
```bash
streamlit cache clear
```

---

## 📚 Need More Help?

- **Full Documentation:** `README.md`
- **Detailed Instructions:** `INSTRUCTIONS.md`
- **Data Info:** `DATA_FILES_INFO.md`
- **Bug Fixes:** `BUG_FIXES.md`

---

**Ready? Let's go!** 🚀

```bash
bash run_analysis.sh
```
