# 🛡️ AI Safety Forecasting Dashboard

**Apart Research Hackathon - Track 1: AI Capability Forecasting & Timeline Models**

A comprehensive analysis and forecasting system for predicting AI model safety performance across 20+ jailbreaking attack methods using model size and release date as inputs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Analysis](#analysis)
- [Dashboard](#dashboard)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This project addresses the AI Forecasting Hackathon challenge by building **transparent, reproducible forecasting models** to predict AI safety performance. We analyze **100+ AI models** across **20+ attack methods** to understand:

- How model characteristics (size, release date) correlate with safety
- Which attack methods are most/least effective
- How different providers approach AI safety
- Whether we can forecast future model safety

### Key Achievements

- ✅ **100+ models analyzed** across 20+ jailbreaking attacks
- ✅ **5 ML models trained** and compared for forecasting
- ✅ **Provider-specific analysis** for major AI companies
- ✅ **Interactive dashboard** with 8 comprehensive sections
- ✅ **Radar plot visualizations** for all models
- ✅ **Real-time predictions** for hypothetical future models

---

## 🚀 Features

### 1. Comprehensive Data Analysis
- Analysis of 100+ AI models from major providers (Anthropic, OpenAI, Meta, Google, etc.)
- 20+ jailbreaking attack methods evaluated
- Temporal trends (2020-2025)
- Provider-specific safety patterns

### 2. Machine Learning Forecasting
- **5 Different Models:**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
- Cross-validation for model stability
- **Comprehensive Feature Importance Analysis:**
  - Tree-based models feature importance
  - Linear models coefficient analysis
  - Comparison across all 5 models
  - Predictive power analysis (Model Size vs Release Date)

### 3. Interactive Streamlit Dashboard
- **8 Main Pages:**
  1. 📖 The Story - Project narrative and key findings
  2. 🔍 Data Source - HydroX benchmark explanation
  3. 📊 Exploratory Analysis - Interactive visualizations
  4. 🎯 Attack Methods Deep Dive - Attack effectiveness analysis
  5. 🏢 Provider Analysis - Company-specific insights with radar plots
  6. 🔮 Forecasting Models - ML model comparison
  7. 📈 Radar Plots Gallery - Visual comparison of all models
  8. 🎲 Make Predictions - Real-time safety forecasting

### 4. Provider Impact Analysis
- Model size impact by provider
- Release date impact by provider
- Provider strategy mapping
- Correlation analysis

---

## ⚡ Quick Start

```bash
# 1. Clone or navigate to the project directory
cd Hackathon_Apart

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis notebook (IMPORTANT - do this first!)
jupyter notebook analysis.ipynb
# Then: Kernel > Restart & Run All

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for analysis)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit >= 1.31.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- plotly >= 5.14.0
- statsmodels >= 0.14.0

### Step 2: Verify Installation

```bash
python -c "import streamlit; import pandas; import plotly; print('All packages installed!')"
```

---

## 🎮 Usage

### Option 1: Using the Shell Script (Recommended)

```bash
bash run_analysis.sh
```

This will:
1. Run the analysis notebook
2. Generate all required data files
3. Launch the Streamlit dashboard

### Option 2: Manual Execution

#### Step 1: Run the Analysis Notebook

```bash
# Open Jupyter
jupyter notebook analysis.ipynb

# In Jupyter:
# - Kernel > Restart & Run All
# - Wait for all cells to complete (~1-2 minutes)
```

**This generates:**
- `data/analysis_data.pkl` - Complete analysis results
- `data/all_models.pkl` - All 5 trained ML models
- `data/best_model.pkl` - Best performing model
- `data/scaler.pkl` - Feature scaler
- `data/predictions.pkl` - Model predictions

#### Step 2: Launch the Dashboard

```bash
streamlit run app.py
```

Or use Python module syntax:

```bash
python -m streamlit run app.py
```

### Option 3: Data Scraping (Optional)

If you want to refresh the dataset:

```bash
python data_scrapper.py
```

This will fetch the latest data from the HydroX leaderboard API.

---

## 📁 Project Structure

```
Hackathon_Apart/
│
├── 📊 Data Files
│   ├── hydrox_attack_methods_with_dates.csv  # Main dataset (100+ models)
│   └── data/                                  # Generated analysis files
│       ├── analysis_data.pkl                  # Complete analysis results
│       ├── all_models.pkl                     # All 5 trained models
│       ├── best_model.pkl                     # Best performing model
│       ├── scaler.pkl                         # Feature scaler
│       └── predictions.pkl                    # Model predictions
│
├── 🔬 Analysis
│   └── analysis.ipynb                         # Jupyter notebook (RUN THIS FIRST!)
│
├── 🎨 Dashboard
│   └── app.py                                 # Streamlit dashboard
│
├── 🛠️ Utilities
│   ├── data_scrapper.py                       # HydroX API scraper
│   └── run_analysis.sh                        # Quick start script
│
├── 📚 Documentation
│   ├── README.md                              # This file
│   ├── INSTRUCTIONS.md                        # Detailed instructions
│   ├── DATA_FILES_INFO.md                     # Data files explanation
│   ├── BUG_FIXES.md                           # Bug fixes documentation
│   ├── RADAR_PLOT_FIX.md                      # Radar plot improvements
│   └── PROVIDER_IMPACT_ADDED.md               # Provider analysis docs
│
└── 📋 Configuration
    └── requirements.txt                       # Python dependencies
```

---

## 📊 Data

### Source: HydroX Leaderboard

The dataset comes from the **HydroX Benchmark**, one of the most comprehensive public datasets for AI safety evaluation.

### Dataset Contents

**100+ AI Models** including:
- Anthropic (Claude 3 Opus, Claude 3.5 Sonnet, etc.)
- OpenAI (GPT-4, GPT-4 Turbo, o1-preview, etc.)
- Meta (Llama 3, Llama 3.1, etc.)
- Google (Gemini Pro, Gemini Flash, etc.)
- And many more...

**20+ Attack Methods:**
- None (baseline)
- ABJ, Adaptive, ArtPrompt, AutoDAN
- Cipher, DAN, DeepInception, Developer
- DRA, DrAttack, GCG, GPTFuzzer
- Grandmother, Masterkey, Multilingual
- PAIR, PastTense, Psychology, ReNeLLM, TAP

**Features:**
- Model name and provider
- Release date
- Model size (parameters in billions)
- Safety scores (0-100) for each attack method
- Overall safety score

---

## 🔬 Analysis

### Exploratory Data Analysis

The notebook performs comprehensive EDA including:

1. **Data Preprocessing**
   - Date conversion and extraction
   - Model size parsing
   - Feature engineering

2. **Statistical Analysis**
   - Correlation analysis (size vs safety, time vs safety)
   - Distribution analysis
   - Provider comparisons

3. **Visualizations**
   - Scatter plots (size vs score, date vs score)
   - Radar plots (40+ models visualized)
   - Bar charts (attack effectiveness)
   - Heatmaps (correlation matrices)
   - Time series (safety trends)

### Machine Learning Models

**Input Features:**
- Model Size (billions of parameters)
- Release Date (days since 2020-01-01)

**Target Variable:**
- Overall Safety Score (0-100)

**Models Trained:**
1. **Linear Regression** - Simple baseline
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization (often best performer)
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Sequential ensemble method

**Evaluation Metrics:**
- R² Score (proportion of variance explained)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- 5-fold Cross-Validation

---

## 🎨 Dashboard

### Page 1: 📖 The Story
**What it shows:**
- Project overview and motivation
- Key statistics (100+ models, 20+ attacks, 5 ML models)
- Research journey timeline
- Main findings summary

**Key Findings:**
- AI models are getting safer over time
- Larger models tend to be more resistant to attacks
- Clear differences in provider approaches
- Some attacks are highly predictable, others vary greatly

### Page 2: 🔍 Data Source
**What it shows:**
- HydroX benchmark explanation
- Data collection methodology
- Complete list of 20+ attack methods with descriptions
- Data limitations and considerations

### Page 3: 📊 Exploratory Analysis
**5 Interactive Tabs:**

1. **📊 Distributions**
   - Overall safety score distribution
   - Model size distribution
   - Statistical summaries

2. **🔗 Correlations**
   - Correlation heatmap
   - Feature relationships
   - Interpretation guides

3. **📅 Temporal Trends**
   - Safety scores over time
   - Trend lines (with LOWESS smoothing)
   - Temporal insights

4. **🏆 Top Models**
   - Top 10 safest models table
   - Bar chart visualization
   - Model details

5. **🏢 Provider Impact** 
   - Model size impact by provider
   - Release date impact by provider
   - Provider strategy map (2D scatter)
   - Quadrant analysis

### Page 4: 🎯 Attack Methods Deep Dive
**What it shows:**
- Attack effectiveness ranking
- Most/least effective attacks
- Attack predictability (variance analysis)
- Individual attack analysis
- Performance by provider

### Page 5: 🏢 Provider Analysis
**What it shows:**
- Provider safety rankings
- Model count by provider
- Provider comparison charts
- **Radar plots by provider** (select any provider to see their models)
- Provider comparison radar overlay

**Interactive Features:**
- Select provider from dropdown
- Choose top/bottom/all models
- View individual or comparison radar plots
- Detailed score tables

### Page 6: 🔮 Forecasting Models
**What it shows:**
- Model performance comparison (R², RMSE)
- Actual vs Predicted scatter plots (all 5 models)
- Detailed metrics table
- Best model highlight
- **Feature Importance Analysis:** ⭐ NEW!
  - Tree-based models (Random Forest, Gradient Boosting)
  - Linear models (Linear, Ridge, Lasso)
  - Comparison across all 5 models
  - Which feature has more predictive power
- **Correlation insights:**
  - Model size impact on safety
  - Temporal trends in safety
  - Interpretation of findings

### Page 7: 📈 Radar Plots Gallery
**What it shows:**
- Comprehensive radar plot gallery
- Filter by provider, view option, sort method
- Individual plots with model info
- Comparison view (overlay multiple models)
- Top 3 strengths for each model

**Filter Options:**
- Top 10/20 models
- Bottom 10/20 models
- All models
- Custom selection
- Filter by provider

### Page 8: 🎲 Make Predictions
**What it shows:**
- Interactive prediction interface
- Input: model size (0.5B - 500B) and release date
- Output: predicted safety score
- **Out-of-range handling:**
  - Warning if prediction > 100 or < 0
  - Clamping to valid range
  - Explanation that benchmarks may need redefinition
- Safety interpretation (Excellent/Good/Moderate/Concerning)
- Comparison with existing models
- Visualization of prediction vs dataset

---

## 📈 Results

### Key Insights

1. **Temporal Trends**
   - Our analysis suggests some improvement in safety over time
   - Improvement patterns vary considerably by provider
   - Some providers show moderate temporal correlation (e.g., OpenAI)
   - Others show stronger size-based patterns (e.g., Meta-Llama)

2. **Model Size Effects**
   - Moderate positive correlation observed overall (~0.36)
   - Effect varies significantly by provider and model architecture
   - Size alone is not deterministic of safety performance

3. **Attack Method Insights**
   - **More Effective Attacks:** Adaptive, DRA, ABJ (lower average scores)
   - **Less Effective Attacks:** PastTense, Psychology, Masterkey (higher average scores)
   - Some attacks show more consistent patterns (e.g., GCG, TAP)
   - Others exhibit high variance across models

4. **Provider Patterns**
   - **Anthropic:** Generally high safety scores in our dataset
   - **OpenAI:** Shows improvement trends over time
   - **Meta-Llama:** Size correlation appears stronger
   - **Google:** Moderate performance with variation across models
   
   *Note: These observations are based on the available benchmark data and may not capture all aspects of each provider's approach.*

### Model Performance

**Best Performing Model:** Lasso Regression
- R² Score: ~0.48
- RMSE: ~12.8
- Interpretation: Moderate predictive power, useful for trend analysis

**Note:** The relatively modest R² indicates that safety is influenced by many factors beyond just size and date (architecture, training data, alignment techniques, etc.)

---

## 🐛 Troubleshooting

### Issue: "Model results not available"
**Solution:** Run the analysis notebook first!
```bash
jupyter notebook analysis.ipynb
# Kernel > Restart & Run All
```

### Issue: "No such file or directory: 'hydrox_attack_methods_with_dates.csv'"
**Solution:** The dataset is missing. Run the scraper:
```bash
python data_scrapper.py
```

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "ValueError: not enough values to unpack"
**Solution:** Clear Streamlit cache:
```bash
streamlit cache clear
```
Then restart the app.

### Issue: Plots not showing or errors with trendlines
**Solution:** Install statsmodels:
```bash
pip install statsmodels
```

### Issue: Streamlit app won't start
**Solution:** Try using Python module syntax:
```bash
python3 -m streamlit run app.py
```

### Issue: Predictions showing > 100%
**This is expected!** The app now handles this gracefully:
- Shows a warning box
- Clamps the value to 100
- Explains that benchmarks may need redefinition
- This indicates the model is extrapolating beyond training data

---

## 🎯 Hackathon Alignment

This project addresses **Track 1: AI Capability Forecasting & Timeline Models** by:

✅ Building transparent, reproducible forecasting methods  
✅ Creating evaluation pipelines for AI safety trends  
✅ Developing quantitative models for safety prediction  
✅ Comparing multiple forecasting approaches  
✅ Providing tools for estimating future AI safety trajectories  
✅ Analyzing temporal trends and provider strategies  
✅ Making all code and analysis open-source  

---

## 🔮 Future Enhancements

- [ ] Incorporate additional features (training data size, architecture type)
- [ ] Time-series forecasting (ARIMA, Prophet)
- [ ] Ensemble methods combining multiple models
- [ ] Uncertainty quantification with confidence intervals
- [ ] Real-time data updates and monitoring
- [ ] Integration with other AI safety benchmarks
- [ ] Provider-specific forecasting models
- [ ] Attack method-specific predictions

---

## 👥 Contributing

This is an open-source project built for the Apart Research AI Forecasting Hackathon.

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

**Areas for contribution:**
- Additional ML models
- New visualizations
- Data pipeline improvements
- Documentation enhancements
- Bug fixes

---

## 📄 License

Open source - feel free to use and modify for AI safety research.

---

## 🙏 Acknowledgments

- **Apart Research** for organizing the hackathon
- **HydroX Benchmark Team** for the comprehensive dataset
- **AI Safety Research Community** for ongoing work in this critical area

---

## 📞 Contact & Support

For questions, issues, or contributions:
- Open an issue in the repository
- Refer to detailed documentation in `INSTRUCTIONS.md`
- Check `DATA_FILES_INFO.md` for data file explanations
- Review `BUG_FIXES.md` for known issues and solutions

---

## 🚀 Quick Reference

### Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (REQUIRED FIRST STEP)
jupyter notebook analysis.ipynb

# Launch dashboard
streamlit run app.py

# Refresh data (optional)
python data_scrapper.py

# Clear cache if needed
streamlit cache clear
```

### File Locations

- **Dataset:** `hydrox_attack_methods_with_dates.csv`
- **Analysis:** `analysis.ipynb`
- **Dashboard:** `app.py`
- **Generated Data:** `data/` folder
- **Documentation:** `*.md` files

---

**Built with ❤️ for AI Safety Research**

*Last Updated: November 2, 2025*
