import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Safety Forecasting - The Complete Story",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .story-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        df = pd.read_csv('hydrox_attack_methods_with_dates.csv')
        
        # Clean and preprocess data
        # Convert Release_Date to datetime, handling errors
        if 'Release_Date' in df.columns:
            df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
        
        # Convert numeric columns that might be strings
        numeric_cols = ['Overall_Score', 'Size', 'Model_Size_Numeric']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Try to load analysis data from data folder first, then root
        analysis_data = None
        predictions_data = None
        all_models = None
        
        for folder in ['data/', '']:
            try:
                with open(f'{folder}analysis_data.pkl', 'rb') as f:
                    analysis_data = pickle.load(f)
                print(f"Loaded analysis_data from {folder if folder else 'root'}")
                break
            except FileNotFoundError:
                continue
        
        # Try to load predictions
        for folder in ['data/', '']:
            try:
                with open(f'{folder}predictions.pkl', 'rb') as f:
                    predictions_data = pickle.load(f)
                break
            except FileNotFoundError:
                continue
        
        # Try to load all models
        for folder in ['data/', '']:
            try:
                with open(f'{folder}all_models.pkl', 'rb') as f:
                    all_models = pickle.load(f)
                break
            except FileNotFoundError:
                continue
        
        return df, analysis_data, predictions_data, all_models
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

df, analysis_data, predictions_data, all_models = load_data()

if df is None:
    st.error("⚠️ Could not load the dataset. Please ensure 'hydrox_attack_methods_with_dates.csv' exists.")
    st.stop()

# Helper function to get ONLY attack score columns (not trials or risks)
def get_attack_score_columns(df):
    """
    Get only the attack method score columns, excluding:
    - Trials columns (*_Trials)
    - Risks columns (*_Risks)
    - Overall_Score
    - Metadata columns
    """
    # Get all columns ending with _Score
    score_cols = [col for col in df.columns if col.endswith('_Score')]
    
    # Remove Overall_Score if present
    score_cols = [col for col in score_cols if col != 'Overall_Score']
    
    # Verify these are actual attack scores (should have corresponding Trials/Risks)
    attack_scores = []
    for col in score_cols:
        attack_name = col.replace('_Score', '')
        # Check if this attack has Trials and Risks columns (validates it's a real attack)
        if f'{attack_name}_Trials' in df.columns or f'{attack_name}_Risks' in df.columns:
            attack_scores.append(col)
    
    return attack_scores if len(attack_scores) > 0 else score_cols

# Helper function to create radar plots
def create_radar_plot(model_data, model_name, attack_methods):
    fig = go.Figure()
    
    values = [model_data.get(method, 0) for method in attack_methods]
    values.append(values[0])  # Close the plot
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=attack_methods + [attack_methods[0]],
        fill='toself',
        name=model_name,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title=f"{model_name} - Safety Profile",
        height=500
    )
    
    return fig

# Sidebar navigation
st.sidebar.markdown("# 🛡️ Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["📖 The Story", "🔍 Data Source", "📊 Exploratory Analysis", 
     "🎯 Attack Methods Deep Dive", "🏢 Provider Analysis", 
     "🔮 Forecasting Models", "📈 Radar Plots Gallery", "🎲 Make Predictions"]
)

# ============================================================================
# PAGE 1: THE STORY
# ============================================================================
if page == "📖 The Story":
    st.markdown('<h1 class="main-header">🛡️ AI Safety Forecasting: The Complete Story</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🎯 The Challenge</h2>
    <p style="font-size: 1.1rem;">
    As AI models become more powerful, understanding and predicting their safety characteristics 
    becomes crucial for AI governance and responsible deployment. But can we forecast how safe 
    future AI models will be? <strong>This is the question we set out to answer.</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3>{len(df)}</h3>
        <p>AI Models Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>20+</h3>
        <p>Attack Methods</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>5</h3>
        <p>ML Models Trained</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>2020-2025</h3>
        <p>Time Period</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🚀 Our Approach</h2>
    <p style="font-size: 1.1rem;">
    We built transparent, reproducible forecasting models that predict AI safety performance 
    using just two key inputs: <strong>model size</strong> (number of parameters) and <strong>release date</strong>.
    Our goal was to create a tool that helps anticipate future AI safety trends and inform governance decisions.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 The Journey")
    
    timeline_data = pd.DataFrame({
        'Step': ['1️⃣ Data Collection', '2️⃣ Exploratory Analysis', '3️⃣ Feature Engineering', 
                 '4️⃣ Model Training', '5️⃣ Provider Analysis', '6️⃣ Interactive Dashboard'],
        'Description': [
            'Scraped HydroX leaderboard with 100+ models across 20+ attack methods',
            'Analyzed correlations, trends, and patterns in the safety data',
            'Extracted model size and dates, created temporal features',
            'Trained 5 different ML models and compared their performance',
            'Built provider-specific models for major AI companies',
            'Created this interactive dashboard for real-time forecasting'
        ],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
    })
    
    st.dataframe(timeline_data, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="success-box">
    <h3>🎯 Key Findings</h3>
    <ul style="font-size: 1.05rem;">
        <li><strong>Temporal Trends:</strong> Our analysis suggests some improvement in safety over time, though the trend is modest and varies by provider</li>
        <li><strong>Size Correlation:</strong> Larger models show moderate positive correlation with safety, but this relationship is not deterministic and varies significantly by provider</li>
        <li><strong>Provider Differences:</strong> We observe notable differences in safety approaches across companies, though more research is needed to understand the underlying factors</li>
        <li><strong>Attack Predictability:</strong> Some attacks (e.g., GCG, TAP) show more consistent patterns, while others exhibit high variance across models</li>
        <li><strong>Forecasting Limitations:</strong> Our models achieve moderate predictive performance (R² ~0.48), indicating that safety depends on many factors beyond just size and release date</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🔮 What You'll Discover</h2>
    <p style="font-size: 1.1rem;">
    Navigate through the sections using the sidebar to explore:
    </p>
    <ul style="font-size: 1.05rem;">
        <li><strong>🔍 Data Source:</strong> Where the data comes from and what it contains</li>
        <li><strong>📊 Exploratory Analysis:</strong> Interactive visualizations and statistical insights</li>
        <li><strong>🎯 Attack Methods:</strong> Deep dive into each jailbreaking technique</li>
        <li><strong>🏢 Provider Analysis:</strong> Company-specific safety patterns and radar plots</li>
        <li><strong>🔮 Forecasting Models:</strong> ML model comparison and performance metrics</li>
        <li><strong>📈 Radar Plots Gallery:</strong> Visual comparison of all models</li>
        <li><strong>🎲 Make Predictions:</strong> Forecast safety for hypothetical future models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DATA SOURCE
# ============================================================================
elif page == "🔍 Data Source":
    st.markdown('<h1 class="main-header">🔍 Data Source & Collection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>📡 The HydroX Benchmark</h2>
    <p style="font-size: 1.1rem;">
    Our data comes from the <strong>HydroX Leaderboard</strong>, a comprehensive benchmark that evaluates 
    AI model safety across multiple jailbreaking attack methods. This is one of the most extensive 
    public datasets for AI safety evaluation, providing standardized testing across diverse models.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h3>🎯 What We Collected</h3>
        <ul>
            <li><strong>Model Information:</strong> Name, provider, size, release date</li>
            <li><strong>Safety Scores:</strong> Performance against 20+ attack methods (0-100 scale)</li>
            <li><strong>Attack Details:</strong> Number of trials and risks for each attack</li>
            <li><strong>Overall Score:</strong> Aggregate safety metric across all attacks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h3>🔧 How We Collected It</h3>
        <ul>
            <li><strong>API Scraping:</strong> Direct access to HydroX JSON endpoints</li>
            <li><strong>Date Extraction:</strong> Parsed release dates from model names and metadata</li>
            <li><strong>Size Extraction:</strong> Extracted parameter counts (e.g., "7B", "70B")</li>
            <li><strong>Data Cleaning:</strong> Handled missing values and standardized formats</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", len(df))
    with col2:
        providers_count = df['Provider'].nunique() if 'Provider' in df.columns else len(df['Model'].str.split().str[0].unique())
        st.metric("Providers", providers_count)
    with col3:
        # Get only attack score columns (not trials or risks)
        attack_score_cols = get_attack_score_columns(df)
        st.metric("Attack Methods", len(attack_score_cols))
    
    st.markdown("### 🔍 Sample Data")
    display_cols = ['Model'] + [col for col in df.columns if col in ['Provider', 'Size', 'Release_Date', 'Overall_Score', 'None', 'GCG', 'TAP']]
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    
    st.markdown("### 🎯 Attack Methods Tested")
    
    attack_methods_info = [
        ("None", "Baseline - No attack applied", "Measures inherent model safety"),
        ("ABJ", "Adversarial Behavior Jailbreak", "Exploits behavioral patterns"),
        ("Adaptive", "Adaptive attack strategies", "Evolving attack techniques"),
        ("ArtPrompt", "ASCII art-based prompts", "Visual encoding attacks"),
        ("AutoDAN", "Automated adversarial prompts", "Automated jailbreak generation"),
        ("Cipher", "Encoded/encrypted prompts", "Obfuscation techniques"),
        ("DAN", "Do Anything Now jailbreak", "Classic role-playing attack"),
        ("DeepInception", "Nested scenario attacks", "Multi-layer deception"),
        ("Developer", "Developer mode exploitation", "Pretending to be in dev mode"),
        ("DRA", "Direct Request Attack", "Straightforward harmful requests"),
        ("DrAttack", "Doctor Attack scenario", "Medical authority exploitation"),
        ("GCG", "Greedy Coordinate Gradient", "Optimization-based attack"),
        ("GPTFuzzer", "Automated fuzzing", "Systematic prompt testing"),
        ("Grandmother", "Grandmother scenario", "Emotional manipulation"),
        ("Masterkey", "Universal jailbreak key", "General-purpose bypass"),
        ("Multilingual", "Non-English attacks", "Language-based evasion"),
        ("PAIR", "Prompt Automatic Iterative Refinement", "Iterative optimization"),
        ("PastTense", "Past tense reformulation", "Temporal framing"),
        ("Psychology", "Psychological manipulation", "Cognitive biases exploitation"),
        ("ReNeLLM", "Reinforcement learning attacks", "RL-based jailbreaks"),
        ("TAP", "Tree of Attacks with Pruning", "Structured attack search")
    ]
    
    attack_df = pd.DataFrame(attack_methods_info, columns=['Attack Method', 'Full Name', 'Description'])
    st.dataframe(attack_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ Data Limitations & Considerations</h3>
    <ul>
        <li><strong>Date Availability:</strong> Not all models have complete release date information</li>
        <li><strong>Size Estimates:</strong> Model sizes are estimates based on public information</li>
        <li><strong>Provider Representation:</strong> Some providers have limited representation in the dataset</li>
        <li><strong>Testing Methodology:</strong> Benchmark results may vary based on testing approach</li>
        <li><strong>Temporal Bias:</strong> More recent models may have been trained with awareness of these attacks</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: EXPLORATORY ANALYSIS
# ============================================================================
elif page == "📊 Exploratory Analysis":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🔍 Understanding the Data</h2>
    <p style="font-size: 1.1rem;">
    Before building forecasting models, we need to understand the relationships in our data. 
    Let's explore how model size, release date, and provider affect safety performance.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key statistics
    st.markdown("### 📈 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['Overall_Score'].mean() if 'Overall_Score' in df.columns else df[[col for col in df.columns if col not in ['Model', 'Provider', 'Size', 'Release_Date']]].mean().mean()
        st.metric("Average Safety Score", f"{avg_score:.1f}")
    
    with col2:
        if 'Size' in df.columns and df['Size'].notna().sum() > 0:
            avg_size = df['Size'].mean()
            st.metric("Average Model Size", f"{avg_size:.1f}B")
        elif 'Model_Size_Numeric' in df.columns and df['Model_Size_Numeric'].notna().sum() > 0:
            avg_size = df['Model_Size_Numeric'].mean()
            st.metric("Average Model Size", f"{avg_size:.1f}B")
        else:
            st.metric("Average Model Size", "N/A")
    
    with col3:
        score_col = 'Overall_Score' if 'Overall_Score' in df.columns else [col for col in df.columns if col not in ['Model', 'Provider', 'Size', 'Release_Date']][0]
        std_score = df[score_col].std()
        st.metric("Score Std Dev", f"{std_score:.1f}")
    
    with col4:
        if 'Release_Date' in df.columns and df['Release_Date'].notna().sum() > 0:
            try:
                date_range = (df['Release_Date'].max() - df['Release_Date'].min()).days
                st.metric("Date Range", f"{date_range} days")
            except:
                st.metric("Date Range", "N/A")
        else:
            st.metric("Date Range", "N/A")
    
    # Visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distributions", "🔗 Correlations", "📅 Temporal Trends", "🏆 Top Models", "🏢 Provider Impact"])
    
    with tab1:
        st.markdown("### Score Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall score distribution
            fig = px.histogram(df, x='Overall_Score' if 'Overall_Score' in df.columns else df.columns[1], 
                             nbins=30, title="Overall Safety Score Distribution",
                             labels={'x': 'Safety Score', 'y': 'Count'})
            fig.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model size distribution
            size_col = None
            if 'Size' in df.columns and df['Size'].notna().sum() > 0:
                size_col = 'Size'
            elif 'Model_Size_Numeric' in df.columns and df['Model_Size_Numeric'].notna().sum() > 0:
                size_col = 'Model_Size_Numeric'
            
            if size_col:
                df_size = df[df[size_col].notna()]
                fig = px.histogram(df_size, x=size_col, nbins=30, title="Model Size Distribution",
                                 labels={size_col: 'Model Size (B)', 'count': 'Count'})
                fig.update_traces(marker_color='#ff7f0e')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model size data not available")
    
    with tab2:
        st.markdown("### Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          text_auto='.2f',
                          aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>🔍 Key Correlations</h4>
            <ul>
                <li>Positive correlations indicate features that increase together</li>
                <li>Negative correlations show inverse relationships</li>
                <li>Values close to 0 suggest weak or no linear relationship</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough numeric columns for correlation analysis")
    
    with tab3:
        st.markdown("### Temporal Trends")
        
        if 'Release_Date' in df.columns and 'Overall_Score' in df.columns:
            df_temporal = df[df['Release_Date'].notna() & df['Overall_Score'].notna()].copy()
            
            if len(df_temporal) > 0:
                df_temporal = df_temporal.sort_values('Release_Date')
                
                # Try to add trendline if statsmodels is available
                try:
                    trendline = "lowess" if len(df_temporal) > 10 else None
                    fig = px.scatter(df_temporal, x='Release_Date', y='Overall_Score',
                                   title="Safety Score Over Time",
                                   trendline=trendline,
                                   labels={'Release_Date': 'Release Date', 'Overall_Score': 'Safety Score'})
                except:
                    # Fallback without trendline if statsmodels not installed
                    fig = px.scatter(df_temporal, x='Release_Date', y='Overall_Score',
                                   title="Safety Score Over Time",
                                   labels={'Release_Date': 'Release Date', 'Overall_Score': 'Safety Score'})
                
                fig.update_traces(marker=dict(size=8, opacity=0.6))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough temporal data available")
            
            st.markdown("""
            <div class="success-box">
            <h4>📈 Temporal Insights</h4>
            <p>The trend line shows how AI safety has evolved over time. An upward trend indicates 
            improving safety performance in newer models.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Temporal data not available")
    
    with tab4:
        st.markdown("### Top Performing Models")
        
        score_col = 'Overall_Score' if 'Overall_Score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        top_models = df.nlargest(10, score_col)
        
        display_cols = ['Model'] + [col for col in top_models.columns if col in ['Provider', 'Size', 'Overall_Score', score_col]]
        st.dataframe(top_models[display_cols].reset_index(drop=True), use_container_width=True)
        
        # Bar chart of top models
        fig = px.bar(top_models.head(10), x='Model', y=score_col,
                    title="Top 10 Safest Models",
                    labels={'Model': 'Model Name', score_col: 'Safety Score'})
        fig.update_traces(marker_color='#2ca02c')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🏢 Provider Impact Analysis")
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Understanding Provider Strategies</h4>
        <p>Different AI providers have different approaches to safety. Let's analyze how model size 
        and release timing impact safety performance overall and for each provider.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall Size Impact Analysis
        st.markdown("#### 📏 Overall Model Size Impact")
        
        size_col = 'Model_Size_Numeric' if 'Model_Size_Numeric' in df.columns else 'Size'
        
        if size_col in df.columns and 'Overall_Score' in df.columns:
            df_size_clean = df[df[size_col].notna() & df['Overall_Score'].notna()].copy()
            
            if len(df_size_clean) > 0:
                # Calculate overall correlation
                overall_corr = df_size_clean[size_col].corr(df_size_clean['Overall_Score'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Scatter plot with trendline
                    fig = px.scatter(df_size_clean, 
                                   x=size_col, 
                                   y='Overall_Score',
                                   color='Overall_Score',
                                   color_continuous_scale='RdYlGn',
                                   title=f"Model Size vs Safety Score<br><sub>Correlation: {overall_corr:.3f}</sub>",
                                   labels={size_col: 'Model Size (Billions of Parameters)', 
                                          'Overall_Score': 'Safety Score'},
                                   hover_data=['Model'] if 'Model' in df_size_clean.columns else None)
                    
                    # Add trendline
                    z = np.polyfit(df_size_clean[size_col], df_size_clean['Overall_Score'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df_size_clean[size_col].min(), df_size_clean[size_col].max(), 100)
                    
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>{overall_corr:.3f}</h3>
                    <p>Overall Correlation</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>{len(df_size_clean)}</h3>
                    <p>Models Analyzed</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation
                    if overall_corr > 0.3:
                        interpretation = "Moderate positive correlation"
                        meaning = "Larger models tend to show somewhat better safety scores"
                    elif overall_corr > 0.1:
                        interpretation = "Weak positive correlation"
                        meaning = "Some relationship between size and safety"
                    else:
                        interpretation = "Very weak correlation"
                        meaning = "Size alone is not a strong predictor"
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <h4>📊 Interpretation</h4>
                    <p><strong>{interpretation}</strong></p>
                    <p style="font-size: 0.9rem;">{meaning}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if 'Provider' in df.columns:
            # Initialize correlation dictionaries
            provider_size_corr = {}
            provider_time_corr = {}
            
            # Calculate regression coefficients by provider (matching notebook approach)
            # Prepare data with Days_Since_2020 and extract model size
            df_temp = df.copy()
            
            # Extract numeric size from Model_Size column if it exists
            if 'Model_Size' in df_temp.columns and 'Model_Size_Numeric' not in df_temp.columns:
                df_temp['Model_Size_Numeric'] = df_temp['Model_Size'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            # Use the appropriate size column
            if 'Model_Size_Numeric' in df_temp.columns:
                size_col = 'Model_Size_Numeric'
            elif 'Size' in df_temp.columns:
                size_col = 'Size'
            else:
                size_col = None
            
            # Create Days_Since_2020
            if 'Release_Date' in df_temp.columns:
                df_temp['Days_Since_2020'] = (df_temp['Release_Date'] - pd.Timestamp('2020-01-01')).dt.days
            
            if size_col and size_col in df_temp.columns and 'Overall_Score' in df_temp.columns and 'Days_Since_2020' in df_temp.columns:
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import r2_score
                
                # Calculate provider-specific forecasting results
                provider_r2_scores = {}
                
                # Get top 10 providers by model count
                provider_counts = df_temp['Provider'].value_counts().head(10)
                
                for provider in provider_counts.index:
                    if pd.notna(provider):
                        provider_df = df_temp[(df_temp['Provider'] == provider) & 
                                             df_temp[size_col].notna() & 
                                             df_temp['Days_Since_2020'].notna() &
                                             df_temp['Overall_Score'].notna()].copy()
                        
                        if len(provider_df) >= 5:  # Need minimum 5 samples like notebook
                            # Prepare features (Size and Date)
                            X_prov = provider_df[[size_col, 'Days_Since_2020']].values
                            y_prov = provider_df['Overall_Score'].values
                            
                            # Scale features (IMPORTANT: this is what the notebook does)
                            scaler_prov = StandardScaler()
                            X_prov_scaled = scaler_prov.fit_transform(X_prov)
                            
                            # Train Linear Regression on SCALED features
                            lr_model = LinearRegression()
                            lr_model.fit(X_prov_scaled, y_prov)
                            
                            # Get coefficients (these are for SCALED features)
                            size_coefficient = lr_model.coef_[0]
                            date_coefficient = lr_model.coef_[1]
                            
                            # Calculate R² score
                            y_pred = lr_model.predict(X_prov_scaled)
                            r2 = r2_score(y_prov, y_pred)
                            
                            # Store results
                            provider_size_corr[provider] = size_coefficient
                            provider_time_corr[provider] = date_coefficient
                            provider_r2_scores[provider] = r2
                
                # Model Size Impact by Provider
                st.markdown("#### 📏 Model Size Impact by Provider")
                
                if provider_size_corr:
                    coef_df = pd.DataFrame(list(provider_size_corr.items()), 
                                          columns=['Provider', 'Coefficient'])
                    coef_df = coef_df.sort_values('Coefficient', ascending=False)
                    
                    # Color code: green for positive, red for negative
                    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
                    
                    fig = go.Figure(go.Bar(
                        x=coef_df['Coefficient'],
                        y=coef_df['Provider'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=coef_df['Coefficient'].round(2),
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Model Size Impact by Provider<br><sub>(Positive = Larger Models Are Safer)</sub>",
                        xaxis_title="Coefficient (Impact on Score)",
                        yaxis_title="Provider",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <p><strong>Interpretation:</strong></p>
                    <ul>
                        <li><span style="color: green;">●</span> <strong>Positive (Green):</strong> Larger models from this provider tend to be safer</li>
                        <li><span style="color: red;">●</span> <strong>Negative (Red):</strong> Size doesn't correlate with safety for this provider</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Release Date Impact by Provider
            st.markdown("#### 📅 Release Date Impact by Provider")
            
            if provider_time_corr:
                coef_df = pd.DataFrame(list(provider_time_corr.items()), 
                                      columns=['Provider', 'Coefficient'])
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
                
                fig = go.Figure(go.Bar(
                    x=coef_df['Coefficient'],
                    y=coef_df['Provider'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=coef_df['Coefficient'].round(4),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Release Date Impact by Provider<br><sub>(Positive = Newer Models Are Safer)</sub>",
                    xaxis_title="Coefficient (Impact on Score)",
                    yaxis_title="Provider",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                <p><strong>Interpretation:</strong></p>
                <ul>
                    <li><span style="color: green;">●</span> <strong>Positive (Green):</strong> This provider's newer models are safer (improving over time)</li>
                    <li><span style="color: red;">●</span> <strong>Negative (Red):</strong> No clear improvement or slight decline over time</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Forecasting Accuracy by Provider
                if provider_r2_scores:
                    st.markdown("#### 📊 Forecasting Accuracy by Provider")
                    
                    r2_df = pd.DataFrame(list(provider_r2_scores.items()), 
                                        columns=['Provider', 'R2_Score'])
                    r2_df = r2_df.sort_values('R2_Score', ascending=False)
                    
                    fig = go.Figure(go.Bar(
                        x=r2_df['Provider'],
                        y=r2_df['R2_Score'],
                        marker=dict(color='steelblue'),
                        text=r2_df['R2_Score'].round(3),
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Forecasting Accuracy by Provider<br><sub>(How Predictable Is Safety?)</sub>",
                        xaxis_title="Provider",
                        yaxis_title="R² Score",
                        height=400,
                        xaxis_tickangle=-45
                    )
                    
                    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <p><strong>Interpretation:</strong></p>
                    <p>R² score shows how well we can predict a provider's safety based on release date. 
                    Higher values mean more predictable safety patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Provider Strategy Map
            if provider_size_corr and provider_time_corr:
                st.markdown("#### 🗺️ Provider Strategy Map")
                
                # Combine both correlations and R² scores
                strategy_data = []
                for provider in set(list(provider_size_corr.keys()) + list(provider_time_corr.keys())):
                    if provider in provider_size_corr and provider in provider_time_corr:
                        r2_score_val = provider_r2_scores.get(provider, 0) if provider_r2_scores else 0
                        strategy_data.append({
                            'Provider': provider,
                            'Size Impact': provider_size_corr[provider],
                            'Date Impact': provider_time_corr[provider],
                            'R2_Score': r2_score_val
                        })
                
                if strategy_data:
                    strategy_df = pd.DataFrame(strategy_data)
                    
                    fig = px.scatter(strategy_df, 
                                   x='Size Impact', 
                                   y='Date Impact',
                                   text='Provider',
                                   color='R2_Score',
                                   color_continuous_scale='RdYlGn',
                                   title="Provider Strategy Map<br><sub>(Size vs Time Focus)</sub>",
                                   labels={'Size Impact': 'Model Size Impact →', 
                                          'Date Impact': 'Temporal Improvement →',
                                          'R2_Score': 'R² Score'})
                    
                    fig.update_traces(marker=dict(size=15), textposition='top center')
                    
                    # Add quadrant lines
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_layout(height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="success-box">
                    <h4>🎯 Strategy Quadrants</h4>
                    <ul>
                        <li><strong>Top Right:</strong> Both size and time improve safety (comprehensive approach)</li>
                        <li><strong>Top Left:</strong> Focus on temporal improvements (iterative safety enhancements)</li>
                        <li><strong>Bottom Right:</strong> Focus on model size (scaling for safety)</li>
                        <li><strong>Bottom Left:</strong> Neither factor strongly correlates (other factors dominate)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("Provider information not available for impact analysis")

# ============================================================================
# PAGE 4: ATTACK METHODS DEEP DIVE
# ============================================================================
elif page == "🎯 Attack Methods Deep Dive":
    st.markdown('<h1 class="main-header">🎯 Attack Methods Deep Dive</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>⚔️ Understanding Jailbreak Attacks</h2>
    <p style="font-size: 1.1rem;">
    Each attack method tests a different vulnerability in AI models. Let's explore which attacks 
    are most effective and how models perform against each one.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get attack method columns (scores only - no trials or risks)
    attack_cols = get_attack_score_columns(df)
    
    if len(attack_cols) > 0:
        # Attack effectiveness ranking
        st.markdown("### 📊 Attack Method Effectiveness")
        
        attack_means = df[attack_cols].mean().sort_values(ascending=True)
        
        fig = px.bar(x=attack_means.values, y=attack_means.index,
                    orientation='h',
                    title="Average Safety Score by Attack Method (Lower = More Effective Attack)",
                    labels={'x': 'Average Safety Score', 'y': 'Attack Method'})
        fig.update_traces(marker_color='#d62728')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Most Effective Attacks</h4>
            <p>These attacks achieve the lowest safety scores, indicating they are most successful 
            at bypassing model safeguards.</p>
            </div>
            """, unsafe_allow_html=True)
            
            most_effective = attack_means.head(5)
            for attack, score in most_effective.items():
                st.write(f"**{attack}**: {score:.1f}")
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Least Effective Attacks</h4>
            <p>These attacks achieve higher safety scores, meaning models are generally resistant 
            to these techniques.</p>
            </div>
            """, unsafe_allow_html=True)
            
            least_effective = attack_means.tail(5)
            for attack, score in least_effective.items():
                st.write(f"**{attack}**: {score:.1f}")
        
        # Attack variance analysis
        st.markdown("### 📈 Attack Predictability")
        
        attack_std = df[attack_cols].std().sort_values(ascending=False)
        
        fig = px.bar(x=attack_std.index, y=attack_std.values,
                    title="Attack Method Variance (Higher = Less Predictable)",
                    labels={'x': 'Attack Method', 'y': 'Standard Deviation'})
        fig.update_traces(marker_color='#9467bd')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Interpretation</h4>
        <ul>
            <li><strong>High Variance:</strong> Attack effectiveness varies greatly across models - harder to predict</li>
            <li><strong>Low Variance:</strong> Attack effectiveness is consistent - easier to forecast</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual attack selection
        st.markdown("### 🔬 Detailed Attack Analysis")
        
        selected_attack = st.selectbox("Select an attack method to analyze:", attack_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution
            fig = px.histogram(df, x=selected_attack, nbins=30,
                             title=f"{selected_attack} - Score Distribution",
                             labels={selected_attack: 'Safety Score'})
            fig.update_traces(marker_color='#17becf')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by provider
            if 'Provider' in df.columns:
                fig = px.box(df, x='Provider', y=selected_attack,
                           title=f"{selected_attack} - Performance by Provider",
                           labels={'Provider': 'Provider', selected_attack: 'Safety Score'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Provider data not available")
    
    else:
        st.warning("No attack method data available")

# ============================================================================
# PAGE 5: PROVIDER ANALYSIS
# ============================================================================
elif page == "🏢 Provider Analysis":
    st.markdown('<h1 class="main-header">🏢 Provider Analysis & Radar Plots</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🏭 Company-Specific Safety Patterns</h2>
    <p style="font-size: 1.1rem;">
    Different AI companies have different approaches to safety. Let's explore how providers 
    compare and visualize their models' safety profiles using radar plots.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Provider' in df.columns:
        # Provider statistics
        st.markdown("### 📊 Provider Overview")
        
        score_col = 'Overall_Score' if 'Overall_Score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        
        # Filter out rows with missing provider or score
        df_provider = df[df['Provider'].notna() & df[score_col].notna()].copy()
        
        if len(df_provider) > 0:
            provider_stats = df_provider.groupby('Provider').agg({
                'Model': 'count',
                score_col: 'mean'
            }).round(2)
        else:
            st.warning("No provider data available")
            provider_stats = pd.DataFrame()
        if len(provider_stats) > 0:
            provider_stats.columns = ['Number of Models', 'Average Safety Score']
            provider_stats = provider_stats.sort_values('Average Safety Score', ascending=False)
        else:
            st.warning("No provider statistics available")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(provider_stats, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Key Insights</h4>
            <p>Providers with higher average scores demonstrate stronger overall safety performance.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Provider comparison chart
        st.markdown("### 📊 Provider Safety Comparison")
        
        fig = px.bar(provider_stats.reset_index(), x='Provider', y='Average Safety Score',
                    title="Average Safety Score by Provider",
                    labels={'Provider': 'Provider', 'Average Safety Score': 'Safety Score'})
        fig.update_traces(marker_color='#2ca02c')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar plots by provider
        st.markdown("### 🎯 Provider Radar Plots")
        
        st.markdown("""
        <div class="story-section">
        <p style="font-size: 1.05rem;">
        Radar plots show the safety profile across different attack methods. Each axis represents 
        a different attack, and the distance from the center indicates the safety score (higher = safer).
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select provider
        providers = sorted(df['Provider'].unique())
        selected_provider = st.selectbox("Select a provider to view their models:", providers)
        
        provider_models = df[df['Provider'] == selected_provider]
        
        st.markdown(f"#### {selected_provider} - {len(provider_models)} Models")
        
        # Get attack columns for radar plot (scores only - no trials or risks)
        attack_cols = get_attack_score_columns(df)
        
        if len(attack_cols) > 0:
            # Select which models to show
            num_models = min(5, len(provider_models))
            show_option = st.radio(
                "Show:",
                [f"Top {num_models} Models", f"Bottom {num_models} Models", "All Models", "Select Specific Model"],
                horizontal=True
            )
            
            if show_option == f"Top {num_models} Models":
                score_col = 'Overall_Score' if 'Overall_Score' in df.columns else attack_cols[0]
                models_to_plot = provider_models.nlargest(num_models, score_col)
            elif show_option == f"Bottom {num_models} Models":
                score_col = 'Overall_Score' if 'Overall_Score' in df.columns else attack_cols[0]
                models_to_plot = provider_models.nsmallest(num_models, score_col)
            elif show_option == "All Models":
                models_to_plot = provider_models
            else:  # Select Specific Model
                model_names = provider_models['Model'].tolist()
                selected_model = st.selectbox("Choose a model:", model_names)
                models_to_plot = provider_models[provider_models['Model'] == selected_model]
            
            # Create radar plots
            for idx, row in models_to_plot.iterrows():
                model_name = row['Model']
                
                # Prepare data for radar plot - handle NaN values
                values = [row[col] if pd.notna(row[col]) else 0 for col in attack_cols if col in row.index]
                
                # Clean attack names for display (remove '_Score' suffix)
                attack_names = [col.replace('_Score', '') for col in attack_cols]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=attack_names + [attack_names[0]],
                    fill='toself',
                    name=model_name,
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title=f"{model_name} - Safety Profile",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show model details
                with st.expander(f"📋 {model_name} - Detailed Scores"):
                    attack_names = [col.replace('_Score', '') for col in attack_cols if col in row.index]
                    attack_values = [row[col] if pd.notna(row[col]) else 0 for col in attack_cols if col in row.index]
                    score_data = pd.DataFrame({
                        'Attack Method': attack_names,
                        'Safety Score': attack_values
                    }).sort_values('Safety Score', ascending=False)
                    st.dataframe(score_data, use_container_width=True, hide_index=True)
        
        # Provider comparison across attacks
        st.markdown("### 🔍 Provider Performance Across Attacks")
        
        if len(attack_cols) > 0:
            selected_attacks = st.multiselect(
                "Select attack methods to compare:",
                attack_cols,
                default=attack_cols[:5] if len(attack_cols) >= 5 else attack_cols
            )
            
            if selected_attacks:
                df_prov_clean = df[df['Provider'].notna()].copy()
                provider_attack_data = df_prov_clean.groupby('Provider')[selected_attacks].mean()
                
                # Clean attack names for display (remove '_Score' suffix)
                selected_attack_names = [col.replace('_Score', '') for col in selected_attacks]
                
                fig = go.Figure()
                
                for provider in provider_attack_data.index:
                    values = provider_attack_data.loc[provider, selected_attacks].tolist()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=selected_attack_names + [selected_attack_names[0]],
                        fill='toself',
                        name=provider
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title="Provider Comparison - Average Performance",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Provider information not available in the dataset")

# ============================================================================
# PAGE 6: FORECASTING MODELS
# ============================================================================
elif page == "🔮 Forecasting Models":
    st.markdown('<h1 class="main-header">🔮 Forecasting Models</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🤖 Machine Learning Models</h2>
    <p style="font-size: 1.1rem;">
    We trained 5 different machine learning models to predict AI safety scores. Let's compare 
    their performance and understand which approaches work best.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    if analysis_data and 'model_results' in analysis_data:
        model_results = analysis_data['model_results']
        
        # Model performance comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        results_df = pd.DataFrame(model_results).T
        results_df = results_df.sort_values('R2_Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### R² Score (Higher is Better)")
            fig = px.bar(results_df.reset_index(), x='index', y='R2_Score',
                        title="R² Score by Model",
                        labels={'index': 'Model', 'R2_Score': 'R² Score'})
            fig.update_traces(marker_color='#2ca02c')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### RMSE (Lower is Better)")
            fig = px.bar(results_df.reset_index(), x='index', y='RMSE',
                        title="RMSE by Model",
                        labels={'index': 'Model', 'RMSE': 'RMSE'})
            fig.update_traces(marker_color='#d62728')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Model Metrics")
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📖 Understanding the Metrics</h4>
        <ul>
            <li><strong>R² Score:</strong> Proportion of variance explained (0-1, higher is better)</li>
            <li><strong>RMSE:</strong> Root Mean Square Error (lower is better, measured in score points)</li>
            <li><strong>MAE:</strong> Mean Absolute Error (average prediction error)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Best model highlight
        best_model = results_df.index[0]
        best_r2 = results_df.loc[best_model, 'R2_Score']
        best_rmse = results_df.loc[best_model, 'RMSE']
        
        st.markdown(f"""
        <div class="success-box">
        <h3>🏆 Best Performing Model</h3>
        <p style="font-size: 1.1rem;">
        <strong>{best_model}</strong> achieved the best performance with an R² score of 
        <strong>{best_r2:.3f}</strong> and RMSE of <strong>{best_rmse:.2f}</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show actual vs predicted plots if predictions available
        if predictions_data:
            st.markdown("### 📊 Actual vs Predicted Scores")
            
            # Create tabs for each model
            model_names = list(predictions_data.keys())
            tabs = st.tabs(model_names)
            
            for tab, model_name in zip(tabs, model_names):
                with tab:
                    pred_data = predictions_data[model_name]
                    y_test = pred_data['y_test']
                    y_pred = pred_data['y_pred']
                    
                    fig = go.Figure()
                    
                    # Scatter plot
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            size=10,
                            color=y_pred,
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Predicted Score")
                        )
                    ))
                    
                    # Perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"{model_name} - Actual vs Predicted",
                        xaxis_title="Actual Safety Score",
                        yaxis_title="Predicted Safety Score",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance Analysis
        st.markdown("### 🎯 Feature Importance Analysis")
        
        st.markdown("""
        <div class="story-section">
        <p style="font-size: 1.05rem;">
        Which feature gives us more predictive power? Let's analyze the importance of 
        <strong>Model Size</strong> vs <strong>Release Date</strong> across all trained models.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance for tree-based models
        if all_models:
            st.markdown("#### 🌲 Feature Importance from Tree-Based Models")
            
            tree_models = {}
            feature_names = ['Model Size (B)', 'Days Since 2020']
            
            # Collect feature importances from tree-based models
            for model_name in ['Random Forest', 'Gradient Boosting']:
                if model_name in all_models:
                    model = all_models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        tree_models[model_name] = model.feature_importances_
            
            if tree_models:
                # Create comparison chart
                fig = go.Figure()
                
                colors = {'Random Forest': '#2ca02c', 'Gradient Boosting': '#9467bd'}
                
                for model_name, importances in tree_models.items():
                    fig.add_trace(go.Bar(
                        name=model_name,
                        x=feature_names,
                        y=importances,
                        text=[f'{imp:.1%}' for imp in importances],
                        textposition='auto',
                        marker_color=colors.get(model_name, '#1f77b4')
                    ))
                
                fig.update_layout(
                    title="Feature Importance Comparison (Tree-Based Models)",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    barmode='group',
                    height=450,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis table
                importance_data = []
                for model_name, importances in tree_models.items():
                    importance_data.append({
                        'Model': model_name,
                        'Model Size Importance': f'{importances[0]:.1%}',
                        'Release Date Importance': f'{importances[1]:.1%}',
                        'Dominant Feature': feature_names[np.argmax(importances)]
                    })
                
                st.markdown("**Feature Importance Summary:**")
                st.dataframe(pd.DataFrame(importance_data), use_container_width=True, hide_index=True)
                
                # Overall interpretation
                avg_size_imp = np.mean([imp[0] for imp in tree_models.values()])
                avg_date_imp = np.mean([imp[1] for imp in tree_models.values()])
                
                dominant = "Model Size" if avg_size_imp > avg_date_imp else "Release Date"
                ratio = avg_size_imp / avg_date_imp if avg_size_imp > avg_date_imp else avg_date_imp / avg_size_imp
                
                st.markdown(f"""
                <div class="success-box">
                <h4>🎯 Tree-Based Models Consensus</h4>
                <p><strong>{dominant}</strong> is consistently more important across tree-based models.</p>
                <p style="font-size: 0.95rem; margin-top: 0.5rem;">
                Average importance: Model Size <strong>{avg_size_imp:.1%}</strong> vs Release Date <strong>{avg_date_imp:.1%}</strong><br>
                {dominant} is <strong>{ratio:.1f}x</strong> more important on average.
                </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Tree-based models not available for feature importance analysis")
        
        # Linear model coefficients analysis
        st.markdown("#### 📐 Linear Model Coefficients Analysis")
        
        if all_models:
            linear_models = {}
            
            for model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                if model_name in all_models:
                    model = all_models[model_name]
                    if hasattr(model, 'coef_'):
                        # Use absolute values to show importance regardless of direction
                        linear_models[model_name] = np.abs(model.coef_)
            
            if linear_models:
                # Create comparison chart
                fig = go.Figure()
                
                colors = {'Linear Regression': '#1f77b4', 'Ridge Regression': '#ff7f0e', 'Lasso Regression': '#d62728'}
                
                for model_name, coefs in linear_models.items():
                    # Normalize to percentages
                    coef_pct = coefs / coefs.sum()
                    fig.add_trace(go.Bar(
                        name=model_name,
                        x=feature_names,
                        y=coef_pct,
                        text=[f'{c:.1%}' for c in coef_pct],
                        textposition='auto',
                        marker_color=colors.get(model_name, '#1f77b4')
                    ))
                
                fig.update_layout(
                    title="Feature Importance from Linear Models<br><sub>(Normalized Absolute Coefficients)</sub>",
                    xaxis_title="Feature",
                    yaxis_title="Relative Importance",
                    barmode='group',
                    height=450,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                <h4>📊 Linear Models Interpretation</h4>
                <p>Linear models show the relative weight each feature has in the prediction equation. 
                Higher values indicate stronger influence on the predicted safety score.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Correlation-based importance
        st.markdown("#### 📊 Correlation-Based Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'correlation_size_safety' in analysis_data and analysis_data['correlation_size_safety']:
                corr_size = abs(analysis_data['correlation_size_safety'])
                st.markdown(f"""
                <div class="metric-card">
                <h3>{corr_size:.3f}</h3>
                <p>Model Size Correlation</p>
                <small style="opacity: 0.8;">Absolute correlation with safety</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'correlation_time_safety' in analysis_data and analysis_data['correlation_time_safety']:
                corr_time = abs(analysis_data['correlation_time_safety'])
                st.markdown(f"""
                <div class="metric-card">
                <h3>{corr_time:.3f}</h3>
                <p>Release Date Correlation</p>
                <small style="opacity: 0.8;">Absolute correlation with safety</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Compare correlations
        if 'correlation_size_safety' in analysis_data and 'correlation_time_safety' in analysis_data:
            corr_size = abs(analysis_data['correlation_size_safety']) if analysis_data['correlation_size_safety'] else 0
            corr_time = abs(analysis_data['correlation_time_safety']) if analysis_data['correlation_time_safety'] else 0
            
            # Create comparison chart
            fig = go.Figure(go.Bar(
                x=['Model Size', 'Release Date'],
                y=[corr_size, corr_time],
                marker=dict(color=['#1f77b4', '#ff7f0e']),
                text=[f'{corr_size:.3f}', f'{corr_time:.3f}'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Feature Correlation Comparison<br><sub>(Higher = More Predictive Power)</sub>",
                xaxis_title="Feature",
                yaxis_title="Absolute Correlation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine which is more important
            if corr_size > corr_time:
                stronger = "Model Size"
                weaker = "Release Date"
                ratio = corr_size / corr_time if corr_time > 0 else float('inf')
            else:
                stronger = "Release Date"
                weaker = "Model Size"
                ratio = corr_time / corr_size if corr_size > 0 else float('inf')
            
            st.markdown(f"""
            <div class="success-box">
            <h4>🎯 Key Finding</h4>
            <p><strong>{stronger}</strong> has stronger predictive power than {weaker}.</p>
            <p style="font-size: 0.95rem; margin-top: 0.5rem;">
            {stronger} is <strong>{ratio:.1f}x</strong> more correlated with safety scores, 
            making it the more important feature for prediction.
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Correlation insights
        st.markdown("### 🔍 Detailed Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'correlation_size_safety' in analysis_data and analysis_data['correlation_size_safety']:
                corr_size = analysis_data['correlation_size_safety']
                st.markdown(f"""
                <div class="insight-box">
                <h4>📏 Model Size Impact</h4>
                <p><strong>Correlation:</strong> {corr_size:.4f}</p>
                <p>{'Larger models show moderate positive correlation with safety' if corr_size > 0.3 else 'Size shows some correlation with safety' if corr_size > 0.1 else 'Size shows weak correlation with safety'}</p>
                <p style="font-size: 0.9rem; color: #666;">
                This suggests model size {'may be one of several factors' if corr_size > 0.3 else 'could play a role alongside other factors'} influencing safety performance.
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'correlation_time_safety' in analysis_data and analysis_data['correlation_time_safety']:
                corr_time = analysis_data['correlation_time_safety']
                st.markdown(f"""
                <div class="insight-box">
                <h4>📅 Temporal Trends</h4>
                <p><strong>Correlation:</strong> {corr_time:.4f}</p>
                <p>{'Newer models show moderate positive correlation with safety' if corr_time > 0.3 else 'Some improvement observed over time' if corr_time > 0.1 else 'No clear temporal trend observed'}</p>
                <p style="font-size: 0.9rem; color: #666;">
                {'This suggests gradual improvements in safety practices' if corr_time > 0.1 else 'Safety improvements may depend more on other factors than release timing'}.
                </p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.warning("Model results not available. Please run the analysis notebook first.")
        
        st.markdown("""
        <div class="story-section">
        <h3>📚 Models We Trained</h3>
        <ul style="font-size: 1.05rem;">
            <li><strong>Linear Regression:</strong> Simple baseline model</li>
            <li><strong>Ridge Regression:</strong> Linear model with L2 regularization</li>
            <li><strong>Lasso Regression:</strong> Linear model with L1 regularization</li>
            <li><strong>Random Forest:</strong> Ensemble of decision trees</li>
            <li><strong>Gradient Boosting:</strong> Sequential ensemble method</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 7: RADAR PLOTS GALLERY
# ============================================================================
elif page == "📈 Radar Plots Gallery":
    st.markdown('<h1 class="main-header">📈 Radar Plots Gallery</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🎨 Visual Safety Profiles</h2>
    <p style="font-size: 1.1rem;">
    Explore radar plots for all models in the dataset. Each plot shows how a model performs 
    across different attack methods, making it easy to identify strengths and weaknesses.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get attack columns (scores only - no trials or risks)
    attack_cols = get_attack_score_columns(df)
    
    if len(attack_cols) > 0:
        # Filter options
        st.markdown("### 🔍 Filter Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_option = st.selectbox(
                "View:",
                ["Top 10 Models", "Bottom 10 Models", "Top 20 Models", "Bottom 20 Models", "All Models", "Custom Selection"]
            )
        
        with col2:
            if 'Provider' in df.columns:
                providers = ['All'] + sorted(df['Provider'].unique().tolist())
                selected_provider_filter = st.selectbox("Filter by Provider:", providers)
            else:
                selected_provider_filter = 'All'
        
        with col3:
            sort_by = st.selectbox(
                "Sort by:",
                ['Overall_Score'] + attack_cols if 'Overall_Score' in df.columns else attack_cols
            )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_provider_filter != 'All' and 'Provider' in df.columns:
            filtered_df = filtered_df[filtered_df['Provider'] == selected_provider_filter]
        
        # Select models based on view option
        if view_option == "Top 10 Models":
            models_to_show = filtered_df.nlargest(10, sort_by)
        elif view_option == "Bottom 10 Models":
            models_to_show = filtered_df.nsmallest(10, sort_by)
        elif view_option == "Top 20 Models":
            models_to_show = filtered_df.nlargest(20, sort_by)
        elif view_option == "Bottom 20 Models":
            models_to_show = filtered_df.nsmallest(20, sort_by)
        elif view_option == "All Models":
            models_to_show = filtered_df.sort_values(sort_by, ascending=False)
        else:  # Custom Selection
            model_names = filtered_df['Model'].tolist()
            selected_models = st.multiselect("Select models:", model_names, default=model_names[:5] if len(model_names) >= 5 else model_names)
            models_to_show = filtered_df[filtered_df['Model'].isin(selected_models)]
        
        st.markdown(f"### Showing {len(models_to_show)} Models")
        
        # Display options
        display_mode = st.radio("Display Mode:", ["Individual Plots", "Comparison View"], horizontal=True)
        
        if display_mode == "Individual Plots":
            # Show individual radar plots
            for idx, row in models_to_show.iterrows():
                model_name = row['Model']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create radar plot - handle NaN values
                    values = [row[col] if pd.notna(row[col]) else 0 for col in attack_cols if col in row.index]
                    
                    # Clean attack names for display (remove '_Score' suffix)
                    attack_names = [col.replace('_Score', '') for col in attack_cols]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=attack_names + [attack_names[0]],
                        fill='toself',
                        name=model_name,
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        showlegend=True,
                        title=f"{model_name}",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Model Info")
                    if 'Provider' in row.index:
                        st.write(f"**Provider:** {row['Provider']}")
                    if 'Size' in row.index:
                        st.write(f"**Size:** {row['Size']}B")
                    if 'Overall_Score' in row.index:
                        st.write(f"**Overall Score:** {row['Overall_Score']:.1f}")
                    
                    # Top 3 strengths
                    attack_scores = {col: row[col] for col in attack_cols if col in row.index and pd.notna(row[col])}
                    if len(attack_scores) > 0:
                        top_3 = sorted(attack_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    else:
                        top_3 = []
                    if len(top_3) > 0:
                        st.markdown("**Top Strengths:**")
                        for attack, score in top_3:
                            attack_name = attack.replace('_Score', '')
                            st.write(f"• {attack_name}: {score:.1f}")
                
                st.markdown("---")
        
        else:  # Comparison View
            # Show multiple models on one plot
            num_to_compare = min(5, len(models_to_show))
            st.info(f"Comparing top {num_to_compare} models (too many lines make the plot hard to read)")
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Clean attack names for display (remove '_Score' suffix)
            attack_names = [col.replace('_Score', '') for col in attack_cols]
            
            for i, (idx, row) in enumerate(models_to_show.head(num_to_compare).iterrows()):
                model_name = row['Model']
                values = [row[col] if pd.notna(row[col]) else 0 for col in attack_cols if col in row.index]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=attack_names + [attack_names[0]],
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Model Comparison",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison table
            st.markdown("### Comparison Table")
            comparison_cols = ['Model'] + ([col for col in ['Provider', 'Size', 'Overall_Score'] if col in models_to_show.columns])
            st.dataframe(models_to_show[comparison_cols].head(num_to_compare), use_container_width=True, hide_index=True)
    
    else:
        st.warning("No attack method data available for radar plots")

# ============================================================================
# PAGE 8: MAKE PREDICTIONS
# ============================================================================
elif page == "🎲 Make Predictions":
    st.markdown('<h1 class="main-header">🎲 Make Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-section">
    <h2>🔮 Forecast Future AI Safety</h2>
    <p style="font-size: 1.1rem;">
    Use our trained models to predict the safety performance of hypothetical future AI models. 
    Simply input the model size and release date to get predictions.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models if available (try data folder first, then root)
    best_model = None
    scaler = None
    
    for folder in ['data/', '']:
        try:
            with open(f'{folder}best_model.pkl', 'rb') as f:
                best_model = pickle.load(f)
            with open(f'{folder}scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            break
        except:
            continue
    
    models_available = best_model is not None and scaler is not None
    
    if not models_available:
        st.warning("⚠️ Trained models not found. Please run the analysis notebook first.")
    
    if models_available:
        st.markdown("### 🎯 Input Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_size = st.slider(
                "Model Size (Billions of Parameters)",
                min_value=0.5,
                max_value=500.0,
                value=7.0,
                step=0.5
            )
        
        with col2:
            release_date = st.date_input(
                "Release Date",
                value=datetime.now(),
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 12, 31)
            )
        
        # Calculate days since 2020
        days_since_2020 = (release_date - datetime(2020, 1, 1).date()).days
        
        if st.button("🔮 Predict Safety Score", type="primary"):
            # Prepare input
            input_data = np.array([[model_size, days_since_2020]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = best_model.predict(input_scaled)[0]
            
            # Clamp prediction to valid range and add warning if needed
            prediction_clamped = max(0, min(100, prediction))
            out_of_range = prediction < 0 or prediction > 100
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            if out_of_range:
                st.markdown(f"""
                <div class="warning-box">
                <h4>⚠️ Prediction Out of Range</h4>
                <p>The model predicted a score of <strong>{prediction:.1f}</strong>, which is outside the valid range (0-100).</p>
                <p><strong>Adjusted to:</strong> {prediction_clamped:.1f}</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                <em>Note: Predictions >100% may indicate that the benchmarks should be redefined or that the model 
                is extrapolating beyond its training data. Consider this prediction with caution.</em>
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>{prediction_clamped:.1f}</h3>
                <p>Predicted Safety Score</p>
                {'<small style="opacity: 0.8;">(Clamped from ' + f'{prediction:.1f}' + ')</small>' if out_of_range else ''}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <h3>{model_size:.1f}B</h3>
                <p>Model Size</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                <h3>{release_date.strftime('%Y-%m-%d')}</h3>
                <p>Release Date</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation (use clamped value)
            if prediction_clamped >= 90:
                interpretation = "🟢 **Excellent** - Highly resistant to attacks"
                box_class = "success-box"
            elif prediction_clamped >= 75:
                interpretation = "🟡 **Good** - Strong safety characteristics"
                box_class = "insight-box"
            elif prediction_clamped >= 60:
                interpretation = "🟠 **Moderate** - Some vulnerabilities present"
                box_class = "warning-box"
            else:
                interpretation = "🔴 **Concerning** - Significant vulnerabilities"
                box_class = "warning-box"
            
            st.markdown(f"""
            <div class="{box_class}">
            <h4>Safety Interpretation</h4>
            <p style="font-size: 1.1rem;">{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison with existing models
            st.markdown("### 📊 Comparison with Existing Models")
            
            score_col = 'Overall_Score' if 'Overall_Score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
            
            # Find similar models
            df_with_pred = df.copy()
            df_with_pred['score_diff'] = abs(df_with_pred[score_col] - prediction)
            similar_models = df_with_pred.nsmallest(5, 'score_diff')
            
            st.markdown("**Models with Similar Predicted Safety Scores:**")
            display_cols = ['Model'] + [col for col in similar_models.columns if col in ['Provider', 'Size', score_col]]
            st.dataframe(similar_models[display_cols], use_container_width=True, hide_index=True)
            
            # Visualize prediction
            fig = go.Figure()
            
            # Add histogram of existing scores
            fig.add_trace(go.Histogram(
                x=df[score_col],
                name='Existing Models',
                opacity=0.7,
                marker_color='#1f77b4'
            ))
            
            # Add prediction line
            fig.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Prediction: {prediction:.1f}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title="Your Prediction vs Existing Models",
                xaxis_title="Safety Score",
                yaxis_title="Count",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown("""
        <div class="insight-box">
        <h4>📚 How Predictions Work</h4>
        <p>Our forecasting models use:</p>
        <ul>
            <li><strong>Model Size:</strong> Number of parameters (in billions)</li>
            <li><strong>Release Date:</strong> When the model was released</li>
        </ul>
        <p>These features are used to predict the overall safety score and individual attack method scores.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛡️ AI Safety Forecasting Dashboard | Built for Apart Research Hackathon</p>
    <p>Data from HydroX Leaderboard | Track 1: AI Capability Forecasting & Timeline Models</p>
</div>
""", unsafe_allow_html=True)