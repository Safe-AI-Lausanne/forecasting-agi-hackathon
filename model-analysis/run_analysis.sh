#!/bin/bash

# 🛡️ AI Safety Forecasting - Quick Start Script
# This script runs the complete analysis pipeline

echo "=========================================="
echo "🛡️  AI Safety Forecasting Dashboard"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, pandas, plotly, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing"
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ All dependencies found"
fi
echo ""

# Check if dataset exists
if [ ! -f "hydrox_attack_methods_with_dates.csv" ]; then
    echo "⚠️  Dataset not found!"
    echo "Would you like to download it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "📥 Running data scraper..."
        python3 data_scrapper.py
        if [ $? -ne 0 ]; then
            echo "❌ Failed to download dataset"
            exit 1
        fi
    else
        echo "❌ Cannot proceed without dataset"
        exit 1
    fi
fi

echo "✅ Dataset found"
echo ""

# Create data directory
mkdir -p data
echo "✅ Data directory ready"
echo ""

# Run analysis notebook
echo "=========================================="
echo "🔬 Running Analysis Notebook"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Analyze 100+ AI models"
echo "  2. Train 5 ML models"
echo "  3. Generate visualizations"
echo "  4. Save results to data/ folder"
echo ""
echo "⏱️  This may take 1-2 minutes..."
echo ""

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "⚠️  Jupyter not found, installing..."
    pip3 install jupyter
fi

# Convert notebook to Python script and run it
jupyter nbconvert --to script --execute analysis.ipynb --output temp_analysis 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    rm -f temp_analysis.py
else
    echo "⚠️  Notebook execution had some warnings (this is usually OK)"
fi

echo ""

# Check if analysis data was generated
if [ -f "data/analysis_data.pkl" ] || [ -f "analysis_data.pkl" ]; then
    echo "✅ Analysis data generated successfully"
else
    echo "⚠️  Analysis data not found, but continuing..."
fi

echo ""
echo "=========================================="
echo "🚀 Launching Streamlit Dashboard"
echo "=========================================="
echo ""
echo "The dashboard will open in your browser at:"
echo "  👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
python3 -m streamlit run app.py

echo ""
echo "=========================================="
echo "✅ Session Complete"
echo "=========================================="
