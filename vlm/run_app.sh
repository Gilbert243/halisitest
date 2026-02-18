#!/bin/bash
# Startup script for Halisi Cosmetics Hair Analysis App

echo "🚀 Starting Halisi Cosmetics Hair Analysis App..."
echo ""

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv env && source env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source env/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if model file exists
if [ ! -f "models/model_mobilenetv2_150_320x320.pth" ]; then
    echo "⚠️  Warning: Model file not found at models/model_mobilenetv2_150_320x320.pth"
    echo "The app may not work properly without the model file."
fi

echo "✅ Launching Streamlit app..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo ""

# Run streamlit
streamlit run app.py
