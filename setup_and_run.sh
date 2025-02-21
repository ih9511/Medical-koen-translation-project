#!/bin/bash

set -e

echo "🐍 Setting up virtual environment..."

# 운영체제 감지
OS_TYPE=$(uname)
if [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "Darwin" ]]; then
    python3.10 -m venv .venv || { echo "❌ Failed to create virtual environment!"; exit 1; }
    echo "🔹 Detected Linux/MacOS - Using bin/activate"
    source .venv/bin/activate
    echo "✅ Virtual environment activated!"
elif [[ "$OS_TYPE" == "MINGW64_NT"* || "$OS_TYPE" == "CYGWIN_NT"* ]]; then
    python -m venv .venv || { echo "❌ Failed to create virtual environment!"; exit 1; }
    echo "🔹 Detected Windows (Git Bash) - Using Scripts/activate"
    source .venv/Scripts/activate
    echo "✅ Virtual environment activated!"
else
    echo "❌ Unsupported OS detected! Please activate venv manually."
    exit 1
fi

echo "🐍 Using Python: $(which python)"

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🔥 Reinstalling PyTorch with CUDA 11.8 support..."
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd src

# 데이터 전처리
echo "Running data preprocessing..."
python ./preprocessing.py

# 모델 파인튜닝 여부 확인
echo "❓ Do you want to fine-tune the base model? (y/n)"
read -r finetune_choice
if [ "$finetune_choice" == "y" ]; then
    echo "🚀 Running model fine-tuning..."
    python finetune.py
else
    echo "⏭ Skipping model fine-tuning."
fi

# 모델 평가 여부 확인
echo "❓ Do you want to evaluate the fine-tuned model? (y/n)"
read -r evaluate_choice
if [ "$evaluate_choice" == "y" ]; then
    echo "📈 Evaluating fine-tuned model..."
    python evaluation.py
else
    echo "⏭ Skipping model evaluation."
fi

# 모델 업로드 여부 확인
echo "❓ Do you want to upload the fine-tuned model to Hugging Face? (y/n)"
read -r upload_choice
if [ "$upload_choice" == "y" ]; then
    echo "☁️ Uploading fine-tuned model to Hugging Face..."
    python upload_model.py
else
    echo "⏭ Skipping model upload."
fi

echo "🎉 All processes completed successfully!"