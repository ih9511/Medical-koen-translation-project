#!/bin/bash

# 초기 세팅
# echo "Cloning the repository..."
# git clone https://github.com/ih9511/Medical-koen-translation-project.git
# cd Medical-koen-translation-project

echo "Setting up virtual environment"
python3.10 -m venv .venv

# 운영체제 감지
OS_TYPE=$(uname)
if [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "Darwin" ]]; then
    echo "Detected Linux/MacOS - Using bin/activate"
    source .venv/bin/activate
elif [[ "$OS_TYPE" == "MINGW64_NT"* || "$OS_TYPE" == "CYGWIN_NT"* ]]; then
    echo "Detected Windows (Git Bash) - Using Scripts/activate"
    source .venv/Scripts/activate
else
    echo "Unsupported OS detected! Please activate venv manually."
    exit 1
fi

echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Reinstalling PyTorch with CUDA 11.8 support..."
pip3 uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face 로그인
echo "Logging into Hugging Face..."
source .env
huggingface-cli login --token $HUGGINGFACE_TOKEN

# 데이터 전처리
echo "Running data preprocessing..."
python3 src/preprocessing.py

# 모델 파인튜닝
echo "Do you want to fine-tune the base model? (y/n)"
read -r finetune_choice
if [ "$finetune_choice" == "y" ]; then
    echo "Running model fine-tuning..."
    python3 src/finetune.py
else
    echo "Skipping model fine-tuning."
fi

# 모델 추론 및 평가
echo "Do you want to evaluate the fine-tuned model? (y/n)"
read -r evaluate_choice
if [ "$evaluate_choice" == "y" ]; then
    echo "Evaluating fine-tuned model..."
    python3 src/evaluation.py
else
    echo "Skipping model evaluation."
fi

# 파인튜닝된 모델 업로드
echo "Do you want to upload the fine-tuned model to Hugging Face? (y/n)"
read -r upload_choice
if [ "$upload_choice" == "y" ]; then
    echo "Uploading fine-tuned model to Hugging Face..."
    python3 src/upload_model.py
else
    echo "Skipping model upload."
fi

echo "All processes completed."