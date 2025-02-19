"""
upload_model.py

이 모듈은 학습된 LLM 모델을 Hugging Face Model Hub에 업로드하는 기능을 제공합니다.
- 저장된 모델과 토크나이저를 불러와 Hugging Face에 업로드
- Hugging Face CLI 인증 필요
- 모델 업로드 후 웹에서 확인 가능
"""
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv

load_dotenv(override=True)
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN_RUNPOD')

def upload_model_to_huggingface(model_path: str, repo_id: str, hf_token: str) -> None:
    """
    로컬에 저장된 모델을 Hugging Face Model Hub에 업로드합니다.
    
    :parameter model_path: 로컬에 저장된 모델 디렉토리 경로
    :parameter repo_id: Hugging face model repository ID
    :parameter hf_token: Hugging face API Access Token
    """
    
    # Hugging face 로그임
    HfFolder.save_token(hf_token)
    
    # 모델, 토크나이저 불러오기
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 모델 업로드
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    
    logging.warning('Model upload complete!')
    
if __name__ == '__main__':
    model_path = '../models/gemma2-2b_finetuned'
    repo_id = 'ih9511/gemma2-2b_medical_translation_en_ko'
    hf_access_token = HUGGINGFACE_TOKEN
    
    upload_model_to_huggingface(model_path, repo_id, hf_access_token)