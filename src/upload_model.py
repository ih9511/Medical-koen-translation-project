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
from peft import PeftModel

load_dotenv(override=True)
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def merge_and_upload_model_to_huggingface(base_model_name: str, adapter_model_path: str, repo_id: str, hf_token: str) -> None:
    """
    로컬에 저장된 모델을 Hugging Face Model Hub에 업로드합니다.
    
    :parameter base_model_name: 원본 사전학습 모델
    :parameter model_path: 로컬에 저장된 모델 디렉토리 경로
    :parameter repo_id: Hugging face model repository ID
    :parameter hf_token: Hugging face API Access Token
    """
    
    # Hugging face 로그인
    HfFolder.save_token(hf_token)
    
    # 원본 모델, 토크나이저 불러오기
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # LoRA 어댑터 로드 및 병합
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model = model.merge_and_unload()
    
    # 병합된 모델 저장
    model.save_pretrained(adapter_model_path, safe_serialization=False)
    tokenizer.save_pretrained(adapter_model_path)
    
    # 병합된 모델 업로드
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    
    logging.warning('Model upload complete!')
    
if __name__ == '__main__':
    base_model = 'google/gemma-2-2b'
    model_path = '../models/gemma2-2b_finetuned'
    repo_id = 'ih9511/gemma2-2b_medical_translation_en_ko'
    hf_access_token = HUGGINGFACE_TOKEN
    
    merge_and_upload_model_to_huggingface(base_model, model_path, repo_id, hf_access_token)