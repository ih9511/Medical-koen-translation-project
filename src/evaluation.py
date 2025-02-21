"""
evaluation.py

이 모듈은 번역 모델의 성능을 평가하는 기능을 제공합니다.
- BLEU 점수를 계산하여 번역 모델 성능 평가
- 모델을 로드하고, 테스트 데이터셋을 기반으로 성능 측정
"""

import os
import torch
import pandas as pd
import evaluate
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Tuple


def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Hugging Face Hub에서 사전학습된 번역 모델을 로드합니다.
    
    :parameter model_name: 불러올 모델의 HuggingFace 저장소 경로
    :return: 로드된 모델과 토크나이저
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.warning('Model and tokenizer load complete!')
    
    return model, tokenizer

def translate_text(model, tokenizer, text: str, max_length: int = 128) -> str:
    """
    입력 텍스트를 번역하는 함수
    
    :parameter model: 로드된 번역 모델
    :parameter tokenizer: 해당 모델의 토크나이저
    :parameter text: 번역할 입력 문장
    :parameter max_length: 번역된 문장의 최대 길이
    :return: 번역된 텍스트
    """
    # 프롬프트
    prompt = (
        "Translate the following medical text from English to Korean with precise medical terminology:\n\n"
        f"English: {text}\n"
        "Korean:"
    )
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            # do_sample=True,
            # temperature=0.9,
            # top_p=0.9,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_translation(model, tokenizer, dataset_path: str):
    """
    번역 모델의 성능을 BLEU 점수를 통해 평가하는 함수 (Hugging Face evaluate 사용)
    
    :parameter model: 로드된 번역 모델
    :parameter tokenizer: 해당 모델의 토크나이저
    :parameter dataset_path: 평가에 사용할 CSV 데이터셋 경로
    """
    
    df = pd.read_csv(dataset_path)
    references = df['output'].tolist()
    hypotheses = [translate_text(model, tokenizer, text) for text in df['input'].tolist()]
    logging.warning('Inference complete!')
    
    for i in range(10):
        print(f'hypothese: {hypotheses[i]}')
        print(f'references: {references[i]}')
        print('\n')
    
    # BLEU 평가
    logging.warning('BLEU evaluation start')
    bleu_metric = evaluate.load('bleu')
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=[[ref] for ref in references])
    
    logging.warning('Evaluation Complete!')
    logging.warning(f"BLEU: {bleu_score['bleu']}")
    
    return {"BLEU": bleu_score['bleu']}


if __name__ == '__main__':
    model_name = 'ih9511/gemma2-2b_medical_translation_en_ko'
    dataset_path = './data/processed_data/test_processed.csv'
    
    model, tokenizer = load_model(model_name)
    evaluation_results = evaluate_translation(model, tokenizer, dataset_path)