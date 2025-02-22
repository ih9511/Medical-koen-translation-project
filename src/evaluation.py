"""
evaluation.py

이 모듈은 번역 모델의 성능을 평가하는 기능을 제공합니다.
- BLEU 점수를 계산하여 번역 모델 성능 평가
- 모델을 로드하고, 테스트 데이터셋을 기반으로 성능 측정
"""
import pandas as pd
import evaluate
import logging
import re

from inference import load_model, translate_text
        

def evaluate_translation(model, tokenizer, dataset_path: str):
    """
    번역 모델의 성능을 BLEU 점수를 통해 평가하는 함수 (Hugging Face evaluate 사용)
    
    :parameter model: 로드된 번역 모델
    :parameter tokenizer: 해당 모델의 토크나이저
    :parameter dataset_path: 평가에 사용할 CSV 데이터셋 경로
    """
    
    df = pd.read_csv(dataset_path)
    logging.warning('Inferencing...')
    references = df['output'].tolist()
    hypotheses = [translate_text(model, tokenizer, './data/processed_data/test_processed.csv') for text in df['input'].tolist()]
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
    base_model_name = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    finetune_model_name = 'ih9511/llama-3-Korean-8B_koen_medical_translation'
    dataset_path = './data/processed_data/test_processed.csv'
    
    model, tokenizer = load_model(base_model_name, finetune_model_name)
    # evaluation_results = evaluate_translation(model, tokenizer, dataset_path)
    translate_text(model, tokenizer, dataset_path)