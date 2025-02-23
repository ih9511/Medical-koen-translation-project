"""
evaluation.py

이 모듈은 번역 모델의 성능을 평가하는 기능을 제공합니다.
- BLEU 점수를 계산하여 번역 모델 성능 평가
- 모델을 로드하고, 테스트 데이터셋을 기반으로 성능 측정
"""
import pandas as pd
import evaluate

from typing import Dict
        

def evaluate_model(data_file_name: str) -> Dict:
    """
    모델이 번역한 결과를 활용하여 METEOR, BLEU 점수를 계산합니다.
    
    :parameter data_file_name: 번역한 결과가 담긴 파일 위치
    :return: METEOR score, BLEU score
    """
    meteor = evaluate.load('meteor')
    bleu = evaluate.load('bleu')
    
    df = pd.read_csv(data_file_name)
    predictions = df['translated'].tolist()
    references = df['output'].tolist()

    meteor_score = meteor.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    return {'meteor': meteor_score, 'bleu': bleu_score}

if __name__ == '__main__':
    dataset_path = './data/finetuned_result/results.csv'
    score = evaluate_model(dataset_path)
    print(f"METEOR: {score['meteor']}, BLEU: {score['bleu']}")