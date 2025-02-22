"""
evaluation.py

이 모듈은 번역 모델의 성능을 평가하는 기능을 제공합니다.
- BLEU 점수를 계산하여 번역 모델 성능 평가
- 모델을 로드하고, 테스트 데이터셋을 기반으로 성능 측정
"""
import pandas as pd
import evaluate
import logging
        

# def evaluate_translation(dataset_path: str):
#     """
#     번역 모델의 성능을 BLEU 점수를 통해 평가하는 함수 (Hugging Face evaluate 사용)
    
#     :parameter dataset_path: 평가에 사용할 데이터 파일 경로
#     :return: BLEU 점수
#     """
    
#     df = pd.read_csv(dataset_path)
#     references = df['output'].tolist()
#     hypotheses = df['translated'].tolist()
#     logging.warning('Data loading complete!')
    
#     for i in range(10):
#         print(f'hypothese: {hypotheses[i]}')
#         print(f'references: {references[i]}')
#         print('\n')
    
#     # BLEU 평가
#     logging.warning('BLEU evaluation start')
#     bleu_metric = evaluate.load('bleu')
#     bleu_score = bleu_metric.compute(predictions=hypotheses, references=[[ref] for ref in references])
    
#     logging.warning('Evaluation Complete!')
#     logging.warning(f"BLEU: {bleu_score}")
    
#     return {"BLEU": bleu_score['bleu']}

meteor = evaluate.load('meteor')
df = pd.read_csv('./data/processed_data/llama_translated_original_model.csv')
predictions = df['translated'].tolist()
references = df['output'].tolist()

results = meteor.compute(predictions=predictions, references=references)
print(results)

# if __name__ == '__main__':
#     dataset_path = './data/processed_data/llama_translated.csv'
#     evaluate_translation(dataset_path)