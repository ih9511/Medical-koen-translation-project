import pandas as pd
import logging
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Tuple


def load_model(base_model_name: str, finetune_model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Hugging Face Hub에서 사전학습된 번역 모델을 로드합니다.
    
    :parameter base_model_name: 토크나이저 세팅을 위한 원본 모델 저장소 경로
    :parameter finetune_model_name: 불러올 모델의 HuggingFace 저장소 경로
    :return: 로드된 모델과 토크나이저
    """
    model = AutoModelForCausalLM.from_pretrained(finetune_model_name, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    logging.warning('Model and tokenizer load complete!')
    
    # 추론 최적화
    model.generation_config.cache_implementation = 'static'
    model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=True)
    
    return model, tokenizer

def translate_text(model, tokenizer, data_dir: str):
    """
    입력 텍스트를 번역하는 함수
    
    :parameter model: 로드된 번역 모델
    :parameter tokenizer: 해당 모델의 토크나이저
    :parameter text: 번역할 입력 문장
    :parameter max_length: 번역된 문장의 최대 길이
    :return: 번역된 텍스트
    """
    translation_pipeline = pipeline(
        'translation',
        model=model,
        tokenizer=tokenizer,
    )
    
    translated_text = []
    test_df = pd.read_csv(data_dir)
    
    for _, row in test_df.iterrows():
        query = row['input']
        prompt = (
            "[[MEDICAL TRANSLATION MODE]] Translate this clinical text to Korean using:"
            "- Exact measurement units (preserve original numbers)"
            "- No idiomatic adaptations"
            f"English: {query}"
            "Korean:"
        )
        
        translated = translation_pipeline(
            prompt,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        translated_text = translated[0]['translation_text']
        
        # 출력 데이터로부터 한국어 문장만 추출
        output_start = translated_text.find('Korean:')
        if output_start != -1:
            extracted_text = translated_text[output_start + len('Korean:'):].strip()
        else:
            extracted_text = translated_text.strip()
        
        extracted_text = re.sub(r'\[\[.*?\]\]', '', extracted_text) # [[END OF MEDICAL TRANSLATION MODE]] 제거
        extracted_text = re.sub(r'[\d\.\s]+$', '', extracted_text) # 문장 끝의 연속된 숫자 제거
        print(extracted_text)
        
        translated_text.append(extracted_text.strip())
        
    test_df['translated'] = translated_text
    
    # 번역 결과 저장
    output_path = './processed_data/llama_translated.csv'
    test_df.to_csv(output_path, index=False)
    logging.warning('Translation result saved')
    

if __name__ == '__main__':
    base_model_name = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    finetune_model_name = 'ih9511/llama-3-Korean-8B_koen_medical_translation'
    dataset_path = './data/processed_data/test_processed.csv'
    
    model, tokenizer = load_model(base_model_name, finetune_model_name)
    translate_text(model, tokenizer, dataset_path)