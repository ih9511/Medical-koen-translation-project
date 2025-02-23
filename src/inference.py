"""
inference.py

이 모듈은 파인튜닝된 모델을 통해 테스트 데이터에 대한 번역을 수행합니다.
- Fine-tuned llama-3-Korean-Bllossom-8B 모델을 로드합니다.
- 원본 모델(llama-3-Korean-Bllossom-8B)의 토크나이저를 로드합니다.
- 테스트 데이터에 대한 번역을 수행하고 번역 데이터를 저장합니다.
"""

import pandas as pd
import logging
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def translate_text(model_id, tokenizer_id, data_load_dir: str, data_save_dir: str):
    """
    입력 텍스트를 번역하는 함수
    
    :parameter model_id: 로드된 번역 모델
    :parameter tokenizer_id: 해당 모델의 토크나이저
    :parameter data_load_dir: 번역할 데이터 주소
    :parameter data_save_dir: 번역된 데이터를 저장할 주소
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    
    # 추론 최적화
    # model.generation_config.cache_implementation = 'static'
    # model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=True)
    logging.warning('Model and tokenizer load complete!')

    model.eval()
    
    test_df = pd.read_csv(data_load_dir, encoding='utf-8')
    translated_text_list = []
    
    i = 1
    for _, row in test_df.iterrows():
        query = row['text']
        prompt = (
            '''
            [[MEDICAL TRANSLATION MODE]] You are a highly accurate professional AI translator. Translate the following English clinical sentence into Korean with strict adherence to the original meaning and structure:
            - Output must be a single, standalone Korean sentence. Do not include any additional information.
            - Preserve exact measurement units (keep original numbers, units, and medical notations unchanged).
            - Ensure precise, formal translation without idiomatic adaptions.
            - Mainatain clinical terminology accuracy.
            - Don't make any fake information about the patient. Only translate the information written in query.
            '''
            f"English: {query}"
        )
        instruction = "Korean:"
        
        messages = [
            {'role': 'system', 'content': f'{prompt}'},
            {'role': 'user', 'content': f'{instruction}'}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(model.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        translated = model.generate(
            input_ids,
            max_new_tokens=1024,
            no_repeat_ngram_size=3, # 반복되는 토큰 배제
            repetition_penalty=1.2, # 반복되는 토큰 배제
            temperature=0.2, # 확률분포 sharpening
            top_p=0.9, # Sharp 확률분포에서 높은 확률의 토큰 선택하도록 유도
            do_sample=True,
            eos_token_id=terminators,
        )
        translated_text = tokenizer.decode(translated[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"✅{i}번째 번역: {translated_text}\n")
        
        translated_text_list.append(translated_text.strip())
        i += 1
        
    test_df['translated'] = translated_text_list
    
    # 번역 결과 저장
    output_path = data_save_dir
    test_df.to_csv(output_path, index=False, encoding='utf-8')
    logging.warning('Translation result saved')
    

if __name__ == '__main__':
    base_model_name = 'ih9511/llama-3-Korean-8B_koen_medical_translation'
    finetune_model_name = 'ih9511/llama-3-Korean-8B_koen_medical_translation'
    # base_model_name = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    # finetune_model_name = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    dataset_path = '../data/mimic_iii_text.csv'
    
    translate_text(model_id=finetune_model_name, tokenizer_id=base_model_name, data_load_dir=dataset_path, data_save_dir='./data/finetuned_result/mimic_iii_text.csv')