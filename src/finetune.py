"""
finetune.py

이 모듈은 LoRA 기법을 활용하여 영어 - 한국어 의료 번역 모델을 파인튜닝하는 기능을 수행합니다.
주요 기능:
- 사전학습 모델 로드 및 LoRA 적용
- 학습 및 검증 데이터셋 로딩
- 학습 루프 구현 및 체크포인트 저장
- TrainingArgumets 및 Trainer를 통한 학습 환경 구성
"""
import logging
import torch
import os
import pandas as pd

from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig # LoRA를 위한 PEFT 라이브러리
from trl import SFTTrainer


load_dotenv(override=True)
DATA_DIR = os.getenv("DATA_DIR")
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'


def format_translation_prompt(train_csv_path: str, validation_csv_path: str) -> Dataset:
    """
    영-한 번역 LLM 파인튜닝용 프롬프트 데이터셋으로 변환합니다.
    
    :parameter train_csv_path: 학습 데이터 CSV 파일 경로
    :parameter validation_csv_path: 검증 데이터 CSV 파일 경로
    :return: 변환된 Dataset 객체 (train, validation)
    """
    dataset = load_dataset('csv', data_files={
        'train': train_csv_path,
        'validation': validation_csv_path,
    }, keep_in_memory=True)
    
    # 'prompt' 필드 생성
    def create_prompt(example):
        prompt = (f"<med_translation>\n"
                  f"<en_to_ko>\n"
                  f"<source_text> {example['input']} </source_text>\n"
                  f"<target_text> {example['output']} </target_text>\n"
                  f"<end_translation>")
        return {'prompt': prompt}
    
    dataset = dataset.map(create_prompt)
    
    return dataset['train'], dataset['validation']
    
    
def tokenizer_after_check_padding_token(model_id: str) -> AutoTokenizer:
    """
    모델의 pad_token 존재 여부를 확인합니다.
    
    :parameter model_id: HuggingFace 모델 이름
    :return: tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
    )
    
    if tokenizer.pad_token is None:
        logging.warning(f'{model_id} has no pad_token. Set pad_token as eos_token.')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        
        return tokenizer
    
    else:
        logging.warning(f'{model_id} has pad_token. {tokenizer.pad_token}')
        tokenizer.padding_side = 'right'
        
        return tokenizer
    
    
def train_pipeline() -> None:
    """
    LoRA를 활용한 모델 파인튜닝의 전체 파이프라인을 실행합니다.
    
    1. 전처리된 학습 데이터셋 및 검증 데이터셋을 로드합니다.
    2. 사전학습 모델과 토크나이저를 로드하고, LoRA를 적용합니다.
    3. TrainingArguments를 설정합니다.
    4. 학습 루프를 실행하고, 최종 모델을 저장합니다.
    """
    
    # 모델 및 토크나이저 로드
    # model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer = tokenizer_after_check_padding_token(model_id=model_id)
    logging.warning('Tokenizer setting complete')
    
    # 전처리된 데이터셋 로드
    # dataset = load_dataset('csv', data_files={
    #     'train': os.path.join(DATA_DIR, "processed_data/train_processed.csv"),
    #     'validation': os.path.join(DATA_DIR, "processed_data/val_processed.csv")
    # })
    
    train_dataset, validation_dataset = format_translation_prompt('./data/processed_data/train_processed.csv', './data/processed_data/val_processed.csv')
    print(train_dataset[1])
    print(validation_dataset[1])
    logging.warning('Successfully load datasets!')
    
    def format_dataset(batch):
        tokens = tokenizer.batch_encode_plus(
            batch['prompt'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
        )
        # example['input_ids'] = tokens
        # return example
        
        return {
            'input_ids': tokens['input_ids'].to(dtype=torch.long), 
            'attention_mask': tokens['attention_mask'].to(dtype=torch.long),
        }
    
    mapped_train_dataset = train_dataset.map(format_dataset, batched=True, remove_columns=['input', 'output'])
    mapped_validation_dataset = validation_dataset.map(format_dataset, batched=True, remove_columns=['input', 'output'])
    logging.warning('Mapping complete!')
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        # attn_implementation='eager', # only for gemma2-2b
    )
    
    # LoRA 설정 적용
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout= 0.1,
        target_modules=["q_proj", "v_proj", "k_proj"],
    )
    
    logging.warning("Model loaded and LoRA applied")
    
    model = get_peft_model(model, lora_config)
    
    # GPU가 bfloat16을 지원하는지 확인
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_supported = torch.cuda.is_available() and not bf16_supported
    logging.warning(f"bf16 support: {bf16_supported}, fp16 support: {fp16_supported}")
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir='../models/llama-3-Korean-8B_koen_medical_translation',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        optim='paged_adamw_32bit', # 로컬 학습 시 페이징 (GPU -> CPU) 기법 활용을 위해 설정
        learning_rate=2e-5, # LoRA 권장 값 (1e-5 ~ 2e-4)
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='../logs/llama-3-Korean-8B_koen_medical_translation',
        logging_steps=50,
        bf16=bf16_supported,
        fp16=fp16_supported,
        report_to='tensorboard',
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_train_dataset,
        eval_dataset=mapped_validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    # 학습 실행
    logging.warning('Training start')
    trainer.train()
    logging.warning('Training complete!')
    trainer.save_model('../models/llama-3-Korean-8B_koen_medical_translation')
    model.half()
    logging.warning('Fine-tuned model saved!')
    
    
    
if __name__ == '__main__':
    train_pipeline()