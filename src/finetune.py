"""
finetune.py

이 모듈은 QLoRA 기법을 활용하여 영어 - 한국어 의료 번역 모델을 파인튜닝하는 기능을 수행합니다.
주요 기능:
- 사전학습 모델 로드 및 QLoRA 적용
- 학습 및 검증 데이터셋 로딩
- 학습 루프 구현 및 체크포인트 저장
- TrainingArgumets 및 Trainer를 통한 학습 환경 구성
"""
import logging
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments # 번역 task를 위한 Seq2SeqLM
from peft import get_peft_model, LoraConfig # QLoRA를 위한 PEFT 라이브러리
from utils import load_dataset # 전처리된 CSV 데이터를 로드


def load_model(model_name: str, qlora_config: dict) -> torch.nn.Module:
    """
    사전학습 모델을 로드하고 QLoRA를 적용합니다.
    
    :parameter model_name: 사전학습 모델의 이름 또는 경로
    :parameter qlora_config: QLoRA 설정을 담은 딕셔너리
    :return: QLoRA가 적용된 모델 객체
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='cuda')
    # QLoRA 설정 적용
    lora_config = LoraConfig(
        r=qlora_config.get("r", 8),
        lora_alpha=qlora_config.get("lora_alpha", 32),
        lora_dropout=qlora_config.get("lora_dropout", 0.1),
        target_modules=qlora_config.get("target_modules", ["q_proj", "v_proj"])
    )
    model = get_peft_model(model, lora_config)
    logging.warning("Model loaded and QLoRA applied")
    
    return model

def train_model(model: torch.nn.Module, tokenizer: AutoTokenizer, train_dataset, val_dataset, output_dir: str, training_args: TrainingArguments) -> None:
    """
    QLoRA가 적용된 모델을 학습 및 검증 데이터셋을 사용해 파인튜닝하고, 결과를 저장합니다.
    
    :parameter model: QLoRA가 적용된 모델
    :parameter tokenizer: 모델과 함께 사용할 토크나이저
    :parameter train_dataset: 학습에 사용할 데이터셋
    :parameter val_dataset: 검증에 사용할 데이터셋
    :parameter output_dir: 학습 결과 체크포인트를 저장할 디렉토리
    :parameter training_args: 학습 설정을 담은 TrainingArguments 객체
    :return: None (모델 체크포인트가 output_dir에 저장됨)
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logging.warning(f"Training completed and model saved to {output_dir}")
    
def train_pipeline() -> None:
    """
    QLoRA를 활용한 모델 파인튜닝의 전체 파이프라인을 실행합니다.
    
    1. 전처리된 학습 데이터셋 및 검증 데이터셋을 로드합니다.
    2. 사전학습 모델과 토크나이저를 로드하고, QLoRA를 적용합니다.
    3. TrainingArguments를 설정합니다.
    4. 학습 루프를 실행하고, 최종 모델을 저장합니다.
    """
    
    # 전처리된 데이터셋 로드
    train_dataset = load_dataset("processed_data/train_processed.csv")
    val_dataset = load_dataset("processed_data/val_processed.csv")
    
    # 모델 및 토크나이저 로드
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # QLoRA 설정
    qlora_config = {
        'r': 8, # update matrics의 rank. 작을수록 trainable param이 적어짐. original weight matrix를 얼마나 줄일 것인지에 대한 계수이므로 작을수록 많이 압축함.
        'lora_alpha': 32, # LoRA scaling factor. scaling 값이 lora_alpha/r로 들어감 
        'lora_dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj'], # lora로 바꿀 모듈
    }
    model = load_model(model_name, qlora_config)
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir='../models',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='../logs',
        logging_steps=50,
        bf16=True,
        report_to='tensorboard',
    )
    
    # 학습 실행
    train_model(model, tokenizer, train_dataset, val_dataset, output_dir='../models', training_args=training_args)
    
if __name__ == '__main__':
    train_pipeline()