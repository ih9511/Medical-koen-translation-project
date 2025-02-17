"""
preprocessing.py

이 모듈은 영어-한국어 의료 번역 모델 파인튜닝을 위한 데이터 전처리 작업을 수행합니다.
주요 기능:
- CSV 파일 형식의 원천 데이터 로딩
- 의학 관련 데이터 필터링
- 텍스트 클리닝, 토큰화, 및 포매팅
- 전처리 파이프라인 구성
"""
import os
import re
import logging
import pandas as pd

from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer


load_dotenv()
TRAINING_DIR = os.getenv("TRAINING_DIR")
VALIDATION_DIR = os.getenv("VALIDATION_DIR")

def remove_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    지정된 컬럼의 문자열에서 양쪽 끝의 큰따옴표를 제거합니다.
    
    :parameter df: 처리할 데이터프레임
    :parameter column: 큰따옴표 제거를 적용할 컬럼 이름
    :return: 큰따옴표가 제거된 데이터프레임
    """
    df.replace({'"': ''})
    
    return df

def preprocess_AIHub_data(csv_file_name: str, load_train=True) -> pd.DataFrame:
    """
    AIHub 데이터를 로드하고 전처리합니다.
    ['ko', 'en']을 추출 후, {'en': 'input', 'ko': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter csv_file_name: AIHub 원천 데이터 이름
    :parameter load_train: 어떤 데이터를 로드할 것인지 선택. True면 train data를, False면 validation data를 로드
    :return: 전처리된 데이터프레임
    """
    if load_train:
        csv_file_dir = os.path.join(TRAINING_DIR, csv_file_name)
    else:
        csv_file_dir = os.path.join(VALIDATION_DIR, csv_file_name)
        
    df = pd.read_csv(csv_file_dir)
    df = df[df['domain'].str.contains('의학')]
    df = df.loc[:, ['en', 'ko']]
    df.rename(columns={'en': 'input', 'ko': 'output'}, inplace=True)
    df = remove_quotes(df=df)
    
    if not load_train:
        validation_output_path = os.path.join(VALIDATION_DIR, 'validation_data.csv')
        df.to_csv(validation_output_path)
        logging.warning("AIHub test data preprocess done")
        return None
    
    logging.warning("AIHub train data preprocess done")
    
    return df

def preprocess_HuggingFace_data(huggingface_path: str) -> pd.DataFrame:
    """
    HuggingFace 데이터를 로드하고 전처리합니다.
    ['kor', 'eng']을 추출 후, {'eng': 'input', 'kor': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter huggingface_path: 데이터를 불러올 huggingface 주소
    :return: 전처리된 데이터프레임
    """
    df = pd.read_parquet(huggingface_path)
    df = df.loc[:, ['kor', 'eng']]
    df.rename(columns={'eng': 'input', 'kor':'output'}, inplace=True)
    df = remove_quotes(df=df)
    logging.warning(f"HuggingFace data preprocess done")
    
    return df

def concat_data(df1: pd.DataFrame, df2: pd.DataFrame, csv_file_name: str) -> pd.DataFrame:
    """
    두 개의 데이터프레임을 합친 후 저장합니다.
    
    :parameter df1: 합치고자 하는 데이터프레임
    :parameter df2: 합치고자 하는 데이터프레임
    :parameter csv_file_name: 합쳐진 데이터프레임을 저장할 이름
    :return: 합쳐진 데이터프레임
    """
    csv_file_dir = os.path.join(TRAINING_DIR, csv_file_name)
    
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(csv_file_dir)
    logging.warning(f"Concatenated dataframe has been saved in {csv_file_dir}")
    
    return df

def validate_and_cleansing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임 내 결측치 및 이상치를 검사하고 정제합니다.
    
    :parameter df: 입력 데이터프레임
    :return: 정제된 데이터프레임
    """
    # 결측치 제거
    df = df.dropna()
    
    # input, output 컬럼의 길이가 너무 짧거나 긴 경우 필터링
    df = df[df['input'].str.len() > 10]
    df = df[df['output'].str.len() > 10]
    logging.warning(f"Data validation and cleansing completed. Total samples: {len(df)}")
    
    return df

def normalize_text(text: str) -> str:
    """
    텍스트 정규화: 불필요한 공백을 제거하고 특수문자를 정규화합니다.
    
    :parameter text: 정규화 할 텍스트
    :return: 정규화된 텍스트
    """
    # 여러 공백을 하나의 공백으로 변경
    text = re.sub(r'\s+', ' ', text)
    # 따옴표 통일
    text = text.replace("“", '"').replace("”", '"')
    
    return text.strip()

def tokenize_and_encode(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128) -> pd.DataFrame:
    """
    데이터프레임의 'input'과 'output' 컬럼에 대해 토큰화 및 인코딩을 수행합니다.
    인코딩 결과를 새로운 컬럼에 저장합니다.
    
    :parameter df: 토큰화 및 임코딩을 수행할 원본 데이터프레임
    :parameter tokenizer: HuggingFace의 AutoTokenizer 객체
    :parameter max_length: 토큰 시퀀스의 최대 길이
    :return: 'input_ids', 'output_ids' 컬럼이 추가된, 토큰화 및 인코딩이 완료된 데이터프레임
    """
    def encode_text(text):
        encoding = tokenizer(
            text,
            add_special_token=True,
            max_length=max_length,
            tuncation=True,
            padding='max_length',
        )
        
        return encoding['input_ids']
    
    df['input_ids'] = df['input'].apply(encode_text)
    df['output_ids'] = df['output'].apply(encode_text)
    
    return df

if __name__ == "__main__":
    aihub_train_data = preprocess_AIHub_data(csv_file_name='1113_tech_train_set_1195228.csv', load_train=True)
    aihub_validation_data = preprocess_AIHub_data(csv_file_name='1113_tech_valid_set_149403.csv', load_train=False)
    huggingface_train_data = preprocess_HuggingFace_data(huggingface_path='hf://datasets/ChuGyouk/chest_radiology_enko/data/train-00000-of-00001.parquet')
    concatenated_data = concat_data(aihub_train_data, huggingface_train_data, csv_file_name='train_data.csv')