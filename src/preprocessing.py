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


load_dotenv()
TRAINING_DIR = os.getenv("TRAINING_DIR")
VALIDATION_DIR = os.getenv("VALIDATION_DIR")

def preprocess_AIHub_data(csv_file_name: str, load_train=True) -> pd.DataFrame:
    """
    AIHub 데이터를 로드하고 전처리합니다.
    ['ko', 'en']을 추출 후, {'en': 'input', 'ko': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter csv_file_name: AIHub 원천 데이터 이름
    :parameter load_train: 어떤 데이터를 로드할 것인지 선택. True면 train data를, False면 validation data를 로드
    :return: 전처리된 DataFrame
    """
    if load_train:
        csv_file_dir = TRAINING_DIR + csv_file_name
    else:
        csv_file_dir = VALIDATION_DIR + csv_file_name
        
    df = pd.read_csv(csv_file_dir)
    df = df[df['domain'].str.contains('의학')]
    df = df.loc[:, ['en', 'ko']]
    df.rename(columns={'en': 'input', 'ko': 'output'}, inplace=True)
    
    
    if not load_train:
        df.to_csv(VALIDATION_DIR + 'validation.csv')
        logging.warning("AIHub test data preprocess done")
        return None
    logging.warning("AIHub train data preprocess done")
    
    return df

def preprocess_HuggingFace_data(huggingface_path: str) -> pd.DataFrame:
    """
    HuggingFace 데이터를 로드하고 전처리합니다.
    ['kor', 'eng']을 추출 후, {'eng': 'input', 'kor': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter huggingface_path: 데이터를 불러올 huggingface 주소
    :return: 전처리된 DataFrame
    """
    df = pd.read_parquet(huggingface_path)
    df = df.loc[:, ['kor', 'eng']]
    df.rename(columns={'eng': 'input', 'kor':'output'}, inplace=True)
    logging.warning(f"HuggingFace data preprocess done")
    
    return df

def concat_data(df1: pd.DataFrame, df2: pd.DataFrame, csv_file_name: str) -> None:
    """
    두 개의 데이터프레임을 합친 후 저장합니다.
    
    :parameter df1: 합치고자 하는 데이터프레임
    :parameter df2: 합치고자 하는 데이터프레임
    :parameter csv_file_name: 합쳐진 데이터프레임을 저장할 이름
    """
    csv_file_dir = TRAINING_DIR + csv_file_name
    
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(csv_file_dir)
    logging.warning(f"Concatenated dataframe has been saved in {csv_file_dir}")

# TODO
def load_data(csv_file_path: str, text_columns: List[str]) -> List[str]:
    """
    주어진 파일 경로에서 원천 데이터를 로드합니다.
    
    :parameter csv_file_path: csv 파일의 경로
    :parameter text_columns: 
    :return: 각 줄이 하나의 텍스트 데이터인 리스트
    """
    pass

# TODO
def filter_medical_text():
    """
    
    """
    pass

# TODO
def clean_text(text: str):
    pass

# TODO
def tokenize_text(text: str):
    pass


if __name__ == "__main__":
    aihub_train_data = preprocess_AIHub_data(csv_file_name='1113_tech_train_set_1195228.csv', load_train=True)
    aihub_validation_data = preprocess_AIHub_data(csv_file_name='1113_tech_valid_set_149403.csv', load_train=False)
    huggingface_train_data = preprocess_HuggingFace_data(huggingface_path='hf://datasets/ChuGyouk/chest_radiology_enko/data/train-00000-of-00001.parquet')
    concat_data(aihub_train_data, huggingface_train_data, csv_file_name='train_data.csv')