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

from datasets import load_dataset
from dotenv import load_dotenv



load_dotenv()
TRAINING_DIR = os.getenv("TRAINING_DIR")
VALIDATION_DIR = os.getenv("VALIDATION_DIR")
DATA_DIR = os.getenv("DATA_DIR")

def remove_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    지정된 컬럼의 문자열에서 양쪽 끝의 큰따옴표를 제거합니다.
    
    :parameter df: 처리할 데이터프레임
    :parameter column: 큰따옴표 제거를 적용할 컬럼 이름
    :return: 큰따옴표가 제거된 데이터프레임
    """
    df.replace({'"': ''})
    
    return df

def preprocess_AIHub_data(csv_file_name: str, load_train: bool=True, is_local: bool=True) -> pd.DataFrame:
    """
    AIHub 데이터를 로드하고 전처리합니다.
    ['ko', 'en']을 추출 후, {'en': 'input', 'ko': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter csv_file_name: AIHub 원천 데이터 이름
    :parameter load_train: 어떤 데이터를 로드할 것인지 선택. True면 train data를, False면 validation data를 로드
    :parameter is_local: 데이터 소스 위치 선택. True면 로컬 데이터를, False면 HuggingFace 데이터를 로드
    :return: 전처리된 데이터프레임
    """
    def preprocess_data(df: pd.DataFrame):
        df = df[df['domain'].str.contains('의학')]
        df = df.loc[:, ['en', 'ko']]
        df.rename(columns={'en': 'input', 'ko': 'output'}, inplace=True)
        df = remove_quotes(df)
        
        return df
        
    if is_local:
        if load_train:
            csv_file_dir = os.path.join(TRAINING_DIR, csv_file_name)
        else:
            csv_file_dir = os.path.join(VALIDATION_DIR, csv_file_name)
            
        df = pd.read_csv(csv_file_dir)
        df = preprocess_data(df)
        
        if not load_train:
            validation_output_path = os.path.join(VALIDATION_DIR, 'validation_data.csv')
            df.to_csv(validation_output_path, index=False)
            logging.warning("AIHub test data (local storage) preprocess done")
            return None
        
        logging.warning("AIHub train data (local storage) preprocess done")
        
        return df
    
    else:
        df = load_dataset('ih9511/medical-translation-en-ko')
        train_df = df['train'].to_pandas()
        validation_df = df['validation'].to_pandas()
        
        train_df = preprocess_data(train_df)
        validation_df = preprocess_data(validation_df)
        
        os.makedirs(TRAINING_DIR, exist_ok=True)
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        
        train_df.to_csv(os.path.join(TRAINING_DIR, 'train_data.csv'))
        validation_df.to_csv(os.path.join(VALIDATION_DIR, 'validation_data.csv'))
        
        logging.warning("AIHub train data (huggingface repo) preprocess done")
        logging.warning("AIHub train data (huggingface repo) preprocess done")
        
        return train_df, validation_df
        

def preprocess_HuggingFace_open_data(huggingface_path: str) -> pd.DataFrame:
    """
    HuggingFace 데이터를 로드하고 전처리합니다.
    ['kor', 'eng']을 추출 후, {'eng': 'input', 'kor': 'output'}으로 컬럼명을 수정합니다.
    
    :parameter huggingface_path: 데이터를 불러올 huggingface 주소
    :return: 전처리된 데이터프레임
    """
    df = pd.read_parquet(huggingface_path)
    df = df.loc[:, ['kor', 'eng']]
    df.rename(columns={'eng': 'input', 'kor':'output'}, inplace=True)
    df = remove_quotes(df)
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
    os.makedirs(TRAINING_DIR, exist_ok=True)
    df.to_csv(csv_file_dir, index=False)
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

def preprocess_pipeline(train_csv_file_name: str, validation_csv_file_name: str) -> None:
    """
    전체 전처리 파이프라인:
        1. train_data.csv 로드 후 텍스트 정규화, 토큰화 및 인코딩 수행 후, 이를 학습 및 검증 데이터셋으로 분할합니다.
        2. validation_data.csv 로드 후 텍스트 정규화, 토큰화 및 인코딩 수행 후, 이를 테스트 데이터셋으로 사용합니다.
        3. 전처리된 데이터셋(train, validation, test)을 지정한 폴더에 CSV 파일로 저장합니다.
        
    :parameter train_csv_file_name: 학습 데이터 csv 파일 이름
    :parameter validation_csv_file_name: 검증 데이터 csv 파일 이름
    """
    # train_data.csv 처리 (학습 데이터셋 생성)
    train_df = pd.read_csv(os.path.join(TRAINING_DIR, train_csv_file_name))
    for col in ['input', 'output']:
        train_df[col] = train_df[col].astype(str).apply(normalize_text)
    
    # validation_data.csv 처리 (검증 데이터셋)
    test_df = pd.read_csv(os.path.join(VALIDATION_DIR, validation_csv_file_name))
    for col in ['input', 'output']:
        test_df[col] = test_df[col].astype(str).apply(normalize_text)
    
    # 전처리된 데이터 저장
    output_dir = os.path.join(os.path.dirname(DATA_DIR), "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
    logging.warning("Preprocessing pipeline completed and files saved.")

if __name__ == "__main__":
    # aihub_train_data = preprocess_AIHub_data(csv_file_name='1113_tech_train_set_1195228.csv', load_train=True)
    # aihub_validation_data = preprocess_AIHub_data(csv_file_name='1113_tech_valid_set_149403.csv', load_train=False)
    aihub_train_data, aihub_validation_data = preprocess_AIHub_data(csv_file_name=None, load_train=None, is_local=False)
    huggingface_train_data = preprocess_HuggingFace_open_data(huggingface_path='hf://datasets/ChuGyouk/chest_radiology_enko/data/train-00000-of-00001.parquet')
    concatenated_data = concat_data(aihub_train_data, huggingface_train_data, csv_file_name='train_data.csv')
    preprocess_pipeline(train_csv_file_name="train_data.csv", validation_csv_file_name="validation_data.csv")