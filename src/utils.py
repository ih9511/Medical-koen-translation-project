import pandas as pd
import logging

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from nltk.translate.bleu_score import sentence_bleu


class TranslationDataset(Dataset):
    """
    CSV 파일로 저장된 전처리된 번역 데이터셋을 PyTorch Dataset으로 래핑합니다.
    
    데이터셋 CSV 파일은 'input_ids'와 'output_ids' 컬럼을 포함해야 합니다.
    이 컬럼들은 이미 토큰화 및 인코딩이 완료된 상태여야 하며, 문자열 형태의 리스트로 저장되어 있습니다.
    
    :parameter csv_file: 데이터셋 CSV 파일의 경로
    :parameter tokenizer: 토크나이저 객체
    :parameter max_length: 토큰 시퀀스의 최대 길이
    """
    def __init__(self, csv_file:str, tokenizer: PreTrainedTokenizer = None, max_length: int = 128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        # CSV에 저장된 input_ids와 output_ids는 문자열로 저장되어 있을 가능성이 있으므로, 리스트로 변환합니다.
        # ex. "[101, 2023, 1012]" -> [101, 2023, 1012]
        input_ids = [int(token.strip()) for token in row['input_ids'].strip('[]').split(',') if token.strip()]
        output_ids = [int(token.strip()) for token in row['output_ids'].strip('[]').split(',') if token.strip()]
        
        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
        }
        
def load_dataset(csv_file: str) -> pd.DataFrame:
    """
    주어진 CSV 파일을 로드하여 데이터프레임을 반환합니다.
    
    :parameter csv_file: 데이터셋 CSV 파일 경로
    :return: 로드된 데이터셋 DataFrame
    """
    try:
        df = pd.read_csv(csv_file)
        logging.warning(f"Dataset loaded successfully from {csv_file}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading dataset from {csv_file}: {e}")
        raise e
    
def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    단일 문장에 대한 BLEU 점수를 계산합니다.
    
    :parameter reference: 참조 문장(정답 문장)
    :parameter hypothesis: 생성된 문장(번역 결과 문장)
    :return: BLEU 점수
    """
    
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    score = sentence_bleu([ref_tokens], hyp_tokens)
    
    return score