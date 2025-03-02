# 1. Abstract
## 1.1 프로젝트 목적
본 프로젝트는 의료 도메인에 특화된 영어-한국어 번역 모델을 파인튜닝하는 것을 목표로 합니다. 파인 튜닝 과정에서 LoRA 기법을 활용하여 Parameter-efficient fine-tuning (PEFT)을 적용하고, 공개된 의료 데이터셋(AI-Hub, Hugging face)을 이용하여 모델을 학습합니다. 번역 품질 분석을 위해 번역 태스크에 최적화된 BLEU, METEOR등 평가지표를 활용하여 모델의 성능을 평가합니다.
## 1.2 프로젝트 주요 과정
  1. 데이터 전처리: AI-Hub 및 Hugging face의 의료 번역 데이터셋을 정제 및 토큰화.
  2. 모델 파인튜닝: `Llama3-Korean-Bllossom-8B` 모델을 PEFT 방식으로 최적화.
  3. 모델 평가: BLEU Score, METEOR Score 기반 성능 평가.
  4. 모델 및 학습 데이터 배포: Hugging face에 업로드하여 활용 가능하도록 구성.
# 2. Data Preprocessing (`preprocessing.py`)
## 2.1 데이터셋 선정
- 학습 데이터: AI-Hub의 한국어-영어 번역 말뭉치(기술과학)
## 2.2 전처리 과정
  1. 데이터 선별: AI-Hub 데이터셋에는 ICT, 전기, 전자, 기계, 의학의 5가지 대분야(`domain`)가 존재. `domain == '의학'`만 추출하여 파인 튜닝에 활용.
  2. 데이터 정제(Cleansing): 특수 문자 및 불필요한 공백 제거(`remove_quotes`, `normalize_text`).
  3. Train, Validation 데이터 분리 및 저장.
  4. 전체 데이터 전처리 과정을 파이프라인화하여 관리.
# 3. Model Fine-tuning (`finetune.py`)
## 3.1 사용 모델
Hugging face hub에 공개되어 있는 `MLP-KTLim/llama-3-Korean-Bllossom-8B` 모델을 활용합니다. 해당 모델은 약 100GB의 한국어 데이터를 통해 Llama3를 풀 튜닝한 모델로, 한국어-영어 Parallel Corpus를 활용하여 지식연결 작업을 수행한 모델입니다. 한국어 문화, 언어를 고려하여 언어학자가 제작한 데이터를 활용하여 파인 튜닝을 진행하여 본 과제를 수행하기 위한 베이스 모델로 활용하기 적합하다고 판단하였습니다.
## 3.2 LoRA 적용
- 추론 성능 유지와 학습 비용 절감을 위해 LoRA 기법 적용.
- 학습 가중치 모듈: `q_proj`, `v_proj`, `k_proj`
- 하이퍼파라미터:
    - Rank (`r`): 8
    - LoRA alpha: 32
    - LoRA Dropout: 0.1
## 3.3 추론 설정
의학 정보의 특성 상, 엄밀한 정보를 정확하게 번역하는 것이 중요합니다. 이에, 확률 기반인 AI 모델을 최대한 높은 확률만을 선택하도록 유도할 필요성이 높습니다. 추론 과정에서 모델의 번역 정확도를 높이기 위해 프롬프트 엔지니어링 기법을 적용하고 샘플링 요소들을 조정하였습니다.  
  
**프롬프트 엔지니어링**
![image](https://github.com/user-attachments/assets/66919088-2610-4fa3-80db-20917e75000c)

# 4. Model Evaluation (`inference.py`, `evaluation.py`)
## 4.1 평가 데이터셋
- 평가 데이터: Hugging face hub의 `ChuGyouk/chest_radiology_enko` 활용.
## 4.2 정량 평가
원본 모델(`MLP-KTLim/llama-3-Korean-Bllossom-8B`)과 파인 튜닝 모델(ih9511/llama-3-Korean-8B_koen_medical_translation)에 대한 정량적인 평가를 위해 METEOR와 BLEU Score를 비교합니다. 원본 모델 대비 각 점수의 증감을 살펴보고 파인 튜닝 전과 후의 차이점에 대해 분석합니다. METEOR Score의 경우 `evaluate.load('meteor')`를 활용하며, BLEU Score의 경우 `evaluate.load('bleu')`를 활용합니다. 각각의 점수는 원본 데이터(`output`)와 번역 결과 데이터(`trained`)를 비교하여 산출합니다.
||Base Model|Fine-tuned Model|
|------|---|---|
|METEOR|21.04|33.08|
|BLEU|3.42|10.30|

  - METEOR Score: 21.04 -> 33.08, **약 12%p 증가**
  - BLEU Score: 3.42 -> 10.30, **약 6%p 증가**
## 4.3 정성 평가
- 원본 문장:
  
>_The inflammatory process within the lung parenchyma had caused significant alterations to the (pulmonary) acinus, leading to reduced gas exchange efficiency and overall respiratory compromise, highlighting the importance of early intervention to prevent further structural damage._

- Base Model Translation:
>_“폐포가 유해한 영향을 받아 폐 조직 내 염증 과정은 가스 교환 효율을 감소시키고 전체 호흡 기능에 악영향을 미치며, 이는 초기 개입이 필수적임을 강조하며 구조적 손상을 방지하기 위해 더욱 중요한 상황입니다._
>_(Translation note:_
>_- "lung parenchyme" is translated as "" which means 'lung tissue' or 'parenchymal tissue'._
>_- "acinus" is directly translated as "", maintaining its technical term._
>_- "gas exchange efficiency" remains as "" for precision._
>_- Overall tone”_

- Fine-tuned Model Translation:
>_“폐질환부 내 염증 과정은 유기체의 상피 변화에 큰 변화를 일으켜 가스 교환 효율과 전반적인 호흡 장애를 감소시켰으며 더 이상 구조적 손상을 방지하기 위한 조기에 개입하는 것이 중요하다는 것을 강조했다.”_

원본 모델의 경우 단순 번역 태스크에는 맞지 않는 추가 정보를 생성한 것과 대비하여, 파인 튜닝 모델의 경우 추가적인 정보 생성 없이 번역 태스크를 잘 수행한 것을 알 수 있습니다. 또한, 예제에는 표현되어 있지 않지만 원본 모델에서 수치적인 의료 정보의 경우 단위가 바뀌거나 숫자가 바뀌는 경우도 다수 관찰할 수 있었습니다. 이러한 점에 비추어 보았을 때 의도한 대로 잘 학습이 되었음을 알 수 있습니다.
# 5. 모델 배포 및 결론
## 5.1 모델 배포
- Hugging face hub 에 업로드(`ih9511/llama-3-Korean-8B_koen_medical_translation`).
- LoRA 어댑터 병합 후 업로드하여 바로 추론 가능하도록 구성.
## 5.2 결론
약 40만개의 한국어-영어 말뭉치 데이터를 사용하여 Llama3-Korean-Bllossom-8B 모델을 파인 튜닝 하였습니다. 효율적인 학습을 위해 PEFT 방법론 중 하나인 LoRA를 활용하여 전체 학습 시간의 경우 8B 모델을 L40S GPU (48GB VRAM) 기준으로 약 5시간이 소요되었습니다. 원본 모델과 어댑터의 병합을 통해 테스트 데이터를 기준으로 정량적, 정성적 분석을 수행하였습니다. 파인 튜닝 모델이 원본 모델 대비 METEOR Score 기준 약 12%p의 증가함을 확인하였고, 번역 품질의 경우 의료 번역이라는 도메인의 성격에 맞게 경어체에서 평서체로 어투가 바뀌었습니다. 단, 추론 속도의 경우 MIMIC-III 데이터 하나 당 약 30초 정도 소요된 점을 고려했을 때, 추론 시간 관점에서 추후 추론 속도가 중요한 태스크에서는 다양한 추론 가속 기법을 활용하여 단축할 필요가 있습니다.
