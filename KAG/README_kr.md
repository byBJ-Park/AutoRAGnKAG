# 한국어 버전  지식 통합 기반 질의 응답에서의 **RAG**, **AutoRAG**, **KAG** 성능 비교 분석  

## 개요  
본 저장소는 **RAG**(**Retrieval Augmented Generation**), **AutoRAG**, **KAG**(**Knowledge-Augmented Generation**)의 성능을 지식 통합 기반 질의 응답 환경에서 비교 분석한 연구를 구현한 코드와 관련 자료를 제공합니다. 동서울대학교 컴퓨터소프트웨어과와 컴퓨터정보보안과 연구팀에서 수행한 이 연구는 특히 **2WikiMultiHopQA**와 **HotpotQA** 데이터셋을 활용하여 다양한 평가 지표를 기준으로 각 모델의 특성을 비교 분석하였습니다.  

## 연구 배경 및 목표  
**RAG** 기법은 **LLM**의 환각 현상을 해결하기 위해 주목받고 있으나, 논리적 추론 및 복잡한 질의 유형에 한계가 있습니다. 이에 본 연구는 **AutoRAG**와 **KAG**의 성능을 비교 분석하여, 엄격한 정답 요구와 복잡한 논리·수치적 추론을 요구하는 질의에 대한 해결 방안을 제시하고자 합니다.  

## 시스템 구성  

### **AutoRAG** 구조  
**AutoRAG**는 자동 최적화 기법을 사용하여 **RAG** 파이프라인의 최적 구성을 자동으로 탐색하는 시스템입니다. 이 과정은 원본 문서를 파싱하여 생성된 질의응답(**QA**) 데이터를 기반으로 다양한 검색 기법 및 프롬프트 조합을 실험적으로 평가하여 최적의 파이프라인 구성을 결정합니다. 본 연구에서는 **2WikiMultiHopQA**와 **HotpotQA** 데이터셋 각각에 대해 **GPT-4o**를 활용하여 1,000개의 최적화용 **QA** 데이터를 생성하여 파이프라인 최적화를 수행했습니다.  
![그림1](https://github.com/user-attachments/assets/439162fd-c52b-4674-aff1-4882f84d601c)  

### **KAG** 구조  
**KAG**는 지식 그래프(**KG**)를 기반으로 구조화된 정보와 논리적 관계를 명확하게 표현하여 복잡한 질의에 효과적으로 대응할 수 있도록 설계된 시스템입니다. 주요 구성요소는 다음과 같습니다.  

- **LLM** Friendly Representation: 계층적으로 구성된 정보를 언어 모델(**LLM**)이 처리하기 쉬운 형태로 변환하여 효율성을 높입니다.  
- Mutual Index Builder: 원본 문서와 **KG** 간의 쌍방향 인덱스를 구축하여 정확하고 빠른 정보 검색을 지원합니다.  
- Logical Form Solver: 자연어 질의를 논리적 형식으로 변환하고, 이를 **Planning**, **Retrieval & Reasoning**, **Generation**의 세 단계로 나누어 문제 해결을 수행합니다.  
- Knowledge Alignment: 지식 그래프 내에서 발생할 수 있는 불일치를 해소하고 검색 결과의 일관성을 유지합니다.  
- **KAG-Model**: **KAG**의 **NLU**(**Natural Language Understanding**), **NLI**(**Natural Language Inference**), **NLG**(**Natural Language Generation**) 능력을 향상시키기 위한 추가 모델 훈련 방법들을 포함할 수 있으나, 본 연구에서는 해당 훈련 방법을 진행하지 않았습니다.  

이러한 구조를 통해 **KAG**는 복잡한 논리 및 수치적 추론이 필요한 질의에 특히 강점을 나타냅니다.  
![그림2](https://github.com/user-attachments/assets/8977e8e9-9c6b-4f0b-99d0-b2c18ba074a3)  

## 실험 환경  
- **GPU:** NVIDIA V100 32GB × 4  
- **Base Model:** DeepSeek-R1-Distill-Llama-8B (**Llama-3.1-8B Distilled**)  
- **평가 데이터셋:** **2WikiMultiHopQA** (395개 질의응답), **HotpotQA**  
- **평가 지표:** **EM**, **Relaxed EM**, **F1**, **Answer Similarity**(**GPT-4o** 기반)  

## 결과 요약  

### **2WikiMultiHopQA** 평가 결과  
**2WikiMultiHopQA** 데이터셋에서의 평가 결과는 **AutoRAG**가 **EM** 지표와 **Bridge Comparison** 유형에서 우수한 성능을 보였으며, **KAG**는 **Relaxed EM**, **F1**, **Answer Similarity** 지표와 **Comparison**, **Compositional** 유형에서 탁월한 성능을 나타냈습니다.  
![그림3](https://github.com/user-attachments/assets/86371b2e-e691-4fb2-9c7e-9647f4491520)  
![그림4](https://github.com/user-attachments/assets/ba215744-4b95-435c-bc37-e98f7c99ac68)  

### **HotpotQA** 평가 결과  
**HotpotQA** 데이터셋 평가에서는 **AutoRAG**가 **EM** 지표에서 우세했으며, **KAG**는 **Relaxed EM**, **Answer Similarity**에서 강점을 보였습니다.  
![그림5](https://github.com/user-attachments/assets/c2851edc-e162-4cc5-be6b-2944dbd26ec8)
![그림6](https://github.com/user-attachments/assets/9e73efc8-eccc-4fd9-bc91-b7e07980d3d7)  

## 최종 결과 및 시사점  
본 연구는 **AutoRAG**와 **KAG**의 성능 비교를 통해 두 모델의 특성과 강점을 명확히 구분하였습니다. **AutoRAG**는 최적화를 통한 정답 표현의 정확성을 높이는 방법으로 **EM** 지표와 **Bridge Comparison** 유형에서 특히 우수했습니다. 이는 **AutoRAG**가 파이프라인 최적화 과정에서 **GPT-4o**를 활용하여 생성된 대규모의 최적화 데이터셋을 바탕으로 가장 효과적인 **Retriever** 구성을 실험적으로 선택했기 때문입니다. 특히 **Bridge Comparison** 유형의 경우, **AutoRAG**가 최적화된 검색 방식을 통해 중개 정보의 정확한 검색과 단계적으로 정확한 정보를 연쇄적으로 찾아내는 방식이 효과적이었기 때문에 우수한 성능을 나타낸 것으로 분석됩니다.

반면 **KAG**는 **KG**를 활용한 논리적이고 수치적인 추론 능력 덕분에, 의미적 유사도와 정보의 핵심을 반영하는 지표(**Relaxed EM**, **F1**, **Answer Similarity**)에서 높은 성능을 보였습니다. 특히, **Comparison** 및 **Compositional** 질의 유형에서 뛰어난 결과를 보인 것은 **KG** 기반의 명확한 논리적 정보 처리 능력이 주요 원인으로 분석됩니다.

이러한 결과는 **AutoRAG**와 **KAG**의 접근법이 서로 상호 보완적이며, 차세대 **RAG** 시스템 설계에서 두 모델의 장점을 결합하는 전략이 매우 유망함을 의미합니다.

## 사용 방법  
본 저장소의 코드와 설정을 통해 **AutoRAG**와 **KAG** 모델을 직접 테스트하고 성능 비교를 수행할 수 있습니다.
