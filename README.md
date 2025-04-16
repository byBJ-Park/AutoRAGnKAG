# 지식 통합 기반 질의 응답에서의 RAG, AutoRAG, KAG 성능 비교 분석

이 저장소는 RAG(Retrieval Augmented Generation), AutoRAG, KAG(Knowledge-Augmented Generation) 세 가지 접근법을 한국어 질의응답 환경에서 비교 분석한 실험 기반 연구 내용을 담고 있습니다.  
본 연구는 동서울대학교 컴퓨터소프트웨어과 및 컴퓨터정보보안과 학부생들에 의해 수행되었습니다.

## 개요

- RAG는 LLM의 환각 현상을 완화하기 위한 대표적인 검색 기반 생성 기법입니다.
- AutoRAG는 RAG 파이프라인의 자동화된 최적 구성 탐색을 통해 응답의 정확도를 높이는 방식입니다.
- KAG는 지식 그래프를 기반으로 논리적, 수치적 추론을 가능하게 하며, 의미 전달력이 요구되는 질의 유형에 강점을 보입니다.

본 실험에서는 2WikiMultiHopQA 데이터셋을 이용해 세 모델의 구조와 성능을 비교하였습니다.

## 시스템 구조 요약

### AutoRAG

- 주요 구성: 데이터 생성 → 자동 파이프라인 최적화
- 질의응답 데이터를 기반으로 다양한 검색 및 생성 조합을 실험하여 최적의 구성 탐색
- Bridge Comparison 질의 유형에서 강점

### KAG

- 주요 구성요소: 
  - LLM Friendly Representation
  - Mutual Index Builder
  - Logical Form Solver
  - Knowledge Alignment
- KG 기반 논리 추론으로 정답의 의미 및 핵심 정보 반영에 우수

## 실험 환경

| 항목 | 내용 |
|------|------|
| GPU | NVIDIA V100 32GB × 4 |
| 모델 | DeepSeek-R1-Distill-Llama-8B (Llama-3.1-8B distilled) |
| 데이터셋 | 2WikiMultiHopQA (총 395개의 Q/A) |

## 평가 지표

- EM (Exact Match)
- Relaxed EM
- F1 Score
- Answer Similarity (GPT-4o 기반 의미 유사도)

## 주요 실험 결과

### 전체 성능 비교

| 모델 | EM | Relaxed EM | F1 | Similarity |
|------|----|------------|----|------------|
| RAG | 0.7% | 50.3% | 0.178 | 0.162 |
| AutoRAG | 12.1% | 38.9% | 0.235 | 0.232 |
| KAG | 6.0% | 69.3% | 0.267 | 0.282 |

- EM 기준으로는 AutoRAG가 가장 우수
- Relaxed EM, F1, Similarity 기준으로는 KAG가 가장 우수

### 질의 유형별 비교 (일부 예시)

| 유형 | 모델 | Relaxed EM | Similarity |
|------|------|-------------|-------------|
| Comparison | KAG | 93.01% | 0.307 |
| Bridge Comparison | AutoRAG | 30.88% | 0.381 |
| Compositional | KAG | 29.82% | 0.336 |

## 결론

- AutoRAG는 파이프라인 최적화를 통해 엄격한 정답 요구 질의에 적합
- KAG는 지식 그래프 기반 구조 덕분에 복잡한 논리적 추론에 효과적
- 두 시스템은 상호 보완적으로 작용할 수 있으며, 향후 통합적 접근이 유망함

## 참고문헌

1. Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", arXiv:2005.11401v4, 2021.  
2. Dongkyu Kim et al., "AutoRAG: Automated Framework for Optimization of Retrieval Augmented Generation Pipeline", arXiv:2410.20878v1, 2024.  
3. Lei Liang et al., "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation", arXiv:2409.13731v3, 2024.

## 공동 연구자

- 박병준 (컴퓨터소프트웨어과)
- 김세빈 (컴퓨터소프트웨어과)
- 정준 (컴퓨터정보보안과)

문의: joshua010106@gmail.com / kksb03@naver.com / jj81271000@gmail.com
