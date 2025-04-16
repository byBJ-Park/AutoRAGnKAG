import os
import json
import pandas as pd
from tqdm import tqdm
import re
import string
from collections import Counter

# LangChain 관련
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers import EnsembleRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever

# deeoseek_r1 모델 불러오기 (deeoseek_r1_model_load.py에 정의되어 있다고 가정)
from ollama_model_load import deepseek_r1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Evaluate 클래스와 유틸리티 함수 직접 정의
def normalize_answer(s):
    """텍스트 정규화 함수"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """F1 스코어 계산"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    """Exact Match 스코어 계산"""
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0

def get_em_f1(prediction, gold):
    """EM과 F1 계산"""
    em = exact_match_score(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    return float(em), f1

class Evaluate:
    """벤치마크 평가 클래스"""
    def __init__(self, embedding_factory="text-embedding-ada-002"):
        self.embedding_factory = embedding_factory

    def evaForSimilarity(self, predictionlist, goldlist):
        """답변 유사도 평가 (미완성 상태로 0.0 반환)"""
        return 0.0  # TODO: 실제 구현 필요 시 추가

    def getBenchMark(self, predictionlist, goldlist):
        """EM, F1, 답변 유사도 계산"""
        total_metrics = {"em": 0.0, "f1": 0.0, "answer_similarity": 0.0}

        for prediction, gold in zip(predictionlist, goldlist):
            em, f1 = get_em_f1(prediction, gold)
            total_metrics["em"] += em
            total_metrics["f1"] += f1

        total_metrics["em"] /= len(predictionlist)
        total_metrics["f1"] /= len(predictionlist)
        total_metrics["answer_similarity"] = self.evaForSimilarity(predictionlist, goldlist)

        return total_metrics

# 1) RAG 검색 수행 함수 (EnsembleRetriever 적용)
def perform_rag(
    question: str,
    text_file: str = "hotpotqa_corpus.json",
    chunk_size: int = 100,
    chunk_overlap: int = 50,
    device: str = "cuda"
) -> str:
    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [Document(page_content=t) for t in splitter.split_text(full_text)]

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    db_faiss = FAISS.from_documents(chunks, embedding=embeddings)
    faiss_retriever = db_faiss.as_retriever(search_kwargs={"k": 2})

    kiwi_bm25_retriever = KiwiBM25Retriever.from_documents(chunks)

    retriever = EnsembleRetriever(
        retrievers=[kiwi_bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        search_type="mmr",
    )

    context_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    return context_text

# 2) LLM 질의 함수 (RAG Prompt 포함)
def query_llm(context: str, question: str) -> str:
    RAG_PROMPT_TEMPLATE = """
    아래 정보(context)를 참고하여 사용자 질문에 답해주세요:
    {context}

    질문:
    {question}

    답변 시, 질문의 핵심만 파악하여 간결하게 1~2문장으로 답변하고, 
    불필요한 설명은 피합니다.

    답변:
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)
    message = HumanMessage(content=formatted_prompt)

    response = deepseek_r1.invoke([message])
    return response.content.strip()

# 3) 전체 평가 함수
def evaluate_model_responses(
    json_file: str = "hotpot_qa.json",
    text_file: str = "hotpotqa_corpus.json",
    output_file: str = "hotpot_result_rag.csv",
    batch_size: int = 5,
    chunk_size: int = 100,
    chunk_overlap: int = 50,
    device: str = "cuda"
):
    processed_count = 0
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, encoding='utf-8-sig')
        except Exception as e:
            print(f"출력 파일 읽기 오류: {e}")
            existing_df = None

        if existing_df is not None and not existing_df.empty:
            if "전체 평균" in str(existing_df.iloc[-1, 0]):
                processed_count = len(existing_df) - 1
                existing_df = existing_df.iloc[:-1]
                existing_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            else:
                processed_count = len(existing_df)

    with open(json_file, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    total_rows = len(qa_list)
    if processed_count >= total_rows:
        print("이미 모든 행이 처리되었습니다.")
        return pd.read_csv(output_file, encoding='utf-8-sig')

    evaluation_results = []
    em_list, f1_list, ans_sim_list = [], [], []
    evalObj = Evaluate()

    for idx in tqdm(range(processed_count, total_rows), desc="평가 진행 중"):
        sample = qa_list[idx]
        _id = sample["_id"]
        type_field = sample["type"]
        question = sample["question"]
        gold = sample["answer"]

        context = perform_rag(
            question=question,
            text_file=text_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=device
        )
        generated_response = query_llm(context, question)

        metrics = evalObj.getBenchMark([generated_response], [gold])
        em_val, f1_val, ans_sim_val = metrics["em"], metrics["f1"], metrics["answer_similarity"]

        em_list.append(em_val)
        f1_list.append(f1_val)
        ans_sim_list.append(ans_sim_val)

        evaluation_results.append({
            "_id": _id,
            "type": type_field,
            "question": question,
            "answer": gold,
            "model_response": generated_response,
            "em": em_val,
            "f1": f1_val,
            "answer_similarity": ans_sim_val
        })

        if (len(evaluation_results) % batch_size == 0) or (idx == total_rows - 1):
            partial_df = pd.DataFrame(evaluation_results)
            partial_df.to_csv(
                output_file,
                mode='a' if os.path.exists(output_file) and processed_count > 0 else 'w',
                index=False,
                header=not (os.path.exists(output_file) and processed_count > 0),
                encoding='utf-8-sig'
            )
            evaluation_results = []
            processed_count = idx + 1

    avg_em = sum(em_list) / len(em_list) if em_list else 0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    avg_ans_sim = sum(ans_sim_list) / len(ans_sim_list) if ans_sim_list else 0

    summary_row = {
        "_id": "전체 평균",
        "type": "",
        "question": "",
        "answer": "",
        "model_response": "",
        "em": avg_em,
        "f1": avg_f1,
        "answer_similarity": avg_ans_sim
    }

    final_df = pd.read_csv(output_file, encoding='utf-8-sig')
    final_df = pd.concat([final_df, pd.DataFrame([summary_row])], ignore_index=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"평가 완료! 결과는 '{output_file}'에 저장되었습니다.")
    return final_df

# 4) 메인 실행 예시
if __name__ == "__main__":
    final_df = evaluate_model_responses(
        json_file="hotpot_qa.json",
        text_file="hotpotqa_corpus.json",
        output_file="hotpot_result_rag.csv",
        batch_size=5,
        chunk_size=100,
        chunk_overlap=50,
        device="cuda"
    )
    print(final_df.head())