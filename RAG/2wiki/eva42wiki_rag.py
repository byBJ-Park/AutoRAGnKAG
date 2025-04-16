import os
import json
import pandas as pd
from tqdm import tqdm
import re
import string
from collections import Counter
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# LangChain 관련
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers import EnsembleRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever

# deepseek_r1 모델 불러오기 (deeoseek_r1_model_load.py에 정의되어 있다고 가정)
from ollama_model_load import deepseek_r1

# OpenAI LLM 호출을 위한 모듈 (compare_summarization_answers에서 필요)
from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Evaluate 클래스와 유틸리티 함수 정의 (evaluate.py 및 evaUtils.py의 모든 기능 포함)
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

def compare_summarization_answers(
    query,
    answer1,
    answer2,
    *,
    api_key="EMPTY",
    base_url="http://127.0.0.1:38080/v1",
    model="gpt-4o-mini",
    language="English",
    retries=3,
):
    """LLM으로 두 답변 비교 (Comprehensiveness, Diversity, Empowerment)"""
    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """
    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, give each answer a score between 0 and 10, choose the better answer (either Answer 1 or Answer 2) and explain why.
    Then, give each answer an overall score between 0 and 10, and select an overall winner based on these three categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Score 1": [Score of Answer 1 - an integer between 0 and 10],
            "Score 2": [Score of Answer 2 - an integer between 0 and 10],
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation in {language} here]"
        }},
        "Diversity": {{
            "Score 1": [Score of Answer 1 - an integer between 0 and 10],
            "Score 2": [Score of Answer 2 - an integer between 0 and 10],
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation in {language} here]"
        }},
        "Empowerment": {{
            "Score 1": [Score of Answer 1 - an integer between 0 and 10],
            "Score 2": [Score of Answer 2 - an integer between 0 and 10],
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation in {language} here]"
        }},
        "Overall": {{
            "Score 1": [Score of Answer 1 - an integer between 0 and 10],
            "Score 2": [Score of Answer 2 - an integer between 0 and 10],
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria in {language}]"
        }}
    }}
    """
    for index in range(retries):
        content = None
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3]
            metrics = json.loads(content)
            return metrics
        except Exception:
            if index == retries - 1:
                message = (
                    f"Comparing summarization answers failed.\n"
                    f"query: {query}\n"
                    f"answer1: {answer1}\n"
                    f"answer2: {answer2}\n"
                    f"content: {content}\n"
                    f"exception:\n{traceback.format_exc()}"
                )
                print(message)
                return None

class Evaluate:
    """벤치마크 평가 클래스"""
    def __init__(self, embedding_factory="text-embedding-ada-002"):
        self.embedding_factory = embedding_factory

    def evaForSimilarity(self, predictionlist: List[str], goldlist: List[str]):
        """답변 유사도 평가 (미완성 상태로 0.0 반환)"""
        return 0.0  # TODO: 실제 구현 필요 시 추가

    def getBenchMark(self, predictionlist: List[str], goldlist: List[str]):
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

    def getSummarizationMetrics(
        self,
        queries: List[str],
        answers1: List[str],
        answers2: List[str],
        *,
        api_key="EMPTY",
        base_url="http://127.0.0.1:38080/v1",
        model="gpt-4o-mini",
        language="English",
        retries=3,
        max_workers=50,
    ):
        """QFS 평가를 위한 LLM 기반 메트릭 계산"""
        responses = [None] * len(queries)
        all_keys = "Comprehensiveness", "Diversity", "Empowerment", "Overall"
        all_items = "Score 1", "Score 2"
        average_metrics = {key: {item: 0.0 for item in all_items} for key in all_keys}
        success_count = 0

        def process_sample(index, query, answer1, answer2):
            metrics = compare_summarization_answers(
                query,
                answer1,
                answer2,
                api_key=api_key,
                base_url=base_url,
                model=model,
                language=language,
                retries=retries,
            )
            if metrics is None:
                print(
                    f"fail to compare answers of query {index + 1}.\n"
                    f"      query: {query}\n"
                    f"    answer1: {answer1}\n"
                    f"    answer2: {answer2}\n"
                )
            else:
                responses[index] = metrics
            return metrics

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_sample, index, query, answer1, answer2)
                for index, (query, answer1, answer2) in enumerate(
                    zip(queries, answers1, answers2)
                )
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Evaluating: "
            ):
                metrics = future.result()
                if metrics is not None:
                    for key in all_keys:
                        for item in all_items:
                            average_metrics[key][item] += metrics[key][item]
                    success_count += 1
        if success_count > 0:
            for key in all_keys:
                for item in all_items:
                    average_metrics[key][item] /= success_count
        result = {
            "average_metrics": average_metrics,
            "responses": responses,
        }
        return result

# 1) RAG 검색 수행 함수 (EnsembleRetriever 적용)
def perform_rag(
    question: str,
    text_file: str = "2wiki_corpus.json",
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
    json_file: str = "2wiki_qa.json",
    text_file: str = "2wiki_corpus.json",
    output_file: str = "2wiki_result_rag.csv",
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
        json_file="2wiki_qa.json",
        text_file="2wiki_corpus.json",
        output_file="2wiki_result_rag.csv",
        batch_size=5,
        chunk_size=100,
        chunk_overlap=50,
        device="cuda"
    )
    print(final_df.head())