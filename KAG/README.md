# English Version: Performance Comparison Analysis of **RAG**, **AutoRAG**, and **KAG** in Integrated Knowledge-based Q&A

## Overview  
This repository provides the code and related materials that implement a research study comparing the performance of **RAG** (**Retrieval Augmented Generation**), **AutoRAG**, and **KAG** (**Knowledge-Augmented Generation**) within an integrated knowledge-based question-answering environment. This study, conducted by the research teams from the Departments of Computer Software and Computer Information Security at Dong Seoul University, specifically utilized the **2WikiMultiHopQA** and **HotpotQA** datasets to analyze the characteristics of each model based on various evaluation metrics.

## Research Background and Objectives  
The **RAG** technique has gained attention as a solution to the hallucination issues of **LLM**s; however, it exhibits limitations in logical reasoning and handling complex query types. Accordingly, this research compares the performance of **AutoRAG** and **KAG** to propose solutions for queries that demand both rigorous answer accuracy and complex logical and numerical reasoning.

## System Configuration

### **AutoRAG** Structure  
**AutoRAG** is a system that automatically searches for the optimal configuration of the **RAG** pipeline using automatic optimization techniques. This process involves parsing the original documents to generate Q&A data, then experimentally evaluating various search methods and prompt combinations to determine the optimal pipeline configuration. In this study, 1,000 optimization Q&A samples were generated using **GPT-4o** for each of the **2WikiMultiHopQA** and **HotpotQA** datasets to perform pipeline optimization.  
![Figure1](https://github.com/user-attachments/assets/439162fd-c52b-4674-aff1-4882f84d601c)

### **KAG** Structure  
**KAG** is designed to effectively address complex queries by clearly representing structured information and logical relationships based on a knowledge graph (**KG**). Its key components include:

- **LLM Friendly Representation**: Converting hierarchically structured information into a format that is easy for the language model (**LLM**) to process, thereby enhancing efficiency.  
- **Mutual Index Builder**: Constructing a bidirectional index between the original document and the **KG** to support accurate and rapid information retrieval.  
- **Logical Form Solver**: Transforming natural language queries into logical forms and dividing the problem-solving process into three stages: **Planning**, **Retrieval & Reasoning**, and **Generation**.  
- **Knowledge Alignment**: Resolving inconsistencies that may arise within the knowledge graph and maintaining consistency in retrieval results.  
- **KAG-Model**: Although additional model training methods to enhance **KAG**'s capabilities in **NLU** (**Natural Language Understanding**), **NLI** (**Natural Language Inference**), and **NLG** (**Natural Language Generation**) can be incorporated, such training was not conducted in this study.

Through this structure, **KAG** demonstrates notable strengths in handling queries that require complex logical and numerical reasoning.  
![Figure2](https://github.com/user-attachments/assets/8977e8e9-9c6b-4f0b-99d0-b2c18ba074a3)

## Experimental Setup  
- **GPU:** NVIDIA V100 32GB × 4  
- **Base Model:** DeepSeek-R1-Distill-Llama-8B (**Llama-3.1-8B Distilled**)  
- **Evaluation Datasets:** **2WikiMultiHopQA** (395 Q&A pairs), **HotpotQA**  
- **Evaluation Metrics:** **EM**, **Relaxed EM**, **F1**, **Answer Similarity** (based on **GPT-4o**)

## Results Summary

### **2WikiMultiHopQA** Evaluation Results  
Evaluation on the **2WikiMultiHopQA** dataset indicated that **AutoRAG** achieved superior performance on the **EM** metric and the Bridge Comparison type, while **KAG** excelled in **Relaxed EM**, **F1**, and **Answer Similarity** metrics as well as in the Comparison and Compositional query types.  
![Figure3](https://github.com/user-attachments/assets/86371b2e-e691-4fb2-9c7e-9647f4491520)  
![Figure4](https://github.com/user-attachments/assets/ba215744-4b95-435c-bc37-e98f7c99ac68)

### **HotpotQA** Evaluation Results  
Evaluation on the **HotpotQA** dataset demonstrated that **AutoRAG** outperformed on the **EM** metric, whereas **KAG** showed strong performance in **Relaxed EM** and **Answer Similarity**.  
 ![그림5](https://github.com/user-attachments/assets/e591b4f5-3cb8-4708-844c-c812baae443f)
![Figure6](https://github.com/user-attachments/assets/9e73efc8-eccc-4fd9-bc91-b7e07980d3d7)

## Final Results and Implications  
This study clearly delineates the characteristics and strengths of **AutoRAG** and **KAG** through their performance comparison. **AutoRAG** showcased exceptional performance in the **EM** metric and Bridge Comparison type by enhancing answer accuracy via optimization. This success is attributed to the effective experimental selection of the optimal **Retriever** configuration, based on a large-scale optimization dataset generated using **GPT-4o**, during the pipeline optimization process. In particular, the Bridge Comparison type benefited from **AutoRAG**’s optimized search method, which accurately retrieves intermediary information and sequentially uncovers precise details.

In contrast, **KAG** demonstrated high performance in metrics that capture semantic similarity and core information extraction (**Relaxed EM**, **F1**, **Answer Similarity**) owing to its logical and numerical reasoning capabilities utilizing **KG**. Its outstanding results in the Comparison and Compositional query types are primarily attributed to the clear logical information processing enabled by the **KG**.

These findings suggest that the approaches of **AutoRAG** and **KAG** are complementary, and combining their advantages could be a promising strategy in the design of next-generation **RAG** systems.

## Usage  
You can directly test the **AutoRAG** and **KAG** models and perform a performance comparison using the code and configurations provided in this repository.
