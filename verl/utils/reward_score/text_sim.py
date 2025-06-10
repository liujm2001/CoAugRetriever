import math
from collections import Counter
import re
import numpy as np
import os

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    return tokens

def compute_tf(doc):
    tf = Counter(doc)
    return tf

def compute_idf(doc_list):
    idf = {}
    total_docs = len(doc_list)
    doc_containing_word = Counter()
    
    for doc in doc_list:
        unique_words = set(doc)
        for word in unique_words:
            doc_containing_word[word] += 1
    
    for word, count in doc_containing_word.items():
        idf[word] = math.log((total_docs - count + 0.5) / (count + 0.5) + 1)
    
    return idf

def bm25_score(q, k, idf, k1=1.2, b=0.75, avg_doc_len=0):
    score = 0
    q_tf = compute_tf(q)
    k_tf = compute_tf(k)
    doc_len = len(k)
    
    for word in q_tf:
        if word in k_tf:
            f = k_tf[word]
            qf = q_tf[word]
            idf_weight = idf.get(word, 0)
            denominator = f + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf_weight * ((f * (k1 + 1)) / denominator) * qf
    
    return score

def compute_bm25_matrix(q_texts, k_texts, idf):
    q_tokenized = [preprocess_text(text) for text in q_texts]
    k_tokenized = [preprocess_text(text) for text in k_texts]

    k_doc_lens = [len(k) for k in k_tokenized]
    avg_doc_len = sum(k_doc_lens) / len(k_doc_lens)
    
    bm25_scores = []
    for q in q_tokenized:
        q_scores = []
        for k in k_tokenized:
            score = bm25_score(q, k, idf, avg_doc_len=avg_doc_len)
            q_scores.append(score)
        bm25_scores.append(q_scores)
    
    return np.round(np.array(bm25_scores), decimals=6)


def compute_dense_matrix(q_texts, k_texts, dense_model_encode):

    embeddings_q = np.array(dense_model_encode(q_texts), dtype=np.float32)
    embeddings_k = np.array(dense_model_encode(k_texts), dtype=np.float32)
    # print(embeddings_q.shape, embeddings_k.shape, flush=True)

    similarity = embeddings_q @ embeddings_k.T

    return np.round(np.array(similarity), decimals=6)




