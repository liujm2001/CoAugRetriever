import logging
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os
from beir import util

import re
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='nfcorpus')
parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--wuser', type=int, default=1)
args = parser.parse_args()

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

### parms
SAVE_LOG = False

model_name = args.model_name.replace('/', '_')
dataset = args.dataset_name
data_path = f'./dataset/{dataset}'
dense_model_path = "./ckpt/bge-base-en-v1.5"

corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split, getsth=False)
# Load bge model using Sentence Transformers
model = DRES(models.SentenceBERT(dense_model_path), batch_size=128)

def readfile(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

if os.path.exists(f"{data_path}/queries_llm_{model_name}.json"):
    with open(f'{data_path}/queries_llm_{model_name}.json', 'r') as file:
        queries_llm = json.load(file)
    for q_key in queries.keys():
        if not args.wuser:
            queries[q_key] = ""
        queries[q_key] = queries[q_key] + '\n' + queries_llm[q_key]['keywords'] 
else:
    print(f'queries_llm_{model_name}.json not exists')
    assert 0

if os.path.exists(f"{data_path}/corpus_llm_{model_name}.json"):
    with open(f'{data_path}/corpus_llm_{model_name}.json', 'r') as file:
        corpus_llm = json.load(file)
    for k_key in corpus.keys():
        if not args.wuser:
            corpus[k_key]['title'] = ""
            corpus[k_key]['text'] = ""
        corpus[k_key]['title'] = corpus[k_key]['title'] + '\n'
        corpus[k_key]['text'] = corpus_llm[k_key]['keywords'] + '\n' + corpus[k_key]['text'] + '\n'
else:
    print(f'corpus_llm_{model_name}.json not exists')
    assert 0


retriever = EvaluateRetrieval(model, score_function="cos_sim")

# Get the searching results
results = retriever.retrieve(corpus, queries)

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
