"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch.

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from collections import Counter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='nfcorpus')
parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--wuser', type=int, default=1)

args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

### parms
SAVE_LOG = False

#### Download scifact.zip dataset and unzip the dataset
model_name = args.model_name.replace('/', '_')
dataset = args.dataset_name
data_path = f'../dataset/{dataset}'

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

'''
corpus: dict[str, dict[str, str]],
queries: dict[str, str],
self.corpus[line.get("_id")] = {
    "text": line.get("text"),
    "title": line.get("title"),
}
'''


import re
import json

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
        queries[q_key] += ' ' + queries_llm[q_key]['keywords']
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
        corpus[k_key]['title'] += ' ' + corpus_llm[k_key]['keywords']
else:
    print(f'corpus_llm_{model_name}.json not exists')
    assert 0


#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "http://localhost:9200"  # localhost
index_name = args.dataset_name 

#### Intialize ####
# (1) True - Delete existing index and re-index all documents from scratch
# (2) False - Load existing index
initialize = True  # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
print(BM25.__module__)
print(BM25.__dict__)
model = BM25(
    index_name=index_name,
    hostname=hostname,
    initialize=initialize,
    number_of_shards=number_of_shards,
)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

os.system('curl -X DELETE "http://localhost:9200/*"')


#### Retrieval Example ####
'''
query_id, scores_dict = random.choice(list(results.items()))
logging.info(f"Query : {queries[query_id]}\n")

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(min(10,len(scores_dict.items()))):
    doc_id = scores[rank][0]
    logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")
'''
