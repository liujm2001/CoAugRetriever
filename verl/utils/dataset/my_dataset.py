# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.reward_score.text_sim import compute_idf, preprocess_text

import pickle
import json
import pandas as pd
from pathlib import Path
import random
from collections import defaultdict

import re

def preprocess_data(queries_dir, corpus_dir, relations_dir_train, relations_dir_test):

    output_dir = os.path.dirname(queries_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    queries = {}
    with open(queries_dir) as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]
    
    corpus = {}
    with open(corpus_dir) as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["_id"]] = f"{obj['title']}. {obj['text']}"
    
    pd.DataFrame({
        "qid": list(queries.keys()),
        "text": list(queries.values())
    }).to_parquet(output_dir / "queries.parquet")
    
    pd.DataFrame({
        "cid": list(corpus.keys()),
        "text": list(corpus.values())
    }).to_parquet(output_dir / "corpus.parquet")


    relation_scores_train = defaultdict(dict)  # {qid: {cid: score}}
    train_df = pd.read_csv(relations_dir_train, sep='\t')
    for _, row in train_df.iterrows():
        relation_scores_train[str(row["query-id"])][str(row["corpus-id"])] = row["score"]
    
    relation_scores_test = defaultdict(dict)  # {qid: {cid: score}}
    test_df = pd.read_csv(relations_dir_test, sep='\t')
    for _, row in test_df.iterrows():
        relation_scores_test[str(row["query-id"])][str(row["corpus-id"])] = row["score"]
    
    with open(output_dir / "relation_scores_train.pkl", "wb") as f:
        pickle.dump(dict(relation_scores_train), f)
    with open(output_dir / "relation_scores_test.pkl", "wb") as f:
        pickle.dump(dict(relation_scores_test), f)
    
    print(len(queries.keys()), len(corpus.keys()), len(relation_scores_train.keys()), len(relation_scores_test.keys()), flush=True)
    print("preprocess_data done!")


class DynamicTagDataset(Dataset):
    def __init__(self,
                 parquet_dir: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_q_files: str,
                 prompt_k_files: str,
                 batch_cases=[1,2,5,8],
                 max_prompt_length=512,
                 truncation='error'
                 ):
        
        self.split = split
        # print(batch_cases, flush=True)

        with open(prompt_q_files, 'r') as f:
            self.prompt_q = f.read()
        with open(prompt_k_files, 'r') as f:
            self.prompt_k = f.read()

        self.q_num = batch_cases[0]
        self.k_pos_num = batch_cases[1]
        self.k_hardneg_num = batch_cases[2]
        self.k_randneg_num = batch_cases[3]

        query_file = os.path.join(parquet_dir, "queries.parquet")
        corpus_file = os.path.join(parquet_dir, "corpus.parquet")
        relation_file = os.path.join(parquet_dir, f"relation_scores_{split}.pkl")

        self.queries = pd.read_parquet(query_file)
        self.corpus = pd.read_parquet(corpus_file)
        with open(relation_file, "rb") as f:
            self.relation_scores = pickle.load(f)  # {qid: {cid: score}}

        corpus_list = [preprocess_text(text) for text in self.corpus['text']]
        self.idf = compute_idf(corpus_list)
        # print(len(self.idf), flush=True)

        valid_qids = set(self.relation_scores.keys())
        self.queries = self.queries[self.queries['qid'].isin(valid_qids)].reset_index(drop=True)

        self.q_map = {row.qid: row.text for row in self.queries.itertuples()}
        self.c_map = {row.cid: row.text for row in self.corpus.itertuples()}
        self.all_cids = self.corpus["cid"].tolist()
        
        self.q_relations = {
            qid: {
                "k_pos": [cid for cid, score in cids.items() if score != 0],
                "k_hardneg": [cid for cid, score in cids.items() if score == 0]
            }
            for qid, cids in self.relation_scores.items()
        }
        
        all_pos_cids = set()
        for relations in self.q_relations.values():
            all_pos_cids.update(relations["k_pos"])

        self.randneg_cids = list(set(self.all_cids) - all_pos_cids)


        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        self.next_cid = 1
        self.cid_str2num = {}

    def extract_numbers(self, text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        else:
            if text not in self.cid_str2num:
                self.cid_str2num[text] = self.next_cid
                self.next_cid += 1
            return self.cid_str2num[text]
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, q_idx):
        qid = self.queries.iloc[q_idx].qid
        qtext = self.q_map[qid]
        
        samples = []
        delta = 0
        
        samples.append({
            "text": qtext,
            "tag": "q",
            "score": 0,
            "qid": self.extract_numbers(qid),
            "cid": self.extract_numbers("0")
        })
        
        k_pos_list = self.q_relations.get(qid, {}).get("k_pos", [])
        delta += max(0, self.k_pos_num - len(k_pos_list))
        
        k_cids = random.choices(
            k_pos_list,
            k=self.k_pos_num - max(0, self.k_pos_num - len(k_pos_list))
        )
        for cid in k_cids:
            samples.append({
                "text": self.c_map[cid],
                "tag": "k_pos",
                "score": self.relation_scores[qid][cid],
                "qid": self.extract_numbers(qid),
                "cid": self.extract_numbers(cid)
            })
        
        v_pos_list = self.q_relations.get(qid, {}).get("k_hardneg", [])
        delta += max(0, self.k_hardneg_num - len(v_pos_list))

        v_cids = random.choices(
            v_pos_list,
            k=self.k_hardneg_num - max(0, self.k_hardneg_num - len(v_pos_list))
        )
        for cid in v_cids:
            samples.append({
                "text": self.c_map[cid],
                "tag": "k_hardneg",
                "score": 0,
                "qid": self.extract_numbers(qid),
                "cid": self.extract_numbers(cid)
            })
        
        w_cids = random.choices(
            self.randneg_cids, 
            k=self.k_randneg_num + delta
        )
        for cid in w_cids:
            score = self.relation_scores.get(qid, {}).get(cid, 0)
            samples.append({
                "text": self.c_map[cid],
                "tag": "k_randneg",
                "score": score,
                "qid": self.extract_numbers(qid),
                "cid": self.extract_numbers(cid)
            })
        
        random.shuffle(samples)
        for sample in samples:
            
            # system_prompt = "You are a helpful assistant. You can summarize the key content of the text and make supplementary associations based on your professional knowledge."
            # system_prompt = "You are an expert at classifying text topics and extracting keywords. "
            system_prompt = self.prompt_q if sample["tag"] == "q" else self.prompt_k
            prompt_with_template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{sample['text']}<|im_end|>\n<|im_start|>assistant\n"


            # qwen prompt template
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_template,
                                                                        tokenizer=self.tokenizer,
                                                                        max_length=self.max_prompt_length,
                                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation=self.truncation)

            position_ids = compute_position_id_with_mask(attention_mask)

            sample.update({
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0]
            })

        return samples



class DynamicBatchLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, drop_last=True, shuffle=True):
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.test_batch_num = 1    

        self.tag2int = {
            "q": 0,
            "k_pos": 1,
            "k_hardneg": 2,
            "k_randneg": 3
        }

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)  
        
        batch_indices = []
        for i in range(0, len(indices), self.batch_size):
            group = indices[i:i+self.batch_size]
            if not self.drop_last or len(group) == self.batch_size:
                batch_indices.append(group)
        
        if self.dataset.split == "test":
            batch_indices = random.sample(batch_indices, self.test_batch_num)
        
        for batch_group in batch_indices:
            batch = []
            for idx in batch_group:
                while True:
                    try:
                        item = self.dataset[idx] 
                        batch.append(item)
                        break  
                    except Exception as e:  
                        print(f"Error processing index {idx}: {str(e)}. Retrying...")   # discarding long text
                        idx = random.randint(0, len(self.dataset)-1)

            yield self.collate_fn(batch)

    def collate_fn(self, batch_list):
            
        batch = {
            "input_ids": torch.stack([s["input_ids"] for samples in batch_list for s in samples]),
            "attention_mask": torch.stack([s["attention_mask"] for samples in batch_list for s in samples]),
            "position_ids": torch.stack([s["position_ids"] for samples in batch_list for s in samples]),
            "qid": torch.tensor([s["qid"] for samples in batch_list for s in samples]),
            "cid": torch.tensor([s["cid"] for samples in batch_list for s in samples]),
            "tags": torch.tensor([self.tag2int[s["tag"]] for samples in batch_list for s in samples]),
            "scores": torch.tensor([s["score"] for samples in batch_list for s in samples]),
        }
        return batch
    