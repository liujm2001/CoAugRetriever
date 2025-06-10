#! /usr/bin/env bash

### parameters
dataset_name='nfcorpus'
model_name='Qwen2.5-7B-Instruct'
use_dense='False'
### 

split='test'

cd ./beir

if [ "$use_dense" = 'True' ]; then
    python beir/run.py --dataset_name "${dataset_name}" --model_name "${model_name}" --split "${split}" --dense
    echo "Result of ${model_name}"
    python examples/retrieval/evaluation/lexical/evaluate_dense.py --dataset_name "${dataset_name}" --model_name "${model_name}" --split "${split}"
else
    python beir/run.py --dataset_name "${dataset_name}" --model_name "${model_name}" --split "${split}"
    echo "Result of ${model_name}"
    python examples/retrieval/evaluation/lexical/evaluate_bm25.py --dataset_name "${dataset_name}" --model_name "${model_name}" --split "${split}"
fi

cd ..

exit




