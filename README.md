# CoAugRetriever
Official Code for "Harnessing the Power of Reinforcement Learning for Language-Model-Based Information Retriever via Query-Document Co-Augmentation" (ArXiv Preprint xxx)

## Installation
```
conda create -n CoAugRetriever python=3.10 -y
conda activate CoAugRetriever

pip install -r requirements.txt
pip install flash-attention==2.7.4.post1
pip install -e ./beir
```

## Dataset & Checkpoint Download
You can use the following script to download the beir dataset and qwen model to start training, or you can use your own prepared data. 
For dense retrieval, you also need to download the encoding model and put it in './ckpt/'.

```
cd ./ckpt
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
cd ..
bash ./scripts/download_dataset.sh
```

## Run training
You can start training directly with the following script. The default environment is a single node with 8 GPUs. The parameter configuration for training can be adjusted in it.
We follow the architecture of TinyZero to implement this work. You can find the entry of the training pipeline in './verl/trainer/main_ppo.py' and './verl/trainer/ppo/ray_trainer.py'.
```
bash ./scripts/run_train.sh
```

## Run Evaluation
You can use the following script to perform performance evaluation. We follow Beir's evaluation method. For sparse retrieval, you need to start the Elastic Search service locally to support BM25.
```
bash ./scripts/run_test.sh
```

