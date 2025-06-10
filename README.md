# CoAugRetriever

## env init
```
conda create -n CoAugRetriever python=3.10 -y
conda activate CoAugRetriever

pip install -r requirements.txt
pip install flash-attention==2.7.4.post1
pip install -e ./beir
```

## ckpts and datasets
```
cd ./ckpt
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
cd ..
bash ./scripts/download_dataset.sh
```

## train
```
bash ./scripts/run_train.sh
```

## test
```
bash ./scripts/run_test.sh
```

## todo
readme
license
prompt_q k dense
