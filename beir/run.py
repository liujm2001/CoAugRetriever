from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.utils import get_distributed_init_method
import torch
import json
from tqdm import tqdm
import csv
import re 
import ray
    
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='nfcorpus')
parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--dense', action='store_true')

args = parser.parse_args()

### params ###
dataset_path = f'./dataset/{args.dataset_name}'
queries_files = f'{dataset_path}/queries.jsonl'
corpus_files = f'{dataset_path}/corpus.jsonl'
test_files = f'{dataset_path}/qrels/{args.split}.tsv'

is_dense = '_dense' if args.dense else ''

prompt_q_files = './dataset/prompt_q' + is_dense
prompt_k_files = './dataset/prompt_k' + is_dense
model_path = f'./ckpt/{args.model_name}'

GPU_MEMORY_UTILIZATION = 0.9
TENSOR_PARALLEL_SIZE = 1
DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

def init_ray_and_params():
    
    if not ray.is_initialized():
        ray.init()
        
    print(ray.available_resources())
    print(ray.nodes())

    nodes = [node for node in ray.nodes() if node["Alive"]]
    num_nodes = len(nodes)
    num_gpu = int(nodes[0]["Resources"].get("GPU", 0)) 
    
    sampling_params = SamplingParams(
        seed=42,
        temperature=0.0,
        max_tokens=2048,
        repetition_penalty=1.2,
        stop=["<|im_end|>"], 
        skip_special_tokens=True
    )

    return num_nodes, num_gpu, sampling_params

def extract_and_combine(text):

    result = {}
    pattern = r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>'
    blocks = re.findall(pattern, text, re.DOTALL)
    for role, content in blocks:
        if role in ['user', 'assistant']:
            cleaned_content = content.strip()
            result[role] = cleaned_content

    think_pattern = r'<think>(.*?)</think>'
    blocks = re.findall(think_pattern, text, re.DOTALL)
    if len(blocks) == 2:
        content = blocks[1]
        cleaned_content = content.strip()
        result['think'] = cleaned_content

    answer_pattern = r'<answer>(.*?)</answer>'
    blocks = re.findall(answer_pattern, text, re.DOTALL)
    if len(blocks) == 2:
        content = blocks[1]
        cleaned_content = content.strip()
        result['answer'] = cleaned_content

    return result

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    return tokens


@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE, num_cpus=TENSOR_PARALLEL_SIZE*8)
def batch_inference(sampling_params, dict_text, system_prompt):
    
    dict_llm = {}

    prompts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{dict_text[key]}<|im_end|>\n<|im_start|>assistant\n" 
        for key in dict_text.keys()
    ]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=DTYPE,
        enforce_eager=True 
    )
    outputs = llm.generate(prompts, sampling_params)
    
    for id in range(len(outputs)):
        key = list(dict_text.keys())[id]
        prompt = prompts[id]
        output = outputs[id]
        result = extract_and_combine(prompt + output.outputs[0].text)

        for tmp_key in result:
            result[tmp_key] = result[tmp_key].replace("Explanation: ", "").replace("Keywords: ", "").replace("Summary: ", "").replace("Title: ", "").replace("Abstract: ", "")
    
        dict_llm[key] = {'keywords': result['answer'] if 'answer' in result else ''}

    return dict_llm

def load_data(corpus_files):

    queries = {}
    with open(queries_files, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")

    corpus = {}
    num_lines = sum(1 for i in open(corpus_files, "rb"))
    with open(corpus_files, encoding="utf8") as fIn:
        for line in tqdm(fIn, total=num_lines):
            line = json.loads(line)
            corpus[line.get("_id")] = f"{line.get('text')}"

    qrels = {}
    reader = csv.reader(
        open(test_files, encoding="utf-8"),
        delimiter="\t",
        quoting=csv.QUOTE_MINIMAL,
    )
    next(reader)

    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score

    queries = {qid: queries[qid] for qid in qrels}
    return queries, corpus, qrels

if __name__ == "__main__":
    
    num_nodes, num_gpus, sampling_params = init_ray_and_params()
    num_nodes *= (num_gpus // TENSOR_PARALLEL_SIZE)

    with open(prompt_q_files, 'r') as f:
        system_prompt_q = f.read()
    with open(prompt_k_files, 'r') as f:
        system_prompt_k = f.read()

    queries, corpus, qrels = load_data(corpus_files)
    model_name_replaced = args.model_name.replace('/', '_')

    queries_items = list(queries.items())
    queries_chunks = [dict(queries_items[i::num_nodes]) for i in range(num_nodes)]
    
    queries_futures = []
    for queries_chunk in queries_chunks:
        queries_future = batch_inference.remote(sampling_params, queries_chunk, system_prompt_q)
        queries_futures.append(queries_future)

    queries_results = ray.get(queries_futures)
    queries_llm = {}
    for idx in range(num_nodes):
        queries_llm.update(queries_results[idx])

    with open(f"{dataset_path}/queries_llm_{model_name_replaced}.json", "w") as file:
        json.dump(queries_llm, file, indent=4)
    print(f'queries_llm_{model_name_replaced} done')

    corpus_items = list(corpus.items())
    corpus_chunks = [dict(corpus_items[i::num_nodes]) for i in range(num_nodes)]
    
    corpus_futures = []
    for corpus_chunk in corpus_chunks:
        corpus_future = batch_inference.remote(sampling_params, corpus_chunk, system_prompt_k)
        corpus_futures.append(corpus_future)

    corpus_results = ray.get(corpus_futures)
    corpus_llm = {}
    for idx in range(num_nodes):
        corpus_llm.update(corpus_results[idx])

    with open(f"{dataset_path}/corpus_llm_{model_name_replaced}.json", "w") as file:
        json.dump(corpus_llm, file, indent=4)
    print(f'corpus_llm_{model_name_replaced} done')
        
