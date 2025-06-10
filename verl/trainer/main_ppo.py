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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score.text_sim import compute_bm25_matrix, compute_dense_matrix, preprocess_text
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import re
import numpy as np
from collections import defaultdict
import time 

from FlagEmbedding import FlagAutoModel

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, config, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

        self.config = config
        # if self.config.use_dense:
        #     self.dense_model = FlagAutoModel.from_finetuned(
        #         self.config.embedmodel_path,
        #         query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        #         torch_dtype="auto",
        #         # devices="auto"
        #     )

    
    def set_idf(self, idf):
        self.idf = idf

    def extract_and_combine(self, text):

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

    def calculate_ndcg(self, bm25_submat, subscores, n=None):
        
        assert len(bm25_submat) == len(subscores)
        if n is None:
            n = len(bm25_submat)
        n = min(n, len(bm25_submat))
        
        sorted_indices = np.argsort(-bm25_submat, kind='stable')
        sorted_scores = subscores[sorted_indices][:n]
        
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(sorted_scores))
        
        ideal_sorted = np.sort(subscores, kind='stable')[::-1][:n] 
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_sorted))
        
        return dcg / idcg if idcg != 0 else 0.0


    def __call__(self, data: DataProto, dense_model_encode=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        print(len(data), flush=True)              # 80 = 5 * 16

        q_data_id = []
        k_data_id = []
        q_data_str = []
        k_data_str = []
        q_qid = []
        k_qid = []
        k_cid = []
        k_scores = []

        format_reward = np.zeros(len(data))

        selected_output_qid = data[0].batch['qid'] # for output cases
        valid_response_length_list = []

        # decode
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_length_list.append(valid_response_length)

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            extracted_dict = self.extract_and_combine(sequences_str)   

            for tmp_key in extracted_dict:
                extracted_dict[tmp_key] = extracted_dict[tmp_key].replace("Explanation: ", "").replace("Keywords: ", "").replace("Summary: ", "").replace("Title: ", "").replace("Abstract: ", "")


            # format reward here
            result_str = extracted_dict['user'] + '\n'
            if not self.config.wuser:
                result_str = ''

            format_penanty = self.config.format_penalty 
            if 'think' in extracted_dict.keys():
                pass   
            else:
                format_reward[i] += format_penanty
            if 'answer' in extracted_dict.keys():
                if data_item.batch['tags'] == 0 and not self.config.k_only:
                    result_str = result_str + '\n' + extracted_dict['answer']
                elif data_item.batch['tags'] != 0 and not self.config.q_only:
                    result_str = result_str + '\n' + extracted_dict['answer']
            else:
                format_reward[i] += format_penanty

            if self.num_examine and data_item.batch['qid'] == selected_output_qid:
                print(data_item.batch['qid'], ' ', data_item.batch['cid'], ' ', format_reward[i], '\n', sequences_str, '\n')

            if data_item.batch['tags'] == 0:
                q_qid.append(data_item.batch['qid'].item())
                q_data_id.append(i)
                q_data_str.append(result_str)
            else:
                k_data_id.append(i)
                k_data_str.append(result_str)
                k_qid.append(data_item.batch['qid'].item())
                k_cid.append(data_item.batch['cid'].item())
                k_scores.append(data_item.batch['scores'].item())

        if not self.config.use_dense:
            sim_mat = compute_bm25_matrix(q_data_str, k_data_str, self.idf)
        else:
            sim_mat = compute_dense_matrix(q_data_str, k_data_str, dense_model_encode)

        k_scores = np.array(k_scores, dtype=np.float32)
        k_scores_mat = np.zeros_like(sim_mat, dtype=np.float32)
        reward_mat = torch.zeros(len(q_data_id), len(k_data_id))
        cnt_mat = torch.zeros(len(q_data_id), len(k_data_id))


        id_to_columns = defaultdict(lambda: defaultdict(list))
        for col_idx in range(len(k_data_id)):
            id_to_columns[k_qid[col_idx]][k_cid[col_idx]].append(col_idx)

        for row_idx in range(len(q_data_id)):
            for cid_key in id_to_columns[q_qid[row_idx]]:
                k_scores_mat[row_idx, id_to_columns[q_qid[row_idx]][cid_key]] = k_scores[id_to_columns[q_qid[row_idx]][cid_key]]

        n_ndcg = 10
        sampling_num = 1 if self.num_examine else len(data) * 64


        if self.config.sampling_method == 'sampling':
            # sampling here  
            for sampling_cnt in range(sampling_num):
                row_idx = np.random.randint(0, len(q_data_id))
                selected_columns = []
                for cid_key in id_to_columns[q_qid[row_idx]]:
                    selected_columns.append(np.random.choice(id_to_columns[q_qid[row_idx]][cid_key]))

                # print(row_idx, selected_columns, flush=True)

                bm25_submat = sim_mat[row_idx, selected_columns]
                k_subscores = k_scores[selected_columns]
                
                reward_ndcg = self.calculate_ndcg(bm25_submat, k_subscores, n_ndcg) 

                # print(bm25_submat, '\n-\n', k_subscores, '\n--\n', reward_ndcg, '\n---\n', flush=True)

                ## only supports 1 q in mircobatch here - without k_tag
                reward_mat[row_idx, selected_columns] += reward_ndcg
                cnt_mat[row_idx, selected_columns] += 1

                # print('\n\n\n\n\n', reward_mat, cnt_mat, flush=True)

        elif self.config.sampling_method == 'batch_sampling':
            # batch_sampling here  
            # 2560 // 160 = 160
            # 2560 * (1*15) => *64 
            # 160 * (32*5 * 480) 12800000 => *64 // 2(microbsz = 8)
            for sampling_cnt in range(max(sampling_num // len(q_data_id) // 2, 1)):

                selected_columns = []
                for qid_key in id_to_columns:
                    for cid_key in id_to_columns[qid_key]:
                        selected_columns.append(np.random.choice(id_to_columns[qid_key][cid_key]))

                for row_idx in range(len(q_data_id)):

                    # print(row_idx, selected_columns, flush=True)

                    bm25_submat = sim_mat[row_idx, selected_columns]
                    k_subscores = k_scores_mat[row_idx, selected_columns]
                    
                    reward_ndcg = self.calculate_ndcg(bm25_submat, k_subscores, n_ndcg) 

                    # print(bm25_submat, '\n-\n', k_subscores, '\n--\n', reward_ndcg, '\n---\n', flush=True)

                    reward_mat[row_idx, selected_columns] += reward_ndcg
                    cnt_mat[row_idx, selected_columns] += 1
                    
                    # k_pos reweight reward
                    kpos_reweight = self.config.kpos_reweight
                    reward_mat[row_idx, selected_columns] += reward_ndcg * k_subscores * kpos_reweight
                    cnt_mat[row_idx, selected_columns] += k_subscores * kpos_reweight

                    

                    # print('\n\n\n\n\n', reward_mat, cnt_mat, flush=True)

        # validate
        if self.num_examine:
            print(f'reward: {reward_ndcg}\n')
            for i in range(len(data)):
                valid_response_length = valid_response_length_list[i]
                reward_tensor[i, valid_response_length - 1] = reward_ndcg + format_reward[i]
            return reward_tensor

        # reward func
        scores = torch.zeros(len(data))

        for qid in range(len(q_data_id)):
            scores[q_data_id[qid]] = 0.0 if cnt_mat[qid].sum() == 0 else reward_mat[qid].sum() / cnt_mat[qid].sum()
        for kid in range(len(k_data_id)):
            scores[k_data_id[kid]] = 0.0 if cnt_mat[:, kid].sum() == 0 else reward_mat[:, kid].sum() / cnt_mat[:, kid].sum()

        for i in range(len(data)):
            if data[i].batch['tags'] == 0 and self.config.k_only:
                scores[i] = 0.0
            elif data[i].batch['tags'] != 0 and self.config.q_only:
                scores[i] = 0.0
            valid_response_length = valid_response_length_list[i]
            if data[i].batch['tags'] == 0:
                reward_tensor[i, valid_response_length - 1] = scores[i] + format_reward[i]
            elif data[i].batch['tags'] == 1:
                reward_tensor[i, valid_response_length - 1] = scores[i] + format_reward[i]
            elif data[i].batch['tags'] == 3:
                reward_tensor[i, valid_response_length - 1] = scores[i] + format_reward[i]
            else:
                # retired
                reward_tensor[i, valid_response_length - 1] = scores[i] + format_reward[i]

        return reward_tensor


import ray
import hydra
from datetime import datetime
import socket
import os
log_path = './log'

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        with open(f"{log_path}/ray_init.log", "a") as f:
            f.write(f"\n===== Ray Initialization at {datetime.now()} =====\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"Environment Variables:\n")
            f.write(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'NOT_SET')}\n")
            f.write(f"  NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'NOT_SET')}\n")
        
        ray.init(
            runtime_env={'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'INFO', 
                'NCCL_DEBUG_FILE': f"{log_path}/nccl_debug.log"
            }},
            # _temp_dir=f"{log_path}",
        )
        # ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        # ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', "PYTHONWARNINGS": "default"}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data

    # if config.reward_model.enable:
    if config.reward_model.rule_based.use_dense:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, config=config.reward_model.rule_based, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, config=config.reward_model.rule_based, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
