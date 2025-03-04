from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from FlagEmbedding import BGEM3FlagModel
import torch

dataset = load_dataset("simplescaling/s1K-1.1", split="train")

def prepare_dataset(example):
    return {
        "prompt": "Human: " + example["question"] + "\n\nAI: ",
        "deepseek_thinking_trajectory": example["deepseek_thinking_trajectory"],
        "deepseek_attempt": example["deepseek_attempt"]
    }

dataset = dataset.map(prepare_dataset)

similarity_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def reward_fn(completions, prompts, **kwargs):
    deepseek_thinking = kwargs['deepseek_thinking_trajectory']
    deepseek_attempt = kwargs['deepseek_attempt']
    
    rewards = []
    for i, completion_text in enumerate(completions):
        reference_text = f"{deepseek_thinking[i]}\n\n{deepseek_attempt[i]}"
        completion_emb = similarity_model.encode(
            [completion_text], 
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
            
        reference_emb = similarity_model.encode(
            [reference_text], 
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
            
        sim = torch.nn.functional.cosine_similarity(
            torch.tensor(completion_emb['dense_vecs']), 
            torch.tensor(reference_emb['dense_vecs'])
        ).item()
        rewards.append(sim)
    return rewards

training_args = GRPOConfig(
    output_dir="Qwen2.5-0.5B-GRPO",
    logging_steps=1,
    max_completion_length=512,
    max_prompt_length=512,
    max_steps=100,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="Qwen2.5-0.5B-GRPO/checkpoint-100",
    repo_id="notlober/kaim",
    repo_type="model",
)