python train_grpo.py
python test.py

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

lm_eval --model hf \
    --model_args pretrained=Qwen2.5-0.5B-GRPO/checkpoint-100 \
    --tasks gpqa \
    --device cuda:0 \
    --batch_size 8

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
    --tasks gpqa \
    --device cuda:0 \
    --batch_size 8