from transformers import AutoTokenizer, pipeline

model_path = "Qwen2.5-0.5B-GRPO/checkpoint-100"
tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = pipeline("text-generation", model=model_path, tokenizer=tokenizer)

test_prompts = [
    "how many r's in strawberry"
]

print("\n===== TESTING FINE-TUNED MODEL =====\n")

for i, prompt in enumerate(test_prompts):
    print(f"Prompt {i+1}: {prompt}")
    
    result = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"Generation: {result[0]['generated_text'][len(prompt):]}\n")
    print("-" * 80 + "\n")