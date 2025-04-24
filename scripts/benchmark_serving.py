import time
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = os.environ.get('MODEL_NAME')
API_URL = "http://localhost:8000/v1/completions"

BENCHMARK_DATASET = [
    "Translate this English sentence to French: 'How are you?'",
    "What is the capital of Canada?",
    "Explain the concept of gravity in simple terms.",
    "Write a short poem about the ocean.",
    "What's the difference between Python and Java?",
    "Summarize the plot of Romeo and Juliet.",
    "Who won the FIFA World Cup in 2018?",
    "List three benefits of a healthy diet.",
    "Describe how a car engine works.",
    "What is machine learning?",
    "Write a haiku about the moon.",
    "Name five countries in Africa.",
    "Give an example of a metaphor.",
    "Explain the importance of the water cycle.",
    "Write a dialogue between a robot and a human about friendship.",
    "What causes earthquakes?",
    "Describe the process of photosynthesis.",
    "What are the advantages of renewable energy?",
    "Define artificial intelligence in one sentence.",
    "Generate a creative story starter about time travel."
]

# vLLM HTTP request
def send_request_vllm(prompt, max_tokens=50, temperature=0.7, logprobs=5):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

# Benchmark vLLM
def benchmark_vllm(tokenizer):
    total_tokens = 0
    total_time = 0
    logprobs_supported = False
    generations = []

    print("\nRunning vLLM HTTP benchmark...")

    for prompt in BENCHMARK_DATASET:
        start = time.time()
        response = send_request_vllm(prompt)
        end = time.time()

        result = response["choices"][0]
        output_text = result["text"]
        tokens = tokenizer.encode(output_text)
        total_tokens += len(tokens)
        total_time += (end - start)

        generations.append({
            "prompt": prompt,
            "output": output_text.strip()
        })

        if "logprobs" in result and result["logprobs"]:
            logprobs_supported = True

    return {
        "engine": "vLLM",
        "total_prompts": len(BENCHMARK_DATASET),
        "total_tokens": total_tokens,
        "average_speed_tps": total_tokens / total_time,
        "logprobs_available": logprobs_supported,
        "generations": generations
    }

# Benchmark Hugging Face Transformers
def benchmark_huggingface(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).half().cuda()
    model.eval()
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    total_tokens = 0
    total_time = 0
    vram_peaks = []
    generations = []

    print("\nRunning Hugging Face benchmark...")

    for prompt in BENCHMARK_DATASET:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        output = hf_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        end = time.time()

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        vram_peaks.append(peak_vram)

        text = output[0]["generated_text"]
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_time += (end - start)

        generations.append({
            "prompt": prompt,
            "output": text.strip()
        })

    avg_vram = sum(vram_peaks) / len(vram_peaks)

    return {
        "engine": "HuggingFace",
        "total_prompts": len(BENCHMARK_DATASET),
        "total_tokens": total_tokens,
        "average_speed_tps": total_tokens / total_time,
        "average_vram_gb": avg_vram,
        "logprobs_available": False,
        "generations": generations
    }

# Run both benchmarks + print results
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    vllm_results = benchmark_vllm(tokenizer)
    hf_results = benchmark_huggingface(tokenizer)

    print("\n=== Benchmark Summary ===")
    for result in [vllm_results, hf_results]:
        print(f"\nEngine: {result['engine']}")
        print(f"Prompts Tested: {result['total_prompts']}")
        print(f"Total Tokens Generated: {result['total_tokens']}")
        print(f"Average Speed: {result['average_speed_tps']:.2f} tokens/sec")
        if "average_vram_gb" in result:
            print(f"Average Peak VRAM: {result['average_vram_gb']:.2f} GB")
        print(f"Logprobs Available: {'Yes' if result['logprobs_available'] else 'No'}")

    print("\n=== Output Comparison ===")
    for i, prompt in enumerate(BENCHMARK_DATASET):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"vLLM: {vllm_results['generations'][i]['output']}")
        print(f"HF  : {hf_results['generations'][i]['output']}")

if __name__ == "__main__":
    main()