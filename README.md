
# On-Premise LLM Benchmark: vLLM vs Hugging Face

This project compares the performance, speed, VRAM usage, and output similarity between two deployment approaches for the BLOOMZ-1B1 language model:

- vLLM: Fast, efficient serving via HTTP
- Hugging Face Transformers: Direct inference on GPU with PyTorch

---

## Why vLLM?

[vLLM](https://github.com/vllm-project/vllm) is an open-source high-throughput and memory-efficient LLM inference engine.

Advantages of vLLM:

- Fast inference with continuous batching and paged attention
- Low VRAM usage even for large models
- HTTP serving support (OpenAI-compatible API)
- Supports logprobs and streaming (optional)

---

## Running vLLM with Docker
Before running the vLLM server, make sure to download the model:

    bash scripts/download.sh

You can spin up a vLLM server using Docker:

    docker compose up

---

## 📡 Example API Usage (vLLM Server)

    import requests

    response = requests.post(
        "http://localhost:8000/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "bigscience/bloomz-1b1",
            "prompt": "What is the capital of Canada?",
            "max_tokens": 50,
            "temperature": 0.7
        }
    )

    print(response.json()["choices"][0]["text"])

---

## Benchmark: vLLM vs Hugging Face

### Step 1: Install dependencies

    pip install -r requirements.txt


### Step 2: Run benchmark

    python scripts/benchmark_serving.py

This will:

- Send 20 prompts to both vLLM (HTTP) and Hugging Face Transformers
- Measure:
  - Speed (tokens per second)
  - VRAM usage (Hugging Face only)

---

## Output

In the terminal:

- Summary stats for both engines
- Side-by-side prompt + completions

---

## License

MIT License. Free to use and modify.
