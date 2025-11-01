import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_name = 'Qwen/Qwen3-0.6B'
    # model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(model_name)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts = [
        "The capital of India is",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


    print(prompts)

if __name__ == "__main__":
    main()
