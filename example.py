from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

def main():

    try:
        model_name = 'Qwen/Qwen3-0.6B'
        # model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        llm = LLM(model_name)
        sp = SamplingParams(temperature=0.6, max_tokens=256)

        prompts = [
            "Introduce yourself",
            "The capital of India is"
        ]

        # if you dont apply chat template, the model is invoked as a base model instead of an SFTed model
        # it just completes next token. we dont know how this behavior is brought about in a model.
        # the difference between a chat_template prompt and non-chat is just few additional special tokens in chat_temp prompt.

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        output = llm.generate(prompts, sp)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()  # Prints full stack trace
    finally:
        if 'llm' in locals():
            llm.exit()


if __name__ == "__main__":
    main()