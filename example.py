from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

def main():
    try:
        model_name = 'Qwen/Qwen3-0.6B'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        llm = LLM(model_name)
        sp = SamplingParams(temperature=0.6, max_tokens=256)

        prompts = [
            "Introduce yourself",
            "The capital of India is"
        ]

        # if you dont apply chat template, the model is invoked as a base model instead of an SFTed model
        # it just completes next token. we dont know how this behavior is brought about in a model.
        # the difference between a chat_template prompt and non-chat is just few additional special tokens in prompt.

        # apply chat_template for each prompt
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
        traceback.print_exc()
    finally:
        if 'llm' in locals():
            llm.exit()


if __name__ == "__main__":
    main()