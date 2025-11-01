import os
import torch.nn as nn
from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.config import Config
from transformers import AutoConfig
from glob import glob
from safetensors import safe_open


def load_model(model: nn.Module, model_path: str):
    """
    phase 1: simple weight loading (single gpu, no packing)
    load pre-trained weights to our model

    Args:
        model: Our Qwen3ForCausalLM instance
        model_path: Path to HF model (local or HF Hub name)
    """
    # download the model weights from huggingface hub if needed
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        print(f'downloading {model_path}...')
        model_path = snapshot_download(model_path)

    # load weights
    for file in glob(os.path.join(model_path, '*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                try:
                    param = model.get_parameter(name)
                    param.data.copy_(f.get_tensor(name))
                except AttributeError:
                    print(f'unable to find the param {name}')
                    continue

def compare_model_params(my_model: nn.Module, hf_model_path: str):
    """
    Compare parameter names between your model and HuggingFace checkpoint.
    
    Args:
        my_model: Your Qwen3ForCausalLM instance
        hf_model_path: Path to HF model (local or HF Hub name like "Qwen/Qwen2.5-0.5B")
    """
    # Download if needed
    if not os.path.exists(hf_model_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading {hf_model_path}...")
        hf_model_path = snapshot_download(hf_model_path)
    
    # Collect HF checkpoint names
    print("=" * 80)
    print("COLLECTING HUGGINGFACE CHECKPOINT WEIGHTS")
    print("=" * 80)
    
    hf_param_names = []
    print(os.path.join(hf_model_path, "*.safetensors"))
    for file in sorted(glob(os.path.join(hf_model_path, "*.safetensors"))):
        print(f"\nReading: {os.path.basename(file)}")
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                hf_param_names.append(name)
    
    hf_param_names = sorted(hf_param_names)
    print(f"\nTotal HF checkpoint parameters: {len(hf_param_names)}")
    
    # Collect your model names
    print("\n" + "=" * 80)
    print("COLLECTING YOUR MODEL PARAMETERS")
    print("=" * 80)
    
    my_param_names = sorted([name for name, _ in my_model.named_parameters()])
    print(f"Total your model parameters: {len(my_param_names)}")
    
    # Side-by-side comparison
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    print(f"{'HF CHECKPOINT':<60} | {'YOUR MODEL':<60}")
    print("-" * 80)
    
    max_len = max(len(hf_param_names), len(my_param_names))
    for i in range(max_len):
        hf_name = hf_param_names[i] if i < len(hf_param_names) else ""
        my_name = my_param_names[i] if i < len(my_param_names) else ""
        
        # Color code matches
        match_symbol = "✓" if hf_name == my_name else "✗"
        print(f"{hf_name:<60} | {my_name:<60} {match_symbol}")
    
    # Summary
    print("\n" + "=" * 80)
    print("MATCHING ANALYSIS")
    print("=" * 80)
    
    hf_set = set(hf_param_names)
    my_set = set(my_param_names)
    
    matching = hf_set & my_set
    only_in_hf = hf_set - my_set
    only_in_mine = my_set - hf_set
    
    print(f"✓ Matching parameters: {len(matching)}/{len(hf_param_names)}")
    print(f"✗ Only in HF checkpoint: {len(only_in_hf)}")
    print(f"✗ Only in your model: {len(only_in_mine)}")
    
    if only_in_hf:
        print(f"\n⚠ Parameters in HF but NOT in your model (first 10):")
        for name in sorted(only_in_hf)[:10]:
            print(f"  - {name}")
        if len(only_in_hf) > 10:
            print(f"  ... and {len(only_in_hf) - 10} more")
    
    if only_in_mine:
        print(f"\n⚠ Parameters in YOUR model but NOT in HF (first 10):")
        for name in sorted(only_in_mine)[:10]:
            print(f"  - {name}")
        if len(only_in_mine) > 10:
            print(f"  ... and {len(only_in_mine) - 10} more")
    
    print("\n" + "=" * 80)
    if len(matching) == len(hf_param_names) == len(my_param_names):
        print("🎉 PERFECT MATCH! All parameter names align.")
    else:
        print("❌ MISMATCH DETECTED! Fix your model architecture naming.")
    print("=" * 80)
    
    return {
        'hf_params': hf_param_names,
        'my_params': my_param_names,
        'matching': matching,
        'only_in_hf': only_in_hf,
        'only_in_mine': only_in_mine
    }

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading model {model_name}...")
    config = Config(model_name)
    hf_config = AutoConfig.from_pretrained(config.model)
    model = Qwen3ForCausalLM(hf_config)
    result = compare_model_params(model, model_name)
    print(result)