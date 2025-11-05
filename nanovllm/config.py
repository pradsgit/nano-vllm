from dataclasses import dataclass
from transformers import AutoConfig
import os

@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 512
    eos: int = None
    hf_config: AutoConfig | None = None
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(
                self.model,
                trust_remote_code=True,
            )
        
        if self.eos is None:
            self.eos = getattr(self.hf_config, 'eos_token_id', None)