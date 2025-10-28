from dataclasses import dataclass
from transformers import AutoConfig
import os

@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 512
    eos: int = -1
    hf_config: AutoConfig | None = None,

    def __post__init__(self):
        os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)