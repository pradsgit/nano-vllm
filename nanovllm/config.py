from dataclasses import dataclass

class Config:
    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 512
    eos: int = -1