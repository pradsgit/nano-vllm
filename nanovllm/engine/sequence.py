# Each Sequence class instance acts as a 'state machine' for prompt request
from enum import Enum
from itertools import count
from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED = 'finished'

class Sequence:
    counter = count()

    def __init__(self, prompt_tokens: list[int], sampling_params: SamplingParams):
        # how should we maintain id? a simple counter? or string?
        self.id = f'seq_{next(Sequence.counter)}'
        self.status = SequenceStatus.WAITING

        self.prompt_tokens = prompt_tokens.copy()
        self.output_tokens = []
        self.num_computed_tokens = 0 # num tokens with saved kv_cache
        self.num_scheduled_tokens = 0 # tokens scheduled to process this step
        # memory management for PagedAttention
        self.block_table = [] # this stores allocated block_ids

        # sampling params
        self.sampling_params = sampling_params

    def __len__(self):
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def last_token(self) -> int:
        """most recently added token"""
        if self.output_tokens:
            return self.output_tokens[-1]

        return self.prompt_tokens[-1]

    @property
    def is_prefill(self):
        """
        if this is a prefill seqeunce request
        """
        return len(self.output_tokens) == 0

    @property
    def token_ids(self) -> list[int]:
        """All tokens (prompt + output)."""
        return self.prompt_tokens + self.output_tokens

    @property
    def num_tokens(self) -> int:
        """Returns total number of prompt and output tokens"""
        return len(self.prompt_tokens) + len(self.output_tokens)

    def add_token(self, token_id: int) -> None:
        """Adds new token to output_tokens"""
        self.output_tokens.append(token_id)

    def get_num_logical_blocks(self, block_size: int) -> int:
        """Calculate number of blocks needed for current token length."""
        # Ceiling division: (num_tokens + block_size - 1) // block_size
        return (self.num_tokens + block_size - 1) // block_size

    def __repr__(self) -> str:
        return (f"Sequence(id={self.id}, status={self.status}, "
                f"tokens={self.num_tokens}, blocks={len(self.block_table)})")
