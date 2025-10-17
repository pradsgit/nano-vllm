# Each Sequence class instance acts as a 'state machine' for prompt request
from enum import Enum, auto
from itertools import counter
from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED = 'finished'

class Sequence:
    count = counter()

    def __init__(self, prompt_tokens: list[int], sampling_params: SamplingParams):
        # how should we maintain id? a simple counter? or string?
        self.id = f'seq_{next(Sequence.counter)}'
        self.status = SequenceStatus.WAITING

        self.prompt_tokens = prompt_tokens.copy()
        self.output_tokens = []

        # memory management for PagedAttention
        self.block_table = []

        # sampling params
        self.smapling_params = sampling_params

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

        def __repr__(self) -> str:
            return (f"Sequence(id={self.id}, status={self.status.value}, "
                    f"tokens={self.num_tokens}, blocks={len(self.block_table)})")
        


        
