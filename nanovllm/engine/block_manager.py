import torch
from collections import deque
from transformers import Qwen3Config

class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        config: Qwen3Config
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # initialize block pool where each block is an actual tensor living in GPU memory!
        # self.gpu_blocks = []

        # for i in range(num_blocks):
        #     block = torch.empty(
        #         2, # for k and v
        #         config.num_hidden_layers, 
        #         block_size, 
        #         config.num_key_value_heads, 
        #         config.head_dim, 
        #         dtype=torch.float16, 
        #         device='cuda'
        #     ) 
        #     self.gpu_blocks.append(block)

        # track free blocks
        self.free_block_ids = deque(range(num_blocks))
        # track which blocks belong to which sequence (seq -> block_ids mapping) 
        self.seq_to_blocks: dict[str, list[int]] = {}

    def allocate_sequence(self, seq_id: str, num_blocks: int) -> list[int]:
        """allocates blocks for a sequence and tracks them"""
        if not self.can_allocate(num_blocks):
            raise ValueError(f"Cannot allocate {num_blocks}")
        
        allocated = []

        for i in range(num_blocks):
            block = self.free_block_ids.popleft()
            allocated.append(block)

        # update the seq -> block_ids mapping
        if seq_id not in self.seq_to_blocks:
            self.seq_to_blocks[seq_id] = []

        self.seq_to_blocks[seq_id].extend(allocated)

        return allocated

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def allocate(self, num_blocks: int) -> list[int]:
        """allocates blocks and updates free_block_ids?"""
        # check if num_blocks are available?
        if not self.can_allocate(num_blocks):
            raise ValueError(f"Cannot allocate {num_blocks}, only {len(self.free_block_ids)} free")
        
        allocated = []

        for i in range(num_blocks):
            block_id = self.free_block_ids.pop()
            allocated.append(block_id)

        return allocated

    def free(self, block_ids: list[int]) -> None:
        """frees blocks with block_ids"""
        for block_id in block_ids:
            self.free_block_ids.append(block_id)

    def get_blocks_for_sequence(self, seq_id: str) -> list[torch.Tensor]:
        """Get actual block tensors for a sequence."""
        block_ids = self.seq_to_blocks.get(seq_id, [])
        return [self.gpu_blocks[bid] for bid in block_ids]

    def free_sequence(self, seq_id: str) -> None:
        """Free all blocks for a sequence."""
        if seq_id in self.seq_to_blocks:
            block_ids = self.seq_to_blocks.pop(seq_id)
            for block_id in block_ids:
                self.free_block_ids.append(block_id)
