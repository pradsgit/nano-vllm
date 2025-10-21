import torch
from collections import deque

class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_blocks: int,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # initialize block pool where each block is an actual tensor living in GPU memory!
        self.gpu_blocks = []

        for i in range(num_blocks):
            # Qwen3-0.6B values
            block = torch.zeros(
                2,
                24, 
                block_size, 
                2, 
                64, 
                dtype=torch.float16, 
                device='cuda'
            ) # each block takes 224 KB of HBM
            self.gpu_blocks.append(block)

        # track free blocks
        self.free_block_ids = deque(range(num_blocks))

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def allocate(self, num_blocks: int) -> list[int]:
        """allocates blocks and updates free_block_ids?"""
        # check if num_blocks are available?
        if not self.can_allocate():
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
