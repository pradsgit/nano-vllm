import torch
from collections import deque
from transformers import Qwen3Config
from typing import Optional
from nanovllm.engine.sequence import Sequence

"""
 - we do not want to compute and store k and v of sequence of tokens that are duplicate  
 - so blocks might be shared by multiple sequences
 - multiple sequences can point to a single block
 - how do we know two seqs have same prefix
"""


class KVCacheBlock:
    def __init__(
        self,
        id
    ):  
        # unique id
        self.id = id
        # will be assigned when the block is full, and reset it when evicted
        self.hash = -1 
        self.ref_count = 0 # number of sequences currently using this block
        self.token_ids = [] # tokens this block contains
        self.prev_free_block: Optional["KVCacheBlock"] = None
        self.next_free_block: Optional["KVCacheBlock"] = None
        
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # a list of KVCacheBlock
        self.block_pool: list[KVCacheBlock] = [KVCacheBlock(id=i) for i in range(num_blocks)]
        self.free_block_pool_head: KVCacheBlock | None # head pointer of block pool
        self.free_block_pool_tail: KVCacheBlock | None # tail pointer of block pool

        self._init_block_pool()
        
        self.cache_blocks: dict[int, int] = dict() # mapping from hash key to block id
        
        # track which blocks belong to which sequence (seq -> block_ids mapping)
        self.seq_to_blocks: dict[str, list[int]] = {}
        self.request_blocks: dict[str, list[int]] = {} # mapping from request to block ids

    def _init_block_pool(self):
        """
        update each KVCacheBlock's previous and next pointers
        """
        N = len(self.block_pool)
        for i in range(N):
            if i > 0:
                self.block_pool[i].prev_free_block = self.block_pool[i-1]
            if i < N - 1:
                self.block_pool[i].next_free_block = self.block_pool[i+1]

        self.free_block_pool_head = self.block_pool[0]
        self.free_block_pool_tail = self.block_pool[N - 1]

    @classmethod
    def compute_hash(cls, token_ids: list[int], prev_hash=-1):
        """
        Compute incremental hash for a block.
        
        Encodes: all prefix tokens (via prev_hash) + current block tokens
        """

        import xxhash
        import numpy as np
        h = xxhash.xxh64()
        if prev_hash != -1:
            h.update(prev_hash.to_bytes(8, "little"))

        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def get_computed_blocks(self, token_ids: list[int]) -> tuple[list[int], int]:
        """
        get the computed(cached) blocks for a sequence's token ids, if cache hits
        note: computed blocks must be full
        finds the longest prefix of blocks that are cached

        returns:
            tuple -> (list of cached block ids, num_cached_tokens)
        """

        cached_blocks = []
        num_cached_tokens = 0
        hash = -1

        blocks_needed = (len(token_ids) + self.block_size - 1) // self.block_size # ceil div

        for block_idx in range(blocks_needed):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(token_ids))
            block_tokens = token_ids[start:end]

            # only hash full blocks
            if len(block_tokens) < self.block_size:
                break

            hash = self.compute_hash(block_tokens, hash)
            # check if cached
            block_id = self.cache_blocks.get(hash, None)
            if block_id is None:
                # cache miss, stop here
                break
            print(f'prefix cache found, reusing this block: {self.block_pool[block_id]}')
            cached_blocks.append(block_id)
            num_cached_tokens += len(block_tokens)

        return cached_blocks, num_cached_tokens
    
    def _remove_from_free_queue(self, block: KVCacheBlock):
        if block.prev_free_block is not None:
            # this is not a head pointer
            block.prev_free_block.next_free_block = block.next_free_block
        
        if block.next_free_block is not None:
            block.next_free_block.prev_free_block = block.prev_free_block
        
        if block.id == self.free_block_pool_head.id:
            self.free_block_pool_head = block.next_free_block
        
        if block.id == self.free_block_pool_tail.id:
            self.free_block_pool_tail = block.prev_free_block

        block.prev_free_block = None
        block.next_free_block = None

    def _add_to_free_queue(self, block: KVCacheBlock):
        """add block to the tail of free queue"""
        if self.free_block_pool_tail is None:
            # Empty queue
            self.free_block_pool_head = block
            self.free_block_pool_tail = block
            block.prev_free_block = None
            block.next_free_block = None
        else:
            # Add to tail
            self.free_block_pool_tail.next_free_block = block
            block.prev_free_block = self.free_block_pool_tail
            block.next_free_block = None
            self.free_block_pool_tail = block

    def _pop_from_free_queue(self) -> KVCacheBlock:
        """
        pop the head block from the queue and return the block
        """
        if self.free_block_pool_head is None:
            raise ValueError("No free blocks available")

        block = self.free_block_pool_head
        self._remove_from_free_queue(block)
        return block
    
    def allocate_slots(
        self, 
        seq: Sequence, 
        cached_blocks: list[int], 
        num_new_blocks: int
    ):
        """
        Allocate blocks with prefix caching support

        returns:
            combined list of all blocks [cached + newly allocated]
        """ 
        # touch cached blocks, update ref_count for this seq
        for block_id in cached_blocks:
            block = self.block_pool[block_id]

            # no one is currently using this cached data
            if block.ref_count == 0:
                # block is in free queue - remove it
                self._remove_from_free_queue(block)

            block.ref_count += 1

        # allocate new blocks
        new_blocks = []
        for _ in range(num_new_blocks):
            new_block = self._pop_from_free_queue()
            # evict if block is cached
            if new_block.hash != -1:
                del self.cache_blocks[new_block.hash]
                new_block.reset()
            else:
                # increase ref count of the block
                new_block.ref_count += 1
            new_blocks.append(new_block.id)

        all_blocks = cached_blocks + new_blocks
        self.request_blocks[seq.id] = all_blocks

        # assign token ids to new blocks and cache them if they get full
        num_cached = len(cached_blocks) * self.block_size
        start_token_idx = num_cached
        # Get prev_hash from last cached block
        if cached_blocks:
            last_block_id = cached_blocks[-1]
            last_block = self.block_pool[last_block_id]
            prev_hash = last_block.hash
        else:
            prev_hash = -1  # No cached blocks

        for block_id in new_blocks:
            block: KVCacheBlock = self.block_pool[block_id]
            # assign token_ids and hash to this block and then cache it
            end_token_idx = min(start_token_idx + self.block_size, len(seq.prompt_tokens))

            block_tokens = seq.prompt_tokens[start_token_idx:end_token_idx]
            block.token_ids = block_tokens
            # if block can be full, update block and cache it
            if len(block_tokens) == self.block_size:
                hash = self.compute_hash(block_tokens, prev_hash)
                block.update(hash, block_tokens)
                self.cache_blocks[hash] = block.id
                prev_hash = hash

            start_token_idx = end_token_idx

        return all_blocks
    
    def append_slots(self, seq: Sequence):
        """
        ensure the seq has sufficient capacity to process the current token
        """
        blocks = self.request_blocks[seq.id]
        last_block_id = blocks[-1]
        last_block = self.block_pool[last_block_id]

        if len(last_block.token_ids) >= self.block_size:
            # block is full, allocate a new block
            if not self.can_allocate(1):
                raise ValueError(f"Cannot allocate block for seq {seq.seq_id}")
            
            new_block = self._pop_from_free_queue()
            # evict if block is cached
            if new_block.hash != -1:
                del self.cache_blocks[new_block.hash]
                new_block.reset()
            else:
                new_block.ref_count += 1

            blocks.append(new_block.id)
            seq.block_table.append(new_block.id)

    def update_block_with_token(self, seq_id: str, token_id: int):
        """
        Add token_id to the last block.
        Cache block if it becomes full.
        """
        blocks = self.request_blocks[seq_id]
        last_block_id = blocks[-1]
        last_block = self.block_pool[last_block_id]
        
        # append token
        last_block.token_ids.append(token_id)
        
        # if block just became full, cache it
        if len(last_block.token_ids) == self.block_size:
            # Get prev_hash from previous block
            if len(blocks) > 1:
                prev_block_id = blocks[-2]
                prev_block = self.block_pool[prev_block_id]
                prev_hash = prev_block.hash
            else:
                prev_hash = -1
            
            # Compute hash and cache
            hash_value = self.compute_hash(last_block.token_ids, prev_hash)
            last_block.hash = hash_value
            self.cache_blocks[hash_value] = last_block_id

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def free_sequence(self, seq_id: str) -> None:
        """Free all blocks for a sequence."""
        
        if seq_id in self.request_blocks:
            block_ids = self.request_blocks.pop(seq_id) # removes the entry in dict
            for block_id in block_ids:
                block = self.block_pool[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    # Return to free queue
                    self._add_to_free_queue(block)