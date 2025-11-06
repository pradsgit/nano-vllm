from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from collections import deque
from nanovllm.config import Config

class Scheduler:
    """A scheduler handles sequnece lifecycles"""
    def __init__(
        self,
        config: Config
        # block_size: int,
        # block_manager: BlockManager
    ):
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # TODO: change to deque
        self.waiting: list[Sequence] = []
        # TODO: change to deque
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        """add a new seq to waiting queue"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting or self.running

    def schedule(self):
        """
        Phase 1: single sequence at a time
        
        Returns:
            (scheduled_seqs, num_prefill)
        """
        # Step 1: Handle running sequence (decode)
        if self.running:
            seq = self.running[0]
            
            if seq.is_finished:
                return [], 0
            
            # Check if we need more blocks for next decode step
            # tokens_after_decode = seq.num_tokens + 1
            # blocks_needed = seq.get_num_logical_blocks(tokens_after_decode)
            # blocks_has = len(seq.block_table)

            # TODO: introduce "token budget" and remove the below look ahead tokens calculation
            tokens_after_decode = seq.num_tokens + 1
            blocks_needed = (tokens_after_decode + self.block_size - 1) // self.block_size
            blocks_has = len(seq.block_table)

            # print(f'{blocks_needed} blocks needed for sequence with total tokens {seq.num_tokens}')
            # print(f'currently {blocks_has} blocks are allocated for sequence with total tokens {seq.num_tokens}')
            if blocks_needed > blocks_has:
                # Need to allocate more blocks
                additional_blocks = blocks_needed - blocks_has
                
                if self.block_manager.can_allocate(additional_blocks):
                    # Allocate and extend block table
                    allocated = self.block_manager.allocate_sequence(
                        seq.id, 
                        additional_blocks
                    )
                    seq.block_table.extend(allocated)  # ← Extend, not replace
                else:
                    # Can't allocate, can't run
                    return [], 0
            
            return [seq], 0  # num_prefill = 0 (decode mode)
        
        # Step 2: No running sequence, try to start one from waiting
        if self.waiting:
            seq = self.waiting.pop(0)
            blocks_needed = seq.get_num_logical_blocks(self.block_size)
            
            if self.block_manager.can_allocate(blocks_needed):
                # Allocate blocks and store in sequence
                allocated = self.block_manager.allocate_sequence(
                    seq.id, 
                    blocks_needed
                )
                seq.block_table = allocated  # ← Store block IDs
                
                # Move to running
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                
                return [seq], 1  # num_prefill = 1
            else:
                # Can't allocate, put back in waiting
                self.waiting.insert(0, seq)
                return [], 0
        
        return [], 0

    def schedule_phase2(self):
        """

        decide which sequences to run this step (a 'step' is a model forward pass).
        this method gets called once every step.

        the V1 scheduler of vllm can handle mix of request types(prefill and decode) in the same step unlike V0(prev version)

        priority:
        1. running sequences (seqs already in RUNNING state)    
        2. waiting sequences (if memory available)
        """

        scheduled_seqs = []
        num_prefill = 0

        # step 1: the scheduler prioritizes decode requests that is handle RUNNING seqs (higher priority)
        for seq in self.running:
            # skip finished seqs
            if seq.is_finished:
                continue

            # check if the sequence has enough blocks allocated for the current size
            blocks_needed = seq.get_num_logical_blocks(self.block_size)
            blocks_has = len(seq.block_table)

            if blocks_needed > blocks_has:
                # need more blocks, allocate
                additional_blocks = blocks_needed - blocks_has
                if self.block_manager.can_allocate(additional_blocks):
                    allocated_blocks = self.block_manager.allocate(additional_blocks)
                    seq.block_table.extend(allocated_blocks)
                else:
                    # TODO: phase2
                    continue
            
            scheduled_seqs.append(seq)

        
        # step2: try to schedule waiting sequences, prefill phase
        for seq in self.waiting[:]:
            # check how many blocks are needed for this seq's prefill phase
            blocks_needed = seq.get_num_logical_blocks(self.block_size)

            if self.block_manager.can_allocate(blocks_needed):
                allocated_blocks = self.block_manager.allocate(blocks_needed)
                seq.block_table = allocated_blocks

                # move WAITING -> RUNNING
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)

                scheduled_seqs.append(seq)
                num_prefill += 1

            else:
                # no blocks are available
                break

        return scheduled_seqs, num_prefill
    
    def free_finished(self):
        """Free blocks from finished sequences and remove from running queue."""
        finished_seqs = []
        
        for seq in self.running[:]:
            if seq.is_finished:
                # Free blocks using BlockManager
                self.block_manager.free_sequence(seq.id)
                seq.block_table = []  # Clear the block table
                
                self.running.remove(seq)
                finished_seqs.append(seq)
        
        return finished_seqs