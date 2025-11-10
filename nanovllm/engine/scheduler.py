from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from collections import deque
from nanovllm.config import Config

class Scheduler:
    """A scheduler handles sequence lifecycle"""
    def __init__(
        self,
        config: Config
    ):
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """add a new seq to waiting queue"""
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

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

            # check if sufficient blocks are allocated for this decode step
            future_cache_size = seq.num_cached_tokens + 1
            blocks_needed = (future_cache_size + self.block_size - 1) // self.block_size
            blocks_has = len(seq.block_table)

            if blocks_needed > blocks_has:
                # Need to allocate more blocks
                additional_blocks = blocks_needed - blocks_has
                
                if self.block_manager.can_allocate(additional_blocks):
                    # Allocate and extend block table
                    allocated = self.block_manager.allocate_sequence(
                        seq.id, 
                        additional_blocks
                    )
                    seq.block_table.extend(allocated)  # extend, not replace
                else:
                    # can't allocate, can't run
                    return [], 0
            
            return [seq], 0  # num_prefill = 0 (decode mode)
        
        # Step 2: No running sequence, try to start one from waiting
        if self.waiting:
            seq = self.waiting.popleft() # gets first item in deque
            blocks_needed = seq.get_num_logical_blocks(self.block_size)
            
            if self.block_manager.can_allocate(blocks_needed):
                # Allocate blocks and store in sequence
                allocated = self.block_manager.allocate_sequence(
                    seq.id, 
                    blocks_needed
                )
                seq.block_table = allocated  # store allocated block IDs
                # Move to running
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                
                return [seq], 1  # num_prefill = 1
            else:
                # Can't allocate, put back in waiting
                self.waiting.appendleft(seq)
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

        while self.waiting:
            seq = self.waiting[0]
            blocks_needed = seq.get_num_logical_blocks(self.block_size)

            if self.block_manager.can_allocate(blocks_needed):
                allocated_blocks = self.block_manager.allocate(blocks_needed)
                seq.block_table = allocated_blocks
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

                scheduled_seqs.append(seq)
                num_prefill += 1

            else:
                break

        # for seq in self.waiting[:]:
        #     # check how many blocks are needed for this seq's prefill phase
        #     blocks_needed = seq.get_num_logical_blocks(self.block_size)

        #     if self.block_manager.can_allocate(blocks_needed):
        #         allocated_blocks = self.block_manager.allocate(blocks_needed)
        #         seq.block_table = allocated_blocks

        #         # move WAITING -> RUNNING
        #         seq.status = SequenceStatus.RUNNING
        #         self.waiting.remove(seq) # this is slow
        #         self.running.append(seq)

        #         scheduled_seqs.append(seq)
        #         num_prefill += 1

        #     else:
        #         # no blocks are available
        #         break

        return scheduled_seqs, num_prefill
    
    def free_finished(self):
        """Free blocks from finished sequences and remove from running queue."""
        finished_seqs = []
        still_running = []
        
        for seq in self.running:
            if seq.is_finished:
                # Free blocks using BlockManager
                self.block_manager.free_sequence(seq.id)
                seq.block_table = []  # Clear the block table
                finished_seqs.append(seq)
            else:
                still_running.append(seq)
        
        self.running = still_running
        return finished_seqs