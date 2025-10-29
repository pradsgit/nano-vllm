from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from collections import deque

class Scheduler:
    """A scheduler handles sequnece lifecycles"""
    def __init__(
        self, 
        block_size: int,
        block_manager: BlockManager
    ):
        self.block_size = block_size
        # TODO: change to deque
        self.waiting: list[Sequence] = []
        # TODO: change to deque
        self.running: list[Sequence] = []
        self.block_manager = block_manager

    def add(self, seq: Sequence) -> None:
        """add a new seq to waiting queue"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting or self.running

    def schedule(self):
        """
        phase 1: single sequence at a time

        returns:
            (scheduled_seqs, num_prefill)
        """

        # handle running sequence
        if self.running:
            seq = self.running[0] # only one sequence for phase 1

            if seq.is_finished:
                return [], 0
            
            # check if we need more blocks for next decode step
            blocks_needed = seq.get_num_logical_blocks(self.block_size)
            blocks_has = len(seq.block_table)

            if blocks_needed > blocks_has:
                # need to allocate more blocks
                additional_blocks = blocks_needed - blocks_has
                # check if block manager can allocate
                if self.block_manager.can_allocate(additional_blocks):
                    # allocate and extend block table
                    allocated_blocks = self.block_manager.allocate(additional_blocks)
                    seq.block_table.extend(allocated_blocks)
                else:
                    # Can't allocate, can't run (should rarely happen in Phase 1)
                    return [], 0
                
            return [seq], 0  # num_prefill = 0 (decode mode)
        # step 2: No running sequence, try to start one from waiting
        if self.waiting:
            print('waiting seqs present')
            seq = self.waiting.pop(0)
            blocks_needed = seq.get_num_logical_blocks(block_size=self.block_size)

            if self.block_manager.can_allocate(blocks_needed):
                allocated_blocks = self.block_manager.allocate(blocks_needed)
                seq.block_table = allocated_blocks

                # move to running status
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

                return [seq], 1
        else:
            # Can't allocate, put back in waiting queue
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
        """free blocks from the finished sequences and remove seqs from running queue"""
        finished_seqs = []
        for seq in self.running[:]:
            if seq.is_finished:
                # free blocks back to block manager
                if seq.block_table:
                    self.block_manager.free(seq.block_table)
                    seq.block_table = [] # clear the block table

                self.running.remove(seq)
                finished_seqs.append(seq)
        
        return finished_seqs