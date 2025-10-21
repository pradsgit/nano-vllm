from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from collections import deque

class Scheduler:
    """A scheduler handles sequnece lifecycles"""
    def __init__(
        self, 
        config,
        block_manager: BlockManager
    ):
        self.block_size = config.block_size
        # TODO: change to deque
        self.waiting: list[Sequence] = []
        # TODO: change to deque
        self.running: list[Sequence] = []
        self.block_manager = block_manager

    def add_request(self, seq: Sequence) -> None:
        """add a new seq to waiting queue"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self):
        """
        decide which sequences to run this step (step is a model forward pass).

        the V1 scheduler of vllm can mix both types of requests(prefill and decode) in the same step unlike V0(prev version)

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




                    
            





        
        