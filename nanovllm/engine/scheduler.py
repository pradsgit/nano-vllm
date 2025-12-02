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
        self.max_num_seqs = config.max_num_seqs
        self.block_size = config.kvcache_block_size
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """add a new seq to waiting queue"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self):
        scheduled_seqs = []
        num_prefill = 0
        num_seqs = 0

        # first schedule RUNNING seqs
        running_seqs = []
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.is_finished:
                continue  # Don't add finished seqs back

            try:
                self.block_manager.append_slots(seq)
            except ValueError as e:
                print(f'Cannot allocate block for seq {seq.seq_id}: {e}')
                #TODO: handle preemption
                continue

            running_seqs.append(seq)
            num_seqs += 1

        # Add running seqs back to self.running (maintain order)
        self.running.extendleft(reversed(running_seqs))
        scheduled_seqs.extend(running_seqs)

        # step2: try to schedule waiting sequences, prefill phase
        while self.waiting and num_seqs < self.max_num_seqs:
            print('waiting seqs found.. adding')
            seq = self.waiting[0]

            # check if blocks are already computed for prompt tokens
            cached_blocks, num_cached_tokens = self.block_manager.get_computed_blocks(seq.prompt_tokens)

            num_new_tokens = len(seq.prompt_tokens) - num_cached_tokens
            # calculate blocks needed for tokens that are not cached
            blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size

            if not self.block_manager.can_allocate(blocks_needed):
                print('cannot allocated blocks to this sequence')
                break  # can't schedule this seq yet
            
            # call allocate_slots of block_manager to get the complete block_table
            allocated_blocks = self.block_manager.allocate_slots(
                seq=seq,
                cached_blocks=cached_blocks,
                num_new_blocks=blocks_needed
            )

            # update sequence
            seq.block_table = allocated_blocks
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

            scheduled_seqs.append(seq)
            num_prefill += 1
            num_seqs += 1

        return scheduled_seqs, num_prefill > 0

    def free_finished(self):
        """Free blocks from finished sequences and remove from running queue."""
        finished_seqs = []
        still_running = []

        if not any(seq.is_finished for seq in self.running):
            return 
        
        for seq in self.running:
            if seq.is_finished:
                self.block_manager.free_sequence(seq.id)
                seq.block_table = []
                finished_seqs.append(seq)
            else:
                still_running.append(seq)

        # Replace deque contents
        self.running.clear()
        self.running.extend(still_running)
        
        return finished_seqs
    
    def update_from_output(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
    ):
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_prefill:
                # we saved prompt_tokens number of tokens in cache
                seq.num_cached_tokens = len(seq.prompt_tokens)
            else:
                seq.num_cached_tokens += 1 # we only cached one token
            seq.add_token(token_id)

            # Update block manager with the new token and cache if full
            self.block_manager.update_block_with_token(seq.id, token_id)
            
            # check if seq is finished
            # 1. max tokens reached or eos token emitted
            if len(seq.output_tokens) >= seq.sampling_params.max_tokens or token_id == self.eos:
                seq.status = SequenceStatus.FINISHED

        # handle finished sequences
        self.free_finished()

                