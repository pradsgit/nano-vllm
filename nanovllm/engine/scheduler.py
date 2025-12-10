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
        self.max_scheduled_tokens = config.max_num_batched_tokens
        self.long_prefill_token_threshold = 512 # maybe move this to config?
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
        token_budget = self.max_scheduled_tokens

        # Step 1: Schedule RUNNING seqs
        running_seqs = []
        while self.running and token_budget > 0:
            seq = self.running.popleft()
            if seq.is_finished:
                continue

            # check if still prefilling or decode
            if seq.num_cached_tokens < len(seq.prompt_tokens):
                # partial prefill
                num_remaining = len(seq.prompt_tokens) - seq.num_cached_tokens
                num_new_tokens = min(num_remaining, self.long_prefill_token_threshold, token_budget)
            else:
                # decode
                num_new_tokens = 1

            try:
                # allocate blocks for decode or continuation of partial prefill
                self.block_manager.append_slots(seq, num_new_tokens)
            except ValueError as e:
                print(f'Cannot allocate block for seq {seq.seq_id}: {e}')
                continue

            seq.num_scheduled_tokens = num_new_tokens
            token_budget -= num_new_tokens
            running_seqs.append(seq)
            num_seqs += 1

        self.running.extendleft(reversed(running_seqs))
        scheduled_seqs.extend(running_seqs)

        # Step 2: Schedule WAITING seqs
        while self.waiting and token_budget > 0:
            seq = self.waiting[0]
            print(f'waiting seqs found.. processing seq: {seq.id}')

            # Check prefix cache
            cached_blocks, num_cached_tokens = (
                self.block_manager.get_computed_blocks(seq.prompt_tokens)
            )
            
            num_new_tokens = len(seq.prompt_tokens) - num_cached_tokens
            threshold = self.long_prefill_token_threshold

            # Apply chunking
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            
            print(f'num new tokens: {num_new_tokens}')
            blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size
            print(f'blocks needed: {blocks_needed}')

            if not self.block_manager.can_allocate(blocks_needed):
                print('cannot allocated blocks to this sequence')
                break
            
            # Allocate slots
            allocated_blocks = self.block_manager.allocate_slots(
                seq=seq,
                cached_blocks=cached_blocks,
                num_new_blocks=blocks_needed
            )

            print(f'seq block_table: {allocated_blocks}')

            # Update sequence
            seq.block_table = allocated_blocks
            seq.num_cached_tokens = num_cached_tokens  # Set, not add!
            seq.num_scheduled_tokens = num_new_tokens
            seq.status = SequenceStatus.RUNNING
            token_budget -= num_new_tokens
            
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_prefill += 1
            num_seqs += 1

        print(f'remaining token budget: {token_budget}')
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
            seq.num_cached_tokens += seq.num_scheduled_tokens

            # Only add token if prefill complete
            if token_id is not None:
                seq.add_token(token_id)
                # Update block manager with the new token and cache if full
                self.block_manager.update_block_with_token(seq.id, token_id)
            
            # check if seq is finished
            # 1. max tokens reached or eos token emitted
            if len(seq.output_tokens) >= seq.sampling_params.max_tokens or token_id == self.eos:
                seq.status = SequenceStatus.FINISHED

        # handle finished sequences
        self.free_finished()

                