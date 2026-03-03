from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class GroupBatchSampler:
    """
    Batch sampler for episode-grouped memory training.

    Packs `group_size` consecutive-timestep samples from the same episode into
    contiguous batch positions. PerMemBank with dataloader_type="group" requires
    this ordering: samples [0..G-1] must be episode A in temporal order, samples
    [G..2G-1] must be episode B in temporal order, etc.

    Groups are shuffled across batches. Within each group, samples are in
    ascending timestep order.
    """

    def __init__(
        self,
        sampler: "SequenceSampler",
        batch_size: int,
        group_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        assert batch_size % group_size == 0, (
            f"batch_size ({batch_size}) must be divisible by group_size ({group_size})"
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed=seed)

        # Build per-episode sorted index lists
        episode_to_items = {}
        for sample_idx in range(len(sampler)):
            episode_id, timestep = sampler.get_episode_id_and_timestep(sample_idx)
            if episode_id not in episode_to_items:
                episode_to_items[episode_id] = []
            episode_to_items[episode_id].append((timestep, sample_idx))

        # Sort by timestep within each episode, then chop into non-overlapping groups
        self.groups = []
        for items in episode_to_items.values():
            items.sort(key=lambda x: x[0])
            indices = [idx for _, idx in items]
            for start in range(0, len(indices) - group_size + 1, group_size):
                self.groups.append(indices[start:start + group_size])

    def __iter__(self):
        groups = list(self.groups)
        self._rng.shuffle(groups)
        n_per_batch = self.batch_size // self.group_size
        for i in range(0, len(groups) - n_per_batch + 1, n_per_batch):
            batch = []
            for group in groups[i:i + n_per_batch]:
                batch.extend(group)
            yield batch
        if not self.drop_last:
            remainder_start = (len(groups) // n_per_batch) * n_per_batch
            if remainder_start < len(groups):
                batch = []
                for group in groups[remainder_start:]:
                    batch.extend(group)
                yield batch

    def __len__(self):
        n_per_batch = self.batch_size // self.group_size
        if self.drop_last:
            return len(self.groups) // n_per_batch
        return (len(self.groups) + n_per_batch - 1) // n_per_batch


class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.episode_ends = episode_ends
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def get_episode_id_and_timestep(self, idx: int) -> tuple:
        """Return (episode_id, timestep) for the given sample index."""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        episode_id = int(np.searchsorted(self.episode_ends, buffer_start_idx, side="left"))
        episode_start = 0 if episode_id == 0 else self.episode_ends[episode_id - 1]
        timestep = float(buffer_start_idx - episode_start)
        return episode_id, timestep

    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0] # np.zeros_like(sample[0])
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
