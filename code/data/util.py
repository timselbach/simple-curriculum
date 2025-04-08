import numpy as np
from torch.utils import data
from code.data.all_datasets import SimpleWikiDataset, SimpleGermanDataset
from torch.utils.data import DataLoader


class myDataLoader(object):
    def __init__(self, data: data.Dataset, batch_size: int, bptt: int):
        sampler = BatchSampler()
        self.indices = sampler.return_indices(data, batch_size, bptt)

        self.batch_size = batch_size
        self.bptt = bptt

        self.loader = DataLoader(data, batch_sampler=self.indices, collate_fn=self.my_collate)

    def my_collate(self, batch):
        seq_len = len(batch[:-self.batch_size]) // self.batch_size
        # TODO: should be a tensor here, not a list
        data = np.asarray(batch[:-self.batch_size]).reshape(seq_len, self.batch_size)
        targets = np.asarray(batch[self.batch_size:])
        return data, targets


class BatchSampler:
    def __init__(self):
        # nothing to do here
        pass

    def return_indices(self, data: data.Dataset, batch_size: int, bptt: int):
        indices = np.asarray(range(0, len(data)))
        batched_indices = self._batchify(indices, batch_size)
        # data_indices, target_indices = self._get_data_target_indices(batched_indices, batch_size, bptt)
        indices = self._get_data_target_indices(batched_indices, batch_size, bptt)
        # reshape indices into data batches and corresponding (flat) targets
        # data has shape [seq_len, batch_size]
        # targets have shape [seq_len * batch_size]

        return indices
        # return data_indices, target_indices

    def _batchify(self, indices, batch_size):
        seq_len = len(indices) // batch_size
        batched = indices[:seq_len * batch_size]
        batched = batched.reshape(batch_size, seq_len).T
        return batched

    def _get_data_target_indices(self, batched_indices, batch_size, bptt):

        indices = []
        for i in range(0, len(batched_indices), bptt):
            seq_len = min(bptt, len(batched_indices) - 1 - i)
            indices.append(batched_indices[i: i + 1 + seq_len].reshape(-1))

        # does not work, because last batch might have different size
        # data_indices = batched_indices[:-1, :].reshape(bptt, batch_size, -1)
        # target_indices = batched_indices[1:, :].reshape(bptt * batch_size, -1)

        return indices


class DataIterator:
    def __init__(self, data: data.Dataset, batch_size: int, bptt: int):
        self.batch_size = batch_size
        self.data = self._batchify(data)
        self.bptt = bptt
        self.num_batches = len(self.data) // self.bptt
        self.i = 0

        # max value of i is num_batches
        self.data_range = list(range(0, len(self.data) - 1, self.bptt))

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > len(self.data):
            self.i = 0
            raise StopIteration

        seq_len = min(self.bptt, len(self.data) - 1 - self.i)
        batch = self.data[self.i: self.i + seq_len]
        target = self.data[self.i + 1: self.i + 1 + seq_len].reshape(-1)
        self.i += self.bptt
        return batch, target

    def _batchify(self, data):
        seq_len = len(data) // self.batch_size
        batched = data[:seq_len * self.batch_size]
        batched = batched.view(self.batch_size, seq_len).t().contiguous()
        # Q: or .to(device) here? or only in the trainer?
        return batched


if __name__ == "__main__":



    data_wiki = SimpleWikiDataset(split="train")
    data_ger = SimpleGermanDataset(split="train")
    batch_size = 3
    bptt = 2
    sampler = BatchSampler()
    indices = sampler.return_indices(data_wiki, batch_size, bptt)

    loader = myDataLoader(SimpleWikiDataset(split="train"), batch_size=batch_size, bptt=bptt)
    for batch in loader.loader:
        data, targets = batch
        print(data)
        print(targets)

