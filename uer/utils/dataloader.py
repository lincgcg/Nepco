import pickle
import random

import torch

from uer.utils.constants import PAD_TOKEN
from uer.utils.mask import mask_seq


class Dataloader(object):
    def __init__(self, args, dataset_path, batch_size, global_rank, world_size, local_rank, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.world_size == self.global_rank:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


class MlmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            masked_words_num = 0

            for ins in instances:
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))

                if len(ins) == 3:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [0] * pad_num)
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking,
                                                      self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(src_single))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [0] * pad_num)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(seg)
