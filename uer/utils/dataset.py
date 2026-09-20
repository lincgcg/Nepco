import os
import pickle
from multiprocessing import Pool

from uer.utils.constants import CLS_TOKEN, SEP_TOKEN
from uer.utils.mask import mask_seq
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed


def merge_dataset(dataset_path, workers_num):
    dataset_dir = os.path.dirname(dataset_path)
    if dataset_dir:
        os.makedirs(dataset_dir, exist_ok=True)

    with open(dataset_path, "wb") as dataset_writer:
        for i in range(workers_num):
            tmp_path = "dataset-tmp-" + str(i) + ".pt"
            with open(tmp_path, "rb") as tmp_dataset_reader:
                while True:
                    tmp_data = tmp_dataset_reader.read(2 ** 20)
                    if not tmp_data:
                        break
                    dataset_writer.write(tmp_data)
            os.remove(tmp_path)


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num):
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert workers_num >= 1
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + \
                        self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) + \
                        [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for document in docs_buffer:
            all_documents += document
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking,
                                    self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = ((src, 0), tgt, seg_pos)
            else:
                instance = ((src, 0), seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]
        if len(src) == 0:
            return instances

        seg_pos = [len(src)]
        pad_num = self.seq_length - len(src)

        if not self.dynamic_masking:
            src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking,
                                self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = ((src, pad_num), tgt, seg_pos)
        else:
            instance = ((src, pad_num), seg_pos)

        instances.append(instance)
        return instances
