import math
import time

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from uer.initialize import init_env
from uer.model_builder import build_model
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.utils import str2dataloader, str2optimizer, str2scheduler, str2tokenizer
from uer.utils.logging import init_logger
from uer.utils.seed import set_seed


def init_model(args):
    model = build_model(args)

    if args.pretrained_model_path is not None:
        model = load_model(model, args.pretrained_model_path)
    else:
        if args.deep_init:
            scaled_factor = 1 / math.sqrt(2.0 * args.layers_num)
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    if "linear_2.weight" in n or "final_linear.weight" in n:
                        p.data.normal_(0, 0.02 * scaled_factor)
                    elif "linear_2.bias" in n or "final_linear.bias" in n:
                        p.data.zero_()
                    else:
                        p.data.normal_(0, 0.02)
        else:
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)
    return model


def init_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    if args.optimizer == "adamw":
        custom_optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            correct_bias=False
        )
    else:
        custom_optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False
        )

    if args.scheduler == "constant":
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler == "constant_with_warmup":
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps * args.warmup)
    elif args.scheduler == "tri_stage":
        custom_scheduler = str2scheduler[args.scheduler](
            custom_optimizer,
            args.total_steps * args.warmup,
            args.total_steps * args.lr_decay,
            args.total_steps
        )
    else:
        custom_scheduler = str2scheduler[args.scheduler](
            custom_optimizer,
            args.total_steps * args.warmup,
            args.total_steps
        )

    return custom_optimizer, custom_scheduler


def train_and_validate(args):
    set_seed(args.seed)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    if args.dist_train:
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args), daemon=False)
    elif args.single_gpu:
        worker(args.local_rank, None, args)
    else:
        worker(None, None, args)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.output_model_path = args.output_model_path
        self.start_time = time.time()
        self.total_loss = 0.0
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.logger = args.logger

    def forward_propagation(self, batch, model):
        raise NotImplementedError

    def report_and_reset_stats(self):
        raise NotImplementedError

    def train(self, args, local_rank, global_rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)

        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if local_rank is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(local_rank)

            loss = self.forward_propagation(batch, model)
            loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and global_rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and global_rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss, correct, denominator = model(src, tgt, seg)
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        return loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
                         "| {:8.2f} tokens/s"
                         "| loss {:7.2f}"
                         "| acc: {:3.3f}".format(
                             self.current_step,
                             self.total_steps,
                             done_tokens / (time.time() - self.start_time),
                             self.total_loss / self.report_steps,
                             self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


str2trainer = {"mlm": MlmTrainer}


def worker(local_rank, gpu_ranks, args):
    set_seed(args.seed)

    args.logger = init_logger(args)
    args.local_rank = local_rank
    init_env(args)
    global_rank = args.global_rank

    model = init_model(args)
    custom_optimizer, custom_scheduler = init_optimizer(args, model)

    if local_rank is not None:
        model.cuda(local_rank)
    optimizer = custom_optimizer
    scheduler = custom_scheduler

    if args.dist_train:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        args.logger.info("Worker %d is training ... " % global_rank)
    else:
        args.logger.info("Worker is training ...")

    if args.dist_train:
        train_loader = str2dataloader[args.data_processor](
            args, args.dataset_path, args.batch_size, global_rank, args.world_size, local_rank, True
        )
    else:
        train_loader = str2dataloader[args.data_processor](
            args, args.dataset_path, args.batch_size, 0, 1, local_rank, True
        )

    trainer = str2trainer[args.data_processor](args)
    trainer.train(args, local_rank, global_rank, train_loader, model, optimizer, scheduler)
