from uer.utils.act_fun import gelu, gelu_fast, linear, relu, silu
from uer.utils.adversarial import FGM, PGD
from uer.utils.dataloader import MlmDataloader
from uer.utils.dataset import MlmDataset
from uer.utils.optimizers import *
from uer.utils.tokenizers import BertTokenizer, SpaceTokenizer


str2tokenizer = {"bert": BertTokenizer, "space": SpaceTokenizer}
str2dataset = {"mlm": MlmDataset}
str2dataloader = {"mlm": MlmDataloader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_with_warmup": get_constant_schedule_with_warmup,
    "inverse_sqrt": get_inverse_square_root_schedule_with_warmup,
    "tri_stage": get_tri_stage_schedule,
}

str2adv = {"fgm": FGM, "pgd": PGD}

__all__ = [
    "BertTokenizer", "SpaceTokenizer", "str2tokenizer",
    "MlmDataset", "str2dataset",
    "MlmDataloader", "str2dataloader",
    "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
    "AdamW", "Adafactor", "str2optimizer",
    "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
    "get_cosine_with_hard_restarts_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_constant_schedule", "get_constant_schedule_with_warmup",
    "get_inverse_square_root_schedule_with_warmup", "get_tri_stage_schedule",
    "str2scheduler", "FGM", "PGD", "str2adv",
]
