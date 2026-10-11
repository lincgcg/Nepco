"""Only utilities required by TraGe's frozen Transformer implementation."""
from uer.utils.tokenizers import BertTokenizer
from uer.utils.act_fun import gelu, gelu_fast, relu, silu, linear

str2tokenizer = {"bert": BertTokenizer}
str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}
