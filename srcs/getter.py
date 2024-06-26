import yaml

from lightning import Trainer
from transformers import AutoConfig, AutoModelForSeq2SeqLM


from srcs.tokenizer import (
    srcTokenizer,
    tgtTokenizer,
    morphTokenizer,
    tagTokenizer,
    lengthTokenizer,
)
from srcs.data.metric import METRIC
from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_args(args):
    with open(args.trainer_args) as f:
        args.trainer_args = yaml.load(f, Loader=yaml.FullLoader)
    return args


def get_datamodule(args, tokenizer):
    dm = DataModule(args, tokenizer)
    dm.setup(args.mode)

    return dm


def get_model(args):
    # tokenizer
    src_tokenizer = srcTokenizer(args)
    tgt_tokenizer = tgtTokenizer(args)
    morph_tokenizer = morphTokenizer(args)
    tag_tokenizer = tagTokenizer(args)
    length_tokenzier = lengthTokenizer(args)

    tokenizers = {
        "src": src_tokenizer,
        "tgt": tgt_tokenizer,
        "morph": morph_tokenizer,
        "tag": tag_tokenizer,
        "length": length_tokenzier,
    }

    # # 모델 및 구성 불러오기
    # config = AutoConfig.from_pretrained(args.model)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    metric = METRIC[args.data]()

    if args.mode == "fit":
        model = LightningWrapper(args, tokenizers, metric)

    if args.mode == "test":
        """
        model = LightningWrapper.load_from_ckpt()
        load from ckpt
        """

    return model, tokenizers


def get_trainer(args):
    trainer = Trainer(**args.trainer_args)

    return trainer
