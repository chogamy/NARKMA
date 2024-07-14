import os
import yaml

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy


from srcs.tokenizer import TOKENIZERS
from srcs.data.metric import METRIC
from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_args(args):
    with open(args.trainer_args) as f:
        args.trainer_args = yaml.load(f, Loader=yaml.FullLoader)
    args.trainer_args["default_root_dir"] = os.path.join(
        "params", args.architecture, args.data
    )
    return args


def get_datamodule(args, tokenizer):
    dm = DataModule(args, tokenizer)
    dm.setup(args.mode)

    return dm


def get_model(args):
    tokenizers = TOKENIZERS[args.architecture](args)

    metric = METRIC[args.data]()

    if args.mode == "fit":
        model = LightningWrapper(args, tokenizers, metric)

    if args.mode == "test":
        path = os.path.join(args.trainer_args["default_root_dir"], "last.ckpt")
        model = LightningWrapper.load_from_checkpoint(
            path, args=args, tokenizers=tokenizers, metric=metric
        )

    return model, tokenizers


def get_callbacks(args):
    ckpt_callback = ModelCheckpoint(
        dirpath=args.trainer_args["default_root_dir"], filename="last"
    )

    return [ckpt_callback]


def get_trainer(args):
    args.trainer_args["callbacks"] = get_callbacks(args)
    args.trainer_args["strategy"] = DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(**args.trainer_args)

    return trainer
