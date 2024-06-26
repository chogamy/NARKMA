import argparse


from lightning import seed_everything

from srcs.getter import get_args, get_model, get_datamodule, get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, default=None, type=str, choices=["fit", "test"]
    )
    parser.add_argument(
        "--architecture",
        required=True,
        default=None,
        type=str,
        choices=[
            "Enc1NARDec2_norm_dict",
            "Enc1NARDec2_norm",
            "Enc1NARDec2_dict",
            "Enc1NARDec2",
            "Enc1ARDec1",
        ],
    )
    parser.add_argument("--trainer_args", default="args/trainer/basic.yaml", type=str)
    parser.add_argument(
        "--data", required=True, default=None, type=str, choices=["sejong"]
    )
    args = parser.parse_args()

    seed_everything(42, workers=True)

    args = get_args(args)
    model, tokenizer = get_model(args)
    dm = get_datamodule(args, tokenizer)
    trainer = get_trainer(args)

    if args.mode == "fit":
        trainer.fit(model=model, datamodule=dm)

    if args.mode == "test":
        trainer.test(model=model, datamodule=dm)
