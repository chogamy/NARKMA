import os
import argparse

from datasets import DatasetDict

from srcs.data.load import LOAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--data", required=True, default=None, type=str, choices=["sejong"]
    )
    args = parser.parse_args()

    train = LOAD[args.data](args, "train")
    valid = LOAD[args.data](args, "valid")
    test = LOAD[args.data](args, "test")

    dataset = DatasetDict({"train": train, "valid": valid, "test": test})

    print(dataset)

    src = list(
        set(
            "".join(dataset["train"]["src"])
            + "".join(dataset["valid"]["src"])
            + "".join(dataset["test"]["src"])
        )
    )
    src.sort()
    morph = list(
        set(
            "".join([item for sub in dataset["train"]["morph"] for item in sub])
            + "".join([item for sub in dataset["valid"]["morph"] for item in sub])
            + "".join([item for sub in dataset["test"]["morph"] for item in sub])
        )
    )
    morph.sort()
    tag = list(
        set(
            [item for sub in dataset["train"]["tag"] for item in sub]
            + [item for sub in dataset["valid"]["tag"] for item in sub]
            + [item for sub in dataset["test"]["tag"] for item in sub]
        )
    )
    tag.sort()

    special_tokens = ["<bos>", "<eos>", "<mask>", "<unk>", "<pad>"]

    dir = os.path.join(os.getcwd(), "data", args.data, args.architecture, "vocab")
    os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, "src.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + src:
            f.write(f"{token}\n")

    with open(os.path.join(dir, "morph.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + morph:
            f.write(f"{token}\n")

    with open(os.path.join(dir, "tag.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + tag:
            if token in special_tokens + [" ", "+"]:
                f.write(f"{token}\n")
            else:
                f.write(f"/{token}\n")
