import os
import argparse

from datasets import DatasetDict

from srcs.data.load import LOAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True, default=None, type=str)
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

    train = LOAD[args.data]("train")
    valid = LOAD[args.data]("valid")
    test = LOAD[args.data]("test")

    dataset = DatasetDict({"train": train, "valid": valid, "test": test})

    print(dataset)

    src = list(
        set(
            "".join(item for sub in dataset["train"]["src"] for item in sub)
            + "".join(item for sub in dataset["valid"]["src"] for item in sub)
            + "".join(item for sub in dataset["test"]["src"] for item in sub)
        )
    )
    src.sort()

    morph = list(
        set(
            "".join([item for sub in dataset["train"]["morph_tgt"] for item in sub])
            + "".join([item for sub in dataset["valid"]["morph_tgt"] for item in sub])
            + "".join([item for sub in dataset["test"]["morph_tgt"] for item in sub])
        )
    )
    morph.sort()

    tag = list(
        set(
            [item for sub in dataset["train"]["tag_tgt"] for item in sub]
            + [item for sub in dataset["valid"]["tag_tgt"] for item in sub]
            + [item for sub in dataset["test"]["tag_tgt"] for item in sub]
        )
    )
    tag.sort()

    # special_tokens = ["<bos>", "<eos>", "<mask>", "<unk>", "<pad>"]
    special_tokens = []

    dir = os.path.join(os.getcwd(), "data", args.data, args.architecture, "vocab")
    os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, "src.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + src:
            f.write(f"{token}\n")

    with open(os.path.join(dir, "tgt.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + tag:
            if token in special_tokens + [" ", "+"]:
                f.write(f"{token}\n")
            else:
                f.write(f"/{token}\n")
        for token in morph:
            if token not in [" ", "+"]:
                f.write(f"{token}\n")

    with open(os.path.join(dir, "morph.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + morph:
            f.write(f"{token}\n")

    with open(os.path.join(dir, "tag.txt"), encoding="utf-8", mode="w") as f:
        for token in special_tokens + tag:
            f.write(f"{token}\n")

            # if token in special_tokens + [" ", "+"]:
            #     f.write(f"{token}\n")
            # else:
            #     f.write(f"/{token}\n")
