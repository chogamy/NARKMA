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
    parser.add_argument("--max_length", default=256, type=int)
    args = parser.parse_args()

    train = LOAD[args.data](args, "train")
    valid = LOAD[args.data](args, "valid")
    test = LOAD[args.data](args, "test")

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

    if "dict" in args.architecture:
        sentence_dict = {}
        # splits = ["train", "valid", "test"]
        splits = ["train"]
        for split in splits:
            for src, tgt in zip(
                dataset[split]["original_src"], dataset[split]["original_tgt"]
            ):
                src = "".join(src)
                tgt = "".join(tgt)
                sentence_dict[src] = tgt

        eoj_dict_with_cnt = {}
        for k, v in sentence_dict.items():
            src_eojs = k.split(" ")
            tgt_eojs = v.split(" ")
            assert len(src_eojs) == len(tgt_eojs)

            for src_eoj, tgt_eoj in zip(src_eojs, tgt_eojs):
                if src_eoj in eoj_dict_with_cnt:
                    if tgt_eoj in eoj_dict_with_cnt[src_eoj]:
                        eoj_dict_with_cnt[src_eoj][tgt_eoj] += 1
                    else:
                        eoj_dict_with_cnt[src_eoj][tgt_eoj] = 1
                else:
                    eoj_dict_with_cnt[src_eoj] = {tgt_eoj: 1}
        print(len(eoj_dict_with_cnt))

        threshold = 10
        eoj_dict = {}
        for k in eoj_dict_with_cnt.keys():
            if len(eoj_dict_with_cnt[k]) == 1:
                for t_k, t_v in eoj_dict_with_cnt[k].items():
                    if t_v > threshold:
                        eoj_dict[k] = t_k
        print(len(eoj_dict))

        with open(os.path.join(dir, "dict.txt"), encoding="utf-8", mode="w") as f:
            for k, v in eoj_dict.items():
                f.write(f"{k}\t{v}\n")
