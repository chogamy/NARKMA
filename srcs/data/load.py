import os

from tqdm import tqdm
from datasets import Dataset


class Text:
    def __init__(self, src: str, tgt: str, max_length, split):

        self.src = src
        self.tgt = tgt
        self.decompose_tgt = self.decompose(tgt)
        self.morph = self.only_morph(tgt)
        self.tag = self.only_pos(tgt)

        self.srcs, self.tgts = self.split(src, tgt, split, max_length)
        self.decompose_tgts = [self.decompose(tgt) for tgt in self.tgts]
        self.morphs = [self.only_morph(tgt) for tgt in self.tgts]
        self.tags = [self.only_pos(tgt) for tgt in self.tgts]

        # self.norm_src, self.norm_tgt = self.normalize(self.src, self.tgt)

    def decompose(self, tgt) -> list:
        eojs = tgt.split(" ")
        split_tgt = []
        for eoj in eojs:
            mts = eoj.split("+")
            for mt in mts:
                m, t = mt.rsplit("/", 1)
                split_tgt.append(m)
                split_tgt.append(f"/{t}")
                split_tgt.append("+")
            split_tgt = split_tgt[:-1]
            split_tgt.append(" ")
        split_tgt = split_tgt[:-1]

        assert "".join(split_tgt) == tgt, f"{split_tgt}\n{tgt}"

        return split_tgt

    def only_pos(self, tgt):
        tags = []
        eojs = tgt.split(" ")
        for eoj in eojs:
            mts = eoj.split("+")
            for mt in mts:
                m, t = mt.rsplit("/", 1)
                tags.append(t)
                tags.append("+")

            tags = tags[:-1]
            tags.append(" ")
        tags = tags[:-1]

        return tags

    def only_morph(self, tgt):
        morphs = []
        eojs = tgt.split(" ")
        for eoj in eojs:
            mts = eoj.split("+")
            for mt in mts:
                m, t = mt.rsplit("/", 1)
                morphs.append(m)
                morphs.append("+")

            morphs = morphs[:-1]
            morphs.append(" ")
        morphs = morphs[:-1]

        return morphs

    def normalize(self):
        pass

    def make_dict(self):
        pass

    def split(self, src, tgt, split, max_length):
        if len(self.decompose_tgt) < max_length:
            return [src], [tgt]

        srcs = []
        tgts = []
        if split in ["valid", "test"]:
            src_eojs = src.split(" ")
            tgt_eojs = tgt.split(" ")

            assert len(src_eojs) == len(tgt_eojs), f"{src_eojs}{tgt_eojs}"

            i = 0
            while i < len(src_eojs):
                if len(self.decompose(" ".join(tgt_eojs[i:]))) < max_length:
                    srcs.append(" ".join(src_eojs[i:]))
                    tgts.append(" ".join(tgt_eojs[i:]))
                    break
                j = len(src_eojs)
                while j > i:
                    if len(self.decompose(" ".join(tgt_eojs[i:j]))) < max_length:
                        srcs.append(" ".join(src_eojs[i:j]))
                        tgts.append(" ".join(tgt_eojs[i:j]))
                        i = j
                        break
                    j -= 1
                i += 1

        elif split in ["train"]:
            src_eojs = src.split(" ")
            tgt_eojs = tgt.split(" ")

            assert len(src_eojs) == len(tgt_eojs), f"{src_eojs}{tgt_eojs}"

            for i in range(len(src_eojs)):
                if len(self.decompose(" ".join(tgt_eojs[i:]))) < max_length:
                    srcs.append(" ".join(src_eojs[i:]))
                    tgts.append(" ".join(tgt_eojs[i:]))
                    break
                for j in range(len(src_eojs), 1, -1):
                    if len(self.decompose(" ".join(tgt_eojs[i:j]))) < max_length:
                        srcs.append(" ".join(src_eojs[i:j]))
                        tgts.append(" ".join(tgt_eojs[i:j]))
                        break

        else:
            raise ValueError("??")

        return srcs, tgts


def sejong(args, split):
    path = os.path.join(os.getcwd(), "data", "sejong")

    src, tgt = [], []
    dataset = []

    with open(os.path.join(path, f"{split}.txt"), encoding="utf-8") as f:
        for line in tqdm(f, desc=f"{split}"):
            line = line.strip()

            if line == "" and src != [] and tgt != []:
                src = " ".join(src)
                tgt = " ".join(tgt)
                text = Text(src, tgt, MAX_LENGTH[args.model], split)
                for src, tgt, decompose_tgt in zip(
                    text.srcs, text.tgts, text.decompose_tgts
                ):
                    dataset.append(
                        {
                            "src": text.src,
                            "tgt": text.tgt,
                            "decompose_tgt": text.decompose_tgt,
                            "split_src": src,
                            "split_tgt": tgt,
                            "split_decompose_tgt": decompose_tgt,
                            "morph": text.morph,
                            "tag": text.tag,
                        }
                    )

                src, tgt = [], []
            else:
                s, t = line.split(" ")
                src.append(s)
                tgt.append(t)

    dataset = Dataset.from_list(dataset)

    return dataset


LOAD = {"sejong": sejong}
MAX_LENGTH = {"lucadiliello/bart-small": 512, "google-t5/t5-small": 512}
