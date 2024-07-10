import os

from tqdm import tqdm
from datasets import Dataset

TAG = [
    "/EC",
    "/EF",
    "/EP",
    "/ETM",
    "/ETN",
    "/IC",
    "/JC",
    "/JKB",
    "/JKC",
    "/JKG",
    "/JKO",
    "/JKQ",
    "/JKS",
    "/JKV",
    "/JX",
    "/MAG",
    "/MAJ",
    "/MM",
    "/NA",
    "/NNB",
    "/NNG",
    "/NNP",
    "/NP",
    "/NR",
    "/SE",
    "/SF",
    "/SH",
    "/SL",
    "/SN",
    "/SO",
    "/SP",
    "/SS",
    "/SW",
    "/VA",
    "/VCN",
    "/VCP",
    "/VV",
    "/VX",
    "/XPN",
    "/XR",
    "/XSA",
    "/XSN",
    "/XSV",
]


class Text:
    def __init__(self, src: str, tgt: str, max_length, split):
        """
        if train:
        ~~~
        else:
        return src
        """

        self.src = src
        self.tgt = tgt
        self.decompose_tgt = self.decompose(tgt)
        self.morph = self.only_morph(self.decompose_tgt)
        self.tag = self.only_tag(self.decompose_tgt)

        self.srcs, self.tgts = self.split(src, tgt, split, max_length)
        self.srcs = [list(src) for src in self.srcs]
        self.decompose_tgts = [self.decompose(tgt) for tgt in self.tgts]
        self.morphs = [self.only_morph(tgt) for tgt in self.decompose_tgts]
        self.tags = [self.only_tag(tgt) for tgt in self.decompose_tgts]
        self.expand_tags = [
            self.expand_tag(tag, morph) for tag, morph in zip(self.tags, self.morphs)
        ]

        self.src = list(self.src)

        # self.encoder_target = [
        #     self.encoder_output(src, morph)
        #     for src, morph in zip(self.srcs, self.morphs)
        # ]
        # self.decoder_input_ids = [self.decoder_input(morph) for morph in self.morphs]

        # self.norm_src, self.norm_tgt = self.normalize(self.src, self.tgt)

    def decompose(self, tgt) -> list:
        """
        morph/tag+morph/tag -> m,o,r,p,h, /tag, m,o,r,p,h, /tag
        """
        eojs = tgt.split(" ")
        split_tgt = []
        for eoj in eojs:
            mts = eoj.split("+")
            for mt in mts:
                m, t = mt.rsplit("/", 1)
                # split_tgt.append(list(m))
                split_tgt.extend(list(m))
                split_tgt.append(f"/{t}")
                split_tgt.append("+")
            split_tgt = split_tgt[:-1]
            split_tgt.append(" ")
        split_tgt = split_tgt[:-1]

        assert "".join(split_tgt) == tgt, f"{split_tgt}\n{tgt}"

        return split_tgt

    def only_tag(self, tgt: list):
        tags = []
        for t in tgt:
            if t in TAG + ["+", " "]:
                tags.append(t)
        return tags

    def only_morph(self, tgt: list):
        morphs = []
        for t in tgt:
            if t not in TAG:
                morphs.append(t)
        return morphs

    def expand_tag(self, tag: list, morph: list):

        morph_counter = []

        cnt = 0
        for m in morph:
            if m in [" ", "+"]:
                morph_counter.append(cnt)
                cnt = 0
            else:
                cnt += 1

        morph_counter.append(cnt)

        expand_tag = []

        for t in tag:
            if t in [" ", "+"]:
                expand_tag.append(t)
            else:
                l = morph_counter.pop(0)
                expand_tag.extend([t] * l)

        # assert len(expand_tag) < 512
        assert len(expand_tag) == len(
            morph
        ), f"{len(expand_tag)}{len(morph)}\n{expand_tag}\n{morph}"
        assert len(morph_counter) == 0

        return expand_tag

    def normalize(self):
        pass

    def make_dict(self):
        pass

    def split(self, src, tgt, split, max_length):
        """
        split by max_length

        (max_length = 2)
        if valid, test:
            abc -> ab, c
        if train:
            abc -> ab, bc
        """
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

    source = "srcs"
    morph_target = "morphs"
    tag_target = "expand_tags"
    target = "decompose_tgts"

    with open(os.path.join(path, f"{split}.txt"), encoding="utf-8-sig") as f:
        for line in tqdm(f, desc=f"{split}"):
            line = line.strip()

            if line == "" and src != [] and tgt != []:
                src = " ".join(src)
                tgt = " ".join(tgt)
                text = Text(src, tgt, args.max_length, split)
                for src, morph_tgt, tag_tgt, tgt in zip(
                    getattr(text, source),
                    getattr(text, morph_target),
                    getattr(text, tag_target),
                    getattr(text, target),
                ):
                    dataset.append(
                        {
                            "original_src": text.src,
                            "original_tgt": text.decompose_tgt,
                            "src": src,
                            "morph_tgt": morph_tgt,
                            "tag_tgt": tag_tgt,
                            "tgt": tgt,
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
