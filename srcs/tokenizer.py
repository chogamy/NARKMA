import os


class BaseTokenizer:
    def __init__(self, path=None) -> None:

        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        if path is not None:
            self.vocab = self.read_vocab(path)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.bos_token_id = self.reverse_vocab[self.bos_token]
        self.eos_token_id = self.reverse_vocab[self.eos_token]
        self.pad_token_id = self.reverse_vocab[self.pad_token]
        self.mask_token_id = self.reverse_vocab[self.mask_token]
        self.unk_token_id = self.reverse_vocab[self.unk_token]

        self.special_tokens = {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id,
            self.mask_token: self.mask_token_id,
            self.unk_token: self.unk_token_id,
        }

    def __call__(self, texts: list, padding=None, truncation=None, max_length=None):
        outputs = {"input_ids": [], "attention_mask": []}
        for text in texts:
            text = self.normalize(text)
            text = self.pre_tokenize(text)
            input_ids = self.encode(text)

            assert len(input_ids) < max_length, f"{text}\n{max_length}\n{input_ids}"

            attention_mask = [1] * len(input_ids)

            if padding == "max_length":
                input_ids = input_ids + [self.pad_token_id] * (
                    max_length - len(input_ids)
                )
                attention_mask = attention_mask + [0] * (
                    max_length - len(attention_mask)
                )

            outputs["input_ids"].append(input_ids)
            outputs["attention_mask"].append(attention_mask)

        return outputs

    def id_to_token(self, id):
        if id in self.vocab:
            return self.vocab[id]
        else:
            return self.unk_token

    def token_to_id(self, token):
        if token in self.reverse_vocab:
            return self.reverse_vocab[token]
        else:
            return self.unk_token_id

    def read_vocab(self, path):
        vocab = {
            0: self.pad_token,
            1: self.mask_token,
            2: self.bos_token,
            3: self.eos_token,
            4: self.unk_token,
        }
        with open(path, encoding="utf-8", mode="r") as f:
            i = 0
            for token in f:
                while i in vocab.keys():
                    i += 1
                token = token.strip()
                if token == "":
                    token = " "
                vocab[i] = token

        return vocab

    def normalize(self, text):
        return text

    def pre_tokenize(self, text):
        return text

    def post_process(self):
        pass

    def encode(self, char_list):
        return [self.token_to_id(char) for char in char_list]

    def decode(self, id_list):
        # if skip_special_token == True:
        #     return [
        #         self.id_to_token(id)
        #         for id in id_list
        #         if id not in self.special_tokens.values()
        #     ]
        # else:
        return [self.id_to_token(id) for id in id_list]

    def batch_decode(self, id_lists):
        return [self.decode(id_list) for id_list in id_lists]


class srcTokenizer(BaseTokenizer):
    def __init__(self, args) -> None:
        path = os.path.join(
            os.getcwd(), "data", args.data, args.architecture, "vocab", "src.txt"
        )
        super().__init__(path)


class tgtTokenizer(BaseTokenizer):
    def __init__(self, args) -> None:
        path = os.path.join(
            os.getcwd(), "data", args.data, args.architecture, "vocab", "tgt.txt"
        )
        super().__init__(path)


class morphTokenizer(BaseTokenizer):
    def __init__(self, args) -> None:
        path = os.path.join(
            os.getcwd(), "data", args.data, args.architecture, "vocab", "morph.txt"
        )
        super().__init__(path)


class tagTokenizer(BaseTokenizer):
    def __init__(self, args) -> None:
        path = os.path.join(
            os.getcwd(), "data", args.data, args.architecture, "vocab", "tag.txt"
        )
        super().__init__(path)


class lengthTokenizer(BaseTokenizer):
    def __init__(self, args):
        self.pad_id = 0
        pass

    def encode(self, srcs, tgts, max_length=None):
        srcs = ["".join(src).split(" ") for src in srcs]
        tgts = ["".join(tgt).split(" ") for tgt in tgts]

        assert len(srcs) == len(tgts)

        lengths = []
        for src, tgt in zip(srcs, tgts):
            assert len(src) == len(tgt)
            length = []
            for s, t in zip(src, tgt):
                length.extend([len(t)] * len(s))
                length.append(0)
            length = length[:-1]
            length = length + [self.pad_id] * (max_length - len(length))
            lengths.append(length)

        return lengths

    def decode(
        self, len_seqs, tgts=None, max_length=512, mode="max", space_id=5, pad_id=-100
    ):
        # len_seq: 123, 321, 0, 345, 12, 0, -100, -100
        outputs = {"input_ids": [], "attention_mask": []}

        for i, len_seq in enumerate(len_seqs):
            inp = []
            part = []
            for l in len_seq:
                if l == pad_id:
                    break
                elif l == 0:
                    if part:
                        if mode == "max":
                            inp.extend([1] * max(part))
                            inp.append(space_id)
                        part = []
                else:
                    part.append(l)

            if part:
                if mode == "max":
                    inp.extend([1] * max(part))
                    inp.append(space_id)
                part = []

            inp = inp[:-1]

            if tgts is not None:
                assert len("".join(tgts[i])) == len(
                    inp
                ), f"{tgts[i]}{inp}\n{len(''.join(tgts[i]))}{len(inp)}"

            if len(inp) > max_length:
                inp = inp[:max_length]

            attention_mask = [1] * len(inp) + [0] * (max_length - len(inp))
            inp = inp + [0] * (max_length - len(inp))
            outputs["input_ids"].append(inp)
            outputs["attention_mask"].append(attention_mask)

        return outputs
