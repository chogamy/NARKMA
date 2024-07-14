import os

import torch
from torch import nn
from torch.nn import functional as F
from .utils import Transformer


class Enc1NARDec2(nn.Module):
    def __init__(self, args, tokenizers):
        super().__init__()

        self.args = args
        self.tokenizers = tokenizers

        self.space_id = tokenizers["tag"].token_to_id(" ")
        assert (
            tokenizers["src"].token_to_id(" ")
            == tokenizers["tag"].token_to_id(" ")
            == tokenizers["morph"].token_to_id(" ")
        )

        self.encoder = Transformer(
            emb_size=tokenizers["src"].vocab_size,
            pos="absolute",
            max_length=args.max_length,
            num_hidden_layers=6,
            hidden_size=512,
            num_attention_heads=8,
            activation_funcion="gelu",
            intermediate_size=2048,
            classifier_size=args.max_length,
            dropout=0.1,
            is_decoder=False,
            is_causal=False,
        )

        self.decoder0 = Transformer(
            emb_size=6,
            pos="absolute",
            max_length=args.max_length,
            num_hidden_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            activation_funcion="gelu",
            intermediate_size=2048,
            classifier_size=tokenizers["morph"].vocab_size,
            dropout=0.1,
            is_decoder=True,
            is_causal=False,
        )
        self.decoder1 = Transformer(
            emb_size=6,
            pos="absolute",
            max_length=args.max_length,
            num_hidden_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            activation_funcion="gelu",
            intermediate_size=2048,
            classifier_size=tokenizers["tag"].vocab_size,
            dropout=0.1,
            is_decoder=True,
            is_causal=False,
        )

        if args.dict:
            self.dict = {}
            with open(
                os.path.join(
                    os.getcwd(),
                    "data",
                    args.data,
                    args.architecture,
                    "vocab",
                    "dict.txt",
                ),
                encoding="utf-8",
                mode="r",
            ) as f:
                for line in f:
                    s, t = line.strip().split("\t")
                    self.dict[s] = t
        else:
            self.dict = None

    def forward(self, batch):
        enc_inp = {
            "input_ids": batch["enc_input_ids"],
            "attention_mask": batch["enc_attention_mask"],
        }
        enc_out = self.encoder(
            **enc_inp, output_last_hidden_state=True, output_logits=True
        )
        length_loss = F.cross_entropy(enc_out.logits.transpose(1, 2), batch["enc_tgt"])

        dec0_inp = {
            "input_ids": batch["dec_input_ids"],
            "attention_mask": batch["dec_attention_mask"],
            "cross_hidden_state": enc_out.last_hidden_state,
            "cross_hidden_attention_mask": enc_inp["attention_mask"],
        }
        dec0_out = self.decoder0(**dec0_inp, output_logits=True)
        dec0_loss = F.cross_entropy(dec0_out.logits.transpose(1, 2), batch["dec0_tgt"])

        dec1_inp = {
            "input_ids": batch["dec_input_ids"],
            "attention_mask": batch["dec_attention_mask"],
            "cross_hidden_state": enc_out.last_hidden_state,
            "cross_hidden_attention_mask": enc_inp["attention_mask"],
        }
        dec0_out = self.decoder1(**dec1_inp, output_logits=True)
        dec1_loss = F.cross_entropy(dec0_out.logits.transpose(1, 2), batch["dec1_tgt"])

        return length_loss + dec0_loss + dec1_loss

    def predict(self, batch):
        enc_inp = {
            "input_ids": batch["enc_input_ids"],
            "attention_mask": batch["enc_attention_mask"],
        }

        enc_out = self.encoder(
            **enc_inp, output_last_hidden_state=True, output_logits=True
        )
        lengths = torch.argmax(enc_out.logits, dim=-1)
        lengths[enc_inp["input_ids"] == self.space_id] = 0
        lengths[(enc_inp["input_ids"] != self.space_id) & (lengths == 0)] = 1
        lengths[enc_inp["input_ids"] == self.tokenizers["src"].pad_token_id] = -100
        dec_inp = self.tokenizers["length"].decode(
            lengths.tolist(),
            space_id=self.space_id,
            max_length=self.args.max_length,
            pad_id=-100,
        )
        lengths = [sum(a) for a in dec_inp["attention_mask"]]

        for k, v in dec_inp.items():
            dec_inp[k] = torch.tensor(v).to(device=enc_out.logits.device)

        dec0_out = self.decoder0(
            **dec_inp,
            cross_hidden_state=enc_out.last_hidden_state,
            cross_hidden_attention_mask=enc_inp["attention_mask"],
            output_logits=True,
        )
        morphs = torch.argmax(dec0_out.logits, dim=-1)
        morphs[dec_inp["input_ids"] == 0] = 0
        morphs[dec_inp["input_ids"] == self.space_id] = self.space_id
        morphs = self.tokenizers["morph"].batch_decode(morphs.tolist())

        dec1_out = self.decoder1(
            **dec_inp,
            cross_hidden_state=enc_out.last_hidden_state,
            cross_hidden_attention_mask=enc_inp["attention_mask"],
            output_logits=True,
        )
        tags = torch.argmax(dec1_out.logits, dim=-1)
        tags[dec_inp["input_ids"] == 0] = 0
        tags[dec_inp["input_ids"] == self.space_id] = self.space_id
        tags = self.tokenizers["tag"].batch_decode(tags.tolist())

        def merge(morphs, tags, lengths):
            seqs = []
            for morph, tag, length in zip(morphs, tags, lengths):
                seq = []
                tag_pointer = ""
                for m, t in zip(morph[:length], tag[:length]):
                    if tag_pointer == "":
                        tag_pointer = t
                    if m in ["+", " "]:
                        seq.append(tag_pointer)
                        seq.append(m)
                        tag_pointer = ""
                    else:
                        seq.append(m)

                if tag_pointer != "":
                    seq.append(tag_pointer)

                seqs.append("".join(seq))
            return seqs

        seqs = merge(morphs, tags, lengths)

        if self.dict is not None:
            new_seqs = []
            srcs = ["".join(src) for src in batch["src"]]
            for src, seq in zip(srcs, seqs):
                eojs = []
                src_eojs = src.split(" ")
                seq_eojs = seq.split(" ")
                for src_eoj, seq_eoj in zip(src_eojs, seq_eojs):
                    answer = self.dict.get(src_eoj, None)
                    if answer is None:
                        eojs.append(seq_eoj)
                    else:
                        eojs.append(answer)
                new_seqs.append(" ".join(eojs))
            return new_seqs

        return seqs
