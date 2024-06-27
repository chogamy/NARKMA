import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import lightning as L
from transformers import BertModel, BertConfig


class Enc1NARDec2(nn.Module):
    def __init__(self, tokenizers):
        super().__init__()

        self.tokenizers = tokenizers

        encoder_config = BertConfig(
            vocab_size=tokenizers["src"].vocab_size,
            num_hidden_layers=6,
            hidden_size=512,
            num_attention_heads=8,
            intermediate_size=2048,
            is_decoder=False,
        )

        decoder_config0 = BertConfig(
            vocab_size=tokenizers["morph"].vocab_size,
            num_hidden_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            intermediate_size=2048,
            is_decoder=True,
            add_cross_attention=True,
        )

        decoder_config1 = BertConfig(
            vocab_size=tokenizers["tag"].vocab_size,
            num_hidden_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            intermediate_size=2048,
            is_decoder=True,
            add_cross_attention=True,
        )

        self.encoder = BertModel(encoder_config)
        self.length_predictor = nn.Linear(512, 512)
        self.decoder0 = BertModel(decoder_config0)
        self.morph_classifier = nn.Linear(512, tokenizers["morph"].vocab_size)
        self.decoder1 = BertModel(decoder_config1)
        self.tag_classifier = nn.Linear(512, tokenizers["tag"].vocab_size)

        self.space_id = tokenizers["tag"].token_to_id(" ")
        assert (
            tokenizers["src"].token_to_id(" ")
            == tokenizers["tag"].token_to_id(" ")
            == tokenizers["morph"].token_to_id(" ")
        )

    def forward(self):
        pass

    def predict(self, batch):
        # print(batch)
        enc_inp = {
            "input_ids": batch["enc_input_ids"],
            "attention_mask": batch["enc_attention_mask"],
        }

        enc_hidden = self.encoder(**enc_inp).last_hidden_state
        length_logits = self.length_predictor(enc_hidden)
        lengths = torch.argmax(length_logits, dim=-1)
        lengths[enc_inp["input_ids"] == self.space_id] = 0
        lengths[(enc_inp["input_ids"] != self.space_id) & (lengths == 0)] = 1
        lengths[enc_inp["input_ids"] == self.tokenizers["src"].pad_token_id] = -100
        dec_inp = self.tokenizers["length"].decode(
            lengths.tolist(),
            space_id=self.space_id,
            pad_id=-100,
        )
        lengths = [sum(a) for a in dec_inp["attention_mask"]]

        for k, v in dec_inp.items():
            dec_inp[k] = torch.tensor(v).to(device=enc_hidden.device)

        dec0_hidden = self.decoder0(
            **dec_inp,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_inp["attention_mask"]
        ).last_hidden_state
        morph_logit = self.morph_classifier(dec0_hidden)
        morphs = torch.argmax(morph_logit, dim=-1)
        morphs[dec_inp["input_ids"] == 0] = 0
        morphs[dec_inp["input_ids"] == self.space_id] = self.space_id
        morphs = self.tokenizers["morph"].batch_decode(morphs.tolist())

        dec1_hidden = self.decoder1(
            **dec_inp,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_inp["attention_mask"]
        ).last_hidden_state
        tag_logit = self.tag_classifier(dec1_hidden)
        tags = torch.argmax(tag_logit, dim=-1)
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

        return seqs


class LightningWrapper(L.LightningModule):
    def __init__(self, args, tokenizers, metric) -> None:
        super().__init__()

        self.model = ARCHITECTURE[args.architecture](tokenizers)

        self.tokenizer = tokenizers
        self.metric = metric

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_id):
        print("train")
        assert 0
        loss = None
        """
        output = self(batch)
        loss = loss_funct(output, batch['target'])
        """

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def predict(self, batch):
        outputs = self.model.predict(batch)

        return outputs

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        targets = ["".join(target) for target in batch["tgt"]]
        outputs = self.predict(batch)

        print(outputs)
        print(targets)
        assert 0

        result = self.metric(outputs, targets)

        self.log_dict({})

    @torch.no_grad()
    def test_step(self, batch, batch_id):
        model_input = {}
        target = None
        """
        model_input = pick(batch)
        """
        outputs = self.predict(**model_input)

        result = self.metric(outputs, target)

        self.log_dict({})

    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict()

    def on_test_epoch_end(self):
        self.eval()
        self.log_dict()

        self.metric.save()

    def configure_optimizers(self):
        # optimizer
        # lr_scheduler
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


ARCHITECTURE = {"Enc1NARDec2": Enc1NARDec2}
