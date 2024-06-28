import torch
import lightning as L
from transformers import get_cosine_schedule_with_warmup


from .model import ARCHITECTURE


class LightningWrapper(L.LightningModule):
    def __init__(self, args, tokenizers, metric) -> None:
        super().__init__()

        self.model = ARCHITECTURE[args.architecture](tokenizers)

        self.tokenizer = tokenizers
        self.metric = metric
        self.args = args

    def forward(self, batch):
        loss = self.model(batch)
        return loss

    def training_step(self, batch, batch_id):
        loss = self(batch)

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def predict(self, batch):
        outputs = self.model.predict(batch)

        return outputs

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        targets = ["".join(target) for target in batch["tgt"]]
        outputs = self.predict(batch)

        self.metric.update(outputs, targets)

    @torch.no_grad()
    def test_step(self, batch, batch_id):
        targets = ["".join(target) for target in batch["tgt"]]
        outputs = self.predict(batch)

        self.metric.update(outputs, targets)

    def on_validation_epoch_end(self):
        self.log_dict(self.metric.compute())
        self.metric.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metric.compute())
        self.metric.reset()

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.trainer.estimated_stepping_batches

        steps = int(self.args.batch_size * len(self.trainer.train_dataloader))
        warmup_steps = int(steps * self.args.warmup_rate)

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
