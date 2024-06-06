import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import lightning as L


class NNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.encoder = model.encoder
        self.decoder = model.decoder

        # ETC.......


class LightningWrapper(L.LightningModule):
    def __init__(self, model, tokenizer, metric) -> None:
        super().__init__()

        self.model = NNWrapper(model)

        self.tokenizer = tokenizer
        self.metric = metric

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_id):
        loss = None
        """
        output = self(batch)
        loss = loss_funct(output, batch['target'])
        """

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def predict(self, batch, batch_id):
        outputs = self(**batch)
        """
        outputs = some_processing(outputs)
        """

        return outputs

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        model_input = {}
        target = None
        """
        model_input = pick(batch)
        """
        outputs = self.predict(**model_input)

        result = self.metric(outputs, target)

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
