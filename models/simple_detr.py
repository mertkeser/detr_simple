import torch
from torch import nn
from torchvision.models import resnet18
from torch.optim import AdamW
import pytorch_lightning as pl

from utils.funcs import format_nuim_targets


class SimpleDETR(pl.LightningModule):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
        self,
        num_classes,
        lr,
        wd,
        loss_func,
        hidden_dim=128,
        nheads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet18()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.lr = lr
        self.wd = wd
        self.loss_func = loss_func

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 512 to 128 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # create a batch of positional encodings
        query_pos_batch = torch.stack(
            [self.query_pos for _ in range(batch_size)], dim=1
        )

        # propagate through the transformer
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1), query_pos_batch
        ).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }

    def training_step(self, batch, batch_idx):
        img, tgts = batch
        # self() calls self.forward()

        pred = self(img)
        formatted_tgts = format_nuim_targets(tgts)
        batch_losses_dict, loss = self.loss_func(pred, formatted_tgts)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, tgts = batch

        pred = self(img)
        formatted_tgts = format_nuim_targets(tgts)
        batch_losses_dict, loss = self.loss_func(pred, formatted_tgts)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
