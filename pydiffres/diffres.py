import torch
import torch.nn as nn
import torch
import logging
import os
import matplotlib.pyplot as plt

from pydiffres.core import Base
from pydiffres.dilated_convolutions_1d.conv import DilatedConv

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4


class DiffRes(Base):
    def __init__(
        self, in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb=False
    ):
        super().__init__(in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb)
        self.feature_channels = 3

        self.model = DilatedConv(
            in_channels=self.input_f_dim,
            dilation_rate=1,
            input_size=self.input_seq_length,
            kernel_size=5,
            stride=1,
        )

    def forward(self, x):
        ret = {}
        score = torch.sigmoid(self.model(x.permute(0, 2, 1)).permute(0, 2, 1))
        score, _ = self.score_norm(score, self.output_seq_length)
        mean_feature, max_pool_feature, mean_pos_enc = self.frame_warping(
            x.exp(), score, total_length=self.output_seq_length
        )

        mean_feature = torch.log(mean_feature + EPS)
        max_pool_feature = torch.log(max_pool_feature + EPS)

        ret["x"] = x
        ret["score"] = score
        ret["resolution_enc"] = mean_pos_enc
        ret["avgpool"] = mean_feature
        ret["maxpool"] = max_pool_feature
        ret["feature"] = torch.cat(
            [
                mean_feature.unsqueeze(1),
                max_pool_feature.unsqueeze(1),
                mean_pos_enc.unsqueeze(1),
            ],
            dim=1,
        )
        ret["guide_loss"], ret["activeness"] = self.guide_loss(
            x, importance_score=score
        )
        return ret

    def frame_warping(self, feature, score, total_length):
        weight = self.calculate_weight(score, feature, total_length=total_length)

        mean_feature = torch.matmul(weight.permute(0, 2, 1), feature)
        max_pool_feature = self.calculate_scatter_maxpool_odd_even_lines(
            weight, feature, out_len=self.output_seq_length
        )
        mean_pos_enc = torch.matmul(weight.permute(0, 2, 1), self.pos_emb)

        return mean_feature, max_pool_feature, mean_pos_enc

    def visualize(self, ret, savepath="."):
        x, y, emb, score = ret["x"], ret["feature"], ret["resolution_enc"], ret["score"]
        y = y[:, 0, :, :]
        for i in range(x.size(0)):
            plt.figure(figsize=(8, 16))
            plt.subplot(411)
            plt.title("Importance score")
            plt.plot(score[i, :, 0].detach().cpu().numpy())
            plt.subplot(412)
            plt.title("Original mel spectrogram")
            plt.imshow(
                x[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(413)
            plt.title(
                "DiffRes mel-spectrogram (Using avgpool frame aggregation function)"
            )
            plt.imshow(
                y[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(414)
            plt.title("Resolution encoding")
            plt.imshow(
                emb[i, ...].detach().cpu().numpy().T,
                aspect="auto",
                interpolation="none",
            )
            plt.savefig(os.path.join(savepath, "%s.png" % i))
            plt.close()


class AvgPool(Base):
    def __init__(
        self, in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb=False
    ):
        super().__init__(in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb)
        self.feature_channels = 1

    def forward(self, x):
        ret = {}
        ret["x"] = x
        ret["score"] = None
        ret["resolution_enc"] = None
        ret["avgpool"] = self.pooling(x.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1)
        ret["maxpool"] = None
        ret["feature"] = ret["avgpool"]
        ret["guide_loss"], ret["activeness"] = None, None
        return ret

    def visualize(self, ret, savepath="."):
        x, y = ret["x"], ret["feature"]
        for i in range(x.size(0)):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.title("Original spectrogram")
            plt.imshow(
                x[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(212)
            plt.title("After average pooling")
            plt.imshow(
                y[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            # path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(savepath, "%s.png" % i))
            plt.close()


# def test_sampler(sampler, data=None):
#     input_tdim = 1056
#     sampler = sampler(input_seq_length=input_tdim, preserve_ratio=0.5)
#     if(data is None): test_input = torch.rand([3, input_tdim, 128])
#     else: test_input = data
#     ret =sampler(test_input)
#     assert "score" in ret.keys()
#     assert "score_loss" in ret.keys()
#     assert "energy" in ret.keys()
#     assert "feature" in ret.keys()
#     sampler.visualize(ret)
#     print("Perfect!", sampler, ret["feature"].size(), ret["score_loss"].size(), ret["score_loss"])
#     return ret["feature"]

# # YZxq2_xOLT8o_0
# if __name__ == "__main__":
#     from HigherModels import *
#     from models.diffres import *
#     from pooling import Pooling_layer
#     import logging

#     import numpy as np

#     logging.basicConfig(
#     filename="log.txt",
#     filemode="a",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s: %(message)s",
#     datefmt="%m/%d/%Y %I:%M:%S %p",
#     )
#     data = torch.rand([3, 1056, 128])

#     out1 = test_sampler(Proposed, data=data)
