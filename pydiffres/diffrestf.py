from turtle import forward
import torch
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt

from pydiffres.core import BaseT, BaseF, PositionalEncoding
from pydiffres.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128
from pydiffres import DiffRes, DiffResF
from pydiffres.pooling import Pooling_layer
import time

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class DiffResTF(nn.Module):
    def __init__(
        self,
        in_t_dim,
        in_f_dim,
        dimension_reduction_rate_t,
        dimension_reduction_rate_f,
        learn_pos_emb=False,
        hidden_t=128,
    ):
        super(DiffResTF, self).__init__()
        
        self.dimension_reduction_rate_f=dimension_reduction_rate_f
        self.dimension_reduction_rate_t=dimension_reduction_rate_t
        self.in_f_dim=in_f_dim
        self.in_t_dim=in_t_dim
        
        self.feature_channels=3
        
        self.tune_dimension_reduction_rate()
    
        self.diffres_f = DiffResF(
            self.in_t_dim,
            self.in_f_dim,
            self.dimension_reduction_rate_f,
            learn_pos_emb,
            hidden_t_dim=hidden_t,
        )
        
        self.diffres_t = DiffRes(
            self.in_t_dim,
            int(self.in_f_dim * (1 - self.dimension_reduction_rate_f)),
            self.dimension_reduction_rate_t,
            learn_pos_emb,
        )

        pos_emb_y = PositionalEncoding(d_model=self.in_f_dim, max_len=self.in_t_dim)(
            torch.zeros((1, self.in_t_dim, self.in_f_dim))
        )
        # [1, 3000, 128]
        self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def tune_dimension_reduction_rate(self):
        out_t_len = int(self.in_t_dim * (1-self.dimension_reduction_rate_t))
        out_f_len = int(self.in_f_dim * (1-self.dimension_reduction_rate_f))
        if(out_t_len % 2 != 0):
            self.dimension_reduction_rate_t = 1-((out_t_len + 1) / self.in_t_dim)
        if(out_f_len % 2 != 0):
            self.dimension_reduction_rate_f = 1-((out_f_len + 1) / self.in_f_dim)
            
    def forward(self, x):
        # start = time.time()
        ret_f = self.diffres_f(x)
        weight_f, avgpool_f, guide_loss_f, activeness_f, score_f = (
            ret_f["weight"], # [1, 128, 64]
            ret_f["avgpool"], # [1, 3000, 64]
            ret_f["guide_loss"], 
            ret_f["activeness"],
            ret_f["score"]
        )
        # print("diffres f", time.time()-start); start = time.time()
        ret_t = self.diffres_t(avgpool_f)
        weight_t, avgpool_t, maxpool_t, guide_loss_t, activeness_t, score_t = (
            ret_t["weight"], # [1, 3000, 1500]
            ret_t["avgpool"], # [1, 1500, 64]
            ret_t["maxpool"], # [1, 1500, 64]
            ret_t["guide_loss"], 
            ret_t["activeness"],
            ret_t["score"],
        )
        # print("diffres t", time.time()-start); start = time.time()
        res_enc = torch.matmul(self.pos_emb, weight_f) # [1, 3000, 64]
        res_enc = torch.matmul(weight_t.permute(0,2,1), res_enc) # [1, 1500, 64]
        
        ret = {}
        ret["x"] = x
        
        ret["score_t"] = score_t
        ret["weight_t"] = weight_t
        ret["score_f"] = score_f
        ret["weight_f"] = weight_f
        
        ret["guide_loss_t"] = guide_loss_t
        ret["activeness_t"] = activeness_t
        ret["guide_loss_f"] = guide_loss_f
        ret["activeness_f"] = activeness_f
        
        ret["resolution_enc"] = res_enc
        ret["feature"] = torch.cat(
            [
                avgpool_t.unsqueeze(1),
                maxpool_t.unsqueeze(1),
                res_enc.unsqueeze(1),
            ],
            dim=1,
        )
        ret["guide_loss"], ret["activeness"] = guide_loss_f + guide_loss_t, activeness_f + activeness_t
        return ret
        
    def visualize(self, ret, savepath="."):
        x, y, emb, score_t, score_f = ret["x"], ret["feature"], ret["resolution_enc"], ret["score_t"], ret["score_f"]
        y = y[:, 0, :, :]
        for i in range(10):  # Visualize 10 images
            if i >= x.size(0):
                break
            plt.figure(figsize=(8, 16))
            plt.subplot(511)
            plt.title("Temporal importance score")
            plt.plot(score_t[i, :, 0].detach().cpu().numpy())
            plt.ylim([0, 1])
            plt.subplot(512)
            plt.title("Freq importance score")
            plt.plot(score_f[i, 0, :].detach().cpu().numpy())
            plt.ylim([0, 1])
            plt.subplot(513)
            plt.title("Original mel spectrogram")
            plt.imshow(
                x[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(514)
            plt.title(
                "DiffRes mel-spectrogram (Using avgpool frame aggregation function)"
            )
            plt.imshow(
                y[i, ...].detach().cpu().numpy().T, aspect="auto", interpolation="none"
            )
            plt.subplot(515)
            plt.title("Resolution encoding")
            plt.imshow(
                emb[i, ...].detach().cpu().numpy().T,
                aspect="auto",
                interpolation="none",
            )
            plt.savefig(os.path.join(savepath, "%s.png" % i))
            plt.close()
