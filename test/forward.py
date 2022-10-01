from pydiffres import DiffRes, AvgPool
import torch

model = DiffRes(
    in_t_dim=3000, in_f_dim=128, dimension_reduction_rate=0.5, learn_pos_emb=False
)
# model = AvgPool(in_t_dim=3000, in_f_dim=128, dimension_reduction_rate=0.5, learn_pos_emb=False)

data = torch.randn(1, 3000, 128)  # Batchsize, t-steps, mel-bins

ret = model(data)

import ipdb

ipdb.set_trace()
