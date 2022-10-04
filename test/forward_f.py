import os
import torch
from pydiffres import DiffRes, AvgPool, AvgMaxPool, ConvAvgPool, ChangeHopSize, DiffResF

def test_t(module):
    print(module)
    model = eval(module)(
        in_t_dim=3000, in_f_dim=128, hidden_t_dim=300, dimension_reduction_rate=0.75, learn_pos_emb=False
    )

    data = torch.randn(1, 3000, 128)  # Batchsize, t-steps, mel-bins

    ret = model(data)
    
    os.makedirs(module, exist_ok=True)
    
    model.visualize(ret, savepath=module)
    
def test_f(module):
    print(module)
    model = eval(module)(
        in_t_dim=98, in_f_dim=256, hidden_t_dim=98, dimension_reduction_rate=0.75, learn_pos_emb=False
    )

    data = torch.randn(128, 98, 256)  # Batchsize, t-steps, mel-bins

    ret = model(data)
    
    os.makedirs(module, exist_ok=True)
    
    model.visualize(ret, savepath=module)
    
# test_t("DiffRes")
# test_t("AvgPool")
# test_t("AvgMaxPool")
# test_t("ConvAvgPool")
# test_t("ChangeHopSize")

test_f("DiffResF")
