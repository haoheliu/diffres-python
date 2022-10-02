import os
import torch
from pydiffres import DiffRes, AvgPool, AvgMaxPool, ConvAvgPool, ChangeHopSize

def test(module):
    print(module)
    model = eval(module)(
        in_t_dim=3000, in_f_dim=128, dimension_reduction_rate=0.75, learn_pos_emb=False
    )

    data = torch.randn(1, 3000, 128)  # Batchsize, t-steps, mel-bins

    ret = model(data)
    
    os.makedirs(module, exist_ok=True)
    
    model.visualize(ret, savepath=module)
    
test("DiffRes")
test("AvgPool")
test("AvgMaxPool")
test("ConvAvgPool")
test("ChangeHopSize")
