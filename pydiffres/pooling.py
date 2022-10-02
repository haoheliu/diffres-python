import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair

def _spectral_crop(input, oheight, owidth):

        cutoff_freq_h = math.ceil(oheight / 2)
        cutoff_freq_w = math.ceil(owidth / 2)

        if oheight % 2 == 1:
                if owidth % 2 == 1:
                        top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
                        bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
                        bottom_right = input[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
                else:
                        top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
                        bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
                        bottom_right = input[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
        else:
                if owidth % 2 == 1:
                        top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
                        bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
                        bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
                else:
                        top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
                        bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
                        bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

        top_combined = torch.cat((top_left, top_right), dim=-1)
        bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
        all_together = torch.cat((top_combined, bottom_combined), dim=-2)

        return all_together

def _spectral_pad(input, output, oheight, owidth):
        cutoff_freq_h = math.ceil(oheight / 2)
        cutoff_freq_w = math.ceil(owidth / 2)

        pad = torch.zeros_like(input)

        if oheight % 2 == 1:
                if owidth % 2 == 1:
                        pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
                        pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
                        pad[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):] = output[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
                else:
                        pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
                        pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
                        pad[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
        else:
                if owidth % 2 == 1:
                        pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
                        pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
                        pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
                else:
                        pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
                        pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
                        pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
                        pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

        return pad

def DiscreteHartleyTransform(input):
    # fft = torch.rfft(input, 2, normalized=True, onesided=False)
    # for new version of pytorch
    fft = torch.fft.fft2(input, dim=(-2, -1), norm='ortho')
    fft = torch.stack((fft.real, fft.imag), -1)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class SpectralPoolingFunction(Function):
        @staticmethod
        def forward(ctx, input, oheight, owidth):
                ctx.oh = oheight
                ctx.ow = owidth
                ctx.save_for_backward(input)

                # Hartley transform by RFFT
                dht = DiscreteHartleyTransform(input)

                # frequency cropping
                all_together = _spectral_crop(dht, oheight, owidth)
                # inverse Hartley transform
                dht = DiscreteHartleyTransform(all_together)
                return dht

        @staticmethod
        def backward(ctx, grad_output):
                input, = ctx.saved_variables

                # Hartley transform by RFFT
                dht = DiscreteHartleyTransform(grad_output)
                # frequency padding
                grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
                # inverse Hartley transform
                grad_input = DiscreteHartleyTransform(grad_input)
                return grad_input, None, None

class SpectralPool2d(nn.Module):
        def __init__(self, scale_factor):
                super(SpectralPool2d, self).__init__()
                self.scale_factor = _pair(scale_factor)
        def forward(self, input):
                H, W = input.size(-2), input.size(-1)
                h, w = math.ceil(H*self.scale_factor[0]), math.ceil(W*self.scale_factor[1])
                return SpectralPoolingFunction.apply(input, h, w)



class Pooling_layer(nn.Module):
    def __init__(self, pooling_type='max', factor=24):
        super(Pooling_layer, self).__init__()
        self.factor = factor
        self.pooling_type = pooling_type

        if self.pooling_type == 'spec':
            self.SpecPool2d = SpectralPool2d(scale_factor=(factor, 1))

    def forward(self, x):
        """
        args:
            x: input mel spectrogram [batch, 1, time, frequency]
        return:
            out: reduced features [batch, 1, factor, frequency]
        """
        factor = int(x.shape[2] * self.factor)

        if self.pooling_type == 'avg':
            size = x.shape[2] // factor
            out = F.avg_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'max':
            size = x.shape[2] // factor
            out =  F.max_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'avg-max':
            size = x.shape[2] // factor
            out1 =  F.max_pool2d(x, kernel_size=(size, 1))
            out2 = F.avg_pool2d(x, kernel_size=(size, 1))
            out = (out1 + out2) / 2
        elif self.pooling_type == 'uniform':
            out =  self.uniform_sample(x, factor)
        elif self.pooling_type == 'spec':
            out = self.SpecPool2d(x)

        return out

    def uniform_sample(self, input, factor):
        """
            args:
                x: input mel spectrogram [batch, 1, time, frequency]
            return:
                out: reduced features [batch, 1, factor, frequency]
            """
        indexes = torch.linspace(0, input.shape[2]-1, factor).tolist()
        indexes = [int(num) for num in indexes]

        output = input[:, :, indexes, :]

        return output



if __name__ == '__main__':
    input = torch.tensor([7.,8.,9.,10.,11.,1.,2.,3.,4.,5.,6.,15.,16.,17.,18.,19.,12.,13.,14.,20.]).reshape(4,5).unsqueeze(0).unsqueeze(0)
    model = Pooling_layer(pooling_type='spec', factor=0.5)
    print(model(input).shape)


if __name__ == '__main__':
    input = torch.randn(4, 1, 100, 64)
    layer = SpectralPool2d(scale_factor=(0.1, 1))
    out = layer(input)