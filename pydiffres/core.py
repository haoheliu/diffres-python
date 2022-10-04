import torch
import torch.nn as nn
import torch
import math
import logging
from torch_scatter import scatter_max, scatter_add

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x).permute(1, 0, 2)

class Base(nn.Module):
    def __init__(
        self, in_t_dim, in_f_dim, dimension_reduction_rate, learn_pos_emb=False
    ):
        super(Base, self).__init__()
        
        assert dimension_reduction_rate < 1.0, "Error: Invalid dimension reduction rate %s" % dimension_reduction_rate
        
        self.dimension_reduction_rate = dimension_reduction_rate
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        self.output_seq_length = int(self.input_seq_length * (1-self.dimension_reduction_rate))

        self.model = None
        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)

        emb_dropout = 0.0
        logging.info("Use positional embedding")
        pos_emb_y = PositionalEncoding(
            d_model=self.input_f_dim, dropout=emb_dropout, max_len=self.input_seq_length
        )(torch.zeros((1, self.input_seq_length, self.input_f_dim)))
        self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self):
        raise NotImplementedError

    def frame_warping(self):
        raise NotImplementedError

    # def denormalize(self, x):
    #     return x * self.std + self.mean

    # def normalize(self, x):
    #     return ( x - self.mean ) / self.std

    def zero_loss_like(self, x):
        return torch.tensor([0.0]).to(x.device)

    def interpolate(self, score):
        return torch.nn.functional.interpolate(
            score, size=self.input_seq_length, mode="linear"
        )

    def pool(self, x):
        return (
            self.pooling(x.permute(0, 2, 1)).permute(0, 2, 1)
            + self.max_pooling(x.permute(0, 2, 1)).permute(0, 2, 1)
        ) / 2

    def calculate_scatter_avgpool(self, score, feature, out_len):
        # cumsum: [3, 1056, 1]
        # feature: [3, 1056, 128]
        bs, in_seq_len, feat_dim = feature.size()
        cumsum = torch.cumsum(score, dim=1)
        # Float point problem here. Need to remove the garbage float points
        cumsum[
            cumsum % 1 < 1e-2
        ] -= (
            1e-2  # In case you perform normalization and the sum is equal to an integer
        )

        int_cumsum = torch.floor(cumsum.float()).permute(0, 2, 1).long()
        int_cumsum = torch.clip(int_cumsum, min=0, max=out_len - 1)
        out = torch.zeros((bs, feat_dim, out_len)).to(score.device)

        # feature: [bs, feat-dim, in-seq-len]
        # int_cumsum: [bs, 1, in-seq-len]
        # out: [bs, feat-dim, out-seq-len]
        out = scatter_add((feature * score).permute(0, 2, 1), int_cumsum, out=out)
        return out.permute(0, 2, 1)

    def calculate_scatter_maxpool(self, score, feature, out_len):

        # cumsum: [3, 1056, 1]
        # feature: [3, 1056, 128]
        bs, in_seq_len, feat_dim = feature.size()
        cumsum = torch.cumsum(score, dim=1)
        # Float point problem here
        cumsum[
            cumsum % 1 < 1e-2
        ] -= (
            1e-2  # In case you perform normalization and the sum is equal to an integer
        )

        int_cumsum = torch.floor(cumsum.float()).permute(0, 2, 1).long()
        int_cumsum = torch.clip(int_cumsum, min=0, max=out_len - 1)
        out = torch.zeros((bs, feat_dim, out_len)).to(score.device)

        # feature: [bs, feat-dim, in-seq-len]
        # int_cumsum: [bs, 1, in-seq-len]
        # out: [bs, feat-dim, out-seq-len]
        out, _ = scatter_max((feature * score).permute(0, 2, 1), int_cumsum, out=out)
        return out.permute(0, 2, 1) * (1 / (1-self.dimension_reduction_rate))

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()

        assert orig_len >= target_len

        weight = torch.tensor([-1.0, 1.0]).expand(target_len, -1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(
            mask.permute(0, 2, 1).float(),
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=target_len,
        )
        value = torch.nn.functional.pad(value, (1, 0))
        value = value.permute(0, 2, 1)
        return value == 1, value == -1

    def calculate_scatter_maxpool_odd_even_lines(self, weight, feature, out_len):
        odd_score, odd_index = self.select_odd_dimensions(weight)
        even_score, even_index = self.select_even_dimensions(weight)
        out_odd = self.calculate_scatter_maxpool(
            odd_score, feature, out_len=int(torch.sum(odd_index).item())
        )
        out_even = self.calculate_scatter_maxpool(
            even_score, feature, out_len=int(torch.sum(even_index).item())
        )

        assert out_odd.size(1) + out_even.size(1) == out_len

        out = torch.zeros(out_odd.size(0), out_len, out_odd.size(2)).to(feature.device)
        out[:, even_index, :] = out_even
        out[:, odd_index, :] = out_odd
        return out

    def calculate_scatter_avgpool_odd_even_lines(self, weight, feature, out_len):
        odd_score, odd_index = self.select_odd_dimensions(weight)
        even_score, even_index = self.select_even_dimensions(weight)

        out_odd = self.calculate_scatter_avgpool(
            odd_score, feature, out_len=int(torch.sum(odd_index).item())
        )
        out_even = self.calculate_scatter_avgpool(
            even_score, feature, out_len=int(torch.sum(even_index).item())
        )

        assert out_odd.size(1) + out_even.size(1) == out_len

        out = torch.zeros(out_odd.size(0), out_len, out_odd.size(2)).to(feature.device)
        out[:, even_index, :] = out_even
        out[:, odd_index, :] = out_odd
        return out

    def select_odd_dimensions(self, weight):
        # torch.Size([1, 10, 5])
        length = weight.size(-1)
        odd_index = torch.arange(length) % 2 == 1
        odd_score = torch.sum(weight[:, :, odd_index], dim=2, keepdim=True)
        return odd_score, odd_index

    def select_even_dimensions(self, weight):
        # torch.Size([1, 10, 5])
        length = weight.size(-1)
        even_index = torch.arange(length) % 2 == 0
        even_score = torch.sum(weight[:, :, even_index], dim=2, keepdim=True)
        return even_score, even_index

    def score_norm(self, score, total_length):
        ####################################################################
        # Trying to rescale the total score
        sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if torch.sum(dims_need_norm) > 0:
            score[dims_need_norm] = (
                score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
            )
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between
        # torch.Size([32, 1056, 1])
        if torch.sum(dims_need_norm) > 0:
            sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
            distance_with_target_length = (total_length - sum_score)[:, 0, 0]
            axis = torch.logical_and(
                score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN
            )  # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if distance_with_target_length[i] >= 1:
                    intervel = 1.0 - score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel)
                    if alpha > 1:
                        alpha = 1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def update_weight(self, weight):
        # Slow version
        weight = weight.permute(0, 2, 1)
        bs, gamma, m = weight.size()
        for b in range(bs):
            i, j, s = 0, 0, 0
            while i < gamma - 1 and j < m - 1:
                if weight[b, i, j] > 0:
                    s += weight[b, i, j]
                    j += 1
                    continue
                else:
                    weight[b, i, j] = 1 - s
                    weight[b, i + 1, j] -= weight[b, i, j]
                    i += 1
                    s = 0
        return weight.permute(0, 2, 1)

    def calculate_weight(self, score, feature, total_length):
        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(
            feature.size(0), feature.size(1), total_length
        )
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum of each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1 - weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:, :, 1:] * one_minus_weight_sum_cumsum[:, :, :-1]
        need_minus = torch.nn.functional.pad(need_minus, (1, 0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add
        weight = torch.clip(weight, min=0.0, max=1.0)  # [66, 1056, 264]
        return weight

    def guide_loss(self, mel, importance_score):
        # If the mel spectrogram is in log scale
        # mel: [bs, t-steps, f-bins]
        # importance_score: [bs, t-steps, 1]
        if torch.min(mel) < 0:
            x = mel.exp()
        score_mask = torch.mean(x, dim=-1, keepdim=True)
        score_mask = score_mask < (torch.min(score_mask) + 1e-6)

        guide_loss_final = self.zero_loss_like(mel)
        activeness_final = self.zero_loss_like(mel)

        for id in range(importance_score.size(0)):
            guide_loss = torch.mean(importance_score[id][score_mask[id]])
            if torch.isnan(guide_loss).item():
                continue
            
            if guide_loss > (1-self.dimension_reduction_rate) * 0.5:
                guide_loss_final = (
                    guide_loss_final + guide_loss / importance_score.size(0)
                )

            activeness = torch.std(importance_score[id][~score_mask[id]])
            if torch.isnan(activeness).item():
                continue
            
            activeness_final = (
                activeness_final + activeness / importance_score.size(0)
            )

        return guide_loss_final, activeness_final