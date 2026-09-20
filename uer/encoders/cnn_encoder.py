import torch
import torch.nn as nn
import torch.nn.functional as F


class Nepco(nn.Module):
    """Final Nepco encoder used in the artifact evaluation package."""

    def __init__(self, args):
        super(Nepco, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.dropout_p = getattr(args, "dropout", 0.0)

        self.project = nn.Linear(self.emb_size, self.hidden_size, bias=True)

        self.weight_h_raw = nn.ParameterList([
            nn.Parameter(torch.empty(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_h = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.weight_g_raw = nn.ParameterList([
            nn.Parameter(torch.empty(self.hidden_size, 1, self.kernel_size))
            for _ in range(self.layers_num)
        ])
        self.bias_g = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_size))
            for _ in range(self.layers_num)
        ])

        self.register_buffer("weight_hg_infer", torch.empty(0), persistent=False)
        self.register_buffer("bias_hg_infer", torch.empty(0), persistent=False)
        self._infer_cache_ready = False

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range((self.layers_num + self.block_size - 1) // self.block_size)
        ])
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        pad_total = self.kernel_size - 1
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

        for w in self.weight_h_raw:
            nn.init.xavier_uniform_(w)
        for w in self.weight_g_raw:
            nn.init.xavier_uniform_(w)
        for b in self.bias_h:
            nn.init.zeros_(b)
        for b in self.bias_g:
            nn.init.zeros_(b)

        self._invalidate_infer_cache()

    def _invalidate_infer_cache(self):
        self.weight_hg_infer = torch.empty(0, device=self.project.weight.device, dtype=self.project.weight.dtype)
        self.bias_hg_infer = torch.empty(0, device=self.project.weight.device, dtype=self.project.weight.dtype)
        self._infer_cache_ready = False

    @torch.no_grad()
    def refresh_infer_cache(self):
        weight_h = torch.stack(
            [F.softmax(w, dim=-1) for w in self.weight_h_raw], dim=0
        ).contiguous()
        weight_g = torch.stack(
            [F.softmax(w, dim=-1) for w in self.weight_g_raw], dim=0
        ).contiguous()

        bias_h = torch.stack([b for b in self.bias_h], dim=0).contiguous()
        bias_g = torch.stack([b for b in self.bias_g], dim=0).contiguous()

        self.weight_hg_infer = torch.cat([weight_h, weight_g], dim=1).contiguous()
        self.bias_hg_infer = torch.cat([bias_h, bias_g], dim=1).contiguous()
        self._infer_cache_ready = True
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._invalidate_infer_cache()
        else:
            self.refresh_infer_cache()
        return self

    def load_state_dict(self, state_dict, strict=True):
        ret = super().load_state_dict(state_dict, strict=strict)
        self._invalidate_infer_cache()
        if not self.training:
            self.refresh_infer_cache()
        return ret

    @torch.no_grad()
    def prepare_for_onnx_export(self):
        self.eval()
        return self

    def _post_block(self, out, res_input, norm_idx):
        out = out + 0.5 * res_input
        out = out.transpose(1, 2)
        out = self.layer_norms[norm_idx](out)
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous()
        return out

    def _forward_impl(self, emb, use_infer_cache: bool):
        x = self.project(emb).transpose(1, 2).contiguous()

        res_input = x
        norm_idx = 0
        out = x

        for i in range(self.layers_num):
            if use_infer_cache:
                weight_hg = self.weight_hg_infer[i]
                bias_hg = self.bias_hg_infer[i]
            else:
                w_h = F.softmax(self.weight_h_raw[i], dim=-1)
                w_g = F.softmax(self.weight_g_raw[i], dim=-1)
                weight_hg = torch.cat([w_h, w_g], dim=0).contiguous()
                bias_hg = torch.cat([self.bias_h[i], self.bias_g[i]], dim=0).contiguous()

            x_padded = F.pad(x, (self.pad_left, self.pad_right))
            hg_out = F.conv1d(
                x_padded,
                weight_hg,
                bias=bias_hg,
                groups=self.hidden_size
            )

            h_out = hg_out[:, :self.hidden_size, :]
            g_out = hg_out[:, self.hidden_size:, :]
            out = h_out * torch.sigmoid(g_out)

            if (i + 1) % self.block_size == 0:
                out = self._post_block(out, res_input, norm_idx)
                res_input = out
                norm_idx += 1

            x = out

        if (self.layers_num % self.block_size) != 0:
            out = self._post_block(out, res_input, norm_idx)

        return out.transpose(1, 2)

    def forward(self, emb, seg=None):
        use_infer_cache = (not self.training) and self._infer_cache_ready
        return self._forward_impl(emb, use_infer_cache=use_infer_cache)
