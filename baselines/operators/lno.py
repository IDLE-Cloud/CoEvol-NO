import math
import torch

ACTIVATION = {
    "Sigmoid": torch.nn.Sigmoid(), "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(), "LeakyReLU": torch.nn.LeakyReLU(0.1),
    "ELU": torch.nn.ELU(), "GELU": torch.nn.GELU()
}


def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    return torch.einsum("bhij,bhjc->bhic", score, v)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.act = act
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)])
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(len(self.hidden)):
            r = r + self.act(self.hidden[i](r))
        return self.output(r)


class SelfAttention(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, attn):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.Wq = torch.nn.Linear(n_dim, n_dim)
        self.Wk = torch.nn.Linear(n_dim, n_dim)
        self.Wv = torch.nn.Linear(n_dim, n_dim)
        self.attn = attn
        self.proj = torch.nn.Linear(n_dim, n_dim)

    def forward(self, x):
        B, N, D = x.size()
        q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.proj(r)


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, act):
        super().__init__()
        self.self_attn = SelfAttention(n_mode, n_dim, n_head, Attention_Vanilla)
        self.ln1 = torch.nn.LayerNorm(n_dim)
        self.ln2 = torch.nn.LayerNorm(n_dim)
        self.drop = torch.nn.Dropout(0.0)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_dim, n_dim * 2), act, torch.nn.Linear(n_dim * 2, n_dim))

    def forward(self, y):
        y = y + self.drop(self.self_attn(self.ln1(y)))
        y = y + self.mlp(self.ln2(y))
        return y


class LNO(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr):
        super().__init__()
        self.n_dim = n_dim
        self.act = ACTIVATION[act]
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        self.y2_dim = 1 if model_attr.get("time") else y2_dim

        self.trunk_projector = MLP(x_dim, n_dim, n_dim, n_layer, self.act)
        self.branch_projector = MLP(y1_dim, n_dim, n_dim, n_layer, self.act)
        self.attention_projector = MLP(n_dim, n_dim, n_mode, n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(
            *[AttentionBlock(n_mode, n_dim, n_head, self.act) for _ in range(n_block)])

    def forward(self, y=None, x=None, t=None):
        if y is None:
            y = x
        x = self.trunk_projector(x)
        y = self.branch_projector(y)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)

        z = torch.einsum("bij,bic->bjc", score_encode, y)
        for block in self.attn_blocks:
            z = block(z)
        return torch.einsum("bij,bjc->bic", score_decode, z)
