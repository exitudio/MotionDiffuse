from torch import nn
import torch
import torch.nn.functional as F
from models.transformer import zero_module
import math


# [TODO] this process is not GPU friendly
# [TODO] Does the frame is missing? Should start from length[i]+1?
def generate_src_mask(T, length):
    B = len(length)
    src_mask = torch.ones(B, T)
    for i in range(B):
        for j in range(length[i], T):
            src_mask[i, j] = 0
    return src_mask


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y)
        return y
    
class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.proj_out = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000
        weight = F.softmax(attention, dim=2)
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y)
        return y


class MotionTransformerOnly(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=196,
                 latent_dim=512,
                 num_layers=8,
                 num_heads=8,
                 output_feats=-1,
                 **kargs):
        super().__init__()

        self.sequence_embedding = nn.Parameter(
            torch.randn(num_frames, latent_dim))

        # Text Transformer

        # Input Embedding
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.temporal_decoder_blocks = nn.ModuleList([
            # LinearTemporalSelfAttention(
            TemporalSelfAttention(
                latent_dim=latent_dim,
                num_head=num_heads,
            )for i in range(num_layers)])
        # Output Module
        if output_feats != -1:
            self.out = zero_module(nn.Linear(latent_dim, output_feats))

    def forward(self, x, src_mask, length=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        for module in self.temporal_decoder_blocks:
            h = module(h, src_mask)
        if hasattr(self, 'out'):
            output = self.out(h).view(B, T, -1).contiguous()
            output = output * src_mask
            return output
        h = h * src_mask
        return h
