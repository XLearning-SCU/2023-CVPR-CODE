import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim, squeezes, shuffle, expan_att_chans):
        super(CondensedAttentionNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        self.ch_sp_squeeze = nn.Sequential(
            nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
        )

        self.channel_attention = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.spatial_attention = SpatialAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)

        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x):
        x = self.ch_sp_squeeze(x)

        group_num = self.sque_ch_dim
        each_group = self.sque_sp_dim // self.sque_ch_dim
        idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        x = x[:, idx, :, :]

        x = self.channel_attention(x)
        nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = x[:, nidx, :, :]

        x = self.spatial_attention(x)
        x = self.sp_ch_unsqueeze(x)
        return x


class DualAdaptiveNeuralBlock(nn.Module):
    def __init__(self, embed_dim):
        super(DualAdaptiveNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        self.group_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Conv2d(embed_dim, embed_dim * 2, 7, 1, 3, groups=embed_dim)
        )
        self.post_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x0, x1 = self.group_conv(x).view(B, C, 2, H, W).chunk(2, dim=2)
        x_ = F.gelu(x0.squeeze(2)) * torch.sigmoid(x1.squeeze(2))
        x_ = self.post_conv(x_)
        return x_


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 squeezes,
                 shuffle,
                 expan_att_chans
                 ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ca = CondensedAttentionNeuralBlock(embed_dim, squeezes, shuffle, expan_att_chans)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.da = DualAdaptiveNeuralBlock(embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x_ = self.norm1(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W).contiguous()
        x = x + self.ca(x_)
        x_ = self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W).contiguous()
        x = x + self.da(x_)
        return x


class CODEFormer(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, expan_att_chans=4,
                 refine_blocks=4, num_blocks=(4, 6, 6, 2), num_shuffles=(16, 8, 4, 2),
                 ch_sp_squeeze=[(4, 8), (4, 4), (4, 2), (4, 1)]):
        super(CODEFormer, self).__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.encoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans
            ) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))])

        self.decoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans
            ) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))][::-1])

        self.downsampler = nn.ModuleList([nn.Sequential(
            nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1)), 3, 1, 1),
            nn.PixelUnshuffle(2)
        ) for i in range(len(num_blocks) - 1)]).append(nn.Identity())

        self.upsampler = nn.ModuleList([nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(embed_dim * 2 ** (i - 1)), int(embed_dim * 2 ** i), 3, 1, 1)
        ) for i in range(len(num_blocks) - 1)][::-1]).append(nn.Identity())

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                embed_dim, ch_sp_squeeze[0], num_shuffles[0], expan_att_chans
            ) for _ in range(refine_blocks)
        ])

        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        x_emb = self.patch_embed(x)

        # Encoder
        x_ = x_emb
        x_ms = []
        for layer, sampler in zip(
                self.encoder, self.downsampler
        ):
            x_ = layer(x_)
            x_ms.append(x_)
            x_ = sampler(x_)

        # Decoder
        x_ = 0
        x_ms.reverse()
        for x_e, layer, sampler in zip(
                x_ms, self.decoder, self.upsampler
        ):
            x_ = sampler(layer(x_ + x_e))

        # Refinement
        x_ = self.refinement(x_ + x_emb)
        x_ = self.conv_last(x_) + x
        return x_
