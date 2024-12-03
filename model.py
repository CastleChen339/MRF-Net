import torch
import torch.nn as nn
import math
import torch.nn.functional as f
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchsummary import summary



class ChannelAttention(nn.Module):  # Based on ECANet
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.cat([x, f.relu(self.conv(x), inplace=True)], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, out_channels, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, out_channels, kernel_size=1)
        self.ca = ChannelAttention(channel=out_channels)
        self.sa = SpatialAttention()
        self.nomalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # return x + self.sa(self.ca(self.lff(self.layers(x))))
        # print(self.sa(self.ca(self.lff(self.layers(x)))).shape)
        return self.nomalization(x + self.sa(self.ca(self.lff(self.layers(x)))))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformer(nn.Module):

    def __init__(self, dim, num_heads=4, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, input_resolution=(240, 240)):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if shift_size > 0:
            attn_mask = self.calculate_mask(input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention(kernel_size=3)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)


        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # if self.input_resolution == x_size:
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #    attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class STBlock(nn.Module):
    def __init__(self, dim, ST_num, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.ST_num = ST_num
        self.num_heads = num_heads
        self.window_size = window_size
        self.ST = nn.ModuleList(
            SwinTransformer(dim=self.dim, num_heads=self.num_heads, window_size=self.window_size) for i in
            range(self.ST_num))
        self.embed = PatchEmbed()
        self.unembed = PatchUnEmbed(embed_dim=self.dim)
        self.conv = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, x_size):
        x0 = x
        x = self.embed(x)
        for i in range(self.ST_num):
            x = self.ST[i](x, x_size)
        x = self.unembed(x, x_size)
        x = self.conv(x)
        return x + x0


class MultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list):
        super(MultiScaleLayer, self).__init__()
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2) for k in kernel_list])
        self.conv1 = nn.Conv2d(out_channels * len(kernel_list), out_channels, kernel_size=1)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(kernel_size=3)
        self.nomalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res_list = []
        for i in range(len(self.conv_list)):
            res_list.append(f.relu(self.conv_list[i](x)))
        res = torch.cat(res_list, dim=1)
        res = f.relu(self.conv1(res), inplace=True)
        return self.nomalization(self.sa(self.ca(res)))


class MRFNet(nn.Module):
    def __init__(self, STB_num=5, ST_num=8, rdb_num=5, denselayer_num=4):
        super(MRFNet, self).__init__()
        self.rdb_num = rdb_num
        self.denselayer_num = denselayer_num
        self.dim = 24
        self.STB_num = STB_num
        self.ST_num = ST_num
        self.conv1 = nn.Conv2d(1, self.dim, kernel_size=3, stride=1, padding=1)
        self.multiscale = MultiScaleLayer(in_channels=self.dim, out_channels=self.dim, kernel_list=[3, 5, 7, 9])
        self.STB_list = nn.ModuleList(
            STBlock(dim=self.dim, ST_num=self.ST_num, num_heads=4, window_size=4)
            for _ in range(self.STB_num))
        self.rdb_list = nn.ModuleList(
            [RDB(in_channels=self.dim, growth_rate=self.dim // 4, out_channels=self.dim, num_layers=self.denselayer_num)
             for _ in range(self.rdb_num)])
        self.ca = ChannelAttention(self.dim * self.rdb_num)
        self.sa = SpatialAttention(kernel_size=3)
        self.conv2 = nn.Conv2d(self.dim * self.rdb_num, self.dim * 2, kernel_size=1)
        self.conv3 = nn.Conv2d(self.dim * 2, self.dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        h1 = f.relu(self.conv1(x), inplace=True)
        h2 = self.multiscale(h1)
        for i in range(self.STB_num):
            h2 = self.STB_list[i](h2, (h2.shape[2], h2.shape[3]))
        rdb_reslist = []
        for i in range(self.rdb_num):
            h2 = self.rdb_list[i](h2)
            rdb_reslist.append(h2)
        h3 = self.sa(self.ca(torch.cat(rdb_reslist, dim=1)))
        h4 = self.conv4(self.conv3(self.conv2(h3)))
        return x + h4


if __name__ == '__main__':
    device = torch.device("cpu")
    model = MRFNet().to(device)
    # print(model)
    # inputs = torch.ones([2, 1, 256, 256]).to(device)
    # outputs = model(inputs)
    # print(outputs)
    summary(model, input_size=(1, 256, 256), device='cpu')

