"""
Transformer window-rize tools
"""

def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(-1).permute(1,0)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x


def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2) # window_size*window_size, num_windows*B, C
    return windows


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x

