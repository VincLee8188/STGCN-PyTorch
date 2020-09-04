import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GConv(nn.Module):
    """ Spectral-based graph convolution function.
    x: tensor, [batch_size, c_in, time_step, n_route].
    theta: tensor, [ks*c_in, c_out], trainable kernel parameters.
    ks: int, kernel size of graph convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, c_out, time_step, n_route].
    """

    #

    def __init__(self, ks, c_in, c_out, graph_kernel):
        super(GConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.graph_kernel = graph_kernel
        self.theta = nn.Linear(ks * c_in, c_out).to(device)

    def forward(self, x):
        # graph kernel: tensor, [n_route, ks*n_route]
        kernel = self.graph_kernel
        # time_step, n_route
        _, _, t, n = x.shape
        # x:[batch_size, c_in, time_step, n_route] -> [batch_size, time_step, c_in, n_route]
        x_tmp = x.transpose(1, 2).contiguous()
        # x_ker = x_tmp * ker -> [batch_size, time_step, c_in, ks*n_route]
        x_ker = torch.matmul(x_tmp, kernel)
        # -> [batch_size, time_step, c_in*ks, n_route] -> [batch_size, time_step, n_route, c_in*ks]
        x_ker = x_ker.reshape(-1, t, self.c_in * self.ks, n).transpose(2, 3)
        # -> [batch_size, time_step, n_route, c_out]
        x_fig = self.theta(x_ker)
        # -> [batch_size, c_out, time_step, n_route]
        return x_fig.permute(0, 3, 1, 2).contiguous()


class TemporalConvLayer(nn.Module):
    """ Temporal convolution layer.
    x: tensor, [batch_size, c_in, time_step, n_route].
    kt: int, kernel size of temporal convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    act_func: str, activation function.
    return: tensor, [batch_size, c_out, time_step-dilation*(kt-1), n_route].
    """

    def __init__(self, kt, c_in, c_out, dilation, act_func='relu'):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.c_out = c_out
        self.c_in = c_in
        self.act_func = act_func
        self.dilation = dilation
        self.conv2d = nn.Conv2d(c_in, c_out, (kt, 1), dilation=(dilation, 1)).to(device)
        self.glu = nn.Conv2d(c_in, 2 * c_out, (kt, 1), dilation=(dilation, 1)).to(device)
        self.modify = nn.Conv2d(c_in, c_out, 1).to(device)

    def forward(self, x):
        b, _, t, n = x.shape
        if self.c_in > self.c_out:
            x_input = self.modify(x)
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat((x, torch.zeros(b, self.c_out - self.c_in, t, n, device=device)), 1)
        else:
            x_input = x

        # keep the original input for residual connection.
        x_input = x_input[:, :, self.dilation * (self.kt - 1):, :]

        if self.act_func == 'GLU':
            # gated liner unit
            x_conv = self.glu(x)
            return (x_conv[:, 0:self.c_out, :, :] + x_input) * torch.sigmoid(x_conv[:, -self.c_out:, :, :])
        else:
            x_conv = self.conv2d(x)
            if self.act_func == 'linear':
                return x_conv
            elif self.act_func == 'sigmoid':
                return torch.sigmoid(x_conv)
            elif self.act_func == 'relu':
                return F.relu(x_conv + x_input)
            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')


class SpatioConvLayer(nn.Module):
    """Spatial graph convolution layer.
    x: tensor, [batch_size, c_in, time_step, n_route].
    ks: int, kernel size of spatial convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, c_out, time_step, n_route].
    """

    def __init__(self, ks, c_in, c_out, graph_kernel):
        super(SpatioConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gconv = GConv(ks, c_in, c_out, graph_kernel)
        self.modify = nn.Conv2d(c_in, c_out, 1).to(device)

    def forward(self, x):
        _, _, t, n, = x.shape
        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = self.modify(x)
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat((x, torch.zeros((x.shape[0], self.c_out - self.c_in, t, n), device=device)), 1)
        else:
            x_input = x

        x_gconv = self.gconv(x)
        return F.relu(x_gconv + x_input)


class StConvBlock(nn.Module):
    """Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    x: tensor, [batch_size, c_in, time_step, n_route].
    ks: int, kernel size of spatial convolution.
    kt: int, kernel size of temporal convolution.
    channels: list, channel configs of a single st_conv block.
    scope: str, variable scope.
    keep_prob: hyper parameter, prob of dropout.
    act_func: str, activation function.
    return: tensor, [batch_size,  c_out, time_step-, n_route]
    """

    def __init__(self, ks, kt, channels, graph_kernel, dilation, keep_prob, act_func='GLU'):
        super(StConvBlock, self).__init__()
        c_si, c_t, c_oo = channels
        self.keep_prob = keep_prob
        self.layer = nn.Sequential(
            TemporalConvLayer(kt, c_si, c_t, dilation, act_func),
            SpatioConvLayer(ks, c_t, c_t, graph_kernel),
            TemporalConvLayer(kt, c_t, c_oo, dilation),
            nn.BatchNorm2d(c_oo).to(device)
        )

    def forward(self, x):
        return F.dropout(self.layer(x), self.keep_prob)


class OutputLayer(nn.Module):
    def __init__(self, c_in, kt, dilation):
        super(OutputLayer, self).__init__()
        self.layer = nn.Sequential(
            TemporalConvLayer(kt, c_in, c_in, dilation, act_func='GLU'),
            nn.BatchNorm2d(c_in).to(device),
            TemporalConvLayer(1, c_in, c_in, dilation, act_func='sigmoid'),
            nn.Conv2d(c_in, 1, 1).to(device)
        )

    def forward(self, x):
        # x: tensor, shape is (batch_size, c_in, time_step, n_route)
        # Returns: shape is (batch_size, 1, 1, n_route)

        return self.layer(x)
