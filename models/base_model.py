import torch.nn as nn

from models.layer_module import *


class STGCN(nn.Module):
    """Build the base model.
    n_his: int, size of historical records for training.
    ks: int, kernel size of spatial convolution.
    kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of StConv blocks.
    return: tensor, [batch_size, 1, n_his, n_route].
    """

    def __init__(self, args, blocks, graph_kernel):
        super(STGCN, self).__init__()
        n_his, ks, kt, dilation, keep_prob = args.n_his, args.ks, args.kt, args.dilation, args.keep_prob
        ko = n_his
        self.st_block = nn.ModuleList()
        # ST-Block
        for channels in blocks:
            self.st_block.append(StConvBlock(ks, kt, channels, graph_kernel, dilation, keep_prob))
            ko -= 2*(kt - 1)*dilation
            # ko>0: kernel size of temporal convolution in the output layer.
        if ko > 1:
            self.out_layer = OutputLayer(blocks[-1][-1], ko, 1)
        else:
            raise ValueError(f'ERROR: kernel size ko must be larger than 1, but recieved {ko}')

    def forward(self, x):
        # x: (batch_size, c_in, time_step, n_route)
        for i in range(len(self.st_block)):
            x = self.st_block[i](x)
        if self.out_layer:
            x = self.out_layer(x)
        return x
