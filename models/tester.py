import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.math_utils import mape, mae, mse
from utils.math_utils import z_inverse


def test_model(dataloader, args, stats, device):
    """Load and test saved model from the checkpoint.
    :param dataloader: instance of class Dataloader, dataloader for test.
    :param args: parameters input from command line
    :param stats: dict, mean and variance for the test dataset.
    :param device: cuda or cpu
    """

    # batch_size: int, the size of batch.
    # n_his: int, the length of historical records for training.
    # n_pred: int, the length of prediction.
    n_his, n_pred, ks, batch_size = args.n_his, args.n_pred, args.ks, args.batch_size
    n_route, epoch = args.n_route, args.epoch
    model_path = './output/model_01.pkl'

    model = torch.load(model_path)
    print(f'>> Loading saved model from {model_path} ...')

    v, v_ = [], []

    with torch.no_grad():
        for j, (x, y_tar) in enumerate(dataloader):
            x, y_tar = x.permute(0, 3, 1, 2), y_tar.permute(0, 3, 1, 2)
            # (batch_size, c_in, time_step, n_route)
            b = x.shape[0]
            step_list = torch.zeros(b, 1, n_pred, n_route, device=device)
            for step in range(n_pred):
                y_pre = model(x)
                x[:, :, 0: n_his - 1, :] = x.clone()[:, :, 1: n_his, :]
                x[:, :, n_his - 1: n_his, :] = y_pre
                step_list[:, :, step: step + 1, :] = y_pre
            v.extend(y_tar.squeeze(1).to('cpu').numpy())
            v_.extend(step_list.squeeze(1).to('cpu').numpy())

        # (batch_size, time_step, n_route)
        v = torch.from_numpy(np.array(v))
        v_ = torch.from_numpy(np.array(v_))
        mae1 = mae(v, v_)
        mape1 = mape(v, v_)
        mse1 = mse(v, v_)

        # convert to original  value
        v = z_inverse(v, stats['mean'], stats['std'])
        v_ = z_inverse(v_, stats['mean'], stats['std'])

        print(f'Preprocess {j:3d}',
              f'mae<{mae1:.3f}> mape<{mape1:.3f}> mse<{mse1:.3f}>')
    print('Testing model finished!')
