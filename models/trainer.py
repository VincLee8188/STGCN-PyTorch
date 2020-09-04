import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.math_utils import mape, mae, mse


def train_model(model, dataloaders, dataset_sizes, args, device):
    """Train the base model while doing validation for parameters choosing.
    :param model:
    :param dataloaders:  dict, include train dataset and validation dataset
    :param dataset_sizes:  dict, size of train dataset and validation dataset.
    :param args: instance of class argparse, args for training.
    :param device: cpu or cuda
    :param patience: int,
    :return:
    """

    # blocks: list, channel configs of st_conv blocks.
    # graph_kernel: tensor, [n_route, ks * n_route].

    batch_size, epoch, lr, opt, n_route = args.batch_size, args.epoch, args.lr, args.opt, args.n_route
    n_his, n_pred, patience = args.n_his, args.n_pred, args.patience
    seq_loaders = dataloaders
    seq_sizes = dataset_sizes
    train_loss = []
    val_loss = []
    loss_func = nn.MSELoss()
    if opt == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif opt == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    writer = SummaryWriter('./tensorboard')

    start_time = time.time()
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    wait = 0

    for i in range(epoch):
        if wait >= patience:
            print('Early stop of training!')
            break
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for j, (x, y_tar) in enumerate(seq_loaders[phase]):
                # x -> (batch_size, c_in, time_step, n_route)
                x, y_tar = x.permute(0, 3, 1, 2), y_tar.permute(0, 3, 1, 2)
                optimizer.zero_grad()

                if phase == 'train':
                    y_pre = model(x)
                    loss = loss_func(y_pre, y_tar[:, :, 0, :])
                    v_ = y_pre.clone().detach()
                    v = y_tar[:, :, 0, :].clone().detach()
                else:
                    b = x.shape[0]
                    v = y_tar
                    v_ = torch.zeros(b, 1, n_pred, n_route, device=device)
                    for step in range(n_pred):
                        y_pre = model(x)
                        x[:, :, 0: n_his - 1, :] = x.clone()[:, :, 1: n_his, :]
                        x[:, :, n_his - 1: n_his, :] = y_pre
                        v_[:, :, step: step + 1, :] = y_pre

                mae1 = mae(v, v_)
                mape1 = mape(v, v_)
                mse1 = mse(v, v_)
                rmse1 = torch.sqrt(mse1)
                print('.', end='')
                if (j+1) % 10 == 0:  # every 10 batches to display information
                    print(f'Phase[{phase}], Epoch {i + 1:2d}, Step {j + 1:3d}:',
                          f'mse<{mse1:.3f}> mae<{mae1:.3f}> mape<{mape1:.3f}> rmse<{rmse1:.3f}>')

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()

            epoch_loss = running_loss / (int(seq_sizes[phase] / args.batch_size) + 1)
            if phase == 'train':
                scheduler.step()
                writer.add_scalar('Loss/train', epoch_loss, i + 1)
                train_loss.append(epoch_loss)
            else:
                writer.add_scalar('Loss/validation', epoch_loss, i + 1)
                val_loss.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    wait = 0
                else:
                    wait += 1
            print()
    # writer.close()
    model.load_state_dict(best_model_wts)
    torch.save(model, './output/model_01.pkl')
    time_elapsed = time.time() - start_time

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:.4f}'.format(best_loss))
    print('Training model finished!')
