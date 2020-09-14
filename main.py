from os.path import join as pjoin
import argparse

from models.base_model import *
from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import train_model
from models.tester import test_model
from models.layer_module import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.7)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--keep_prob', type=float, default=0.3)
parser.add_argument('--patience', type=int, default=10)


args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
ks, kt, patience = args.ks, args.kt, args.patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
wa = weight_matrix(pjoin('./dataset', f'W_{n}.csv'))


# Calculate graph kernel
la = scaled_laplacian(wa)
lk = cheb_poly_approx(la, ks, n)
graph_kernel = torch.tensor(lk).type(torch.float32)

# Data Preprocessing
data_file = pjoin('./dataset', f'V_{n}.csv')
n_train, n_val, n_test = 34, 5, 5
# data: [batch_size, seq_len, n_route, C_0].
dataloaders, dataset_sizes, test_data_gen, stats = loader_gen(data_file, n_train, n_val,
                                                              n_test, n_his, n_pred, n, args.batch_size, device)
print('>> Loading dataset with Mean: {:.2f}, STD: {:.2f}'.format(stats['mean'], stats['std']))

if __name__ == '__main__':
    model = STGCN(args, blocks, graph_kernel)
    train_model(model, dataloaders, dataset_sizes, args, device)
    print(f'the model has {count_parameters(model)} parameters!')
    test_model(test_data_gen, args, stats, device)
