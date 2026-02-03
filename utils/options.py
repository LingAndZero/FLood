import argparse
from utils.dataset import get_num_classes


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument('--seed', type=int, default=2, help="random seed")
    parser.add_argument('--device', type=str, default='cuda:0', help='the device to run the program')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./checkpoint/", help='model directory path')
    parser.add_argument('--datadir', type=str, required=False, default="./dataset/", help="dataset directory path")
    parser.add_argument('--wandb', type=bool, required=False, default=False, help='use wandb or not')
    # benchmark
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for training')
    parser.add_argument('--model', type=str, default='resnet', help='model for training')
    parser.add_argument('--algorithm', type=str, default='flood', help='algorithm for training')

    # fl setting
    parser.add_argument('--rounds', type=int, default=2000, help='number of training rounds')
    parser.add_argument('--num_users', type=int, default=100, help='number of users or clients')
    parser.add_argument('--fraction', type=float, default=0.1, help='proportion of participating users per round')
    parser.add_argument('--partition', type=str, default='drichlet', help='the data partitioning strategy (iid, drichlet, pathological)')
    parser.add_argument('--drichlet_beta', type=float, default=0.1, help='parameter of drichlet distribution')
    parser.add_argument('--shard_per_user', type=int, default=3, help='shard of pathological distribution')

    # training setting
    parser.add_argument('--local_bs', type=int, default=50, help='batch size for training')
    parser.add_argument('--bs', type=int, default=128, help='batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--local_ep', type=int, default=5, help='number of local epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # parser.add_argument('--lr_decay_epoch', type=int, default=1, help='learning rate decay epoch')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.998, help='learning rate decay gamma')

    # for specific algorithm
    parser.add_argument('--mu', type=float, default=0.01, help='mu for fedprox')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for moon')
    parser.add_argument('--disco_a', type=float, default=0.5, help='a for feddisco')
    parser.add_argument('--disco_b', type=float, default=0.1, help='b for feddisco')
    parser.add_argument('--beta', type=float, default=0.1, help='beta for fedavgM')

    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for flood')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold for flood')
    parser.add_argument('--a', type=int, default=50, help='a for flood')
    parser.add_argument('--T', type=int, default=1000, help='T for flood')

    args = parser.parse_args()
    args.num_classes = get_num_classes(args.dataset)

    return args