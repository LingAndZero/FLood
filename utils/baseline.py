import os
from utils.utils import *
from utils.dataset import *
from server.server import Server
from server.feddisco_server import FedDiscoServer
from server.fednova_server import FedNovaServer
from server.serverM import ServerM
from server.scaffold_server import SCAFFOLDServer
from server.fedmr_server import FedMRServer
from server.fednp_server import FedNPServer
from server.fedimpro_server import FedImproServer
from server.flood_server import FLoodServer

from client.client import Client
from client.fedprox_client import FedProxClient
from client.moon_client import MoonClient
from client.scaffold import SCAFFOLDClient
from client.feddisco_client import FedDiscoClient
from client.fednova_client import FedNovaClient
from client.feddecorr_client import FedDecorrClient
from client.fednp_client import FedNPClient
from client.fedimpro_client import FedImproClient
from client.flood_client import FLoodClient


def get_server_and_clients(args):
    checkpoint_logdir = os.path.join(args.modeldir, args.algorithm, args.dataset, args.partition, str(args.drichlet_beta))
    mkdirs(checkpoint_logdir)

    print("Split and load the training dataset")
    dataset_train, dataset_test = load_dataset(args)
    dict_users = niid_distribution(dataset_train, args)

    if args.algorithm == "fedavg":
        print("----FedAvg----\n")
        server = Server(args, dataset_test, checkpoint_logdir)
        clients_pool = [Client(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fedprox":
        print("----FedProx----\n")
        server = Server(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedProxClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fednova":
        print("----FedNova----\n")
        server = FedNovaServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedNovaClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "moon":
        print("----MOON----\n")
        server = FLoodServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [MoonClient(args, dataset_train, dict_users[i], server.model_g) for i in range(args.num_users)]

    elif args.algorithm == "scaffold":
        print("----SCAFFOLD----\n")
        server = SCAFFOLDServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [SCAFFOLDClient(args, dataset_train, dict_users[i], server.global_variate) for i in range(args.num_users)]

    elif args.algorithm == "feddisco":
        print("----FedDisco----\n")
        server = FedDiscoServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedDiscoClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fedavgM":
        print("----FedAvgM----\n")
        server = ServerM(args, dataset_test, checkpoint_logdir)
        clients_pool = [Client(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "feddecorr":
        print("----FedDecorr----\n")
        server = Server(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedDecorrClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fednp":
        print("----FedNP----\n")
        server = FedNPServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedNPClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fedimpro":
        print("----FedImpro----\n")
        server = FedImproServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [FedImproClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "fedmr":
        print("----FedMR----\n")
        server = FedMRServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [Client(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    elif args.algorithm == "flood":
        print("----FLood----\n")
        server = FLoodServer(args, dataset_test, checkpoint_logdir)
        clients_pool = [FLoodClient(args, dataset_train, dict_users[i]) for i in range(args.num_users)]

    return server, clients_pool
