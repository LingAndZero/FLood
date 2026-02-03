import copy
import math
import numpy as np
from tqdm import tqdm
from utils.options import get_args
from utils.utils import *
from utils.baseline import *


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)
    fix_random_seed(args.seed)

    logger = Logger(args)
    # logger.info("#" * 100)

    server, clients_pool = get_server_and_clients(args)

    weight = 1.0
    sigma = lambda x: 1 / (1 + np.exp(-x))

    print("\nTraining begins!")
    for round in range(args.rounds):
        # logger.info("in communication round:" + str(round))
        print("In communication round:" + str(round))

        # random select the client
        clients = server.select_clients()
        print("select clients {} to train".format(clients))

        local_models = []
        local_n_train = []
        local_others = []

        for i, idx in enumerate(tqdm(clients)):
            # server.model_g.load_state_dict(server.w_locals[i])
            local_model, n_train, others = clients_pool[idx].train(copy.deepcopy(server.model_g), weight)
            local_models.append(local_model)
            local_n_train.append(n_train)
            local_others.append(others)

        server.aggregate(local_models, local_n_train, local_others)
        accuracy = server.test(round)
        print(f"Accuracy: {accuracy}")
        logger.info(accuracy)

        args.lr = args.lr * args.lr_decay_gamma
        if round <= args.T:
            # weight = args.a * (1 - math.cos(math.pi * ((round) / args.T)))
            # weight = 2 * args.a * round / args.T
            # weight = 2 * args.a * (round / args.T) ** 2
            # weight = 2 * args.a * (1 - np.exp(-5.0 * round)) / (1 - np.exp(-5.0 * args.T))
            weight = 2 * args.a * (sigma(10.0 * (round - args.T/2)) - sigma(-10.0 * args.T/2)) / (sigma(10.0 * args.T/2) - sigma(-10.0 * args.T/2))


    if args.partition == "drichlet":
        model_logdir = os.path.join(args.modeldir, args.algorithm, args.dataset, args.partition, str(args.drichlet_beta), args.model)
    elif args.partition == "pathological":
        model_logdir = os.path.join(args.modeldir, args.algorithm, args.dataset, args.partition, str(args.shard_per_user), args.model)

    # os.makedirs(model_logdir, exist_ok=True)
    # save_path = os.path.join(model_logdir, "final_model.pth")
    # torch.save(server.model_g.state_dict(), save_path)
    # print(f"Model saved to {save_path}")