from torchvision import transforms
import numpy as np


def get_num_classes(dataset):

    if dataset in ["mnist", "cifar10", "svhn"]:
        num_classes = 10

    elif dataset == "cifar100":
        num_classes = 100
        
    else:
        print("Invalid dataset")
        raise Exception("Invalid Dataset")
    
    return num_classes


def load_dataset(args):

    if args.dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        dataset_train = CIFAR10("./data/cifar10", train=True, transform=train_transform, download=True)
        dataset_test = CIFAR10("./data/cifar10", train=False, transform=test_transform, download=True)

    elif args.dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        dataset_train = CIFAR100("./data/cifar100", train=True, transform=train_transform, download=True)
        dataset_test = CIFAR100("./data/cifar100", train=False, transform=test_transform, download=True)
    
    elif args.dataset == "svhn":
        from torchvision.datasets import SVHN
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset_train = SVHN("./data/svhn", split='train', transform=transform, download=True)
        dataset_test = SVHN("./data/svhn", split='test', transform=transform, download=True)

    else:
        print("Invalid dataset")
        raise Exception("Invalid Dataset")

    return dataset_train, dataset_test


def niid_distribution(dataset, args):
    
    if args.partition == "iid":
        num_items = int(len(dataset) / args.num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(args.num_users):
            dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
      
    elif args.partition == "drichlet":
        min_size = 0
        min_require_size = 10
        K = args.num_classes

        if hasattr(dataset, 'targets'):
            y_train = np.array(dataset.targets) 
        elif hasattr(dataset, 'labels'):
            y_train = np.array(dataset.labels)
        N = len(dataset)
        dict_users = {}

        idx_batch = None
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.num_users)]
            
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.drichlet_beta, args.num_users))
                proportions = np.array(
                    [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)
        for j in range(args.num_users):
            # np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]

    elif args.partition == "pathological":
        dict_users = {i: [] for i in range(args.num_users)}
        idxs_dict = {}

        if hasattr(dataset, 'targets'):
            y_train = np.array(dataset.targets) 
        elif hasattr(dataset, 'labels'):
            y_train = np.array(dataset.labels)

        for i in range(len(y_train)):
            label = y_train[i].item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

        num_classes = len(np.unique(y_train))
        shard_per_class = int(args.shard_per_user * args.num_users / num_classes)

        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x

        import random
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((args.num_users, -1))

        for i in range(args.num_users):
            rand_set_label = rand_set_all[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set).astype("int")

    return dict_users
    