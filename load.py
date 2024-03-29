import os
import numpy as np
import torch
from utils import SEEDS


def load_data(data_path, lang, batch_size):
    datasets = []

    for directory in os.listdir(data_path):
        if lang == directory[:2]:
            for i in range(len(SEEDS)):
                print("Loading data from directoy {}...".format(os.path.join(data_path, directory, "split_{}".format(i))))

                train_X = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_X.npy")))
                train_mask = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_mask.npy")))
                train_y = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_y.npy"))).float()

                assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

                dev_X = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_X.npy")))
                dev_mask = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_mask.npy")))
                dev_y = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_y.npy"))).float()

                assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]

                test_X = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_X.npy")))
                test_mask = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_mask.npy")))
                test_y = torch.from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_y.npy"))).float()

                assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

                dataset = torch.utils.data.TensorDataset(train_X, train_mask, train_y)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                dataset = torch.utils.data.TensorDataset(dev_X, dev_mask, dev_y)
                dev_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                dataset = torch.utils.data.TensorDataset(test_X, test_mask, test_y)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

                datasets.append((train_loader, dev_loader, test_loader, train_y.shape[1]))

    return datasets