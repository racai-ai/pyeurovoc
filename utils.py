import torch
import numpy as np


SEEDS = [110, 221, 332, 443, 554]


def f1k_score(y_true, probs, k, eps=1e-10):
    true_labels = [torch.nonzero(labels, as_tuple=True)[0] for labels in y_true]
    pred_labels = torch.sort(probs, descending=True)[1][:, :k]

    pk_scores = [np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps for true, pred in zip(true_labels, pred_labels)]
    rk_scores = [np.intersect1d(true, pred).shape[0] / true.shape[0] + eps for true, pred in zip(true_labels, pred_labels)]
    f1k_scores = [2 * recall * precision / (recall + precision) for recall, precision in zip(pk_scores, rk_scores)]

    return sum(f1k_scores) / len(f1k_scores)


class Meter:
    """
     This class is used to keep track of the metrics in the train and dev loops.
    """
    def __init__(self):
        """
        :param target_classes: The classes for whom the metrics will be calculated.
        """
        self.loss = 0
        self.f1k = 0

        self.it = 0

    def update_params(self, loss, logits, y_true):
        f1k = f1k_score(y_true, torch.sigmoid(logits), 6)

        self.f1k = (self.f1k * self.it + f1k) / (self.it + 1)
        self.loss = (self.loss * self.it + loss) / (self.it + 1)

        self.it += 1

        return self.loss, self.f1k

    def reset(self):
        """
        Resets the metrics to the 0 values. Must be used after each epoch.
        """
        self.loss = 0

        self.prec = 0
        self.recall = 0
        self.f1 = 0

        self.it = 0