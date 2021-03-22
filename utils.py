import torch
import numpy as np
import json


SEEDS = [110, 221, 332, 443, 554]


class Meter:
    """
     This class is used to keep track of the metrics in the train and dev loops.
    """
    def __init__(self, mlb_encoder=None, mt_labels=None, k=6):
        """
        :param target_classes: The classes for whom the metrics will be calculated.
        """
        self.loss = 0
        self.f1k = 0
        self.f1k_mt = 0
        self.f1k_domain = 0

        self.it = 0

        self.mlb_encoder = mlb_encoder
        self.mt_labels = mt_labels
        self.k = k

    def f1k_scores(self, y_true, probs, eps=1e-10):
        true_labels = [torch.nonzero(labels, as_tuple=True)[0] for labels in y_true]
        pred_labels = torch.sort(probs, descending=True)[1][:, :self.k]

        pk_scores = [np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps for true, pred in
                     zip(true_labels, pred_labels)]
        rk_scores = [np.intersect1d(true, pred).shape[0] / true.shape[0] + eps for true, pred in
                     zip(true_labels, pred_labels)]
        f1k_scores = [2 * recall * precision / (recall + precision) for recall, precision in zip(pk_scores, rk_scores)]

        return sum(f1k_scores) / len(f1k_scores)

    def f1k_mt_scores(self, y_true, probs, eps=1e-10):
        true_labels = self.mlb_encoder.inverse_transform(y_true)
        pred_indexs = torch.sort(probs, descending=True)[1][:, :self.k]

        pred_labels_array = np.zeros_like(y_true)
        for i in range(y_true.shape[0]):
            pred_labels_array[i][pred_indexs[i]] = 1
        pred_labels = self.mlb_encoder.inverse_transform(pred_labels_array)

        true_labels_mt = []
        true_labels_domain = []
        for labels in true_labels:
            true_labels_mt.append(np.unique([self.mt_labels[str(label)] for label in labels]).astype(np.int32))
            true_labels_domain.append(np.unique([self.mt_labels[str(label)][:2] for label in labels]).astype(np.int32))

        pred_labels_mt = []
        pred_labels_domain = []
        for labels in pred_labels:
            pred_labels_mt.append(np.unique([self.mt_labels[str(label)] for label in labels]).astype(np.int32))
            pred_labels_domain.append(np.unique([self.mt_labels[str(label)][:2] for label in labels]).astype(np.int32))

        pk_mt_scores = [np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps for true, pred in
                        zip(true_labels_mt, pred_labels_mt)]
        rk_mt_scores = [np.intersect1d(true, pred).shape[0] / true.shape[0] + eps for true, pred in
                        zip(true_labels_mt, pred_labels_mt)]
        f1k_mt_scores = [2 * recall * precision / (recall + precision) for recall, precision in zip(pk_mt_scores, rk_mt_scores)]

        pk_domain_scores = [np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps for true, pred in
                            zip(true_labels_domain, pred_labels_domain)]
        rk_domain_scores = [np.intersect1d(true, pred).shape[0] / true.shape[0] + eps for true, pred in
                            zip(true_labels_domain, pred_labels_domain)]
        f1k_domain_scores = [2 * recall * precision / (recall + precision) for recall, precision in
                             zip(pk_domain_scores, rk_domain_scores)]

        return sum(f1k_mt_scores) / len(f1k_mt_scores), sum(f1k_domain_scores) / len(f1k_domain_scores)

    def update_params(self, loss, logits, y_true):
        f1k = self.f1k_scores(y_true, torch.sigmoid(logits))

        self.f1k = (self.f1k * self.it + f1k) / (self.it + 1)
        self.loss = (self.loss * self.it + loss) / (self.it + 1)

        self.it += 1

        return self.loss, self.f1k, self.f1k_mt, self.f1k_domain

    def update_params_eval(self, logits, y_true):
        f1k = self.f1k_scores(y_true, torch.sigmoid(logits))
        f1k_mt, f1k_domain = self.f1k_mt_scores(y_true, torch.sigmoid(logits))

        self.f1k = (self.f1k * self.it + f1k) / (self.it + 1)
        self.f1k_mt = (self.f1k_mt * self.it + f1k_mt) / (self.it + 1)
        self.f1k_domain = (self.f1k_domain * self.it + f1k_domain) / (self.it + 1)

        self.it += 1

        return self.f1k

    def reset(self):
        """
        Resets the metrics to the 0 values. Must be used after each epoch.
        """
        self.loss = 0

        self.prec = 0
        self.recall = 0
        self.f1 = 0

        self.it = 0