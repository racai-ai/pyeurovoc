import os
from run_jex import SEEDS
import numpy as np
import xml.etree.ElementTree as ET
import json


jex_dir = "eurovoc"


def parse_true_labels(directory):
    dict_labels = {}

    with open(os.path.join(jex_dir, directory, "workspace", "cf", "acquis.cf"), "r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split(" ")

            file.readline()

            labels = [int(token) for token in tokens[:-2]]
            doc_id = tokens[-1][:-1]

            dict_labels[doc_id] = labels

    with open(os.path.join(jex_dir, directory, "workspace", "cf", "opoce.cf"), "r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split(" ")

            file.readline()

            labels = [int(token) for token in tokens[:-2]]
            doc_id = tokens[-1][:-1]

            dict_labels[doc_id] = labels

    return dict_labels


def parse_pred_labels(directory, i):
    dict_labels = {}

    tree = ET.parse(os.path.join(jex_dir, directory, "workspace", "results", "Assign_Result_{}.xml".format(i)))
    root = tree.getroot()

    for document in root:
        doc_id = document.attrib["id"].split()[-1]
        dict_labels[doc_id] = []

        for category in document:

            dict_labels[doc_id].append(int(category.attrib["code"]))

    return dict_labels


def evaluate_f1_id(dict_true_labels, dict_pred_labels, eps=1e-6, k=6):
    total_f1_id = 0

    for doc_id, pred in dict_pred_labels.items():
        true = np.asarray(dict_true_labels[doc_id], dtype=np.float32())
        pred = np.asarray(pred, dtype=np.float32())[:k]

        if pred.shape[0] == 0:
            f1 = 0
        else:
            precision = np.intersect1d(true, pred).shape[0] / pred.shape[0]
            recall = np.intersect1d(true, pred).shape[0] / true.shape[0]

            if precision == 0 or recall == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)

        total_f1_id += f1

    return total_f1_id / len(dict_pred_labels) * 100


def evaluate_f1_mt(dict_true_labels, dict_pred_labels, mt_labels, eps=1e-6, k=5):
    total_f1_mt = 0

    for doc_id, pred in dict_pred_labels.items():
        true_mt_labels = np.unique([mt_labels.get(str(label), 0) for label in dict_true_labels[doc_id]]).astype(np.int32)
        pred_mt_labels = np.unique([mt_labels.get(str(label), 0) for label in pred]).astype(np.int32)[:k]

        true_mt_labels = true_mt_labels[true_mt_labels != 0]
        pred_mt_labels = pred_mt_labels[pred_mt_labels != 0]

        true = np.asarray(true_mt_labels, dtype=np.float32())
        pred = np.asarray(pred_mt_labels, dtype=np.float32())

        if pred.shape[0] == 0:
            f1 = 0
        else:
            precision = np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps
            recall = np.intersect1d(true, pred).shape[0] / true.shape[0] + eps
            f1 = 2 * recall * precision / (recall + precision)

        total_f1_mt += f1

    return total_f1_mt / len(dict_pred_labels) * 100


def evaluate_f1_do(dict_true_labels, dict_pred_labels, mt_labels, eps=1e-6, k=4):
    total_f1_do = 0

    for doc_id, pred in dict_pred_labels.items():
        true_do_labels = np.unique([mt_labels.get(str(label), "00")[:2] for label in dict_true_labels[doc_id]]).astype(
            np.int32)[:k]
        pred_do_labels = np.unique([mt_labels.get(str(label), "00")[:2] for label in pred]).astype(np.int32)[:k]

        true_do_labels = true_do_labels[true_do_labels != 0]
        pred_do_labels = pred_do_labels[pred_do_labels != 0]

        true = np.asarray(true_do_labels, dtype=np.float32())
        pred = np.asarray(pred_do_labels, dtype=np.float32())

        if pred.shape[0] == 0:
            f1 = 0
        else:
            precision = np.intersect1d(true, pred).shape[0] / pred.shape[0] + eps
            recall = np.intersect1d(true, pred).shape[0] / true.shape[0] + eps
            f1 = 2 * recall * precision / (recall + precision)

        total_f1_do += f1

    return total_f1_do / len(dict_pred_labels) * 100


if __name__ == '__main__':
    with open("mt_labels.json", "rb") as file:
        mt_labels = json.load(file)

    for directory in os.listdir(jex_dir):
        true_labels = parse_true_labels(directory)

        list_f1_id = []
        list_f1_mt = []
        list_f1_do = []
        for i in range(len(SEEDS)):
            pred_labels = parse_pred_labels(directory, i)

            list_f1_id.append(evaluate_f1_id(true_labels, pred_labels))
            list_f1_mt.append(evaluate_f1_mt(true_labels, pred_labels, mt_labels))
            list_f1_do.append(evaluate_f1_do(true_labels, pred_labels, mt_labels))

        print("Lang {} - F1@6_ID: {:.2f}, F1@5_MT: {:.2f}, F1@4_DO: {:.2f}".format(directory[:2],
                                                                                   sum(list_f1_id) / len(list_f1_id),
                                                                                   sum(list_f1_mt) / len(list_f1_mt),
                                                                                   sum(list_f1_do) / len(list_f1_do)))