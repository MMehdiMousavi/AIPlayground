import numpy as np
import torch


class NNVal:
    def __init__(self):
        self.value = 0.0
        self.count = 0.0
        self.eps = 1e-5

    def add(self, loss):
        self.value += loss
        self.count += 1

    @property
    def average(self):
        return self.value / max(self.count, self.eps)

    def reset(self):
        self.value = 0.0
        self.count = 0.0

    def accumulate(self, other):
        self.value += other.value
        self.count += other.count


class Prf1a:
    def __init__(self):
        super().__init__()
        self.eps = 1e-3
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def add(self, tn=0, fp=0, fn=0, tp=0):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def add_tensor(self, y_pred_tensor, y_true_tensor):

        y_true = y_true_tensor.clone().int().view(1, -1).squeeze()
        y_pred = y_pred_tensor.clone().int().view(1, -1).squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += torch.sum(y_cases == 3).item()
        self.fp += torch.sum(y_cases == 1).item()
        self.tn += torch.sum(y_cases == 0).item()
        self.fn += torch.sum(y_cases == 2).item()

    def add_array(self, arr_2d=None, truth=None):
        x = arr_2d.copy()
        y = truth.copy()
        x[x == 255] = 1
        y[y == 255] = 1
        xy = x + (y * 2)
        self.tp += xy[xy == 3].shape[0]
        self.fp += xy[xy == 1].shape[0]
        self.tn += xy[xy == 0].shape[0]
        self.fn += xy[xy == 2].shape[0]

    def accumulate(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    @property
    def precision(self):
        try:
            p = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            p = 0
        return round(p, 5) + self.eps

    @property
    def recall(self):
        try:
            r = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            r = 0
        return round(r, 5) + self.eps

    @property
    def accuracy(self):
        try:
            a = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        except ZeroDivisionError:
            a = 0
        return round(a, 5) + self.eps

    @property
    def f1(self):
        return round(self.f_beta(beta=1), 4)

    def f_beta(self, beta=1):
        try:
            f_beta = (1 + beta ** 2) * self.precision * self.recall / (((beta ** 2) * self.precision) + self.recall)
        except ZeroDivisionError:
            f_beta = 0
        return round(f_beta, 5) + self.eps

    def prfa(self, beta=1):
        return [self.precision, self.recall, self.f_beta(beta=beta), self.accuracy]

    @property
    def overlap(self):
        try:
            o = self.tp / (self.tp + self.fp + self.fn)
        except ZeroDivisionError:
            o = 0
        return round(o, 5) + self.eps


class ConfusionMatrix:
    """
    x-axis is predicted. y-axis is true lable.
    F1 score from average precision and recall is calculated
    """

    def __init__(self, num_classes=None, device='cpu'):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes).float()
        self.eps = 10e-5
        self.device = device

    def reset(self):
        self.matrix = torch.zeros(self.num_classes, self.num_classes).float()
        return self

    def accumulate(self, other):
        self.matrix += other.matrix
        return self

    def add(self, y_true, y_pred):
        y_true = y_true.clone().long().reshape(1, -1).squeeze()
        y_pred = y_pred.clone().long().reshape(1, -1).squeeze()
        self.matrix += torch.sparse.LongTensor(
            torch.stack([y_true, y_pred]).to(self.device),
            torch.ones_like(y_true).long().to(self.device),
            torch.Size([self.num_classes, self.num_classes])).to_dense().to(self.device)

    def precision(self, average=True):
        precision = [0] * self.num_classes
        for i in range(self.num_classes):
            try:
                precision[i] = self.matrix[i, i] / (torch.sum(self.matrix[i, :]) + self.eps)
            except ZeroDivisionError:
                precision[i] = 0.0

        precision = np.array(precision) + self.eps
        return round(precision.sum() / self.num_classes if average else precision, 4)

    def recall(self, average=True):
        recall = [0] * self.num_classes
        for i in range(self.num_classes):
            try:
                recall[i] = self.matrix[i, i] / (torch.sum(self.matrix[:, i]) + self.eps)
            except ZeroDivisionError:
                recall[i] = 0.0
        recall = np.array(recall) + self.eps
        return round(recall.sum() / self.num_classes if average else recall, 4)

    def f1(self, average=True):
        f_1 = []
        precision = [self.precision(average)] if average else self.precision(average)
        recall = [self.recall(average)] if average else self.recall(average)
        for p, r in zip(precision, recall):
            try:
                f_1.append(2 * p * r / (p + r))
            except ZeroDivisionError:
                f_1.append(0)
        return round(f_1[0] if average else f_1, 4)

    def accuracy(self):
        return round(self.matrix.trace().item() / (self.matrix.sum().item() + self.eps), 4)

    def prf1a(self):
        return [self.precision(True), self.recall(True), self.f1(True), self.accuracy()]
