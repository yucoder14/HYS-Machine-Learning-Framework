import numpy as np
from cccaatl_ml.core.tensor import Tensor, Function, _track
from cccaatl_ml.core.activations import log_softmax

BCE_EPS = 1e-7


def _raw(x):
    return x.data if isinstance(x, Tensor) else np.asarray(x)


class Loss:
    def forward(self, predictions, targets):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class MSEBackward(Function):
    def backward(self, grad):
        (pred,) = self.inputs
        diff = pred.data - self.target
        # d/dpred mean((pred - t)^2) = 2 (pred - t) / N
        return [grad * (2.0 / diff.size) * diff]


class MSELoss(Loss):
    def forward(self, predictions, targets):
        t = _raw(targets)
        out = Tensor(np.mean((predictions.data - t) ** 2))
        return _track(out, MSEBackward, (predictions,), target=t)


class CrossEntropyBackward(Function):
    def backward(self, grad):
        sm = np.exp(self.log_probs)  # softmax = exp(log_softmax)
        onehot = np.zeros_like(sm)
        onehot[np.arange(sm.shape[0]), self.idx] = 1.0
        # d/dlogits of the mean NLL = (softmax - onehot) / batch
        return [grad * (sm - onehot) / sm.shape[0]]


class CrossEntropyLoss(Loss):
    # predictions are raw logits (batch, num_classes); targets are class indices
    def forward(self, predictions, targets):
        log_probs = log_softmax(predictions, dim=-1)
        idx = _raw(targets).astype(int)
        picked = log_probs.data[np.arange(log_probs.data.shape[0]), idx]
        out = Tensor(-np.mean(picked))  # negative mean log-prob of true class
        return _track(out, CrossEntropyBackward, (predictions,),
                      log_probs=log_probs.data, idx=idx)


class BCEBackward(Function):
    def backward(self, grad):
        (pred,) = self.inputs
        p = np.clip(pred.data, BCE_EPS, 1.0 - BCE_EPS)
        # d/dp mean(BCE) = (p - y) / (p (1 - p)) / N, zero where clip saturated
        d = (p - self.target) / (p * (1.0 - p))
        mask = (pred.data > BCE_EPS) & (pred.data < 1.0 - BCE_EPS)
        return [grad * d * mask / p.size]


class BinaryCrossEntropyLoss(Loss):
    # predictions are probabilities in (0, 1); targets are 0/1 labels
    def forward(self, predictions, targets):
        y = _raw(targets)
        p = np.clip(predictions.data, BCE_EPS, 1.0 - BCE_EPS)
        bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        out = Tensor(np.mean(bce))
        return _track(out, BCEBackward, (predictions,), target=y)
