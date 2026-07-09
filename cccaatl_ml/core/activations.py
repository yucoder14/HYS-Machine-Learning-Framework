import numpy as np
from tensor import Tensor, Function, _track

class Activation:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        # Activations are stateless -- no learnable parameters.
        return []

    def backward(self, grad: Tensor) -> Tensor:
        # Stub -- automatic differentiation arrives in Module 06.
        raise NotImplementedError("backward() is implemented in Module 06 (Autograd)")

class ReLU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data)
        return Tensor(result)

class Sigmoid(Activation):
    def forward(self, x: Tensor) -> Tensor:
        z = np.clip(x.data, -500, 500)  # belt-and-suspenders overflow guard
        result_data = np.zeros_like(z, dtype=np.float64)

        # Branch 1 -- non-negative inputs: 1 / (1 + exp(-x)) is safe here
        # because exp(-x) <= 1, never overflows.
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Branch 2 -- negative inputs: exp(x) / (1 + exp(x)) is safe here
        # because exp(x) < 1 when x < 0, never overflows.
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

class Tanh(Activation):
    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)

class GELU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid()(x * 1.702) * x

class Softmax(Activation):
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        # Step 1: find the max along `dim`, keep the dimension so it
        # still broadcasts cleanly against the original tensor.
        x_max_data = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data)

        # Step 2: shift every value down by that max. Largest value in
        # each row is now exactly 0; everything else is <= 0.
        x_shifted = x - x_max

        # Step 3: exponentiate the shifted values. Since nothing is
        # above 0, exp(...) is always in (0, 1] -- no overflow risk.
        exp_values = Tensor(np.exp(x_shifted.data))

        # Step 4: sum the exponentials along `dim` (this is the
        # denominator -- the normalizing constant for the distribution).
        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data)

        # Step 5: divide each exponential by the row sum so everything
        # along `dim` sums to exactly 1.
        result = exp_values / exp_sum
        return result


class LogSoftmaxBackward(Function):
    def backward(self, grad):
        # y = log_softmax(x); dx = g - softmax(x) * sum(g along dim)
        grad = np.asarray(grad)
        sm = np.exp(self.output)  # softmax = exp(log_softmax)
        return [grad - sm * np.sum(grad, axis=self.dim, keepdims=True)]


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    # log-sum-exp trick: subtract the max before exp so nothing overflows.
    # log_softmax(x) = (x - m) - log(sum(exp(x - m)))
    x_max = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    out = Tensor(shifted - log_sum_exp)
    return _track(out, LogSoftmaxBackward, (x,), output=out.data, dim=dim)