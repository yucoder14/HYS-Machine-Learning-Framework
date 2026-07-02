import numpy as np
from cccaatl_ml.core.tensor import Tensor

INIT_SCALE_FACTOR = 1
DROPOUT_MIN_PROB = 0.0
DROPOUT_MAX_PROB = 1.0

class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        pass
    
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        return []

class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None
    
    def forward(self, x):
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x, training=True):
        if not training or self.p == DROPOUT_MIN_PROB:
            return x
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data), requires_grad=x.requires_grad)
        
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)
        scale = Tensor(np.array(1.0 / keep_prob), requires_grad=False)

        output = x * mask_tensor * scale
        return output

    def parameters(self):
        return []


class Sequential(Layer):
    def __init__(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
