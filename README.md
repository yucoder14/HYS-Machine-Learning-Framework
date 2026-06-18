# HYS-Machine-Learning-Framework

# TODO 

Week 1
------

- [ ] Have a working forward pass (CPU)
- [ ] Person A — Tensor: construction, arithmetic, reshape (ships first — unblocks B and C fastest)
      - [ ] __init__ wrapping a NumPy array; set data, shape, size, dtype
      - [ ] Arithmetic dunders: __add__, __sub__, __mul__, __truediv__ (handle Tensor+Tensor and Tensor+scalar)
      - [ ] reshape() with -1 dimension inference + element-count validation
      - [ ] transpose() (default: swap last two dims)
      - [ ] Tests: broadcasting compatibility, scalar ops, reshape size-mismatch errors

- [ ] Person B — Tensor: matmul, reductions + demo + tooling
      - [ ] matmul() with inner-dimension shape validation (a.shape[-1] == b.shape[-2])
      - [ ] Reductions: sum(), mean(), max() with axis / keepdims
      - [ ] Light tooling: repo scaffolding (core/tensor.py, core/activations.py, core/layers.py), requirements.txt, mb could set up a CI to run the test suite
      - [ ] End-to-end demo: assemble the Sequential MLP, run a forward pass on MNIST-shaped dummy input, verify output shape (batch, 10) and that Softmax rows sum to 1
      - [ ] Tests: matmul shape errors, reduction axis semantics

- [ ] Person C — Activations (depends on Tensor arithmetic + max/sum)
      - [ ] Activation base pattern: forward(), __call__(), backward() stub
      - [ ] ReLU (np.maximum(0, x))
      - [ ] Sigmoid (numerically stable piecewise formula + clip at ±500)
      - [ ] Tanh (np.tanh)
      - [ ] GELU (fast approximation, reuses Sigmoid: sigmoid(1.702 * x) * x)
      - [ ] Softmax (subtract per-row max for stability, dim argument)
      - [ ] Tests: output ranges, Softmax sums to 1, no NaN on large logits (e.g. 100)

- [ ] Person D — Layers (depends on Tensor matmul + __add__)
      - [ ] Layer base class: forward(), __call__(), parameters()
      - [ ] Linear: LeCun init sqrt(1/in_features), weight (in, out) + bias (out,), forward = xW + b
      - [ ] Dropout: training/inference modes, inverted scaling 1/(1-p)
      - [ ] Sequential: chain forward, flatten per-layer parameters
      - [ ] Tests: parameter counts, shape composition across stacked layers, dropout passthrough at inference


- [ ] Get familiar with Numba framework (or jump straight to CUDA) (GPU)
    - [ ] How do you do a matrix multiplication 
    - [ ] etc

Week 2
------

- [ ] Have a working backward pass (CPU)
- [ ] visualizing the computation graph(stretch)
- [ ] Figure out `.to(device)`
- [ ] Have a working forward pass (GPU)  


Week 3
------

- [ ] Optimizer? (CPU)
- [ ] Convolution
- [ ] Have a working backward pass (GPU)
- [ ] Integrate CPU and GPU together 

Week 4
------

- [ ] Iterate 
