# HYS-Machine-Learning-Framework

# TODO 

Week 1
------
- [ ] Person A — Tensor: construction, arithmetic, reshape (ships first — unblocks C and D fastest) - Catherine
  - [ ] `__init__` wrapping a NumPy array; set data, shape, size, dtype
  - [ ] Arithmetic dunders: `__add__`, `__sub__`, `__mul__`, `__truediv__` (Tensor+Tensor and Tensor+scalar)
  - [ ] `reshape()` with -1 dimension inference + element-count validation
  - [ ] `transpose()` (default: swap last two dims)
  - [ ] Tests: broadcasting compatibility, scalar ops, reshape size-mismatch errors
- [ ] Person B — Tensor: matmul, reductions + demo + tooling - Anand
  - [ ] `matmul()` with inner-dimension shape validation (`a.shape[-1] == b.shape[-2]`)
  - [ ] Reductions: `sum()`, `mean()`, `max()` with axis / keepdims
  - [ x] Light tooling: repo scaffolding, requirements.txt, shared pytest harness
  - [x ] CI: GitHub Actions workflow to run the test suite on every push/PR (test job only)
  - [ ] End-to-end demo: assemble Sequential MLP, run forward pass, verify output shape (batch, 10) and Softmax rows sum to 1
  - [ ] Tests: matmul shape errors, reduction axis semantics
- [ ] Person C — Activations (depends on Tensor arithmetic + max/sum) - Tina 
  - [ ] Activation base pattern: `forward()`, `__call__()`, `backward()` stub
  - [ ] `ReLU` (`np.maximum(0, x)`)
  - [ ] `Sigmoid` (numerically stable piecewise + clip at ±500)
  - [ ] `Tanh` (`np.tanh`)
  - [ ] `GELU` (fast approx, reuses Sigmoid: `sigmoid(1.702 * x) * x`)
  - [ ] `Softmax` (subtract per-row max for stability, dim argument)
  - [ ] Tests: output ranges, Softmax sums to 1, no NaN on large logits (e.g. 100)
- [ ] Person D — Layers (depends on Tensor matmul + __add__) - Trisha 
  - [ ] `Layer` base class: `forward()`, `__call__()`, `parameters()`
  - [ ] `Linear`: LeCun init sqrt(1/in_features), weight (in, out) + bias (out,), forward = xW + b
  - [ ] `Dropout`: training/inference modes, inverted scaling 1/(1-p)
  - [ ] `Sequential`: chain forward, flatten per-layer parameters
  - [ ] Tests: parameter counts, shape composition, dropout passthrough at inference

     
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
