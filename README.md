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
  - [x] `matmul()` with inner-dimension shape validation (`a.shape[-1] == b.shape[-2]`)
  - [x] Reductions: `sum()`, `mean()`, `max()` with axis / keepdims
  - [x] Light tooling: repo scaffolding, requirements.txt, shared pytest harness
  - [x] CI: GitHub Actions workflow to run the test suite on every push/PR (test job only)
  - [x] End-to-end demo: assemble Sequential MLP, run forward pass, verify output shape (batch, 10) and Softmax rows sum to 1
  - [x] Tests: matmul shape errors, reduction axis semantics
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

     
- Everyone who wants to work on GPU acceleration: understanding GPUs architecture and numba.cuda framework 
  - Recommend : https://www.youtube.com/watch?v=9bBsvpg-Xlk 
  - Understanding GPU architecture hierarchy and memory organization 
    - [ ] read/skim: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html 
    - Threads, warps, blocks, grid 
    - SIMT (Single instruction, multiple threads) 
  - Basics of writing cuda kernels with numba.cuda 
    - [ ] read/skim: https://nvidia.github.io/numba-cuda/user/index.html 
    - Memory transfers and thread mapping for 1d and nd tensors 
    - Handling alignments 

Week 2
------

- [ ] Specialized layers (CNN or attention) 
- [ ] Have a working backward pass (CPU)
- [ ] visualizing the computation graph

- GPU Person A - memory management, basic arithmetic, activations  
  - [ ] `detect()` to check for GPU support 
  - [ ] `.to(device)` to logically move `Tensor` to `device` ("cpu" or "cuda")
  - [ ] `.numpy()` physically fetch data from GPU to CPU 
  - [ ] Kernels for basic operations: `__add__`, `__subtract__`, `__mul__`, `__truediv__` 
  - [ ] Kernels for activation layers: `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`

- GPU Person B - matrix multiplication, reductions, testing/profiling suite 
  - [ ] CI to benchmark CPU operations 
  - [ ] CI to confirm correctness of CUDA kernels 
  - [ ] Naive implementation of `__matmul__`
  - [ ] tiled implementation of `__matmul__`
  - [ ] Reductions 

Week 3
------

- [ ] Optimizer? (CPU)
- [ ] Convolution
- [ ] Have a working backward pass (GPU)
- [ ] Integrate CPU and GPU together 

- [ ] Kernel fusion 

Week 4
------

- [ ] Iterate 
