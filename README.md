# HYS-Machine-Learning-Framework

# TODO 

Week 1
------
- [x] Person A — Tensor: construction, arithmetic, reshape (ships first — unblocks C and D fastest) - Catherine
  - [x] `__init__` wrapping a NumPy array; set data, shape, size, dtype
  - [x] Arithmetic dunders: `__add__`, `__sub__`, `__mul__`, `__truediv__` (Tensor+Tensor and Tensor+scalar)
  - [x] `reshape()` with -1 dimension inference + element-count validation
  - [x] `transpose()` (default: swap last two dims)
  - [x] Tests: broadcasting compatibility, scalar ops, reshape size-mismatch errors
- [x] Person B — Tensor: matmul, reductions + demo + tooling - Anand
  - [x] `matmul()` with inner-dimension shape validation (`a.shape[-1] == b.shape[-2]`)
  - [x] Reductions: `sum()`, `mean()`, `max()` with axis / keepdims
  - [x] Light tooling: repo scaffolding, requirements.txt, shared pytest harness
  - [x] CI: GitHub Actions workflow to run the test suite on every push/PR (test job only)
  - [x] End-to-end demo: assemble Sequential MLP, run forward pass, verify output shape (batch, 10) and Softmax rows sum to 1
  - [x] Tests: matmul shape errors, reduction axis semantics
- [x] Person C — Activations (depends on Tensor arithmetic + max/sum) - Tina 
  - [x] Activation base pattern: `forward()`, `__call__()`, `backward()` stub
  - [x] `ReLU` (`np.maximum(0, x)`)
  - [x] `Sigmoid` (numerically stable piecewise + clip at ±500)
  - [x] `Tanh` (`np.tanh`)
  - [x] `GELU` (fast approx, reuses Sigmoid: `sigmoid(1.702 * x) * x`)
  - [x] `Softmax` (subtract per-row max for stability, dim argument)
  - [x] Tests: output ranges, Softmax sums to 1, no NaN on large logits (e.g. 100)
- [x] Person D — Layers (depends on Tensor matmul + __add__) - Trisha 
  - [x] `Layer` base class: `forward()`, `__call__()`, `parameters()`
  - [x] `Linear`: LeCun init sqrt(1/in_features), weight (in, out) + bias (out,), forward = xW + b
  - [x] `Dropout`: training/inference modes, inverted scaling 1/(1-p)
  - [x] `Sequential`: chain forward, flatten per-layer parameters
  - [x] Tests: parameter counts, shape composition, dropout passthrough at inference

     
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

Week 2 & 3 & Part of Week 4
------
- [ ] Person A — Losses + log_softmax helper - Trisha
  - [ ] No blockers — depends only on Tensor (Sprint 1)
  - [ ] `log_softmax()` with log-sum-exp trick (numerical stability, subtract max before exp)
  - [ ] `MSELoss.forward()` (subtract, square, mean)
  - [ ] `CrossEntropyLoss.forward()` (calls log_softmax, indexes correct class, negative mean)
  - [ ] `BinaryCrossEntropyLoss.forward()` (clamp predictions with eps=1e-7, BCE formula)
  - [ ] Tests: NaN on large logits (e.g. 100), shape mismatch errors, logits vs probabilities distinction

- [ ] Person B — DataLoader - Catherine
  - [ ] No blockers — depends only on Tensor (Sprint 1)
  - [ ] `Dataset` abstract base class (`__len__`, `__getitem__` with @abstractmethod)
  - [ ] `TensorDataset` wrapping multiple tensors, validates matching first dimension
  - [ ] `DataLoader.__init__()` (store dataset, batch_size, shuffle flag)
  - [ ] `DataLoader.__iter__()` (index shuffling, batch grouping, lazy generator with yield)
  - [ ] `DataLoader._collate_batch()` (np.stack per position into batch tensors)
  - [ ] Tests: mismatched tensor dimensions, last batch smaller than batch_size, shuffle vs no shuffle

- [x] Person C — Autograd - Anand 
  - [x] No blockers — depends only on Tensor (Sprint 1)
  - [x] `Function` base class (stores `saved_tensors`, defines `apply()`)
  - [x] `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward` (gradient rules for arithmetic)
  - [x] `MatmulBackward` (grad_A = grad @ B.T, grad_B = A.T @ grad)
  - [x] `SumBackward`, `ReshapeBackward`, `TransposeBackward`
  - [x] `Tensor.backward()` (seed gradient to 1.0 for scalars, accumulate into .grad, recurse through _grad_fn)
  - [x] `Tensor.zero_grad()` (set .grad to None)
  - [x] `enable_autograd()` (monkey-patch Tensor ops to attach _grad_fn on output)
  - [x] Tests: x.grad correct for arithmetic chains, zero_grad resets accumulation, requires_grad=False skips graph

- [ ] Person D — Optimizers + Trainer - Tina
  - [ ] ⚠️ Blocked by Person C — needs Tensor.backward() and .grad before optimizer steps can be tested end-to-end. However can get started without implementation by assuming function
  - [ ] `Optimizer` base class (`zero_grad()`, `step()` interface)
  - [ ] `SGD` with momentum (velocity buffer, lazy init, weight decay)
  - [ ] `Adam` (m and v buffers, bias correction 1 - β^t, adaptive step)
  - [ ] `AdamW` (same as Adam but weight decay applied after gradient update, decoupled)
  - [ ] `CosineSchedule.get_lr()` (cosine annealing formula)
  - [ ] `clip_grad_norm()` (global norm across all params, uniform scale if exceeds max_norm)
  - [ ] `Trainer.train_epoch()` (forward, loss, backward, clip, step, zero_grad, scheduler update)
  - [ ] `Trainer.evaluate()` (model.training=False, forward only, loss + accuracy)
  - [ ] `Trainer.save_checkpoint()` / `load_checkpoint()` (pickle full state)
  - [ ] Tests: Adam memory 3x params, zero_grad between iterations, checkpoint round-trip
    
- GPU Person A - memory management, basic arithmetic, activations - Lily
  - [ ] `detect()` to check for GPU support 
  - [ ] `.numpy()` physically fetch data from GPU to CPU 
  - [ ] Kernels for basic operations: `__add__`, `__subtract__`, `__mul__`, `__truediv__` (need to manually broadcast)
  - [ ] Kernels for activation layers: `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`

- GPU Person B - matrix multiplication, reductions, testing/profiling suite - Changwoo
  - [x] `.to(device)` to logically move `Tensor` to `device` ("cpu" or "cuda")
  - [ ] CI to benchmark CPU operations 
  - [ ] CI to confirm correctness of CUDA kernels 
  - [ ] Naive implementation of `__matmul__` (need to manually broadcast)
  - [ ] tiled implementation of `__matmul__` (need to manually boradcast)
  - [ ] Reductions 

Week 4:

Week 3
------

- [ ] Have a working backward pass (GPU)
- [ ] Integrate CPU and GPU together 

- [ ] Kernel fusion 

Week 4
------
- [ ] Review Demo 
- [ ] Iterate 
