# ResNet18 from Scratch → Distributed Data Parallel (DDP) Benchmarking

```{=html}
<p align="center">
```
`<b>`{=html}From implementing a convolutional neural network from first
principles to measuring multi-GPU scaling with PyTorch DDP.`</b>`{=html}
```{=html}
</p>
```
## Project Overview

This project is an end-to-end study of deep-learning training systems
built around a ResNet18 image classifier trained on CIFAR-10.

The project progresses from:

1.  ResNet18 implementation from scratch
2.  CIFAR-10 data augmentation and normalization
3.  Single-GPU training and validation
4.  Automatic Mixed Precision (AMP)
5.  Learning-rate scheduling and checkpointing
6.  TensorBoard experiment tracking
7.  PyTorch profiling and GPU-kernel analysis
8.  Controlled performance experiments
9.  DistributedDataParallel (DDP)
10. DistributedSampler and distributed metric reduction
11. Linux/WSL2 + NCCL distributed execution
12. 1-GPU vs 2-GPU benchmarking
13. Quantitative scaling analysis

This is best presented as a **deep-learning systems engineering and
experimental research project**, rather than as a novel ML-algorithm
contribution. The value is the end-to-end investigation,
reproducibility, profiling, distributed implementation, and quantitative
benchmarking.

## Research / Engineering Questions

-   How is ResNet18 constructed internally?
-   What are the computational bottlenecks during convolutional
    training?
-   What does AMP change?
-   How can profiling identify expensive operators and CUDA kernels?
-   What changes when a single-GPU training loop becomes DDP?
-   How are data, gradients, metrics, and checkpoints handled across
    processes?
-   How close does 2-GPU training get to ideal 2× scaling?
-   Where does the remaining scaling overhead come from?

## 1. ResNet18 From Scratch

The model is implemented using PyTorch primitives rather than importing
a pre-built ResNet.

A residual block learns:

``` text
F(x) + x
```

rather than requiring the stacked layers to directly learn the complete
mapping.

Conceptually:

``` text
             x
             │
        ┌────┴────┐
        │         │
        │         ▼
        │    Conv → BN → ReLU
        │         │
        │    Conv → BN
        │         │
        └─────── (+)
                 │
                 ▼
                ReLU
```

When dimensions change, the shortcut uses a projection such as a 1×1
convolution followed by normalization.

For CIFAR-10, the architecture is adapted to 32×32 images rather than
blindly using an ImageNet-sized stem.

## 2. CIFAR-10 Pipeline

Training uses:

``` python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
transforms.ToTensor()
transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)
```

Validation uses deterministic tensor conversion and normalization
without random augmentation.

## 3. Training Stack

The training loop uses:

-   CrossEntropyLoss
-   Adam
-   CosineAnnealingLR
-   CUDA AMP
-   validation loss
-   validation accuracy
-   checkpointing

The training flow is:

``` text
batch → GPU → forward → loss → backward → optimizer step
```

AMP uses `torch.autocast` and gradient scaling.

## 4. TensorBoard

TensorBoard tracks training loss, validation loss, validation accuracy,
learning rate, model graph, and selected training images.

Run locally with:

``` bash
tensorboard --logdir runs
```

Then open:

``` text
http://localhost:6006
```

Recommended layout:

``` text
runs/
├── 1gpu/
│   └── events.out.tfevents....
└── 2gpu/
    └── events.out.tfevents....
```

### Screenshots to add

The repository should include screenshots of the final TensorBoard
visuals. Add them under `docs/images/` when available.

``` markdown
![Training Loss Comparison](docs/images/tensorboard_train_loss.png)
![Validation Accuracy Comparison](docs/images/tensorboard_val_accuracy.png)
![Learning Rate](docs/images/tensorboard_learning_rate.png)
![ResNet18 Model Graph](docs/images/tensorboard_model_graph.png)
```

These are intentional placeholders; the screenshots can be added after
the final dashboard comparison is prepared.

## 5. Profiling

The project uses `torch.profiler` to inspect:

-   CPU activity
-   CUDA activity
-   PyTorch operators
-   GPU kernels
-   execution traces

The profiling results showed that convolutional computation dominates
the workload.

In the detailed profiling window:

-   `aten::convolution_backward` was about 57.9% of self CUDA time.
-   `aten::cudnn_convolution` was about 37.6%.
-   CUDA kernels including `sm75_xmma_fprop...`, `sm75_xmma_dgrad...`,
    and `volta_sgemm_32x128_nt` appeared in the trace.

The important takeaway is the execution hierarchy:

``` text
Training code
  ↓
PyTorch operators
  ↓
Autograd
  ↓
cuDNN / CUDA
  ↓
GPU kernels
  ↓
GPU hardware
```

Profiler reference: https://docs.pytorch.org/docs/stable/profiler.html

## 6. Performance Experiments Before DDP

Initial experiments were performed on a GTX 1660 Ti Max-Q.

Baseline:

``` text
Batch size:       128
Workers:          0
pin_memory:       False
non_blocking:     False
AMP:              Enabled
cuDNN benchmark:  False
```

Steady-state baseline (epochs 3--5):

``` text
≈ 178.95 s / epoch
```

Experiments showed that obvious DataLoader and transfer changes did not
materially improve the workload:

  Experiment                       Steady-state result
  ------------------------------ ---------------------
  Baseline                             ≈178.95 s/epoch
  4 workers                            ≈180.81 s/epoch
  4 workers + pin + persistent         ≈178.77 s/epoch
  \+ non_blocking                      ≈187.26 s/epoch
  Batch size 256                       ≈240.51 s/epoch
  cuDNN benchmark=True                 ≈185.25 s/epoch

This led to an important engineering decision: stop micro-optimizing the
input pipeline and investigate distributed scaling.

## 7. DistributedDataParallel

DDP uses one process per GPU.

``` text
                 Dataset
                    │
           DistributedSampler
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Process 0           Process 1
        GPU 0               GPU 1
          │                   │
       ResNet18             ResNet18
          │                   │
       Backward             Backward
          └─────────┬─────────┘
                    ▼
                 AllReduce
                    │
                    ▼
            synchronized update
```

The model is wrapped with:

``` python
model = DDP(
    model,
    device_ids=[local_rank],
    broadcast_buffers=False,
)
```

PyTorch DDP documentation:
https://docs.pytorch.org/docs/main/notes/ddp.html

### DistributedSampler

``` python
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
```

At each epoch:

``` python
train_sampler.set_epoch(epoch)
```

This ensures correct distributed shuffling.

### Distributed metrics

Each process accumulates local totals, then global totals are
reconstructed with `dist.all_reduce`:

``` python
dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
```

This prevents validation metrics from representing only rank 0's local
subset.

### Rank-aware logging

Only rank 0 creates the TensorBoard writer and saves checkpoints:

``` python
if rank == 0:
    writer = SummaryWriter(...)
```

and:

``` python
if rank == 0:
    save_checkpoint(...)
```

## 8. Windows → WSL2 → NCCL

A significant systems lesson came from attempting CUDA DDP on Windows.

Standalone CUDA training worked, but the Windows distributed execution
path encountered backend/runtime failures. The workload was moved to
Linux through WSL2 and run with NCCL.

The resulting stack was:

``` text
Windows
  ↓
WSL2
  ↓
Ubuntu
  ↓
CUDA
  ↓
NCCL
  ↓
PyTorch DDP
```

This demonstrated that distributed training depends not only on Python
code but also on the interaction between OS, CUDA, PyTorch, launcher,
communication backend, and hardware.

## 9. Reproducible DDP Execution

Single GPU:

``` bash
torchrun --standalone --nproc-per-node=1 train_ddp.py
```

Two GPUs:

``` bash
torchrun --standalone --nproc-per-node=2 train_ddp.py
```

PyTorch Distributed overview:
https://docs.pytorch.org/tutorials/beginner/dist_overview.html

## 10. Final Benchmark Setup

Both final runs used:

``` text
Model:              ResNet18
Dataset:            CIFAR-10
Epochs:             5
Per-GPU batch size: 128
AMP:                Enabled
Optimizer:          Adam
Scheduler:          CosineAnnealingLR
DataLoader workers: 0
Backend:            NCCL
```

Hardware:

``` text
1-GPU experiment: 1 × RTX 4090
2-GPU experiment: 2 × RTX 4090
```

## 11. Final 1-GPU Results

    Epoch   Train Loss   Val Loss   Val Accuracy   Train Time
  ------- ------------ ---------- -------------- ------------
        1       1.4209     1.4516         52.33%       7.79 s
        2       0.9349     0.9045         68.77%       7.49 s
        3       0.7061     0.7253         75.04%       7.73 s
        4       0.5484     0.6327         79.18%       7.51 s
        5       0.4370     0.4645         84.15%       7.51 s

Steady-state average over epochs 2--5:

``` text
7.56 s / epoch
```

Approximate throughput:

``` text
≈ 6,614 images/s
```

## 12. Final 2-GPU Results

    Epoch   Train Loss   Val Loss   Val Accuracy   Train Time
  ------- ------------ ---------- -------------- ------------
        1       1.5024     1.2705         54.66%       4.65 s
        2       1.0109     0.9770         65.79%       4.50 s
        3       0.7878     0.7757         72.45%       4.37 s
        4       0.6300     0.7416         74.75%       4.35 s
        5       0.5093     0.5314         81.69%       4.35 s

Steady-state average over epochs 2--5:

``` text
4.39 s / epoch
```

Approximate throughput:

``` text
≈ 11,390 images/s
```

## 13. Scaling Analysis

  Metric                                1 GPU           2 GPUs
  --------------------------- --------------- ----------------
  Hardware                        1× RTX 4090      2× RTX 4090
  Steady-state epoch                   7.56 s           4.39 s
  Throughput                    \~6,614 img/s   \~11,390 img/s
  Final validation accuracy            84.15%           81.69%

Speedup:

``` text
7.56 / 4.39 ≈ 1.72×
```

Parallel scaling efficiency:

``` text
1.72 / 2 × 100 ≈ 86%
```

### Main result

> Moving from 1× to 2× RTX 4090 reduced steady-state epoch time from
> approximately 7.56 s to 4.39 s, producing a 1.72× speedup and
> approximately 86% parallel scaling efficiency.

## 14. Why Not 2×?

The theoretical 2× target would be:

``` text
7.56 / 2 = 3.78 s
```

The measured result was:

``` text
4.39 s
```

The difference comes from distributed-system overheads such as:

-   gradient synchronization
-   communication latency
-   process synchronization
-   input overhead
-   GPU utilization
-   workload size

DDP performs gradient all-reduce during backward propagation, so
multi-GPU scaling is never simply "add another GPU and divide the time
by two."

## 15. Benchmarking Caveat

The experiment uses a fixed **per-GPU** batch size:

``` text
1 GPU → global batch 128
2 GPUs → global batch 256
```

Therefore this is not a strict strong-scaling experiment with fixed
global batch size.

A future strong-scaling experiment should use the same global batch:

``` text
1 GPU → batch 256
2 GPUs → batch 128 per GPU
```

and compare the wall-clock time for the same amount of work.

This distinction is intentionally documented because rigorous
benchmarking requires specifying what is held constant.

## 16. Accuracy Caveat

Final validation accuracy was:

``` text
1 GPU → 84.15%
2 GPUs → 81.69%
```

This should not be interpreted as DDP reducing model quality.

The global batch size changes between the experiments, which changes the
optimization trajectory. The primary benchmark target here is
training-system performance, not a controlled accuracy comparison.

A future study should control:

-   global batch size
-   learning-rate scaling
-   random seed
-   number of epochs / optimization steps

before drawing accuracy conclusions.

## 17. Project Structure

``` text
resnet18_ddp/
├── configs/
├── datasets/
│   └── cifar10.py
├── models/
│   └── resnet.py
├── engine/
├── scripts/
├── runs/
│   ├── 1gpu/
│   └── 2gpu/
├── checkpoints/
├── train.py
├── train_ddp.py
├── checkpoint.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

Adjust filenames above if the repository structure changes.

## 18. Reproducibility

Clone:

``` bash
git clone https://github.com/a-pt/resnet18_ddp.git
cd resnet18_ddp
```

Install:

``` bash
uv sync
```

The CIFAR-10 loader downloads the dataset automatically. Keep `data/`
out of Git.

For a distributed CUDA run, use a Linux/WSL2 environment with a
functioning NCCL setup.

## 19. Future Work

### Strong scaling

Keep global batch size fixed and compare 1 vs 2 GPUs.

### More GPUs

Extend:

``` text
1 → 2 → 4 → 8 GPUs
```

and plot:

``` text
GPU count vs speedup
GPU count vs efficiency
GPU count vs throughput
```

### Communication profiling

Measure computation and all-reduce communication separately.

### Larger models

Repeat the experiment with ResNet50 or a vision transformer.

### Precision study

Compare:

``` text
FP32
AMP FP16
AMP BF16
```

for speed, memory, and accuracy.

### Larger workloads

Repeat the scaling study with larger images, datasets, or models where
distributed overhead is a smaller fraction of total computation.

## 20. What This Project Demonstrates

### Deep learning

-   residual networks
-   convolutional architectures
-   BatchNorm
-   global average pooling
-   optimization
-   learning-rate scheduling
-   mixed precision

### PyTorch

-   `nn.Module`
-   autograd
-   DataLoader
-   checkpointing
-   TensorBoard
-   `torch.profiler`

### GPU systems

-   CUDA execution
-   CPU → GPU transfers
-   cuDNN
-   GPU kernels
-   AMP
-   synchronization

### Distributed systems

-   process groups
-   ranks
-   world size
-   local rank
-   DistributedSampler
-   DDP
-   NCCL
-   all-reduce
-   distributed metrics
-   rank-aware checkpointing

## 21. Research Positioning

This repository should be presented as a **deep-learning systems /
distributed-training experimental project**.

It does not claim a new neural architecture or a new optimization
algorithm.

Its research value comes from the experimental methodology:

``` text
Implement
   ↓
Measure
   ↓
Profile
   ↓
Hypothesize
   ↓
Change one variable
   ↓
Benchmark
   ↓
Scale
   ↓
Analyze limitations
```

That methodology is directly transferable to larger research workloads
such as computer vision models, generative models, multimodal models,
and large-scale training systems.

## 22. References

-   PyTorch DistributedDataParallel:
    https://docs.pytorch.org/docs/main/notes/ddp.html
-   PyTorch Distributed Overview:
    https://docs.pytorch.org/tutorials/beginner/dist_overview.html
-   Getting Started with DDP:
    https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
-   PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html
-   PyTorch Profiler Recipe:
    https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
-   TensorBoard with PyTorch:
    https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

## 23. Author

**Athira PT**

M.Tech --- Computer Science and Engineering, IIT Madras

Interests:

-   Deep Learning
-   Computer Vision
-   Generative AI
-   Distributed Training
-   Efficient AI Systems
-   Multimodal Learning
-   Large-Scale Model Training
