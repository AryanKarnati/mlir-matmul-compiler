# MLIR Matmul Compiler

## Overview

A lightweight MLIR compiler that compiles a neural-network layer — `D = relu(A·B + C)` — from high-level tensor operations down to optimized machine code. The compiler demonstrates end-to-end lowering through MLIR dialects, optimization passes (operator fusion and loop tiling), and JIT execution, achieving **17.7× speedup** over naive implementation.

## Why This Matters

Modern AI accelerators (TPUs, GPUs, custom silicon) require compilers that can automatically optimize tensor computations. This project explores how that optimization happens: taking an abstract operation and progressively lowering it through representations that are easier to optimize, until arriving at efficient machine code.

## The Problem

A matrix multiplication followed by a bias addition and ReLU activation is the fundamental building block of neural networks: 

D = relu(A·B + C)

Without optimization, this computation requires three separate passes through memory: one for the matmul, one for the bias add, and one for ReLU. Each pass loads and stores 512×512 matrices, causing cache misses and memory bandwidth bottlenecks. The goal is to fuse these operations and tile the loops so intermediate results stay in fast cache.

## The approach
This project uses a four-stage IR lowering pipeline to optimize the computation from abstract tensor operations down to machine code.

Linalg dialect -> Affine dialect -> LLVM dialect -> Machine code

### Stage 1: Linalg Dialect (Input)

The computation is expressed using high-level operations: `linalg.matmul` for matrix multiplication, `linalg.add` for elementwise addition, and `linalg.generic` for ReLU. These operations describe *what* to compute without specifying *how* — the matmul hides the three nested loops, the add hides the iteration pattern, etc. This abstraction makes optimization possible because the compiler can reason about entire operations rather than individual loop iterations.

### Stage 2: Affine Dialect (Optimization)

The fusion pass (`-linalg-fuse-elementwise-ops`) observes that the output of matmul is immediately consumed by add, and add's output is immediately consumed by ReLU. Instead of three separate loops with memory round-trips between them, it merges these into a single fused loop nest. The tiling pass (`-affine-loop-tile`) then restructures the loops into cache-friendly blocks (64×64×64), ensuring hot data stays in L1/L2 cache rather than spilling to main memory.

### Stage 3: LLVM Dialect (Lowering)

A series of conversion passes lower away the remaining abstraction: affine loops become explicit branches, memory references become pointers, arithmetic becomes LLVM intrinsics. At this level, the optimizer can see exactly how many loads/stores occur and in what order.

### Stage 4: Machine Code (Execution)

The MLIR JIT takes the LLVM dialect IR and compiles it to x86-64 machine instructions, which execute on the CPU.

## Results

The compiler was benchmarked on a 512×512 matrix multiplication with bias and ReLU on a single CPU core. Each configuration was run once; timing is wall-clock time.

| Stage | Operation | Time | Speedup vs. Baseline | Notes |
|-------|-----------|------|----------------------|-------|
| Sprint 3 | Naive (no optimization) | 10.15s | 1.0× | Three separate loops, memory round-trips |
| Sprint 4 | Fused | 1.24s | 8.2× | Matmul + bias + ReLU in one loop; eliminates memory writes/reads between ops |
| Sprint 5 | Fused + Tiled | 0.574s | 17.7× | Loops tiled to 64×64×64 blocks; data stays in L1/L2 cache |

**Key insight:** Most of the speedup (8.2×) comes from fusion thats eliminates memory round trips. Tiling adds another 2.16× by improving cache locality. Together, the compiler transforms a memory-bound computation into a cache-efficient one.




