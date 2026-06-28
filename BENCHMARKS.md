# Benchmark Results

## Sprint 3 — Naive (baseline)
- Matrix size: 512×512
- Operation: `relu(A @ B + bias)`
- Time: **10.15s**
- Notes: No fusion, no tiling, no vectorization

## Sprint 4 — Fusion
- Matrix size: 512×512
- Operation: `relu(A @ B + bias)` fused
- Time: **1.24s**
- Speedup vs. baseline: **8.2×**
- Notes: Fused matmul + bias + relu into single loop; eliminated memory round-trips

## Sprint 5 — Tiling
(coming soon)

## Sprint 6 — Vectorization
(coming soon)
