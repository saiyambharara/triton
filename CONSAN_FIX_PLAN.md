# Tile ConSan state operations without reducing coverage

Branch: `keren/consan-bounded-state-tensors`

Status: **Proposal only; no native/compiler implementation applied.**

Reproducer: [04-attention-tensor-limit](/mnt/keren/tmp/consan-unfixed-repros-20260925/04-attention-tensor-limit/README.md). Baseline results were verified on `831e8e2517`; runtime checks use only physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

## Evidence and diagnosis

The eight-CTA attention reproducer fails the 1,048,576-element verifier limit. Generated state includes 2,097,152-element tensors. `lib/Dialect/TritonInstrument/IR/Utility.cpp::populateAndPassToWarpSpecialize` constructs read tracking with dimensions `[numCTAs, numBufs, numCTAs, numBarriers, numCTAs, 2]`; full-state broadcasting and pointer construction can multiply these axes beyond the limit.

## Proposed fix

Separate logical state storage from the tensor shape materialized by each helper. Preserve the full backing allocation and its original strides, but process the buffer/barrier axes in bounded tiles. Initially keep the dense state semantics unchanged and target representability. Use explicit offset/stride views and a runtime tile loop where needed rather than unrolling another oversized tensor. Split initialization, loads/stores, masks, broadcasts and reductions consistently; fixing only the original splat is insufficient.

Combine reduction results across tiles with the original operator and identity. Keep both barrier phases, CTA ownership, thread frontiers, unknown-alias masks and broadcast-recipient relations. State identity must not be truncated, sampled, or discarded. Do not raise the tensor-element limit to conceal the scaling problem.

Primary touch points: state storage/type planning and scratch pointer construction in `lib/Dialect/TritonInstrument/IR/Utility.cpp`; tiled helper generation/reductions in `lib/Dialect/TritonInstrument/IR/FunctionBuilder.cpp`. Retain existing one-buffer fast paths and their full-backing-type cache keys. This branch is independent of the separate proposal to skip unselected state traffic for speed.

## Acceptance checks

After `make`, verify the saved eight-CTA input no longer produces any over-limit tensor with the normal verifier enabled. Add an existing-file lit case crossing the old threshold and a smaller control. Compare results against the dense implementation on small configurations for each frontier/phase transition, and preserve existing negative ConSan tests. Run 2/4/8-CTA synchronization checks on GPU 2. Passing the element-limit stage does not by itself establish complete attention correctness; rerun the entire pipeline.
