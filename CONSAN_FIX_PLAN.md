# Account for instrumentation-created layout-conversion scratch

Branch: `keren/consan-helper-scratch`

Status: **Proposal only; no native/compiler implementation applied.**

Reproducer: [03-attention-allocation-offset](/mnt/keren/tmp/consan-unfixed-repros-20260925/03-attention-allocation-offset/README.md). Baseline results were verified on `831e8e2517`; runtime checks use only physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

## Evidence and diagnosis

The four-CTA causal attention IR reproducibly aborts in `getSharedMemoryBase` because an operation has no `allocation.offset`. The pipeline allocates original shared memory before ConSan generates helper functions and further layout conversions. `FunctionBuilder.cpp::getOrCreateFunction` requests warp-shuffle conversion for generated helpers, so identify the exact conversion falling back to shared memory before selecting a remedy; not every conversion without an offset needs scratch.

A fresh IR dump immediately before ConvertTritonGPUToLLVM is saved in the investigation directory. The production lowering assertion must remain: inventing offset zero could alias live user allocations.

## Proposed fix

Make the allocation/late-instrumentation contract explicit. First ensure generated helper conversions that are intended to be register/warp-local remain so through canonicalization. For conversions that genuinely need shared memory, reserve and annotate dedicated instrumentation scratch in a transformation/analysis pass before LLVM lowering. Preserve existing allocation offsets and sizes used to construct ConSan's physical-region registry; a blind second allocation pass could silently invalidate that registry.

Scope the late allocation to new helper/call-frame scratch, append it after reserved original storage, account for call graphs and warp-specialized capture reservations, and update the module shared-memory requirement. Shared-memory conversions in helpers must receive real `allocation.offset` / `allocation.size` metadata or be eliminated legitimately. Cross-CTA reasoning belongs in that analysis/transformation, not in the lowering.

Likely touch points: `lib/Dialect/TritonInstrument/IR/FunctionBuilder.cpp`, `lib/Dialect/TritonInstrument/Transforms/PrepareConSanCaptures.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp`, the allocation utilities, and pass ordering in `third_party/nvidia/backend/compiler.py`. Keep the patch limited to the actual missing-scratch path once isolated.

## Acceptance checks

After `make`, the saved full MLIR must lower with verification enabled. Add a reduced regression to an existing instrumentation/allocation lit file, checking nonoverlapping helper scratch and preserved original offsets. Exercise ordinary calls and warp-specialized helper calls; check capture reservation remains sufficient. Then run causal/noncausal 2/4-CTA attention numerics and ConSan on GPU 2. Any subsequent failure must be reported separately rather than calling compilation success a synchronization pass.
