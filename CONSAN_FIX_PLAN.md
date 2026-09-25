# Limit barrier instrumentation to selected state slices

Branch: `keren/consan-selective-state-access`

Status: **Proposal only; no native/compiler implementation applied.**

Reproducer: [06-instrumentation-overhead](/mnt/keren/tmp/consan-unfixed-repros-20260925/06-instrumentation-overhead/README.md). Baseline results were verified on `831e8e2517`; runtime checks use only physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

## Evidence and diagnosis boundary

The frozen Gluon example passes numerics with and without ConSan. On GPU 2 it measures 373.091 ms versus 0.192448 ms in the current warm-cache CUDA-event harness; the instrumented kernel reports 255 registers and 634 spills. This shows substantial overhead, not that every lost cycle has been attributed to a particular helper.

## Proposed fix

Reduce full-table traffic and live tensor state in barrier/frontier helpers. Narrow accesses to the selected barrier, phase and owner-CTA set when those indices are known, retaining all required source/observer CTA relationships. Stream frontier reductions in bounded pieces and release temporaries before loading the next state table. Prefer scalar/indexed selection of backing storage to loading the entire table and masking it afterward.

Existing one-buffer paths already narrow `createSetReadVisibilityCall` and related operations; do not reimplement them. Target remaining broad operations such as `createTrackVisibleAccessesCall`, `createTransferVisibleAccessesCall`, `createCompleteBarrierWaitCall`, read/write visibility verification and cluster-frontier publication in `lib/Dialect/TritonInstrument/IR/FunctionBuilder.cpp`. Use current stride-aware scratch helpers in `Utility.cpp`. Preserve the generic masked fallback where effect sets or alias candidates are not singular.

Select the first helper using measured instruction/memory/spill evidence. Keep lock acquisition/release, synchronization instrumentation, proxy ordering and allocation poisoning intact. Do not optimize by dropping checks, skipping CTAs, or assuming only the leader owns the state. The bounded-state-tensor proposal handles representability; this proposal reduces the amount of work and register lifetime even when the state already fits.

## Acceptance checks

After `make`, rerun the frozen example with the exact same tile/stage configuration and numerical checks on GPU 2. Compare instrumented before/after time using matched timing methodology; record spills, registers, scratch bytes, and helper-level evidence. Check both cold and warm cache if drawing broader conclusions. Run existing positive and negative ConSan tests covering barrier phases, mixed consumers, physical aliases, explicit cross-CTA effects and proxy ordering. A reduction in spills alone is not a correctness or performance result.
