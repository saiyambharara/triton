# Reduce repeated verification of large lowered aggregate types

Branch: `keren/consan-type-verification-cost`

Status: **Proposal only; no native/compiler implementation applied.**

Reproducer: [05-eight-cta-compile-time](/mnt/keren/tmp/consan-unfixed-repros-20260925/05-eight-cta-compile-time/README.md). Baseline results were verified on `831e8e2517`; runtime checks use only physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

## Evidence and diagnosis boundary

The saved instrumented eight-CTA matmul input exceeds 180 seconds when replaying ConvertTritonGPUToLLVM with verification enabled. A debugger interruption at approximately 15 seconds finds the process in `AttrTypeWalker`, `verifyOpTypeSymbolUses`, and `verifySymbolTable`, called by pass verification. This identifies an actionable path, but one stack sample is not a complete profile and does not prove it dominates every run. `perf` sampling was unavailable because of host perf_event permissions; no system settings were changed.

A diagnostic replay with the embedded `verify_each` flag changed to false also exceeded a 60-second limit. This does not isolate verification as the sole cost and does not establish that the command bypassed every verification stage. Keep the cache proposal provisional until multiple samples/timing phases confirm it.

## Proposed fix

Investigate repeated symbol-use traversal of large lowered LLVM aggregate types. If confirmed, memoize reusable type-walk results within a verification invocation and symbol-table scope, so repeated SSA values with the same large aggregate type do not repeatedly traverse the entire type graph. Retain all verification. Cache type traversal separately from operation-dependent symbol resolution; preserve operation-specific diagnostics, nested symbol-table scopes, recursive/identified types and mutation boundaries.

The relevant implementation is in the LLVM/MLIR dependency (`mlir/lib/IR/SymbolTable.cpp`, `verifyOpTypeSymbolUses` / `verifySymbolTable`, and the attribute/type walker), not the ConSan barrier protocol. This repository branch records the Triton reproducer and dependency-fix integration plan; an LLVM-side patch and compatible dependency build/pin may be required. Do not present `verify_each=false` or a longer timeout as the fix.

Before committing to that change, collect multiple interrupted stacks or an allowed CPU profile and distinguish conversion time, verification time and printing time. A no-verification run is diagnostic only. If traversal is not the dominant cost, revise this proposal around measured evidence rather than landing speculative caching.

## Acceptance checks

Use the exact saved post-instrumentation input, current compiler build, and unchanged verification settings for before/after CPU wall-time and peak-RSS measurements. Require successful completion rather than merely moving past a timeout. Add LLVM verifier coverage for repeated large aggregate types, recursive types, mutable identified types, and symbol scopes; run Triton lit checks to catch dependency integration effects. Keep instrumentation semantics and generated kernel behavior unchanged. GPU checks, if needed after a dependency rebuild, use only physical GPU 2.
