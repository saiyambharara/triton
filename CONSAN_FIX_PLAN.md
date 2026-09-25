# Preserve mixed-consumer release frontiers

Branch: `keren/consan-mixed-consumer-frontiers`

Status: **Proposal only; no native/compiler implementation applied.**

Reproducer: [02-mixed-consumers](/mnt/keren/tmp/consan-unfixed-repros-20260925/02-mixed-consumers/README.md). Baseline results were verified on `831e8e2517`; runtime checks use only physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

## Evidence and diagnosis boundary

The ordinary-load-plus-MMA consumer kernel triggers `Buffer being accessed has outstanding reads` for both one-CTA and paired two-CTA warp specialization, with allocation poisoning disabled. Both uninstrumented numerical controls pass. This is not enough to classify the report as a false positive. A real synchronization bug remains possible; numerical success does not rule it out.

## Proposed fix

Target the handoff from a buffer's ordinary reader and its asynchronous MMA reader back to the producer. Preserve the union of both readers' release frontiers until the producer observes completion of the matching barrier phase. Do not replace one reader's frontier when another consumer signals the same barrier, and do not clear it before buffer reuse is synchronized.

Start by tracing the failing write's physical region, owner CTA set, reader slot, barrier phase, and producer/consumer execution region. Use compiler-generated diagnostics or captured instrumented IR, not source-level lane-varying scalars. Compare that trace with the generated aref barrier protocol. This diagnostic gate determines where the patch belongs:

- If the generated waits/arrivals order both reads correctly, repair frontier accumulation/transfer in `lib/Dialect/TritonInstrument/IR/FunctionBuilder.cpp`: `createTrackVisibleAccessesCall`, `createVerifyAndUpdateBarrierStateCall`, `createTransferVisibleAccessesCall`, and `createCompleteBarrierWaitCall`. Audit matching masks in `ConcurrencySanitizer.cpp`.
- If a required consumer release or producer wait is absent, fix the NVWS producer/consumer analysis and barrier protocol in `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp` and its analysis inputs. Do not weaken ConSan to accommodate a compiler race.

The proposed invariant is common to one and multiple CTAs. Preserve physical alias checks, execution-region identity, loop iteration/phase, predication, and asynchronous completion. Do not special-case away mixed consumers or disable the assertion. The exact faulty transition has not yet been isolated; this branch is a proposal, not a claimed implementation.

## Acceptance checks

After `make`, rerun both standalone cases on GPU 2 with ConSan and poisoning disabled to isolate the issue, then with default poisoning when the separate initialization fix is available. Extend the existing mixed-consumer test in `python/test/unit/language/test_matmul.py` only as needed. Include a valid negative ConSan case with a genuinely missing read-release edge, proving the fix still detects a race. Check two consecutive barrier phases and buffer reuse, not only the first iteration.
