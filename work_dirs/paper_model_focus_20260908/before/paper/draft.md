# State Preservation Is Not Output Equivalence: Conditional Error and Pipeline-Cost Analysis of Change-Gated Video Depth

> 최신 통합 영문 원고 · 2026-09-07. `paper/submission/manuscript.tex`에서 동기화한 전체 Markdown판입니다.
> [PDF](submission/manuscript.pdf) · [40항목 self-revision 대응표](self-revision/r3/revision-status.md) · [정정 기록](submission/REVISION_NOTES.md) · [과거 초안](draft_legacy_20260907.md).
> 저자 정보·독립 검토·권리 확인은 대기 중이며 투고 완료본이 아닙니다. 그림은 저장된 실측/평가 결과와 구현에서 생성했습니다.

## Authors

Author information to be completed

Affiliation and institutional address to be completed by the author.

Correspondence: to be completed by the author.

## Abstract

Conditional updates can preserve recurrent state exactly while approximating the output of a dense video model. We study this distinction in a fixed 4.19-million-parameter depth model with change-gated selective state-space blocks and output caches. We derive conditional defect-propagation, cache-age and refresh bounds and an additive pipeline-cost criterion. Counterexamples show why small pixel changes do not certify small dense-state errors. On four development sequences, same-state local defects vanish at keyframes while differences from a continuously dense history remain. The FP32 PNG-to-depth pipeline takes 7.496 ms per frame for the sparse path and 7.521 ms for its dense counterpart; input processing accounts for 69.4% of a separate synchronized profile. A frozen evaluation on four previously reserved, same-scene TUM sequences gives L256 scale–shift AbsRel 0.1869 for the reference sparse model, 0.1837 for dense carry, and 0.1849 for MiDaS Small. The study does not establish general sparse acceleration, scene-disjoint generalization, or a global accuracy certificate. Its contribution is an implementation-specific separation of exact preservation, approximation error and paid computation.

## Introduction

Video depth inference often processes similar images repeatedly. Change-based computation offers a way to reuse work, but several distinct claims are easily conflated: copying an internal state exactly, approximating a dense network accurately, producing correct depth, and reducing wall-clock time. None follows automatically from another.

We examine these distinctions using a fixed SOKKANAEM checkpoint. A binary pixel-change mask multiplies the discretization step of each temporal selective SSM. An inactive step is an identity on hidden state; optional temporal and spatial output caches introduce additional approximations. The objective is not to establish a new accuracy leader, but to determine what the implementation and its measurements actually support.

The contributions are (i) an implementation-aligned conditional error analysis with explicit counterexamples; (ii) same-checkpoint interventions separating state carry, gating, readout reuse and spatial context; (iii) a measured input-to-output cost decomposition; and (iv) a frozen, limited transfer evaluation after development-based model selection. The algebraic inequalities are applications of established perturbation reasoning, not new universal stability theorems. Author and independent mathematical review remain required before submission of this working manuscript.

## Related Work and Scope of Novelty

Skip RNN explicitly uses a binary update/copy operation [\[3\]](#ref-campos2018). Mamba introduces input-selective state-space dynamics and hardware-aware scanning [\[4\]](#ref-gu2024). We do not claim either binary preservation or input selection as a first contribution. DeltaCNN propagates sparse frame differences through CNN operators [\[5\]](#ref-deltacnn), and Eventful Transformers reuse computation based on changing tokens [\[6\]](#ref-eventful). Change-based reuse itself is therefore also established. Spiking SSMs such as SPikE-SSM [\[7\]](#ref-spikessm) and SpikySpace [\[8\]](#ref-spikyspace) investigate sparse dynamics in sequence learning and forecasting. Their architectures and tasks differ from dense RGB video-depth readout, but they preclude a broad claim of the first sparse or event-driven SSM.

The distinction investigated here is the combination of externally selected patch updates, selective SSM states, separate output caches, and measured depth/pipeline error. The literature comparison is targeted, not an exhaustive priority search. We do not report cross-task numerical comparisons against those methods. DPT [\[9\]](#ref-dpt), MiDaS [\[10\]](#ref-midas) and Depth Anything V2 [\[11\]](#ref-da2) provide independent-image depth references, not a complete benchmark of causal and noncausal video-depth systems. Video Depth Anything (VDA) [\[1\]](#ref-vda) adds a spatial–temporal depth head; online VDA [\[2\]](#ref-ovda) is a distinct online formulation. Historical VDA scores in this repository do not have frame-ID and weight/source-hash linkage sufficient for inclusion in the frozen comparison. The audited historical wrapper calls a 32-frame offline, unmasked temporal-attention path, not frame-causal inference. The supplementary audit preserves those results separately. A matched modern video-depth baseline remains a limitation, not a completed comparison.

## Model and Conditional Analysis

### Frozen implementation

The model uses a stride-16 RGB embedding, four alternating temporal/spatial blocks, 192-dimensional tokens, inner dimension 384, state dimension 16, four-direction spatial scans, local spatial convolution, and a dense DPT-style binned depth decoder. It has 4,185,872 parameters. The baseline is the seed-0 EMA checkpoint at 256 pixels, with pixel-MSE hysteresis thresholds 0.05/0.025, keyframe interval 30, both output caches, and an all-active fallback above activity 0.4. GMC is disabled in the primary model. The recorded training ancestry contains 60k, 60k, 25k and 8k stage steps; this is not an independently verified optimizer-update count or evidence of teacher-free ancestry.

![Figure 1. Frozen-model schematic](submission/figures/r3_architecture.png)

Figure 1. Frozen-model schematic. Binary inactive temporal steps copy hidden state; readout reuse and spatial-context omission remain separate approximations. The diagram distinguishes these claims from measured pipeline cost, and does not specify kernel execution order.


### What preservation guarantees

For a shared token input, write $F_t(h)=\Phi_t h+q_t$, with $\Phi_t=\operatorname{diag}(\exp(-\lambda_j\Delta_{t,j}))$ and $q_{t,j}=\Delta_{t,j}(B_tx_t)_j$. Binary gating implements $\widetilde h_t=M_tF_t(\widetilde h_{t-1})+(I-M_t)\widetilde h_{t-1}$. Zero step preserves state, but the input term is a first-order approximation to exact ZOH. Current-input-dependent readout and spatial-context omission need separate analysis.

With $e_t=\widetilde h_t-h_t$ and $r_t=(I-M_t)((I-\Phi_t)\widetilde h_{t-1}-q_t)$, the exact shared-input identity is $e_t=\Phi_te_{t-1}+r_t$. Products of transition norms bound the propagated initial error and injected defects. A uniform geometric bound additionally requires a uniform contraction and a valid defect bound. Different deep-layer token trajectories add parameter/input perturbations; a negative diagonal $A$ alone does not certify the full network.

The first temporal block admits a conditional cache bound $L_R\lVert W\rVert a\sqrt{Cp^2\tau_{\rm on}}$ after $a$ inactive steps, provided the state-fixed readout has the stipulated Lipschitz constant. It is not a tau-only depth guarantee. Keyframes bound cache age by $K-1$ and remove the local approximation at the same incoming state, but they do not recover the state of a continuously dense history. Full assumptions, proofs, counterexamples and the no-fit AbsRel connection are in Appendix A.

### Paid computation

Under the explicitly additive model $C_d=F+V$ and $\overline C_s=F+\overline a_{\rm eff}V+\overline H$, normalized speedup is $[f+(1-f)\overline a_{\rm eff}+\omega]^{-1}$. Overhead must be smaller than saved variable cost for acceleration. This is Amdahl-style reasoning [\[12\]](#ref-amdahl1967), not a conversion of MAC reductions into observed time. Keyframes and fallback events are counted as a union. At 256 frames and $K=30$, even a static stream has nine keyframes. The analysis links two constraints on refresh: a certified cache-only tolerance would impose an upper bound on $K$, whereas a target speedup in the fixed-cost model imposes a lower bound. Their intersection is a conditional design check, not a selected operating point or an accuracy certificate. The constants needed for a useful numerical cache guarantee have not been certified for this network.

## Evaluation Protocol

### Development and reserved transfer sequences

The previously reused TUM/Bonn holdout is development data, not an independent test. It contains 488 L8 clips, of which 487 have scorable GT, and 13 L256 clips (3328 frames) from four sequences. The final evaluation uses four reserved TUM Freiburg3 sequences: sitting_xyz, sitting_rpy, walking_xyz and walking_rpy. The frozen lists contain 462 L8 clips and 13 L256 clips. These are new sequences in the same capture setting; they do not establish scene-disjoint or fixed-camera generalization. L32 is not evaluated in this final protocol. The lengths share underlying footage and are not independent samples. RGB/depth timestamps are paired within 20 ms, with possible reuse of a depth frame. Sensor validity, resize/crop and depth scale follow the frozen evaluation contract and the TUM data documentation [\[13\]](#ref-tumdata); Bonn supplies dynamic indoor development sequences [\[14\]](#ref-bonndata).

### Comparisons and scoring

Final conditions were fixed before reserved predictions: K30, K5, dense carry, dense reset, output hold, MiDaS v2.1 Small at 256, DA2 Small at common target 256 (actual input 252), and DPT-Large at common 256. No model was retrained or promoted after test results. Native controls share the frozen weights; external baselines have different training data and budgets, whose overlap is not fully known. DPT’s four missing checkpoint parameters belong to an unused branch; execution hooks reject any use of that branch.

All models score the same 256-pixel ROI. Native metric outputs have a no-GT-fit panel; relative models are not relabeled as metric. Median and disparity scale–shift alignment are separate panels. The main shape comparison uses one fit per clip. Frame-fit L256 scores are supplementary shape diagnostics without temporal scores. Nonpositive fitted disparity is counted as an alignment failure; the resulting errors remain in aggregates. The frozen inverse-disparity conversion uses $1/\max(d,10^{-3})$, so an invalid fitted disparity can produce a 1000-unit fallback depth. This is not a physical depth estimate; it explains the sensitivity of RMSE to failed fits and is reported rather than removed. Nonfinite predictions fail execution. No-GT clips are explicitly counted, not substituted. Optical-flow-based temporal error uses a shared RAFT estimate and the frozen scoring equations; flow and GT are never fed to a predictor. Dynamic masks are a flow-residual heuristic, not semantic annotations.

### Statistics and timing

Source pixel-pooled and equal-sequence means are reported separately. Paired sequence bootstrap enumerates $4^4$ ordered resamples, while sign flipping enumerates $2^4$ cases. The minimum possible two-sided sign-flip $p$ is 0.125. Intervals are exploratory, unadjusted for multiple comparisons, and not finite-sample coverage guarantees. Frames, neighboring clips and repeated timing runs are not independent sequence replicates. The three existing training seeds differ only in the final 8k stage from a common parent.

Timing is development-only: the first L256 clip of each development sequence, 1024 matched frames, batch one, FP32 eager, TF32 off, RTX 4090, one complete warmup and five repeats. The boundary includes warm-cache PNG read/decode, crop/normalization, transfer, actual detector/fallback/keyframes, model, interpolation and CPU output copy. Model loading, GT, flow, fitting, scoring and output-file writes are excluded. A separate synchronized pass measures components. No final-test accuracy is paired with development latency as a same-quality speedup. No edge timing or energy claim is made.

## Results

### Frozen final transfer evaluation

| Model           | AbsRel |   RMSE | $\delta_1$ |    TCE | Fail |
|:----------------|-------:|-------:|-------------:|-------:|-----:|
| Sparse K30      | 0.1869 | 1.3742 |       0.7382 | 0.0352 | 0/13 |
| Sparse K5       | 0.1828 | 1.3531 |       0.7468 | 0.0357 | 0/13 |
| Dense carry     | 0.1837 | 1.3488 |       0.7465 | 0.0328 | 0/13 |
| Dense reset     | 0.1909 | 1.3778 |       0.7412 | 0.0348 | 0/13 |
| Output hold     | 0.1955 | 1.3877 |       0.7332 | 0.0345 | 0/13 |
| DA2 Small 252   | 0.2166 | 2.1519 |       0.6686 | 0.0709 | 2/13 |
| MiDaS Small 256 | 0.1849 | 1.3476 |       0.7369 | 0.0640 | 0/13 |
| DPT-Large 256   | 0.1894 | 1.3559 |       0.7201 | 0.0690 | 0/13 |

Table 1. Final TUM L256: clip scale–shift alignment, source pixel-pooled. AbsRel, RMSE and TCE lower is better; $\delta_1$ higher is better. Fail is the number of clips with any invalid fitted disparity; failures remain in aggregates.

| Model           | AbsRel |    RMSE | $\delta_1$ |    TCE |   Fail |
|:----------------|-------:|--------:|-------------:|-------:|-------:|
| Sparse K30      | 0.1840 | 10.1931 |       0.8017 | 0.0578 |  7/462 |
| Sparse K5       | 0.1837 | 10.1759 |       0.8032 | 0.0590 |  6/462 |
| Dense carry     | 0.1946 | 10.9170 |       0.8020 | 0.0572 |  7/462 |
| Dense reset     | 0.1669 |  8.6731 |       0.7979 | 0.0503 |  8/462 |
| Output hold     | 0.1677 |  8.5613 |       0.7965 | 0.0484 |  8/462 |
| DA2 Small 252   | 0.2074 | 16.9349 |       0.8305 | 0.1285 | 45/462 |
| MiDaS Small 256 | 0.1539 |  9.5211 |       0.8270 | 0.0716 | 22/462 |
| DPT-Large 256   | 0.1810 | 10.4890 |       0.8402 | 0.1004 |  9/462 |

Table 2. Final TUM L8: clip scale–shift alignment, source pixel-pooled. AbsRel, RMSE and TCE lower is better; $\delta_1$ higher is better. Fail is the number of clips with any invalid fitted disparity; failures remain in aggregates.

These panels measure aligned relative shape, not calibration-free metric depth. The selected K30 model is retained irrespective of its test ranking. All 13 L256 and 462 L8 clips were processed; 0 clips had no scorable GT.

| Gauge      | AbsRel |   RMSE | $\delta_1$ |    TCE |
|:-----------|-------:|-------:|-------------:|-------:|
| none       | 0.2014 | 1.3372 |       0.7553 | 0.0468 |
| median     | 0.2094 | 1.3231 |       0.7416 | 0.0480 |
| scaleshift | 0.1869 | 1.3742 |       0.7382 | 0.0352 |

Table 3. Final L256 native K30: alignment gauges must not be conflated.

| Comparator      | Difference |         95% interval | $p$ |
|:----------------|-----------:|---------------------:|------:|
| Dense carry     |    +0.0023 | \[-0.0052, +0.0094\] | 0.625 |
| Sparse K5       |    +0.0035 | \[-0.0007, +0.0082\] | 0.375 |
| MiDaS Small 256 |    +0.0054 | \[-0.0292, +0.0359\] | 0.625 |
| DPT-Large 256   |    +0.0011 | \[-0.0398, +0.0430\] | 0.750 |

Table 4. Exploratory equal-sequence L256 AbsRel differences (K30 minus comparator), $n=4$. Unadjusted percentile sequence-bootstrap intervals; exact sign flips.

### Development diagnostics and pipeline cost

Across 1024 development frames, all 32 post-initial keyframes have zero same-state local defect. Nevertheless, the median concatenated hidden/cache state RMS differences at those keyframes are 0.932, 0.490, 0.721, 1.069 in manifest sequence order. These are internal-state units, not depth errors. Cache age never exceeds 29. The FP64 first-block shared-input shadow bound holds at all 1024 steps; this does not certify FP32 end-to-end error. Full sequence names and frame traces are in the supplement. On the four matched development clips, sparse K30 takes 7.496 ms/frame versus 7.521 for dense carry, a 0.33% reduction. MiDaS Small takes 9.439 ms with matched-subset AbsRel 0.1637 versus 0.1630 for K30. That operating-point comparison is not statistical equivalence: on all 13 development L256 clips, K30/MiDaS AbsRel is 0.2091/0.1749. The native sparse stream uses 13.501 MiB of persistent state versus 12.000 MiB dense. Read/decode/preprocessing accounts for 69.4% of a separate synchronized profile. The associated idealized fixed-input ceiling is 1.44 times that profile, not a measured speedup. Existing final-stage seed replicates give source-balanced scale–shift AbsRel $0.1146\pm0.0007$ on development L8 and $0.2040\pm0.0077$ on development L256 (mean and sample SD, three seeds sharing the earlier training stages).

![Figure 2. Development-only extended-state diagnostics](submission/figures/development_state_diagnostic.png)

Figure 2. Development-only extended-state diagnostics. Dotted lines mark keyframes. The same-state local defect vanishes at refresh, while the difference from the separately evolved dense history persists. Curves are RMS over concatenated hidden states and caches, not GT depth error or uniquely allocated memory.


### Post-hoc scale-variation diagnostic

To address the earlier self-review, we summarize statistics already stored during final evaluation. This analysis was added after test access, not preregistered as a primary endpoint. It includes all eight frozen models and both lengths without tuning or new inference. For each valid frame, $s_t=\operatorname{median}(G_t)/\operatorname{median}(D_t)$ uses the same GT-valid pixels. We report the within-clip sample coefficient of variation $\operatorname{sd}(s_t)/\overline{s}$, sample standard deviation of $\log s_t$, and mean absolute log difference between consecutive retained valid frames. The scorer skips frames without valid GT and returns zero if fewer than two remain; these conventions are unchanged. Native models use no-fit depth; relative models use their median-gauge inverse-disparity depth, not an affine-fit depth. Away from numerical clamp activation, a positive constant clip scaling cancels from these statistics; an affine disparity shift does not. The stored scorer clamps the predicted median at $10^{-6}$, the scale factor at $10^{-12}$ and the CV denominator at $10^{-6}$.

| Model           |  CV L8 | CV L256 |   L256 CV interval | Log-SD | Log-step |
|:----------------|-------:|--------:|-------------------:|-------:|---------:|
| Sparse K30      | 0.0312 |  0.2185 | \[0.0770, 0.3601\] | 0.1800 |   0.0180 |
| Sparse K5       | 0.0318 |  0.2131 | \[0.0677, 0.3585\] | 0.1752 |   0.0183 |
| Dense carry     | 0.0312 |  0.2101 | \[0.0642, 0.3560\] | 0.1723 |   0.0169 |
| Dense reset     | 0.0355 |  0.2463 | \[0.0677, 0.4248\] | 0.1903 |   0.0235 |
| Output hold     | 0.0338 |  0.2466 | \[0.0683, 0.4250\] | 0.1905 |   0.0218 |
| DA2 Small 252   | 0.0744 |  0.2477 | \[0.1753, 0.3201\] | 0.2418 |   0.0696 |
| MiDaS Small 256 | 0.0603 |  0.1892 | \[0.1104, 0.2680\] | 0.1760 |   0.0559 |
| DPT-Large 256   | 0.0774 |  0.2554 | \[0.1613, 0.3496\] | 0.2219 |   0.0704 |

Table 5. Post-hoc scale statistics: clip means within each sequence, then four-sequence means. The CV interval enumerates sequence-bootstrap resamples; it is exploratory and unadjusted. Full intervals and gauge/failure counts are supplied as CSV.

A smaller scale statistic alone does not establish better depth, and per-frame fitting can hide scale errors that a metric deployment still incurs. L8 and L256 use partly different footage and reset schedules, so their difference is not an isolated causal effect of history length. Local shape changes can also affect the scale medians. These summaries are not an additive decomposition of AbsRel into scale and shape terms. Confidence intervals containing a null difference do not establish equivalence.

![Figure 3. All eight frozen paths on L8 and L256, using the source pixel-pooled clip scale–shift AbsRel estimator of the main tables](submission/figures/r3_accuracy.png)

Figure 3. All eight frozen paths on L8 and L256, using the source pixel-pooled clip scale–shift AbsRel estimator of the main tables. Zero-based axes and all failed fits are retained. These point estimates have no sequence uncertainty bars: the exploratory paired sequence estimator is reported separately. The two lengths share some footage but are not identical history interventions.


![Figure 4. Development-only measured cost for five representative operating points](submission/figures/r3_cost.png)

Figure 4. Development-only measured cost for five representative operating points. Latency includes the same PNG-to-depth boundary on four matched L256 clips; whiskers show the minimum/maximum of five repeat means, not confidence intervals. Persistent stream memory excludes weights and transient buffers; zero state in an image model does not mean zero total memory. Final-test accuracy is not used to claim matched-quality acceleration.


![Figure 5. Post-hoc scale diagnostics for all eight frozen L256 paths](submission/figures/r3_scale.png)

Figure 5. Post-hoc scale diagnostics for all eight frozen L256 paths. Four-sequence means and exploratory, unadjusted percentile sequence-bootstrap intervals; clip means are averaged within each sequence first. Native predictions use no-fit depth, while relative predictions use median-gauge depth. CV describes within-clip variation and log-step describes short-step variation; neither is a substitute for depth accuracy or proof of general temporal superiority.


## Discussion and Limitations

The central distinction is between preservation and approximation. A constant input can leave an ungated SSM approaching equilibrium while a skipped state remains frozen. Similarly, recomputing every cache entry on a keyframe does not erase the history carried by temporal states. Local defect elimination and global trajectory synchronization are different operations, as the development diagnostic demonstrates.

The shared-input bound is useful for identifying omitted innovations, but an observed small shadow-model residual does not certify the complete FP32 depth network. The pixel threshold does not control all deeper features, and the spatial subsequence path omits context even at active coordinates. Global Lipschitz constants have not been certified. Synthetic checks and this internal derivation audit do not replace mathematical review by the authors and an independent domain expert.

The external efficiency comparison also cannot isolate architecture from training. MiDaS Small and DA2 Small are larger than the native network, and we have not covered all similarly sized CNNs or modern causal video baselines. The present study supports bounded, implementation-specific statements rather than broad accuracy or efficiency superiority. More independent scenes and full-training replicates would be needed for stronger generalization and stability conclusions. Test access was a one-time finalization stage; any subsequent development would require a new untouched evaluation set. Our 1024 development diagnostic frames comprise four separate L256 clips, not one continuous L1024 trajectory. The older L1024 evidence concerns v9/v10, not the present v11; its PointOdyssey aggregate scores and frame curves use 20 and 8 clips, respectively. Nearly equal endpoints cannot establish stability throughout a trajectory. We therefore do not inherit the earlier continuous-stream or scale/shape decomposition claims. Current-checkpoint qualitative sequences and longer continuous evaluation remain useful extensions.

## Conclusions

Change-gated temporal state copying is exact in its algebraic scope, whereas readout reuse, spatial-context omission and end-to-end speed are separate questions. Conditional defect bounds and refresh analysis clarify these boundaries; matched implementation experiments show why low activity alone is insufficient. The evidence is best understood as an analysis of when claims hold and where they stop, not as a universal acceleration or accuracy certificate.

## Declarations

### Supplementary materials

The accompanying package contains complete per-model/per-sequence tables, frame-level development diagnostics, derivations, source hashes and reproduction commands. Raw third-party datasets and baseline weights are obtained from their original providers.

### Author contributions

AUTHOR TO COMPLETE: actual contributions and approval by every author.

### Funding

AUTHOR TO COMPLETE: funding sources and grant identifiers, or a verified no-funding statement.

### Institutional review

AUTHOR TO CONFIRM: applicability of institutional review and dataset-use permissions.

### Informed consent

AUTHOR TO CONFIRM: applicability of consent requirements for the reused datasets.

### Data availability

Derived results, code and manifest hashes are included in the local reproducibility package. Public repository URL/DOI and redistribution permissions must be supplied and confirmed by the authors before submission. Original TUM/Bonn data and external weights remain subject to their providers’ terms.

### Acknowledgments and AI assistance

AI assistance was used for code development, experiment orchestration, mathematical drafting, analysis and manuscript preparation. The authors must supply tool/version details, review and edit all outputs, and confirm responsibility before submission. This working draft does not assert that such author review has already occurred.

### Conflicts of interest

AUTHOR TO COMPLETE: actual competing interests or a verified absence thereof.

## Appendix A. Analytic Proofs

### Implemented recurrence and zero step

Vectorize the inner-channel and state coordinates of one temporal SSM. For a shared exogenous token sequence, write

$$
F_t(h)=\Phi_t h+q_t,\qquad
 \Phi_t=\operatorname{diag}(e^{-\lambda_j\Delta_{t,j}}),\quad
 q_{t,j}=\Delta_{t,j}(B_tx_t)_j,
$$

where $\lambda_j>0$ and $\Delta_{t,j}\ge0$. A channel’s step size is repeated over its state coordinates. The code uses $A=-\exp(A_{\log})$ and a softplus step size. Norms are Euclidean and induced operator norms unless specified.

##### Proposition 1 (Binary zero-step identity).

For a diagonal binary mask $M_t$, multiplying the step size by the mask gives

$$
\widetilde h_t=M_tF_t(\widetilde h_{t-1})+(I-M_t)\widetilde h_{t-1}.
$$

Thus each inactive hidden-state coordinate is unchanged in real arithmetic.

##### Proof.

On an active coordinate the original update is used. On an inactive coordinate the transition is $e^0=1$ and the input term is zero. These cases give the stated expression. Fractional masks do not generally have this update/copy interpretation because $e^{m\Delta A}\ne m e^{\Delta A}+1-m$. $\square$

The input term is first order, not exact zero-order hold (ZOH). With the input held constant, the ZOH coefficient is $(1-e^{-\lambda\Delta})/\lambda$; see the ZOH definition in Mamba [\[4\]](#ref-gu2024). Its difference from the implemented coefficient obeys

$$
\left|\Delta b-\frac{1-e^{-\lambda\Delta}}{\lambda}b\right|
 \le \frac{\lambda\Delta^2}{2}|b|.
$$

Indeed, the coefficient difference is $\int_0^\Delta(1-e^{-\lambda s})\,ds\le\int_0^\Delta\lambda s\,ds$. This local discretization error is distinct from skipping error: both paths below use the same implemented recurrence. State identity also does not imply readout identity, since the current input affects $C,x,z,D$ and the residual. Finite CPU/CUDA tests supplement the real-arithmetic statement; they do not establish bitwise identity for exceptional values or all floating-point scan reassociations.

### Skipping defects and state error

##### Theorem 1 (Shared-input defect propagation).

Let $h_t=F_t(h_{t-1})$ and let $\widetilde h_t$ satisfy (the binary-gate equation), using the same $\Phi_t,q_t$. Define

$$
e_t=\widetilde h_t-h_t,\qquad
 r_t=(I-M_t)((I-\Phi_t)\widetilde h_{t-1}-q_t),\qquad d_t=\lVert r_t\rVert.
$$

Then $e_t=\Phi_t e_{t-1}+r_t$. If $\lVert\Phi_t\rVert\le\rho_t$, then

$$
\lVert e_T\rVert\le\left(\prod_{j=1}^T\rho_j\right)\lVert e_0\rVert
 +\sum_{i=1}^T\left(\prod_{j=i+1}^T\rho_j\right)d_i.
$$

Empty products equal one. In particular, if $\rho_t\le\rho<1$ and $d_t\le\epsilon$, the bound is $\rho^T\lVert e_0\rVert+\epsilon(1-\rho^T)/(1-\rho)$.

##### Proof.

Add and subtract $\Phi_t\widetilde h_{t-1}+q_t$ in the gated update and subtract the dense update. Iteration gives a sum of propagated defects. Submultiplicativity and the triangle inequality give (the defect-propagation bound); the uniform case follows by summing a geometric series. $\square$

A uniform lower bound $\lambda_j\ge\lambda_{\min}>0$ and $\Delta_{t,j}\ge\Delta_{\min}>0$ would yield $\rho=e^{-\lambda_{\min}\Delta_{\min}}<1$ for the ungated comparison transition. The actual gated transition equals one on skipped coordinates. Without a useful uniform contraction bound, the nonexpansive estimate $\lVert e_T\rVert\le\lVert e_0\rVert+\sum_i d_i$ remains available.

##### Counterexample: zero input change.

For $F(h)=h/2+1$, equal zero initial states and all updates skipped, $\widetilde h_t=0$ while $h_t=2(1-2^{-t})$, despite constant inputs. The error at $t=4$ is $1.875$. An all-active first frame does not remove the subsequent discrepancy. Thus a small pixel-change threshold alone does not bound the omitted innovation, even when the input change is exactly zero.

##### Different token trajectories.

If the gated branch uses $\widetilde\Phi_t,\widetilde q_t$, define its skip defect using those parameters. The error recurrence acquires the extra term $(\widetilde\Phi_t-\Phi_t)\widetilde h_{t-1}+\widetilde q_t-q_t$. Its norm is at most $\lVert\widetilde\Phi_t-\Phi_t\rVert\lVert\widetilde h_{t-1}\rVert
+\lVert\widetilde q_t-q_t\rVert$. This term cannot be dropped for deeper blocks whose inputs depend on previously approximated features.

##### Theorem 2 (Whole-network conditional perturbation).

Fix the frame sequence and the masks selected by the sparse trajectory. Let $z$ comprise all temporal states and block output caches, with the dense comparison overwriting an identically shaped set of caches at every step. Let $F_t$ be the all-active network step and $G_t$ the sparse step. Put $d_t^{\rm net}=\lVert G_t(\widetilde z_{t-1})-F_t(\widetilde z_{t-1})\rVert$. If $F_t$ is $L_t$-Lipschitz on a region containing both states, then

$$
E_t^{\rm net}\le L_tE_{t-1}^{\rm net}+d_t^{\rm net}.
$$

Equation (the defect-propagation bound) holds with $L_t,d_t^{\rm net}$ in place of $\rho_t,d_t$. For dense and sparse output maps $P_t,\widetilde P_t$, if $P_t$ is $L_{P,t}$-Lipschitz and $\eta_t=\lVert\widetilde P_t(\widetilde z_{t-1})-P_t(\widetilde z_{t-1})\rVert$, then $\lVert\widetilde y_t-y_t\rVert\le L_{P,t}E_{t-1}^{\rm net}+\eta_t$.

##### Proof.

Add and subtract $F_t(\widetilde z_{t-1})$, or respectively $P_t(\widetilde z_{t-1})$, and apply the assumed Lipschitz inequality. Iterate the state bound. $\square$

No whole-network $L_t<1$ or practically small global constant is certified here. Negative diagonal $A$ alone is insufficient: the elementary map $F(h)=0.5h+u(h)$ with $u(h)=h$ expands by $1.5$. This is a counterexample to an inference about feedback, not an identification of v11 with that map. Computing a counterfactual dense defect also costs work; the detector does not provide it as a free online certificate. Threshold-induced mask changes under perturbed images are outside this fixed-mask comparison.

For $n$ fixed pixels with ground truth $d_i\ge d_*>0$, the reverse triangle inequality gives the additional, no-fit statement

$$
|\operatorname{AbsRel}(\widetilde y,d)-\operatorname{AbsRel}(y,d)|
 \le \frac1n\sum_i\frac{|\widetilde y_i-y_i|}{d_i}
 \le \frac{\lVert\widetilde y-y\rVert_1}{nd_*}.
$$

This does not remove the dense reference’s own ground-truth error. Separately refitted scale–shift alignment requires conditioning assumptions of its own; threshold accuracy metrics require margin assumptions. No new depth cutoff has been introduced into the evaluation protocol.

### Cached readout and refresh

##### Proposition 2 (Conditional first-block cache bound).

Consider a pixel-gated patch with $C p^2$ pixel values that remains inactive for $a=t-s$ steps after its last update. Suppose its embedding is $u=WI+b$ and the state-fixed block readout $R(u,h_s)$ is $L_R$-Lipschitz in $u$ on the relevant region. Then

$$
\lVert R(u_t,h_s)-R(u_s,h_s)\rVert
 \le L_R\lVert W\rVert\,a\sqrt{Cp^2\tau_{\rm on}}.
$$


##### Proof.

Inactivity implies patch MSE at most the applicable hysteresis threshold, which is at most $\tau_{\rm on}$. Dilation and forced activation only turn zeros into ones and preserve this implication for a final zero mask. Each consecutive patch difference is at most $\sqrt{Cp^2\tau_{\rm on}}$. Sum these differences, apply the embedding norm and then the readout bound. The cached readout at $s$ used the already updated state $h_s$, which remains unchanged during the inactive interval. $\square$

The readout includes normalization, residual and input-dependent projections. This first-block, pixel-MSE statement does not directly apply to deeper-layer tokens or GMC relative-feature scores. Spatial active-subsequence computation has an additional context defect $\lVert S_A(u_A)-[S(u)]_A\rVert$, even on active tokens. For example, the spatial recurrence $s_i=\rho s_{i-1}+b_i$ with $b_1=1,b_2=0,s_0=0$ produces $s_2=\rho$; omitting token 1 gives zero, even if the current inputs equal past inputs. Current GMC warps the comparison image, not the hidden-state or output caches.

##### Theorem 3 (Refresh is not state reset).

An all-active keyframe makes $G_t=F_t$ at the same incoming extended state; hence $d_t^{\rm net}=0$, but generally $E_t^{\rm net}\le L_tE_{t-1}^{\rm net}\ne0$. For the shared-input SSM of Theorem 1, assume a common $\rho<1$, skip defects at most $\epsilon$, and a refresh every $K$ steps. If $B_n$ is the error immediately after refresh $n$, then

$$
\begin{aligned}
 B_{n+1}&\le\rho^KB_n+\rho\epsilon\frac{1-\rho^{K-1}}{1-\rho},\\
 \limsup_{n\to\infty}B_n&\le
 \frac{\rho\epsilon(1-\rho^{K-1})}{(1-\rho)(1-\rho^K)}.
\end{aligned}
$$


##### Proof.

All-active branches recompute block outputs but pass the carried temporal states to the recurrence. Thus only the same-state local defect vanishes. For the scalar norm recurrence, propagate $K-1$ bounded defects and apply the final zero-defect refresh. Each injected term receives at least one factor $\rho$. Sum the resulting cycle recurrence as a geometric series. $\square$

At $j\le K-1$ steps after a refresh, the bound is $\rho^jB_n+\epsilon(1-\rho^j)/(1-\rho)$. Actual errors need not be monotone in $K$: changing $K$ also changes the trajectories and defects. Keyframes bound cache age by $K-1$, not dense-history error by zero. Resetting only one path to zero is a different intervention. True state synchronization would require the reference state or replay. Rolling refresh bounds per-patch age but is not an all-active whole-network refresh.

### Fixed cost, refresh frequency and overhead

##### Proposition 3 (Conditional pipeline speedup).

Assume an additive cost model with dense reference $C_d=F+V>0$ and mean sparse cost $\overline C_s=F+\overline a_{\rm eff}V+\overline H$, where $F,V,\overline H\ge0$ and $\overline C_s>0$. Define $f=F/C_d$ and $\omega=\overline H/C_d$. Then

$$
S=\frac{C_d}{\overline C_s}
 =\frac1{f+(1-f)\overline a_{\rm eff}+\omega},\qquad
 S>1\ \Longleftrightarrow\ \omega<(1-f)(1-\overline a_{\rm eff}).
$$

For $f>0$, $S\le1/f$.

##### Proof.

Divide by $C_d$ and compare the positive denominator with one. Dropping its other nonnegative terms yields the upper bound. $\square$

This is an application of Amdahl-style fixed-cost reasoning [\[12\]](#ref-amdahl1967), not a new universal speed law. Detector, gather/scatter, cache-copy and launch overheads must be counted without double counting. Keyframe and fallback events are a union: $a_{{\rm eff},t}=1$ on either event, and otherwise equals the normal activity. Padding and nonlinear kernel costs must be modeled separately if active fraction does not scale variable work linearly.

For $T$ frames starting at frame index zero, the keyframe fraction is $\lceil T/K\rceil/T$, even on a static stream. At $T=256,K=30$ this is $9/256$, not exactly $1/30$. In a long stream without fallback and with constant nonkeyframe mean $a$, the effective activity tends to $a+(1-a)/K$. If a verified cache-only coefficient $c=L_R\lVert W\rVert\sqrt{Cp^2\tau}>0$ and tolerance $E_c$ are available, (the first-block cache bound) gives the sufficient condition $K\le1+\lfloor E_c/c\rfloor$. In the nondegenerate fixed-cost model, a target speedup $S_*>1$ requires

$$
D=1/S_*-f-(1-f)a-\omega>0,\qquad
 K\ge\left\lceil\frac{(1-f)(1-a)}{D}\right\rceil.
$$

Nonoverlap means these sufficient error conditions cannot certify a feasible $K$; it is not a proof that no accurate fast policy exists.

The existing K30 synchronized profile assigns about $69.4\%$ of its time to image read/decode/preprocessing. Holding that component fixed and deleting all remaining profiled time gives an idealized ceiling of about $1.44\times$ relative to that same profile. This is neither a measured speedup nor a bound for a redesigned input path or GPU-resident pipeline. Main matched-subset latencies remain $7.496$ ms sparse and $7.521$ ms dense. Persistent temporal states are densely allocated:

$$
bBN\sum_{\ell\in\mathrm{temporal}}P_\ell S_\ell
$$

bytes, independent of activity. For v11 this is $12$ MiB in FP32; output caches add storage.

## Appendix B. Reproduction Details and Evidence Boundaries

### Checkpoint, training record and interventions

The archived effective configuration beside the seed-0 v11 checkpoint records 8000 final-stage steps, clip length 24, batch size 2, image size 256, learning rate $3\times10^{-4}$, 1000 warmup steps, gradient-norm clipping at 1 and EMA decay 0.999. The checkpoint resumes partially from v10. The recorded loss weights are multiscale gradient 0.5, normal 0.05, warp 2, spread 0.5, edge 2, and bin classification 0.2. Teacher and feature-distillation weights are zero in this stage only. Training uses random masks with maximum skip fraction 0.5; the image-change detector is not used to generate these training masks.

The archived training implementation uses AdamW (default weight decay 0.01), warmup followed by cosine learning-rate decay, and source-balanced weighted sampling over TUM, Bonn, VKITTI2, TartanAir2 and PointOdyssey. Clip-consistent augmentation comprises zoom/crop with scale 0.55–1.0, horizontal flip with probability 0.5, brightness/contrast factors 0.75–1.3, saturation 0.7–1.4, and gamma 0.8–1.25. The effective v11 sidecar records augmentation enabled and the dataset paths, but not all of these sampler/optimizer implementation details or the RNG trajectory. These are reproduction settings read from the archived implementation, not an independently reconstructed historical run. The sidecar’s recorded source commit is `0a3f5e06abbf448898a65c0a42c53534531912e5`. The four-stage ancestry, rather than the earlier three-stage summary, must be used when describing training cost. Full-training seed variation remains unknown.

Only one reference operating configuration is selected: K30, thresholds 0.05/0.025, fallback 0.4, both caches on and GMC off. K5, dense carry, dense reset and output hold are frozen diagnostic controls, not post-test deployment recommendations. The development cache/readout interventions are available in the study-5–8 tables. Historical threshold/fallback sweeps remain in the older draft’s Table 9, not a newly validated frozen-protocol sensitivity panel. None of these should be mixed with final-test scores. A token-drop control zeroes the inactive temporal residual contribution (retaining the identity bypass and frozen hidden state); it is not equivalent to dropping input video frames. No universal accuracy gain over that control is claimed.

### Temporal-score equations and masking

Let $D_t$ be depth after the declared clip gauge and $G_t$ ground truth. RAFT Small uses RGB mapped from $[0,1]$ to $[-1,1]$ at the common evaluation resolution and its final refinement output, with the default pretrained weights resolved by the frozen dependency record. Flow maps the current frame to the preceding frame. Let $W_t$ warp that preceding frame with nearest-neighbor sampling, border padding and aligned grid corners. The common mask $m_t$ is the product of the in-bounds indicator, current GT validity and warped previous GT validity. There is no forward–backward flow-consistency or semantic occlusion filter. With $N_m=\max(1,\sum_{t,i}m_{t,i})$,

$$
\begin{aligned}
 \mathrm{OPW}&=\frac{1}{N_m}\sum_{t,i}m_{t,i}
  \frac{|(W_tD_{t-1})_i-D_{t,i}|}{\max(G_{t,i},10^{-6})},\\
 \mathrm{TCE}&=\frac{1}{N_m}\sum_{t,i}m_{t,i}
  \frac{|(W_tD_{t-1})_i-D_{t,i}-(W_tG_{t-1})_i+G_{t,i}|}
       {\max(G_{t,i},10^{-6})}.
\end{aligned}
$$

Both are dimensionless. Raw temporal delta is the mean absolute consecutive depth difference over all pixels, without warping or a GT-validity mask, after the declared clip alignment; it has depth units and is not itself a ground-truth temporal-accuracy measure. Temporal scores are not reported after per-frame fitting, which would change the temporal signal. All code-level clamps and invalid-fit conventions remain those of the frozen scorer.

### Memory, edge evidence and portability

For $N$ independent streams, persistent storage is $W+NS$, before input/output buffers, temporary activations and allocator overhead. The native FP32 weights occupy about 15.968 MiB; K30 state plus caches occupies 13.501 MiB per stream, versus 12.000 MiB for dense carry without output caches. This is not peak VRAM, and inactive states are still densely allocated. Fewer parameters or updates therefore do not by themselves establish lower latency or multi-stream memory.

Historical Nano Developer Kit B01 logs cover user-confirmed 5 W and 10 W modes, but use a reference scan, fixed-ratio activity and repeated synthetic inputs. They are not the fused desktop path or the measured PNG-to-depth pipeline. The power source is module VDD_IN, not whole-system wall power. Missing historical weight/code linkage and independent power repetitions prevent those logs from establishing final-model real-video energy savings. Their separate audit is supplied without a quantitative edge claim in this paper.

The private package includes source, derived scores, manifests, configuration and native weights; original media and third-party baseline weights require separate authorized acquisition. Frozen paths may need a versioned relocation manifest on another machine. Full clean-machine training/inference reproduction and independent human verification have not been performed. The complete 40-item internal-review response is provided separately; partial and withdrawn items must not be read as experimentally resolved.

## References

<div id="ref-vda">

</div>

##### \[1\]

Chen, S.; Guo, H.; Zhu, S.; Zhang, F.; Huang, Z.; Feng, J.; Kang, B. Video Depth Anything: Consistent Depth Estimation for Super-Long Videos. CVPR, 2025. <https://arxiv.org/abs/2501.12375>.

<div id="ref-ovda">

</div>

##### \[2\]

Feiden, J.-F.; Küchler, T.; Zavadski, D.; Savchynskyy, B.; Rother, C. Online Video Depth Anything: Temporally-Consistent Depth Prediction with Low Memory Consumption. arXiv:2510.09182, 2025. <https://arxiv.org/abs/2510.09182>.

<div id="ref-campos2018">

</div>

##### \[3\]

Campos, V.; Jou, B.; Giró-i-Nieto, X.; Torres, J.; Chang, S.-F. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks. ICLR, 2018. <https://arxiv.org/abs/1708.06834>.

<div id="ref-gu2024">

</div>

##### \[4\]

Gu, A.; Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752v2, 2024. <https://arxiv.org/abs/2312.00752>.

<div id="ref-deltacnn">

</div>

##### \[5\]

Parger, M.; Tang, C.; Twigg, C.D.; Keskin, C.; Wang, R.; Steinberger, M. DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos. CVPR, 2022. <https://arxiv.org/abs/2203.03996>.

<div id="ref-eventful">

</div>

##### \[6\]

Dutson, M.; Li, Y.; Gupta, M. Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers. ICCV, 2023. <https://arxiv.org/abs/2308.13494>.

<div id="ref-spikessm">

</div>

##### \[7\]

Zhong, Y.; Zhao, R.; Wang, C.; Guo, Q.; Zhang, J.; Lu, Z.; Leng, L. SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning. arXiv:2410.17268, 2024. <https://arxiv.org/abs/2410.17268>.

<div id="ref-spikyspace">

</div>

##### \[8\]

Tang, K.; Zheng, J.; Jin, Y.; Qiu, Y.; Sun, G.; Yan, Z.; Wong, W.-F. SpikySpace: A Spiking State Space Model for Energy-Efficient Time Series Forecasting. arXiv:2601.02411v2, 2026. <https://arxiv.org/abs/2601.02411>.

<div id="ref-dpt">

</div>

##### \[9\]

Ranftl, R.; Bochkovskiy, A.; Koltun, V. Vision Transformers for Dense Prediction. ICCV, 2021. <https://arxiv.org/abs/2103.13413>.

<div id="ref-midas">

</div>

##### \[10\]

Ranftl, R.; Lasinger, K.; Hafner, D.; Schindler, K.; Koltun, V. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. <https://arxiv.org/abs/1907.01341>.

<div id="ref-da2">

</div>

##### \[11\]

Yang, L.; Kang, B.; Huang, Z.; Zhao, Z.; Xu, X.; Feng, J.; Zhao, H. Depth Anything V2. arXiv:2406.09414, 2024. <https://arxiv.org/abs/2406.09414>.

<div id="ref-amdahl1967">

</div>

##### \[12\]

Amdahl, G.M. Validity of the single processor approach to achieving large scale computing capabilities. AFIPS Spring Joint Computer Conference, 1967, 483–485. <https://doi.org/10.1145/1465482.1465560>.

<div id="ref-tumdata">

</div>

##### \[13\]

Sturm, J.; Engelhard, N.; Endres, F.; Burgard, W.; Cremers, D. A Benchmark for the Evaluation of RGB-D SLAM Systems. IROS, 2012, 573–580. <https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf>.

<div id="ref-bonndata">

</div>

##### \[14\]

Palazzolo, E.; Behley, J.; Lottes, P.; Giguère, P.; Stachniss, C. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. IROS, 2019. <https://arxiv.org/abs/1905.02082>.
