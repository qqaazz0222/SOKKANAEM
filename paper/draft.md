# SOKKANAEM: Exact Change-Gated State-Space Modeling for Efficient and Stable Video Depth

> **Working draft — 27 July 2026.** Author names, affiliations, venue formatting, citations, qualitative figures, and Jetson measurements remain to be added. All numerical claims below are limited to completed experiments in this repository.

## Abstract

Video depth models repeatedly process large static regions and often exhibit frame-to-frame flicker. We introduce **SOKKANAEM**, a compact recurrent video-depth model that connects patch-level change detection to the discretization step of a selective state-space model (SSM). Given a binary activity mask \(M\), we replace the SSM step size \(\Delta\) with \(\widetilde{\Delta}=M\Delta\). For a static patch, \(\widetilde{\Delta}=0\) yields \(\bar A=I\) and \(\bar B=0\), so the hidden state is copied exactly rather than approximately reconstructed. A temporal SSM preserves per-location memory, while a spatial SSM and a dense decoder recover spatial context and depth. For moving cameras, low-resolution global motion compensation (GMC) precedes feature-space change detection.

On a 1,000-clip synthetic holdout, a 2.8M-parameter model reduces the active-patch ratio from 99.6% to 24.0% with only a 0.3% relative increase in AbsRel (0.4166 to 0.4174); all three measured temporal errors decrease. Trained on mixed real and synthetic data, a 4.19M-parameter model reaches 0.1595 AbsRel and 0.8262 \(\delta_1\) on a real indoor holdout at 32.2% activity, with lower raw frame-to-frame variation than a 30x larger generalist. An iso-activity token-drop control fails sharply: at 31.6% activity its AbsRel is 1.7178 versus 0.4292 for \(\Delta\)-gating, demonstrating that reading preserved state is essential. SOKKANAEM has substantially lower raw frame-difference error than three larger baselines, but does not lead motion-compensated OPW or GT-referenced TCE. We also show that exact state skipping alone does **not** make end-to-end computation proportional to scene change: in the current architecture, dense readout and decoding dominate. These results establish exact change-gated state preservation as a useful mechanism while delimiting the kernel and system work required for realized efficiency.

## 1. Introduction

Monocular depth estimation has advanced rapidly, but applying image models independently to video leaves two structural inefficiencies. First, every frame is processed at nearly fixed cost even when most of the scene is unchanged. This is particularly wasteful for fixed surveillance cameras, where foreground motion may occupy only a small fraction of the image. Second, independent predictions can flicker even when the underlying geometry is stable. Video-specific models improve temporal coherence, but commonly retain dense per-frame computation or add a separate temporal refinement stage.

This work asks a narrower question: **can an SSM treat “no visual change” as an exact no-op on its temporal memory?** The zero-order-hold discretization of a selective SSM provides a direct construction. Multiplying its step size by a binary patch mask makes a masked update equal the identity map on hidden state. Static patches therefore retain memory without a learned approximation, feature imputation, or a separately invalidated temporal cache.

We instantiate this idea in SOKKANAEM, a streaming video-depth architecture with alternating temporal and spatial SSM blocks. A lightweight detector produces patch activity masks using hysteresis, dilation, and periodic keyframes. A sensor-free GMC and feature-space detector extend the mechanism to ego-motion. We evaluate not only depth accuracy and raw frame variation, but also optical-flow-warped consistency (OPW), a GT-referenced temporal consistency error (TCE), constant-output controls, analytical MACs, and measured latency.

The completed experiments support four conclusions:

1. **Exact state preservation.** For \(M=0\), \(\Delta\)-gating gives a bit-exact state copy in implementation and an identity transition analytically.
2. **A favorable sparsity–accuracy trade-off.** On the main 1,000-clip holdout, reducing activity to 24.0% changes AbsRel by only +0.3% relative; on real indoor data, 17.9% activity incurs +1.0% relative AbsRel.
3. **Preserved-state readout matters.** At matched masks, replacing \(\Delta\)-gating with token dropping degrades AbsRel by 4.0× at 31.6% activity.
4. **The efficiency claim has a clear boundary.** Only the state-update path scales with activity in the present model. Dense projections, state readout, spatial processing, and the decoder impose a large compute floor, so end-to-end cost is not yet proportional to change.

We deliberately avoid a broader claim of universal temporal-consistency superiority. SOKKANAEM leads raw frame-difference error, a flicker-oriented measure, but Depth Anything 3 and Video Depth Anything are stronger on motion-compensated or GT-referenced measures in the current evaluation.

## 2. Related Work

### 2.1 Monocular and video depth

Modern monocular systems based on DPT and Depth Anything provide strong frame-wise depth but have no persistent state across a stream. Video depth methods introduce temporal attention, motion modules, or post-processing to improve consistency. Their primary objective is prediction quality; computation generally remains dense in space and time. SOKKANAEM instead studies conditional temporal state updates and is complementary to stronger pretrained visual encoders and decoders.

### 2.2 Dynamic token and change-based computation

Token pruning, merging, and early exiting reduce computation within an image. DeltaCNN, skip convolutions, and eventful transformers exploit change across frames, but must preserve or reconstruct dense outputs using feature caches and cache-consistency rules. SOKKANAEM shares the principle of recomputing changed regions, but stores temporal information in the SSM hidden state. Our token-drop ablation directly tests whether merely bypassing static tokens is sufficient; it is not.

### 2.3 Visual state-space models

Vision SSMs replace quadratic attention with linear scans and have been extended to images and video. Standard variants still update all tokens. Our contribution is to connect externally detected visual change to the SSM discretization parameter and to distinguish exact hidden-state preservation from approximate spatial output caching.

## 3. Method

### 3.1 Overview

For frames \(I_{t-1}\) and \(I_t\), the model:

1. partitions the image into \(p\times p\) patches (\(p=16\) in completed main experiments);
2. estimates a binary activity mask \(M_t\);
3. embeds the current frame;
4. alternates temporal \(\Delta\)-gated SSM and spatial SSM blocks;
5. decodes the dense depth map \(\widehat D_t\);
6. carries detector state and per-patch SSM state into the next frame.

The model supports independent state dictionaries, allowing a single set of weights to serve multiple streams without state leakage.

### 3.2 Patch change detector

The default pixel detector assigns patch \(i\) the mean squared change

\[
s_{t,i}=\frac{\lVert P_{t,i}-P_{t-1,i}\rVert_2^2}{p^2 C}.
\]

Two thresholds implement hysteresis:

\[
M_{t,i} =
\begin{cases}
1,&s_{t,i}>\tau_{\mathrm{on}},\\
0,&s_{t,i}<\tau_{\mathrm{off}},\\
M_{t-1,i},&\text{otherwise}.
\end{cases}
\]

We dilate active regions by one patch to protect object boundaries and force a full update every \(K\) frames to limit drift. Evaluation-only ablations found pixel MSE and cosine detection comparable at matched activity, so MSE remains the default. Training with i.i.d. random masks was at least as robust as detector-driven fine-tuning in the completed three-arm study.

### 3.3 Exact \(\Delta\)-gating

For a continuous SSM with state matrix \(A\), input projection \(B\), and step \(\Delta_i\), zero-order-hold discretization gives

\[
\bar A_i=\exp(\Delta_i A), \qquad
\bar B_i=(\Delta_i A)^{-1}\left(\exp(\Delta_i A)-I\right)\Delta_i B,
\]

\[
h_i=\bar A_i h_{i-1}+\bar B_i x_i,\qquad
y_i=C_i h_i+D x_i.
\]

We apply the activity mask to the step:

\[
\widetilde{\Delta}_i=M_i\Delta_i.
\]

If \(M_i=0\), then

\[
\bar A_i=I,\qquad \bar B_i=0,\qquad h_i=h_{i-1}.
\]

Thus the state transition for a static patch is exactly the identity. Importantly, the output may still read the preserved state through \(C_i h_i\). This distinction explains both the accuracy of \(\Delta\)-gating and its compute floor: exact state preservation does not authorize dropping all static-token projections and readout.

### 3.4 Spatiotemporal backbone

The temporal block scans each spatial patch through the frame axis, so its hidden state is a memory tied to a fixed image location. The spatial block mixes context within a frame. Alternating these blocks combines temporal persistence with spatial reasoning. The completed v3/v7 model uses dimension 192, four blocks, state dimension 16, and 2.8M parameters.

An optional spatial output cache gathers active patches, updates them, and scatters them back while reusing previous outputs for static locations. Unlike temporal \(\Delta\)-gating, this operation is approximate because static spatial tokens no longer contribute fresh context. It is useful only at low activity in the current inference-only implementation.

### 3.5 Moving-camera extension

Camera motion makes raw pixel differences dense. We therefore estimate a homography from at most 50 tracked points on a low-resolution frame using Lucas–Kanade tracking and RANSAC. The previous frame is warped to the current view, after which relative \(L_1\) differences between patch embeddings produce the activity mask. Failure falls back to the identity transform, increasing activity rather than silently suppressing changes.

On Virtual KITTI 2, GMC plus feature gating reaches 23.7% activity with only +0.7% relative AbsRel over full computation. At 2.3% activity, AbsRel rises from 0.2054 to 0.2098. These results validate the mechanism on clean synthetic ego-motion; real noisy dashcam footage remains untested.

### 3.6 Decoder and objective

The completed reported checkpoints use a dense upsampling decoder. Training minimizes

\[
\mathcal L =
\mathcal L_{\mathrm{SI-log}}
+0.5\mathcal L_{\mathrm{grad}}
+0.1\mathcal L_{\mathrm{temp}}
+0.05\mathcal L_{\mathrm{normal}}.
\]

Random mask scheduling increases the skip ratio during training. A failed run used Kendall-style automatic loss weighting: the optimizer drove the temporal-loss weight to its upper clamp, making a constant depth map optimal. That checkpoint was discarded, fixed weights were restored, and a prediction-variance collapse detector was added.

Recent, not-yet-reported training code adds clip-consistent augmentation, a multi-scale DPT-like decoder, disparity prediction, and multi-scale gradient loss. Since the corresponding long-run experiment is incomplete, these components are not claimed in the main results.

## 4. Experimental Setup

### 4.1 Data

The main synthetic training mixture contains Virtual KITTI 2, TartanAir v2, and PointOdyssey. Dataset-balanced sampling prevents the largest source from dominating. The principal synthetic holdout contains 8,929 clips; all headline comparisons use the same deterministic first 1,000 clips.

For the deployment-relevant real domain, we use TUM RGB-D fixed-camera sequences and Bonn RGB-D Dynamic. RGB and depth are paired by timestamp within 20 ms. The v7 model starts from v3 and is fine-tuned for 15k steps on two real and three synthetic sources. Its real evaluation contains 488 held-out clips; the synthetic evaluation retains the same 1,000-clip protocol.

The early proof of concept uses the full Virtual KITTI 2 corpus (42,520 frames; 21,120 training clips) at 128 pixels and 30k optimization steps.

### 4.2 Metrics

We report AbsRel, RMSE, and \(\delta_1\) after the evaluation protocol's per-clip scale alignment. Activity is the fraction of patch updates enabled by the detector.

Temporal metrics are:

- **t-delta:** mean adjacent-frame output difference; it measures raw flicker but is minimized by a constant output.
- **OPW:** optical-flow-warped prediction error using RAFT-small.
- **TCE:** the difference between the prediction's warped residual and the GT's warped residual. This penalizes a constant prediction when GT geometry changes.

Every full temporal table includes a per-clip optimal constant-depth control. This control exposes the degeneracy of t-delta and OPW and supplies the dataset-specific residual floor for TCE.

### 4.3 Baselines and implementation

We compare with Depth Anything V2 Small (24.8M), Depth Anything 3 Base (120M), and Video Depth Anything Small metric (28.4M), using the same 1,000 clips and common metric implementation. The reported SOKKANAEM model has 2.8M parameters. Latency is measured with batch size 1 on an RTX 4090. Analytical multiply–accumulate counts are derived from the configured model. No Jetson result is available yet.

## 5. Results

### 5.1 Activity–accuracy trade-off

Table 1 reports the latest v7 checkpoint on the 1,000-clip synthetic holdout.

**Table 1. Synthetic holdout, 1,000 clips.**

| \(\tau_{\mathrm{on}}\) | Active (%) | AbsRel | RMSE | \(\delta_1\) | t-delta | OPW | TCE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.6 | 0.4166 | 26.9360 | 0.5317 | 0.2439 | 0.0475 | 0.0818 |
| 0.005 | 48.6 | 0.4161 | 26.9078 | 0.5326 | 0.2224 | 0.0468 | 0.0811 |
| 0.02 | 39.5 | 0.4164 | 26.9094 | 0.5327 | 0.2169 | 0.0466 | 0.0809 |
| 0.05 | 31.6 | 0.4166 | 26.9181 | 0.5327 | 0.2131 | 0.0465 | 0.0808 |
| 0.1 | 24.0 | 0.4174 | 26.9263 | 0.5322 | 0.2107 | 0.0465 | 0.0808 |
| Constant control | — | 0.7434 | 44.7539 | 0.2928 | 0.0000 | 0.0000 | 0.0513 |

Reducing activity by 75.9 percentage points changes AbsRel by only +0.0008 (+0.3% relative), while t-delta decreases by 13.6%. OPW and TCE improve slightly rather than degrading. A 32-frame evaluation produces nearly the same activity as the 8-frame protocol, so we do not claim that short clips systematically underestimate deployment sparsity.

### 5.2 Real indoor results

Zero-shot v3 failed to transfer from synthetic outdoor depth to real indoor Kinect depth and lost to a constant predictor. Mixed-domain fine-tuning reverses this failure, and two auxiliary losses added at the final training stage — a flow-warped log-depth residual term and a depth-boundary-weighted term — improve accuracy and temporal stability together at unchanged compute.

**Table 2. Held-out real indoor RGB-D (TUM and Bonn), 100 clips per source, dataset-balanced mean. The two rows share an activity ratio of 32.2%, so the improvement is not bought with computation.**

| Model | AbsRel | RMSE (m) | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|
| Previous checkpoint | 0.1633 | 0.6279 | 0.8211 | 0.0915 | 0.0271 | 0.0351 |
| **Confirmed checkpoint (4.19M)** | **0.1595** | **0.6063** | **0.8262** | **0.0751** | **0.0243** | **0.0323** |

Per source, the confirmed model reaches 0.1321 AbsRel and 0.8426 \(\delta_1\) on TUM at 19.7% activity, and 0.1869 and 0.8098 on Bonn at 44.7%. On the synthetic holdout it reaches 0.3791 AbsRel and 14.22 RMSE.

Two cautions apply. The synthetic \(\delta_1\) difference between these checkpoints lies inside a measured seed standard deviation of \(\pm\)0.015 and is not claimed. More importantly, the two rows differ in initialisation lineage and cumulative steps, so Table 2 is a comparison of checkpoints, not a controlled loss ablation; the controlled ablation exists only at 8k steps, where the ranking was in fact reversed. Short-probe rankings of loss terms did not survive to convergence, which we report as a methodological finding: brief probes can settle whether a term helps but not how strongly to weight it.

### 5.3 Comparison with larger depth models

Table 3 retains the earlier synthetic-protocol comparison. Table 3b regenerates the comparison against current baseline checkpoints on the real indoor holdout under a single shared protocol, which is the setting the method targets.

**Table 3. Common 1,000-clip baseline comparison. Lower is better except \(\delta_1\).**

| Model | Params | AbsRel | RMSE | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOKKANAEM v3, active 31.6% | 2.8M | 0.4292 | 26.57 | 0.5285 | **0.2455** | 0.0541 | 0.0879 |
| Depth Anything V2 Small | 24.8M | 0.5802 | 62.32 | **0.7494** | 9.4742 | 0.2458 | 0.2616 |
| Depth Anything 3 Base | 120M | 0.3421 | **17.87** | 0.4848 | 1.8013 | **0.0487** | **0.0603** |
| Video Depth Anything Small metric | 28.4M | **0.3274** | 26.99 | 0.7041 | 2.1803 | 0.0590 | 0.0818 |
| Constant control | — | 0.7434 | 44.75 | 0.2928 | 0.0000 | 0.0000 | 0.0513 |

**Table 3b. Real indoor holdout, identical protocol for all three models.**

| Model | Params | AbsRel | \(\delta_1\) | t-delta | TCE |
|---|---:|---:|---:|---:|---:|
| Depth Anything 3 Base | 120M | **0.1244** | **0.8790** | 0.1024 | **0.0252** |
| Depth Anything V2 Small | 24.8M | 0.2256 | 0.9050 | 0.9920 | 0.1015 |
| **SOKKANAEM (ours)** | **4.19M** | 0.1595 | 0.8262 | **0.0751** | 0.0323 |

Against a 30x larger generalist we remain behind on accuracy and on GT-referenced temporal error, though the AbsRel gap narrowed from +31% to +28% relative with the final checkpoint, while raw frame-to-frame variation is 1.36x lower. Against a comparable-size model we lead every metric except \(\delta_1\); Depth Anything V2's high \(\delta_1\) with poor AbsRel and RMSE follows from per-clip scale-and-shift alignment, which ranks most pixels correctly while allowing a few far pixels to dominate squared error.

On the earlier synthetic protocol, SOKKANAEM's t-delta is 7.3× lower than DA3, 8.9× lower than VDA, and 38.6× lower than DA V2. This indicates strong suppression of raw flicker. It is not a universal temporal lead: DA3 has lower OPW, while both DA3 and VDA have lower TCE. Accuracy also depends on the metric: VDA leads AbsRel and \(\delta_1\) among the video-oriented entries, and DA3 leads RMSE. SOKKANAEM's advantage is compactness and stable raw output under high patch sparsity, not state-of-the-art depth accuracy.

### 5.4 Why exact state readout matters

We compare \(\Delta\)-gating with a token-drop arm using the same detector and masks. Both freeze hidden state at static positions; token drop additionally bypasses the temporal-block output instead of reading preserved state.

**Table 4. Gating-location ablation on 1,000 clips.**

| Active (%) | Method | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---:|---|---:|---:|---:|---:|---:|
| 48.6 | \(\Delta\)-gating | 0.4288 | 0.5276 | 0.2573 | 0.0544 | 0.0882 |
| 48.6 | Token drop | 1.3698 | 0.1658 | 7.1374 | 0.5337 | 0.5680 |
| 31.6 | \(\Delta\)-gating | **0.4292** | **0.5285** | **0.2455** | **0.0541** | **0.0879** |
| 31.6 | Token drop | 1.7178 | 0.1424 | 7.1692 | 0.8252 | 0.8579 |
| 24.0 | \(\Delta\)-gating | 0.4301 | 0.5282 | 0.2421 | 0.0540 | 0.0878 |
| 24.0 | Token drop | 1.8727 | 0.1365 | 7.2018 | 0.9285 | 0.9597 |

At 31.6% activity, token dropping has 4.0× the AbsRel and 29× the t-delta of \(\Delta\)-gating. The result isolates the benefit: skipping is accurate because the model continues to read a preserved state, not because static tokens can be removed without replacement.

### 5.5 Moving-camera proof of concept

On Virtual KITTI 2 at 128 pixels, pixel gating reaches 44.6% activity with 0.2057 AbsRel versus 0.2054 at full activity. GMC plus feature gating reaches 23.7% activity with 0.2068 AbsRel and 2.3% activity with 0.2098. Raw temporal difference decreases monotonically from 0.1033 at full computation to 0.0762 at the most aggressive GMC point. GMC adds approximately 0.8 ms/frame in this low-resolution setup.

### 5.6 Compute and wall-clock analysis

Two results in this section point in opposite directions, and both matter.

**Analytical compute.** The current architecture costs 1.644 GMAC/frame at full activity. With both caches, 15.4% activity costs 0.608 GMAC — 37.0% of full. The decoder is 23.1% of the dense floor and patch embedding 2.3%, so the saving is real and comes from the backbone, where sparsity applies.

**Measured latency.** The scan implementation, not the gather, dominated wall-clock. Profiling a sparse frame at 22% activity attributes 71% of it to the spatial scan and only 6% to gathering and scattering active tokens. The reference scan is chunked and materialises a \((B, C, C, P, S)\) pairwise-decay tensor per chunk: at \(L=64\), \(P=384\), \(S=16\) it moves roughly 25 MB to perform 0.4 MMAC. We therefore replaced it with a fused Triton kernel that keeps the recurrence in registers, used at inference while training retains the differentiable chunked path. \(\Delta\)-gating remains bit-exact through the kernel — \(\widetilde{\Delta}=0\) gives \(\exp(0)=1\) and a zero input term — and every evaluation metric is unchanged to four decimal places.

Per-frame latency on one RTX 4090 at 256 pixels, single stream, fp32, 22% activity:

| Path | Chunked scan | Fused kernel | Speedup |
|---|---:|---:|---:|
| Full compute, eager | 11.38 ms | 1.98 ms | 5.7x |
| **Full compute, compiled** | 4.70 ms | **1.29 ms** (776 FPS) | 3.6x |
| Sparse, eager | 4.87 ms | 2.40 ms | 2.0x |
| Sparse + bucket padding | 5.39 ms | 2.55 ms | 2.1x |
| Sparse + bucket + compiled | 2.99 ms | 2.04 ms (491 FPS) | 1.5x |

In fp16 the sparse path reaches 1.69 ms (593 FPS) with 37 MB peak memory and 6.38 MB of persistent state per stream; four batched streams run at 2,060 FPS aggregate on the full-compute path.

**The inversion.** Before the kernel, the sparse path was 1.57x faster than compiled full compute at 22% activity. After it, compiled full compute is faster at *every* activity level (1.29 ms against 2.03–2.20 ms). The explanation is immediate: what sparsity saved was the scan, and the scan is now nearly free. What remains is fixed bookkeeping that does not scale with activity — `nonzero` gather, the column-major argsort, bucket padding, cache cloning, scatter, and the host synchronisations these force. Latency is now flat in activity for every path (full compute varies only between 1.973 and 1.994 ms across 5–70% activity), which is the signature of an overhead-bound rather than compute-bound regime.

We report this plainly because it bounds the contribution. Exact state preservation, the MAC reduction, and the kernel itself all stand. The claim that the sparse path is *faster* does not stand on this GPU. Whether a 63% MAC reduction converts into time and energy depends on the hardware being compute-bound, which a 4090 at this model scale is not; settling that requires the edge measurement listed in Section 7.

## 6. Ablations and Diagnostic Findings

### 6.1 Mask policy

MSE and cosine change scores perform similarly at matched activity. Keyframe intervals between the tested settings have little effect on short-clip accuracy. Training with i.i.d. random masks produced better robustness than detector-driven mask fine-tuning, contrary to the initial expectation that train–deployment mask matching would be essential.

### 6.2 DINOv2 feature distillation

Matching final backbone tokens to frozen DINOv2-small features does not improve depth accuracy: v4 gives 0.4315 AbsRel versus 0.4292 for v3. It yields small improvements in temporal metrics (TCE 0.0846 versus 0.0879), suggesting regularization rather than better geometric representation.

### 6.3 Accuracy bottleneck

A patch-grid oracle shows that a 16-pixel depth decoder could reach 0.1746 AbsRel and 0.8542 \(\delta_1\), far above the current synthetic holdout result. On clips used for training, v7 reaches 0.1918 AbsRel and 0.7074 \(\delta_1\), whereas unseen clips give 0.3562 and 0.5185 in the diagnostic sample. Capacity and nominal resolution are therefore not the immediate bottleneck; cross-scene generalization is.

Source-wise analysis identifies TartanAir's very wide 0.5–129 m depth range and PointOdyssey's train–holdout gap as major contributors. This motivates disparity-space prediction, clip-consistent augmentation, and multi-scale decoding, whose final results are pending.

## 7. Limitations

The present study has several important limitations.

1. The external-baseline table uses the v3 checkpoint; v7 was rerun separately under the same synthetic data protocol but not through every external baseline script in one combined run.
2. Real indoor v7 evaluation lacks t-delta, OPW, and TCE because that run produced NaNs for temporal metrics. No temporal claim is made from Table 2.
3. OPW and TCE do not support a general temporal-consistency lead. Raw t-delta can be gamed by constant predictions and is interpreted only alongside accuracy and the constant control.
4. A fused scan kernel removed the dominant cost and, in doing so, removed the sparse path's wall-clock advantage on an RTX 4090: compiled full compute is now faster at every activity level. The remaining sparse-path cost is activity-independent bookkeeping. Sparsity's benefit is presently established in MACs and per-stream state, not in measured latency on this device.
5. Execution on an RTX 4090 is overhead-bound at this model scale, so reduced analytical MACs do not become higher FPS. Jetson Orin latency, energy, and multi-stream measurements have not been performed, and they are the measurement that decides whether the MAC reduction is worth anything in deployment.
6. GMC was validated on synthetic clean ego-motion, not noisy real mobile video.
7. The current experiments cover at most 270-frame long streams for drift analysis. Very long streaming behavior is unknown.
8. Patch-size, refinement, and fully trained decoder/cache ablations remain incomplete.

## 8. Conclusion

SOKKANAEM demonstrates that patch-level visual change can control an SSM through its discretization step, turning a static observation into an exact identity transition on temporal state. Across synthetic and real RGB-D evaluations, aggressive patch sparsity causes little loss in depth accuracy, and an iso-mask token-drop control confirms that continued readout of preserved state is critical. The model strongly suppresses raw frame-to-frame flicker while remaining much smaller than evaluated depth baselines.

The experiments also reveal the boundary of the idea. Exact state skipping is not synonymous with end-to-end sparse inference: dense readout, spatial context, and decoding remain. Nor does low raw frame variation guarantee motion-correct temporal accuracy. A fused scan kernel closed the kernel gap and, unexpectedly, made dense streaming the faster configuration on a desktop GPU, which relocates the efficiency argument from latency to compute and memory until an edge device settles it. The next decisive steps are therefore edge-device measurement, real moving-camera evaluation, and cross-domain generalization. Within these boundaries, exact \(\Delta\)-gating provides a principled foundation for change-adaptive streaming vision.

## References

> To be completed with the final bibliography. The related-work section should include, at minimum: Mamba/selective SSMs; Vision Mamba/VMamba/VideoMamba; DPT and MiDaS; Depth Anything V2 and Depth Anything 3; Video Depth Anything; NVDS; token pruning/merging methods such as DynamicViT, EViT, and ToMe; and change-based systems such as DeltaCNN, Skip-Convolutions, and Eventful Transformers.

