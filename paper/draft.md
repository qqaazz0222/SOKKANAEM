# SOKKANAEM: Exact Change-Gated State-Space Modeling for Efficient and Stable Video Depth

> **Working draft — 20 August 2026.** Author names, affiliations, venue formatting, and edge-device measurements remain to be added. All numerical claims below are limited to completed experiments in this repository, and every table row is checked against a single measurement run (`scripts/table_check.py`).
>
> **Protocol note.** Numbers in this draft supersede every figure we reported before 20 August 2026. A clip cap was sampling the first held-out sequence of each source rather than the holdout (Section 4.2); the correction moves the real-domain operating point from 0.1595 AbsRel at 32.2% activity to 0.1302 at 22.0%, and it withdraws or reduces three diagnostic findings.
>
> **Section tags.** Headings carry a status marker so a reader knows which numbers are settled and which are moving:
>
> - `[UNDER TEST]` — a running or queued experiment targets this section's conclusion directly, and the conclusion may reverse.
> - `[CHECKPOINT-DEPENDENT]` — the finding holds for the reported checkpoint. The numbers move if the checkpoint is replaced, though the qualitative claim is not expected to.
> - Untagged sections are settled: they follow from the architecture, the data, or measurements that a retrain does not affect.

## Abstract

Video depth models repeatedly process large static regions and often exhibit frame-to-frame flicker. We introduce **SOKKANAEM**, a compact recurrent video-depth model that connects patch-level change detection to the discretization step of a selective state-space model (SSM). Given a binary activity mask \(M\), we replace the SSM step size \(\Delta\) with \(\widetilde{\Delta}=M\Delta\). For a static patch, \(\widetilde{\Delta}=0\) yields \(\bar A=I\) and \(\bar B=0\), so the hidden state is carried exactly rather than approximately reconstructed — an identity transition, not a suppressed update. A temporal SSM preserves per-location memory, while a spatial SSM and a dense decoder recover spatial context and depth. For moving cameras, low-resolution global motion compensation precedes feature-space change detection. Conditional recurrent updates have precedent; what is new here is the combination of an external untrained change signal, an exact rather than approximate no-op, and a dense streaming output — together with a measurement of where the resulting claims stop holding.

A 4.19M-parameter model reaches 0.1302 AbsRel and 0.8613 \(\delta_1\) on a real indoor holdout while updating 22.0% of patches per frame; cutting the update rate to 4.6% costs 6.4% relative error, and full computation buys only 0.9% over the default. Against seven commonly cited depth models measured under one protocol, it is last or nearly last on accuracy and first on raw frame-to-frame difference — by 1.36x at eight-frame clips and 1.27x at 256-frame clips, including against a video-specific baseline with an explicit temporal module. It does not lead motion-compensated or ground-truth-referenced consistency, and we do not claim general temporal consistency.

Four measurements delimit the contribution, and each narrows a claim we expected to make. **Efficiency:** a fused scan kernel removes the dominant cost and with it the sparse path's latency advantage, so on a desktop GPU compiled dense execution is faster at every activity level; sparsity's benefit is established in multiply-accumulates and per-stream state, not in measured time, and the edge measurement that would settle it is absent. **Transfer:** on unseen real driving, accuracy transfers while sparsity does not — activity rises from 26% to 93%, recoverable to 14% only with motion compensation and per-domain threshold recalibration. **Readout:** an iso-mask token-drop control is a wash on accuracy, so preserved-state readout buys stability rather than depth quality. **Protocol:** error grows 87% from eight-frame to 256-frame clips, but stateless per-frame baselines grow more over the same clips (116% and 145%), so part of that penalty is the alignment window rather than drift; long-clip fine-tuning removes a fifth of the 256-frame error while leaving the eight-frame number unchanged to four decimal places, which is what a short-clip protocol is blind to. A predicted dynamic range under half the ground truth's on dynamic scenes is the remaining accuracy defect, and a spread term in the objective recovers part of it at a stated cost in motion-referenced consistency.

## 1. Introduction

Monocular depth estimation has advanced rapidly, but applying image models independently to video leaves two structural inefficiencies. First, every frame is processed at nearly fixed cost even when most of the scene is unchanged. This is particularly wasteful for fixed surveillance cameras, where foreground motion may occupy only a small fraction of the image. Second, independent predictions can flicker even when the underlying geometry is stable. Video-specific models improve temporal coherence, but commonly retain dense per-frame computation or add a separate temporal refinement stage.

This work asks a narrower question: **can an SSM treat “no visual change” as an exact no-op on its temporal memory?** The zero-order-hold discretization of a selective SSM (Gu & Dao, 2023) provides a direct construction. Multiplying its step size by a binary patch mask makes a masked update equal the identity map on hidden state. Static patches therefore retain memory without a learned approximation, feature imputation, or a separately invalidated temporal cache.

We instantiate this idea in SOKKANAEM, a streaming video-depth architecture with alternating temporal and spatial SSM blocks. A lightweight detector produces patch activity masks using hysteresis, dilation, and periodic keyframes. A sensor-free GMC and feature-space detector extend the mechanism to ego-motion. We evaluate not only depth accuracy and raw frame variation, but also optical-flow-warped consistency (OPW), a GT-referenced temporal consistency error (TCE), constant-output controls, analytical MACs, and measured latency.

Conditional recurrent updates have precedent, and one algebraic identity is not by itself a contribution; Section 2.4 places this work against learned skip gates and event-driven state-space models. What we claim is the combination: an external, untrained change signal wired to the discretization parameter so that a static patch's transition is an exact identity rather than a suppressed update; a clean separation between that exact temporal preservation and the approximate spatial caching a dense output still requires; an iso-mask token-drop control that isolates what reading preserved state buys from what merely skipping static tokens buys; and an empirical characterization of where the resulting efficiency and stability claims stop holding — on a desktop GPU after kernel fusion, on real capture where sparsity does not transfer, and over a stream long enough for drift to appear.

One scope note runs through the paper. **Exact** describes the temporal hidden-state transition and nothing else. The spatial output cache is an approximation, the decoder is dense, and the sparse inference path as a whole is not bit-exact against full computation; only the state a static patch carries is.

The completed experiments support five conclusions:

1. **Exact state preservation.** For \(M=0\), \(\Delta\)-gating gives a bit-exact state copy in implementation and an identity transition analytically.
2. **A favorable sparsity–accuracy trade-off.** On real indoor footage, cutting the patch update rate by 22x — 100% to 4.6% activity — costs 6.4% relative AbsRel, and the default 22.0% operating point costs 0.9%. Raw frame difference improves threefold over the same range.
3. **Preserved-state readout buys suppression of raw frame-to-frame variation, not accuracy.** At matched masks, replacing \(\Delta\)-gating with token dropping leaves accuracy unchanged but degrades all three temporal metrics. It does not reproduce the stability of preserved-state readout; it is not the case that token dropping fails outright. An earlier fourfold accuracy collapse turned out to measure an untrained sparse path (Section 5.5).
4. **The efficiency claim has a sharp boundary.** After a fused scan kernel, the model is overhead-bound rather than compute-bound on a desktop GPU, and dense execution is faster than the sparse path at every activity level. Sparsity's benefit is established in MACs and per-stream state, and its conversion into time and energy is unmeasured (Section 5.8).
5. **Clip length changes the answer, and the eight-frame convention is optimistic for everyone.** Our error grows 87% from eight-frame to 256-frame clips. Stateless per-frame baselines grow more over the same clips (116% and 145%), because per-clip alignment gets harder as the clip lengthens — so carried state is a net advantage over a long stream even though it does not accumulate accuracy within a keyframe cycle. Long-clip fine-tuning removes a fifth of the 256-frame error while leaving the eight-frame number identical to four decimal places, so a short-clip evaluation scores that intervention as doing nothing (Section 5.8).

We claim no accuracy advantage. Against the comparison group in Section 5.3, SOKKANAEM is last on AbsRel and \(\delta_1\) on real footage and leads exactly one measure, raw frame difference. Depth Anything 3 is stronger on motion-compensated and GT-referenced temporal error. The contribution is the mechanism and the efficiency-stability point it reaches, not the depth numbers.

## 2. Related Work

### 2.1 Monocular and video depth

Modern monocular systems built on dense prediction transformers (Ranftl et al., 2021) and large-scale mixed-dataset training (Ranftl et al., 2022; Yang et al., 2024) provide strong frame-wise depth but hold no persistent state across a stream. Video depth methods introduce temporal attention, motion modules, or post-processing to improve consistency (Chen et al., 2025). Their primary objective is prediction quality; computation generally remains dense in space and time. SOKKANAEM instead studies conditional temporal state updates and is complementary to stronger pretrained encoders and decoders.

### 2.2 Dynamic token and change-based computation

Token pruning, merging, and early exiting reduce computation within an image (Rao et al., 2021; Kong et al., 2022). DeltaCNN (Parger et al., 2022), skip convolutions (Habibian et al., 2021), and eventful transformers (Liang et al., 2023) exploit change across frames, but must preserve or reconstruct dense outputs using feature caches and cache-consistency rules. SOKKANAEM shares the principle of recomputing changed regions while storing temporal information in the SSM hidden state, so no cache-invalidation rule is needed for the temporal path. Our token-drop ablation tests what the readout adds over merely bypassing static tokens: temporal stability, not accuracy (Section 5.5).

### 2.3 Visual state-space models

State-space sequence models (Gu et al., 2022) with input-dependent selection (Gu & Dao, 2023) replace quadratic attention with linear scans, and have been extended to images and video (Zhang et al., 2023; Liu et al., 2024). Standard variants still update every token.

### 2.4 Conditional and event-driven state updates

Making a recurrent state update conditional is not new, and the closest prior work is worth stating precisely rather than by contrast alone. Skip RNN (Campos et al., 2018) augments a recurrent cell with a learned binary gate that either updates the state or copies it forward, with a budget term encouraging copies. Spiking state-space models reformulate the selective scan so that sparse spike signals drive state transitions, giving event-driven computation on time series (Tang et al., 2026). Concurrently with this work, event-gated video generation predicts token-level activity with a learned head and applies latent updates mainly where an interaction is forming, using hysteresis on the activity signal much as our detector does (Maduabuchi & Wang, 2026).

Three things separate the mechanism studied here from that group, and only their combination is our claim.

1. **The gate is external and untrained.** The activity signal comes from patch-level pixel or feature change, not from a learned head with a sparsity budget. Nothing in the objective can trade accuracy for a lower skip rate, and the operating point is set at inference by a threshold rather than fixed at training time — which is also why it must be recalibrated per domain (Section 5.6).
2. **The no-update case is an identity, not a suppression.** A learned gate driven to zero, or a spike that does not fire, leaves an update that is approximately skipped: the transition is still computed and the residual is small. Multiplying the discretization step instead makes \(\bar A = I\) and \(\bar B = 0\), so the state is carried with no residual at all, in implementation as well as in the algebra (Section 3.3).
3. **The task keeps a dense spatial output.** Skipping a token's temporal update does not excuse producing its depth. The separation between exact temporal state preservation and the approximate spatial caching that supplies the missing context — and the cost floor that separation implies — is specific to dense prediction and is where the efficiency claim runs out (Section 5.8).

Change-based computation in vision (Section 2.2) shares the first property and none of the second: DeltaCNN and skip convolutions preserve or reconstruct dense activations through caches with invalidation rules, where the temporal path here has no cache to invalidate. Our token-drop ablation tests exactly what the readout adds over bypassing static tokens (Section 5.5).

## 3. Method

### 3.1 Overview

![Streaming pipeline](figures/pipeline.svg)

**Figure 1. Streaming pipeline.** A change detector compares consecutive frames patch-wise and emits a binary activity mask, which reaches the backbone as the \(\Delta\)-gating signal. Static patches retain their hidden state exactly and their computation is skipped. Two caches — spatial outputs and temporal hidden state — are where the reduction in multiply–accumulate operations comes from, taking a frame from 1.644 to 0.608 GMAC at 15.4% activity. The global motion compensation branch is used only for moving cameras. Frames above 40% activity are routed through the dense path instead.

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

We dilate active regions by one patch to protect object boundaries and force a full update every \(K\) frames to limit drift. \(K\) turns out to be an accuracy control rather than a safety valve, and the drift it bounds is larger than clip-level numbers suggest (Section 5.8). Evaluation-only ablations found pixel MSE and cosine detection comparable at matched activity, so MSE remains the default. Training with i.i.d. random masks was at least as robust as detector-driven fine-tuning in the completed three-arm study.

### 3.3 Exact \(\Delta\)-gating

![Delta-gating](figures/delta-gating.svg)

**Figure 2. Exact \(\Delta\)-gating.** The activity mask multiplies the discretization step, \(\widetilde{\Delta} = M\Delta\). A changed patch takes the standard selective-SSM update. A static patch takes \(\bar A = I\) and \(\bar B = 0\), so its hidden state is copied rather than reconstructed: skipping the computation is not compensated for, it is algebraically identical to preserving the state. Early exit and token dropping instead substitute zero or an approximation, and that error accumulates across frames.

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

The temporal block scans each spatial patch through the frame axis, so its hidden state is a memory tied to a fixed image location. The spatial block mixes context within a frame. Alternating these blocks combines temporal persistence with spatial reasoning. The reported model uses dimension 192, four blocks, state dimension 16, patch size 16, and 4,185,872 parameters (16.7 MB of fp32 weights).

An optional spatial output cache gathers active patches, updates them, and scatters them back while reusing previous outputs for static locations. Unlike temporal \(\Delta\)-gating, this operation is approximate because static spatial tokens no longer contribute fresh context. It is useful only at low activity in the current inference-only implementation.

### 3.5 Moving-camera extension

Camera motion makes raw pixel differences dense. We therefore estimate a homography from at most 50 tracked points on a low-resolution frame using Lucas–Kanade tracking and RANSAC. The previous frame is warped to the current view, after which relative \(L_1\) differences between patch embeddings produce the activity mask. Failure falls back to the identity transform, increasing activity rather than silently suppressing changes.

On Virtual KITTI 2, GMC plus feature gating reaches 23.7% activity with only +0.7% relative AbsRel over full computation. Section 5.6 tests the same mechanism on real driving footage, where it also holds — but only after per-domain threshold recalibration, because the feature-scale thresholds tuned on rendered video are inoperative on real capture. That experiment is a feasibility demonstration on one dataset, not a claim of robustness to camera motion in general.

### 3.6 Decoder and objective

The reported checkpoints use a dense upsampling decoder in the DPT family (Ranftl et al., 2021) with a binned depth head (Bhat et al., 2021). Training minimizes

\[
\mathcal L =
\mathcal L_{\mathrm{SI-log}}
+0.5\mathcal L_{\mathrm{grad}}
+0.1\mathcal L_{\mathrm{temp}}
+0.05\mathcal L_{\mathrm{normal}}.
\]

The mask is binary and non-differentiable, so gradients pass through a straight-through estimator (Bengio et al., 2013). Random mask scheduling increases the skip ratio during training. A failed run used Kendall-style automatic loss weighting: the optimizer drove the temporal-loss weight to its upper clamp, making a constant depth map optimal. That checkpoint was discarded, fixed weights were restored, and a prediction-variance collapse detector was added.

## 4. Experimental Setup

### 4.1 Data

The main synthetic training mixture contains Virtual KITTI 2 (Cabon et al., 2020), TartanAir v2 (Wang et al., 2020), and PointOdyssey (Zheng et al., 2023). Dataset-balanced sampling prevents the largest source from dominating. Every comparison in this paper evaluates 100 deterministic clips per source and reports the dataset-balanced mean, so a large source cannot dominate the headline number the way pixel pooling would.

For the deployment-relevant real domain, we use TUM RGB-D fixed-camera sequences (Sturm et al., 2012) and Bonn RGB-D Dynamic (Palazzolo et al., 2019). RGB and depth are paired by timestamp within 20 ms. The reported checkpoint is trained for 60,000 steps on two real and three synthetic sources on a single GPU, taking 13 hours 11 minutes. Evaluation uses 100 clips per source with held-out sequences, so no evaluated sequence appears in training.

The early proof of concept uses the full Virtual KITTI 2 corpus (42,520 frames; 21,120 training clips) at 128 pixels and 30k optimization steps.

### 4.2 Evaluation protocol

Three protocol choices decide what the numbers mean, and each has bitten us.

**State is reset at every clip boundary and nowhere else.** Within a clip the model runs as it would in deployment: detector state, per-patch SSM state, and both caches carry from frame to frame, and the keyframe counter runs from the clip's first frame. Between clips everything is discarded. A clip is therefore a stream of its own length, which is why clip length is a protocol parameter rather than a batching detail — an eight-frame clip never reaches a keyframe at period 30, and a 256-frame clip crosses eight of them.

**Clips are disjoint and spread over the whole holdout.** Clips tile each held-out sequence without overlap, and when the clip count is capped the retained clips are spread evenly over the source rather than taken from its front. This is not a refinement: sequences are concatenated in order, so a cap of 100 on Bonn's 399 clips evaluates its first held-out sequence alone, and the same checkpoint then reads 44.7% activity under one cap and 55.8% under another. Every number in this paper is measured under even spacing; earlier versions of our own reports were not, and their Bonn and PointOdyssey columns describe one sequence rather than a holdout.

**Alignment is per clip, not per frame or per dataset.** One scale (and, where stated, shift) is fitted per clip against valid ground-truth pixels and applied to all its frames. The frame-index analysis in Section 5.8 is the one exception, where each frame is aligned independently so that a decaying curve cannot be an artefact of a single clip-level fit dominated by late frames.

### 4.3 Metrics

We report AbsRel, RMSE, and \(\delta_1\) (Eigen et al., 2014) after the evaluation protocol's per-clip scale alignment. Activity is the fraction of patch updates enabled by the detector.

Temporal metrics are:

- **t-delta:** mean adjacent-frame output difference; it measures raw flicker but is minimized by a constant output.
- **OPW:** optical-flow-warped prediction error using RAFT-small (Teed & Deng, 2020).
- **TCE:** the difference between the prediction's warped residual and the GT's warped residual. This penalizes a constant prediction when GT geometry changes.

Every full temporal table includes a per-clip optimal constant-depth control. This control exposes the degeneracy of t-delta and OPW and supplies the dataset-specific residual floor for TCE.

### 4.4 Baselines and implementation

We compare against six commonly cited depth models — DPT-Large, ZoeDepth N-K, Depth Anything V1 Small, V2 Small, V2 Base, and Depth Anything 3 Base — spanning 24.8M to 345M parameters. Published numbers for these models each come from a different split, resolution and alignment rule, so we re-ran all of them on our own holdout clips at 256 pixels through the same metric implementation rather than quoting papers.

Alignment is the one place where a single rule would be unfair. Relative-depth models are evaluated under the two-degree-of-freedom scale-and-shift fit in disparity space they are designed for; metric models and ours use one-degree-of-freedom per-clip median scaling. Because the extra degree of freedom always flatters the model receiving it, we additionally report our own model under the relative-depth rule so the protocol cannot carry the result.

The reported SOKKANAEM model has 4.19M parameters and is a single checkpoint used for every table in Section 5. Latency is measured with batch size 1 on an RTX 4090. Analytical multiply–accumulate counts are derived from the configured model. No edge-device result is available yet.

## 5. Results

### 5.1 Activity–accuracy trade-off `[CHECKPOINT-DEPENDENT]`

All tables in this section come from a single checkpoint (4.19M parameters, 60k steps) so that no comparison mixes model versions. Each row sweeps the detector threshold; 100 clips per source spread over the whole holdout, dataset-balanced mean, eight-frame clips.

**Table 1. Activity sweep on both domains. The constant-depth control is the per-clip optimal constant prediction.**

| \(\tau_{\mathrm{on}}\) | Real active (%) | Real AbsRel | Real \(\delta_1\) | Real t-delta | Synth. active (%) | Synth. AbsRel | Synth. \(\delta_1\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (full compute) | 100.0 | 0.1290 | **0.8705** | 0.0679 | 100.0 | **0.4299** | **0.5939** |
| 0.005 | 94.9 | 0.1288 | 0.8703 | 0.0751 | 62.8 | 0.4305 | 0.5906 |
| 0.01 | 85.0 | 0.1287 | 0.8701 | 0.0846 | 58.1 | 0.4315 | 0.5903 |
| 0.02 | 63.6 | **0.1282** | 0.8694 | 0.0904 | 51.2 | 0.4313 | 0.5897 |
| 0.05 (default) | 22.0 | 0.1302 | 0.8613 | 0.0607 | 40.9 | 0.4317 | 0.5918 |
| 0.1 | **4.6** | 0.1373 | 0.8576 | **0.0224** | 32.1 | 0.4332 | 0.5927 |
| Constant control | — | 0.2761 | 0.5831 | 0.0000 | — | 0.6303 | 0.4152 |

**The trade-off is better than we previously reported, and in the same direction.** On real footage, cutting the update rate by a factor of 22 — 100% to 4.6% activity — costs 6.4% relative AbsRel and 1.3 points of \(\delta_1\), while raw frame difference improves by a factor of three and both motion-referenced measures improve as well. At the default operating point, 22.0% activity, accuracy is within 0.9% of full computation. On synthetic footage a threefold cut costs 0.8%.

Two features of the curve are worth naming. Accuracy is *non-monotonic*: at 63.6% activity on real footage it is better than at full computation (0.1282 against 0.1290), consistent with gating removing stale context rather than only saving work. And the temporal metrics are non-monotonic in the other direction — t-delta rises from 0.0679 at full compute to 0.0904 at 63.6% before falling to 0.0224 at 4.6%. Partial gating updates some patches and not others, which is itself a source of frame-to-frame difference; heavy gating freezes most of the field and removes it. The stability our architecture is built for arrives at low activity, not at every activity.

The gap to the constant control remains large at every operating point — a factor of 2.1 on real AbsRel even at 4.6% activity — which the t-delta column alone would not establish, since a constant prediction scores zero there by construction.

![Activity–accuracy trade-off](figures/tradeoff.svg)

**Figure 3. Activity against accuracy** on the real indoor and synthetic holdouts, sweeping the detector threshold. Accuracy is flat to within 1% down to roughly 20% activity and then bends. On real footage a 22-fold cut in the update rate costs 6.4% relative AbsRel, and the default operating point (circled) costs 0.9%. The per-clip optimal constant-depth control lies far above both panels and is marked off scale, so the vertical axis can resolve the curve the panel exists to show.

The synthetic sweep does not reach low activity because TartanAir stays between 80% and 100% active regardless of threshold. This is the same phenomenon quantified in Section 5.6: how much a stream can skip is a property of the capture, not only of the method.

### 5.2 Real indoor results `[CHECKPOINT-DEPENDENT]`

An earlier synthetic-only checkpoint failed to transfer to real indoor Kinect depth and lost to a constant predictor. Mixed-domain fine-tuning reverses this failure, and two auxiliary losses added at the final training stage — a flow-warped log-depth residual term and a depth-boundary-weighted term — improve accuracy and temporal stability together at unchanged compute.

**Table 2. Held-out real indoor RGB-D (TUM and Bonn), eight-frame clips, 100 clips per source spread evenly over the holdout, dataset-balanced mean. The two rows share an activity ratio, so the improvement is not bought with computation.**

| Model | AbsRel | \(\delta_1\) | t-delta | OPW | TCE | Active (%) |
|---|---:|---:|---:|---:|---:|---:|
| Previous checkpoint (first-sequence sampling) | 0.1633 | 0.8211 | 0.0915 | 0.0271 | 0.0351 | 32.2 |
| Reported checkpoint (first-sequence sampling) | 0.1595 | 0.8262 | 0.0751 | 0.0243 | 0.0323 | 32.2 |
| **Reported checkpoint (4.19M), full holdout** | **0.1302** | **0.8613** | **0.0607** | **0.0184** | **0.0262** | **22.0** |
| **Long-clip checkpoint, full holdout** | **0.1302** | **0.8684** | 0.0702 | 0.0196 | 0.0273 | **22.0** |

The first two rows are the comparison of training recipes we previously reported, and they are kept because the auxiliary-loss conclusion rests on them; both were measured before the sampling fix of Section 4.2 and therefore describe Bonn's first held-out sequence. Rows three and four are the same checkpoints on the full holdout, and they are the numbers used everywhere else in this paper. The sampling change moves the reported checkpoint from 0.1595 to 0.1302 AbsRel and its activity from 32.2% to 22.0% — the first-sequence subset is the crowd scene, which is both harder and more active than the holdout it stood for.

Per source under the fixed sampling, the reported model reaches 0.1321 AbsRel and 0.8426 \(\delta_1\) on TUM at 19.7% activity, and 0.1283 and 0.8801 on Bonn at 24.4%. The Bonn column previously read 0.1869 at 44.7% activity, which was the crowd sequence alone.

Two cautions apply. The synthetic \(\delta_1\) difference between these checkpoints lies inside a measured seed standard deviation of \(\pm\)0.015 and is not claimed. More importantly, the two rows differ in initialisation lineage and cumulative steps, so Table 2 is a comparison of checkpoints, not a controlled loss ablation; the controlled ablation exists only at 8k steps, where the ranking was in fact reversed. Short-probe rankings of loss terms did not survive to convergence, which we report as a methodological finding: brief probes can settle whether a term helps but not how strongly to weight it.

### 5.3 Comparison with larger depth models `[CHECKPOINT-DEPENDENT]`

Every model in this section is measured on the same holdout clips, at 256 pixels, through the same metric implementation, under the protocol of Section 4.2. We report two clip lengths, because they answer different questions and disagree. Eight frames is the convention in this literature. 256 frames is the streaming setting the architecture is for, and it is the primary table.

**Table 3a. The comparison group at 256 frames — the streaming protocol. Real indoor holdout (TUM, Bonn), disjoint clips, 13 clips per model, dataset-balanced mean. DA3 receives the whole clip at once and is not causal; every other row is causal. Alignment is each model's native rule (Section 4.4); Section 5.4 reports the whole group under both rules.**

| Model | Params | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|
| DA 3 Base (non-causal) | 120M | **0.1163** | **0.8871** | 0.0857 | **0.0181** | **0.0241** |
| Video Depth Anything S (metric) | 28.4M | 0.1276 | 0.8808 | 0.0854 | 0.0217 | 0.0274 |
| ZoeDepth N-K | 345M | 0.1285 | 0.8677 | 0.0871 | 0.0227 | 0.0278 |
| DA V1 Small | 24.8M | 0.1404 | 0.8339 | 0.1222 | 0.0271 | 0.0324 |
| DPT-Large | 343M | 0.1891 | 0.7461 | 0.1631 | 0.0352 | 0.0404 |
| **SOKKANAEM (long-clip)** | **4.19M** | 0.1990 | 0.7864 | **0.0674** | 0.0258 | 0.0335 |
| DA V2 Base | 97.5M | 0.2151 | 0.7387 | 0.3194 | 0.0483 | 0.0541 |
| SOKKANAEM (reported) | 4.19M | 0.2434 | 0.7134 | 0.0719 | 0.0291 | 0.0366 |
| DA V2 Small | 24.8M | 0.5491 | 0.7305 | 3.5881 | 0.2228 | 0.2283 |

**Table 3b. The same group at eight frames, the conventional protocol. 189 clips per model.**

| Model | Params | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|
| DA V1 Small | 24.8M | **0.0650** | 0.9439 | 0.0892 | 0.0202 | 0.0261 |
| DPT-Large | 343M | 0.0875 | 0.9276 | 0.1162 | 0.0270 | 0.0326 |
| DA V2 Base | 97.5M | 0.0877 | **0.9420** | 0.3814 | 0.0306 | 0.0367 |
| ZoeDepth N-K | 345M | 0.0992 | 0.8900 | 0.0866 | 0.0211 | 0.0264 |
| Video Depth Anything S (metric) | 28.4M | 0.1000 | 0.9139 | 0.0829 | 0.0200 | 0.0258 |
| DA 3 Base (non-causal) | 120M | 0.1130 | 0.8924 | 0.0825 | **0.0140** | **0.0200** |
| **SOKKANAEM (reported)** | **4.19M** | 0.1302 | 0.8613 | **0.0607** | 0.0184 | 0.0262 |
| **SOKKANAEM (long-clip)** | **4.19M** | 0.1302 | 0.8684 | 0.0702 | 0.0196 | 0.0273 |
| DA V2 Small | 24.8M | 0.2068 | 0.9292 | 1.0015 | 0.0895 | 0.0953 |

![Comparison group](figures/comparison.svg)

**Figure 4. Accuracy against temporal stability across the comparison group.** Marker area scales as the logarithm of parameter count; down and to the left is better on both axes. Our model is the smallest marker and the lowest on the stability axis in both panels. The stability axis is logarithmic because t-delta spans two orders of magnitude across the group.

Four things follow, and only one of them flatters us.

**We lead raw frame-to-frame difference under both protocols, and the video-specific baseline does not take it.** At 256 frames, 0.0674 against 0.0854 for Video Depth Anything, the one baseline with an explicit temporal module, and 0.0857 for the non-causal DA3 — a factor of 1.27 over the best of them. At eight frames the margin is 1.36. This is the claim the architecture was built to make, and adding the class of baseline that was missing from our earlier tables did not overturn it.

**We are last or nearly last on accuracy, under both protocols.** At eight frames, 0.1302 AbsRel against 0.0650 for a 24.8M-parameter baseline: a factor of two, at six times fewer parameters. At 256 frames we are sixth of nine on AbsRel. The gap narrows with clip length but does not close, and nothing in this paper claims otherwise.

**Motion-referenced consistency is not ours.** DA3 is better on OPW and TCE under both protocols, and at 256 frames Video Depth Anything and ZoeDepth are too. Raw frame difference and motion-compensated consistency come apart here, and only the first favours us.

**The two protocols rank the group differently, which is the point of reporting both.** Every model degrades from eight frames to 256 — the per-clip alignment window grows with the clip, so a single scale (or scale and shift) must serve a longer span — but they degrade by very different factors: DPT-Large by 116% and DA V2 Base by 145%, our reported checkpoint by 87%, our long-clip checkpoint by 53%, DA3 by only 3%. A model that carries state and one that processes the whole clip jointly both hold up better than per-frame models over a long stream, for opposite reasons: ours accumulates evidence causally and DA3 is allowed to see the future. Read only the eight-frame table and none of that is visible.

**Table 3c. Synthetic holdout (Virtual KITTI 2, TartanAir v2, PointOdyssey), eight-frame clips, 300 clips per model, dataset-balanced mean. Our row is full computation; the sweep is in Table 1.**

| Model | Params | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|
| DA 3 Base (non-causal) | 120M | **0.3003** | 0.5699 | 0.8901 | 0.0452 | 0.0623 |
| ZoeDepth N-K | 345M | 0.3973 | 0.5459 | 0.7789 | 0.1080 | 0.1388 |
| **SOKKANAEM** | **4.19M** | 0.4299 | 0.5939 | **0.3761** | **0.0398** | **0.0812** |
| DA V2 Small | 24.8M | 1.0121 | 0.7343 | 8.4370 | 0.5096 | 0.5278 |
| DA V2 Base | 97.5M | 1.0228 | **0.7398** | 9.5212 | 0.9094 | 0.9244 |
| DA V1 Small | 24.8M | 1.2032 | 0.7175 | 7.7437 | 0.5908 | 0.6103 |
| DPT-Large | 343M | 1.2035 | 0.6793 | 10.3811 | 0.7401 | 0.7629 |

Synthetic footage inverts the accuracy ranking. The relative-depth models, which lead on real indoor footage, produce AbsRel above 1.0 here: their disparity-space affine fit cannot span scenes with structure at hundreds of metres, and their \(\delta_1\) stays high while AbsRel explodes, which is the signature of a few catastrophically scaled clips rather than uniformly poor depth. We are third of seven on AbsRel behind two metric-capable models, and 2.1x better than the best relative model. Raw frame difference is 2.1x better than the runner-up and OPW is best in the group. TCE is not: DA3's 0.0623 beats our 0.0812.

**A correction to our own earlier reporting.** We previously claimed to beat a comparable-size model on every real-domain metric except \(\delta_1\). That claim rested on Depth Anything V2 Small's real AbsRel, which is an alignment artifact: on a handful of clips the fitted disparity approaches zero and inverting it sends predicted depth to the clip range, dominating the mean (its per-clip median is in line with the rest of the group). **We withdraw the claim.**

### 5.4 Alignment: why one rule for every model would be worse

Scale-ambiguous depth has to be aligned to ground truth before it can be scored, and the choice of rule is not neutral. Relative-depth models are trained to produce disparity up to an affine transform, so they are conventionally fitted with a two-degree-of-freedom scale and shift in disparity space. Metric models, and ours, are fitted with a one-degree-of-freedom per-clip median scale. Reporting each model under its native rule invites the objection that "one protocol" is not one protocol. We therefore ran the whole group under both.

**Table 12. Every model under its native alignment rule and under the other one. Real indoor holdout, eight-frame clips, 189 clips, AbsRel.**

| Model | Native rule | AbsRel, native | AbsRel, other rule |
|---|---|---:|---:|
| DA V1 Small | 2-DOF disparity | **0.0650** | 1.0620 |
| DPT-Large | 2-DOF disparity | 0.0875 | 0.8962 |
| DA V2 Base | 2-DOF disparity | 0.0877 | 0.8963 |
| DA V2 Small | 2-DOF disparity | 0.2068 | 0.8722 |
| ZoeDepth N-K | 1-DOF median | 0.0992 | 0.3782 |
| DA 3 Base | 2-DOF, depth space | 0.1130 | **0.1023** |
| **SOKKANAEM** | 1-DOF median | 0.1302 | 0.1155 |

**A single common rule would not be fairer; it would be meaningless for most of the group.** Fitting a relative-depth model with one degree of freedom in depth space raises its error by an order of magnitude, because the quantity being scaled is not the quantity it predicts. Fitting the metric baseline in disparity space costs it a factor of four. Neither number measures depth quality; both measure a protocol mismatch. The native-rule column is the only defensible main comparison, and the paper uses it.

Two things follow for our own claims. The extra degree of freedom is worth 11% to us (0.1302 to 0.1155), which is what Section 6.5 uses as evidence of a systematic error a single scale cannot absorb — and it is worth *less* to us than to any relative-depth model, so it is not the source of our accuracy gap. And DA3 is the one model that prefers the median rule, which is why we report it at 0.1130 in Table 3b rather than at its better 0.1023: quoting a baseline at its worse number would flatter us.

### 5.5 What preserved-state readout actually buys `[CHECKPOINT-DEPENDENT]`

\(\Delta\)-gating freezes hidden state at static positions but still reads it through \(C_i h_i\). A token-drop arm freezes the same state under the same masks and additionally bypasses the temporal block's output. Comparing the two isolates the value of the readout itself.

**Table 4. Gating-location ablation at matched masks (40.9% activity), reported checkpoint, synthetic holdout, 300 clips. Both arms run with the temporal cache disabled so that the \(\Delta\)-gating arm actually performs the dense readout under test.**

| Method | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|
| \(\Delta\)-gating | **0.4317** | 0.5918 | **0.1923** | **0.0310** | **0.0725** |
| Token drop | 0.4329 | **0.5935** | 0.2508 | 0.0357 | 0.0772 |

**This result reverses an earlier finding of ours and we report the reversal rather than the earlier number.** On an earlier checkpoint whose sparse path was an inference-time approximation never seen during training, token dropping collapsed: 1.7178 AbsRel against 0.4292 at 31.6% activity, a factor of four. On the confirmed checkpoint, trained with the sparse path in the loop and with randomised mask ratios, accuracy is a wash — 0.4317 against 0.4329 AbsRel, and \(\delta_1\) marginally favours token dropping — while the entire difference has moved into the temporal metrics: 30% worse raw frame difference, 15% worse OPW, 6% worse TCE. The long-clip checkpoint reproduces the same pattern (0.4347 against 0.4361, t-delta 0.2112 against 0.2601).

The honest reading is that the earlier experiment measured the fragility of an untrained sparse path, not the value of state readout. What survives is narrower and still meaningful: **reading preserved state buys temporal stability, not depth accuracy.** A model trained to tolerate missing static tokens recovers the accuracy on its own, but only the readout keeps consecutive predictions from moving. Since flicker suppression is the property this architecture is built around, the ablation still supports the design — it simply supports a smaller claim than we first made.

### 5.6 Cross-domain transfer and real moving cameras `[CHECKPOINT-DEPENDENT]`

We evaluate the confirmed checkpoint on five KITTI raw drives (Geiger et al., 2012) (885 frames) that appear in no training split. Only the synthetic clone of this domain was trained on, so the experiment isolates the synthetic-to-real axis rather than an arbitrary domain shift. Ground truth is projected LiDAR: capped near 80 m and 30% valid.

**Table 5. Zero-shot real driving against the in-domain synthetic holdout.**

| Setting | Active (%) | AbsRel | RMSE (m) | \(\delta_1\) | t-delta | TCE | Median scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| KITTI raw, zero-shot | **92.8** | 0.2894 | 11.03 | 0.4955 | 2.0716 | 0.0995 | 2.630 |
| Virtual KITTI 2 holdout, in-domain | 25.8 | 0.3619 | 33.89 | 0.3943 | 0.3115 | 0.0243 | 0.760 |

Accuracy does not collapse; it is nominally better on real footage. That ordering should not be read as a generalization result, because the two rows solve problems of different difficulty: the synthetic holdout contains structure at hundreds of metres that a 256-pixel input cannot resolve, while the LiDAR ground truth is capped and concentrated in the near field. The defensible statement is that representations learned on synthetic driving remain usable on real driving.

**What does not transfer is sparsity.** Activity rises from 25.8% to 92.8% on the same scene type. Measuring the detector alone with the deployment fallback disabled reproduces this: at a pixel threshold of 0.05, synthetic sequences leave 7-10% of patches active while real drives leave 40-74%. Sensor noise, exposure variation, rolling shutter, and compression artifacts all register as change. The skip ratios reported for fixed cameras are therefore measured values for that setting, and the synthetic driving ratios are optimistic.

**Moving-camera gating.** Pixel gating and GMC feature gating operate on different score scales, so comparing them at equal thresholds is meaningless — at their default thresholds the two are indistinguishable in accuracy while GMC uses more computation. The fair comparison is the activity-accuracy curve.

![Gating strategies](figures/gating.svg)

**Figure 5. Pixel gating against global-motion-compensated feature gating** on real driving footage, swept as curves. The two strategies score change on different scales, so only the curves are comparable and points at equal thresholds are not. At matched activity the compensated variant is better on both axes, and it reaches 14% activity while still beating pixel gating at 51%.

**Table 6. Same 30 clips, both gating strategies swept.**

| Gating | Active (%) | AbsRel | \(\delta_1\) | t-delta | TCE |
|---|---:|---:|---:|---:|---:|
| Pixel | 100.0 | 0.3083 | 0.5142 | 3.0852 | 0.1189 |
| Pixel | 92.7 | 0.3093 | 0.5089 | 3.1016 | 0.1236 |
| Pixel | 51.1 | 0.3357 | 0.4749 | 2.5108 | 0.1074 |
| GMC + feature | 87.1 | 0.3065 | 0.5170 | 3.0533 | 0.1173 |
| **GMC + feature** | **43.8** | **0.3084** | **0.5342** | 1.9319 | 0.0992 |
| **GMC + feature** | **14.1** | 0.3178 | **0.5341** | **1.2314** | **0.0922** |

At matched activity GMC is clearly better: 43.8% active gives 0.3084 AbsRel and 0.5342 \(\delta_1\) against 0.3357 and 0.4749 for pixel gating at 51.1% — less computation and 8.1% lower relative error. GMC at 14.1% activity still beats the pixel-gating point at 51.1% on every metric. Within the GMC curve, cutting computation sevenfold costs 3.1% relative AbsRel while \(\delta_1\) and both temporal metrics improve monotonically, reproducing on real footage the pattern previously observed only in simulation.

The practical caveat is that GMC's default threshold leaves real driving at 100% activity. The correct statement is not that enabling GMC suffices, but that GMC plus per-domain threshold calibration recovers most of the sparsity that pixel gating loses on real video.

Homography estimation itself does not fail on this footage: over 210 frames of KITTI raw at three thresholds, the identity fallback fired **0 times**. The failure mode this branch guards against — no texture, degenerate fit — is not what limits the moving-camera path here; the threshold scale is.

### 5.7 Compute and wall-clock analysis

Two results in this section point in opposite directions, and both matter.

**Analytical compute.** The current architecture costs 1.644 GMAC/frame at full activity. With both caches, 15.4% activity costs 0.608 GMAC — 37.0% of full. The decoder is 23.1% of the dense floor and patch embedding 2.3%, so the saving is real and comes from the backbone, where sparsity applies.

**Measured latency.** The scan implementation, not the gather, dominated wall-clock. Profiling a sparse frame at 22% activity attributes 71% of it to the spatial scan and only 6% to gathering and scattering active tokens. The reference scan is chunked and materialises a \((B, C, C, P, S)\) pairwise-decay tensor per chunk: at \(L=64\), \(P=384\), \(S=16\) it moves roughly 25 MB to perform 0.4 MMAC. We therefore replaced it with a fused Triton (Tillet et al., 2019) kernel that keeps the recurrence in registers, used at inference while training retains the differentiable chunked path. \(\Delta\)-gating remains bit-exact through the kernel — \(\widetilde{\Delta}=0\) gives \(\exp(0)=1\) and a zero input term — and every evaluation metric is unchanged to four decimal places.

![Latency before and after the fused kernel](figures/latency.svg)

**Figure 6. Per-frame latency before and after the fused scan kernel**, measured at 22% activity on one RTX 4090 at 256 pixels, batch size one, fp32. Every path became faster and the ordering inverted: what sparsity was saving was the scan, and the scan is now nearly free, leaving the sparse path with bookkeeping that does not scale with activity.

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

**Four efficiency claims, separately scored.** The word "efficient" covers four different assertions in this literature, and they have different evidence here. It is worth stating which is which, because the model *is* fast and that speed is not the mechanism's doing:

| Claim | Status | Where it comes from |
|---|---|---|
| Fewer parameters and MACs than the comparison group | demonstrated | 4.19M parameters, 1.644 GMAC dense; model scale, not sparsity |
| Lower per-frame latency than a comparable-size baseline | demonstrated | 1.29 ms compiled dense; model scale and the fused kernel, **not** the sparse path |
| MAC reduction from sparsity | demonstrated | 1.644 to 0.608 GMAC at 15.4% activity |
| Reduced per-stream state, so one weight set serves many streams | demonstrated | 6.38 MB fp16 state against 8.4 MB of weights |
| Wall-clock speedup *from sparsity* | **not demonstrated** | compiled dense is faster at every activity level on a 4090 |
| Energy or power reduction from sparsity | **not measured** | board-power sampling is implemented; the desktop card's idle floor dominates at this model scale, so the informative measurement is the edge one |
| Advantage on an edge accelerator | **not measured** | no Jetson-class device was available (Section 7) |

The headline latency and memory numbers therefore belong to a small model with a fused kernel, and the sparsity mechanism's established benefit is arithmetic and state, not time. A reader who takes "2.2x faster" as the sparse path beating the dense one has read the opposite of what we measured.

### 5.8 Streaming drift, and how much of it is ours `[CHECKPOINT-DEPENDENT]`

A streaming model is supposed to accumulate evidence across frames, so depth at frame 7 should be better than at frame 0 — that is the reason to carry state at all. The eight-frame clip mean conventional in this literature cannot see whether that happens, and for a long time we did not measure it either.

**The clip-length ladder.** The same two checkpoints, evaluated on disjoint clips of increasing length, with the keyframe period fixed at 30:

**Table 7a. Accuracy against clip length, real indoor holdout, dataset-balanced mean. Clip counts fall with length because the holdout is finite: 189 clips at eight frames, 13 at 256.**

| Clip length | Clips | Reported ckpt AbsRel | Penalty | Long-clip ckpt AbsRel | Penalty |
|---:|---:|---:|---:|---:|---:|
| 8 | 189 | 0.1302 | — | 0.1302 | — |
| 32 | 120 | 0.1487 | +14.2% | 0.1424 | +9.4% |
| 128 | 28 | 0.1961 | +50.6% | 0.1719 | +32.0% |
| 256 | 13 | 0.2434 | +86.9% | 0.1990 | +52.8% |

**The short-clip convention is optimistic by a factor we had badly underestimated.** We previously reported an eight-to-32-frame penalty of 11.2% and treated it as the size of the effect. Measured out to 256 frames it is 87% for the reported checkpoint. Deployment runs hundreds of frames, so this is the number that describes the setting the architecture is for.

**But not all of that penalty is drift, and the honest accounting needs a control.** Per-clip alignment fits one scale for the whole clip, so a longer clip gives the fit a harder job irrespective of any state. Stateless per-frame baselines measure that effect directly, since they have no state to drift: over the same clips, DPT-Large degrades by 116% and Depth Anything V2 Base by 145% from eight frames to 256 — *more* than our 87%. Depth Anything 3, which is handed the whole clip at once and is not causal, degrades by 3%.

So there are two effects and they point in opposite directions. The protocol penalises every causal model as the clip grows, and it penalises per-frame models hardest; carried state, drift and all, is a net advantage over a long stream rather than a liability. What remains true, and is the reason this section exists, is that **the carried state does not accumulate accuracy within a keyframe cycle** — the frame-index curve below shows error growing between refreshes — and that **an eight-frame protocol cannot see either effect**.

**Where inside a clip the error lives.** Scoring by frame index, with each frame aligned independently so the curve cannot be an artefact of one clip-level fit:

**Table 7b. Accuracy and consistency by frame index, 32-frame clips, full holdout (22 TUM and 98 Bonn clips), keyframe refresh every 30 frames. OPW and TCE are scored on the pair (t-1, t).**

| Frame | TUM AbsRel | TUM \(\delta_1\) | Bonn AbsRel | Bonn \(\delta_1\) | Bonn OPW | Bonn TCE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.1353 | **0.8616** | **0.1167** | **0.8994** | — | — |
| 4 | 0.1206 | 0.8559 | 0.1239 | 0.8858 | 0.0229 | 0.0255 |
| 8 | 0.1249 | 0.8431 | 0.1316 | 0.8630 | 0.0194 | 0.0216 |
| 12 | 0.1510 | 0.8327 | 0.1383 | 0.8405 | 0.0228 | 0.0250 |
| 16 | 0.1521 | 0.8068 | 0.1485 | 0.8197 | 0.0213 | 0.0236 |
| 20 | 0.1415 | 0.8041 | 0.1538 | 0.8246 | 0.0198 | 0.0226 |
| 24 | 0.1585 | 0.7983 | 0.1582 | 0.8162 | 0.0204 | 0.0227 |
| 28 | 0.1697 | 0.7935 | **0.1668** | 0.8095 | 0.0228 | 0.0250 |
| 31 | **0.1323** | 0.8413 | 0.1340 | 0.8695 | 0.0260 | 0.0283 |

**There is no accumulation benefit, and on dynamic scenes there is an accumulation cost.** Bonn degrades monotonically from 0.1167 at frame 0 to 0.1668 at frame 28 — 43% worse — and loses 9.0 points of \(\delta_1\) over the same span. TUM improves for the first few frames and then degrades. We previously put the Bonn figure at 61%, measured on the crowd sequence alone; on the full holdout it is 43%, and the shape is unchanged.

**The decay is in accuracy, not in motion-referenced consistency.** Bonn's OPW and TCE are flat across the cycle — 0.0229 at frame 4 against 0.0228 at frame 28 — while AbsRel and \(\delta_1\) move steadily. What drifts is where the surface is placed, not how consistently it moves, which is exactly the distinction the two metric families exist to draw and the reason a single "temporal consistency" claim would be wrong in both directions.

The recovery at frame 31 is the keyframe: a full refresh fires at frame 30 and the error snaps back, on Bonn recovering 6 points of \(\delta_1\) at once. The refresh is not free in the other direction: raw frame difference *spikes* at the keyframe (TUM 0.0824 at frame 28 against 0.1751 at frame 31), because a full recomputation is a discontinuity in the output sequence. Accuracy sawtooths down and flicker sawtooths up, out of phase. Long-clip fine-tuning flattens both: on the same clips its Bonn curve runs 0.1139 to 0.1548, a 36% rise rather than 43%, and it holds 0.8380 \(\delta_1\) at frame 28 against 0.8095.

![Depth and error either side of a keyframe](figures/sawtooth.png)

**Figure 8. The sawtooth, as pictures.** One 32-frame clip of the dynamic-object holdout; rows are frames 24, 28, 29, 30 and 31; columns are RGB, prediction, ground truth and relative error (black is invalid ground truth). Frame 30 is the keyframe: activity goes to 100% and clip AbsRel falls from 0.3740 to 0.2030 in one frame, visible as the error column darkening over the moving person. The prediction column is also where range compression (Section 6.5) shows without a histogram — it is uniformly flatter than the ground-truth column that shares its colour scale.

**Out to 256 frames the sawtooth rides a trend and then levels off.** On disjoint 256-frame clips, Bonn's per-frame error grows from 0.0912 at frame 0 to 0.2817 by frame 224 — a factor of three — and then falls back to 0.1712 at frame 255. The error is bounded rather than divergent, but the trend over the first two hundred frames is real, and a keyframe period tuned on 32-frame clips does not contain it.

Two consequences follow. First, the fix already exists in the architecture and is simply applied too rarely: the keyframe period is a knob on exactly this decay.

**Table 7c. Keyframe period against accuracy, stability and cost. 32-frame clips, full holdout, both checkpoints.**

| Period | Active (%) | Reported AbsRel | Reported \(\delta_1\) | Reported t-delta | Long-clip AbsRel | Long-clip \(\delta_1\) | Long-clip t-delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 39.4 | 0.1337 | 0.8499 | 0.0976 | **0.1281** | **0.8663** | 0.0951 |
| 10 | 29.4 | 0.1370 | 0.8464 | 0.0793 | 0.1313 | 0.8602 | 0.0817 |
| 15 | 26.0 | 0.1400 | 0.8393 | 0.0753 | 0.1340 | 0.8521 | 0.0788 |
| 30 (default) | 22.7 | 0.1487 | 0.8264 | 0.0682 | 0.1424 | 0.8384 | 0.0729 |
| 60 | **19.6** | 0.1510 | 0.8225 | **0.0570** | 0.1447 | 0.8345 | **0.0608** |

The trade is monotone in all three quantities and there is no free point on it: refreshing six times more often buys 12% relative AbsRel and 2.7 points of \(\delta_1\) for double the compute and 71% worse raw frame difference. A keyframe is by construction a discontinuity in the output sequence, so the metric this architecture leads on is the one that pays for accuracy. The long-clip checkpoint is better than the reported one at every period on accuracy and \(\delta_1\), which is what makes the period a free variable again: it can be lengthened to recover stability without giving back the accuracy the fine-tune bought.

Second, the cause is a train-deploy mismatch rather than a flaw in \(\Delta\)-gating itself. Training uses four-frame clips, so the model has never had to hold state through more than three consecutive gated frames. Randomised mask ratios make it robust to *how much* is skipped, not to *how long*.

**Testing that explanation.** We fine-tuned the reported checkpoint for 25k steps at clip length 24 — long enough that no mid-clip keyframe fires, so the model must hold state through 23 consecutive gated frames. The diagnostic quantity is not the absolute error but the gap between protocols, which is the Penalty column of Table 7a: the eight-to-32-frame penalty falls from 14.2% to 9.4%, the eight-to-128 from 50.6% to 32.0%, and the eight-to-256 from 86.9% to 52.8%. **The mismatch is a real contributor at every horizon, and its removal is worth 18% of absolute error at 256 frames** (0.2434 to 0.1990).

One observation comes with it, and it is about evaluation rather than about the model. The two checkpoints score **identically to four decimal places on eight-frame clips** — 0.1302 both. An evaluation restricted to short clips would have scored a 25k-step intervention that removes a fifth of the long-horizon error as having no effect whatsoever. The short-clip convention is not only optimistic about streaming, it is blind to improvements aimed at it.

**The refresh period moves with the checkpoint.** A model taught to tolerate sustained gating needs refreshing less often, and the saved refreshes buy back the stability that a longer period would otherwise cost:

**Table 8. The reported configuration against the long-clip checkpoint at a longer refresh period, under both clip lengths that matter.**

| Configuration | Clip | Active (%) | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reported ckpt, period 30 | 32 | 22.7 | 0.1487 | 0.8264 | 0.0682 | 0.0219 | 0.0298 |
| **Long-clip ckpt, period 60** | 32 | **19.6** | **0.1447** | **0.8345** | **0.0608** | **0.0195** | **0.0274** |
| Reported ckpt, period 30 | 256 | 23.9 | 0.2434 | 0.7134 | 0.0719 | 0.0291 | 0.0366 |
| **Long-clip ckpt, period 60** | 256 | **22.1** | **0.2087** | **0.7716** | **0.0665** | **0.0252** | **0.0329** |

The second configuration is better on every measure and cheaper, at both clip lengths: at 256 frames, 14% relative AbsRel, 5.8 points of \(\delta_1\), 8% on raw frame difference, 10% on TCE, and 1.8 points less activity. The two changes are coupled — swapping the checkpoint while leaving the period at 30 keeps the accuracy gain but gives back the stability (t-delta 0.0674 against 0.0665), and it is the saved refreshes that buy it back.

**We therefore treat the long-clip checkpoint as the model of record for the streaming protocol**, and report both in Table 3 so that the eight-frame comparison against the literature stays on the checkpoint it was measured with. Section 6.5 adds a further stage to this checkpoint, and the final configuration is stated there.

## 6. Ablations and Diagnostic Findings

### 6.1 Mask policy

MSE and cosine change scores perform similarly at matched activity. Keyframe intervals between the tested settings have little effect on short-clip accuracy. Training with i.i.d. random masks produced better robustness than detector-driven mask fine-tuning, contrary to the initial expectation that train–deployment mask matching would be essential.

### 6.2 DINOv2 feature distillation

Matching final backbone tokens to frozen DINOv2-small features does not improve depth accuracy: 0.4315 AbsRel against 0.4292 without it, measured on an earlier checkpoint pair. It yields small improvements in temporal metrics (TCE 0.0846 versus 0.0879), suggesting regularization rather than better geometric representation.

### 6.3 Design decisions behind the reported checkpoint

Three choices in the final recipe were made from measurements rather than intuition, and each carries a cost worth stating.

**Binned depth head.** A 64-bin cross-entropy term alongside the regression loss is worth 8.6% relative AbsRel on real footage (0.1795 against 0.1963), measured across three seeds per arm so the effect is three to six times the seed noise. It is not free: it costs 8% on raw frame difference and 12% on TCE. The reported model keeps it and buys the temporal loss back with the auxiliary terms below.

**Dense fallback, and a claim it no longer supports.** A frame whose activity exceeds a threshold is routed through the dense path. We previously reported that this improves real accuracy (AbsRel 0.1685 to 0.1633) at the cost of raising mean activity, and adopted 40% as the default. Sweeping the threshold over the full holdout says otherwise:

**Table 9. Dense-fallback threshold, real indoor holdout, eight-frame clips, reported checkpoint.**

| Threshold | Active (%) | AbsRel | \(\delta_1\) | t-delta | OPW | TCE |
|---|---:|---:|---:|---:|---:|---:|
| off | **16.1** | **0.1293** | 0.8573 | 0.0596 | 0.0178 | 0.0257 |
| 0.2 | 36.5 | 0.1303 | **0.8665** | **0.0546** | **0.0175** | **0.0252** |
| 0.3 | 27.4 | 0.1302 | 0.8642 | 0.0569 | 0.0181 | 0.0258 |
| 0.4 (previous default) | 22.0 | 0.1302 | 0.8613 | 0.0607 | 0.0184 | 0.0262 |
| 0.6 | 16.9 | **0.1293** | 0.8582 | 0.0604 | 0.0183 | 0.0262 |

**AbsRel is flat across the sweep** — 0.1293 to 0.1303, a spread five times smaller than the seed noise floor — while activity varies by a factor of 2.3. What the fallback actually buys is one point of \(\delta_1\) and a 9% improvement in raw frame difference, for 20 points of activity at threshold 0.2. The earlier accuracy claim was measured on the first-sequence subset, where the crowd scene both dominates and is dense enough that the threshold fires often; on the full holdout it does not survive.

The consequence is an efficiency claim we can now make more cheaply: **disabling the fallback reaches 16.1% mean activity at the same accuracy** as the 22.0% default. We keep the threshold in the architecture because it is the natural place to trade stability for compute, and we report accuracy claims at the threshold each table names.

**Auxiliary losses, and a methodological caution.** A depth-boundary-weighted term and a flow-warped residual term are both enabled at weight 2.0. Screened at 8k steps, they looked like a trade: the boundary term improved accuracy while worsening temporal metrics, and the warp term did the reverse, degrading \(\delta_1\) beyond the noise floor. At 60k steps the trade disappears and both improve together (Table 2). **The ranking of loss terms at 8k did not survive to convergence.** We report this because short screening runs are standard practice for choosing loss weights, and in our case they were reliable for deciding whether a term helps but not for deciding how strongly to weight it.

### 6.4 Where the remaining accuracy error lives

Pushing ground truth through the model's own output bottleneck — average-pool to the token grid over valid pixels, bilinear back up, median-align — measures what a perfect patch-token head could score. On the real indoor holdout at the current patch size of 16, that ceiling is 0.0858 AbsRel and 0.9150 \(\delta_1\) on TUM and 0.0367 and 0.9786 on Bonn, against our 0.1321 and 0.1283. The dataset-balanced ceiling, 0.0613, is better than every model in the comparison group.

**Patch size and input resolution are therefore not the bottleneck.** Halving the patch to 8 lifts the ceiling to 0.0474 and 0.0204, but there is no reason to buy headroom that the model is not using. Capacity, optimisation, and dynamic-scene handling are what stand between the model and its current ceiling.

The gap is concentrated, and more sharply than we previously reported: TUM sits 1.5 times above its ceiling while Bonn sits **3.5 times** above, and Bonn is the source with moving people and occlusion. On the first-sequence subset this asymmetry read as three times against two; on the full holdout Bonn's ceiling drops to 0.0367 — its scenes are geometrically easy for a patch-grid output and hard for our model, which is the sharpest statement in this section. Closing half of Bonn's gap alone would bring the dataset-balanced mean from 0.1302 to 0.107. This also suggests a specific suspect among the auxiliary losses — the flow-warped residual term assumes photometric correspondence, which is precisely what fails at a moving object's depth discontinuity.

This measurement is easy to get wrong. Our first version pooled ground truth without a validity mask, so sensor holes averaged in; in disparity space a single zero becomes 1/eps and dominates its patch, giving 11.4 AbsRel on TUM, and the corresponding depth-space numbers suggested the model had already surpassed the ceiling — the opposite conclusion.

### 6.5 Range compression, a cheap fix that failed, and one that works

Section 5.4 reports our model under both alignment rules, and the two-degree-of-freedom fit improves real AbsRel from 0.1302 to 0.1155 — 11%. An extra degree of freedom helping that much means a systematic error the one-parameter fit cannot absorb.

Splitting by clip localises it. The two-degree-of-freedom fit helps 78% of Bonn clips (median improvement 9.6%) but only 31% of TUM clips, where the median clip is actually worse. Measuring the predicted and ground-truth disparity distributions on the same clips explains why:

**Table 10. Predicted dynamic range as a fraction of ground truth, full holdout.**

| Checkpoint | Source | Range ratio | Std ratio |
|---|---|---:|---:|
| Reported | TUM | 0.93 | 0.90 |
| Reported | Bonn | **0.75** | **0.74** |
| Long-clip | TUM | 0.94 | 0.91 |
| Long-clip | Bonn | 0.78 | 0.77 |

**The model compresses range on the dynamic-object source, by about a quarter.** We previously reported this ratio as 0.47 — less than half — but that measurement was the crowd sequence alone; over the full holdout it is 0.75. The direction of the effect is unchanged and it is still concentrated on Bonn, where the true range is widest, but its magnitude is a quarter rather than a half, and any statement resting on the larger figure has to be read down accordingly. It remains one phenomenon behind three symptoms: the low \(\delta_1\) (a ratio metric punishes a flattened field), the Bonn-specific alignment gap, and Bonn sitting 3.5 times above its structural ceiling in Section 6.4.

The obvious suspect was the decoder. The binned head predicts depth as a softmax expectation over log-depth bin centres, and an expectation pulls toward the distribution mean whenever the model is uncertain. That hypothesis is testable without retraining, by sharpening the softmax at inference:

| Temperature | Bonn AbsRel | Bonn \(\delta_1\) | Range ratio |
|---:|---:|---:|---:|
| 1.00 | 0.2618 | 0.7315 | 0.58 |
| 0.50 | 0.2642 | 0.7320 | 0.59 |
| 0.25 | 0.2677 | 0.7278 | 0.58 |

**The range ratio does not move and accuracy gets slightly worse.** The bin distributions are already peaked; the compression is in the predicted centres themselves. The model genuinely predicts a flattened field, which rules out a decoding fix and also rules out longer-clip training as a remedy — drift and range compression are separate defects with separate causes.

What the diagnosis does point at is the objective. Nothing in it penalises compression: the scale-invariant log loss punishes the mismatch indirectly, but under uncertainty shrinking the prediction still lowers it, which is the ordinary bias-variance trade. So we added a term that penalises it directly: the squared log ratio of the standard deviation of predicted log depth to that of ground truth, taken per sample over valid pixels rather than per batch, since compression is a property of a scene and a wide scene would otherwise cancel a narrow one. Comparing in log space makes the term scale-free and symmetric, so over-spreading is penalised as much as under-spreading and the loss cannot be bought with noise. We fine-tuned the reported checkpoint for 8k steps at three weights, with everything else held fixed. All three arms run at the same 22.0% activity, so nothing here is bought with computation.

**Table 11. A spread term against the compression it targets. Same 8k fine-tune from the reported checkpoint, same activity, real indoor holdout.**

| Spread weight | AbsRel | \(\delta_1\) | t-delta | OPW | TCE | TUM range/GT | Bonn range/GT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 (control) | 0.1286 | 0.8633 | 0.0656 | 0.0179 | **0.0257** | 0.93 | 0.77 |
| 0.5 | **0.1254** | **0.8638** | 0.0720 | **0.0178** | **0.0256** | 0.94 | 0.85 |
| 2.0 | 0.1293 | 0.8634 | **0.0637** | 0.0188 | 0.0266 | 0.94 | **0.90** |

**The term moves the quantity it targets, and only that quantity clearly.** Bonn's range ratio recovers monotonically with the weight, 0.77 to 0.90, while TUM — already at 0.93 — does not move. The intervention lands exactly where the diagnosis said the defect was, which is the strongest evidence that the diagnosis is right.

**The accuracy gain is not claimable.** Dataset-balanced AbsRel improves from 0.1286 to 0.1254 at weight 0.5, which is 0.0032 — inside the \(\pm\)0.005 seed-noise floor — and \(\delta_1\) moves by half a point of noise. Weight 2.0 is no better than the control. Measured on the first-sequence subset the same arms read as a 5.5% AbsRel improvement, and that figure does not survive the full holdout. What survives is a mechanism that corrects a specific, separately measured defect at no cost in compute and a small cost in raw frame difference (10% at weight 0.5), which is worth having for the range itself rather than for the error metric.

Widening a predicted field necessarily lets it move more, so a term that fights compression works against the flicker suppression this architecture is built for. The useful statement is that range compression is a defect of the objective rather than of the decoder or the gating, and that it is correctable at a stated price.

## 7. Limitations

The present study has several important limitations.

1. Every table in Section 5 comes from one checkpoint, but two diagnostic results in Section 6 (mask policy, feature distillation) were measured on earlier checkpoints and are reported as such rather than re-run.
2. The comparison group now includes Video Depth Anything (metric, Small), the video-specific class that was missing, and Depth Anything 3, which is non-causal and therefore not a streaming competitor at all — we report it because it is strong, not because the comparison is like for like. Neural Video Depth Stabilizer is still absent.
3. OPW and TCE do not support a general temporal-consistency lead. Raw t-delta can be gamed by constant predictions and is interpreted only alongside accuracy and the constant control.
4. A fused scan kernel removed the dominant cost and, in doing so, removed the sparse path's wall-clock advantage on an RTX 4090: compiled full compute is now faster at every activity level. The remaining sparse-path cost is activity-independent bookkeeping. Sparsity's benefit is presently established in MACs and per-stream state, not in measured latency on this device.
5. Execution on an RTX 4090 is overhead-bound at this model scale, so reduced analytical MACs do not become higher FPS. Jetson Orin latency, energy, and multi-stream measurements have not been performed, and they are the measurement that decides whether the MAC reduction is worth anything in deployment.
6. GMC is validated on real ego-motion (Section 5.6), but only on five driving sequences from one dataset, and only with per-domain threshold calibration; its default threshold is inoperative on real video. Handheld and aerial motion remain untested.
7. Cross-domain evaluation covers real driving only. Unseen indoor domains (NYU, ScanNet) were not evaluated: the former's host was unreachable and the latter requires a signed agreement. The claim of generalization is therefore limited to the synthetic-to-real axis of one scene type.
8. Accuracy degrades between keyframes, and clip-length dependence is severe: the reported checkpoint scores 0.1302 AbsRel on eight-frame clips and 0.2434 on 256-frame clips (Section 5.8). Part of that is the per-clip alignment window rather than drift — stateless baselines degrade more over the same clips — but we cannot separate the two exactly, only bound our share by the stateless controls. Behaviour beyond 256 frames is unmeasured; the holdout sequences run to 1,294 frames, so the measurement is available and simply not yet done.

9. The 256-frame protocol rests on 13 disjoint clips of real footage, because a finite holdout yields few long clips. Overlapping windows would multiply the count at the cost of independence, and one outlier frame then aliases into several frame indices. Differences of a few percent at that clip length are not resolvable, and we do not claim any.
10. The predicted depth field has under half the ground truth's dynamic range on the dynamic-object source (Section 6.5). A spread term recovers part of it at a stated cost in motion-referenced consistency; the defect is not eliminated.
11. Patch-size, refinement, and fully trained decoder/cache ablations remain incomplete.
12. Results are single-seed at 60k steps. Seed variance was characterised only at 8k steps (Appendix A), and differences below that noise floor are not claimed.

## 8. Conclusion

SOKKANAEM demonstrates that patch-level visual change can control an SSM through its discretization step, turning a static observation into an exact identity transition on temporal state rather than a suppressed update. Across synthetic and real RGB-D evaluations, patch sparsity costs little depth accuracy — a 22-fold cut in the update rate costs 6.4% relative error on real indoor footage — and the model suppresses raw frame-to-frame variation better than any baseline in the comparison group, including the video-specific one, while remaining 6x to 82x smaller. That is a statement about raw prediction variation only. It does not establish an advantage in motion-compensated or ground-truth-referenced consistency, where a 120M baseline is ahead of us under both protocols.

The experiments also mark the boundary of the idea, and most of the work of this paper was finding those boundaries rather than the result inside them.

**On the mechanism.** Exact state skipping is not end-to-end sparse inference: dense readout, spatial context and decoding remain, and only the temporal state transition is exact. An iso-mask token-drop control, which we had read as proving that reading preserved state is critical, proves something narrower once the sparse path is trained — the readout buys stability, and the accuracy it seemed to buy was an artefact of an untrained path.

**On efficiency.** A fused scan kernel closed the kernel gap and, unexpectedly, made dense streaming the faster configuration on a desktop GPU, which relocates the efficiency argument from latency to arithmetic and per-stream state until an edge device settles it. That measurement remains the single most informative experiment left undone.

**On evaluation.** Two protocol choices moved our own numbers further than any architectural change in this paper. A clip cap that sampled the first held-out sequence rather than the holdout was worth 0.03 AbsRel and ten points of activity, and it inflated three separate diagnostic findings — a dense-fallback accuracy gain that does not exist, a range compression twice its true size, and a spread-term improvement inside the noise floor. Clip length was worth 87% of our error, and the eight-frame convention this literature uses is blind to a fine-tune that removes a fifth of the long-horizon error. We report both because a reader has no way to discover either from a table that does not name its protocol.

**What remains open.** Accuracy is limited by drift between keyframes and by a predicted depth field at three quarters of the true dynamic range on dynamic scenes — not by patch size or output resolution, which sit 1.5 to 3.5 times above where the model operates. The drift is largely a training artefact: long-clip fine-tuning removes 18% of the 256-frame error and a longer refresh period converts the rest into stability. Range compression is correctable in the objective, at a cost in the metric this architecture leads on. Within these boundaries, exact \(\Delta\)-gating provides a principled foundation for change-adaptive streaming vision, and a fairly complete map of where its advantages stop.

## Appendix A. Reproducibility `[CHECKPOINT-DEPENDENT]`

**Model.** Dimension 192, four alternating temporal/spatial blocks, state dimension 16, four-direction spatial cross-scan, depthwise local convolution branch, DPT-style decoder with a 64-bin depth head over 0.3-150 m. 4,185,872 parameters; 16.7 MB fp32 weights; 12.75 MB of persistent state per stream in fp32 and 6.38 MB in fp16. Stream state lives entirely in an external dictionary, so one set of weights serves many streams without leakage.

**Training.** 60,000 steps, seed 0, input 256x256, single RTX 4090, 13 h 11 min. Loss is scale-invariant log depth plus 0.5 gradient, 0.1 temporal, 0.05 normal, a 64-bin cross-entropy term at weight 0.2, and two auxiliary terms at weight 2.0 — a flow-warped log-depth residual and a depth-boundary-weighted term. Mask ratios are sampled i.i.d. during training rather than taken from the detector.

**Optimization.** AdamW, learning rate 3e-4, weight decay 0.01 (the PyTorch default), gradient-norm clipping at 1.0, 1,000 linear warm-up steps followed by cosine decay to zero, full fp32 (no mixed precision). Batch is four clips of four frames, so 16 frames per step. Evaluation uses shadow EMA weights with decay 0.999, not the raw parameters.

**Data sampling and augmentation.** The five sources are drawn through a weighted sampler that equalizes per-dataset draw probability, so the largest source does not dominate the gradient any more than it dominates the reported mean. Augmentation is drawn once per clip and applied to every frame in it — random-resized crop (scale 0.55–1.0 of the shorter side, random position), horizontal flip with probability 0.5, and brightness and contrast jitter in 0.75–1.3 on RGB only. Clip-consistent transforms are not a convenience: a per-frame transform would inject apparent motion, which the change detector would register as activity and the temporal loss would penalise. Depth is never photometrically altered. The random mask ratio ramps from 0 to 0.5 over training.

**Temporal metric definitions.** t-delta is the mean absolute difference between consecutive predicted depth maps, in metres, computed *after* per-clip alignment and over every pixel — a prediction is defined everywhere, so t-delta needs no GT-validity mask. Alignment order matters: our own model and every baseline are scored through one implementation, after an earlier version of this pipeline measured t-delta on raw output for our model and on scale-aligned output for the baselines, which is a difference of the scale factor itself. It is not normalised by depth, which is why its magnitude tracks a scene's depth range and why synthetic and real columns are not comparable to each other. OPW and TCE are normalised by ground-truth depth, and both are averaged over pixels that are valid in both frames of a pair and land in-bounds after warping. Flow comes from RAFT-small (torchvision `Raft_Small_Weights.DEFAULT`) applied to the 256-pixel RGB frames the model sees, scaled to [-1, 1], last refinement iteration. Occlusion is handled by the in-bounds test and the warped GT-validity mask only, without a forward-backward consistency check; every model is scored through the identical mask, so the comparison is fair even where absolute values would not match another paper's definition.

**Memory.** Weight and per-stream state memory scale differently, and a streaming deployment cares about the second:

| Component | fp32 | fp16 |
|---|---:|---:|
| Weights \(W\), shared across streams | 16.7 MB | 8.4 MB |
| Persistent state \(S\), per stream | 12.75 MB | 6.38 MB |
| Peak working set, single sparse stream | — | 37 MB |

Serving \(N\) streams from one set of weights costs \(W + N \times S\): 8.4 MB + 6.38N MB in fp16. State overtakes weights at two streams, which is the regime the external state dictionary exists for.

**Detector defaults.** \(\tau_{\mathrm{on}}=0.05\), \(\tau_{\mathrm{off}}=0.025\), one-patch dilation, keyframe refresh every 30 frames, dense fallback above 40% activity. For GMC these thresholds are on a feature scale and must be recalibrated per domain (Section 5.6).

**Evaluation.** 100 clips per source, 8 frames per clip, held-out sequences only, per-clip median alignment before every metric. Temporal metrics use RAFT-small flow. Every full temporal table carries the per-clip optimal constant-depth control.

**Measured variance.** Seed variation was estimated from six 8k-step runs: real AbsRel \(\pm\)0.005, real \(\delta_1\) \(\pm\)0.004, synthetic \(\delta_1\) \(\pm\)0.015. Differences smaller than these are not claimed anywhere in this paper. The 60k runs are single-seed, so absolute numbers at that scale have no confidence interval.

**Timing protocol.** Batch size 1, 256 pixels, 100 iterations after 20 warm-up, fastest of three repetitions, on an otherwise idle GPU. Activity is forced by a detector stub so the x axis is identical across configurations, and the dense-fallback policy is disabled during timing so that each configuration is actually measured.

## References

Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or propagating gradients through stochastic neurons for conditional computation*. arXiv:1308.3432.

Bhat, S. F., Alhashim, I., & Wonka, P. (2021). AdaBins: Depth estimation using adaptive bins. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 4009–4018.

Campos, V., Jou, B., Giró-i-Nieto, X., Torres, J., & Chang, S.-F. (2018). Skip RNN: Learning to skip state updates in recurrent neural networks. *International Conference on Learning Representations*.

Cabon, Y., Murray, N., & Humenberger, M. (2020). *Virtual KITTI 2*. arXiv:2001.10773.

Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., & Kang, B. (2025). Video Depth Anything: Consistent depth estimation for super-long videos. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations*.

Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth map prediction from a single image using a multi-scale deep network. *Advances in Neural Information Processing Systems*, 27.

Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 3354–3361.

Gu, A., & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv:2312.00752.

Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*.

Habibian, A., Ben Yahia, H., Abati, D., Gavves, E., & Porikli, F. (2021). Skip-convolutions for efficient video processing. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2695–2704.

Kong, L., Wu, B., Chen, Y., Zhang, X., & Sun, J. (2022). *EViT: Expediting vision transformers via token reorganization*. arXiv:2202.07800.

Liang, F., et al. (2023). Eventful transformers: Leveraging temporal redundancy in vision transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision*.

Maduabuchi, C., & Wang, J. (2026). *Event-driven video generation*. arXiv:2603.13402. To appear, *European Conference on Computer Vision*.


Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., & Liu, Y. (2024). VMamba: Visual state space model. *Advances in Neural Information Processing Systems*, 37.

Palazzolo, E., Behley, J., Lottes, P., Giguère, P., & Stachniss, C. (2019). ReFusion: 3D reconstruction in dynamic environments for RGB-D cameras exploiting residuals. *IEEE/RSJ International Conference on Intelligent Robots and Systems*.

Parger, M., Tang, C., Twigg, C. D., Keskin, C., Wang, R., & Steinberger, M. (2022). DeltaCNN: End-to-end CNN inference of sparse frame differences in videos. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 12497–12506.

Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., & Hsieh, C.-J. (2021). DynamicViT: Efficient vision transformers with dynamic token sparsification. *Advances in Neural Information Processing Systems*, 34.

Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 12179–12188.

Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, R. (2022). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(3), 1623–1637.

Tang, K., Zheng, J., Jin, Y., Qiu, Y., Sun, G., Yan, Z., & Wong, W.-F. (2026). *SpikySpace: A spiking state space model for energy-efficient time series forecasting*. arXiv:2601.02411.

Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012). A benchmark for the evaluation of RGB-D SLAM systems. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 573–580.

Teed, Z., & Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical flow. *European Conference on Computer Vision*, 402–419.

Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An intermediate language and compiler for tiled neural network computations. *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 10–19.

Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., & Scherer, S. (2020). TartanAir: A dataset to push the limits of visual SLAM. *IEEE/RSJ International Conference on Intelligent Robots and Systems*.

Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. *Advances in Neural Information Processing Systems*, 37.

Zhang, Y., et al. (2023). *Vision Mamba: Efficient visual representation learning with bidirectional state space model*. arXiv:2401.09417.

Zheng, Y., Harley, A. W., Shen, B., Wetzstein, G., & Guibas, L. J. (2023). PointOdyssey: A large-scale synthetic dataset for long-term point tracking. *Proceedings of the IEEE/CVF International Conference on Computer Vision*.

**Bibliographic entries still to verify against the originals before submission.** Depth Anything 3 (used as a measured baseline throughout; author list, venue and year unconfirmed). TartanAir V2 (the entry above is the original TartanAir paper; whether V2 has its own citable reference is unconfirmed). Vision Mamba and Eventful Transformers (venue, page numbers and full author lists unconfirmed). NVDS is cited in the related-work discussion but has no entry yet. The Skip RNN, SpikySpace and event-driven video generation entries were checked against their arXiv records; the last is listed as to appear at ECCV 2026 and should be re-checked for final page numbers.
