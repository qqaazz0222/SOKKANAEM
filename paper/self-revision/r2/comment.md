라운드 2 기준으로 보면 **원고가 상당히 많이 좋아졌습니다.** self-revision-r1에서 제가 Major로 지적했던 핵심 사항 대부분이 실제 추가 실험으로 해결되었습니다. 특히 ① 256/512/1024-frame 장기 streaming 평가, ② Video Depth Anything 추가, ③ Jetson Nano/Raspberry Pi 실측, ④ alignment protocol 분석, ⑤ GMC calibration split, ⑥ dense fallback sensitivity, ⑦ 일부 multi-seed 측정까지 들어간 것은 단순 rebuttal 수준이 아니라 논문의 성격 자체를 훨씬 탄탄하게 만들었습니다.

제 Round-2 recommendation은 **Minor Revision에 가까운 Weak Accept**입니다. 다만 아래의 **Major Comment 1은 claim을 반드시 수정해야 한다고 봅니다.** 추가 대규모 실험이 반드시 필요하다기보다는, 현재 새로 얻은 결과의 해석을 조금 더 엄밀하게 만들어야 합니다.

---

# Round 2 Review

## Overall assessment

The revised manuscript is substantially stronger than the previous version. Most of my previous major concerns have been directly addressed through new experiments rather than only through textual clarification. In particular, the authors now include a video-specific baseline, long-horizon streaming evaluations up to 1,024 frames, edge-device latency/energy measurements, explicit cross-alignment comparisons, a non-oracle GMC calibration experiment, dense-fallback sensitivity analysis, and partial multi-seed characterization.

The paper’s strongest aspect remains its unusually careful effort to identify the boundary of its own claims. The revised manuscript now makes a much more defensible contribution: **exact change-gated temporal state preservation is useful for suppressing raw prediction variation and reducing conditional computation, but neither temporal accuracy nor hardware speedup follows automatically from the mechanism.** This positioning is much clearer than in the previous version. The introduction explicitly limits “exact” to the temporal hidden-state transition and separates it from approximate spatial caching and dense decoding. 

I am therefore broadly positive about the revision. My remaining concerns are primarily about the interpretation of the long-horizon alignment experiment and statistical strength of the 256/512-frame comparisons.

**Recommendation: Minor Revision / Weak Accept after clarification.**

---

# Response to previous Major Comments

## R1-1. Short-clip evaluation was inconsistent with the streaming claim  
### Status: **Largely resolved**

이 부분은 매우 잘 보완되었습니다.

이전에는 사실상 8-frame 결과가 main result였는데, revised version에서는 **256-frame을 streaming primary protocol로 승격**했고, 동일 baseline들을 8-frame과 256-frame에서 함께 비교합니다. 

또한 clip-length ladder를 8 → 32 → 128 → 256 → 512까지 확장했고, base model의 AbsRel이 0.1302에서 0.2972까지 악화되는 것을 명시적으로 보여줍니다. Long-clip fine-tuning의 효과 역시 horizon이 길어질수록 커집니다. 

더 중요한 것은 1,024-frame experiment를 추가하여 clip-level score와 independently aligned per-frame score를 분리한 것입니다. Long-clip checkpoint에서 per-frame AbsRel은 frame 0→1023 동안 1.1%만 증가하지만 clip-level penalty는 36%입니다. 

이전 코멘트의 핵심 요구는 충분히 충족되었습니다.

단, 이 결과의 **해석 방식**에 대해서는 아래 Major Comment 1이 남습니다.

---

## R1-2. 개선된 long-clip checkpoint를 발견했는데 main model에 반영하지 않음  
### Status: **Resolved**

이번에는 training lineage를 Base → Long-clip → Final로 명확히 정의했고, final checkpoint가 실제 reported model입니다. 

특히 final checkpoint는 long-clip training 뒤 range-spread stage까지 포함하고 있으며, 왜 period 30을 최종 설정으로 택했는지도 period 60과 직접 비교해서 설명합니다. Final model에서는 period 30이 256-frame accuracy/δ1 면에서 더 좋고, period 60의 stability gain이 크지 않다는 근거가 있습니다. 

따라서 이전 버전에서 느껴졌던 “research log가 아직 끝나지 않았다”는 문제는 거의 사라졌습니다.

---

## R1-3. Video-specific baseline 부재  
### Status: **Resolved**

Video Depth Anything Small이 추가되었고, causal clip-wise evaluation이라는 점도 명시했습니다. DA3는 non-causal model임을 별도로 표시하고 있습니다. 

256-frame에서 SOKKANAEM의 t-delta 0.0692가 VDA의 0.0854보다 낮고, 8-frame에서도 0.0750 vs 0.0829로 raw frame difference advantage가 유지됩니다. 

이제 “video model과 비교하지 않았다”는 제 이전 비판은 해소되었습니다.

Neural Video Depth Stabilizer가 여전히 빠져 있지만 저자들도 limitation에서 이를 명시하고 있습니다.  저는 이것만으로 acceptance를 막지는 않겠습니다.

---

## R1-4. Temporal consistency claim이 과도함  
### Status: **Resolved**

이 수정은 특히 좋습니다.

Abstract부터

> “It does not lead motion-compensated or ground-truth-referenced consistency, and we claim no general temporal-consistency advantage.”

라고 명확히 제한했습니다. 

Conclusion도 raw prediction variation과 OPW/TCE를 명확히 구분합니다. 

Token-drop에 대해서도 이전의 “does not suffice” 대신 “accuracy는 동일하지만 preserved-state readout의 stability를 재현하지 못한다”로 수정되어 훨씬 정확합니다. 

이 코멘트는 해결되었습니다.

---

## R1-5. Efficiency claim: MAC reduction과 real speedup을 분리해야 함  
### Status: **Strongly resolved**

제가 Round 1에서 가장 중요하게 요구했던 추가 실험 중 하나인데, 기대 이상으로 보완되었습니다.

저자들은 이제 efficiency를 여러 claim으로 분해하여:

- small model 자체의 latency,
- MAC reduction,
- per-stream state,
- sparsity-induced wall-clock speedup,
- energy reduction

을 각각 구분합니다. 

RTX 4090에서는 여전히 dense compiled path가 sparse보다 빠르다는 결과를 그대로 유지하면서, Jetson Nano에서는 5.1% activity에서 **13.7× latency speedup / 15.3× energy reduction**, Raspberry Pi에서는 **14.2× speedup**을 측정했습니다.  

더 좋았던 점은 “energy가 줄어드는 이유가 instantaneous power 감소가 아니라 execution time 감소”라는 점까지 분리했다는 것입니다. 

다만 두 edge device 모두 shipped fused implementation을 실행하지 못한다는 limitation도 정직하게 남겨두었습니다. 따라서 이 부분은 현재 충분히 publishable한 수준이라고 생각합니다.

---

## R1-6. Novelty positioning 부족  
### Status: **Resolved**

새 Section 2.4가 상당히 효과적입니다.

Skip RNN, event-driven/spiking SSM, event-gated video generation을 직접 비교하면서 novelty를 “conditional update 자체”로 주장하지 않고,

1. external/untrained gate,
2. exact identity transition,
3. dense spatial output에서 temporal exactness와 spatial approximation의 분리

의 **combination**으로 한정했습니다.  

또 Introduction에서 “one algebraic identity is not by itself a contribution”이라고 직접 적은 것도 positioning 측면에서 좋습니다. 

---

## R1-7. GMC calibration이 oracle인지 불명확  
### Status: **Resolved**

이전에는 상당히 우려했던 부분인데 이번에는 제대로 통제했습니다.

저자들이 기존 sweep이 같은 test clips에서 threshold를 고른 oracle임을 직접 인정하고, **1개 drive에서 calibration → 나머지 4개 unseen drives 평가**로 분리했습니다. 

또 calibration target이 ground truth가 아니라 detector 자체의 activity ratio임을 명시하여 deployment 가능한 calibration임을 보여줍니다. 

Homography identity fallback도 1,260 frames에서 0회라는 failure statistic까지 추가되었습니다. 

충분히 해결되었습니다.

---

## R1-8. Statistical reliability  
### Status: **Partially resolved**

Final stage에 대해 3 seeds를 수행하고 8/32/256-frame protocol을 다시 평가한 것은 좋은 개선입니다. 8-frame AbsRel std가 ±0.0012, 256-frame period-30이 ±0.0019 등으로 나타납니다. 

또 저자들이 이 variance가 **전체 training pipeline variance가 아니라 마지막 8k-step stage의 variance일 뿐**이라고 정확히 제한하고 있습니다. 

따라서 과거보다는 훨씬 낫지만, base 60k 및 long-clip initialization이 single seed라는 점은 여전히 남습니다. Acceptance blocker까지는 아니지만 아래 Minor/Major에서 조금 더 언급하겠습니다.

---

## R1-9. Alignment protocol fairness  
### Status: **Resolved experimentally, but leads to a new interpretational issue**

이번에는 모든 model을 native alignment와 opposite rule 둘 다 적용했습니다. 

이는 제가 이전에 요구했던 sanity check를 충분히 충족합니다. Relative model을 1-DOF depth scaling하면 error가 거의 order-of-magnitude 커지고, metric model을 disparity affine alignment하면 역시 크게 나빠지므로 native rule을 main comparison에 쓰는 것이 타당하다는 설명에 동의합니다.

다만 long-clip degradation을 “alignment window artifact”라고 해석하는 부분은 별개의 문제가 생겼습니다.

---

# Remaining / New Major Comments

## Major Comment 1. “Most of the drift is the alignment window”라는 해석은 현재 증거보다 강합니다.

이번 Round 2에서 제가 가장 중요하게 수정 요청할 부분입니다.

Table 7d의 실험 자체는 매우 유용합니다. Long-clip checkpoint에서:

- clip-level AbsRel: 8-frame 대비 +36%
- independently aligned per-frame AbsRel: frame 0→1023에서 +1.1%

이라는 결과는 분명 중요합니다. 

하지만 여기서

> “Almost all of the clip-level penalty is the protocol”

또는

> “A benchmark that fits one scale to a long clip will report a degradation that says more about its own protocol than about any model”

까지 가는 것은 조금 지나친 해석입니다. 

왜냐하면 **per-frame alignment는 model의 temporal scale inconsistency를 정의상 제거하기 때문입니다.**

예를 들어 model이 매 frame마다 depth *shape*는 잘 예측하지만 global scale이 \(s_t\) 형태로 서서히 흔들린다면:

- per-frame alignment → 거의 완벽하게 제거
- one-scale-per-clip alignment → error 증가

가 발생합니다.

이 현상을 단순히 “benchmark artifact”라고 부르기는 어렵습니다. Metric-scale deployment라면 **frame-to-frame scale stability 자체도 모델의 temporal quality**이기 때문입니다.

Stateless baselines가 긴 clip에서 더 심하게 악화된다는 것도 “protocol이 잘못되었다”는 증거라기보다, 오히려 그 baseline들이 frame-to-frame scale consistency가 낮다는 증거일 수 있습니다.

### 제가 권하는 수정

추가 training은 필요하지 않습니다. 기존 prediction으로 다음 하나만 분석해도 충분합니다.

각 frame의 independently fitted scale factor \(s_t\)를 저장하고:

- std/log-std of \(s_t\)
- \(|\log s_t-\log s_{t-1}|\)
- clip length에 따른 scale variation

을 SOKKANAEM과 1–2개의 stateless baseline에 대해 보고해 주세요.

그러면 long-horizon penalty를

\[
\text{shape/depth drift} + \text{global scale drift} + \text{single-window alignment effect}
\]

정도로 더 정확하게 분해할 수 있습니다.

실험을 추가하지 않더라도 wording은 반드시 완화해야 합니다.

현재:

> “almost all … is the protocol”

보다는

> “most of the clip-level penalty disappears under per-frame alignment, indicating that temporally varying global scale/alignment rather than local depth-shape drift accounts for much of the difference”

정도가 더 defensible합니다.

이 수정은 중요합니다. 오히려 이 분석을 제대로 하면 **논문의 protocol contribution이 더 강해질 수 있습니다.**

---

## Major Comment 2. 256-frame main ranking에 대한 uncertainty를 더 직접적으로 보여줄 필요가 있습니다.

256-frame Table 3a는 main streaming result인데 **13 disjoint clips**에 기반합니다. 

저자들도 limitation에서 “differences of a few percent at that clip length are not resolvable”이라고 정확히 인정합니다. 

그렇다면 Table 3a에는 가능하면 **clip-level bootstrap CI** 또는 최소한 per-clip standard error를 넣는 것이 좋겠습니다.

특히 다음 claim:

> SOKKANAEM 0.0692 vs Video Depth Anything 0.0854, 1.23×

은 핵심 headline claim 중 하나입니다. 

차이가 충분히 커 보여 실제로 유지될 가능성이 높지만, 13 clips라는 작은 N을 고려하면 baseline pair의 uncertainty를 함께 보여주는 것이 훨씬 설득력 있습니다.

새 모델 training은 필요 없고 기존 per-clip outputs로 bootstrap하면 됩니다.

제가 acceptance를 막을 정도의 문제는 아니지만, **256-frame을 primary protocol이라고 선언한 이상 해당 primary table에는 uncertainty가 있어야 한다**고 봅니다.

---

# Minor Comments

1. **Abstract의 “clip-level score falls 36%” 표현은 방향상 조금 혼동됩니다.** AbsRel이 증가하는 것이므로 “worsens by 36%”가 더 자연스럽습니다. 현재 abstract에서는 per-frame error는 “rises”인데 clip-level score는 “falls”라고 써서 lower-is-better metric인지 순간적으로 헷갈립니다. 

2. Abstract가 여전히 상당히 깁니다. Round 1보다 message는 훨씬 명확하지만 1,024-frame, Nano, Pi까지 모두 들어가면서 거의 executive summary가 되었습니다. 핵심 result 3개 정도만 남겨도 충분해 보입니다.

3. **“22-fold cut in update rate”와 “MAC reduction”을 동일시하지 않도록 계속 주의해야 합니다.** Patch activity가 100→4.6%라고 해서 end-to-end MAC이 22× 감소하는 것은 아닙니다. 본문은 이를 알고 있지만 Figure 3 caption의 “22-fold cut in update rate” 표현을 그대로 유지하는 것이 좋으며 “compute”로 바꾸면 안 됩니다. 

4. Section 5.3의 “At matched accuracy we are 25–53% better…”는 regression-based normalization이므로 다소 강합니다.  “relative to the cross-model regression trend at our accuracy”라는 qualifier를 같은 문장 안에 넣는 것을 권합니다. Cross-model regression은 causal evidence가 아닙니다.

5. DPT-Large와 256-frame AbsRel이 0.1891 vs 0.1907로 거의 같은 상태에서 temporal metrics가 더 좋다는 matched-accuracy comparison은 regression보다 훨씬 설득력 있습니다. 이 사례를 먼저 강조하고 regression은 secondary analysis로 두는 편이 좋겠습니다.

6. **Final model과 deployment-preferred alternative가 약간 복잡합니다.** Final+period30은 accuracy-oriented, Long-clip+period60은 flicker-oriented라는 설명은 타당하지만, 한 표나 작은 “recommended operating configurations” box로 정리하면 독자 이해가 쉬워질 것입니다. 

7. Dense fallback 분석은 잘 추가되었습니다. 특히 fallback off에서 activity가 16.1%로 내려가면서 AbsRel이 사실상 유지되는 점은 중요한 결과입니다.  Abstract의 “16.1% at unchanged AbsRel and half a point less δ1”도 이 결과에 기반하므로 Table 9를 main text에서 조금 더 눈에 띄게 연결해 주세요.

8. GMC calibration에서 “one drive is enough to calibrate”라는 문구는 조금 강합니다. 한 calibration drive로 다른 4 drives에 transfer한 것은 보여주었지만, 다른 drive를 calibration source로 했을 때도 동일한지는 아직 알 수 없습니다. “one drive was sufficient in this split”이 더 정확합니다.

9. Jetson Nano와 Pi 결과에서 sparse path가 reference scan이고 desktop은 fused scan이라는 점은 매우 중요합니다. 현재도 여러 차례 명시되어 있으므로 좋지만, Figure/Table 제목에 **reference scan**을 굵게 넣어 독자가 숫자만 가져가는 것을 방지하면 좋겠습니다. 

10. Reproducibility appendix는 크게 개선되었습니다. AdamW, LR, warm-up, cosine schedule, clipping, batch, EMA 등이 추가되어 있습니다.  Round 1의 reproduction comment는 사실상 해결된 것으로 봅니다.

11. Memory도 \(W + N S\) 형태로 명시되어 다중-stream deployment interpretation이 좋아졌습니다. 

12. 여전히 **Data Availability / Acknowledgments가 placeholder**입니다. 실제 submission 전에는 반드시 채워야 합니다. 

13. Full-pipeline variance가 없는 것은 limitation으로 충분히 고지되어 있습니다.  다만 최종 manuscript에서 “2.5σ / 4.4σ” 표현을 쓸 때 이 σ가 final-stage conditional variance임을 독자가 바로 알 수 있게 표 caption에도 한 번 더 표시하는 것이 좋겠습니다.

14. 제목이 이전보다 훨씬 낫습니다. “Exact”의 대상도 본문에서 충분히 제한하고 있어 Round 1의 title concern은 해소되었습니다.

---

# Final Recommendation

제 Round 1 평가는 대략 **Major Revision / 6점 수준**이었는데, 이번에는 확실히 올라갑니다.

현재 제 판단은:

**Novelty: 7.5/10**  
**Technical quality: 8/10**  
**Experimental completeness: 8/10**  
**Transparency/reproducibility: 9/10**  
**Overall: 7.5–8/10**

**Recommendation: Minor Revision / Weak Accept**

핵심적으로는 더 이상 “실험이 부족해서 논문의 주장을 판단할 수 없다”는 상태가 아닙니다. 오히려 지금은 **실험이 충분히 많이 쌓였고, 그 결과를 얼마나 정확하게 해석하느냐가 마지막 문제**입니다.

특히 제가 editor에게 한 문장으로 쓴다면 다음 정도일 것 같습니다:

> *The revision addresses nearly all substantive concerns from the previous round through extensive new experiments, including long-horizon streaming evaluation, a video-specific baseline, edge-device measurements, calibration controls, and improved reproducibility. I am now broadly supportive of publication. My main remaining concern is interpretational: the manuscript currently attributes most long-horizon degradation to the per-clip alignment protocol, whereas independently aligning every frame also removes temporally varying global-scale error produced by the model. This distinction should be quantified or the corresponding claims should be softened. Subject to this clarification and minor presentation revisions, I would support acceptance.*

즉, **이번 라운드에서는 reject/major revision 쪽으로 갈 이유는 많이 사라졌습니다.** 가장 먼저 손볼 부분은 새로운 실험을 크게 더 하는 것보다 **“alignment artifact”를 “time-varying global scale/alignment component”로 엄밀하게 재해석하는 것**이라고 봅니다.