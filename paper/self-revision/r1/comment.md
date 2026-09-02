네. 전체 원고를 리뷰어 관점에서 보면, **아이디어 자체는 흥미롭고 저자들이 부정적 결과까지 상당히 정직하게 보고 있다는 점은 강점**입니다. 특히 “변화가 없는 patch에서는 selective SSM의 discretization step을 0으로 만들어 hidden state transition을 정확한 identity로 만든다”는 핵심 메커니즘은 깔끔합니다. 저자들도 정확도 우위를 주장하기보다 mechanism/efficiency–stability trade-off를 기여로 한정하고 있습니다.  다만 현재 버전은 **핵심 실험 프로토콜과 최종 모델 설정이 아직 완전히 정리되지 않았고, 실제 streaming/deployment 효율성에 대한 주장도 충분히 검증되지 않은 상태**라서, 제 판단은 **Major Revision**입니다.

## Overall assessment

제가 실제 리뷰어라면 대략 다음과 같이 평가할 것 같습니다.

**Recommendation: Major Revision**

논문의 장점은 ① ∆-gating의 수학적 정의가 단순하면서 명확하고, M=0일 때 state transition이 identity가 된다는 점을 논리적으로 제시한다는 것, ② token-drop control, constant-depth control, frame-index analysis처럼 자기 주장에 불리할 수 있는 실험까지 수행했다는 것, ③ sparse implementation이 실제 GPU latency에서는 이득이 없다는 사실까지 명시했다는 것입니다. 특히 fused kernel을 적용하면 RTX 4090에서 dense path가 sparse path보다 모든 activity level에서 더 빠르다는 결과를 숨기지 않은 점은 좋습니다. 

반면, 가장 중요한 문제는 **논문이 주장하는 대상이 “streaming video depth”인데 headline 결과 대부분은 8-frame clip protocol에서 산출되었고, 저자 스스로 이 protocol이 실제 streaming 성능을 낙관적으로 평가한다는 것을 보여주었다는 점**입니다. 같은 checkpoint가 8-frame에서는 AbsRel 0.1595이지만 32-frame에서는 0.1774로 악화되고, Bonn에서는 keyframe 사이 error가 61% 증가합니다.  이 문제는 단순 limitation이라기보다 논문의 main evaluation protocol 자체와 연결되므로 수정 후 재평가가 필요합니다.

---

# Major Comments

### 1. Main evaluation protocol과 논문의 “streaming” claim이 서로 맞지 않습니다.

현재 제가 가장 중요하게 보는 문제입니다.

논문은 streaming architecture를 핵심으로 제시하지만, 저자 스스로 “모든 기존 headline number가 8-frame clip mean이었다”고 밝히고 있으며, 32-frame 평가에서는 Bonn error가 frame 0의 0.1642에서 frame 28의 0.2636으로 증가합니다. 즉 약 61% 악화됩니다.  또한 동일한 checkpoint의 AbsRel이 8-frame 기준 0.1595에서 32-frame 기준 0.1774로 바뀝니다. 

이 결과를 발견하고 보고한 것은 매우 좋지만, 현재 manuscript에서는 이 사실이 **추가 diagnostic**처럼 들어가 있고, 여전히 주요 baseline comparison과 abstract headline은 short-clip 결과에 의존하고 있습니다.

따라서 저는 다음을 요구할 것 같습니다.

- Table 3의 주요 baseline comparison을 최소한 32-frame 또는 충분히 긴 continuous stream protocol로 다시 수행할 것.
- 가능하면 실제 deployment를 모사하는 **100–300+ frame continuous evaluation**을 main result로 추가할 것.
- 각 clip 시작 시 state를 reset하는지, dataset sequence 전체를 continuous하게 흘리는지 명확히 구분할 것.
- frame-index curve를 AbsRel뿐 아니라 δ1/TCE/OPW에도 제시할 것.
- short-clip evaluation은 보조 결과로 내리고, streaming claim은 long-horizon 결과를 기준으로 작성할 것.

특히 Limitations에서도 behavior beyond 270 frames는 측정하지 않았으며 “worse rather than flat”일 것으로 예상한다고 저자들이 인정합니다.  이런 상태에서 “streaming video-depth model”의 deployment relevance를 강하게 주장하기는 어렵습니다.

---

### 2. 저자들이 발견한 개선 모델을 최종 모델로 사용하지 않았기 때문에, 현재 paper의 main operating point가 사실상 obsolete합니다.

Section 5.7에서 저자들은 long-clip fine-tuning을 통해 8→32 frame penalty를 +11.2%에서 +4.8%로 절반 이하로 줄였습니다.  더 나아가 fine-tuned checkpoint + refresh period 60 조합은 기존 period 30보다 activity, AbsRel, t-delta, TCE를 동시에 개선합니다. 

즉 paper의 마지막 실험에서 **현재 reported checkpoint보다 더 나은 configuration을 이미 발견했습니다.**

그런데 main tables는 여전히 이전 checkpoint / K=30 설정으로 되어 있습니다. 실제로 Section 5.1에는 “default operating point is under review”라는 draft note까지 남아 있으며, K=10을 택하면 activity figure들이 약 6 percentage points 변한다고 적혀 있습니다. 

저널 submission이라면 이 상태는 받아들이기 어렵습니다.

저라면 다음 중 하나를 요구하겠습니다.

1. **새 checkpoint와 최종 K를 확정한 뒤 모든 main table을 재실행**, 또는  
2. 기존 checkpoint를 paper의 공식 model로 유지한다면, 왜 개선된 checkpoint를 채택하지 않았는지 methodologically 명확한 이유를 제시.

현재처럼 “더 나은 방법을 이미 찾았지만 baseline protocol이 달라 아직 main model로 승격하지 않았다”는 상태는 research log로서는 정직하지만, final manuscript로서는 incomplete합니다.

---

### 3. 가장 중요한 baseline인 video-specific depth model이 빠져 있습니다.

이 논문의 주요 주장 중 하나는 temporal stability입니다. 그런데 저자 스스로 Limitations에서 **Video Depth Anything이 baseline table에 없으며, 비교 대상 6개가 모두 가장 관련성 높은 video-specific class를 포함하지 않는다고 인정**합니다. 

References에는 Video Depth Anything과 Neural Video Depth Stabilizer가 인용되어 있는데, 실제 quantitative main comparison에서는 video-specific baseline이 없습니다. 

이는 상당히 큰 문제입니다. 이미지 기반 depth model들과 비교해서 t-delta가 낮다는 것은 어느 정도 예상되는 결과일 수 있기 때문입니다. 지속적인 temporal modeling이나 video stabilization을 목적으로 하는 모델과 비교해야 본 논문의 핵심 기여가 설득력을 갖습니다.

따라서 최소한 다음 중 하나 이상이 필요합니다.

- Video Depth Anything
- Neural Video Depth Stabilizer
- 가능하면 다른 lightweight video depth / recurrent depth baseline

그리고 동일한 input resolution, clip/stream length, alignment rule, temporal metric implementation에서 비교해야 합니다.

저라면 이 항목은 **acceptance 전 필수 보완사항**으로 보겠습니다.

---

### 4. Temporal consistency에 관한 main claim을 더 좁고 정확하게 정리해야 합니다.

Table 3 결과를 보면 SOKKANAEM이 명확하게 우수한 것은 **raw t-delta**입니다. 반면 real-domain에서 Depth Anything 3가 OPW와 TCE에서 더 좋습니다. 저자도 이를 명시하고 있습니다. 

그리고 t-delta는 constant output으로 trivially 최적화될 수 있기 때문에, 그 자체로는 temporal consistency의 강한 증거가 아닙니다. 저자들은 constant-depth control을 포함해 이 문제를 잘 인지하고 있습니다. 

문제는 일부 wording이 여전히 “temporal stability/consistency”를 다소 넓게 사용한다는 점입니다.

저는 논문의 핵심 claim을 다음 정도로 제한하는 것이 적절하다고 봅니다.

> “The method substantially suppresses raw frame-to-frame prediction variation, but does not establish a general advantage in motion-compensated or ground-truth-referenced temporal consistency.”

특히 title/abstract/introduction/conclusion에서 **stability, flicker, temporal consistency**를 엄격히 구분해야 합니다.

---

### 5. Efficiency claim은 현재 “computational sparsity”이지 “faster inference”가 아닙니다.

이 부분 역시 중요한 issue입니다.

Analytical MAC은 full 1.644 GMAC에서 15.4% activity일 때 0.608 GMAC까지 줄어듭니다.  하지만 fused kernel 이후 RTX 4090에서는 compiled dense full-compute가 1.29 ms이고 sparse path는 약 2.03–2.20 ms로 **dense가 더 빠릅니다.** 

따라서 paper의 efficiency story는 현재 다음과 같습니다.

- MAC reduction: demonstrated
- persistent-state / memory reduction: partly demonstrated
- wall-clock speedup from sparsity: **not demonstrated**
- energy reduction: **not measured**
- edge deployment advantage: **not measured**

이 구분은 본문에서 비교적 솔직히 쓰고 있지만, abstract 초반의 “0.378 ms/frame, 2.2× faster…” 문장은 독자가 sparse mechanism이 가져온 speedup이라고 오해할 가능성이 큽니다. 실제 abstract 후반에는 fused kernel이 sparse latency advantage를 없앴다고 다시 설명되어 있어 내부적으로도 message가 복잡합니다. 

저라면 다음을 요청하겠습니다.

- “model-level compactness/kernel optimization speed”와 “sparsity-induced speedup”을 분리해 보고할 것.
- sparse vs dense 동일 architecture의 latency를 headline으로 제시할 것.
- energy, power, latency를 Jetson/edge device에서 최소 1개 이상 측정할 것.
- activity에 따른 latency curve뿐 아니라 **measured energy/frame** 또는 throughput/W도 포함할 것.

저자들 역시 edge measurement가 “single most informative experiment left”라고 결론에서 인정하고 있습니다.  저 역시 동의하며, deployment/efficiency를 주요 기여로 유지하려면 이 실험은 사실상 필요해 보입니다.

---

### 6. Novelty를 더 정밀하게 정립할 필요가 있습니다.

핵심 수식인

- activity mask를 selective SSM의 ∆에 곱하고,
- M=0이면 \(\bar A=I,\bar B=0\),
- 따라서 \(h_t=h_{t-1}\)

라는 아이디어는 매우 깔끔합니다. 

다만 리뷰어 입장에서는 다음 질문이 자연스럽게 생깁니다.

> “∆를 0으로 설정해 SSM transition을 identity로 만드는 것은 selective SSM의 직접적인 algebraic consequence인데, 이것 자체가 충분한 methodological novelty인가?”

현재 related work는 DynamicViT, DeltaCNN, skip convolutions, Eventful Transformers, Vision Mamba/VMamba 등을 간략히 다룹니다.  하지만 “externally gated SSM update”, “conditional state update”, “event-driven recurrent/state-space computation”과의 차이가 더 깊게 논의되어야 합니다.

저라면 novelty를 단순히 “M=0 gives identity”에 두기보다 다음 **조합된 contribution**으로 재정의하라고 권하겠습니다.

1. external visual-change signal과 SSM discretization parameter의 직접 결합,
2. exact temporal state preservation과 approximate spatial caching의 명확한 분리,
3. iso-mask token-drop control을 통한 preserved-state readout 효과의 분리,
4. streaming drift / train-deploy mismatch의 empirical characterization.

이렇게 해야 “간단한 trick”을 넘어 paper-level contribution으로 더 설득력이 생깁니다.

---

### 7. Cross-domain/moving-camera claim은 아직 너무 제한적입니다.

KITTI raw 5개 drive, 885 frame에서 zero-shot 실험을 수행한 것은 좋은 추가 실험입니다. 하지만 sparsity가 synthetic 25.8%에서 real 92.8%까지 올라가며, GMC + **per-domain threshold recalibration**을 해야 14.1%까지 줄어듭니다. 

즉 실제 deployment에서는 change detector threshold가 domain-dependent합니다.

이 결과에서 제가 궁금한 것은:

- calibration에 몇 frame이 필요한가?
- calibration에는 ground truth가 필요한가?
- test sequence 자체를 보면서 threshold를 tune한 것인가?
- calibration target은 activity인가, accuracy인가?
- domain별 threshold를 oracle로 선택한 것은 아닌가?

이 protocol을 명확히 설명해야 합니다.

또 저자 스스로 handheld/aerial motion은 미검증이며 real moving camera validation이 한 dataset의 5 sequence뿐이라고 적고 있습니다. 

따라서 “moving-camera extension”은 feasibility demonstration 정도로 표현하는 편이 타당하며, 일반적인 camera motion robustness를 주장해서는 안 됩니다.

---

### 8. Statistical reliability가 부족합니다.

최종 60k-step 결과는 **single seed**이고, variance는 8k-step six-run에서만 추정되어 있습니다. 

이런 상황에서 2–3% 수준의 improvement를 핵심 결론으로 사용하는 것은 조심해야 합니다.

예를 들어 Table 1에서 96.2% activity 대비 32.2% activity의 AbsRel 차이가 0.1552→0.1595로 작습니다. 이것이 실질적으로 reproducible한 trade-off인지 확인하려면 적어도 주요 operating points에 대해 multiple seed가 필요합니다.

전체 실험을 multi-seed로 다시 돌리는 것이 비용상 어렵다면 최소한:

- dense/full baseline,
- default sparse setting,
- low-activity setting,
- long-clip fine-tuned setting

정도는 3 seeds로 반복하고 confidence interval 또는 std를 보고하는 것이 좋겠습니다.

---

### 9. Alignment protocol의 fairness가 여전히 다소 복잡하고 결과 해석을 어렵게 합니다.

Relative-depth model에는 disparity-space 2-DOF scale-and-shift, metric model과 본 모델에는 1-DOF median scale을 사용하고, 본 모델을 2-DOF에서도 추가 평가한 것은 공정성을 의식한 좋은 시도입니다. 

하지만 이 setup은 model ranking을 상당히 바꿀 수 있습니다. 실제로 SOKKANAEM의 real AbsRel이 1-DOF에서는 0.1595, 2-DOF에서는 0.1321로 약 17% 개선됩니다. 이후 분석에서는 이 차이를 range compression 문제의 증거로 사용합니다. 

따라서 메인 comparison에서 저는 다음을 권합니다.

- 모든 모델에 적용 가능한 common alignment protocol 하나를 main table로 제공하고,
- 각 모델이 원래 사용하는 “native protocol”은 secondary table로 분리,
- 혹은 최소한 1-DOF / 2-DOF 모두를 모든 relevant baseline에 보고.

현재 방식은 합리적인 근거가 있지만, 독자 입장에서는 “같은 protocol”이라는 표현과 실제 서로 다른 alignment degree-of-freedom이 혼재되어 혼란스러울 수 있습니다.

---

# Minor Comments

1. **Draft note들을 최종 제출 전에 반드시 제거해야 합니다.** Section 5.1의 “default operating point is under review”, Section 5.7의 “causal claim … what remains open is adoption”, Section 6.5의 “loss term … queued” 등은 현재 manuscript가 아직 실험 진행 중인 draft라는 인상을 줍니다. 예를 들어 Section 6.5에는 아직 “queued”된 loss experiment가 명시되어 있습니다. 

2. **Abstract가 지나치게 많은 결과와 caveat를 담고 있습니다.** 현재 abstract가 거의 mini-results/discussion 수준이라 핵심 message가 흐려집니다. mechanism, 핵심 trade-off, 가장 중요한 limitation 1개 정도로 압축하는 것이 좋습니다.

3. **“Exact”라는 표현의 범위를 더 엄격히 해야 합니다.** Exact한 것은 temporal hidden-state transition이지 전체 feature/output 또는 sparse inference pipeline이 아닙니다. 저자도 spatial output cache는 approximate하다고 인정합니다. 이 distinction을 title/introduction에서부터 반복적으로 명확히 하는 것이 좋습니다.

4. **Figure 1의 “1.644 → 0.608 GMAC”은 특정 activity(15.4%) 조건임을 그림 안에서도 바로 알 수 있도록 표시**하면 좋겠습니다. 현재 caption에는 있지만 그림 자체만 보면 universal reduction처럼 보일 수 있습니다. 

5. **Token-drop ablation의 의미를 더 명확히 설명할 필요가 있습니다.** 이 실험에서 accuracy는 token drop이 오히려 약간 좋고, temporal metrics만 ∆-gating이 좋습니다. 따라서 “token dropping does not suffice”보다 “does not reproduce the temporal stability of preserved-state readout” 정도가 더 정확합니다. 

6. **t-delta 단위/normalization을 더 명확히 설명**해 주세요. baseline 간 output scaling/alignment 이후 metric을 계산하는 순서에 따라 값이 크게 달라질 수 있습니다.

7. **OPW/TCE에서 사용한 RAFT-small의 input preprocessing과 occlusion 처리**를 명시하면 reproducibility가 좋아집니다. Appendix에는 RAFT-small을 쓴다고만 되어 있습니다. 

8. **GMC failure rate를 보고하면 좋겠습니다.** Homography estimation 실패 시 identity fallback을 사용한다고 되어 있으므로, KITTI에서 실제 fallback이 몇 % 발생했는지가 중요합니다.

9. **40% dense fallback threshold의 sensitivity analysis**가 있으면 좋겠습니다. 이 threshold는 accuracy/activity를 꽤 크게 바꾸므로 하나의 heuristic hyperparameter 이상으로 중요합니다. 현재 dense fallback 적용으로 activity가 22.2→32.2% 증가하면서 accuracy가 개선됩니다. 

10. **Memory comparison에서 weight memory와 per-stream state memory를 분리해 표로 정리**하는 것이 좋습니다. streaming deployment에서는 stream 수가 증가할수록 persistent state가 중요하므로 총 memory \(W + N\times S\) 형태의 scaling plot도 유용합니다.

11. **“6× to 82× fewer parameters”는 비교 대상 전체 범위를 의미하므로**, 해당 숫자만으로 efficiency superiority를 강조하면 다소 과장돼 보일 수 있습니다. parameters, MAC, actual latency를 별도 축으로 구분하십시오.

12. **실제 qualitative depth video examples**가 있으면 좋겠습니다. 특히 Bonn에서 keyframe 직전/직후 depth map과 error map을 보여주면 Figure 7의 sawtooth phenomenon이 훨씬 직관적일 것입니다.

13. **Range compression 문제는 중요한 finding이므로 Section 6 diagnostic이 아니라 main limitation/result로 일부 승격할 가치가 있습니다.** Bonn에서 predicted dynamic range가 GT의 0.47배라는 것은 낮은 δ1의 중요한 원인입니다. 

14. **Reproducibility appendix에 optimizer, learning rate, LR schedule, batch size, augmentation, dataset sampling ratio** 등을 추가해야 합니다. 현재 architecture/loss/steps는 비교적 잘 적혀 있지만 독립 재현에는 정보가 부족합니다. 

15. Data Availability, Funding, Acknowledgments가 아직 placeholder 상태입니다. 최종 제출 전에 정리해야 합니다. 

---

## 리뷰어 스타일로 한 문단으로 요약하면

이 논문에서 제가 가장 높게 평가하는 부분은 **저자들이 자신의 가설을 검증하다가 나온 반례와 실패를 숨기지 않았다는 점**입니다. Token-drop 결과가 초기 해석을 뒤집었고, fused kernel이 sparse latency advantage를 없앴으며, long-stream evaluation에서 기존 8-frame 결과가 optimistic하다는 사실까지 논문 안에서 직접 보여줍니다. 이 때문에 연구 자체의 신뢰감은 오히려 높습니다. 다만 바로 그 결과들 때문에 현재 manuscript의 핵심 baseline protocol과 최종 configuration을 다시 정리해야 합니다. 특히 **video-specific baseline 부재, short-clip 중심의 main evaluation, 아직 채택되지 않은 improved long-clip checkpoint, 그리고 edge-device efficiency 검증 부재**는 acceptance 전에 해결해야 할 major issue라고 판단합니다. 저자 스스로도 이 한계 대부분을 정확히 인식하고 있습니다. 

제가 점수를 준다면 대략 **Novelty 7/10, Technical quality 6/10, Experimental completeness 5/10, Writing/Transparency 8/10, Overall 6/10 — Major Revision** 정도입니다. 핵심 아이디어 때문에 reject보다는 **“실험과 positioning을 마무리하면 충분히 다시 볼 가치가 있는 논문”** 쪽으로 판단할 것 같습니다.