# 작업 12 — skip 오차, refresh, 고정 비용의 수학적 분석

2026-09-06. 대상은 [고정 v11 구현](freeze/baseline.json)이며 학습·모델 변경·최종 test 추론은
하지 않았다. 본 문서는 작업 12의 분석·증명이다. [영문 LaTeX 이론 절](theory12.tex)은 별도
삽입용 원고이며 기존 전체 초안의 과거 주장을 자동으로 유효하게 만들지 않는다.

## 0. 결과와 보장 범위

| 결과 | 증명된 내용 | 증명하지 않은 내용 |
|---|---|---|
| 명제 1 | binary Δ-gating의 hidden-state update/copy 항등식 | active 입력 항의 exact ZOH, dense 출력과의 동등성 |
| 정리 2 | 공통 입력의 단일 SSM에서 skip 결손의 누적 오차식·상한 | 픽셀 threshold만으로 그 결손이 작다는 보장 |
| 정리 3 | 전체 네트워크의 조건부 perturbation 상한 | 전체 네트워크의 수축성·실용적인 전역 상수 |
| 명제 4 | 첫 temporal block의 조건부 캐시 오차와 최대 보유 나이 K−1 | 공간 문맥 생략, 깊은 층, GMC까지 같은 tau 상한 적용 |
| 정리 5 | refresh의 국소 결손 제거와 가정하 주기별 오차 상한 | 누적 상태 오차 리셋, K 단축에 따른 정확도 단조 개선 |
| 명제 6 | 고정 비용·키프레임·fallback·overhead를 포함한 가속 조건 | MAC 비율을 실측 지연/에너지로 변환하는 무조건적 법칙 |

핵심 결론: **상태를 그대로 보관하는 정확성은 dense 상태를 정확하게 근사한다는 뜻이 아니다.**
이 결과들은 재귀식 전개와 표준 perturbation/Amdahl 분석을 본 구현에 적용한 것이다.
정리 이름을 붙였다는 이유로 새로운 일반 수학 정리나 저널 게재 적합성이 입증되는 것은 아니다.

## 1. 표기, 구현 대응, 가정

한 패치·한 temporal SSM의 inner-channel/state 축을 벡터화한다. 실수 연산에서

\[
\Phi_t=\operatorname{diag}(e^{-\lambda_j\Delta_{t,j}}),\quad
q_{t,j}=\Delta_{t,j}(B_tx_t)_j,\quad
F_t(h)=\Phi_t h+q_t,\qquad \lambda_j>0,\ \Delta_{t,j}\ge0.
\tag{1}
\]

실제 `Bp*x`의 broadcast를 벡터화한 표기다. 채널별 Δ는 state 차원에 반복된다.
`A=-exp(A_log)`이고 Δ는 `softplus(dt_proj(x))`다. 한 패치의 binary mask는 그 패치의
모든 state 좌표에 동일하게 적용한다. 여러 패치를 묶으면 좌표별 대각 mask가 된다.

이하 노름은 Euclidean norm, 행렬은 그 유도 노름이다. \(e_t=\widetilde h_t-h_t\),
\(E_t=\|e_t\|\), \(M_t=\operatorname{diag}(m_{t,j})\), \(m_{t,j}\in\{0,1\}\).
정리 2에서 dense와 gated 경로는 **동일한 외생 토큰 입력 및 \(\Phi_t,q_t\)**를 사용한다.
시점 0은 초기 상태이고 정리의 update는 1부터 센다. 실제 detector의 첫 프레임 인덱스는 0이다.

| 수학 대상 | 구현 위치 | 확인 사항 |
|---|---|---|
| \(\Phi,q\), zero step | `ssm.py`: `_params`, `forward`, `step` | exact exponential transition + first-order input |
| state/readout 분리 | `model.py`: `TemporalBlock.step` | C, x, z, D 및 residual은 현재 입력에 의존 |
| temporal output hold | `TemporalBlock.step_cached` | 비활성 state와 block output을 유지 |
| 공간 근사 | `SpatialBlock.forward_cached` | active subsequence scan + 비활성 output cache |
| 강제 갱신 | `ChangeDetector.is_keyframe`, `SOKKANAEM.step` | all-active mask이며 기존 hs를 넘김 |
| 실측 비용 | [작업 9~11](STUDY_9_11.md) | PNG-to-CPU-depth와 별도 동기화 profile 구분 |

조건부 상한에서 쓰는 \(\rho,\epsilon,L_t,L_R,L_D\)는 유효한 상한이라는 **가정**이다.
고정 가중치의 A가 음수라는 사실만으로 전체 모델의 수축성이나 작은 decoder 상수가 따라오지 않는다.
유한 표본의 최대값·평균값·회귀 기울기를 전역 상수로 대신하지 않는다.

## 2. 명제 1 — zero-step identity와 입력 이산화

Binary mask 아래 실제 gated update는

\[
\widetilde h_t=M_tF_t(\widetilde h_{t-1})+(I-M_t)\widetilde h_{t-1}.
\tag{2}
\]

증명: 각 좌표에서 m=1이면 (1), m=0이면 \(e^0=1\), \(0\cdot Bx=0\)이다.
따라서 비활성 좌표는 유지된다. 이는 binary update/copy와 동일하며, fractional mask의
\(e^{m\Delta A}\)는 일반적으로 \(m e^{\Delta A}+1-m\)와 같지 않다. □

이 원리는 [Skip RNN의 update/copy 식 (3)](https://arxiv.org/html/1708.06834v3)을 포함한
선행 조건부 재귀 갱신과 구분해 기술한다. 항등식 자체의 최초성을 주장하지 않는다.

정확 ZOH의 scalar 입력 계수는
\((1-e^{-\lambda\Delta})/\lambda\)이다. 구현은 이를 Δ로 대체한다.
[Mamba의 ZOH 정의 식 (4)](https://arxiv.org/html/2312.00752v2)와 구분해야 한다.
\(b=(Bx)_j\)를 한 step 동안 고정하면

\[
|\Delta b-\tfrac{1-e^{-\lambda\Delta}}{\lambda}b|
\le \tfrac12\lambda\Delta^2|b|.
\tag{3}
\]

증명: 계수 차이는 \(\int_0^\Delta(1-e^{-\lambda s})\,ds\)이고
\(0\le1-e^{-\lambda s}\le\lambda s\)를 적분한다. □
이는 **입력 이산화의 국소 오차**이지 skip 오차가 아니다. 이후 dense/gated 비교는 둘 다
동일한 구현 입력 항을 써서 이산화 차이를 끼워 넣지 않는다.

실수 항등식과 부동소수점 검증은 구분한다. 유한 정상 입력의 one-step CPU/CUDA 테스트와
캐시의 inactive index 보존을 검사한다. NaN/Inf, overflow, signed-zero 비트 패턴이나 긴
chunked scan의 연산 재결합까지 모든 입력에서 bitwise equality라고 일반화하지 않는다.

## 3. 정리 2 — 공통 입력 SSM의 skip 결손과 누적 오차

다음을 skip 결손으로 정의한다.

\[
r_t=(I-M_t)\{(I-\Phi_t)\widetilde h_{t-1}-q_t\},\quad d_t=\|r_t\|.
\tag{4}
\]

이는 gated 경로의 현재 상태에 dense update를 적용했을 때 **생략한 상태 이동의 음수**다.
(1)–(2)에서 정확히

\[
e_t=\Phi_t e_{t-1}+r_t,
\quad
e_T=\Big(\prod_{j=1}^T\Phi_j\Big)e_0+
\sum_{i=1}^T\Big(\prod_{j=i+1}^T\Phi_j\Big)r_i.
\tag{5}
\]

일반 행렬 표기라면 뒤 시점이 왼쪽인 순서곱이며, 여기서는 대각이라 순서가 교환된다.
빈 곱은 I다. \(\|\Phi_t\|\le\rho_t\)이면

\[
E_T\le\Big(\prod_{j=1}^T\rho_j\Big)E_0+
\sum_{i=1}^T\Big(\prod_{j=i+1}^T\rho_j\Big)d_i.
\tag{6}
\]

증명: (2)에서 \(\Phi_t\widetilde h_{t-1}+q_t\)를 더하고 빼면 첫 식이다.
귀납적으로 전개한 뒤 유도 노름의 submultiplicativity와 삼각부등식을 적용한다. □

추가로 \(\rho_t\le\rho<1\), \(d_t\le\epsilon\)이면

\[
E_T\le\rho^T E_0+\epsilon\frac{1-\rho^T}{1-\rho}.
\tag{7}
\]

\(\lambda_j\ge\lambda_{\min}>0\), ungated \(\Delta_{t,j}\ge\Delta_{\min}>0\)라는
균일 하한이 있으면 \(\rho=e^{-\lambda_{\min}\Delta_{\min}}<1\)을 쓸 수 있다.
여기서 \(\Phi_t\)는 skip하지 않은 **비교 기준의 transition**이다. gated transition은
skip 시 1이며 strict contraction이 아니다. 유용한 Δ 하한을 확보하지 못하면 \(\rho=1\)의
\(E_T\le E_0+\sum_i d_i\)만 쓰며, 1로 나눌 수 없는 (7)을 적용하지 않는다.

### 반례 A — 픽셀 변화 0이어도 dense-state 오차는 생긴다

\(F(h)=h/2+1\), \(h_0=\widetilde h_0=0\), 입력은 계속 동일, 모든 update를 skip하면
\(\widetilde h_t=0\), \(h_t=2(1-2^{-t})\)다. t=4에서 오차는 1.875다.
실제 첫 프레임을 all-active로 시작해도 그 뒤 dense는 계속 평형으로 이동하고 gated는 멈춘다.
따라서 \(\|I_t-I_{t-1}\|=0\)은 \(d_t=0\)을 뜻하지 않는다.
**임계값 tau만으로 dense-state 오차가 O(tau) 또는 O(sqrt(tau))라는 주장은 성립하지 않는다.**

### 동일 입력 가정이 깨질 때

\(\widetilde\Phi_t,\widetilde q_t\)가 sparse 경로의 다른 토큰에서 계산되면 (5)의 결손에
\((\widetilde\Phi_t-\Phi_t)\widetilde h_{t-1}+(\widetilde q_t-q_t)\)가 추가된다
(skip 결손은 \(\widetilde\Phi,\widetilde q\)로 정의).
따라서 추가 노름은
\(\|\widetilde\Phi_t-\Phi_t\|\,\|\widetilde h_{t-1}\|+\|\widetilde q_t-q_t\|\) 이하이다.
깊은 temporal block은 앞선 temporal/spatial 출력에 의존하므로 이 차이를 0으로 놓을 수 없다.

## 4. 정리 3 — 캐시와 층간 피드백을 포함한 전체 모델 상한

고정 입력 영상과 sparse 경로에서 선택된 mask열을 조건으로 둔다. z는 모든 hs와 block output
cache를 합친 확장 상태다. Dense 경로에도 같은 모양의 cache를 기록하되 매 프레임 덮어쓰게
정의하면 비교 공간이 일치한다. \(F_t\)는 all-active 전체 step, \(G_t^{m_t}\)는 실제
sparse/cache step이다. detector의 threshold를 미분하거나 작은 입력 변화에 mask가 같다고
가정하지 않는다. 여기서는 관측된 mask열을 고정한 trajectory 비교다.

\[
z_t=F_t(z_{t-1}),\quad\widetilde z_t=G_t^{m_t}(\widetilde z_{t-1}),\quad
d_t^{\rm net}=\|G_t^{m_t}(\widetilde z_{t-1})-F_t(\widetilde z_{t-1})\|.
\tag{8}
\]

두 경로를 포함하는 영역에서 \(F_t\)가 Lipschitz 상수 \(L_t\)를 가지면
\(E_t^{\rm net}\le L_tE_{t-1}^{\rm net}+d_t^{\rm net}\)이고,
(6)의 \(\rho_t,d_t\)를 \(L_t,d_t^{\rm net}\)로 바꾼 식이 성립한다.
증명: \(F_t(\widetilde z_{t-1})\)를 더하고 빼고 삼각부등식을 적용한 뒤 귀납한다. □

전체 depth 출력 함수 \(P_t\), sparse 출력 \(\widetilde P_t\)에 대해서도
\(\eta_t=\|\widetilde P_t(\widetilde z_{t-1})-P_t(\widetilde z_{t-1})\|\),
dense 출력 함수의 상수 \(L_{P,t}\)를 쓰면

\[
\|\widetilde y_t-y_t\|\le L_{P,t}E_{t-1}^{\rm net}+\eta_t.
\tag{9}
\]

이 식은 가정을 드러내는 **조건부 perturbation 정리**다. 비용을 치러 counterfactual dense
step을 실행하면 d를 진단할 수 있지만, 이는 현재 detector가 제공하는 무료 online 인증이 아니다.
\(L_t<1\)은 입증하지 않았다. 단순 반례 \(F(h)=0.5h+u(h),u(h)=h\)는 내부 전이 0.5에도
전체 map의 계수가 1.5임을 보여 준다. 이 scalar 예가 v11의 정확한 구조라는 뜻은 아니며,
음의 A만으로 전체 네트워크 수축을 추론하는 논리를 반박한다.

Depth decoder의 Lipschitz 상수 \(L_D\)를 별도로 확보하면 feature 차이에 이를 곱할 수 있다.
양의 GT \(d_i\ge d_*>0\)인 n개 동일 픽셀의 **no-fit** AbsRel에는

\[
|\operatorname{AbsRel}(\widetilde y,d)-\operatorname{AbsRel}(y,d)|
\le\frac1n\sum_i\frac{|\widetilde y_i-y_i|}{d_i}
\le\frac{\|\widetilde y-y\|_1}{n d_*}.
\tag{10}
\]

증명은 \(||a|-|b||\le|a-b|\)의 픽셀별 적용이다. Dense 자체의 GT 오차는 별도로 남는다.
현재 데이터 계약에 새로운 d_* cutoff를 도입하지 않았다. GT별 scale–shift 재적합은 추가
연산자이고 ill-conditioned fit/역수 변환 때문에 같은 상수를 사용할 수 없다.
δ1 같은 threshold 지표도 별도의 margin 가정 없이 이 방식의 연속 오차 보장을 갖지 않는다.

## 5. 명제 4 — cache age, 픽셀 변화, 공간 문맥의 분리

Pixel detector의 비활성 패치에는 hysteresis의 해당 임계값 \(\theta_t\le\tau_{on}\)에 대해
\(\operatorname{MSE}(I_t,I_{t-1})\le\theta_t\)가 성립한다. dilation/강제 갱신/fallback은
0을 1로 바꾸므로 **최종 mask=0이면** 이 함의가 보존된다. C채널 p×p 패치를 벡터화하면

\[
\|I_t-I_{t-1}\|_2\le\sqrt{Cp^2\tau_{on}}.
\tag{11}
\]

마지막 갱신 s 이후 현재 t까지 이 패치가 계속 비활성이고 a=t−s이면
\(\|I_t-I_s\|_2\le a\sqrt{Cp^2\tau_{on}}\).
첫 patch embedding \(u=W I+b\)와, 상태를 고정한 temporal block readout
\(R(u,h)\)의 입력 Lipschitz 상수 \(L_R\)가 해당 영역에서 존재하면

\[
\|R(u_t,h_s)-R(u_s,h_s)\|
\le L_R\|W\|_2\,a\sqrt{Cp^2\tau_{on}}.
\tag{12}
\]

증명: (11)을 a회 합하고 선형 embedding과 readout의 Lipschitz 부등식을 적용한다. □
여기서 R은 residual/LayerNorm/C/x/z/D를 포함한 **state 고정 출력 함수**이며 Δ를 재갱신하는
active-step 함수가 아니다. 캐시 저장 시에는 갱신된 h_s를 읽었고 이후 skip 동안 h_s가 유지된다.
tau는 MSE 단위이므로 sqrt(tau), 누적 나이 a, 가중치/함수 상수를 생략하면 안 된다.

고정 v11의 C=3,p=16,tau_on=0.05,K=30에서는 (12)의 순수 입력 계수
\((K-1)\sqrt{Cp^2\tau_{on}}\)만 약 179.706다. 영상이 [0,1]에 있으므로 더 단순한
\(\|I_t-I_s\|_2\le\sqrt{Cp^2}\approx27.713\)로 cap할 수 있다. 이만큼 보수적인
계수에 \(L_R\|W\|\)도 곱해야 하므로 현재 설정에서 작은 수치 오차 인증을 얻었다고 하지 않는다.

### 깊은 층과 공간 캐시는 별도 결손이다

깊은 층의 토큰은 다른 패치/이전 상태에 의존하므로 (12)에 해당 입력 변화 상한을 추가해야 한다.
공간 block의 dense 연산 S와 active subsequence 연산 S_A에 대해서는 active 좌표도
\(\xi_A(u)=\|S_A(u_A)-[S(u)]_A\|\)라는 문맥 생략 결손이 있다.
비활성 캐시 age의 상한만으로 \(\xi_A\)를 제어하지 못한다.

반례 B: scalar spatial recurrence \(s_i=\rho s_{i-1}+b_i\), \(s_0=0\)에서
\(b_1=1,b_2=0\). dense의 두 번째 출력은 ρ, 첫 토큰을 생략한 active subsequence에서는 0이다.
현재 입력이 과거와 똑같아도 이 차이가 존재한다. 실제 `SpatialBlock.forward_cached`에서도
동일 토큰을 넣고 부분 mask를 적용하면 active 출력 차이가 발생함을 테스트했다.
이는 해당 mask에서 공간 생략 연산자가 다른 연산이라는 반례이며, 정적 영상에서 detector가
그 부분 mask를 반드시 선택한다는 주장은 아니다.

GMC의 score는 warped feature 상대 L1이며 (11)의 pixel MSE 가정과 다르다.
현재 구현은 RGB 비교용 warp만 수행하고 hs/sp/tc를 좌표 정합하지 않는다. GMC를 켰다는
사실만으로 moving-camera 상태 오차의 같은 상한을 주장하지 않는다.

## 6. 정리 5 — refresh는 국소 근사 제거이지 dense-history 복원은 아니다

키프레임은 all-active mask를 설정하고 각 block cache를 현재 계산으로 덮어쓴다.
따라서 (8)의 **같은 입력 상태** 기준으로 \(G_t^{\mathbf1}=F_t\), \(d_t^{net}=0\)이다.
하지만 두 경로의 입력 상태가 다르면

\[
E_t^{net}\le L_t E_{t-1}^{net},\qquad
\text{일반적으로 }E_t^{net}\ne0.
\tag{13}
\]

정리 2의 공통 입력 단일 SSM이라면 \(e_t=\Phi_te_{t-1}\)이다.
증명: all-active branch에는 hold/gather 근사가 없으나 기존 hs를 그대로 전달한다.
따라서 (8)의 국소 결손만 0이다. □

### 주기별 조건부 상한

공통 입력 단일 SSM에서 매 step \(\|\Phi_t\|\le\rho<1\), 비refresh 결손
\(d_t\le\epsilon\), refresh 간격 K를 가정한다. refresh 직후 상한을 \(B_n\)이라 하면
그 다음 K−1 step과 마지막 refresh에 (6)을 적용하여

\[
B_{n+1}\le\rho^K B_n+
\rho\epsilon\frac{1-\rho^{K-1}}{1-\rho},
\quad
\limsup_{n\to\infty}B_n\le
\frac{\rho\epsilon(1-\rho^{K-1})}{(1-\rho)(1-\rho^K)}.
\tag{14}
\]

refresh 이후 j≤K−1 step의 상한은
\(E_{nK+j}\le\rho^j B_n+\epsilon(1-\rho^j)/(1-\rho)\)다.
증명: 마지막 step의 결손은 0이므로 앞선 K−1 결손에 최소 한 번 ρ가 곱해진다.
주기 recurrence를 기하급수로 합하면 (14)다. K=1에서는 새 skip 결손이 없다. □

주의: 이는 K를 바꿔도 유효한 \(\rho,\epsilon\) 상한을 가정한 정리다. 실제 K 변경은
토큰/마스크/hysteresis 이력과 결손도 바꾼다. 실제 AbsRel이 K에 따라 단조 개선된다는 증명이 아니다.
한 경로만 state=0으로 reset하는 것은 또 다른 개입이며 dense 역사와의 오차를 제거하지 않는다.
진정한 비교 상태 동기화에는 dense reference 상태를 복사하거나 필요한 과거를 재실행해야 한다.
이를 현재 refresh의 기능으로 기술하지 않는다.

K≥1인 `refresh=keyframe`에서는 첫 프레임과 0,K,2K,… 프레임이 all-active이므로 각 캐시의
최대 보유 나이는 K−1이다. 추가 activation/fallback은 이 나이를 줄일 수 있다.
Rolling refresh도 패치별 나이를 제한하지만 한 프레임의 전체 문맥을 dense하게 복원하지 않으므로
\(G_t=F_t\)인 전역 refresh나 동일 latency라고 간주하지 않는다.

## 7. 명제 6 — 고정 비용, 활성률, overhead의 가속 조건

동일 입출력 경계에서 dense 기준 비용을 \(C_d=F+V>0\)로 두자. F는 최적화 후에도 남는
공통 비용, V는 dense 기준의 절감 가능한 비용이다. 다음 **가산·선형 비용 모델**과
\(\overline C_s>0\)을 가정한다(모든 sparse 비용이 0인 퇴화적 이상화는 무한 가속으로 처리).

\[
\overline C_s=F+\overline a_{eff}V+\overline H,
\quad f=F/C_d,\quad\omega=\overline H/C_d\ge0,
\quad S=\frac1{f+(1-f)\overline a_{eff}+\omega}.
\tag{15}
\]

H에는 검출기 추가 비용, gather/scatter, cache copy, launch/shape 관리 등 dense 대비
추가 비용을 포함한다. 어떤 항을 F와 H에 두는지 정하고 중복 합산하지 않는다.
이 가정 아래

\[
S>1\iff\omega<(1-f)(1-\overline a_{eff}),\qquad
S\le 1/f\quad(f>0).
\tag{16}
\]

증명: 양수인 (15)의 분모를 1과 비교한다. 상한은 비음수인 나머지 항을 버리면 얻어진다. □
이는 [Amdahl의 고정 비용 한계](https://doi.org/10.1145/1465482.1465560)를 현재 실행 경로에
적용한 것이며 새로운 보편 가속 법칙을 주장하지 않는다. GPU overlap, shape별 비선형 시간,
하드웨어 변경, detector/입력 경로 변경은 모델 가정을 바꾸므로 실측 검증이 따로 필요하다.

### 키프레임과 fallback은 합집합으로 센다

정상 gate 활성률 \(a_t\in[0,1]\), keyframe 사건 k_t, dense fallback 사건 b_t에 대해

\[
a_{eff,t}=\begin{cases}1,&k_t\lor b_t,\\a_t,&\text{그 외},\end{cases}
\quad \overline a_{eff}=T^{-1}\sum_t a_{eff,t}.
\tag{17}
\]

겹치는 사건을 두 번 세지 않는다. 실제 실행의 post-fallback mask를 쓰면 이미 반영되어 있다.
마스크 활성률과 bucket padding/서브연산별 실제 처리 비율은 다를 수 있다. (15)는 실제 V가 이
활성률에 선형 비례할 때만 쓸 수 있으며, padding/비선형 비용은 별도 항 또는 실측값으로 다룬다.

완전 정적 영상이라도 K주기, T프레임, 첫 프레임 0에서 시작하면 keyframe 수는
\(\lceil T/K\rceil\)이므로 \(\overline a_{eff}\ge\lceil T/K\rceil/T\).
T=256,K=30이면 9/256≈3.516%이며 정확히 1/30이 아니다. 장기 평균에서만 1/K에 접근한다.
비keyframe 평균 a, fallback 없음이라는 추가 조건에서는
\(\overline a_{eff}\to a+(1-a)/K\).

### 오차–비용 조건을 함께 사용할 때

첫 block의 cache-only 허용 오차 E_c와 (12)의 \(c=L_R\|W\|\sqrt{Cp^2\tau}>0\)가
검증되었다면 충분조건은 \(K\le1+\lfloor E_c/c\rfloor\)다.
고정 f,a,ω인 장기 비용 모델에서 목표 S_*>1은

\[
D=1/S_*-f-(1-f)a-\omega>0,\qquad
K\ge\left\lceil\frac{(1-f)(1-a)}{D}\right\rceil
\tag{18}
\]

을 요구한다(정수 K≥1, 비퇴화 경우 \((1-f)(1-a)>0\)). 두 범위가 겹치지 않으면 **이 가정의
충분조건으로 인증할 수 있는 K가 없다**. 실제 가능성 전체의 불가능성 정리가 아니다.
cache-only 조건은 dense-state/공간 문맥/GT 오차 보장을 대신하지 않는다.

### 기존 실측에 대한 제한된 적용

[작업 10 구성요소 표](tables/study10_components.csv)의 sparse K30에서 read/decode/preprocess는
source-balanced profile 총시간의 약 69.4%다. 이 입력 비용을 고정하고 **나머지 계측 시간을
전부 0으로 만드는 이상화**라면 해당 profile 경로 대비 상한은 약 1/0.694≈1.44×다.
이는 profile 내부의 가정적 한계이며, sparse/dense 주 latency의 속도 비나 GPU-resident 실행,
입력 경로 자체를 개선한 시스템에 대한 상한이 아니다. 실측 상한 인증·도달 가능한 가속이라고
표기하지 않는다. 본문의 정확한 재계산은 [검증 요약](tables/theory12_checks.json)에 남긴다.

실제 동일 subset 주 latency는 sparse 7.496 ms, dense 7.521 ms였다.
네트워크의 작은 가중치 수와 sparsity의 실행시간 절감은 분리해야 한다.
현재 dense하게 할당한 temporal state 크기는
\(b\,B\,N\sum_{\ell\in temporal}P_\ell S_\ell\) bytes다. 활성률을 곱하지 않는다.
v11 FP32,B=1,N=256,P=384,S=16,temporal block=2에서 12 MiB다. 캐시는 추가 메모리이며
실측 sparse stream state 13.501 MiB가 dense 12 MiB보다 작지 않았던 것과 일치한다.

## 8. 검증, 출처, 원고 적용

실행 가능한 식: [`sokkanaem/theory.py`](../sokkanaem/theory.py).
수치 항등식·반례·실제 block/refresh 동작 검증: [`tests/test_theory.py`](../tests/test_theory.py).
요약/해시 생성: [`scripts/paper_theory_check.py`](../scripts/paper_theory_check.py).
합성 검증은 유한 예제의 회귀 테스트이며 해석적 증명이나 학습된 모델의 전역 인증을 대체하지 않는다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_theory.py -q
python scripts/paper_theory_check.py
python scripts/paper_freeze.py --verify
```

검증기는 모델 학습/영상 채점 없이 실제 가중치의 부호·구조, 기존 실측 표의 산술, 합성
오차식을 확인하고 가중치·코드·기존 보고서·새 산출물 해시를 기록한다. 그 결과를
실제 데이터에서 d_t/전역 Lipschitz 상수를 인증한 것으로 읽지 않는다.
LaTeX는 영문 절과 증명을 제공하지만 현재 환경에 TeX 엔진이 없어 PDF 조판은 미검증이다.

원고 반영 시 `TemporalBlock.step_cached`의 과거 docstring에 있는 “bounded by tau”는
(12)의 조건부 문장으로 대체해야 한다. frozen `model.py`는 재현성을 위해 수정하지 않았다.
기존 초안의 “refresh가 누적 오차를 제거”, “희소화가 stream state를 줄임”, “negative A이므로
전체 출력 안정” 같은 해석도 이 문서의 제한으로 대체한다. 전체 초안 통합은 작업 15 범위다.

참고문헌은 각 관련 문장에 연결했다. Mamba/Skip RNN의 원문 정의 및 Amdahl의 원 출판 기록을
확인했으며, 여기의 식 (3)–(18)의 적용·증명과 반례는 이 저장소의 분석이다. 계열 전체 선행연구
조사나 독립적인 동료 심사를 완료했다는 의미는 아니다.
