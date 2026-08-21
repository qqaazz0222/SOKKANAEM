# 엣지 기기 지연·전력 측정 실행 문서

`scripts/edge_bench.py`를 Raspberry Pi 4와 Jetson Nano Developer Kit B01에서 돌리는 절차.

## 0. 이 측정이 필요한 이유

데스크톱 GPU에서는 ∆-게이팅의 MAC 절감이 시간으로 환산되지 않는다. 융합 Triton 스캔 이후 RTX
4090은 이 모델 규모에서 **오버헤드 제약**이라, 지연 시간이 활성률에 평평하고(활성 5~70% 구간
1.973~1.994 ms) 컴파일된 dense 1.29 ms가 희소 경로 2.03~2.20 ms를 모든 활성률에서 이긴다
(`paper/draft.md` §5.7). 절감이 시간·전력이 되는지는 **연산이 제약인 하드웨어**에서만 답할 수 있다.

기기별로 닫히는 주장이 다르다.

| 주장 | Pi 4 (CPU) | Nano B01 (Maxwell sm_53) | TX2 (Pascal sm_62) | Orin 이상 |
|---|---|---|---|---|
| 연산 제약 환경에서 게이팅이 지배 연산을 제거한다 | **닫힘** (런치 오버헤드 교란 없음) | 닫힘 (참조 경로) | 닫힘 (참조 경로) | 닫힘 |
| 논문이 보고하는 융합 커널 구현에서도 그렇다 | 해당 없음 | **불가** (Triton sm_70+) | **불가** (Triton sm_70+) | 닫힘 |
| 엣지 배치에서 실시간 | 안 닫힘 | 안 닫힘 | 측정값에 따라 | 닫힘 가능 |
| 에너지/프레임 감소 | 외부 전력계 필요 | 온보드 INA3221 (보드 전체) | 온보드 INA3221 (**GPU 레일 분리**) | 온보드 |
| fp16 측정 의미 | 해당 없음 | 없음 (Maxwell은 FP32와 동률) | **있음** (Pascal 2:1) | 있음 |

### 4점 교차점 특성화

기기를 연산 풍부도 순으로 늘어놓는 것이 이 라운드의 목표다. 현재 논문은 4090 한 점만 있어
"이 GPU에서는 환산되지 않는다"까지만 말한다.

| 기기 | 성격 | 예상 |
|---|---|---|
| Pi 4 (CPU 4스레드) | 극단적 연산 제약 | 희소 압승 (데스크톱 CPU에서 이미 활성 5%에서 15.1배) |
| Jetson Nano B01 | 연산 + 대역 제약 (25.6 GB/s) | 희소 우세 |
| Jetson TX2 | 연산 제약, 대역 여유 (59.7 GB/s) | 희소 우세, 배수 감소 예상 |
| RTX 4090 (융합 커널) | 오버헤드 제약 | **dense 압승** (측정 완료) |

네 점이 있으면 "희소성의 시간 이득이 어느 등급에서 뒤집히는가"를 곡선으로 제시할 수 있다.

**요점은 기기 연식이 아니라 영역이다.** 세 기기 모두 2019~2020년대 초 하드웨어이므로, 본문에서
"구형 기기 측정"으로 읽히지 않게 연산 제약 영역을 측정한다는 의도를 먼저 밝힌다.

## 1. 기준 수치 (데스크톱 CPU, 4스레드)

같은 스크립트를 이 저장소의 개발 PC CPU에서 4스레드로 제한해 돌린 결과. 기기 수치가 이보다
느린 것이 정상이며, **비율**이 비교 대상이다.

```
device=cpu dtype=fp32 size=256 threads=4 torch=2.8.0+cu128
cache  active%   ms/frame      FPS
on        5.1      14.12     70.8
on       30.1      72.41     13.8
on      100.0     207.35      4.8
off       5.1     213.46      4.7
off      30.1     212.29      4.7
off     100.0     215.82      4.6
```

dense(캐시 off)는 활성률과 무관하게 평평하고, 희소는 거의 비례한다 — 5%에서 15.1배, 30%에서
2.9배. 4090에서 dense가 항상 이겼던 것과 부호가 반대다.

## 2. 공통 준비

체크포인트와 코드를 기기로 옮긴다. 체크포인트는 16.7 MB, `config.toml`이 **같은 디렉터리에
있어야** 모델 설정이 복원된다.

```bash
# 개발 PC에서
CK=work_dirs/v11-longclip-spread-s0
tar czf edge-bundle.tar.gz sokkanaem scripts/edge_bench.py pyproject.toml \
    "$CK/latest.pt" "$CK/config.toml"
scp edge-bundle.tar.gz <user>@<device>:~/
```

기기에서:

```bash
tar xzf edge-bundle.tar.gz && cd ~
```

`edge_bench.py`는 torch만 요구하고 Python 3.6 문법으로 작성돼 있다(JetPack 4.6 대응). 패키지
설치 없이 저장소 루트에서 바로 실행된다.

## 3. Raspberry Pi 4

64-bit Raspberry Pi OS 기준. 32-bit OS면 aarch64 휠이 없어 설치가 막히므로 `uname -m`이
`aarch64`인지 먼저 확인한다.

```bash
uname -m                      # aarch64 여야 한다
python3 -V                    # 3.9 이상 권장
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

측정:

```bash
python3 scripts/edge_bench.py \
    --ckpt v11-longclip-spread-s0/latest.pt \
    --device cpu --threads 4 --iters 20 --warmup 5
```

전력계가 있으면(INA219를 5V 라인에 붙이거나 USB 전력 미터의 값을 읽는 스크립트) watts 한 줄을
stdout에 출력하는 명령을 넘긴다.

```bash
python3 scripts/edge_bench.py --ckpt v11-longclip-spread-s0/latest.pt \
    --device cpu --threads 4 --power cmd --power-cmd "python3 read_watts.py"
```

주의:

- 방열판·팬 없이 돌리면 80~85°C에서 스로틀한다. 측정 중 `vcgencmd measure_temp`와
  `vcgencmd get_throttled`를 확인하고, `get_throttled`가 `0x0`이 아니면 그 회차는 버린다.
- `--threads`를 지정하지 않으면 4코어를 모두 쓰면서도 값이 회차마다 흔들린다. 항상 명시한다.
- 전원 어댑터가 부족하면(공식 3A 미만) 언더볼티지로 클럭이 떨어진다.

## 4. Jetson Nano Developer Kit B01

Maxwell sm_53 / JetPack 4.6.x(CUDA 10.2, Python 3.6). **Triton이 sm_53을 지원하지 않으므로 융합
스캔 커널은 동작하지 않는다.** 스크립트가 `fused scan available: False`를 출력하며, 그 로그가
"참조 chunked 스캔으로 측정했다"는 기록이 된다.

```bash
cat /etc/nv_tegra_release      # JetPack 버전 확인
python3 -V                     # 3.6.x
```

torch는 NVIDIA가 배포하는 JetPack 4.6용 휠을 쓴다(PyPI 휠은 CUDA를 못 잡는다).

```bash
sudo apt-get install -y libopenblas-base libopenmpi-dev
# JetPack 4.6 / Python 3.6 대상 torch 1.10 휠 (NVIDIA 포럼 배포 링크)
pip3 install --user <torch-1.10.0-cp36-cp36m-linux_aarch64.whl>
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

측정 전 전력 모드와 클럭을 고정한다. 고정하지 않으면 DVFS 때문에 지연·전력 둘 다 재현되지 않는다.

```bash
sudo nvpmodel -m 0            # 0 = 10W(MAXN), 1 = 5W
sudo jetson_clocks            # 클럭 락
sudo nvpmodel -q              # 현재 모드 확인
```

측정:

```bash
python3 scripts/edge_bench.py \
    --ckpt v11-longclip-spread-s0/latest.pt \
    --device cuda --iters 20 --warmup 5 --power tegra
```

`--power tegra`는 `tegrastats` 출력의 `VDD_IN` 또는 `POM_5V_IN` mW를 파싱한다. 보드 전체 전력이라
GPU 레일만 분리하려면 `--power cmd`로 원하는 sysfs 경로를 읽는 명령을 넘긴다.

```bash
# 예: INA3221 채널을 직접 읽기 (경로는 커널 버전에 따라 다르다)
--power cmd --power-cmd "cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
```

주의:

- 4GB 공유 메모리다. 데스크톱 환경(GUI)을 끄고 측정하는 편이 안정적이다: `sudo systemctl isolate multi-user.target`
- fp16(`--half`)은 Maxwell에 텐서코어가 없어 FP32와 처리량이 같다. 의미 없으므로 기본 fp32로 둔다.
- 5W 모드(`nvpmodel -m 1`)에서도 한 번 재두면 전력-지연 곡선의 양 끝이 생긴다.

## 4b. Jetson TX2

Pascal sm_62 / JetPack 4.6.x. Nano B01과 소프트웨어 제약은 같다(Python 3.6, torch 1.10 휠,
**Triton 불가 → 참조 chunked 스캔**). 달라지는 것은 측정 품질이다 — 대역폭 59.7 GB/s로 Nano의
2.3배라 참조 스캔의 메모리 병목 왜곡이 작고, Pascal은 FP16을 2:1로 처리하므로 `--half`가 의미를
가진다.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks     # MAXN, 클럭 락
python3 scripts/edge_bench.py --ckpt v11-longclip-spread-s0/latest.pt \
    --device cuda --iters 20 --warmup 5 --power tegra

# fp16 (Nano B01과 달리 여기서는 유의미하다)
python3 scripts/edge_bench.py --ckpt v11-longclip-spread-s0/latest.pt \
    --device cuda --half --iters 20 --warmup 5 --power tegra

# 저전력 끝점
sudo nvpmodel -m 1 && sudo jetson_clocks     # 7.5W Max-Q
python3 scripts/edge_bench.py --ckpt v11-longclip-spread-s0/latest.pt \
    --device cuda --iters 20 --warmup 5 --power tegra
```

GPU 레일만 분리해 재려면 보드 전체(`VDD_IN`) 대신 `VDD_SYS_GPU` 채널을 읽는다. 경로는 커널 버전에
따라 다르므로 먼저 `ls /sys/bus/i2c/drivers/ina3221x/*/iio:device0/`로 채널 이름을 확인한다.

```bash
--power cmd --power-cmd "cat /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input"
```

Denver2 코어는 기본 비활성인 경우가 있다. `sudo nvpmodel -q`로 활성 코어 구성을 기록해 둔다 —
CPU 측 런치 오버헤드가 결과에 들어가므로 어떤 코어 구성이었는지가 재현에 필요하다.

## 5. 기록할 것

두 기기 모두 다음을 그대로 남긴다. `paper/draft.md` §5.7의 표에 행으로 들어간다.

1. 스크립트 헤더 전체 (device, dtype, threads/gpu, torch 버전, `fused scan available`)
2. 표 전체 (cache on/off × 활성률 6점)
3. Jetson은 `nvpmodel -q` 출력과 모드, TX2는 활성 코어 구성까지
4. Pi는 측정 종료 후 `vcgencmd get_throttled` (0x0이 아니면 그 회차 폐기)
5. 전력계 종류와 어느 레일을 읽었는지 (보드 전체 / GPU 레일)
6. TX2는 fp32와 fp16 두 벌

판정 기준은 **dense와 sparse의 순서가 역전되는지**, 그리고 활성률에 대한 희소 경로의 기울기다.
4090에서는 dense가 평평하게 이겼다. 연산 제약 기기에서 희소가 활성률에 비례해 내려가면 §5.7의
"환산 여부는 하드웨어가 연산 제약인지에 달렸다"는 서술이 추측에서 측정으로 바뀐다.

## 6. 결과 붙일 위치

- `paper/draft.md` §5.7 "Four efficiency claims, separately scored" 표의 "희소성에 의한 실측 속도
  향상" 행과 그 아래 본문
- `paper/draft_ko.md` 같은 절
- `paper/self-revision/r1-revision-status.md` major 5번 항목
- 한계 4·5번 문장 (엣지 미측정 서술)
