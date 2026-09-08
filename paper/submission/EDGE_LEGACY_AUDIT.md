# Jetson Nano B01: historical measurement audit

2026-09-07. The user confirmed **Jetson Nano Developer Kit B01**, measured in **5W and
10W power modes**. Existing logs were located in `measures/nano-b01-5w.log` and
`measures/nano-b01-10w.log`. The earlier blanket statement that no edge measurements
existed was incorrect. No new device run was performed during this audit.

## Recorded measurements

Both logs record CUDA, FP32, size 256, PyTorch 1.10.0, NVIDIA Tegra X1 and
`fused scan available: False`. The 5W log records two CPU threads; the 10W log records four.
Each contains six forced activity levels with caches on/off (12 rows per mode).
The following is the **5.1% activity operating point**, not the mean activity of real video.

| User-confirmed mode | Cache | ms/frame | FPS | Sensor mean W | Reported mJ/frame |
|---|---|---:|---:|---:|---:|
| 5W | on | 157.93 | 6.3 | 3.91 | 617.40 |
| 5W | off | 2130.72 | 0.5 | 4.16 | 8863.44 |
| 10W | on | 114.27 | 8.8 | 5.55 | 634.62 |
| 10W | off | 1564.05 | 0.6 | 6.20 | 9701.15 |

At matched 5.1% activity, off/on time ratios are **13.49× (5W)** and **13.69× (10W)**;
reported energy ratios are **14.36×** and **15.29×**. Cache-off at the same mask is the
comparator, not necessarily the final paper's all-active dense-carry implementation.
The cache-on 10W point takes 27.65% less time but reports 2.79% more energy/frame than
the 5W point. These are descriptive ratios of single aggregate log rows, without confidence
intervals. Mode and thread count change together; this does not isolate a power-setting effect.
Printed watts/time are rounded, so their product need not exactly equal printed energy.

## Measurement boundary and limitations

The current `scripts/edge_bench.py` uses a repeated random GPU-resident image, a fixed
active-coordinate mask, disabled activity fallback, and warmed recurrent/cache state.
It times `model.step` and does not read real PNGs, execute the natural change detector,
include file/decode/transfer/CPU-output costs, or evaluate GT accuracy. Its forced mask
does not reproduce natural keyframe/refresh schedules. Therefore these results are a
**historical synthetic activity/cache microbenchmark**, not 13× end-to-end video speedup
with preserved depth accuracy. The exact source revision used then is not in the logs;
the current code explains the apparent procedure but is not proof of historical code identity.

The logs identify `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input`.
[NVIDIA's Nano power documentation](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html)
maps this channel to **VDD_IN, main module power input**. It is not automatically wall-socket
or whole developer-kit peripheral power. The same documentation describes 5W/MaxN modes;
their labels are configuration budgets, not measured constant consumption.

The current script computes energy as sampled mean power times average inference duration.
Raw power samples, sample counts, independent timing repeats, idle baseline, temperatures,
throttling/clock traces, exact launch command and `nvpmodel -q` output are absent from these logs.
Successful aggregate power output does not prove that every sensor read succeeded.
Checkpoint/config hashes and source commit are also absent. The operating mode is supported
by the filename and user confirmation, not by an embedded device-mode dump.

Thus retain this evidence as an explicitly qualified supplement. Before promoting it to
the frozen model's principal efficiency claim, link weights/code/config, repeat the actual
video pipeline in both modes, check runtime accuracy and record uncertainty/power scope.
This does not require repeating the historical sweep merely because SSH is unavailable.

## Preserved bytes

These hashes were calculated now; they do not retroactively establish run-time provenance.

| File | SHA-256 |
|---|---|
| `measures/nano-b01-5w.log` | `b38ed5eec1a869853253f772a965eda2481f3083aa23f84095affe7c76be6766` |
| `measures/nano-b01-10w.log` | `60d769f4813a89f412fcf49d4b3964c66d72f6bac16accfa48fcdb129ef3c729` |
| current `scripts/edge_bench.py` | `738997b812309b5af8516cd33820e02ebe0c8b0d145b0ada6972b3016f66d0fc` |

The previously generated edge ZIP is preserved as a dated preparation artifact. Its enclosed
instructions predate device identification; use the current `EDGE_RUN.md` and this audit.
