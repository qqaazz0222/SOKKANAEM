# Pi 4 / Jetson measurement handoff

Status: user confirms Jetson Nano Developer Kit B01, already measured in 5W and 10W modes.
Both historical logs are present; see [their audit](EDGE_LEGACY_AUDIT.md). They are synthetic
activity/cache measurements, not validation of the new real-PNG runner. No new device run or
SSH connection has been performed in this closeout. For additional real-video runs only,
provide `user@host` and whether an existing SSH key is usable, or return locally produced JSON.
Do not send passwords. The existing edge ZIP predates this clarification; use these updated instructions.

The [official Nano power guide](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html)
and [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) distinguish Nano
from newer Orin devices. **Do not reflash, upgrade JetPack, or install desktop CUDA wheels
on an unidentified device.** Existing device-specific PyTorch compatibility must be checked first.
The package's old-Python syntax is intentional, but ARM/JetPack runtime compatibility is not yet tested.

## Private bundle

`work_dirs/paper_edge_bundle.zip` contains the exact EMA tensors exported from the frozen
checkpoint, recorded model configuration/code, and 1024 development RGB frames (no final-test
data and no GT). This is a private local convenience copy; authors must check third-party
dataset rights before sharing it outside their authorized research environment.
It does not include a Python installation. The runner uses existing Python, PyTorch, NumPy,
Pillow and a TOML reader (`tomllib`, `tomli`, or `toml`, as appropriate). GMC is disabled.

After extracting, change into `paper_edge_bundle`. Inspect hardware without importing PyTorch:

```bash
python3 scripts/paper_edge_input_bench.py --inspect
```

Pi CPU smoke, then full evaluation only if smoke is finite and the machine remains stable:

```bash
python3 scripts/paper_edge_input_bench.py --device cpu --threads 4 --reference-scan --smoke --output pi_smoke.json
python3 scripts/paper_edge_input_bench.py --device cpu --threads 4 --reference-scan --output pi_full.json
```

Nano B01 with an already working CUDA PyTorch installation (example uses four threads;
for a 5W run explicitly record an appropriate thread count and use a different output filename):

```bash
python3 scripts/paper_edge_input_bench.py --device cuda --threads 4 --reference-scan --smoke --output jetson_smoke.json
python3 scripts/paper_edge_input_bench.py --device cuda --threads 4 --reference-scan --output jetson_full.json
```

Reference-scan measurements must be labeled as such: they do not measure the desktop Triton
kernel. The full run processes K30 and same-weight dense carry, one warmup plus five repeats,
alternating model order, batch one, FP32, all four development sequences at 256 pixels.
Frames stream from PNG files and outputs are copied to CPU. No clip is preloaded as a tensor.
Loading, checksums, telemetry and writing JSON are outside the timing boundary. Full runs can
be slow or exceed a small device's memory; report failure rather than silently reducing input size.

Record RAM, OS, storage medium, power mode, cooling, ambient conditions and throttling indicators.
The runner records available board/OS/thermal/memory information and does not change clocks.
CPU peak resident memory is not measured by this runner; available-memory snapshots are not a
substitute. GPU allocated memory is not total system memory on a shared-memory Jetson.

Energy in the **new real-video runner** remains unmeasured; the historical Nano logs do record
sensor-based aggregate power/energy, with the limitations in the audit. No assumed mode wattage,
unidentified sensor or desktop power is
converted into joules/frame. An identified calibrated board-input meter with aligned timing,
idle baseline and documented coverage is required for an energy claim. Device output hashes
are checked across repetitions, not for bitwise equality across architectures. No-GT timing
alone does not establish preservation of depth accuracy on the new runtime.
