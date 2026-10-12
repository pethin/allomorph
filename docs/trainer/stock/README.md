# Stock Engine Architecture & Reference Manual

**Engine Implementation:** [`src/allomorph/trainer/stock/`](../../../src/allomorph/trainer/stock/) | [`engine.py`](../../../src/allomorph/trainer/stock/engine.py)  
**Framework:** Upstream [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler) + PyTorch Lightning  
**Architecture Standard:** Architecture 2 (A2) Slimmable WaveNet (`channels_3` + `channels_8`)  
**Target Hosts:** Darkglass Anagram Block 1 (Preamp / NAM Host), Desktop DAW Plugins, Tone3000 Storefront  

---

## 1. Overview & Purpose

The **Stock Engine** (`allomorph.trainer.stock`) serves as the official upstream reference standard within the Allomorph modeling pipeline. It interfaces directly with the upstream `neural-amp-modeler` library and PyTorch Lightning trainer to train compliant Architecture 2 (A2) neural network digital twins.

While the bespoke **Native Engine** (`allomorph.trainer.native`) provides in-VRAM block permutation and fast 60–100 epoch convergence, the Stock Engine provides a standard, un-modified upstream NAM training baseline operating on a fixed 200-epoch schedule.

---

## 2. Model Architecture: Architecture 2 (A2) Slimmable

The stock engine packages models using the NAM Architecture 2 (A2) standard, which trains a dual-tier slimmable WaveNet in a single container:

```
+-----------------------------------------------------------------------------+
|                     Architecture 2 Container (.nam JSON)                   |
|                                                                             |
|  +--------------------------------+   +----------------------------------+  |
|  |       Lite Tier (channels_3)   |   |       Full Tier (channels_8)     |  |
|  |  Receptive Field: ~132.2 ms    |   |  Receptive Field: ~132.2 ms      |  |
|  |  Parameters: 1,871 weights     |   |  Parameters: 12,146 weights      |  |
|  |  Target: Low-power DSP hosts   |   |  Target: Darkglass Anagram Blk 1 |  |
|  +--------------------------------+   +----------------------------------+  |
+-----------------------------------------------------------------------------+
```

### 2.1 Submodel Dimensions
- **Full Tier (`channels_8`):** 8 internal dilated convolution channels ($12,146$ parameters). Captures complete non-linear magnetic feel, RLC resonance peaks, and spatial aperture notches for high-performance DSP hosts.
- **Lite Tier (`channels_3`):** 3 internal channels ($1,871$ parameters). Provides an ultra-efficient fallback tier for low-power microcontrollers and embedded hosts.

---

## 3. Training Hyperparameters & Standards

The stock engine strictly adheres to upstream NAM baseline defaults:

| Hyperparameter | Stock Standard | Description |
| :--- | :---: | :--- |
| **Max Epochs (`epochs`)** | `200` | Fixed 200-epoch training schedule (`STOCK_DEFAULT_MAX_EPOCHS`) |
| **Early Stopping** | None | Runs complete 200-epoch schedule to match upstream standards |
| **Optimizer** | Adam | Standard upstream Adam optimizer (`lr = 0.01`) |
| **Loss Formulation** | ESR + MSE | Standard upstream multi-tier Error-to-Signal Ratio and MSE |
| **Batch Size (`batch_size`)** | `"auto"` | Dynamically resolved based on host VRAM (`STOCK_DEFAULT_BATCH_SIZE`) |
| **Hardware Precision** | `"auto"` | Dynamically resolved to `bf16-mixed`, `16-mixed`, or `32-true` |
| **Worker Threads** | `"auto"` | Resolves to optimal CPU worker count (`STOCK_DEFAULT_NUM_WORKERS`) |

### 3.1 Hardware Precision Resolution
- **`bf16-mixed`:** Selected on native bfloat16 hardware (Apple Silicon MPS, modern NVIDIA/AMD GPUs).
- **`16-mixed`:** Selected on hardware supporting standard FP16 mixed precision.
- **`32-true`:** Fallback for CPU execution or architectures lacking native half-precision support.

### 3.2 Dynamic Batch Size Resolution
- $\ge 12\text{ GB}$ VRAM: Batch size `32`
- $6\text{--}12\text{ GB}$ VRAM: Batch size `16`
- $< 6\text{ GB}$ VRAM / CPU: Batch size `8`

---

## 4. Audio Pipeline & Dataset Integration

The stock engine seamlessly integrates with Allomorph's synthetic and measured audio stems:

```
[Dry Excitation Stem: input.wav] ──┐
                                   ├──> [configure_a2_architecture] ──> [PyTorch Lightning Trainer]
[Wet Target Stem: target.wav]   ──┘
```

1. **Dry/Wet Input Detection:** [`configure_a2_architecture`](file:///home/peter/Projects/pethin/allomorph/src/allomorph/trainer/stock/engine.py#L53-L135) hooks `_detect_input_version` and `_check_data` in upstream NAM, allowing arbitrary dry excitation and wet target stem pairs to be fed directly without dataset conversion scripts.
2. **Audio File Validation:** Verifies that audio files exist, have matching sample rates ($48\text{ kHz}$), matching channel layouts, and satisfy 24-bit PCM requirements.

---

## 5. Telemetry & Terminal Progress Logging

The stock engine uses [`EsrProgressCallback`](file:///home/peter/Projects/pethin/allomorph/src/allomorph/trainer/callbacks.py) to provide real-time terminal telemetry during the 200-epoch run:

```
Epoch 042/200 [1.85s] Train: 0.0018 | Val: 0.0012 | ESR Ch8: 0.000854 (-30.7 dB) | ESR Ch3: 0.001241 (-29.1 dB)
```

- **Per-Epoch Timing:** Tracks epoch compute duration and GPU throughput.
- **Dual-Tier ESR Monitoring:** Displays Error-to-Signal Ratio in both linear decimal and decibels ($\text{dB}$) for both `channels_8` and `channels_3`.
- **Learning Rate Tracking:** Reports active learning rate across scheduler decay cycles.

---

## 6. Container Serialization & Studio Metadata

Upon completion of the 200-epoch training schedule, the stock engine exports the final converged model directly into a compliant `.nam` JSON container with Allomorph studio reference metadata:

```json
{
  "version": "0.5.3",
  "architecture": "Architecture 2",
  "description": "Allomorph Universal Bass Digital Twin",
  "metadata": {
    "allomorph_version": "6.1.1",
    "dsp_generation": 2,
    "git_commit": "1bc3052...",
    "source_instrument": "34in_standard_p",
    "source_voicing": "precision_vintage",
    "target_voicing": "jazz_pair_active",
    "training": {
      "engine": "stock",
      "epochs": 200,
      "batch_size": 16,
      "precision": "bf16-mixed"
    }
  },
  "weights": { ... }
}
```

---

## 7. CLI Usage

### Direct Invocation via `scripts/train_nam.py`
```bash
# Train a target voice using the stock 200-epoch engine:
uv run python scripts/train_nam.py \
  --source-inst 34in_standard_p \
  --voice-id jazz_pair_active \
  --engine stock \
  --epochs 200 \
  --batch-size auto
```

### Pipeline Invocation via `allomorph`
```bash
# Run training stage across all configured voices with the stock engine:
uv run allomorph --stage train --engine stock
```
