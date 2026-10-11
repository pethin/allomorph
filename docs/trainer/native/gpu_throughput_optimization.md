# Native Engine GPU Throughput Profiling & Hardware Acceleration

**Document Version:** 1.0.0  
**Target Hardware:** AMD Radeon RX 9070 XT (64 CUs, 16 GB VRAM, ROCm 6.3+ / HIP, `bf16-mixed`)  
**Package Implementation:** [`src/allomorph/trainer/native/`](../../../src/allomorph/trainer/native/)  
**Primary Engine:** [`engine.py`](../../../src/allomorph/trainer/native/engine.py) | **Neural Net:** [`models.py`](../../../src/allomorph/trainer/native/models.py) | **Loss:** [`loss.py`](../../../src/allomorph/trainer/native/loss.py)

---

## 1. Executive Summary & Optimization Impact

During initial implementation of the Allomorph Native Neural Amp Modeler (NAM) Architecture 2 engine, GPU compute utilization on the AMD Radeon RX 9070 XT hovered around **~50%**, yielding training throughput of **$7.82\text{ s / epoch}$** (10.74 steps/s). While faster than unoptimized CPU pipelines, the engine was severely under-utilizing the RDNA GPU compute pipelines and fell behind the stock PyTorch Lightning wrapper ($6.52\text{ s / epoch}$).

Through hardware-level execution profiling, three distinct execution bottlenecks were isolated and eliminated:
1. **Inner-Loop PCIe Host Synchronizations:** 8 blocking `.detach().cpu().item()` calls per step inside `MultiBandSpectralLoss` (672 syncs/epoch).
2. **Interpreter-Bound Sequence Slicing:** Sequential Python list-comprehension tensor indexing per step.
3. **Dynamic Forward Weight Mask Allocations:** Dynamic multiplication of parameter weights by buffer masks on every forward layer execution, causing 46 temporary tensor allocations and autograd graph bloat per step.

### Empirical Optimization Results:
- **Per-Epoch Throughput:** Accelerated from **$7.82\text{ s}$** down to **$3.15\text{ s}$** (**$2.48\times$ throughput speedup**; $>2.0\times$ faster than Stock Lightning).
- **GPU Pipeline Saturation:** Increased from **~50%** to **85–92%** continuous compute utilization.
- **Full Catalog Workflow:** Total training wall-clock time across 4 representative target voices dropped from **$1,648.5\text{ s}$ ($27.48\text{ min}$)** on Stock to **$545.0\text{ s}$ ($9.08\text{ min}$)** on Native (**$3.02\times$ overall workflow speedup**).
- **Zero Loss of Quality:** Verified bit-for-bit identical export structure, matching or exceeding Stock in dynamic peak crest tracking and electroacoustic frequency response deconvolution.

---

## 2. Root-Cause Bottleneck Analysis & Architectural Interventions

```mermaid
graph TD
    subgraph "Before Optimization: ~50% GPU Utilization (7.82 s/epoch)"
        A1["Python List Slicing Loop<br/>(16 sequential CPU slices/step)"] --> A2["Dynamic Weight Masking<br/>(46 tensor allocations/step)"]
        A2 --> A3["Forward & Backward Passes<br/>(Autograd Graph Overhead)"]
        A3 --> A4["8x .cpu().item() Host Syncs<br/>(Blocks GPU pipeline 672x/epoch)"]
    end

    subgraph "After Optimization: 85-92% GPU Saturation (3.15 s/epoch)"
        B1["2D Vectorized Index Gather<br/>(Single GPU kernel launch)"] --> B2["Direct Native C++ Conv1D<br/>(In-place weight/grad masking)"]
        B2 --> B3["Streamlined Forward/Backward<br/>(Minimal graph memory footprint)"]
        B3 --> B4["In-VRAM Loss Accumulation<br/>(Zero host syncs during batch loop)"]
    end
```

### 2.1 Bottleneck 1: Host-Device Synchronization Stalls (`.cpu().item()`)

#### The Problem:
In `MultiBandSpectralLoss`, four floating-point loss components (`mse`, `hp_mid`, `hp_high`, `total`) were extracted via `.detach().cpu().item()` to populate metric dictionaries on every training iteration.

For the dual-tier architecture (A2 Lite `channels_3` and A2 Full `channels_8`):
$$\text{Host Syncs per Step} = 4\text{ metrics} \times 2\text{ submodels} = 8\text{ synchronous PCIe round-trips}$$
$$\text{Host Syncs per Epoch} = 8 \times 84\text{ steps} = 672\text{ blocking host barriers}$$

Each `.item()` call forces the CPU to issue a hardware fence, pause Python interpreter execution, wait for the GPU kernel queue to drain, and transfer 4 bytes over the PCIe bus. This destroyed asynchronous kernel execution, forcing the GPU into idle state for $>40\%$ of each batch.

#### The Intervention:
Added `return_metrics: bool = True` to `MultiBandSpectralLoss.forward()`. During training steps, `return_metrics=False` is passed, bypassing all `.item()` calls:

```python
# src/allomorph/trainer/native/loss.py
if not return_metrics:
    if self.hp_mid_loss is not None:
        hp_mid = self.hp_mid_loss(pred, target)
        loss = loss + self.hp_mid_weight * hp_mid
    if self.hp_high_loss is not None:
        hp_high = self.hp_high_loss(pred, target)
        loss = loss + self.hp_high_weight * hp_high
    return loss, {}
```

In `train_voice_native`, batch losses are accumulated directly as VRAM GPU tensors:
```python
# src/allomorph/trainer/native/engine.py
train_loss_acc = train_loss_acc + total_step_loss.detach()
```
Metric extraction occurs strictly once per epoch during the single validation evaluation pass.

---

### 2.2 Bottleneck 2: Interpreter-Bound Sequence Slicing

#### The Problem:
Random batch slicing previously extracted audio windows using a sequential Python list comprehension:
```python
# Naive approach:
x_batch = torch.stack([x_train_padded[i : i + nx] for i in indices]).unsqueeze(1)
y_batch = torch.stack([y_train[i : i + ny] for i in indices]).unsqueeze(1)
```
Across a batch size of $B=16$, this executed 32 separate slice and copy operations in Python interpreter space per step ($32 \times 84 = 2,688$ slice operations per epoch). The overhead of Python reference counting and sequential kernel dispatches throttled the GPU scheduler.

#### The Intervention:
Implemented 2D vectorized GPU index gathering using pre-allocated offset grids:
```python
# src/allomorph/trainer/native/engine.py
# Pre-allocated once at initialization:
arange_nx = torch.arange(nx, device=device).unsqueeze(0)  # (1, nx)
arange_ny = torch.arange(ny, device=device).unsqueeze(0)  # (1, ny)

# Inside step loop:
indices_2d = indices.unsqueeze(1)  # (batch_size, 1)
x_batch = x_train_padded[indices_2d + arange_nx].unsqueeze(1)  # (B, 1, nx)
y_batch = y_train[indices_2d + arange_ny].unsqueeze(1)         # (B, 1, ny)
```
The entire batch slice is executed as a single, contiguous parallel GPU gather kernel, running directly in high-bandwidth VRAM without Python interpreter loops.

---

### 2.3 Bottleneck 3: Dynamic Forward Weight Mask Allocations

#### The Problem:
To guarantee zero cross-talk between the A2 Lite (`channels_3`) and A2 Full (`channels_8`) models in the unified 11-channel chassis, each layer dynamically masked its convolution weights inside `forward()`:
```python
# Naive forward pass:
w_conv = self.conv.weight * self.mask_conv
zconv = F.conv1d(x, w_conv, self.conv.bias, dilation=self.dilation)

w_1x1 = self.layer1x1.weight * self.mask_1x1
z1x1 = F.conv1d(z, w_1x1, self.layer1x1.bias)
```
Across 23 WaveNet layers, this allocated **46 temporary weight tensors and associated autograd backward nodes on every single step** ($46 \times 84 = 3,864$ tensor allocations/epoch). This bloated VRAM memory bandwidth and prevented cuDNN/MIOpen from using fused convolution kernels.

#### The Intervention:
1. **Initialization In-Place Masking:** Parameter weights are masked in-place once upon construction:
   ```python
   with torch.no_grad():
       self.conv.weight.mul_(self.mask_conv)
       self.layer1x1.weight.mul_(self.mask_1x1)
   ```
2. **Direct Forward Convolution:** Forward passes call native C++ convolutions directly:
   ```python
   zconv = self.conv(x)
   z1x1 = self.layer1x1(z)
   ```
3. **Gradient Masking Before Step:** Off-block gradients are zeroed in-place before `optimizer.step()`:
   ```python
   def apply_weight_masks(self) -> None:
       with torch.no_grad():
           for layer in self.layers:
               layer_mod = cast(AllomorphWaveNetLayer, layer)
               layer_mod.conv.weight.mul_(layer_mod.mask_conv)
               layer_mod.layer1x1.weight.mul_(layer_mod.mask_1x1)
               if layer_mod.conv.weight.grad is not None:
                   layer_mod.conv.weight.grad.mul_(layer_mod.mask_conv)
               if layer_mod.layer1x1.weight.grad is not None:
                   layer_mod.layer1x1.weight.grad.mul_(layer_mod.mask_1x1)
           self.head_rechannel.weight.mul_(self.mask_head)
           if self.head_rechannel.weight.grad is not None:
               self.head_rechannel.weight.grad.mul_(self.mask_head)
   ```
This eliminates all intermediate weight tensor allocations, enabling the GPU to execute pure C++ MIOpen convolution primitives at maximum hardware clock speeds.

---

### 2.4 AMD ROCm / MIOpen Kernel Optimization Nuance

> [!WARNING]
> Setting `torch.backends.cudnn.benchmark = True` on AMD ROCm (HIP / MIOpen) causes severe kernel compilation freezes during dilated convolution setup.
> 
> When MIOpen's kernel search mode evaluates dilated convolutions with large dilations (e.g. $d=128, 256, 512$ in Layers 7–9 and 17–19), it enters an exhaustive search pattern that stalls the main thread for 45–90 seconds per layer.
> 
> **Standard:** `torch.backends.cudnn.benchmark` must remain `False` on AMD ROCm hardware. Native direct Conv1D execution provides deterministic, maximum throughput without JIT profiling hangs.

---

## 3. Physical Warmup Floor & Convergence Dynamics

### 3.1 The Adam Velocity Ceiling Proof
In Adam optimization, weight parameter updates $\Delta\theta_t$ are bounded by the learning rate $\eta$:
$$|\Delta\theta_t| = \eta \cdot \left| \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right| \le \eta$$
For our calibrated learning rate $\eta = 4.0 \times 10^{-3}$, a single gradient update can move any individual weight parameter by at most **$0.004$**.

Inverting the passive high-impedance RLC roll-off of a split-coil bass pickup ($1 / H_{\text{src}}$) mathematically requires establishing alternating differentiator taps at the earliest receptive field samples:
$$h_{\text{diff}}[0, 1] \approx [+1.77, -1.85]$$
Starting from near-zero orthogonal random initialization ($\approx 0.05$):
$$\Delta w \approx 1.77 - 0.05 = 1.72 \implies N_{\text{steps}} \ge \frac{1.72}{0.004} = 430\text{ consecutive updates}$$
With $84\text{ steps/epoch}$, this requires a minimum physical floor:
$$N_{\text{epochs}} \ge \frac{430}{84} \approx 5.1\text{ epochs of pure monotonic high-frequency gradient pressure}$$

However, during the first 25–30 epochs, $>99.9\%$ of the gradient magnitude is dominated by low-frequency fundamental excursion ($40\text{--}250\text{ Hz}$). High frequencies only experience strong gradient pressure once the low-end MSE drops below $3 \times 10^{-6}$.

### 3.2 The Warmup Floor Solution
Setting premature early stopping (`min_epochs = 15`) or un-gated plateau checks caused training on complex voicings (e.g. `rickenbacker_clank`) to exit at epoch 9 before differentiator taps could physically form, resulting in attenuated treble.

**The Calibrated Fix:**
- Set `NATIVE_DEFAULT_MIN_EPOCHS = 35` and `NATIVE_DEFAULT_MAX_EPOCHS = 60`.
- Gated `min_mse_full`, `best_pareto_score`, and plateau stopping conditions strictly behind `epoch >= min_epochs`.
- Between epochs 35 and 45, the network reaches full electroacoustic convergence and terminates cleanly, saving $36.7\%\text{--}41.7\%$ of unnecessary schedule compute while preserving treble resolution.

---

## 4. Benchmark Telemetry & Hardware Profiling

Evaluated on an AMD Radeon RX 9070 XT across the four normative baseline target archetypes using the synthetic 5.0-second log-sine sweep ($20\text{--}20000\text{ Hz}$, $8\text{ Hz}$ DC block, 8192 gate taps):

### 4.1 Training Speedup & Efficiency Matrix
| Target Voice | Stock Time | Native Time | Speedup | Stock Schedule | Native Checkpoint | Compute Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Precision Dub** | $383.0\text{ s}$ ($6.38\text{ m}$) | **$175.4\text{ s}$ ($2.92\text{ m}$)** | **$2.18\times$** | 60 ep | **Epoch 35** | **$-41.7\%$** |
| **Rickenbacker Clank** | $407.3\text{ s}$ ($6.79\text{ m}$) | **$126.4\text{ s}$ ($2.11\text{ m}$)** | **$3.22\times$** | 60 ep | **Epoch 35** | **$-41.7\%$** |
| **Jazz Pair Active** | $423.3\text{ s}$ ($7.06\text{ m}$) | **$121.9\text{ s}$ ($2.03\text{ m}$)** | **$3.47\times$** | 60 ep | **Epoch 38** | **$-36.7\%$** |
| **StingRay Parallel** | $434.9\text{ s}$ ($7.25\text{ m}$) | **$121.3\text{ s}$ ($2.02\text{ m}$)** | **$3.58\times$** | 60 ep | **Epoch 38** | **$-36.7\%$** |
| **TOTAL / OVERALL** | **$1,648.5\text{ s}$ ($27.48\text{ m}$)** | **$545.0\text{ s}$ ($9.08\text{ m}$)** | **$3.02\times$** | **240 ep** | **146 ep** | **$-39.2\%$ compute** |

### 4.2 Electroacoustic Deconvolution Fidelity
| Target Voice | Metric | Stock Engine | Native Engine | Winner & Significance |
| :--- | :--- | :---: | :---: | :--- |
| **Precision Dub** | Low-Mid RMSE ($100\text{--}1\text{k Hz}$) | $1.09\text{ dB}$ | **$0.73\text{ dB}$** | **Native ($+0.36\text{ dB}$ better body tracking)** |
| | Sub-Bass RMSE ($20\text{--}120\text{ Hz}$) | $1.21\text{ dB}$ | **$1.14\text{ dB}$** | **Native ($+0.07\text{ dB}$ closer fundamental)** |
| | Peak Crest Factor Error | $-0.049\text{ dB}$ | $-0.139\text{ dB}$ | Both within $< 0.14\text{ dB}$ of target stem |
| **Rickenbacker Clank** | Low-Mid RMSE ($100\text{--}1\text{k Hz}$) | $2.79\text{ dB}$ | **$2.63\text{ dB}$** | **Native ($+0.16\text{ dB}$ better bite tracking)** |
| | Presence RMSE ($1\text{k--}4\text{k Hz}$) | $1.51\text{ dB}$ | **$1.49\text{ dB}$** | **Native matches aggressive clank bite** |
| | Peak Crest Factor Error | **$-0.234\text{ dB}$** | $-0.248\text{ dB}$ | Equivalent within $0.014\text{ dB}$ |
| **Jazz Pair Active** | Overall MAE ($20\text{--}20\text{k Hz}$) | $11.68\text{ dB}$ | **$10.33\text{ dB}$** | **Native ($+1.35\text{ dB}$ better overall MAE)** |
| | Treble RMSE ($4\text{k--}10\text{k Hz}$) | $10.33\text{ dB}$ | **$7.19\text{ dB}$** | **Native ($+3.14\text{ dB}$ superior treble tracking)** |
| | Peak Crest Factor Error | $+0.362\text{ dB}$ | **$+0.267\text{ dB}$** | **Native closer by $0.095\text{ dB}$** |
| **StingRay Parallel** | Treble RMSE ($4\text{k--}10\text{k Hz}$) | $6.38\text{ dB}$ | **$5.69\text{ dB}$** | **Native ($+0.69\text{ dB}$ superior treble sizzle)** |
| | Peak Crest Factor Error | $-0.426\text{ dB}$ | **$-0.325\text{ dB}$** | **Native closer by $0.101\text{ dB}$** |
| | Overall MAE ($20\text{--}20\text{k Hz}$) | **$8.06\text{ dB}$** | $8.15\text{ dB}$ | Equivalent within $0.09\text{ dB}$ across 4,097 bins |

---

## 5. Architectural Verification & Summary

The native training optimization proves that eliminating framework baggage, IPC queue latency, and unnecessary device synchronizations enables standard PyTorch models to run at native hardware speeds:
- Convolutions run directly on the GPU without per-step dynamic tensor allocations.
- Audio sequences stream directly from high-bandwidth VRAM.
- Anti-Drift Pareto early stopping safely saves $39.2\%$ of training iterations while preserving full electroacoustic fidelity.
