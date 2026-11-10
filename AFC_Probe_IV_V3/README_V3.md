# Probe-IV AFC V3：有限样本精确矩 + 交叉拟合结构融合

## 状态和重要结论

V3 为 V2 的增量版本，**没有改写** `afc.py`、`optimized_iv.py`、V1/V2 的实验结果和原始 MAT 文件。V3 的辨识器仅接收观测到的麦克风、实际扬声器、独立探测信号。**真实反馈路径 h 和原始目标 s 都不参与辨识和模型选择**，仅用于仿真环境的外部评分。

20 个独立、在开发阶段未使用的随机种子，分别对比 V2 与 V3；同一实验行里的两个算法接收逐点相同的麦克风、扬声器、探测数据：

| 辨识时长 | 算法 | 平均 NFCE | 平均 ASG | 最小 ASG | 10秒独立目标回声抑制 |
| --- | --- | ---: | ---: | ---: | ---: |
| 60 秒 | V2 结构化 IV | 0.1667 | +9.145 dB | — | 8.237 dB |
| 60 秒 | **V3 IV 结构融合** | **0.1429** | **+9.747 dB** | +7.875 dB | **8.870 dB** |
| 120 秒 | V2 结构化 IV | 0.1076 | +10.852 dB | — | 10.108 dB |
| 120 秒 | **V3 IV 结构融合** | **0.0936** | **+11.372 dB** | +9.099 dB | **10.778 dB** |

- 60 秒：20 次中的 14 次相对 V2 获得更高 ASG；平均配对提升 **+0.601 dB**，正态近似 95% 区间 [+0.083,+1.120] dB，ASG ≥10 dB 的次数 **8/20**。
- 120 秒：20 次中的 13 次相对 V2 获得更高 ASG；平均配对提升 **+0.519 dB**，正态近似 95% 区间 [+0.088,+0.951] dB，ASG ≥10 dB 的次数 **15/20**。
- 固定反馈路径：归档的 128 tap 实测反馈 FIR；16 kHz；前向增益 1；延迟 D=1；高通有色近端噪声；探测为**有界白色二值信号** RMS=0.05，始终未提高。
- **没有达到**之前提出的“60 秒平均 +10 dB、120 秒平均 +13 dB”研究目标；绝不把本结果写成达到。
- 上面的 ASG 依旧为 **MSG−MSG0** 的 16384 点 FFT 网格指标，**不是实物设备测得的增益临界值**。

## 方法

1. **修正有限长度相关矩阵边界**：对完整有效 FIR 回归行，精确求解 `G=Z.T@X/(N-L)`、`b=Z.T@mic/(N-L)`；使用 Toeplitz 批量互相关加上首尾边界修正，没有创建巨大 `N×128` 矩阵。`test_v3_iv.py` 使用显式矩阵乘法逐项验证等价性。
2. **交叉拟合**：分半辨识获得原始 IV 估计 `r1,r2`，用它们的差异估计每 tap 噪声，构造旧 SURE 候选和 50 个数据驱动连续支撑 / 局部 Wiener 收缩候选。
3. **按近似交叉验证风险平均**：对于候选 k，计算 `loss_k=||denoise_k(r1)-r2||²+||denoise_k(r2)-r1||²`，并按 `exp(-(loss_k-minloss)/(2*T))` 赋予权重，`T=3*sqrt(2*L)*sigma_half²`。最终在整个样本上重新计算各个候选，再按权重融合。系数与误差评估真值不参与权重选择。
4. **保留安全的控制组和回退**：V2 仍可调用。直接增大全局系数的交叉拟合标量增益，以及额外的后处理硬/软阈值，在开发种子上没有持续帮助或使结果变差，因此 V3 默认关闭。

参数 `temperature_factor=3` 是从**开发集**（seed 20500100 起的 10 次 60 秒实验）选择的。两个最终评估集 seed **20661102**（60 秒）和 **20761102**（120 秒）均未用于选择该参数。开发种子与这些验证种子不重叠，但**所有实验仍使用同一条物理反馈路径**，因此不能据此宣称适用于其他路径。

## 运行 Python

安装和测试（Python 依赖见 `requirements.txt`）：

```bash
pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -v
python stream_v3_example.py --seconds 60 --probe-rms 0.05
```

逐次重新生成结果（使用新的输出文件夹，不覆盖归档结果）：

```bash
python compare_v3.py --seconds 60 --probe-rms 0.05 --trials 20 --seed 20661102 --output verify_v3_60s
python compare_v3.py --seconds 120 --probe-rms 0.05 --trials 20 --seed 20761102 --output verify_v3_120s
```

实际音频流接入方式沿用 `ProbeAFC`：

```python
from afc import ProbeAFC
from v3_iv import identify_v3

afc = ProbeAFC(taps=128, delay=1, gain=1.0, probe_rms=0.05)
# 音频实时循环中：
error, speaker, probe = afc.process_mic(microphone_sample)
# 向物理扬声器输出 speaker；要保证已标定其实际延迟和增益。

# 在非实时工作线程完成标定后：
estimate = identify_v3(afc.microphones, afc.loudspeakers, afc.probes, 128)
afc.activate(estimate.to_identification(), fade_samples=1600)
afc.probe_rms = 0.0
afc.clear_recorded_audio()
```

**不要把矩阵辨识放在音频实时回调中。** 此流式接口和 `stream_v3_example.py` 是实测 FIR 作为被控对象的仿真，不是声学硬件认证。

## MATLAB 版本

提供 `identify_iv_v3.m` 和 `run_v3_iv_afc.m`：

```matlab
r = run_v3_iv_afc('data/afc_inputs_10s.mat',60,0.05);
```

MATLAB 移植版遵循 Python 有效 FIR 行的确切样本索引和权重公式；**当前环境没有 MATLAB 或 Octave，尚未执行数值一致性验证**。不能宣称 MATLAB 与 Python 已经逐点一致。

## 文件说明

- `v3_iv.py`：V3 核心，无源信号/真值入口，可直接适配 `ProbeAFC.activate()`。
- `compare_v3.py`：20 种子公平成对对照；CSV、JSON、PNG 输出；外部仿真评分。
- `stream_v3_example.py`：逐点闭环仿真、标定、平滑激活、关闭探测、独立输入测试。
- `test_v3_iv.py`：精确相关矩阵、未知路径延迟、零探测拒绝、参数检查。
- `results_v3_heldout_60s_20seed`、`results_v3_heldout_120s_20seed`：独立实验归档。
- `identify_iv_v3.m`、`run_v3_iv_afc.m`：待 MATLAB 本地验证的移植版。

## 未解决的实际问题

- 120 秒固定 RMS=0.05 标定时间过长，且探测噪声的可闻度尚未评估。
- 还没有对真人语音、音乐、多条不同反馈路径、路径突变、高前向增益，以及扬声器非线性做系统性验证。
- 系统尚无生产级硬件增益控制、功率限制器、啸叫检测和声学延迟校准；真实设备必须单独完成这类保护。
- V3 改进幅度有限，**不是**取代 V2 的普遍最优模型。其他路径可能不提升。
