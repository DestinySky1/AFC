# Probe-IV AFC V2：交叉验证的结构化反馈路径估计

**这是与 V1 并存的增量实现；旧版 `afc.py`、`demo.py` 和原有结果保持不变。**

## 已验证结果

同一128点实测反馈路径，采样率16 kHz，D=1，gain=0 dB；目标是65系数高通滤波有色高斯噪声。每组20个独立随机种子，扬声器辨识信号为**有界二值白噪声**，RMS始终为0.05。前一阶段 Probe-IV+SURE 和 V2 使用**逐次完全相同**的麦克风、扬声器、探测信号数据，比较公平。结果均来自Python闭环仿真。

| 辨识时长 | 算法 | 平均 NFCE | 平均ASG / dB | 最低ASG / dB | 独立目标回声抑制 / dB |
| --- | --- | ---: | ---: | ---: | ---: |
| 60秒 | V1 Probe-IV+SURE | 0.2696 | +6.497 | — | 5.898 |
| 60秒 | **V2 结构化IV** | **0.1692** | **+8.673** | +6.884 | **8.129** |
| 120秒 | V1 Probe-IV+SURE | 0.1623 | +8.411 | — | 8.087 |
| 120秒 | **V2 结构化IV** | **0.1065** | **+11.230** | +8.627 | **10.111** |

V2相对同期同数据V1的平均ASG提升：

- 60秒：+2.176 dB；配对20次的近似95% t区间 [+1.329,+3.022] dB。
- 120秒：+2.820 dB；配对20次的近似95% t区间 [+2.121,+3.518] dB。

60秒V2的20/20次ASG为正，2/20次ASG达到10 dB；120秒V2的20/20次为正，15/20次达到10 dB。**因此60秒以内平均ASG≥10 dB这一研究目标尚未达成；120秒组平均达到，但并非每次达到。**

两组的校准扬声器峰值分别不超过0.568、0.595（归一化仿真幅值），无非有限输出；这里没有真实设备音量校准。所称ASG仍是FFT16384网格的`MSG-MSG0`，并非硬件上实测的增益临界值。

## V2到底改进了什么

保持工具变量识别方程不变：

`G = E[z x^T]`，`b = E[z m]`，`G * h ≈ b`；仪器变量`z`来自**独立扬声器探测信号**。

V1对全128维IV估计做SURE软阈值。V2发现，这条实测反馈路径的主要响应能量在短时间区间内集中，因此进一步采用：

1. **自动识别响应区间**：用估计到的IV系数和从数据分半得到的噪声尺度，扫描连续区间；不从真实路径读取起始位置。
2. **局部Wiener收缩**：根据估计系数的局部能量，保留主要响应并抑制噪声；比逐系数硬阈值更平滑。
3. **两半记录交叉验证**：第一半数据生成候选估计，第二半原始IV估计作为独立近似验证目标，反向再做一次。选择平均预测误差更低的区间宽度、局部平滑宽度，最后在整段数据上重新估计。
4. **旧版SURE保底候选**：当结构化模型得不到较好的交叉验证分数时，可以保留旧版估计。

候选长度为16、20、24、28、32、36、40、48、64、全长（按实际滤波器长度过滤）；局部窗口4、8、12、16、24。V2的参数是**基于可观测数据选择**，并非通过在运行时读取真实路径、NFCE或ASG选取。真值只在外部仿真评分模块出现。

**边界**：交叉验证优化的是近似路径均方误差，不是直接最小化最坏频点的ASG；有限样本时V2有个别记录不如V1。短时间内路径剧烈变化、稠密长路径或多簇路径需要另行验证。

## 额外尝试但未纳入默认方案的模块

- AR预测误差加权IV：5条开发记录中，对ASG提升很小，未显示可靠的优势。
- 将同等RMS探测信号强烈高通整形：部分情形造成工具变量相关矩阵病态，未选用。
- 单纯逐系数硬阈值：在同样数据上劣于结构化局部估计。

保留**独立白色有界探测信号**与原样反馈消除更新，避免引入未经验证的复杂度和增大探测幅度。

## 运行

首先：

```bash
pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -v
python stream_optimized_example.py --seconds 60 --probe-rms 0.05
```

分别重做60秒与120秒的多种子实验（请使用新输出目录，不覆盖归档结果）：

```bash
python compare_optimized.py --seconds 60 --probe-rms 0.05 --trials 20 --seed 20421102 --output verify_60s
python compare_optimized.py --seconds 120 --probe-rms 0.05 --trials 20 --seed 20461102 --output verify_120s
```

每次运行输出：`per_trial.csv`、`summary.json`、`comparison.png`、首个种子的两个估计路径。新实验的SURE、V2共享观察数据，不使用目标/真实路径作为算法输入。

### Python逐样本音频接口

```python
from afc import ProbeAFC
from optimized_iv import identify_structured

afc = ProbeAFC(taps=128,delay=1,gain=1.,probe_rms=0.05)
# 辨识阶段：在音频回调中连续处理真实麦克风样本：
error, speaker, probe = afc.process_mic(microphone_sample)
# 将 speaker 输出到物理扬声器；采集的麦克风应包含它之后传播的声反馈。

# 在非实时线程中，60秒或120秒以后：
model = identify_structured(afc.microphones,afc.loudspeakers,afc.probes,128)
afc.activate(model.to_identification(),fade_samples=1600)
afc.probe_rms = 0.0
afc.clear_recorded_audio()
# 继续处理麦克风输入，已关闭外加探测信号。
```

路径变化时，需要重新打开探测并重新辨识。`ProbeAFC`每个采样的抵消核心未修改；求解大矩阵不应该放在实时音频回调线程中。

### MATLAB

```matlab
r = run_structured_iv_afc('data/afc_inputs_10s.mat',60,0.05);
```

`identify_structured_iv.m`是Python算法的MATLAB独立移植版，但**当前环境没有MATLAB，因此尚未运行验证**；请让本地Codex和MATLAB逐项核对。

## 包含的主要新增文件

- `optimized_iv.py`：交叉拟合、自动区间定位、局部收缩，导出可直接送入`ProbeAFC.activate`的结果。
- `compare_optimized.py`：20个独立随机种子和外部评分，成对比较新旧算法。
- `stream_optimized_example.py`：真正逐样本的闭环仿真、辨识、系数激活、停止探测。
- `test_optimized_iv.py`：不同路径位置、反馈路径变化、零探测拒绝、可直接激活等测试。
- `identify_structured_iv.m`、`run_structured_iv_afc.m`：MATLAB移植与实验入口。
- `results_structured_cv_20seed`和`results_structured_cv_120s_20seed`：实际运行结果和图。

## 仍未解决的问题

- **实物设备验证尚未完成**。实测反馈FIR是仿真被控对象，不代表真实硬件的音频I/O、相位/延迟、增益或声学路径标定已正确。
- 探测信号RMS=0.05未经过听感或心理声学掩蔽验证，可能可闻。120秒持续探测也可能影响实际使用体验。
- 当前ASG以路径残差的频域幅值估计，不能替代包含真实相位、完整总延迟、扬声器非线性和硬件增益扫描的稳定性试验。
- 与V1相比，60秒显著改善但平均未达10 dB，且每条记录不保证单调改善。
- 当前主要验证的是这条实测128点路径和合成有色输入；不同长度、稠密、多簇或动态路径需额外验证，不能宣称普遍最优。
