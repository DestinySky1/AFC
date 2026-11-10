# 可验证的闭环声反馈消除（Probe-IV AFC）

这是一个**独立的新实现**。它不覆盖原始 openMHA、NLMS、LMS 或任何旧实验。
目标是在已测得的128点反馈路径上，处理原先使普通NLMS明显失配的**有色目标信号**。

## 结论与方法

普通 LMS/NLMS 可以放在闭环中使用；但当扬声器输出与麦克风目标声音相关时，用整个扬声器信号直接做 LMS/NLMS 参考，反馈路径估计可能有偏。

本项目采用**探测信号辅助的工具变量闭环辨识**（Probe-IV），而不声称这是论文中某个NLMS公式的逐式复现。

1. 辨识时，扬声器输出为 `u[n] = gain * e[n-D] + p[n]`，其中 `p` 是设备生成、与外界目标声音独立的有界二值探测信号。
2. 麦克风为 `m[n] = s[n] + h^T x[n]`，`x[n]=[u[n-1],...,u[n-L]]`。这保留了原 openMHA 基线中的**额外一个扬声器缓冲区延迟**。
3. 以 `z[n]=[p[n-1],...,p[n-L]]` 为工具变量，用 `E[z x^T] * h = E[z m]` 估计反馈路径。此关系在闭环中成立，条件是探测信号与外界目标声音独立、路径在辨识窗内近似不变、且相关矩阵可逆。
4. 使用两半数据之间的系数差估计噪声，并通过近似 SURE 软阈值处理稀疏反馈路径，减少微弱真实路径的估计噪声。该步骤适合本实测路径；**不保证对所有稠密路径都有效**。
5. 辨识完成后，关闭探测信号；此时每个麦克风样本实时执行 `e[n]=m[n]-w^T x[n]`。若路径发生变化，可临时重新启用探测信号并重新辨识。

与论文相关但并非逐式复现的参考：

- Shusina & Rafaely, *Unbiased Adaptive Feedback Cancellation in Hearing Aids by Closed-Loop Identification*, IEEE TASLP, 2006。
- Spriet et al., *An Instrumental Variable Method for Adaptive Feedback Cancellation in Hearing Aids*, ICASSP, 2005。
- *Acoustic feedback cancellation in hearing aids using dual adaptive filtering and gain-controlled probe signal*, Biomedical Signal Processing and Control, 2019。

## 已执行的闭环仿真（真实128点路径）

系统：16 kHz，D=1，gain=0 dB，原始MATLAB导出的10秒高通有色输入（第1次），其余独立高通随机输入。Python中仅用真实路径 `h` **生成仿真麦克风数据和计算验收分数**，不让估计器读取 `h` 或 `s`。新生成的有色输入使用 SciPy FIR（对应 `fir1(64,.1,'high')` 的设计），不声称跨语言随机数组逐位相同。

| 辨识时长 | 探测信号峰值及RMS | 独立种子 | 平均NFCE | 平均ASG | 最低ASG | 独立目标的平均反馈抑制 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10秒 | 0.10（有界二值） | 10 | 0.3297 | +5.7216 dB | +3.3284 dB | 5.1823 dB |
| 60秒 | 0.05（有界二值） | 10 | 0.3004 | +5.8871 dB | +4.3559 dB | 5.3488 dB |

两组均为10/10次NFCE<1且ASG>0。原有标准NLMS基线在10秒、有色输入、μ=.01时NFCE≈598、ASG≈-29 dB。注意这不是相同的辅助信号条件；新方案通过额外探测信号获得可辨识性。

**ASG的定义**：`MSG - MSG0`，其中 `MSG=-20log10(max|FFT(h-w)|)`，`MSG0=-20log10(max|FFT(h)|)`，FFT16384，仅是频域幅值网格指标，不等价于实际硬件的严格稳定增益。`NFCE=||h-w||²/||h||²`。

探测信号可能被听到：10秒方案RMS=0.10，约等于目标输入RMS=0.094；60秒方案RMS=0.05，约比目标低5.5 dB。**未进行听感测试、扬声器幅度标定或实际硬件测试。**

## 文件

- `afc.py`：工具变量估计、SURE稀疏去噪、因果逐样本闭环处理器 `ProbeAFC`、仿真专用反馈路径类。
- `simulation.py`：与原始缓冲区/延迟约定一致的快速闭环验证器。
- `demo.py`：多随机种子实验、独立目标测试、CSV/JSON/PNG结果。
- `stream_example.py`：端到端逐样本闭环运行，先辨识，再停止探测并继续实时抵消。
- `run_probe_iv_afc.m`：相同思路的MATLAB单文件版本（此环境无MATLAB，**未运行验证**）。
- `test_afc.py`：样本级闭环等价、无探测拒绝、真实路径、路径变化后重新辨识等测试。
- `data/afc_inputs_10s.mat`：用户原先提供的MATLAB 10秒输入与真实128点路径，未修改。
- `results_binary_010/`、`results_binary_005_60s/`：实际运行的十种子结果。

## 运行

Python 3.10+：

```bash
pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -v
python stream_example.py
python demo.py --probe-kind binary --probe-rms 0.1 --calibration-seconds 10 --trials 10 --output results_binary_010
python demo.py --probe-kind binary --probe-rms 0.05 --calibration-seconds 60 --trials 10 --output results_binary_005_60s
```

MATLAB（需Signal Processing Toolbox）：

```matlab
cd('此文件夹')
r = run_probe_iv_afc('data/afc_inputs_10s.mat',0.1);
```

MATLAB的随机流与Python不同，不能逐位比对；该MATLAB脚本尚待你本机验证。

## 与实时麦克风／扬声器程序对接

以下说明接口，不表示硬件已经通过验证：

```python
from afc import ProbeAFC

afc = ProbeAFC(taps=128, delay=1, gain=1.0, probe_rms=0.1)

# 每到一个麦克风采样：
error, speaker_sample, injected_probe = afc.process_mic(microphone_sample)
# 将 speaker_sample 送入扬声器驱动；真实声学路径将其反馈到之后的麦克风采样。

# 在收集至少10秒且输入条件合适后，由非实时线程调用：
estimate = afc.identify(last_samples=160000)
afc.activate(estimate, fade_samples=1600)   # 100毫秒平滑切换系数（16 kHz）
afc.probe_rms = 0.0   # 关闭额外探测信号，继续运行逐样本反馈抵消
afc.clear_recorded_audio()  # 若不需要持续记录，释放校准历史
```

如果声学路径变化，重新打开探测信号，收集新的数据并再次 `identify/activate`。不建议在音频实时线程里求解矩阵。

## 必须知道的限制

- 此成果是**使用真实测量FIR作为被控对象的闭环仿真成功**，并非已在助听器或扬声器上完成物理实测。
- 尚缺原始路径的采样率/绝对延迟/声电增益/单位及设备I/O标定。真实音频的块处理延迟不能简单沿用仿真D=1。
- 额外探测信号会影响用户听感。上线前需进行增益约束、峰值限制、听感实验、硬件互锁与路径漂移测试。
- 工具变量估计假定探测信号与目标独立，且校准窗口中路径变化不大；声源主动响应探测信号、非线性剪切和设备时钟漂移均会破坏假设。
- 当前路径较弱并呈稀疏特征，所以噪声收缩很有效；更长、更强或稠密路径需要另行测试及调整。对所有闭环系统或所有步长均不提供无条件收敛保证。
- 实时类提供逐样本算法接口，不等于已交付具备低延迟调度、硬件驱动及声学安全保障的音频设备固件。
