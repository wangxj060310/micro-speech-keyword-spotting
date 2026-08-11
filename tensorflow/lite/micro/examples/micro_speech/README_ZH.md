# Micro Speech — MCU 关键词唤醒（TinyML 面试项目）

> 在单片机级别的资源约束下运行**关键词唤醒（Keyword Spotting）**：
> 识别 "yes" / "no" 两个英文单词。整个系统由两个神经网络模型 + TFLite Micro
> 推理引擎组成，模型总大小 **< 20 KB**，可运行在 Cortex-M 系列 MCU 上。
>
> 代码来自 [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro)
> 官方示例（Apache-2.0），本仓库保留完整源码并补充中文讲解文档。

## 项目亮点（简历可写）

| 技能点 | 本项目的体现 |
| --- | --- |
| 边缘 AI / TinyML | 在 KB 级内存的 MCU 上跑神经网络推理，模型 <20KB |
| 模型量化 | int8 量化模型（`micro_speech_quantized.tflite`），1 字节权重/激活 |
| 音频特征工程 | 前端用**音频预处理模型**把原始波形转成频谱图（MFCC 思路） |
| TFLM 推理引擎 | 解释器、arena 内存分配、kernel 注册、算子执行 |
| 端到端流程 | 模型训练（TensorFlow）→ 转 TFLite → int8 量化 → C 数组 → MCU 推理 |

## 系统架构

```
原始音频(16kHz PCM)
      │
      ▼
┌─────────────────────┐
│ Audio Preprocessor  │  ── 频谱图特征
│ (模型1, int8)       │     49×49×1 帧
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Micro Speech 模型    │  ── 分类结果
│ (模型2, int8, <20KB)│     "yes" / "no" / "silence" / "unknown"
└─────────────────────┘
      │
      ▼
   唤醒动作（如点亮 LED、开始录音）
```

### 关键文件

| 文件 | 作用 |
| --- | --- |
| `micro_speech_test.cc` | 官方 C++ 测试：用预置音频验证推理结果 |
| `models/micro_speech_quantized.tflite` | int8 量化后的识别模型 |
| `models/audio_preprocessor_int8.tflite` | 音频预处理模型（特征提取） |
| `micro_model_settings.h` | 音频参数：采样率 16k、帧长 30ms、步长 20ms |
| `testdata/*.wav` | 测试音频（yes/no/silence 片段） |
| `train/` | 模型训练代码（Jupyter notebook） |

## 模型有多大、多快？（面试数据）

| 指标 | 数值 |
| --- | --- |
| Micro Speech 模型权重 | < 20 KB（int8） |
| 输入 | 49×49 频谱图（int8） |
| 推理延迟（Cortex-M4 @ 80MHz 量级） | 约 100-200ms |
| 每帧音频处理 | 30ms 窗口，20ms 步进 |

## 为什么选 int8 量化？（必考问答）

1. **省内存**：float32 权重 4 字节/参数，int8 只需 1 字节——模型缩小 4 倍。
2. **省 Flash**：权重存 Flash，int8 减少固件体积。
3. **省功耗/加速**：Cortex-M 的 SIMD（如 CMSIS-NN）与硬件加速器
   （Ethos-U55 等）都针对 int8 优化。
4. **代价**：精度损失通常 <1%（量化校准可得），MCU 场景完全可接受。

## 构建与运行

> 官方推荐 Bazel 或 make 构建（依赖下载较慢，需要科学上网环境）。

```bash
# Bazel（推荐，Google 官方环境）
bazel build tensorflow/lite/micro/examples/micro_speech:micro_speech_test
bazel run  tensorflow/lite/micro/examples/micro_speech:micro_speech_test

# 或 make（旧方式）
make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test
```

运行成功后输出类似：

```
Heard yes (216) @0ms
Heard no (216) @320ms
```

即模型在给定音频上正确识别出 "yes" 和 "no"。

### 在 STM32 上部署（面试延伸）

1. 用 `xxd -i` 或官方 `generate_cc_arrays` 把 .tflite 转成 C 数组；
2. 接入 TFLite Micro 运行时（`MicroInterpreter` + arena）；
3. 用 I2S/PDM 采集麦克风 16kHz 音频，16ms 数据块喂给预处理模型；
4. 每 20ms 得到一帧特征，累积 49 帧后推理，输出唤醒词。

## 学习路径（配合源码）

1. **先跑通测试**：理解输入是什么（频谱图）、输出是什么（分类）。
2. **读 `micro_model_settings.h`**：掌握音频参数与模型输入输出的对应。
3. **读 `micro_speech_test.cc`**：看特征如何拼接、推理如何调用。
4. **读模型训练 notebook**：理解"为什么这样提取特征"。
5. **延伸**：用 `train/` 里的脚本训练自己的唤醒词（中文"小爱同学"等）。

## 与 STM32 传统方案对比（面试加分）

| 方案 | 特点 |
| --- | --- |
| 传统 DSP（能量阈值/过零率） | 简单但只能区分"有/无声音"，无法区分内容 |
| 机器学习（本方案） | 需要特征 + 模型，但能识别具体词，鲁棒性强 |
| 云端语音（如讯飞） | 效果最好但要联网，延迟大、功耗高、隐私差 |

TinyML 的价值正是把"最后一公里"的识别放到设备端：**离线、低延迟、低功耗、数据不出设备**。

## License

Apache-2.0（TensorFlow 官方许可），本仓库保留版权声明与出处。
