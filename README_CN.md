# VoiceClone - AI 声音转换管道

一个通用的声音转换管道，使用 AI（RVC - 基于检索的声音转换）将任意歌曲中的人声替换为您自己的声音。

## 概述

该管道执行以下步骤：
1. **下载** - 从 YouTube 下载歌曲
2. **分离** - 使用 Demucs 分离人声和伴奏
3. **训练** - 在您的声音样本上训练 RVC 模型
4. **转换** - 将原始人声转换为您的音色
5. **混音** - 将转换后的人声与原始伴奏混合

## 示例用例

例如，您可以使用此管道用自己的声音演唱汉密尔顿的《My Shot》：
- 从 YouTube 下载《My Shot》
- 使用 10-15 分钟的您说话的声音进行训练
- 转换并混音以创建您自己的版本

## 目录结构

```
voiceclone/
├── input/                    # 在此处放置输入文件
│   ├── song_original.mp3     # 下载的歌曲（自动生成）
│   ├── song_original.wav     # 转换的 WAV 文件（自动生成）
│   └── my_voice.m4a          # 您的声音录音（放置在此处）
├── intermediate/             # 临时处理文件
│   ├── vocals.wav            # 提取的人声（自动生成）
│   ├── accompaniment.wav     # 伴奏音轨（自动生成）
│   └── sliced/               # 用于训练的切片声音片段
├── output/                   # 最终输出文件
│   ├── song_my_voice.mp3     # 最终混音歌曲（MP3）
│   └── song_my_voice.wav     # 最终混音歌曲（WAV）
├── exp/                      # 训练实验（自动创建）
│   └── YYYYMMDD_HHMMSS/      # 带时间戳的训练运行
│       ├── my_voice.pth      # 训练好的模型
│       ├── my_voice.index    # 特征索引
│       └── vocals_converted.wav  # 转换后的人声
└── rvc_workspace/            # RVC 仓库和模型
    └── Retrieval-based-Voice-Conversion-WebUI/
```

## 前置要求

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

必需的包包括：
- `torch`, `torchaudio` - 深度学习框架
- `demucs` - 人声分离
- `rvc-python` - 声音转换（可选，回退到 RVC 仓库）
- `pydub`, `soundfile`, `librosa` - 音频处理
- `yt-dlp` - YouTube 下载
- `faiss-cpu` 或 `faiss-gpu` - 特征索引
- `scikit-learn` - K-means 聚类

### 2. 下载 RVC 仓库（手动步骤）

由于网络限制，您需要手动下载 RVC 仓库：

```bash
# 在有网络访问的机器上：
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git

# 复制到项目：
# 将文件夹放置在：./rvc_workspace/Retrieval-based-Voice-Conversion-WebUI/
```

### 3. 下载 RVC 模型

```bash
cd rvc_workspace/Retrieval-based-Voice-Conversion-WebUI
python tools/download_models.py
```

## 步骤 1：准备您的声音录音

### 文件放置

将您的声音录音放置在：
```
input/my_voice.m4a
```

或修改 `config.py` 指向您的文件：
```python
USER_VOICE_FILE = INPUT_DIR / "your_voice.wav"
```

### 录音要求

**时长：**
- **最低：** 10 分钟清晰语音
- **推荐：** 10-15 分钟以获得最佳质量
- **越多越好：** 最长 30 分钟可以改善效果

**内容指南：**
- 用多变的语调、情感和节奏说话
- 充满活力地朗读诗歌、新闻文章或故事
- 涵盖不同的说话风格：耳语、正常说话、热情洋溢的说话
- **您不需要唱歌或说唱** - RVC 会处理从语音到歌唱的转换

**技术质量：**
- **格式：** M4A、WAV 或 MP3（首选 44.1 kHz）
- **环境：** 安静的房间，尽量减少背景噪音
- **麦克风：** 靠近麦克风（6-12 英寸），如有条件请使用防喷罩
- **混响：** 最小化房间混响（在挂满衣服的衣柜中录音效果最佳）
- **削波：** 避免音频削波/失真

**专业提示：**
- 在一次连续会话中录音以保持一致性
- 可以休息但保持麦克风位置不变
- 阅读您熟悉的材料以听起来自然
- 包含一些笑声和情感表达

## 步骤 2：运行管道

### 完整管道（所有步骤）

```bash
python pipeline.py --all
```

### 单独步骤

```bash
# 步骤 1：下载歌曲
python pipeline.py --download

# 步骤 2：分离人声和伴奏
python pipeline.py --separate

# 步骤 3：训练声音模型
python pipeline.py --train

# 步骤 4：将人声转换为您的声音
python pipeline.py --convert

# 步骤 5：混音最终输出
python pipeline.py --mix
```

### 快速调试模式

用于以最短训练时间测试管道：

```bash
python pipeline.py --all --quick
```

这仅使用 2 个训练轮次而非 200 个，完成速度更快但质量较低。

### 恢复训练

如果训练被中断，您可以从检查点恢复：

```bash
python pipeline.py --train --continue-from ./exp/20260328_143000
```

### 使用特定实验

要使用特定的训练模型进行转换/混音：

```bash
python pipeline.py --convert --ckpt ./exp/20260328_143000
python pipeline.py --mix --ckpt ./exp/20260328_143000
```

## 配置

编辑 `config.py` 以自定义管道：

### 声音转换后端

```python
# 使用 RVC 进行声音转换（默认）
VOICE_CONVERSION_BACKEND = "rvc"

# 或跳过声音转换（按原样复制人声）
VOICE_CONVERSION_BACKEND = "passthrough"
```

### RVC 训练参数

```python
RVC_SAMPLE_RATE = 40000      # 40000 (v1) 或 48000 (v2)
RVC_TRAINING_EPOCHS = 200    # 10 分钟数据推荐 200-300
RVC_BATCH_SIZE = 32          # 根据您的 GPU 内存调整
RVC_F0_METHOD = "rmvpe"      # 音高提取方法
```

### 输出设置

```python
VOCAL_VOLUME_ADJUST_DB = -2.0     # 调整人声音量
ACCOMPANIMENT_VOLUME_ADJUST_DB = 0.0  # 调整伴奏音量
OUTPUT_MP3_BITRATE = "320k"       # 输出 MP3 质量
```

### GPU 配置

```python
# 使用特定 GPU（用于多 GPU 系统）
RVC_CUDA_DEVICE = "0"  # 使用 GPU 0
```

## 输出文件

成功完成后，您将找到：

| 文件 | 位置 | 描述 |
|------|------|------|
| 最终 MP3 | `output/song_my_voice.mp3` | MP3 格式的最终歌曲（320kbps） |
| 最终 WAV | `output/song_my_voice.wav` | 未压缩 WAV 格式的最终歌曲 |
| 模型 | `exp/YYYYMMDD_HHMMSS/my_voice.pth` | 训练好的 RVC 模型（可重用） |
| 索引 | `exp/YYYYMMDD_HHMMSS/my_voice.index` | 模型的特征索引 |
| 转换后的人声 | `exp/YYYYMMDD_HHMMSS/vocals_converted.wav` | 您的声音演唱的歌曲 |

## 故障排除

### YouTube 下载问题

如果 YouTube 下载因身份验证失败：

```bash
# 使用浏览器 cookie
python pipeline.py --download --cookies-from-browser chrome

# 或使用 cookie 文件
python pipeline.py --download --cookies-file /path/to/cookies.txt
```

### CUDA 内存不足

在 `config.py` 中减小批量大小：
```python
RVC_BATCH_SIZE = 16  # 对于内存较小的 GPU 甚至可以使用 8
```

### 无 GPU 可用

管道将回退到 CPU，但训练会非常慢。建议：
- 使用 `--quick` 模式进行测试
- 使用云 GPU 服务
- 如果有预训练模型则使用预训练模型

### 音频质量问题

- **机器音：** 增加训练轮次（300+）
- **背景噪音：** 改善录音环境
- **音高问题：** 在 config.py 中调整 `RVC_TRANSPOSE`
- **闷声：** 检查录音质量，确保无削波

## 高级用法

### 使用不同的歌曲

要转换不同的歌曲，在 `config.py` 中修改：

```python
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# 或使用搜索：
YOUTUBE_SEARCH_QUERY = "Your Song Name Artist"
```

您也可以更改输出文件名：

```python
OUTPUT_MP3 = OUTPUT_DIR / "your_song_your_voice.mp3"
OUTPUT_WAV = OUTPUT_DIR / "your_song_your_voice.wav"
```

### 批量处理

您可以训练一次并转换多首歌曲：

```bash
# 训练一次
python pipeline.py --train

# 对于每首新歌：
python pipeline.py --download --separate
python pipeline.py --convert --ckpt ./exp/YOUR_TIMESTAMP
python pipeline.py --mix --ckpt ./exp/YOUR_TIMESTAMP
```

## 许可证

本项目使用 RVC，采用 MIT 许可证。详情请参阅 RVC 仓库。
