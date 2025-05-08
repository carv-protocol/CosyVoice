import os
import subprocess
from pydub import AudioSegment
from pathlib import Path

# ✅ 设置你的目标
VIDEO_URL = "https://www.bilibili.com/video/BV11V411X7Fn"  # 或B站视频链接
OUTPUT_DIR = "loli_dataset"
SEGMENT_LENGTH = 10000  # 每段音频时长（毫秒）
OVERLAP = 500           # 重叠部分防止断句（毫秒）

# ✅ 创建文件夹结构
audio_dir = Path(OUTPUT_DIR) / "wavs"
text_dir = Path(OUTPUT_DIR) / "txts"
audio_dir.mkdir(parents=True, exist_ok=True)
text_dir.mkdir(parents=True, exist_ok=True)

# ✅ 下载视频并转为音频
print("⬇️ Downloading audio...")
subprocess.run([
    "yt-dlp",
    "-x", "--audio-format", "wav",
    "-o", "temp.%(ext)s",
    VIDEO_URL
])

# ✅ 音频切割
print("✂️ Splitting audio...")
audio = AudioSegment.from_wav("temp.wav")
length = len(audio)

i = 0
start = 0
while start < length:
    end = min(start + SEGMENT_LENGTH, length)
    segment = audio[start:end]
    filename = f"{i:04d}.wav"
    segment.export(audio_dir / filename, format="wav")
    with open(text_dir / filename.replace(".wav", ".txt"), "w", encoding="utf-8") as f:
        f.write("")  # 你可以之后填上文字或用 Whisper 自动转写
    i += 1
    start += SEGMENT_LENGTH - OVERLAP

# ✅ 清理
os.remove("temp.wav")
print(f"✅ Done! {i} segments saved to {OUTPUT_DIR}/wavs")

print("\n📦 目录结构示例：")
print(f"{OUTPUT_DIR}/wavs/0000.wav")
print(f"{OUTPUT_DIR}/txts/0000.txt")