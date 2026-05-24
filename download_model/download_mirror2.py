#!/usr/bin/env python3
"""
使用镜像站下载 GPTQ 模型 - 改进版
"""
import os
import sys
import urllib.request
from pathlib import Path
import time

sys.stdout.reconfigure(line_buffering=True)

MIRROR_BASE = "https://hf-mirror.com"
PROJECT_DIR = Path("/data2/user/quyiyang/gptq")
os.chdir(PROJECT_DIR)

print("=" * 50)
print("   使用镜像站下载模型 v2")
print(f"   开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

models_dir = PROJECT_DIR / "models"
models_dir.mkdir(exist_ok=True)

MODELS = [
    ("facebook/opt-350m", "models/opt-350m"),
    ("facebook/opt-1.3b", "models/opt-1.3b"),
    ("facebook/opt-2.7b", "models/opt-2.7b"),
    ("facebook/opt-6.7b", "models/opt-6.7b"),
    ("facebook/opt-13b", "models/opt-13b"),
    ("facebook/opt-66b", "models/opt-66b"),
    ("bigscience/bloom-560m", "models/bloom-560m"),
    ("bigscience/bloom-1b1", "models/bloom-1b1"),
    ("bigscience/bloom-1b7", "models/bloom-1b7"),
    ("bigscience/bloom-3b", "models/bloom-3b"),
    ("bigscience/bloom-7b1", "models/bloom-7b1"),
]

REQUIRED_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]

def download_file(url, dest, timeout=1200):
    """下载单个文件"""
    if dest.exists():
        print(f"  ✅ {dest.name} 已存在", flush=True)
        return True

    print(f"  📥 {dest.name}...", flush=True)
    try:
        # 创建请求，添加 User-Agent
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"  ✅ {dest.name} 完成 ({dest.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"  ❌ {dest.name} 失败: {e}", flush=True)
        if dest.exists():
            dest.unlink()  # 删除不完整的文件
        return False

def download_model(model_id, local_dir):
    """下载一个模型"""
    target_path = Path(local_dir)
    if target_path.exists() and (target_path / "config.json").exists():
        print(f"✅ {model_id} 已存在", flush=True)
        return True

    print(f"📥 {model_id} -> {local_dir}", flush=True)
    target_path.mkdir(exist_ok=True)

    success_count = 0
    for filename in REQUIRED_FILES:
        mirror_url = f"{MIRROR_BASE}/{model_id}/resolve/main/{filename}"
        dest_file = target_path / filename

        if download_file(mirror_url, dest_file):
            success_count += 1

    return success_count > 0

# 开始下载
print(f"\n下载 {len(MODELS)} 个模型...\n", flush=True)

completed = 0
for i, (model_id, local_dir) in enumerate(MODELS, 1):
    print(f"[{i}/{len(MODELS)}] {model_id}", flush=True)
    if download_model(model_id, local_dir):
        completed += 1
    print("", flush=True)

print(f"\n完成: {completed}/{len(MODELS)} 个模型", flush=True)

# 保存完成标志
with open(PROJECT_DIR / "download_complete.log", "w") as f:
    f.write(f"下载完成于: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"完成: {completed}/{len(MODELS)} 个模型\n")

print("\n" + "=" * 50, flush=True)
print("   下载完成!", flush=True)
print(f"   完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 50, flush=True)
