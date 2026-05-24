#!/usr/bin/env python3
"""
使用镜像站下载 Meta-Llama-3-8B
"""
import os
import sys
import urllib.request
import json
from pathlib import Path
import time

sys.stdout.reconfigure(line_buffering=True)

MIRROR_BASE = "https://hf-mirror.com"
MODEL_ID = "NousResearch/Meta-Llama-3-8B"
LOCAL_DIR = Path("/data2/user/quyiyang/gptq/models/llama-3-8b")  # python llama.py models/llama-3-8b c4 --wbits 3

os.chdir(LOCAL_DIR.parent)

print("=" * 50)
print("   下载 Meta-Llama-3-8B")
print(f"   开始时间: {time.strftime('%Y-%m-%d-%H:%M:%S')}")
print("=" * 50)


def get_model_files(model_id):
    """从镜像站获取模型文件列表"""
    api_url = f"{MIRROR_BASE}/api/models/{model_id}"
    print(f"获取文件列表: {api_url}", flush=True)
    try:
        req = urllib.request.Request(
            api_url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            siblings = data.get("siblings", [])
            files = [s["rfilename"] for s in siblings]
            print(f"共 {len(files)} 个文件", flush=True)
            for f in files:
                print(f"  {f}", flush=True)
            return files
    except Exception as e:
        print(f"获取文件列表失败: {e}", flush=True)
        print("将使用默认文件列表", flush=True)
        return None


def download_file(url, dest, timeout=1800, max_retries=5):
    """下载单个文件，带大小校验和重试"""
    for attempt in range(1, max_retries + 1):
        if dest.exists():
            print(f"  [skip] {dest.name} 已存在 ({dest.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
            return True

        if attempt > 1:
            wait = attempt * 5
            print(f"  [retry {attempt}/{max_retries}] {dest.name}，等待 {wait}s...", flush=True)
            time.sleep(wait)

        print(f"  [down] {dest.name}...", flush=True)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total = response.headers.get('Content-Length')
                if total:
                    total = int(total)
                    print(f"    预期大小: {total / 1024 / 1024:.1f} MB", flush=True)

                downloaded = 0
                last_report = 0
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if total and (now - last_report) > 30:
                            pct = downloaded / total * 100
                            print(f"    进度: {pct:.1f}% ({downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB)", flush=True)
                            last_report = now

            actual_size = dest.stat().st_size
            if total and actual_size < total:
                print(f"  [warn] {dest.name} 不完整: {actual_size / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB，将重试", flush=True)
                dest.unlink()
                continue

            print(f"  [done] {dest.name} 完成 ({actual_size / 1024 / 1024:.1f} MB)", flush=True)
            return True
        except Exception as e:
            print(f"  [fail] {dest.name} 失败: {e}", flush=True)
            if dest.exists():
                dest.unlink()

    print(f"  [abort] {dest.name} 重试 {max_retries} 次后仍失败", flush=True)
    return False


# 尝试自动获取文件列表
files = get_model_files(MODEL_ID)

if files is None:
    # Llama-2-7b-hf 默认文件列表
    files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]

# 过滤：跳过隐藏文件和 pytorch_model.bin（safetensors 格式已足够）
download_files = [f for f in files if not f.startswith(".") and "pytorch_model" not in f]

print(f"\n需要下载 {len(download_files)} 个文件到 {LOCAL_DIR}\n", flush=True)

LOCAL_DIR.mkdir(exist_ok=True)

success = 0
failed = []
for filename in download_files:
    url = f"{MIRROR_BASE}/{MODEL_ID}/resolve/main/{filename}"
    dest = LOCAL_DIR / filename

    # 如果文件在子目录中，创建子目录
    dest.parent.mkdir(parents=True, exist_ok=True)

    if download_file(url, dest):
        success += 1
    else:
        failed.append(filename)
    print("", flush=True)

print("\n" + "=" * 50)
print(f"   完成: {success}/{len(download_files)} 个文件")
if failed:
    print(f"   失败: {', '.join(failed)}")
print(f"   结束时间: {time.strftime('%Y-%m-%d-%H:%M:%S')}")
print("=" * 50)
