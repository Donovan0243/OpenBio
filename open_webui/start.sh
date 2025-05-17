#!/usr/bin/env bash

# 设置默认值
PORT="${PORT:-9099}"
HOST="${HOST:-0.0.0.0}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINES_DIR="${PIPELINES_DIR:-$SCRIPT_DIR/pipelines}"

# 安装基本依赖
echo "正在安装基本依赖..."
pip install fastapi uvicorn python-dotenv aiohttp pydantic

# 确保 pipelines 目录存在
if [ ! -d "$PIPELINES_DIR" ]; then
    mkdir -p "$PIPELINES_DIR"
    echo "创建 pipelines 目录: $PIPELINES_DIR"
fi

# 切换到脚本所在目录
cd "$SCRIPT_DIR"
echo "当前工作目录: $(pwd)"

# 设置 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# 启动服务器
echo "启动服务器..."
exec uvicorn main:app --host "$HOST" --port "$PORT" --reload

