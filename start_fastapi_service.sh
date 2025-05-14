#!/bin/bash
# 在 conda 环境中启动 CosyVoice2 FastAPI 服务

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 默认配置
CONDA_ENV="cosyvoice"
API_PORT=9880
PRELOAD_MODEL=true
LOG_LEVEL="info"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --port)
      API_PORT="$2"
      shift 2
      ;;
    --no-preload)
      PRELOAD_MODEL=false
      shift
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [--env 环境名] [--port 端口号] [--no-preload] [--log-level 日志级别]"
      echo ""
      echo "选项:"
      echo "  --env <环境名>     指定 conda 环境名 (默认: cosyvoice)"
      echo "  --port <端口号>    指定 API 端口 (默认: 9880)"
      echo "  --no-preload       禁用模型预加载 (提高启动速度，但首次请求较慢)"
      echo "  --log-level <级别> 指定日志级别 (debug, info, warning, error) (默认: info)"
      echo "  --help             显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--env 环境名] [--port 端口号] [--no-preload] [--log-level 日志级别] [--help]"
      exit 1
      ;;
  esac
done

# 显示启动信息
echo "========================================"
echo "CosyVoice2-Ex FastAPI 服务启动器 (Conda)"
echo "========================================"
echo "conda 环境: $CONDA_ENV"
echo "API 端口: $API_PORT"
echo "模型预加载: $PRELOAD_MODEL"
echo "日志级别: $LOG_LEVEL"
echo "服务地址: http://localhost:$API_PORT"
echo "API 文档: http://localhost:$API_PORT/docs"
echo "========================================"

# 检查 conda 是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令。请确保已安装 Anaconda 或 Miniconda。"
    exit 1
fi

# 设置环境变量
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH
export COSYVOICE_API_PORT=$API_PORT
if [ "$PRELOAD_MODEL" = "true" ]; then
    export PRELOAD_MODEL=1
else
    export PRELOAD_MODEL=0
fi
export COSYVOICE_DEBUG=1  # 启用调试模式

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p logs
mkdir -p voices
mkdir -p output
mkdir -p audios

# 检查 conda 环境是否存在
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "错误: conda 环境 '$CONDA_ENV' 不存在"
    echo "是否要创建此环境? (y/n)"
    read -r create_env
    if [[ $create_env == "y" || $create_env == "Y" ]]; then
        echo "创建 conda 环境: $CONDA_ENV"
        conda create -n $CONDA_ENV python=3.10 -y
        if [ $? -ne 0 ]; then
            echo "创建环境失败，请检查 conda 安装"
            exit 1
        fi
    else
        echo "退出安装"
        exit 1
    fi
fi

# 在 conda 环境中安装依赖
echo "在 conda 环境 '$CONDA_ENV' 中安装依赖..."
conda run -n $CONDA_ENV pip install -r requirements-api.txt

if [ $? -ne 0 ]; then
    echo "警告: 安装基本依赖可能不完整，将尝试继续..."
fi

# 运行依赖安装脚本
echo "运行依赖安装脚本..."
conda run -n $CONDA_ENV python api/setup_deps.py

if [ $? -ne 0 ]; then
    echo "警告: 依赖安装脚本可能未完全成功，将尝试继续..."
fi

# 尝试针对特定系统进行额外安装
echo "检查系统特定依赖..."
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    # Apple Silicon Mac
    echo "检测到 Apple Silicon Mac，安装兼容依赖..."
    conda run -n $CONDA_ENV pip install onnxruntime-silicon
elif conda run -n $CONDA_ENV python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    # CUDA 可用
    echo "检测到 CUDA 环境，安装 GPU 加速依赖..."
    conda run -n $CONDA_ENV pip install onnxruntime-gpu
fi

# 最终依赖检查
echo "执行最终依赖检查..."
conda run -n $CONDA_ENV python -c "
try:
    import fastapi, uvicorn, torch, numpy, transformers, modelscope, onnxruntime
    print('基本依赖检查通过!')
except ImportError as e:
    print(f'错误: 仍然缺少依赖: {e}')
    print('请手动安装缺失的依赖后重试')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "依赖检查失败，无法启动服务"
    echo "请查看上述错误信息，手动安装缺失的依赖后重试"
    exit 1
fi

# 配置 API 服务
echo "配置 API 服务..."
# 创建临时配置脚本
TMP_CONFIG=$(mktemp)
cat > $TMP_CONFIG << EOF
import sys, os

# 配置文件路径
config_path = 'api/config.py'

# 更新配置
with open(config_path, 'r') as f:
    config = f.read()

# 更新端口
config = config.replace("PORT = 9880", f"PORT = {$API_PORT}")

# 更新预加载设置
if ${PRELOAD_MODEL}:
    if "PRELOAD_MODEL = False" in config:
        config = config.replace("PRELOAD_MODEL = False", "PRELOAD_MODEL = True")
    elif not "PRELOAD_MODEL" in config:
        config += "\n# 模型预加载设置\nPRELOAD_MODEL = True\n"
else:
    if "PRELOAD_MODEL = True" in config:
        config = config.replace("PRELOAD_MODEL = True", "PRELOAD_MODEL = False")
    elif not "PRELOAD_MODEL" in config:
        config += "\n# 模型预加载设置\nPRELOAD_MODEL = False\n"

# 确保调试模式开启
if not "DEBUG = True" in config:
    config += "\n# 调试模式\nDEBUG = True\n"

# 写回配置
with open(config_path, 'w') as f:
    f.write(config)

print("API 配置已更新")
EOF

# 执行配置脚本
conda run -n $CONDA_ENV python $TMP_CONFIG
rm $TMP_CONFIG

# 创建日志目录
LOG_FILE="logs/api_$(date +"%Y%m%d_%H%M%S").log"
echo "日志将保存到: $LOG_FILE"

# 启动 FastAPI 服务 - 使用run-api.py直接启动
echo "启动 FastAPI 服务..."
conda run -n $CONDA_ENV python -c "
import os
import sys
import logging
import uvicorn

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('$LOG_FILE')
    ]
)

logger = logging.getLogger('cosyvoice_api')

# 设置环境变量
os.environ['PYTHONPATH'] = '$SCRIPT_DIR:' + os.environ.get('PYTHONPATH', '')
os.environ['COSYVOICE_API_PORT'] = '$API_PORT'
os.environ['COSYVOICE_DEBUG'] = '1'

# 启动 uvicorn 服务器
logger.info('启动 FastAPI 服务')

try:
    # 直接使用 uvicorn 启动，不使用reload模式，以便日志直接显示在当前终端
    uvicorn.run(
        'api.main:app',
        host='0.0.0.0',
        port=$API_PORT,
        log_level='$LOG_LEVEL'
    )
except Exception as e:
    logger.error(f'启动服务出错: {str(e)}')
    sys.exit(1)
"

echo "服务已停止。查看日志文件获取详细信息: $LOG_FILE" 