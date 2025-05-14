#!/bin/bash
# CosyVoice2 FastAPI 服务启动脚本 (Conda)

# 默认使用 cosyvoice 环境
CONDA_ENV="cosyvoice"

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

# 调用完整的脚本
./start_fastapi_service.sh --env $CONDA_ENV

# 脚本结束后的提示
echo ""
echo "如需更多选项，请使用 ./start_fastapi_service.sh --help 查看帮助" 