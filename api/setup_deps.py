#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CosyVoice2 依赖安装脚本
检查并安装所有必要的依赖包
"""

import os
import sys
import subprocess
import platform
import time

# 彩色输出
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color):
    """输出彩色文本"""
    print(f"{color}{text}{Colors.ENDC}")

def run_command(command):
    """运行命令并返回结果"""
    print_colored(f"执行命令: {command}", Colors.BLUE)
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def check_and_install_dependencies():
    """检查和安装依赖"""
    print_colored("=== CosyVoice2 依赖检查 ===", Colors.HEADER)
    
    # 检查当前环境
    python_version = platform.python_version()
    print_colored(f"Python版本: {python_version}", Colors.BLUE)
    
    # 安装requirements.txt中的所有依赖
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print_colored(f"错误: 未找到依赖文件 {requirements_path}", Colors.RED)
        return False
    
    print_colored("开始安装基础依赖...", Colors.GREEN)
    code, out, err = run_command(f"{sys.executable} -m pip install -r {requirements_path}")
    
    if code != 0:
        print_colored(f"安装基础依赖失败: {err}", Colors.RED)
        return False
    
    # 特别检查conformer
    print_colored("检查conformer模块...", Colors.BLUE)
    try:
        import conformer
        print_colored("conformer模块已安装", Colors.GREEN)
    except ImportError:
        print_colored("conformer模块未安装，尝试使用pip安装...", Colors.YELLOW)
        code, out, err = run_command(f"{sys.executable} -m pip install git+https://github.com/sooftware/conformer.git")
        
        if code != 0:
            print_colored(f"安装conformer失败: {err}", Colors.RED)
            print_colored("尝试使用备用方法安装...", Colors.YELLOW)
            code, out, err = run_command(f"{sys.executable} -m pip install conformer")
            
            if code != 0:
                print_colored("所有安装方法都失败，建议手动安装conformer", Colors.RED)
                return False
    
    # 特别检查diffusers
    print_colored("检查diffusers模块...", Colors.BLUE)
    try:
        import diffusers
        print_colored("diffusers模块已安装", Colors.GREEN)
    except ImportError:
        print_colored("diffusers模块未安装，尝试使用pip安装...", Colors.YELLOW)
        code, out, err = run_command(f"{sys.executable} -m pip install diffusers")
        
        if code != 0:
            print_colored(f"安装diffusers失败: {err}", Colors.RED)
            return False
            
    # 特别检查hydra-core
    print_colored("检查hydra-core模块...", Colors.BLUE)
    try:
        import hydra
        print_colored("hydra-core模块已安装", Colors.GREEN)
    except ImportError:
        print_colored("hydra-core模块未安装，尝试使用pip安装...", Colors.YELLOW)
        code, out, err = run_command(f"{sys.executable} -m pip install hydra-core")
        
        if code != 0:
            print_colored(f"安装hydra-core失败: {err}", Colors.RED)
            return False
            
    # 特别检查pytorch-lightning
    print_colored("检查pytorch-lightning模块...", Colors.BLUE)
    try:
        import lightning
        print_colored("lightning模块已安装", Colors.GREEN)
    except ImportError:
        print_colored("lightning模块未安装，尝试使用pip安装...", Colors.YELLOW)
        code, out, err = run_command(f"{sys.executable} -m pip install lightning")
        
        if code != 0:
            print_colored(f"安装lightning失败: {err}", Colors.RED)
            print_colored("尝试安装pytorch-lightning替代...", Colors.YELLOW)
            code, out, err = run_command(f"{sys.executable} -m pip install pytorch-lightning")
            if code != 0:
                print_colored(f"安装pytorch-lightning也失败了: {err}", Colors.RED)
                return False
    
    # 检查其他可能缺失的依赖
    problematic_modules = [
        ("inflect", "inflect>=6.0.0"),
        ("hyperpyyaml", "hyperpyyaml>=1.0.0"),
        ("onnxruntime", "onnxruntime"),
        ("modelscope", "modelscope"),
        ("transformers", "transformers>=4.31.0"),
        ("pyyaml", "pyyaml"),
        ("accelerate", "accelerate"),
        ("torchaudio", "torchaudio")
    ]
    
    for module_name, package_name in problematic_modules:
        print_colored(f"检查 {module_name} 模块...", Colors.BLUE)
        try:
            __import__(module_name)
            print_colored(f"{module_name} 模块已安装", Colors.GREEN)
        except ImportError:
            print_colored(f"{module_name} 模块未安装，正在安装...", Colors.YELLOW)
            code, out, err = run_command(f"{sys.executable} -m pip install {package_name}")
            
            if code != 0:
                print_colored(f"安装 {module_name} 失败: {err}", Colors.RED)
                return False
    
    print_colored("=== 所有依赖安装完成 ===", Colors.GREEN)
    return True

if __name__ == "__main__":
    success = check_and_install_dependencies()
    
    if success:
        print_colored("依赖安装成功，现在可以运行 FastAPI 服务了!", Colors.GREEN)
        print_colored("运行以下命令启动服务:", Colors.YELLOW)
        print_colored("./start_fastapi_service.sh", Colors.BLUE)
    else:
        print_colored("依赖安装过程中遇到错误，请查看上面的错误信息", Colors.RED)
        sys.exit(1) 