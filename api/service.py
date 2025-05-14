#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CosyVoice2 API 服务模块
包含TTS服务的核心功能实现
"""

import io
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# os.environ['TORIO_FFMPEG_BINARY'] = 'ffmpeg'  # 使用系统安装的ffmpeg
# os.environ['TORIO_USE_FFMPEG'] = '0'  # 禁用torio内置的FFmpeg加载机制
# os.environ['TORIO_NO_FFMPEG'] = '1'  # 防止重复尝试加载失败的库

# # 在macOS上配置动态库路径
# if sys.platform == 'darwin':
#     if 'DYLD_LIBRARY_PATH' not in os.environ:
#         if os.path.exists('/opt/homebrew/lib'):  # Apple Silicon
#             os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'
#             logger.info("设置Apple Silicon库路径: /opt/homebrew/lib")
#         else:  # Intel Mac
#             os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/lib'
#             logger.info("设置Intel Mac库路径: /usr/local/lib")
#     else:
#         logger.info(f"使用已设置的库路径: {os.environ['DYLD_LIBRARY_PATH']}")

import torch
import torchaudio
from typing import Dict, List, Generator, Optional, Union, Any

# 确保第三方库可以导入
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'third_party/AcademiCodec'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

# 导入相关依赖
try:
    # 尝试导入 CosyVoice2 模块
    from cosyvoice.cli.cosyvoice import CosyVoice2
except ImportError as e:
    error_msg = f"导入 cosyvoice 模块失败: {e}"
    logger.error(error_msg)
    logger.error("请确保已安装所需依赖:")
    logger.error("1. 运行 'pip install -r api/requirements.txt'")
    logger.error("2. 或使用 'conda activate cosyvoice' 切换到正确的环境")
    raise ImportError(error_msg)

# 导入配置
from . import config

def process_audio(tts_speeches, sample_rate=22050, format="wav"):
    """处理音频数据并返回字节流"""
    buffer = io.BytesIO()
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save(buffer, audio_data, sample_rate, format=format)
    buffer.seek(0)
    return buffer

def load_voice_data(speaker):
    """加载语音数据"""
    voice_path = f"{config.VOICES_DIR}/{speaker}.pt"
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(voice_path):
            return None
        voice_data = torch.load(voice_path, map_location=device)
        return voice_data.get('audio_ref')
    except Exception as e:
        logger.error(f"加载音色文件失败: {e}")
        raise ValueError(f"加载音色文件失败: {e}")

def get_available_voices() -> List[Dict[str, str]]:
    """获取所有可用的声音列表，包括预训练和自定义音色"""
    voices = []
    
    # 获取已有的 CosyVoice 模型
    tts_service = TTSService()
    
    # 获取预训练音色
    for name in tts_service.model.list_available_spks():
        voices.append({
            "name": name,
            "voice_id": name
        })
    
    # 获取自定义音色
    if os.path.exists(config.VOICES_DIR):
        for name in os.listdir(config.VOICES_DIR):
            if name.endswith('.pt'):
                voice_name = name.replace(".pt", "")
                voices.append({
                    "name": voice_name,
                    "voice_id": voice_name
                })
    
    return voices

class TTSService:
    """TTS服务单例类，负责管理CosyVoice2模型实例"""
    _instance = None
    
    def __new__(cls):
        # 单例模式，确保模型只加载一次
        if cls._instance is None:
            cls._instance = super(TTSService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        # 延迟初始化
        if not getattr(self, "initialized", False):
            self.initialize_model()
    
    def initialize_model(self):
        """初始化CosyVoice2模型
        
        加载预训练模型并准备好使用。如果加载失败，会抛出异常。
        """
        try:
            logger.info("初始化CosyVoice2模型...")
            model_path = config.MODEL_PATH  # 使用配置中的模型路径
            
            # 检查模型是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                raise FileNotFoundError(f"未找到TTS模型文件夹: {model_path}，请确保下载了预训练模型")
            
            # 确保FFmpeg环境变量设置正确
            if 'ffmpeg' not in os.environ.get('TORIO_FFMPEG_BINARY', ''):
                os.environ['TORIO_FFMPEG_BINARY'] = 'ffmpeg'
            
            # 在macOS上尝试解决动态库加载问题
            if sys.platform == 'darwin':
                # 设置动态库搜索路径
                if 'DYLD_LIBRARY_PATH' not in os.environ:
                    os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/lib:/opt/homebrew/lib'
                
                # 尝试直接加载libsndfile
                try:
                    import ctypes
                    libsndfile_paths = [
                        '/usr/local/lib/libsndfile.dylib',
                        '/opt/homebrew/lib/libsndfile.dylib'
                    ]
                    for path in libsndfile_paths:
                        if os.path.exists(path):
                            try:
                                ctypes.CDLL(path)
                                logger.info(f"成功加载 {path}")
                                break
                            except Exception as e:
                                logger.warning(f"尝试加载 {path} 失败: {e}")
                except Exception as e:
                    logger.warning(f"尝试预加载libsndfile失败，但将继续: {e}")
            
            # 初始化模型前检查CUDA可用性
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"使用设备: {device}")
            
            # 初始化模型
            self.model = CosyVoice2(model_path)
            self.sample_rate = self.model.sample_rate
            self.default_voices = self.model.list_available_spks()
            
            # 扫描自定义音色
            self.spk_custom = []
            if os.path.exists(config.VOICES_DIR):
                for name in os.listdir(config.VOICES_DIR):
                    if name.endswith('.pt'):
                        self.spk_custom.append(name.replace(".pt", ""))
            
            logger.info(f"默认音色: {self.default_voices}")
            logger.info(f"自定义音色: {self.spk_custom}")
            self.initialized = True
            logger.info("CosyVoice2模型初始化完成")
        except Exception as e:
            logger.error(f"初始化CosyVoice2模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"初始化TTS模型失败: {e}")
    
    def generate_tts(self, 
                    text: str, 
                    speaker: str, 
                    instruct: Optional[str] = None,
                    streaming: bool = False,
                    speed: float = 1.0):
        """生成语音
        
        Args:
            text: 需要转换的文本
            speaker: 音色ID
            instruct: 指令，用于控制语音风格
            streaming: 是否使用流式输出
            speed: 语速，范围0.5-2.0
        
        Returns:
            语音生成器对象
        
        Raises:
            ValueError: 当参数无效或音色不可用时
            RuntimeError: 当生成过程中出错时
        """
        try:
            # 检查音色是否存在
            if speaker not in self.default_voices and speaker not in self.spk_custom:
                available_voices = self.default_voices + self.spk_custom
                raise ValueError(f"音色 '{speaker}' 不存在。可用音色: {', '.join(available_voices)}")
            
            # 处理 instruct 模式
            if instruct:
                prompt_speech_16k = load_voice_data(speaker)
                if prompt_speech_16k is None:
                    raise ValueError(f"无法加载音色 '{speaker}' 的参考音频数据，请确保文件存在且格式正确")
                
                return self.model.inference_instruct2(
                    text, instruct, prompt_speech_16k, stream=streaming, speed=speed
                )
            else:
                return self.model.inference_sft(
                    text, speaker, stream=streaming, speed=speed
                )
        except Exception as e:
            # 记录详细错误
            logger.error(f"语音生成失败: {str(e)}")
            
            # 提供友好的错误信息
            if "memory" in str(e).lower():
                raise RuntimeError(f"内存不足，无法处理文本。请尝试减少文本长度或使用流式处理: {e}")
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise RuntimeError(f"网络连接错误，无法访问模型服务: {e}")
            elif "not implemented" in str(e).lower():
                raise RuntimeError(f"当前平台不支持该功能: {e}")
            else:
                raise RuntimeError(f"语音生成过程中发生错误: {e}") 