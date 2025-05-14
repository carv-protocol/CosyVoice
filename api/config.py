import os
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "audio")
VOICES_DIR = os.path.join(ROOT_DIR, "voices")

# 确保目录存在
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# 服务器配置
HOST = "0.0.0.0"  # 绑定到所有网络接口
PORT = int(os.environ.get("COSYVOICE_API_PORT", 9880))  # 使用环境变量或默认端口 9880

# 模型预加载配置
preload_env = os.environ.get("PRELOAD_MODEL", "True")
if preload_env.lower() in ("true", "1", "t", "yes", "y"):
    PRELOAD_MODEL = True
else:
    PRELOAD_MODEL = True

logger.info(f"环境变量PRELOAD_MODEL={preload_env}, 解析为: {PRELOAD_MODEL}")

# 音频处理参数
MAX_VAL = 0.8
PROMPT_SR = 16000

# 模型路径
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_models/CosyVoice2-0.5B")
# LOAD_JIT = True

# # 后处理参数
# TOP_DB = 60
# HOP_LENGTH = 220
# WIN_LENGTH = 440

# API密钥配置
API_KEYS = [
    "cosyvoice-api-demo",  # dev 测试用 API 密钥
]

# 不需要API密钥验证的路径
NO_AUTH_PATHS = [
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
]

# 是否启用API认证（默认启用）
ENABLE_API_AUTH = True

# 添加允许所有源的CORS配置
CORS_ORIGINS = ["*"]  # 测试环境允许所有源，生产环境应该设置为特定域名

# 目录路径
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audios")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# 调试模式
DEBUG = True
