import io
import os
import sys
import logging
import torch
from fastapi import FastAPI, Request, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Generator
import time

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到 PATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 配置导入
try:
    from api.config import HOST, PORT, CORS_ORIGINS, API_KEYS, ENABLE_API_AUTH, NO_AUTH_PATHS, PRELOAD_MODEL
    from api.service import TTSService, get_available_voices, process_audio, load_voice_data
except ImportError as e:
    logger.error(f"导入基础模块失败: {e}")
    raise RuntimeError(f"无法导入必要的模块: {e}")

# 创建 API 密钥头字段
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# 创建 FastAPI 应用
app = FastAPI(
    title="CosyVoice2 API",
    description="CosyVoice2 文本转语音 API - 提供高质量的语音合成服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 TTS 服务
tts_service = None

def get_tts_service():
    global tts_service
    if tts_service is None:
        try:
            logger.info("初始化TTS服务...")
            tts_service = TTSService()
            logger.info("TTS服务初始化完成")
        except Exception as e:
            logger.error(f"初始化TTS服务失败: {e}")
            raise HTTPException(status_code=500, detail=f"初始化TTS服务失败: {str(e)}")
    return tts_service

# 应用启动事件处理
@app.on_event("startup")
async def startup_event():
    """应用启动时执行预加载操作"""
    global tts_service
    
    # 输出环境变量检查
    preload_env = os.environ.get("PRELOAD_MODEL", "未设置")
    logger.info(f"PRELOAD_MODEL环境变量值: '{preload_env}'")
    logger.info(f"配置解析后的PRELOAD_MODEL值: {PRELOAD_MODEL}")
    
    # 仅在PRELOAD_MODEL为True时预加载模型
    if PRELOAD_MODEL:
        logger.info("启动时预加载模型模式已启用...")
        try:
            # 设置FFmpeg环境变量
            logger.info("设置FFmpeg环境变量")
            os.environ['TORIO_FFMPEG_BINARY'] = 'ffmpeg'  # 指定使用系统的ffmpeg
            os.environ['TORIO_USE_FFMPEG'] = '0'  # 禁用torio内置的FFmpeg加载
            os.environ['TORIO_NO_FFMPEG'] = '1'  # 防止重复尝试加载失败的库
            
            # 在macOS上设置动态库搜索路径
            if sys.platform == 'darwin':
                if 'DYLD_LIBRARY_PATH' not in os.environ:
                    if os.path.exists('/opt/homebrew/lib'):
                        os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'
                    else:
                        os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/lib'
                        
            # 预加载TTS服务
            logger.info("开始预加载TTS服务...")
            tts_service = get_tts_service()
            
            # 预热 - 获取可用音色列表
            logger.info("获取可用音色列表...")
            voices = get_available_voices()
            logger.info(f"模型预加载成功，可用音色数: {len(voices)}")
            logger.info("API 服务启动成功")
        except Exception as e:
            logger.error(f"模型预加载失败: {e}", exc_info=True)
            logger.warning("将在第一次请求时尝试加载模型")
    else:
        logger.info("模型预加载已禁用，将在第一次请求时加载")

# 数据模型
class TTSRequest(BaseModel):
    text: str = Field(..., description="需要转换为语音的文本")
    speaker: str = Field(..., description="音色ID")
    instruct: Optional[str] = Field(None, description="指令，用于控制语音风格")
    streaming: bool = Field(False, description="是否使用流式输出")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="语速，范围0.5-2.0")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("文本不能为空")
        return v
    
    @validator('speaker')
    def speaker_not_empty(cls, v):
        if not v.strip():
            raise ValueError("音色ID不能为空")
        return v

    @validator('speaker')
    def speaker_exists(cls, v):
        # 这里不能直接调用get_available_voices()，因为验证器是静态方法
        # 但可以在路由函数中进行额外验证
        return v

# API 密钥验证函数
async def verify_api_key(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    # 如果API认证被禁用，则跳过验证
    if not ENABLE_API_AUTH:
        return True
        
    # 检查路径是否需要验证
    if request.url.path in NO_AUTH_PATHS or any(
        request.url.path.startswith(f"{path}/") for path in NO_AUTH_PATHS if path != "/"
    ):
        return True
    
    # 验证API密钥
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="未提供API密钥"
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    
    return True

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_size = 16 * 1024 * 1024  # 16MB
    
    if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "请求体过大，最大允许16MB"}
        )
    
    return await call_next(request)

# 自定义异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"请求验证错误: {exc}")
    return await request_validation_exception_handler(request, exc)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"处理请求时发生错误: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    )

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# API 路由
@app.get("/", summary="文本转语音", tags=["TTS"])
@app.get("/tts", summary="文本转语音", tags=["TTS"])
async def tts(
    request: Request,
    text: str = Query(..., description="需要转换为语音的文本"),
    speaker: str = Query(..., description="音色ID"),
    instruct: Optional[str] = Query(None, description="指令，用于控制语音风格"),
    streaming: int = Query(0, description="是否使用流式输出 (0: 否, 1: 是)"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="语速，范围0.5-2.0"),
    auth: bool = Depends(verify_api_key)
):
    """文本转语音 GET 接口
    
    将文本转换为语音，支持多种音色和指令控制。可以选择流式输出或一次性输出。
    
    - **text**: 需要转换的文本
    - **speaker**: 音色ID，可以使用 /speakers 接口获取可用音色列表
    - **instruct**: 可选的指令，用于控制语音风格，如"用开心的语气说"
    - **streaming**: 是否使用流式输出，适用于长文本，0表示否，1表示是
    - **speed**: 语速，范围0.5-2.0，1.0为正常速度
    """
    try:
        service = get_tts_service()
    except Exception as e:
        logger.error(f"获取TTS服务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        # 验证必要参数
        if not text or not speaker:
            raise HTTPException(status_code=400, detail="文本和音色ID不能为空")
        
        # 验证音色是否存在
        available_voices = get_available_voices()
        available_voice_ids = [voice["voice_id"] for voice in available_voices]
        
        if speaker not in available_voice_ids:
            # 提供可用的音色列表
            voice_list = ", ".join(available_voice_ids)
            raise HTTPException(
                status_code=400, 
                detail=f"音色ID '{speaker}' 不存在。可用的音色: {voice_list}"
            )
        
        # 处理请求
        streaming_bool = bool(streaming)
        
        logger.info(f"处理TTS请求: text='{text[:30]}{'...' if len(text)>30 else ''}', speaker='{speaker}', instruct='{instruct}'")
        
        # 定义推理函数
        def inference_func():
            try:
                if instruct:
                    prompt_speech_16k = load_voice_data(speaker)
                    if prompt_speech_16k is None:
                        raise HTTPException(status_code=400, detail=f"无法加载音色 '{speaker}' 的参考音频数据")
                    
                    return service.generate_tts(
                        text=text, 
                        speaker=speaker, 
                        instruct=instruct, 
                        streaming=streaming_bool, 
                        speed=speed
                    )
                else:
                    return service.generate_tts(
                        text=text, 
                        speaker=speaker, 
                        streaming=streaming_bool, 
                        speed=speed
                    )
            except Exception as e:
                logger.error(f"TTS生成失败: {e}", exc_info=True)
                error_msg = str(e)
                if "model inputs" in error_msg:
                    raise ValueError(f"模型输入错误: 文本可能过长或包含不支持的字符")
                elif "memory" in error_msg:
                    raise ValueError(f"内存不足: 请尝试减少文本长度或关闭流式处理")
                elif "not found" in error_msg or "音色" in error_msg:
                    raise ValueError(f"音色错误: {error_msg}")
                else:
                    raise
        
        # 处理流式输出
        if streaming_bool:
            async def generate():
                try:
                    for i, out in enumerate(inference_func()):
                        buffer = process_audio([out['tts_speech']], format="ogg")
                        yield buffer.read()
                except Exception as e:
                    logger.error(f"流式生成过程中发生错误: {str(e)}")
                    # 由于已经开始返回流，无法发送错误响应，只能记录日志
            
            return StreamingResponse(
                generate(),
                media_type="audio/ogg",
                headers={"Content-Disposition": "attachment; filename=sound.ogg"}
            )
        
        # 处理非流式输出
        tts_speeches = [i['tts_speech'] for i in inference_func()]
        buffer = process_audio(tts_speeches, format="wav")
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=sound.wav"}
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except ValueError as e:
        # 处理验证错误
        logger.warning(f"TTS输入验证错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/tts", summary="文本转语音(POST)", tags=["TTS"])
async def tts_post(
    request: Request,
    request_data: TTSRequest,
    auth: bool = Depends(verify_api_key)
):
    """文本转语音 POST 接口
    
    使用JSON格式请求体，将文本转换为语音。
    
    请求体格式：
    ```json
    {
        "text": "需要转换的文本",
        "speaker": "音色ID",
        "instruct": "用开心的语气说",  // 可选
        "streaming": false,         // 可选，默认false
        "speed": 1.0                // 可选，默认1.0
    }
    ```
    """
    return await tts(
        request=request,
        text=request_data.text,
        speaker=request_data.speaker,
        instruct=request_data.instruct,
        streaming=1 if request_data.streaming else 0,
        speed=request_data.speed
    )


@app.get("/speakers", summary="获取可用音色", tags=["Character"])
async def speakers(
    request: Request,
    auth: bool = Depends(verify_api_key)
):
    """获取所有可用音色列表
    
    返回系统中所有可用的音色列表，包括预训练音色和自定义音色。
    
    返回格式：
    ```json
    [
        {
            "name": "音色名称",
            "voice_id": "音色ID"
        },
        ...
    ]
    ```
    """
    try:
        voices = get_available_voices()
        logger.info(f"获取可用音色列表: 共{len(voices)}个音色")
        return voices
    except Exception as e:
        logger.error(f"获取音色列表错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/health", summary="健康检查", tags=["System"])
async def health_check():
    """健康检查接口
    
    用于监控系统健康状态，返回服务名称和状态。
    
    返回格式：
    ```json
    {
        "status": "healthy",
        "service": "CosyVoice2 API"
    }
    ```
    """
    return {"status": "healthy", "service": "CosyVoice2 API"}


# 启动服务器
if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动 FastAPI 服务: http://{HOST}:{PORT}")
    logger.info(f"API 文档: http://{HOST}:{PORT}/docs")
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=True)