"""
多平台API服务模块
支持自定义OpenAI API、OpenRouter、Ollama、LMStudio等平台
"""

import httpx
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
import os
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

def debug_print(*args, **kwargs):
    """统一的DEBUG输出函数，只在DEBUG_MODE启用时输出"""
    if DEBUG_MODE:
        print(*args, **kwargs)

class PlatformType(Enum):
    """平台类型枚举"""
    CUSTOM_OPENAI = "custom_openai"  # 自定义OpenAI API
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"

@dataclass
class PlatformConfig:
    """平台配置"""
    platform_type: PlatformType
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    timeout: int = 30

@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    name: str
    platform: PlatformType
    enabled: bool = True
    description: str = ""

class PlatformClient:
    """平台客户端基类"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.client = None
    
    async def get_models(self) -> List[ModelInfo]:
        """获取可用模型列表"""
        raise NotImplementedError
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """聊天补全接口"""
        raise NotImplementedError
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            models = await self.get_models()
            return len(models) > 0
        except Exception as e:
            logger.error(f"Platform {self.config.platform_type} connection test failed: {e}")
            return False

class CustomOpenAIClient(PlatformClient):
    """自定义OpenAI API客户端"""
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com"
    
    async def get_models(self) -> List[ModelInfo]:
        """获取自定义OpenAI API模型列表"""
        logger.info("🔍 [CustomOpenAI] 开始获取模型列表...")
        
        if not self.config.api_key:
            logger.warning("⚠️ [CustomOpenAI] API Key未配置，跳过获取模型")
            return []
        
        try:
            logger.info(f"🌐 [CustomOpenAI] 请求URL: {self.base_url}/v1/models")
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                logger.info(f"📡 [CustomOpenAI] API响应状态: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    logger.info(f"📋 [CustomOpenAI] 响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    
                    # 解析标准OpenAI API模型列表格式
                    if "data" in data:
                        for model in data["data"]:
                            model_info = ModelInfo(
                                id=model.get("id", ""),
                                name=model.get("id", ""),
                                platform=PlatformType.CUSTOM_OPENAI,
                                description=model.get("description", f"创建时间: {model.get('created', 'Unknown')}")
                            )
                            models.append(model_info)
                    else:
                        # 如果API返回格式不匹配，添加一些常见的默认模型
                        logger.info("⚠️ [CustomOpenAI] API响应格式不匹配，使用默认模型列表")
                        default_models = [
                            {"id": "gpt-4", "name": "gpt-4", "description": "GPT-4 模型"},
                            {"id": "gpt-4-turbo", "name": "gpt-4-turbo", "description": "GPT-4 Turbo 模型"},
                            {"id": "gpt-3.5-turbo", "name": "gpt-3.5-turbo", "description": "GPT-3.5 Turbo 模型"},
                            {"id": "claude-3-opus", "name": "claude-3-opus", "description": "Claude 3 Opus 模型"},
                            {"id": "claude-3-sonnet", "name": "claude-3-sonnet", "description": "Claude 3 Sonnet 模型"},
                            {"id": "claude-3-haiku", "name": "claude-3-haiku", "description": "Claude 3 Haiku 模型"},
                        ]
                        
                        for model in default_models:
                            model_info = ModelInfo(
                                id=model["id"],
                                name=model["name"],
                                platform=PlatformType.CUSTOM_OPENAI,
                                description=model["description"]
                            )
                            models.append(model_info)
            
                    
                    logger.info(f"✅ [CustomOpenAI] 成功获取 {len(models)} 个模型")
                    return models
                else:
                    logger.error(f"❌ [CustomOpenAI] API错误: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ [CustomOpenAI] 获取模型失败: {e}")
            return []
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """自定义OpenAI API聊天补全"""
        if not self.config.api_key:
            yield json.dumps({"error": "API key not configured"})
            return
        
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream:
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.strip():
                                    if line.startswith("data: "):
                                        data = line[6:]
                                        if data.strip() == "[DONE]":
                                            break
                                        yield data
                                    else:
                                        yield line
                        else:
                            error_msg = await response.aread()
                            yield json.dumps({"error": f"API error: {response.status_code} - {error_msg.decode()}"})
                else:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        yield response.text
                    else:
                        yield json.dumps({"error": f"API error: {response.status_code} - {response.text}"})
                        
        except Exception as e:
            logger.error(f"CustomOpenAI chat completion error: {e}")
            yield json.dumps({"error": f"Request failed: {str(e)}"})

class OpenRouterClient(PlatformClient):
    """OpenRouter客户端"""
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        self.base_url = "https://openrouter.ai/api/v1"
    
    async def get_models(self) -> List[ModelInfo]:
        """获取OpenRouter模型列表"""
        if not self.config.api_key:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    if "data" in data:
                        for model in data["data"]:
                            models.append(ModelInfo(
                                id=model.get("id", ""),
                                name=model.get("name", model.get("id", "")),
                                platform=PlatformType.OPENROUTER,
                                description=model.get("description", "")
                            ))
                    
                    return models
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to get OpenRouter models: {e}")
            return []
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """OpenRouter聊天补全"""
        if not self.config.api_key:
            yield json.dumps({"error": "API key not configured"})
            return
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream:
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.strip():
                                    # 直接 yield 原始行，让转换器处理格式
                                    yield line
                        else:
                            error_msg = await response.aread()
                            yield json.dumps({"error": f"API error: {response.status_code} - {error_msg.decode()}"})
                else:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        yield response.text
                    else:
                        yield json.dumps({"error": f"API error: {response.status_code} - {response.text}"})
                        
        except Exception as e:
            logger.error(f"OpenRouter chat completion error: {e}")
            yield json.dumps({"error": f"Request failed: {str(e)}"})

class OllamaClient(PlatformClient):
    """Ollama客户端"""
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    async def get_models(self) -> List[ModelInfo]:
        """获取Ollama模型列表"""
        logger.info("🔍 [Ollama] 开始获取模型列表...")
        logger.info(f"🌐 [Ollama] 请求URL: {self.base_url}/api/tags")
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                logger.info(f"📡 [Ollama] API响应状态: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    logger.info(f"📋 [Ollama] 响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    
                    if "models" in data:
                        for model in data["models"]:
                            model_info = ModelInfo(
                                id=model.get("name", ""),
                                name=model.get("name", ""),
                                platform=PlatformType.OLLAMA,
                                description=f"Size: {model.get('size', 'Unknown')}"
                            )
                            models.append(model_info)
            
                    
                    logger.info(f"✅ [Ollama] 成功获取 {len(models)} 个模型")
                    return models
                else:
                    logger.error(f"❌ [Ollama] API错误: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ [Ollama] 获取模型失败: {e}")
            return []
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = True,  # Ollama默认使用流式
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Ollama聊天补全"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream:
                    async with client.stream(
                        "POST", url, json=payload
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        # 转换Ollama格式到OpenAI格式
                                        openai_chunk = self._convert_ollama_to_openai(data)
                                        yield json.dumps(openai_chunk)
                                        
                                        if data.get("done", False):
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            error_msg = await response.aread()
                            yield json.dumps({"error": f"API error: {response.status_code} - {error_msg.decode()}"})
                else:
                    # 非流式模式需要手动收集所有响应
                    full_response = ""
                    async with client.stream("POST", url, json=payload) as response:
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "message" in data and "content" in data["message"]:
                                        full_response += data["message"]["content"]
                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    openai_response = {
                        "id": "chatcmpl-ollama",
                        "object": "chat.completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_response
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    yield json.dumps(openai_response)
                        
        except Exception as e:
            logger.error(f"Ollama chat completion error: {e}")
            yield json.dumps({"error": f"Request failed: {str(e)}"})
    
    def _convert_ollama_to_openai(self, ollama_data: Dict[str, Any]) -> Dict[str, Any]:
        """将Ollama响应格式转换为OpenAI格式"""
        content = ""
        if "message" in ollama_data and "content" in ollama_data["message"]:
            content = ollama_data["message"]["content"]
        
        return {
            "id": "chatcmpl-ollama",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": ollama_data.get("model", "unknown"),
            "choices": [{
                "index": 0,
                "delta": {
                    "content": content
                } if content else {},
                "finish_reason": "stop" if ollama_data.get("done", False) else None
            }]
        }

class LMStudioClient(PlatformClient):
    """LMStudio客户端"""
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:1234"
    
    async def get_models(self) -> List[ModelInfo]:
        """获取LMStudio模型列表"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    if "data" in data:
                        for model in data["data"]:
                            models.append(ModelInfo(
                                id=model.get("id", ""),
                                name=model.get("id", ""),
                                platform=PlatformType.LMSTUDIO,
                                description="LMStudio local model"
                            ))
                    
                    return models
                else:
                    logger.error(f"LMStudio API error: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to get LMStudio models: {e}")
            return []
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """LMStudio聊天补全"""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream:
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.strip():
                                    if line.startswith("data: "):
                                        data = line[6:]
                                        if data.strip() == "[DONE]":
                                            break
                                        yield data
                                    else:
                                        yield line
                        else:
                            error_msg = await response.aread()
                            yield json.dumps({"error": f"API error: {response.status_code} - {error_msg.decode()}"})
                else:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        yield response.text
                    else:
                        yield json.dumps({"error": f"API error: {response.status_code} - {response.text}"})
                        
        except Exception as e:
            logger.error(f"LMStudio chat completion error: {e}")
            yield json.dumps({"error": f"Request failed: {str(e)}"})

class PlatformManager:
    """平台管理器"""
    
    def __init__(self):
        self.platforms: Dict[PlatformType, PlatformClient] = {}
    
    def add_platform(self, config: PlatformConfig):
        """添加平台"""
        if config.platform_type == PlatformType.CUSTOM_OPENAI:
            client = CustomOpenAIClient(config)
        elif config.platform_type == PlatformType.OPENROUTER:
            client = OpenRouterClient(config)
        elif config.platform_type == PlatformType.OLLAMA:
            client = OllamaClient(config)
        elif config.platform_type == PlatformType.LMSTUDIO:
            client = LMStudioClient(config)
        else:
            raise ValueError(f"Unsupported platform type: {config.platform_type}")
        
        self.platforms[config.platform_type] = client
    
    def get_platform(self, platform_type: PlatformType) -> Optional[PlatformClient]:
        """获取平台客户端"""
        return self.platforms.get(platform_type)
    
    async def get_all_models(self) -> List[ModelInfo]:
        """获取所有平台的模型列表"""
        logger.info("🚀 [PlatformManager] 开始获取所有平台模型列表...")
        
        all_models = []
        for platform_type, platform in self.platforms.items():
            try:
                logger.info(f"📞 [PlatformManager] 调用 {platform_type.value} 平台...")
                models = await platform.get_models()
                logger.info(f"📦 [PlatformManager] {platform_type.value} 返回 {len(models)} 个模型")
                all_models.extend(models)
            except Exception as e:
                logger.error(f"❌ [PlatformManager] {platform_type.value} 平台获取模型失败: {e}")
        
        logger.info(f"🎯 [PlatformManager] 总共获取到 {len(all_models)} 个模型")
        return all_models
    
    async def test_all_connections(self) -> Dict[PlatformType, bool]:
        """测试所有平台连接"""
        results = {}
        for platform_type, client in self.platforms.items():
            results[platform_type] = await client.test_connection()
        
        return results