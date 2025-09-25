import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import requests

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    print("[⚠️] Azure OpenAI library not installed")
    AZURE_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class ModelConfig:
    name: str
    provider: str 
    api_type: str
    endpoint: str
    api_key: str
    deployment_name: str = ""
    api_version: str = ""
    model_name: str = ""
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)

class BaseLLMBackend(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.request_count = 0
        self.success_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.request_count if self.request_count > 0 else 0,
            "total_tokens": self.total_tokens,
            "provider": self.config.provider,
            "model": self.config.name
        }

class AzureOpenAIBackend(BaseLLMBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not AZURE_AVAILABLE:
            raise ImportError("Azure OpenAI library not available")
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", config.api_key),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", config.api_version),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", config.endpoint)
        )
        
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", config.deployment_name)
        
        print(f"[🔧] Azure OpenAI backend initialized: {config.name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.request_count += 1
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    timeout=self.config.timeout
                )
                
                result = response.choices[0].message.content.strip()
                self.success_count += 1
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Azure OpenAI generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"[FAIL] {str(e)}"
    
    def is_available(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"Azure OpenAI unavailable: {e}")
            return False

class OpenAIBackend(BaseLLMBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        self.client = openai.OpenAI(api_key=config.api_key)
        
        print(f"[🔧] OpenAI backend initialized: {config.name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.request_count += 1
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    timeout=self.config.timeout
                )
                
                result = response.choices[0].message.content.strip()
                self.success_count += 1
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                
                return result
                
            except Exception as e:
                self.logger.warning(f"OpenAI generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"[FAIL] {str(e)}"
    
    def is_available(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"OpenAI unavailable: {e}")
            return False

class DeepSeekBackend(BaseLLMBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        
        print(f"[🔧] DeepSeek backend initialized: {config.name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.request_count += 1
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        headers.update(self.config.headers)
        
        payload = {
            "model": self.config.model_name or "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.session.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result_data = response.json()
                result = result_data["choices"][0]["message"]["content"].strip()
                
                self.success_count += 1
                self.total_tokens += result_data.get("usage", {}).get("total_tokens", 0)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"DeepSeek generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"[FAIL] {str(e)}"
    
    def is_available(self) -> bool:
        try:
            test_response = self.generate("Hello", max_tokens=10)
            return "[FAIL]" not in test_response
        except Exception as e:
            self.logger.error(f"DeepSeek unavailable: {e}")
            return False

class QwenBackend(BaseLLMBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        
        print(f"[🔧] Qwen backend initialized: {config.name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.request_count += 1
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable"
        }
        headers.update(self.config.headers)
        
        payload = {
            "model": self.config.model_name or "qwen-turbo",
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        }
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.session.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result_data = response.json()
                
                if result_data.get("output") and result_data["output"].get("choices"):
                    result = result_data["output"]["choices"][0]["message"]["content"].strip()
                else:
                    result = str(result_data)
                
                self.success_count += 1
                usage = result_data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Qwen generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"[FAIL] {str(e)}"
    
    def is_available(self) -> bool:
        try:
            test_response = self.generate("Hello", max_tokens=10)
            return "[FAIL]" not in test_response
        except Exception as e:
            self.logger.error(f"Qwen unavailable: {e}")
            return False

class ClaudeBackend(BaseLLMBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        
        print(f"[🔧] Claude backend initialized: {config.name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.request_count += 1
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        headers.update(self.config.headers)
        
        payload = {
            "model": self.config.model_name or "claude-3-sonnet-20240229",
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.session.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result_data = response.json()
                result = result_data["content"][0]["text"].strip()
                
                self.success_count += 1
                usage = result_data.get("usage", {})
                self.total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Claude generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"[FAIL] {str(e)}"
    
    def is_available(self) -> bool:
        try:
            test_response = self.generate("Hello", max_tokens=10)
            return "[FAIL]" not in test_response
        except Exception as e:
            self.logger.error(f"Claude unavailable: {e}")
            return False

class MultiModelManager:
    def __init__(self):
        self.backends: Dict[str, BaseLLMBackend] = {}
        self.current_backend = None
        self.fallback_order = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("[🎯] Multi-model manager initialized")
    
    def add_backend(self, name: str, backend: BaseLLMBackend):
        self.backends[name] = backend
        if self.current_backend is None:
            self.current_backend = name
        print(f"[➕] Backend added: {name}")
    
    def set_current_backend(self, name: str):
        if name in self.backends:
            self.current_backend = name
            print(f"[🔄] Switched to backend: {name}")
        else:
            print(f"[❌] Backend not found: {name}")
    
    def set_fallback_order(self, order: List[str]):
        self.fallback_order = [name for name in order if name in self.backends]
        print(f"[🔄] Fallback order set: {self.fallback_order}")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try_order = [self.current_backend] if self.current_backend else []
        try_order.extend([name for name in self.fallback_order if name != self.current_backend])
        
        if not try_order:
            return {"success": False, "result": "No available backend", "backend": None}
        
        for backend_name in try_order:
            if backend_name not in self.backends:
                continue
                
            backend = self.backends[backend_name]
            
            try:
                if not backend.is_available():
                    print(f"[⚠️] Backend {backend_name} unavailable, trying next")
                    continue
    
                start_time = time.time()
                result = backend.generate(prompt, **kwargs)
                generation_time = time.time() - start_time
                
                if "[FAIL]" not in result:
                    return {
                        "success": True,
                        "result": result,
                        "backend": backend_name,
                        "generation_time": generation_time,
                        "stats": backend.get_stats()
                    }
                else:
                    print(f"[⚠️] Backend {backend_name} generation failed, trying next")
                    
            except Exception as e:
                print(f"[❌] Backend {backend_name} exception: {e}")
                continue
        
        return {"success": False, "result": "All backends failed", "backend": None}
    
    def get_available_backends(self) -> List[str]:
        available = []
        for name, backend in self.backends.items():
            try:
                if backend.is_available():
                    available.append(name)
            except:
                continue
        return available
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: backend.get_stats() for name, backend in self.backends.items()}

def create_enhanced_multi_model_backend() -> MultiModelManager:
    manager = MultiModelManager()
    # Allow forcing MOCK backend via environment for offline/deterministic runs
    if os.getenv("USE_MOCK") == "1":
        class MockBackend(BaseLLMBackend):
            def __init__(self):
                super().__init__(ModelConfig(name="mock", provider="mock", api_type="mock", endpoint="", api_key=""))
            def generate(self, prompt: str, **kwargs) -> str:
                import hashlib
                self.request_count += 1
                h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
                self.success_count += 1
                return f"[MOCK] deterministic response {h}"
            def is_available(self) -> bool:
                return True
        manager.add_backend("mock", MockBackend())
        manager.set_current_backend("mock")
        print("[ℹ️] USE_MOCK=1 detected; using MOCK backend only")
        return manager
    if AZURE_AVAILABLE:
        try:
            from config.azure_config import get_azure_openai_config_from_env, setup_lab_environment
            setup_lab_environment()
            lab_config = get_azure_openai_config_from_env()
            azure_config = ModelConfig(
                name="Azure-GPT-4o",
                provider="azure",
                api_type="azure",
                endpoint=lab_config["endpoint"],
                api_key=lab_config["api_key"],
                deployment_name=lab_config["deployment_name"],
                api_version=lab_config["api_version"],
                max_tokens=lab_config["max_tokens"],
                temperature=lab_config["temperature"]
            )
            
            if azure_config.api_key and azure_config.endpoint:
                azure_backend = AzureOpenAIBackend(azure_config)
                manager.add_backend("azure-gpt4o", azure_backend)
                print("[✅] Azure OpenAI GPT-4o backend added")
                print(f"[ℹ️] Deployment: {azure_config.deployment_name}")
                print(f"[ℹ️] API version: {azure_config.api_version}")
            else:
                print("[⚠️] Azure OpenAI config incomplete, skipping")
                
        except Exception as e:
            print(f"[❌] Azure OpenAI backend init failed: {e}")
    try:
        deepseek_config = ModelConfig(
            name="DeepSeek-Chat",
            provider="deepseek",
            api_type="openai_compatible",
            endpoint="https://api.deepseek.com/v1/chat/completions",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model_name="deepseek-chat",
            max_tokens=2000,
            temperature=0.7
        )
        
        if deepseek_config.api_key:
            deepseek_backend = DeepSeekBackend(deepseek_config)
            manager.add_backend("deepseek", deepseek_backend)
            print("[✅] DeepSeek backend added")
        else:
            print("[⚠️] DeepSeek API key not set, skipping")
            
    except Exception as e:
        print(f"[❌] DeepSeek backend init failed: {e}")
    try:
        qwen_config = ModelConfig(
            name="Qwen-Turbo",
            provider="qwen",
            api_type="dashscope",
            endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            api_key=os.getenv("QWEN_API_KEY", ""),
            model_name="qwen-turbo",
            max_tokens=2000,
            temperature=0.7
        )
        
        if qwen_config.api_key:
            qwen_backend = QwenBackend(qwen_config)
            manager.add_backend("qwen", qwen_backend)
            print("[✅] Qwen backend added")
        else:
            print("[⚠️] Qwen API key not set, skipping")
            
    except Exception as e:
        print(f"[❌] Qwen backend init failed: {e}")
    try:
        claude_config = ModelConfig(
            name="Claude-3-Sonnet",
            provider="claude",
            api_type="anthropic",
            endpoint="https://api.anthropic.com/v1/messages",
            api_key=os.getenv("CLAUDE_API_KEY", ""),
            model_name="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.7
        )
        
        if claude_config.api_key:
            claude_backend = ClaudeBackend(claude_config)
            manager.add_backend("claude", claude_backend)
            print("[✅] Claude backend added")
        else:
            print("[⚠️] Claude API key not set, skipping")
            
    except Exception as e:
        print(f"[❌] Claude backend init failed: {e}")
    if OPENAI_AVAILABLE:
        try:
            openai_config = ModelConfig(
                name="GPT-4",
                provider="openai",
                api_type="openai",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name="gpt-4",
                max_tokens=2000,
                temperature=0.7
            )
            
            if openai_config.api_key:
                openai_backend = OpenAIBackend(openai_config)
                manager.add_backend("openai-gpt4", openai_backend)
                print("[✅] OpenAI backend added")
            else:
                print("[⚠️] OpenAI API key not set, skipping")
                
        except Exception as e:
            print(f"[❌] OpenAI backend init failed: {e}")
    manager.set_fallback_order(["azure-gpt4o", "openai-gpt4", "deepseek", "qwen", "claude"])
    
    available_backends = manager.get_available_backends()
    print(f"[📊] Available backends: {available_backends}")
    
    if not available_backends:
        class MockBackend(BaseLLMBackend):
            def __init__(self):
                super().__init__(ModelConfig(name="mock", provider="mock", api_type="mock", endpoint="", api_key=""))
            def generate(self, prompt: str, **kwargs) -> str:
                import hashlib
                self.request_count += 1
                h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
                self.success_count += 1
                return f"[MOCK] deterministic response {h}"
            def is_available(self) -> bool:
                return True
        manager.add_backend("mock", MockBackend())
        manager.set_current_backend("mock")
        print("[ℹ️] No API backends available; using MOCK backend for reproducibility")
    
    return manager
def create_multi_model_backend():
    manager = create_enhanced_multi_model_backend()
    
 
    class LegacyWrapper:
        def __init__(self, manager):
            self.manager = manager
        
        def generate(self, prompt: str) -> str:
            result = self.manager.generate(prompt)
            return result.get("result", "[FAIL]")
    
    return LegacyWrapper(manager)

if __name__ == "__main__":
    manager = create_enhanced_multi_model_backend()
    test_prompt = "Briefly introduce artificial intelligence."
    result = manager.generate(test_prompt)
    
    print("\n=== Test Result ===")
    print(f"Success: {result['success']}")
    print(f"Backend: {result.get('backend', 'None')}")
    print(f"Result: {result['result'][:100]}...")
    
    print("\n=== Backend Stats ===")
    stats = manager.get_all_stats()
    for name, stat in stats.items():
        print(f"{name}: {stat}") 