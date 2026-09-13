"""
services/llm_services.py

豆包大模型服务封装。

通过火山方舟（ARK）的 OpenAI 兼容接口访问豆包 Doubao-Seed-2.1-turbo，
对外提供流式（SSE）与非流式两种调用方式，供 main.py 使用。

说明：
- 大模型初始化在模块底部以全局单例 `llm_service` 完成。
- RTC「接入第三方大模型 / Agent」（CustomLLM）会以 POST 请求携带
  OpenAI 风格的 messages 到本服务，本服务再转发给方舟并流式回传结果。
"""
import json
from typing import AsyncGenerator, Dict, Optional

import httpx

from rag_llm_server.config import settings


class LLMService:
    """豆包大模型服务（火山方舟 ARK OpenAI 兼容接口）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.ARK_API_KEY
        self.base_url = (base_url or settings.ARK_BASE_URL).rstrip("/")
        self.model = model or settings.ARK_MODEL

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    # ---------- 请求构造 ----------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_request_payload(self, incoming: Dict) -> Dict:
        """
        将 RTC 传入的请求体转换为方舟 ARK 的请求体。

        RTC 会透传以下字段（参见《接入第三方大模型或 Agent》）：
          messages / stream / temperature / max_tokens / model / top_p /
          custom / parallel_tool_calls / service_tier 等。
        """
        payload = {
            # 优先使用请求体中的 model，否则使用本地配置的豆包模型
            "model": incoming.get("model") or self.model,
            "messages": incoming.get("messages", []),
            "stream": True,
        }

        # 常用采样参数透传
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if incoming.get(key) is not None:
                payload[key] = incoming[key]

        # 业务自定义数据透传
        for key in ("custom", "parallel_tool_calls", "service_tier"):
            if key in incoming:
                payload[key] = incoming[key]

        # 在流式响应末尾附带 token 用量统计
        payload.setdefault("stream_options", {"include_usage": True})

        return payload

    # ---------- 调用 ----------

    async def stream_chat(self, incoming: Dict) -> AsyncGenerator[str, None]:
        """
        流式调用豆包模型，逐条产出 SSE 事件字符串：

            data: {json}\n\n
            ...
            data: [DONE]\n\n

        直接转发方舟返回的 OpenAI 兼容流式数据，并保证以 data: [DONE] 结束。
        """
        payload = self.build_request_payload(incoming)
        seen_done = False

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.chat_url,
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    yield self._format_error(resp.status_code, await resp.aread())
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # 方舟返回的是 SSE 行：data: {...}，直接透传
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            seen_done = True
                        yield f"data: {data}\n\n"

        # 兜底：确保 RTC 能判定本轮回答结束
        if not seen_done:
            yield "data: [DONE]\n\n"

    async def chat(self, incoming: Dict) -> Dict:
        """非流式调用（备用），返回完整 JSON。"""
        payload = self.build_request_payload(incoming)
        payload["stream"] = False

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.chat_url,
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ---------- 内部工具 ----------

    def _format_error(self, status_code: int, body: bytes) -> str:
        """将上游错误格式化为 SSE 事件，供 RTC 侧感知。"""
        try:
            err = json.loads(body.decode("utf-8"))
        except Exception:
            err = {
                "error": {
                    "message": f"upstream LLM error (HTTP {status_code})",
                    "type": "upstream_error",
                }
            }
        return f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


# 全局单例：大模型初始化入口
llm_service = LLMService()