"""
services/rag_services.py

火山引擎知识库（Knowledge Base）检索服务（RAG）。

调用 /api/knowledge/collection/search_knowledge 接口，对电商客服知识库进行
「混合检索 + 多轮改写 + 重排」，返回与用户问题最相关的切片内容，供 LLM 在
回答前注入到 message 中（检索增强生成）。

接口文档（search_knowledge）：https://www.volcengine.com/docs/84313/
"""
from typing import Dict, List, Optional

import httpx

from rag_llm_server.config import settings


class RAGService:
    """火山引擎知识库检索服务"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.KB_API_KEY
        self.collection = collection or settings.KB_COLLECTION
        self.project = project or settings.KB_PROJECT
        self.base_url = (base_url or settings.KB_BASE_URL).rstrip("/")

    @property
    def search_url(self) -> str:
        return f"{self.base_url}/api/knowledge/collection/search_knowledge"

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "Host": settings.KB_DOMAIN,
            "Authorization": f"Bearer {self.api_key}",
        }

    def build_search_params(
        self,
        query: str,
        messages: Optional[List[Dict]] = None,
        image_query: Optional[str] = None,
    ) -> Dict:
        """构造 search_knowledge 请求体（电商客服小知识库调优参数）。"""
        params = {
            "project": self.project,
            "name": self.collection,
            "query": query,
            # 召回数量：小知识库取 4 条，太少易漏、太多易混入噪声
            "limit": settings.KB_LIMIT,
            # 混合检索中稠密向量的权重（0.5 = 关键词与语义并重，适合电商 FAQ）
            "dense_weight": settings.KB_DENSE_WEIGHT,
            "pre_processing": {
                # 不拼接 instruction，由本服务自行注入上下文
                "need_instruction": True,
                # 多轮改写：解决「那怎么退」「运费呢」这类指代缺失问题
                "rewrite": settings.KB_REWRITE,
                "messages": messages or [],
                "return_token_usage": False,
            },
            "post_processing": {
                # 开启重排，保证小知识库排序准确
                "rerank_switch": settings.KB_RERANK,
                "rerank_model": settings.KB_RERANK_MODEL,
                "rerank_only_chunk": False,
                # 进入重排的切片数（需 >= limit）
                "retrieve_count": settings.KB_RETRIEVE_COUNT,
                # 按文档顺序聚合切片，上下文更连贯
                "chunk_group": True,
                "get_attachment_link": False,
            },
        }
        if image_query:
            params["image_query"] = image_query
        return params

    async def search(
        self,
        query: str,
        messages: Optional[List[Dict]] = None,
        image_query: Optional[str] = None,
    ) -> List[Dict]:
        """检索知识库，返回命中的切片列表（原始 result_list）。"""
        if not self.api_key:
            return []
        payload = self.build_search_params(query, messages, image_query)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.search_url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            print(f"[RAG] 检索返回异常: code={data.get('code')} message={data.get('message')}")
            return []
        return data.get("data", {}).get("result_list", []) or []

    @staticmethod
    def chunks_to_context(result_list: List[Dict]) -> str:
        """把检索切片整理成可注入 LLM 的上下文文本。"""
        if not result_list:
            return ""
        blocks = []
        for i, item in enumerate(result_list, 1):
            content = (item.get("content") or "").strip()
            title = (item.get("chunk_title") or "").strip()
            if not content:
                continue
            blocks.append(f"[资料{i}] {title + '：' if title else ''}{content}")
        return "\n\n".join(blocks)


# 全局单例：知识库检索入口
rag_service = RAGService()