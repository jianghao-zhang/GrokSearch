import asyncio
from dataclasses import dataclass

from .config import config
from .providers.grok import GrokSearchProvider
from .sources import merge_sources, split_answer_and_sources


@dataclass(slots=True)
class SearchRunResult:
    content: str
    sources: list[dict]
    model: str


def _extra_results_to_sources(
    tavily_results: list[dict] | None,
    firecrawl_results: list[dict] | None,
) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    if firecrawl_results:
        for r in firecrawl_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "firecrawl"}
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            desc = (r.get("description") or "").strip()
            if desc:
                item["description"] = desc
            sources.append(item)

    if tavily_results:
        for r in tavily_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "tavily"}
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            content = (r.get("content") or "").strip()
            if content:
                item["description"] = content
            sources.append(item)

    return sources


async def _call_tavily_search(query: str, max_results: int = 6) -> list[dict] | None:
    import httpx

    api_key = config.tavily_api_key
    if not api_key:
        return None
    endpoint = f"{config.tavily_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "include_answer": False,
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in results
            ] if results else None
    except Exception:
        return None


async def _call_firecrawl_search(query: str, limit: int = 14) -> list[dict] | None:
    import httpx

    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{config.firecrawl_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"query": query, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("data", {}).get("web", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                }
                for r in results
            ] if results else None
    except Exception:
        return None


async def run_search(
    query: str,
    platform: str = "",
    model: str = "",
    extra_sources: int = 0,
    *,
    suppress_grok_errors: bool = False,
) -> SearchRunResult:
    api_url = config.grok_api_url
    api_key = config.grok_api_key
    effective_model = model or config.grok_model
    grok_provider = GrokSearchProvider(api_url, api_key, effective_model)

    has_tavily = bool(config.tavily_api_key)
    has_firecrawl = bool(config.firecrawl_api_key)
    firecrawl_count = 0
    tavily_count = 0
    if extra_sources > 0:
        if has_firecrawl and has_tavily:
            firecrawl_count = extra_sources
            tavily_count = 0
        elif has_firecrawl:
            firecrawl_count = extra_sources
        elif has_tavily:
            tavily_count = extra_sources

    async def _safe_grok() -> str:
        try:
            return await grok_provider.search(query, platform)
        except Exception:
            if suppress_grok_errors:
                return ""
            raise

    async def _safe_tavily() -> list[dict] | None:
        if tavily_count <= 0:
            return None
        return await _call_tavily_search(query, tavily_count)

    async def _safe_firecrawl() -> list[dict] | None:
        if firecrawl_count <= 0:
            return None
        return await _call_firecrawl_search(query, firecrawl_count)

    coros: list = [_safe_grok()]
    if tavily_count > 0:
        coros.append(_safe_tavily())
    if firecrawl_count > 0:
        coros.append(_safe_firecrawl())

    gathered = await asyncio.gather(*coros)
    grok_result: str = gathered[0] or ""
    tavily_results: list[dict] | None = None
    firecrawl_results: list[dict] | None = None
    idx = 1
    if tavily_count > 0:
        tavily_results = gathered[idx]
        idx += 1
    if firecrawl_count > 0:
        firecrawl_results = gathered[idx]

    answer, grok_sources = split_answer_and_sources(grok_result)
    extra = _extra_results_to_sources(tavily_results, firecrawl_results)
    all_sources = merge_sources(grok_sources, extra)
    return SearchRunResult(content=answer, sources=all_sources, model=effective_model)
