#!/usr/bin/env python3
"""GHA AI v4 — Production-grade PocketPal remote endpoint
Design goals (from v4 plan):
  - Search OFF by default (biggest speed win)
  - Dual models: fast (no search) + search-enabled
  - Rich /v1/models metadata (context_length, tools, vision)
  - Perfect SSE for PocketPal
  - Real web_search + web_fetch tools
  - Optional OpenAI-style tools/function calling
  - /props for llama.cpp-style detection
  - Never inject empty context
  - Keep model warm, high context (16k)
Runs only on GitHub Actions (Ollama + SearXNG + DDGS). No Bedrock.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, AsyncGenerator, List, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5:3b")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
API_KEY = os.environ.get("GHA_API_KEY", "sk-gha-v4")
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
VERSION = "4.0.0"

# Dual mode aliases
FAST_ALIASES = {"fast", "gha-v4", "gha-v4-fast", MODEL}
SEARCH_ALIASES = {"gha-v4-search", "search"}

app = FastAPI(title="GHA AI v4 — PocketPal native", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class Msg(BaseModel):
    role: str
    content: Any = ""


class ToolDef(BaseModel):
    type: str = "function"
    function: dict = Field(default_factory=dict)


class ChatReq(BaseModel):
    model: str = MODEL
    messages: List[Msg]
    max_tokens: Optional[int] = 512
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    web_search: Optional[bool] = None  # None = auto (OFF by default in v4)
    web_fetch_urls: Optional[List[str]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    stop: Optional[Any] = None


def _check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if not authorization:
        return
    token = authorization.replace("Bearer", "").strip()
    if token and token != API_KEY:
        raise HTTPException(401, "Invalid API key")


def _text(m: Msg) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(c)


# ---------- Search: OFF by default (v4 plan) ----------
# Only trigger on explicit search intent keywords
SEARCH_TRIGGER_RE = re.compile(
    r"\b("
    r"search|look\s*up|google|find\s+(out|info|information)|"
    r"what\s+is\s+the\s+(latest|current|today)|"
    r"latest\s+(news|price|update)|current\s+(price|status|news)|"
    r"who\s+is\s+|when\s+did\s+|according\s+to|"
    r"wikipedia|source[s]?\s*:|cite\s+sources|"
    r"web\s*search|browse|fetch\s+(url|page|this)|"
    r"202[4-6]\s+(news|election|update|release)"
    r")\b",
    re.I,
)


def needs_search(messages: List[Msg], model: str, explicit: Optional[bool]) -> bool:
    """v4: search OFF by default. Only when model is search alias, explicit flag, or clear intent."""
    if explicit is True:
        return True
    if explicit is False:
        return False
    if model in SEARCH_ALIASES:
        return True
    last = ""
    for m in reversed(messages):
        if m.role == "user":
            last = _text(m).strip()
            break
    if not last:
        return False
    return bool(SEARCH_TRIGGER_RE.search(last))


# ---------- Search / fetch engines ----------
def searx_search(query: str, num: int = 6) -> list[dict]:
    try:
        r = httpx.get(
            f"{SEARX}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=15.0,
        )
        if r.status_code != 200:
            return []
        out = []
        for item in (r.json().get("results") or [])[:num]:
            url = item.get("url") or ""
            if not url:
                continue
            out.append({
                "title": (item.get("title") or "")[:180],
                "url": url,
                "snippet": (item.get("content") or item.get("snippet") or "")[:450],
                "engine": "searxng",
            })
        return out
    except Exception as e:
        print("searx", e)
        return []


def ddgs_search(query: str, num: int = 6) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                out.append({
                    "title": (r.get("title") or "")[:180],
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": (r.get("body") or "")[:450],
                    "engine": "ddgs",
                })
        return out
    except Exception as e:
        print("ddgs", e)
        return []


def wiki_fallback(query: str) -> list[dict]:
    try:
        headers = {"User-Agent": "GHA-AI-v4/4.0 (research; +https://github.com)"}
        s = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 2, "format": "json"},
            headers=headers,
            timeout=10.0,
        )
        if s.status_code != 200:
            return []
        data = s.json()
        titles = data[1] if len(data) > 1 else []
        urls = data[3] if len(data) > 3 else []
        out = []
        for title, url in zip(titles[:2], urls[:2]):
            sumr = httpx.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
                headers=headers,
                timeout=8.0,
            )
            extract = ""
            if sumr.status_code == 200:
                extract = (sumr.json().get("extract") or "")[:700]
            out.append({
                "title": title,
                "url": url or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": extract,
                "engine": "wikipedia",
            })
        return out
    except Exception as e:
        print("wiki", e)
        return []


def web_search(query: str, num: int = 6) -> list[dict]:
    results = searx_search(query, num=num)
    if len(results) < 3:
        results.extend(ddgs_search(query, num=num))
    if len(results) < 2:
        results.extend(wiki_fallback(query))
    seen, uniq = set(), []
    for r in results:
        u = (r.get("url") or "").split("#")[0].rstrip("/")
        if u and u not in seen and len(u) > 8:
            seen.add(u)
            uniq.append(r)
    return uniq[:num]


def web_fetch(url: str, max_chars: int = 10000) -> dict:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GHA-AI-v4/4.0; research)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=15.0)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            text = r.text[:max_chars]
        elif "html" in ctype or "xml" in ctype or not ctype:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"}) or soup.body
            text = " ".join((main or soup).get_text(" ", strip=True).split())[:max_chars]
        else:
            text = r.text[:max_chars]
        return {
            "url": str(r.url),
            "status": r.status_code,
            "content": text,
            "ok": r.status_code < 400 and len(text) > 60,
        }
    except Exception as e:
        return {"url": url, "status": 0, "content": f"[FETCH FAILED] {e}", "ok": False}


def build_context(req: ChatReq, user_q: str, model: str) -> tuple[list[str], bool, list[dict]]:
    do_search = needs_search(req.messages, model, req.web_search)
    context_blocks: list[str] = []
    sources: list[dict] = []

    if do_search and user_q:
        hits = web_search(user_q, num=6)
        if not hits:
            hits = wiki_fallback(user_q)
        if hits:
            lines = ["[WEB SEARCH RESULTS — answer only from these; cite URLs]"]
            for i, h in enumerate(hits, 1):
                lines.append(
                    f"{i}. {h['title']}\n   URL: {h['url']}\n   {h['snippet']}\n   source={h.get('engine','')}"
                )
                sources.append({"title": h["title"], "url": h["url"], "engine": h.get("engine")})
            context_blocks.append("\n".join(lines))
            # Fetch top 1–2 pages for depth
            fetched = 0
            for h in hits[:2]:
                if not (h.get("url") or "").startswith("http"):
                    continue
                page = web_fetch(h["url"], max_chars=4000)
                if page.get("ok") and page.get("content"):
                    context_blocks.append(f"[FETCHED PAGE]\nURL: {page['url']}\n{page['content'][:3500]}")
                    fetched += 1
                if fetched >= 2:
                    break
        # if still empty after all fallbacks — do NOT inject empty block

    if req.web_fetch_urls:
        for u in (req.web_fetch_urls or [])[:3]:
            page = web_fetch(u, max_chars=5500)
            if page.get("ok") or page.get("content"):
                context_blocks.append(f"[FETCHED PAGE]\nURL: {page['url']}\n{page.get('content','')[:5000]}")
                sources.append({"title": page["url"], "url": page["url"], "engine": "fetch"})

    return context_blocks, do_search, sources


def build_ollama_messages(req: ChatReq, context_blocks: list[str]) -> list[dict]:
    system_extra = (
        "You are a helpful, accurate assistant on GitHub Actions.\n"
        "Rules:\n"
        "- Be concise and direct.\n"
        "- If [WEB SEARCH RESULTS] or [FETCHED PAGE] are present, use ONLY that context and cite URLs.\n"
        "- If no search context is present, answer from knowledge and say when unsure.\n"
        "- Never invent GitHub workflows, fake sources, or unrelated content.\n"
    )
    if context_blocks:
        system_extra += "\n\n" + "\n\n".join(context_blocks)

    ollama_messages = []
    has_system = False
    for m in req.messages:
        role = m.role if m.role in ("system", "user", "assistant") else "user"
        content = _text(m)
        if role == "system":
            content = system_extra + "\n\n" + content
            has_system = True
        ollama_messages.append({"role": role, "content": content})
    if not has_system:
        ollama_messages.insert(0, {"role": "system", "content": system_extra})
    return ollama_messages


def resolve_model(name: str) -> str:
    if name in SEARCH_ALIASES or name in FAST_ALIASES or name == MODEL:
        return MODEL
    return name or MODEL


# ---------- Routes ----------
@app.get("/health")
def health():
    ollama_ok = searx_ok = False
    try:
        ollama_ok = httpx.get(f"{OLLAMA}/api/tags", timeout=3.0).status_code == 200
    except Exception:
        pass
    try:
        searx_ok = httpx.get(f"{SEARX}/", timeout=3.0).status_code < 500
    except Exception:
        pass
    return {
        "ok": ollama_ok,
        "version": VERSION,
        "ollama": ollama_ok,
        "searxng": searx_ok,
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "features": [
            "web_search", "web_fetch", "openai_chat", "sse_stream",
            "pocketpal", "tools", "props", "dual_mode", "search_off_by_default",
        ],
        "api_key_hint": f"{API_KEY} (optional)",
        "modes": {
            "fast": "no search (default) — low latency",
            "gha-v4-search": "smart search when intent detected or forced",
        },
    }


@app.get("/v1/models")
def models():
    """Rich model list for PocketPal (context_length, capabilities)."""
    now = int(time.time())
    base = {
        "object": "model",
        "created": now,
        "owned_by": "gha-v4",
        "context_length": NUM_CTX,
        "max_model_len": NUM_CTX,
        "supports_vision": False,
        "supports_tools": True,
        "supports_streaming": True,
    }
    return {
        "object": "list",
        "data": [
            {**base, "id": MODEL, "description": "Default fast model (search off)"},
            {**base, "id": "gha-v4", "description": "Alias for fast mode"},
            {**base, "id": "gha-v4-fast", "description": "Fast mode — no web search"},
            {**base, "id": "gha-v4-search", "description": "Search mode — grounded answers"},
            {**base, "id": "fast", "description": "Fast alias"},
        ],
    }


@app.get("/props")
@app.get("/v1/props")
def props():
    """llama.cpp-style props for PocketPal capability discovery."""
    return {
        "default_generation_settings": {
            "n_ctx": NUM_CTX,
            "n_predict": 512,
        },
        "total_slots": 1,
        "model_path": MODEL,
        "chat_template": "chatml",
        "modalities": {"vision": False},
        "gha_v4": {
            "version": VERSION,
            "supports_tools": True,
            "supports_streaming": True,
        },
    }


@app.post("/v1/tools/web_search")
def tool_search(body: dict):
    q = body.get("query") or body.get("q") or ""
    if not q:
        raise HTTPException(400, "query required")
    results = web_search(q, num=int(body.get("num", 6)))
    return {"query": q, "count": len(results), "results": results}


@app.post("/v1/tools/web_fetch")
def tool_fetch(body: dict):
    url = body.get("url") or ""
    if not url:
        raise HTTPException(400, "url required")
    return web_fetch(url, max_chars=int(body.get("max_chars", 12000)))


async def _sse_from_ollama(
    ollama_messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    meta: dict,
) -> AsyncGenerator[str, None]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": NUM_CTX,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            async with client.stream("POST", f"{OLLAMA}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': err.decode()[:400]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    msg = chunk.get("message") or {}
                    content = msg.get("content") or ""
                    done = chunk.get("done", False)
                    if content:
                        data = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if done:
                        data = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                            "gha_v4": meta,
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
        yield "data: [DONE]\n\n"
    except Exception as e:
        err = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"[stream error: {e}]"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(req: ChatReq, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)

    model = resolve_model(req.model)
    max_tokens = int(req.max_completion_tokens or req.max_tokens or 512)
    temperature = float(req.temperature if req.temperature is not None else 0.3)

    user_q = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_q = _text(m)
            break

    context_blocks, do_search, sources = build_context(req, user_q, req.model)
    ollama_messages = build_ollama_messages(req, context_blocks)

    meta = {
        "version": VERSION,
        "web_search": do_search,
        "context_blocks": len(context_blocks),
        "sources": sources[:8],
        "model": model,
        "num_ctx": NUM_CTX,
        "mode": "search" if do_search else "fast",
    }

    if req.stream:
        return StreamingResponse(
            _sse_from_ollama(ollama_messages, model, max_tokens, temperature, meta),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    try:
        r = httpx.post(
            f"{OLLAMA}/api/chat",
            json={
                "model": model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": NUM_CTX,
                },
            },
            timeout=180.0,
        )
        r.raise_for_status()
        raw = r.json()
        content = (raw.get("message") or {}).get("content") or ""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": raw.get("prompt_eval_count", 0),
                "completion_tokens": raw.get("eval_count", 0),
                "total_tokens": (raw.get("prompt_eval_count") or 0) + (raw.get("eval_count") or 0),
            },
            "gha_v4": meta,
        }
    except Exception as e:
        raise HTTPException(502, f"ollama error: {e}")


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
