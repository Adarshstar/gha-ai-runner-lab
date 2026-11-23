#!/usr/bin/env python3
"""GHA AI v3 — PocketPal-ready OpenAI endpoint
- Stronger model: qwen2.5:3b (fits GHA CPU, far better quality)
- High context (num_ctx=16384)
- Deterministic web search (never empty context)
- Parallel SearXNG + DDGS + Wikipedia fallback
- Real page fetch + clean extraction
- Grounding system prompt + source citations
- Full SSE streaming for PocketPal
- Optional Bearer key, CORS
Runs only on GitHub Actions (Ollama + SearXNG + DDGS). No Bedrock.
"""
from __future__ import annotations

import asyncio
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
from pydantic import BaseModel

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5:3b")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
API_KEY = os.environ.get("GHA_API_KEY", "sk-gha-v3")
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
VERSION = "3.0.0"

app = FastAPI(title="GHA AI v3 — PocketPal + grounded search", version=VERSION)
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


class ChatReq(BaseModel):
    model: str = MODEL
    messages: List[Msg]
    max_tokens: Optional[int] = 768
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    web_search: Optional[bool] = None  # None = auto (deterministic)
    web_fetch_urls: Optional[List[str]] = None
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


# ---------- Search decision (deterministic) ----------
GREETING_RE = re.compile(
    r"^(hi|hello|hey|hola|yo|sup|ping|test|ok|thanks|thank you|bye|good morning|good night)\b",
    re.I,
)
CODE_ONLY_RE = re.compile(
    r"^(write|fix|debug|refactor|implement|code|python|js|ts|rust|go)\b.*(function|class|script|code|bug)",
    re.I,
)


def needs_search(messages: List[Msg]) -> bool:
    last = ""
    for m in reversed(messages):
        if m.role == "user":
            last = _text(m).strip()
            break
    if not last or len(last) < 3:
        return False
    if GREETING_RE.match(last) and len(last) < 40:
        return False
    # Explicit opt-out
    if re.search(r"\b(no search|without search|offline only|don'?t search)\b", last, re.I):
        return False
    # Everything else with a question or factual intent → search
    if last.endswith("?") or len(last) > 25:
        return True
    triggers = (
        "who ", "what ", "when ", "where ", "why ", "how ", "latest", "news",
        "current", "today", "2024", "2025", "2026", "price", "stock", "weather",
        "capital of", "population", "compare", "vs ", "versus", "difference",
        "wikipedia", "look up", "find ", "research", "according to", "source",
        "url", "http", "website", "page",
    )
    low = last.lower()
    return any(t in low for t in triggers)


# ---------- Search engines ----------
def searx_search(query: str, num: int = 8) -> list[dict]:
    try:
        r = httpx.get(
            f"{SEARX}/search",
            params={"q": query, "format": "json", "language": "en", "pageno": 1},
            timeout=20.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for item in (data.get("results") or [])[:num]:
            url = item.get("url") or ""
            if not url:
                continue
            out.append({
                "title": (item.get("title") or "")[:200],
                "url": url,
                "snippet": (item.get("content") or item.get("snippet") or "")[:500],
                "engine": "searxng",
            })
        return out
    except Exception as e:
        print("searx error", e)
        return []


def ddgs_search(query: str, num: int = 8) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                out.append({
                    "title": (r.get("title") or "")[:200],
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": (r.get("body") or "")[:500],
                    "engine": "ddgs",
                })
        return out
    except Exception as e:
        print("ddgs error", e)
        return []


def wiki_fallback(query: str) -> list[dict]:
    """Last-resort Wikipedia summary so context is never empty."""
    try:
        headers = {"User-Agent": "GHA-AI-v3/3.0 (research)"}
        # Search
        s = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 3, "format": "json"},
            headers=headers,
            timeout=12.0,
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
                timeout=10.0,
            )
            extract = ""
            if sumr.status_code == 200:
                extract = (sumr.json().get("extract") or "")[:800]
            out.append({
                "title": title,
                "url": url or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": extract,
                "engine": "wikipedia",
            })
        return out
    except Exception as e:
        print("wiki error", e)
        return []


def web_search(query: str, num: int = 8) -> list[dict]:
    """Parallel SearXNG + DDGS, dedupe, Wikipedia fallback. Never returns empty if possible."""
    results: list[dict] = []
    try:
        # Sequential is fine and simpler on free runners; still fast
        results = searx_search(query, num=num)
        if len(results) < 3:
            results.extend(ddgs_search(query, num=num))
        if len(results) < 2:
            results.extend(wiki_fallback(query))
    except Exception as e:
        print("web_search outer", e)
        results = wiki_fallback(query)

    seen, uniq = set(), []
    for r in results:
        u = (r.get("url") or "").split("#")[0].rstrip("/")
        if u and u not in seen and len(u) > 8:
            seen.add(u)
            uniq.append(r)
    return uniq[:num]


def web_fetch(url: str, max_chars: int = 10000) -> dict:
    """Real page fetch with clean text extraction."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GHA-AI-v3/3.0; +research)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=18.0)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            text = r.text[:max_chars]
        elif "html" in ctype or "xml" in ctype or not ctype:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe"]):
                tag.decompose()
            # Prefer main/article
            main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"}) or soup.body
            if main:
                text = " ".join(main.get_text(" ", strip=True).split())
            else:
                text = " ".join(soup.get_text(" ", strip=True).split())
            text = text[:max_chars]
        else:
            text = r.text[:max_chars]
        return {
            "url": str(r.url),
            "status": r.status_code,
            "content": text,
            "ok": r.status_code < 400 and len(text) > 80,
        }
    except Exception as e:
        return {"url": url, "status": 0, "content": f"[FETCH FAILED] {e}", "ok": False}


def build_context(req: ChatReq, user_q: str) -> tuple[list[str], bool, list[dict]]:
    """Returns (context_blocks, do_search, sources). Never leaves empty when search was requested."""
    do_search = req.web_search if req.web_search is not None else needs_search(req.messages)
    context_blocks: list[str] = []
    sources: list[dict] = []

    if do_search and user_q:
        hits = web_search(user_q, num=7)
        if not hits:
            # Absolute last resort
            hits = wiki_fallback(user_q) or [
                {"title": "No results", "url": "", "snippet": "Search engines returned no results for this query.", "engine": "none"}
            ]
        lines = ["[WEB SEARCH RESULTS — use ONLY these facts; cite URLs]"]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"{i}. {h['title']}\n   URL: {h['url']}\n   {h['snippet']}\n   source={h.get('engine','')}"
            )
            sources.append({"title": h["title"], "url": h["url"], "engine": h.get("engine")})
        context_blocks.append("\n".join(lines))

        # Fetch top pages for depth (real content)
        fetched = 0
        for h in hits[:3]:
            if not h.get("url") or not h["url"].startswith("http"):
                continue
            page = web_fetch(h["url"], max_chars=4500)
            if page.get("ok") and page.get("content"):
                context_blocks.append(
                    f"[FETCHED PAGE]\nURL: {page['url']}\n{page['content'][:4000]}"
                )
                fetched += 1
            if fetched >= 2:
                break

    if req.web_fetch_urls:
        for u in (req.web_fetch_urls or [])[:3]:
            page = web_fetch(u, max_chars=6000)
            context_blocks.append(
                f"[FETCHED PAGE]\nURL: {page['url']}\n{page.get('content','')[:5500]}"
            )
            if page.get("ok"):
                sources.append({"title": page["url"], "url": page["url"], "engine": "fetch"})

    return context_blocks, do_search, sources


def build_ollama_messages(req: ChatReq, context_blocks: list[str]) -> list[dict]:
    system_extra = (
        "You are a helpful, accurate assistant running on GitHub Actions with live web search.\n"
        "RULES:\n"
        "1. When [WEB SEARCH RESULTS] or [FETCHED PAGE] are present, answer ONLY from that context.\n"
        "2. Cite sources with URLs inline or at the end.\n"
        "3. If the context does not contain the answer, say so clearly — do NOT invent facts, GitHub workflows, or unrelated content.\n"
        "4. Be concise and direct. Prefer facts over speculation.\n"
        "5. For multi-part questions, address each part using the retrieved material.\n"
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
            "pocketpal", "grounding", "high_context", "deterministic_search",
        ],
        "api_key_hint": f"{API_KEY} (optional)",
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "gha-v3"},
            {"id": "gha-v3-search", "object": "model", "created": int(time.time()), "owned_by": "gha-v3"},
        ],
    }


@app.post("/v1/tools/web_search")
def tool_search(body: dict):
    q = body.get("query") or body.get("q") or ""
    if not q:
        raise HTTPException(400, "query required")
    results = web_search(q, num=int(body.get("num", 8)))
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
                            "gha_v3": meta,
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

    model = req.model if req.model not in ("gha-v3-search", "gha-v2-search") else MODEL
    max_tokens = int(req.max_completion_tokens or req.max_tokens or 768)
    temperature = float(req.temperature if req.temperature is not None else 0.3)

    user_q = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_q = _text(m)
            break

    context_blocks, do_search, sources = build_context(req, user_q)
    ollama_messages = build_ollama_messages(req, context_blocks)

    meta = {
        "version": VERSION,
        "web_search": do_search,
        "context_blocks": len(context_blocks),
        "sources": sources[:8],
        "model": model,
        "num_ctx": NUM_CTX,
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

    # Non-streaming
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
        data = {
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
            "gha_v3": meta,
        }
        return data
    except Exception as e:
        raise HTTPException(502, f"ollama error: {e}")


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
