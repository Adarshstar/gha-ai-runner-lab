#!/usr/bin/env python3
"""GHA v2 gateway — OpenAI-compatible + local web search/fetch.
Fully compatible with PocketPal Android remote server:
  - GET  /v1/models
  - POST /v1/chat/completions  (stream=true → SSE, stream=false → JSON)
  - Optional Bearer API key (sk-gha-v2)
  - CORS *
  - Auto web_search + page fetch when query needs it
Runs only on GitHub Actions runner (Ollama + SearXNG + DDGS). No Bedrock.
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
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5:0.5b")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
API_KEY = os.environ.get("GHA_API_KEY", "sk-gha-v2")  # optional; empty = no auth

app = FastAPI(title="GHA AI v2 — PocketPal ready", version="2.1.0")

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
    max_tokens: Optional[int] = 512
    max_completion_tokens: Optional[int] = None  # OpenAI newer name
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    web_search: Optional[bool] = None  # None = auto-detect
    web_fetch_urls: Optional[List[str]] = None
    stop: Optional[Any] = None


def _check_auth(authorization: Optional[str]) -> None:
    """Optional Bearer key. If GHA_API_KEY is set, require it (or allow missing for probe)."""
    if not API_KEY:
        return
    if not authorization:
        # PocketPal models probe and some clients send no key on first contact
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


def needs_search(messages: List[Msg]) -> bool:
    last = ""
    for m in reversed(messages):
        if m.role == "user":
            last = _text(m).lower()
            break
    if not last:
        return False
    if re.match(r"^(hi|hello|hey|ping|test|ok|thanks|thank you)\b", last):
        return False
    triggers = (
        "search", "who is", "what is", "when did", "latest", "news", "current",
        "today", "2024", "2025", "2026", "price", "stock", "weather", "how many",
        "wikipedia", "look up", "find", "research", "according to", "web",
        "capital of", "president", "ceo of",
    )
    return any(t in last for t in triggers) or last.endswith("?")


def searx_search(query: str, num: int = 5) -> list[dict]:
    try:
        r = httpx.get(
            f"{SEARX}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=25.0,
        )
        if r.status_code == 200:
            data = r.json()
            out = []
            for item in (data.get("results") or [])[:num]:
                out.append({
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "snippet": (item.get("content") or item.get("snippet") or "")[:400],
                    "engine": "searxng",
                })
            if out:
                return out
    except Exception as e:
        print("searx error", e)
    return []


def ddgs_search(query: str, num: int = 5) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                out.append({
                    "title": r.get("title") or "",
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": (r.get("body") or "")[:400],
                    "engine": "ddgs",
                })
        return out
    except Exception as e:
        print("ddgs error", e)
        return []


def web_search(query: str, num: int = 6) -> list[dict]:
    results = searx_search(query, num=num)
    if len(results) < 2:
        results = ddgs_search(query, num=num) or results
    seen, uniq = set(), []
    for r in results:
        u = r.get("url") or ""
        if u and u not in seen:
            seen.add(u)
            uniq.append(r)
    return uniq[:num]


def web_fetch(url: str, max_chars: int = 12000) -> dict:
    try:
        headers = {
            "User-Agent": "GHA-AI-v2/2.1 (+PocketPal; research)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
        }
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=20.0)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            text = r.text[:max_chars]
        elif "html" in ctype or "xml" in ctype or not ctype:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())
            text = text[:max_chars]
        else:
            text = r.text[:max_chars]
        return {"url": url, "status": r.status_code, "content": text, "ok": r.status_code < 400}
    except Exception as e:
        return {"url": url, "status": 0, "content": f"[FETCH FAILED] {e}", "ok": False}


def build_context(req: ChatReq, user_q: str) -> tuple[list[str], bool]:
    do_search = req.web_search if req.web_search is not None else needs_search(req.messages)
    context_blocks: list[str] = []
    if do_search and user_q:
        hits = web_search(user_q, num=5)
        if hits:
            lines = ["[WEB SEARCH RESULTS]"]
            for i, h in enumerate(hits, 1):
                lines.append(
                    f"{i}. {h['title']}\n   URL: {h['url']}\n   {h['snippet']}\n   source={h['engine']}"
                )
            context_blocks.append("\n".join(lines))
            for h in hits[:2]:
                if h.get("url"):
                    page = web_fetch(h["url"], max_chars=4000)
                    if page.get("ok") and page.get("content"):
                        context_blocks.append(
                            f"[FETCHED PAGE]\nURL: {page['url']}\n{page['content'][:3500]}"
                        )
    if req.web_fetch_urls:
        for u in req.web_fetch_urls[:3]:
            page = web_fetch(u, max_chars=5000)
            context_blocks.append(
                f"[FETCHED PAGE]\nURL: {page['url']}\n{page.get('content', '')[:4000]}"
            )
    return context_blocks, do_search


def build_ollama_messages(req: ChatReq, context_blocks: list[str]) -> list[dict]:
    system_extra = (
        "You are a helpful assistant running on GitHub Actions with local web search. "
        "When search/page context is provided, use it and cite URLs. Be concise and accurate."
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


@app.get("/")
@app.get("/health")
def health():
    ollama_ok = False
    searx_ok = False
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
        "version": "2.1.0",
        "ollama": ollama_ok,
        "searxng": searx_ok,
        "model": MODEL,
        "features": ["web_search", "web_fetch", "openai_chat", "sse_stream", "pocketpal"],
        "api_key_hint": "sk-gha-v2 (optional)",
    }


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gha-v2",
            },
            {
                "id": "gha-v2-search",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gha-v2",
            },
        ],
    }


@app.post("/v1/tools/web_search")
def tool_search(body: dict, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    q = body.get("query") or body.get("q") or ""
    if not q:
        raise HTTPException(400, "query required")
    return {"query": q, "results": web_search(q, num=int(body.get("num", 6)))}


@app.post("/v1/tools/web_fetch")
def tool_fetch(body: dict, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    url = body.get("url") or ""
    if not url:
        raise HTTPException(400, "url required")
    return web_fetch(url, max_chars=int(body.get("max_chars", 12000)))


async def _sse_from_ollama(
    ollama_messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream Ollama /api/chat and convert to OpenAI SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", f"{OLLAMA}/api/chat", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': err.decode()[:300]})}\n\n"
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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if done:
                        data = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
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
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"[stream error: {e}]"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(
    req: ChatReq,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    model = req.model if req.model not in ("gha-v2-search",) else MODEL
    max_tokens = int(
        req.max_completion_tokens
        or req.max_tokens
        or 512
    )
    temperature = float(req.temperature if req.temperature is not None else 0.2)

    user_q = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_q = _text(m)
            break

    context_blocks, do_search = build_context(req, user_q)
    ollama_messages = build_ollama_messages(req, context_blocks)

    # ---- STREAMING (PocketPal uses this) ----
    if req.stream:
        return StreamingResponse(
            _sse_from_ollama(ollama_messages, model, max_tokens, temperature),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # ---- NON-STREAMING ----
    try:
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": model,
                "messages": ollama_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # fallback to native /api/chat
        try:
            r2 = httpx.post(
                f"{OLLAMA}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120.0,
            )
            r2.raise_for_status()
            raw = r2.json()
            content = (raw.get("message") or {}).get("content") or ""
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        except Exception as e2:
            raise HTTPException(502, f"ollama error: {e} | fallback: {e2}")

    if do_search:
        data.setdefault("gha_v2", {})
        data["gha_v2"]["web_search"] = True
        data["gha_v2"]["context_blocks"] = len(context_blocks)
    return data


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
