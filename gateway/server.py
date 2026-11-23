#!/usr/bin/env python3
"""GHA v2 gateway: OpenAI-compatible chat + local web_search + web_fetch.
Runs entirely on the Actions runner (Ollama + SearXNG + DDGS). No Bedrock.
"""
from __future__ import annotations
import json, os, re, time, uuid
from typing import Any, List, Optional
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5:0.5b")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))

app = FastAPI(title="GHA AI v2 — search + fetch", version="2.0.0")


class Msg(BaseModel):
    role: str
    content: Any = ""


class ChatReq(BaseModel):
    model: str = MODEL
    messages: List[Msg]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    # v2 extras
    web_search: Optional[bool] = None  # None = auto
    web_fetch_urls: Optional[List[str]] = None


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
    # skip pure coding/hello
    if re.match(r"^(hi|hello|hey|ping|test)\b", last):
        return False
    triggers = (
        "search", "who is", "what is", "when did", "latest", "news", "current",
        "today", "2024", "2025", "2026", "price", "stock", "weather", "how many",
        "wikipedia", "look up", "find", "research", "according to", "web",
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
        from ddgs import DDGS
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
    # dedupe by url
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
            "User-Agent": "GHA-AI-v2/2.0 (+research; self-hosted)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
        }
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=20.0)
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype:
            text = r.text[:max_chars]
        elif "html" in ctype or "xml" in ctype or not ctype:
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())
            text = text[:max_chars]
        else:
            text = r.text[:max_chars]
        return {"url": url, "status": r.status_code, "content": text, "ok": r.status_code < 400}
    except Exception as e:
        return {"url": url, "status": 0, "content": f"[FETCH FAILED] {e}", "ok": False}


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
        "version": "2.0.0",
        "ollama": ollama_ok,
        "searxng": searx_ok,
        "model": MODEL,
        "features": ["web_search", "web_fetch", "openai_chat"],
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL, "object": "model", "owned_by": "gha-v2"},
            {"id": "gha-v2-search", "object": "model", "owned_by": "gha-v2"},
        ],
    }


@app.post("/v1/tools/web_search")
def tool_search(body: dict):
    q = body.get("query") or body.get("q") or ""
    if not q:
        raise HTTPException(400, "query required")
    return {"query": q, "results": web_search(q, num=int(body.get("num", 6)))}


@app.post("/v1/tools/web_fetch")
def tool_fetch(body: dict):
    url = body.get("url") or ""
    if not url:
        raise HTTPException(400, "url required")
    return web_fetch(url, max_chars=int(body.get("max_chars", 12000)))


@app.post("/v1/chat/completions")
def chat(req: ChatReq, authorization: Optional[str] = Header(None)):
    model = req.model if req.model not in ("gha-v2-search",) else MODEL
    user_q = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_q = _text(m)
            break

    do_search = req.web_search if req.web_search is not None else needs_search(req.messages)
    context_blocks = []

    if do_search and user_q:
        hits = web_search(user_q, num=5)
        if hits:
            lines = ["[WEB SEARCH RESULTS]"]
            for i, h in enumerate(hits, 1):
                lines.append(f"{i}. {h['title']}\n   URL: {h['url']}\n   {h['snippet']}\n   source={h['engine']}")
            context_blocks.append("\n".join(lines))
            # fetch top 1-2 pages for depth
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
            context_blocks.append(f"[FETCHED PAGE]\nURL: {page['url']}\n{page.get('content','')[:4000]}")

    ollama_messages = []
    system_extra = (
        "You are a helpful assistant running on GitHub Actions with local web search. "
        "When search/page context is provided, use it and cite URLs. Be concise and accurate."
    )
    if context_blocks:
        system_extra += "\n\n" + "\n\n".join(context_blocks)

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

    payload = {
        "model": model if model != "gha-v2-search" else MODEL,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": float(req.temperature if req.temperature is not None else 0.2),
            "num_predict": int(req.max_tokens or 512),
        },
    }
    try:
        r = httpx.post(f"{OLLAMA}/v1/chat/completions", json={
            "model": payload["model"],
            "messages": ollama_messages,
            "max_tokens": int(req.max_tokens or 512),
            "temperature": float(req.temperature if req.temperature is not None else 0.2),
        }, timeout=120.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"ollama error: {e}")

    # annotate that search was used
    if do_search:
        data.setdefault("gha_v2", {})
        data["gha_v2"]["web_search"] = True
        data["gha_v2"]["context_blocks"] = len(context_blocks)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
