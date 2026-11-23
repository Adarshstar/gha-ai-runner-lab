# PocketPal + GHA AI v3

## Live connection

| Field | Value |
|-------|--------|
| Base URL | (see LIVE_OPENAI_BASE_V3.txt on live-endpoint-v3 branch) |
| API Key | `sk-gha-v3` |
| Model | `qwen2.5:3b` or `gha-v3-search` |

## In PocketPal

1. Settings → Servers / Remote models → Add
2. Base URL must end with `/v1`
3. API Key: `sk-gha-v3`
4. Test connection → select remote model (cloud icon)
5. Chat

## What v3 fixed

- Model upgraded 0.5b → **qwen2.5:3b**
- Context **16k**
- Search is **deterministic** (almost always runs for real questions)
- Context is **never empty** when search runs (Wikipedia fallback)
- Real **web_fetch** of top pages
- Strong grounding + citation instructions
- Cloudflare tunnel (stable for mobile)
- Full SSE for PocketPal streaming
