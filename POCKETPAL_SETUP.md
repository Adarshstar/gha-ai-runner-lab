# PocketPal Android ↔ GHA v2 OpenAI endpoint

This endpoint runs **entirely on GitHub Actions** (Ollama + SearXNG + DDGS web search).
No Bedrock. No EC2. Auto web search + page fetch for factual questions.

## Current live URL

Check the `live-endpoint-v2` branch files:

- `LIVE_OPENAI_BASE_V2.txt` → base URL ending in `/v1`
- `LIVE_API_KEY_V2.txt` → `sk-gha-v2`
- `LIVE_MODEL_V2.txt` → `qwen2.5:0.5b`

Or open:
https://github.com/Adarshstar/gha-ai-runner-lab/blob/live-endpoint-v2/LIVE_ENDPOINT_V2.md

## PocketPal setup (step by step)

1. Update PocketPal to a build that includes **remote server support** (PR #597 / releases after that).
2. Open **Settings** → **Servers** / **Remote models** (or the server sheet).
3. **Add server**:
   - **Name:** `GHA v2 Search`
   - **Base URL:** `https://xxxxx.loca.lt/v1`  
     (exact value from `LIVE_OPENAI_BASE_V2.txt` — must end with `/v1`)
   - **API Key:** `sk-gha-v2` (optional but recommended)
4. Tap **Test connection**.  
   PocketPal calls `GET /v1/models` — you should see `qwen2.5:0.5b` and `gha-v2-search`.
5. Select the remote model in the model picker (cloud icon).
6. Chat normally. Streaming uses SSE (`stream=true`).

### Web search behaviour

- Questions that look factual / “what is / latest / 2026 / capital of …” automatically trigger:
  1. Local SearXNG (+ DDGS fallback)
  2. Fetch of top 1–2 result pages
  3. Context injected into the system prompt before the model answers
- You can also force search by including words like “search”, “look up”, “research”.

## Manual curl tests (from phone or PC)

```bash
BASE="https://YOUR-SUBDOMAIN.loca.lt"
KEY="sk-gha-v2"

# models probe (what PocketPal does first)
curl -s -H "Bypass-Tunnel-Reminder: 1" -H "Authorization: Bearer $KEY" "$BASE/v1/models"

# non-stream
curl -s -H "Bypass-Tunnel-Reminder: 1" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hi"}],"stream":false}' \
  "$BASE/v1/chat/completions"

# SSE stream (PocketPal path)
curl -N -H "Bypass-Tunnel-Reminder: 1" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -H "Accept: text/event-stream" \
  -d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Count 1 to 3"}],"stream":true}' \
  "$BASE/v1/chat/completions"
```

## Notes

- **localtunnel** may show a browser reminder page. PocketPal and curl with header `Bypass-Tunnel-Reminder: 1` skip it. If the app still fails on first connect, open the base URL once in the phone browser and click through, then retry in PocketPal.
- Job default hold = **3 hours** (max 5). After that the URL dies until you re-trigger the workflow.
- To re-trigger: push any change to `TRIGGER_V2` or run **openai-endpoint-v2** via Actions → workflow_dispatch.

## Compatibility checklist (implemented)

| Feature                         | Status |
|--------------------------------|--------|
| `GET /v1/models`               | ✅     |
| `POST /v1/chat/completions`    | ✅     |
| SSE streaming (`stream=true`)  | ✅     |
| Optional Bearer API key        | ✅     |
| CORS `*`                       | ✅     |
| Auto web search + page fetch   | ✅     |
| OpenAI chunk format            | ✅     |
| `data: [DONE]` terminator      | ✅     |
