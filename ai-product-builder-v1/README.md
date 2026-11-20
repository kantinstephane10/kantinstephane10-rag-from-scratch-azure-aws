# AI Product Builder V1.1 — Reference Dashboard + In-app OpenAI Setup

A local **AI Agent / Agentic AI / RAG Product Builder control-plane prototype** with 49 specialist roles and nine organization layers. V1.1 rebuilds the live workflow dashboard around the operator's screenshot and adds browser-based OpenAI API configuration, per-agent model routing, and token-usage limits.

**Scope:** This app manages project intake, planning via OpenAI, specialist messaging, documented handoffs, meetings, cancellation, artifact uploads and review. It is **not** a finished autonomous code-writing/testing/deployment factory. An `agent replied` receipt is not proof of source files built, tests executed, LangChain/LangGraph installed, or a production RAG deployed. Do not authorize a release based only on generated text.

## Run directly from GitHub (Windows PowerShell)

If you already cloned `kantinstephane10/kantinstephane10-rag-from-scratch-azure-aws` using `--single-branch`:

```powershell
# Run in the repository ROOT, not inside the old ai-product-builder folder

git remote set-branches --add origin feat/ai-product-builder-v11-reference-ui-settings
git fetch origin
git switch -c feat/ai-product-builder-v11-reference-ui-settings origin/feat/ai-product-builder-v11-reference-ui-settings
cd ai-product-builder-v1

& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Open **http://127.0.0.1:8767/** by typing it directly into Firefox or Edge. Keep PowerShell open. This application uses Python's standard library only; it does not need Ollama or a separate Streamlit installation.

### Setup entirely in the UI

1. Click the round workflow icon to open the sidebar, then **OpenAI & model settings**. You can also open it from **Projects → Provider API keys**.
2. Paste your own OpenAI API key and choose **Save API key**. The key is never returned in API responses. On Windows, the application encrypts it at rest using Windows DPAPI scoped to the logged-in user. On other platforms it is session-only and not persisted. The DPAPI path must be verified on your actual Windows installation.
3. Click **Find available models**. This asks the OpenAI models API for the models accessible to your key.
4. Configure model IDs for the **Astra** (architecture), **Sol** (engineering), **Terra** (business), and **Luna** (routine) routing groups. *These are labels from the requested design, not proof that a model with the pictured ID exists.* Replace any example IDs with actual API model IDs from your account. Configure overrides for any of the **49 individual workers**, and optionally a meeting-specific model.
5. Set maximum output tokens per response, number of GPT calls per project, and total reported tokens per project. Set mode to **Live OpenAI GPT** and click **Save routing, budgets & mode**. Without a key, switching to live is blocked. API calls can incur separate charges.
6. In **Projects**, describe the AI Agent / Agentic AI / RAG project, upload reference documents (up to 1 MB each), check OpenAI processing consent, leave **Start the Product Manager agent upon creation** checked, and select **Validate & start project**.

Use **Meeting studio** to watch Product Manager-chaired agent conversations, take the floor, and ask specialists direct questions. The Product Manager can convene agent-only meetings without your participation, subject to meeting quotas. Select tasks to cancel individually or in batches and supply a required reason. Cancelling prevents late results from being published but may not cancel an API call already in flight.

## Token-budget enforcement

- The configured `max_output_tokens` value is sent to OpenAI for each Responses API request.
- An atomic SQLite ledger reserves **one request** before a model call; when the call budget is exhausted, subsequent calls fail closed.
- Reported `usage.total_tokens` is recorded after successful responses. When the recorded total exceeds the configured threshold, subsequent requests are blocked. **This is not a strict hard cap on the tokens consumed by a currently running request**, nor a guaranteed dollar-cost cap.
- Model requests with HTTP errors still count toward the request count; token usage for failed requests may be unavailable.
- The usage count appears in the live project activity panel. Demo-mode responses do not consume OpenAI tokens.

## Security and limitations

- Local server binds to `127.0.0.1` and the settings/key endpoints accept **localhost-only** calls; origin checks protect writes from unrelated browser pages. Do **not** expose this raw HTTP server to the internet.
- API keys are never stored in source code, project briefs, artifacts or the shared database, nor echoed by the settings endpoints. The local encrypted credential is in `data/openai-key.dpapi` on Windows; `data/` is Git-ignored.
- Project files/handoffs include checksums, status and evidence markers. Text-only planning is explicitly distinguishable from real executed tests. There is no production-grade sandbox, CI/CD, app generation, public internet-tool broker, LangSmith tracing pipeline, or notification delivery in this version.
- The screenshots' green ticks are **not** shown indiscriminately. A worker tile only becomes green if that role has a completed recorded task. Human approval is a separate operator gate.

## Validation

```powershell
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" -m compileall -q .
```

Node.js is optional for a JS syntax check: `node --check static/app.js`. Offline Python, localhost HTTP, and browser layout tests were performed here; Windows DPAPI and live paid OpenAI inference require verification on the user's Windows computer.
