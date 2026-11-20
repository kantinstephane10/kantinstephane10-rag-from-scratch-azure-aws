# AI Product Builder V1.1 — Reference Dashboard + In-app OpenAI Setup

A local **AI Agent / Agentic AI / RAG Product Builder control-plane prototype** with 49 specialist roles and nine organization layers. V1.3 adds an executable document RAG pathway and project deletion while preserving the reference-style dashboard and browser-managed OpenAI model routing.

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
5. Switch to **Live OpenAI GPT** and save the mode and routing. V1.2+ removed application-level token and call ceilings; OpenAI billing and provider limits still apply. Without a key, switching to live is blocked. API calls can incur separate charges.
6. In **Projects**, describe the AI Agent / Agentic AI / RAG project, upload reference documents (up to 8 MiB each), check OpenAI processing consent, leave **Start the Product Manager agent upon creation** checked, and select **Validate & start project**.

Use **Meeting studio** to watch Product Manager-chaired agent conversations, take the floor, and ask specialists direct questions. The Product Manager can convene agent-only meetings without your participation, subject to meeting quotas. Select tasks to cancel individually or in batches and supply a required reason. Cancelling prevents late results from being published but may not cancel an API call already in flight.

## Usage tracking (no local token caps)

The project records requests and OpenAI-reported tokens for observability, without an application token/call ceiling. Provider quotas, model context limits and API costs still apply.

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


## V1.2 — Automatic advisory work and uncapped GPT usage

- Removed locally configured output, per-project GPT call, and token caps. GPT requests no longer send `max_output_tokens`; provider-side limits and billing still apply. Existing persisted cap keys are ignored.
- Usage counters remain visible but no longer block subsequent GPT work.
- Already-queued advisory planning tasks continue even when an agent requested human review. Human approval for protected releases remains mandatory.
- Queue processing is sequential and respects mandatory verified handoffs; a task may still block on invalid handoffs, provider errors, cancellation, or missing consent.


## V1.3 — Document RAG and project deletion

This release implements a **local, project-scoped** RAG execution path, rather than merely asking LLM workers to describe one. It supports readable PDF text extraction (PyMuPDF), LangChain recursive character splitting, OpenAI embeddings (`text-embedding-3-small`), an atomic SQLite cosine-similarity index, citation-linked answer retrieval, integrity checks, background indexing, and persistent ingestion receipts. No OCR, per-user document ACL enforcement, quality benchmark, sandboxed code execution, external deployment or production readiness is claimed.

### Windows setup

Run PowerShell from the `ai-product-builder-v1` folder with the same working interpreter used for the server:

```powershell
& "$HOME\anaconda3\python.exe" -m pip install -r requirements-rag.txt
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Open `http://127.0.0.1:8767/`, then in **OpenAI & model settings** save an OpenAI key, select available models, and set **Live OpenAI GPT**. In **Projects**, create a **RAG assistant** project, enable document-processing consent, upload an unlocked, text-based PDF or supported text files (8 MiB maximum per file), then start the Product Manager. Document indexing starts as a background task, or select **Build / rebuild searchable index** in the project panel. Refresh status until the job is complete, then ask a question in **Document RAG execution**.

The index is invalidated when indexed document sets change, and downstream answers fail closed until a rebuild. Ingestion receipts document real extraction/embedding checks, but do **not** imply successful retrieval quality or production authorization.

**Delete project** opens a confirmation dialog requiring the exact project name. This is irreversible: it cancels its work and meetings and removes project-scoped files, handoffs, index and database records while preserving unrelated projects. In-flight paid OpenAI requests cannot necessarily be cancelled or refunded.

### Validation scope

82 automated tests passed in this build, including actual PDF extraction with an installed PyMuPDF interpreter and mocked embedding/answer services, integrity, indexing failures, stale index detection, deletion isolation and HTTP deletion endpoints, and invalid-meeting-question resilience. JavaScript syntax and Python compilation checks also passed. **No paid OpenAI model calls or actual Windows DPAPI execution were run as part of these tests.**
