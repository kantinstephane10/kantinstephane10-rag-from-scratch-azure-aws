# AI Product Builder V1.5 — Website Builder, Local RAG & Runtime Audit

A local **AI Agent / Agentic AI / RAG Product Builder control-plane prototype** with 49 specialist roles and nine organization layers. V1.3 adds an executable document RAG pathway and project deletion while preserving the reference-style dashboard and browser-managed OpenAI model routing.

**Scope:** This app manages project intake, planning via OpenAI, specialist messaging, documented handoffs, meetings, cancellation, artifact uploads and review. It is **not** a finished autonomous code-writing/testing/deployment factory. An `agent replied` receipt is not proof of source files built, tests executed, LangChain/LangGraph installed, or a production RAG deployed. Do not authorize a release based only on generated text.


## V1.6.1 — Unattended Python on the Windows host

**Agent-generated Python now runs automatically with no per-script approval or project-name confirmation.** The local worker persists execution requests in SQLite, compiles newly written Python, runs entrypoints (`app.py`, `main.py`, `server.py`, `run.py` or files with `__main__`), runs discovered unittest suites for new tests, and executes additional project-local `.py` files / `unittest`, `pytest`, `compile` when agents explicitly request them. Other source modules are compiled and exercised when imported by entrypoints or tests; they are not incorrectly launched as standalone applications.

- Each workspace has **automatic execution ON** by default. Use **Pause automatic Python**, **Resume automatic Python**, or **Stop running / cancel queued jobs** in the Project code workspace. These are operational controls, not approval prompts.
- The server starts a dedicated auto-run worker and resumes pending jobs after restart. Each job is tied to a source-content fingerprint, preventing later unreviewed revisions being executed under old receipts. Duplicate revision/command pairs are not rerun; results record exit codes and terminal statuses.
- Windows Python runs through the same interpreter used to launch `server.py`. Runtime subprocesses are stopped on cancellation or timeout, output is bounded, and the child environment omits inherited API keys. **This does not isolate files, OS credentials, process execution, or outbound network**. Scripts can still access anything the Windows account can.
- The app remains localhost-only for Windows runs. In opt-in multiuser mode, an administrator must control auto-execution and workspace file writes when auto-execution is on, to prevent an ordinary editor from escalating privileges. **There is no per-script permission prompt.**
- This is an explicitly unsafe/trusted-code operating mode. Use a dedicated, non-admin Windows account and a machine without sensitive files or credentials. Uploaded documents can contain prompt injection that alters generated code. Do not expose this application to the internet.
- Approval for **public releases and protected deployment operations** is separate and remains required.

Run tests: `python -m unittest discover -s tests -v`. Windows-native execution and Windows subprocess-tree termination still require verification on the actual Windows host; Linux tests do not certify it.


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

Open **http://127.0.0.1:8767/** by typing it directly into Firefox or Edge. Keep PowerShell open. The dashboard and static website builder use Python's standard library; local RAG requires packages from `requirements-rag.txt`. No Ollama is used.

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

## V1.3.1 — reliable multi-agent relay recovery

- Fixed the `Agent relay limit reached; no further peer messages allowed.` failure: a worker's valid written result is now saved even when its optional delegation is suppressed. The event is recorded and disclosed in the worker's response.
- Expanded the bounded agent-to-agent chain from 4 to 12 hops. This is a **workflow safety guard, not a GPT token cap**. At the final hop, further requests are dropped safely and the worker explains any follow-up needed; no infinite agent loop is permitted.
- On startup, legacy failed messages carrying the exact old relay-limit error are requeued once automatically, including messages in projects awaiting human review. Retrying an old paid GPT request can incur another charge.
- The Live Agent Activity pane now has a **Retry failed task** button for other failures. The retry API verifies that the task belongs to the selected project.
- Invalid optional delegation and meeting requests no longer discard an otherwise valid worker answer. Supported release and human-approval gates are unchanged.

From an existing `ai-product-builder-v1` directory on Windows:

```powershell
git fetch origin refs/heads/feat/ai-product-builder-v131-relay-recovery
git switch -c feat/ai-product-builder-v131-relay-recovery FETCH_HEAD
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

This package does **not** promise autonomous production code generation, CI/CD, or deployment. For a real RAG artifact, use the Document RAG execution panel and inspect its recorded build/index/test evidence.


## Mandatory Business Analyst participation (V1.3.2)

Every **newly kicked-off project** schedules Business Analyst 1 and Business Analyst 2 before the Product Manager's kickoff. The Product Manager cannot begin until both initial reviews finish or terminate; the PM receives both recorded assessments and must flag any failed review. These three assignments are inserted atomically and persist across restart. No extra agent relay-depth or token caps are imposed. Existing project histories are not rewritten automatically.

## V1.3.3 update — large files and per-agent file reading

See [RELEASE_NOTES_V1.3.3.md](RELEASE_NOTES_V1.3.3.md). Highlights: 1 GiB streamed uploads, project/meeting file attachments, real PDF/TXT source access for all 49 registered agents, short kickoff assignments for long project briefs, and **Start agents** for an existing draft. All files remain project/meeting-scoped. A 1 GiB upload does not imply full 1 GiB semantic indexing; the existing index limits still apply.


## V1.4.1 — local free Hugging Face embeddings

- Uses local `sentence-transformers/all-MiniLM-L6-v2` by default (384-dimensional embedding vectors). Optionally select `BAAI/bge-small-en-v1.5` from **OpenAI & model settings**. Both are free to run locally and the first use downloads model weights from Hugging Face. Model downloads, CPU/RAM and disk usage still apply.
- Incremental indexing in **Projects → Document RAG execution** uses the V2 disk-backed SQLite + FTS5 index, including optional RapidOCR scanned-PDF processing (`pip install -r requirements-ocr.txt`). Indices are per project and saved; Resume and Cancel controls are available. Indexing does not send document embeddings to OpenAI and does not need an OpenAI API key.
- **Answer generation** still needs a valid OpenAI API key, Live OpenAI mode, and explicit per-project consent. Hugging Face embeddings are separate from OpenAI GPT. No Ollama.
- Configure the embedding model and OCR mode through **OpenAI & model settings**. **Rebuild the index after changing embedding models.** The selected model is persisted per-index so queries use a matching model.
- Python installation on Windows: from `ai-product-builder-v1`, `& "$HOME\anaconda3\python.exe" -m pip install -r requirements-rag.txt`; optionally install `requirements-ocr.txt`, and run `server.py --port 8767`. Large indexing may use substantial disk and CPU.
- `--demo` still simulates agent messages; indexing is real and local even in demo mode. **Do not report production deployment, LangGraph code generation, full 1 GiB benchmark, ACL certification, or live provider verification as passed:** none of those is proven by this release.


## V1.5 — Website Builder & honest production prerequisite audit

**New website product type.** Select **Website / landing page** when creating a project. A responsive, mobile-ready multi-page website is generated immediately, *without a GPT call or token usage*. In the Website Builder project panel you can edit the site name/tagline, choose **Business**, **Portfolio**, or **Documentation**, select a color theme, rebuild, preview Home/About/Contact safely inside a script-disabled sandbox, and export all files in a ZIP. Generated HTML/CSS/metadata/README are project-scoped, checksummed and versioned. The previous site remains intact if a rebuild fails. Project deletion also removes the website directory. Business Analysts and Product Manager kickoff still work for website projects when explicitly enabled with provider consent.

The Website Builder produces real static HTML/CSS source files, **not arbitrary GPT-authored code**, a dynamic backend, a deployed website, hosted forms, user auth, or custom domain. A generated landing page is not an agent-generated and independently validated production web application. External hosting and any protected deployment require operator approval.

**Runtime audit.** Select **Settings → Runtime and production prerequisite audit → Run audit**. This reports actual Python module versions, presence of Git/Node/Tesseract and free disk, plus missing enterprise controls. A successful local run does **not** set `production_ready` true. For RAG install `requirements-rag.txt`; optionally install `requirements-ocr.txt`; additional optional LangChain/LangGraph/Chroma libraries are declared in `requirements-agent-optional.txt` (installing them does not create an agent command runner).

**Windows PowerShell — existing checkout:**

```powershell
& "$HOME\anaconda3\python.exe" -m pip install -r requirements-rag.txt
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Open `http://127.0.0.1:8767/`. The `run_windows.ps1` launcher now accepts API keys configured in the UI without requiring an environment variable. Python 3.11–3.13 is the recommended tested ecosystem target; test external libraries carefully if your Anaconda interpreter is Python 3.14.

**Production exclusions.** This local single-operator application still lacks multiuser authentication and fine-grained ACLs, hardened public TLS/auth/rate limiting, a sandboxed arbitrary code runner for all 49 agents, full CI/CD, reliable distributed job queue, restore drills, monitoring/alerting certification, and a measured 1 GiB end-to-end ingestion benchmark. The UI enumerates those gaps. Do **not** expose the built-in `ThreadingHTTPServer` directly to the internet.
