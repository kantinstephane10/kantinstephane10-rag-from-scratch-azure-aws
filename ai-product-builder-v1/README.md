# AI Product Builder V1.0 — GPT Meeting Studio & Project Workspace

This is a **working local control-plane prototype** for the 9-layer / 49-role AI Agent, Agentic AI and RAG Product Builder. It provides a project brief/estimate form, document ledger, persisted GPT agent messaging and handoffs, project activity, a Product Manager-chaired meeting studio, and cancellation controls. **It is not yet a full code-generating and deploying AI product factory.** Code/tool execution and production deployment require additional work and explicit human approval.

The application uses **OpenAI GPT models only** for live responses; there is no Ollama integration. Demo mode generates synthetic responses and **never calls OpenAI**. GPT API calls may incur charges; a cancellation prevents publishing late results but cannot guarantee cancellation of a network call already made.

## Launch on your Windows machine

Use **PowerShell** in the extracted `ai-product-builder` directory. Your existing `rag-training` Python shortcut was pointing to a missing Python 3.12 installation; avoid using that broken interpreter. Your Anaconda Python 3.14.6 was previously confirmed to work.

```powershell
& "$HOME\anaconda3\python.exe" --version
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --demo --port 8767
```

Open **http://127.0.0.1:8767/** directly. The PowerShell process must remain running. If you prefer, use `./run_windows.ps1 -Demo -Port 8767` after allowing scripts in your session.

For live GPT calls, stop the demo process with `Ctrl+C`. Set your key for the current PowerShell session **without putting it in code or Git**:

```powershell
$key = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Model IDs are configured by role in `config.py`. Your API project must have access to the relevant models. Override environment variables, e.g. `$env:GPT_MODEL_ENGINEERING = 'gpt-6.2-sol'`. No fallback to Ollama is attempted.

## Create your first project

1. Open **Projects** from the dark navigation sidebar or click **Create a project** on the architecture page.
2. Choose **RAG assistant**, **AI agent**, or **Agentic AI system**. Provide a project name and detailed goals/requirements.
3. Optionally attach PDF/text/code/reference files, up to **1 MB per file**. The UI uploads each selected document to the project artifact ledger **before** starting its Product Manager. Uploads are stored locally, SHA-256 hashed and accessible in **Handoffs & artifacts**. Current agents do **not** automatically parse these attachments into a RAG index.
4. Enter cost assumptions, select a GPT model profile, and enable **provider consent** before live GPT execution. Demo mode doesn't contact OpenAI.
5. Leave **Start the Product Manager agent upon creation** checked, then click **Validate & start project**. The project is created, documents are saved, and the Product Manager's initial task is queued.
6. Under **Agent sequence**, track real queued/running/done/failed states. Selecting a worker displays its actual assignment and response. **A completed planning message is not equivalent to code execution or passed tests.**
7. Under **Meeting studio**, create a meeting with any agents. The Product Manager chairs. You can observe, join, ask directed questions, dictate if your browser supports it, or take the floor to prevent ongoing agent responses from being published.

### Cancel meetings and individual/bulk agent tasks (V1.0)

- For a scheduled or live meeting, click **Cancel meeting**. A custom in-app dialog requires a **5–500 character reason**. The transcript and reason persist in SQLite, pending meeting turns are cancelled, and results of prior in-flight turns are suppressed if they arrive later.
- To cancel one worker assignment, select its row, click **Cancel task**, and enter a reason in the custom dialog.
- To cancel several assignments, check the boxes beside eligible tasks, or click **Select all queued**. Click **Cancel selected (N)** and enter a reason. The API only cancels the **explicitly selected** IDs and rejects the entire batch if any ID is invalid, from another project, or no longer cancellable. A separate **Clear selection** control is included.
- **End meeting** uses the same in-app confirmation dialog but is separate from cancellation. Completed or previously cancelled work cannot be cancelled again; cancelled tasks block final approval.
- **A pending API model response may continue billing after cancellation**, even though no late response or new outgoing task is published.

### Meetings, handoffs and safety

- All 49 registered roles can communicate using a shared SQLite-backed mailbox. Product Manager chairs meetings. A directed question queues the addressed agent first.
- Human floor ownership preempts automatic contribution publishing. Agent meetings are capped to avoid uncontrolled conversation loops.
- Agent-to-agent project messages require a verified sender handoff record. The next agent must inspect/acknowledge it. The ledger lists artifacts, hashes and `test_status` (currently `not_executed`, because this prototype does not execute generated code).
- Approval of a project through the tracked API is guarded by a human operator and refuses unresolved tasks, cancelled tasks, or unacknowledged handoffs. **There is no real deployment endpoint**.
- The app binds to localhost and uses restrictive origin checks. Do not expose its Python HTTP server to the public internet without adding HTTPS, authentication, RBAC, rate limits and other hardening.

## Verify locally

```powershell
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" -m compileall -q .
```

The release passed **60 deterministic unit/integration tests**, Python compilation, Node.js JS syntax validation, a live local-server startup check, and a headless Chromium UI walkthrough with a Python-to-HTTP bridge (used because the automated browser environment blocks direct localhost navigation). These tests did not call OpenAI, verify your Windows browser, or exercise live GPT account credentials.

The test report is in `docs/TEST_REPORT.md` and a sample project brief is in `docs/EXAMPLE_PROJECT.md`.

## Source map

- `server.py` — local HTTP server, JSON routes, access restrictions
- `static/` — project intake, organization, meeting, task, and evidence interfaces
- `engine.py`, `gpt_client.py`, `config.py` — GPT agent processing and model routing
- `store.py` — project/work queues, SQLite cancellation receipts, human approval guard
- `meetings.py` — scheduled and live rooms, interrupted turns, audited cancellation
- `handoffs.py` — project documents, SHA-256 integrity and mandatory handoffs
- `roles.py`, `architecture.py` — 49 roles and 9 organizational layers
- `tests/` — standard-library tests; no OpenAI service needed

## Honest scope boundary

V1.0 **can start and track an AI product project and coordinate specialist GPT messages and meetings**. It does not yet write files on behalf of agents, run tool sandboxes, execute pytest on generated code, create Git commits, ingest a vector index, or deploy the resulting RAG/AI products. Additional implementation and verification are required before relying on it for autonomous software delivery.
