# AI Product Builder V1.6 — Windows Python execution & controlled operations

## Implemented and locally verified

- Project-specific source workspaces (`data/workspaces/<project_id>/`) and immutable source snapshots (`data/workspace_versions/`).
- GPT agent responses can include real source-file proposals (`workspace_files`), saved by the engine with SHA-256 receipts and verified in subsequent handoffs. No GPT response automatically runs Python.
- Human-approved Python execution using the **same interpreter that starts `server.py`**; works with your Windows Anaconda Python. Supports any project-local `.py` script, unittest discovery, pytest, and compile checks. Runs use subprocess argument lists (no shell), explicit confirmation, fixed working directory, minimal inherited environment, no inherited OpenAI key, bounded output, and a 90-second maximum timeout. Run logs record exit code, timing and output.
- **Security warning:** Windows-host Python execution is NOT sandboxed. Untrusted code can access your Windows user files/network and install or spawn processes. Review agent-produced code before running. Dedicated low-privilege Windows accounts and Windows Sandbox/VM remain recommended for untrusted workloads. Docker remains optional.
- Optional `--multiuser` mode: user passwords stored using scrypt; HttpOnly/SameSite sessions, CSRF token checks, local login UI, viewer/editor/owner project roles, admin-only secret changes and Windows-host script execution. Multiuser mode is deliberately localhost-only and requires further hardening before internet-facing use.
- Human-approved **local** website release staging with source hash verification, atomic active-version pointer, and tamper rejection. Not public cloud deployment.
- `/api/metrics`, `/api/system/capacity`, and `/metrics` provide low-cardinality HTTP counters, disk and active work metrics. In-process metrics reset on restart. Metrics require admin in multiuser mode.
- `load_test.py` runs repeatable bounded, non-destructive localhost HTTP health load checks. This does not constitute a full throughput, ingestion, or 1-GiB load certification.
- Existing website builder, large-document RAG indexing, local Hugging Face embeddings, OCR options, and 49-agent model routing are preserved.

## Not production certified

No arbitrary unattended code execution; user confirmation per Windows run. No public HTTPS deployment pipeline, OS sandbox, durable distributed queue, network policy, complete user/session administration UI, audited backup and recovery automation, or validated full 1-GiB load. Live OpenAI and full Windows-host integration must be tested on your actual installation.

## Windows quick start

In `ai-product-builder-v1`:

```powershell
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Open `http://127.0.0.1:8767/`, select a project and use **Project code workspace · Windows Python**. Save an `.py` script, review it, type the exact project name, then click **Run on Windows Python**. The executable shown in the UI is the Python interpreter running the server.

### Multiuser (opt-in)

From a terminal with the server stopped:

```powershell
& "$HOME\anaconda3\python.exe" manage_users.py bootstrap-admin operator
& "$HOME\anaconda3\python.exe" manage_users.py add-user analyst1
& "$HOME\anaconda3\python.exe" manage_users.py grant <project-id> analyst1 editor
& "$HOME\anaconda3\python.exe" server.py --multiuser --port 8767
```

Use the built-in login screen. The first administrator bootstrap requires no existing users. For existing projects, administrator access always applies; other users see only granted projects. Only administrators can run host Python. The same `data/builder.sqlite3` database is used; make a verified backup before migrating critical data. Do not expose this localhost HTTP service directly to the internet.

### Test local HTTP capacity

```powershell
& "$HOME\anaconda3\python.exe" load_test.py --url http://127.0.0.1:8767 --requests 500 --workers 16
```

Use a dedicated test instance and data copy for substantial load testing. This exercises `/api/health`, not GPT or ingestion.
