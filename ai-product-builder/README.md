# AI Product Builder — GPT Meeting Studio v0.5

A locally hosted AI product engineering **control-plane prototype**, derived from the nine-layer architecture PDF. All 49 specialist roles are registered and can exchange persisted project messages; agents and operators can call meetings, the Product Manager chairs, and questions addressed to a specific agent receive priority turns. The browser lets the operator view a live-updating transcript, direct questions to a specialist, join the conversation, and interrupt the AI speaker queue.

This version **exclusively uses the OpenAI Responses API** (role-configurable GPT models). It does **not use a local LLM, Ollama, or an automatic model fallback**.

## Run on Windows — Python 3.10 or newer

Unzip this archive, then from a PowerShell terminal in the extracted directory:

```powershell
python -m unittest discover -s tests -v
$env:OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
python server.py
```

Open **http://127.0.0.1:8766/** on that same computer. The API key is **not** stored by this app; do not commit it to Git or paste it into a public chat. Using the OpenAI API can incur separate API charges and model access may vary by organization.

For an **offline interface demo** that never calls OpenAI, run:

```powershell
python server.py --demo
```

The dashboard displays a clear **OFFLINE DEMO — no GPT calls** label. Do not treat demo text as actual AI reasoning or engineering output.

## Direct-response meeting behavior

1. Click **New meeting**, provide a topic and select any participants. The Product Manager is always added and chairs the meeting.
2. Select a room. The transcript automatically refreshes approximately every second. These are completed turns, **not token-by-token streaming**.
3. Use **Ask** to select an agent, including one not already in the room. Enter your question and click **Send question**. Your question is saved; the floor then goes back to the AI group, and the addressee responds before normal agenda turns.
4. During any AI meeting turn, an agent may send up to **one explicitly addressed question** in its structured `outgoing` field. The target is automatically invited and queued to respond as the next priority turn. Agent-to-agent follow-ups are bounded to prevent ping-pong loops.
5. Click **Take the floor** to pause agent responses. Any GPT response that was already running is **prevented from being published**, even if the network request finishes later; it is not necessarily cancelled or refunded by OpenAI. Use **Return floor to agents** to let discussion resume.
6. **Dictate** uses the browser's optional Web Speech Recognition implementation; starting dictation takes the floor immediately. Review/edit the recognized words before sending. The browser must support SpeechRecognition and permission for microphone use. The app does **not** implement always-on listening or live synthesized audio.
7. The transcript, speaker queue, meeting status, participants, and state are saved in SQLite. Scheduled meetings are activated when the server is running and their scheduled time arrives.

## Agent model routing

Default role groups are mapped to GPT model IDs in `config.py`. Override them as needed:

```powershell
$env:GPT_MODEL_ARCHITECTURE = 'gpt-6.2-astra'
$env:GPT_MODEL_ENGINEERING = 'gpt-6.2-sol'
$env:GPT_MODEL_BUSINESS = 'gpt-6.1-terra'
$env:GPT_MODEL_ROUTINE = 'gpt-6-luna'
```

Individual role overrides use the exact role name converted to uppercase underscore, e.g. `GPT_MODEL_DEV_1`, `GPT_MODEL_RAG_ENGINEER`, `GPT_MODEL_PRODUCT_MANAGER`.

## File map

- `server.py`: same-origin localhost dashboard and REST API
- `static/`: responsive HTML/CSS/JS interface, no external CDN requirement
- `roles.py`, `architecture.py`: 49-role/nine-layer organization
- `engine.py`, `gpt_client.py`, `config.py`: agent work loop and GPT Responses API integration
- `store.py`: persistent projects, messages, project authorization, events
- `meetings.py`: scheduled rooms, prioritized Q&A, floor interruption, restart recovery
- `tests/test_meeting_studio.py`: deterministic tests including actual HTTP exchange with a local test server
- `docs/TEST_REPORT.md`: tested scope and limitations

## Status and security boundaries

This is **a working local control-plane and meeting prototype, not a fully deployed production AI product factory**. In particular, tool execution, artifact synthesis, sandboxed builds, CI/CD, production observability, and AI-product evaluation pipelines are future work. Human approval is enforced for the tracked project approval API, but the app does not have a production deployment function.

The local server binds to `127.0.0.1` by default and blocks cross-origin writes. **Do not expose the raw HTTP service to the internet.** A production deployment requires authentication, HTTPS, RBAC, rate limits, hardened audit logging, queue supervision, secret management, and penetration testing. Browser functionality was not verified in a real Windows browser in this environment.

## v0.6 — project intake and live worker review

This release integrates the *project setup / agent sequence* elements shown in the user's references into the GPT Meeting Studio v0.5 base:

- Project type (`agent`, `agentic`, `rag`) and GPT routing profile (`standard`, `deep-review`, `cost-aware`).
- Explicit provider-consent checkbox before a live GPT project can begin. No API key is collected by the browser.
- Operator-entered budget fields: engineering hours × rate + one-time expense, then contingency. Monthly cost is displayed separately; **not an AI-produced cost quote**.
- SQLite-backed intake settings and a live-refreshing agent sequence using actual message status (`queued`, `processing`, `done`, `failed`).
- Selecting an agent task shows its recorded assignment, response, model ID, errors and retry count. **Completed messages are not claims of successful code execution.**
- **Ask this worker in a meeting** preselects the worker and project in the existing Product Manager-chaired meeting scheduler.

The only implemented project engine remains a persistent GPT *messaging/planning* prototype. There is no live code execution, evidence-based approval from Git commit/test receipts, RAG ingestion, attachment upload, or deployment in this package. The UI does not fake these steps.

To choose the optional deep-review or cost-aware GPT model for a project, configure `GPT_MODEL_DEEP_REVIEW` or `GPT_MODEL_COST_AWARE` with an actual OpenAI API model ID available to your account. Without an override, the usual per-role model routing remains in effect. Consult the OpenAI API for model availability; the repository's example model names are not guarantees of access.

### Validation

Run from the project root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
```

`node --check static/app.js` verifies syntax if Node.js is installed. The HTTP tests use a local fake client and **never contact OpenAI**. Real Windows browser behavior and live GPT calls require validation in the actual runtime environment.

### Git branch

Prepared locally as `feat/project-workspace-live-agent-review`. This package does **not** contain a GitHub remote or credentials and has **not** been pushed. To push, specify an existing destination repository where the GitHub connector grants push permission. Do not publish environment variables, SQLite databases, private project documents, or API keys.
