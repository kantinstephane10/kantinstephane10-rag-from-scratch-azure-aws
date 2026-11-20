# V1.0 test report

Tested in the development container with Python 3.13, Node.js 22, Chromium headless and temporary SQLite databases.

- `python -m unittest discover -s tests -q`: **60 passed**.
- `python -m compileall -q .`: **passed**.
- `node --check static/app.js`: **passed**.
- HTML ID uniqueness, JS references and removal of browser `confirm()`: **passed**.
- Actual server subprocess launch, `GET /api/health`, `GET /`: **passed**.
- Headless Chromium browser UI walkthrough: **passed**, using an HTTP bridge to the real local API due to browser network blocking. Covered project creation, document upload, precise two-of-three task cancellation, custom reason modal, meeting creation/cancellation, artifact ledger and JavaScript runtime errors.

New regression cases cover missing/invalid reasons, all-or-nothing bulk selection, mixed projects, stale completions, cancelled-task approval gates, persistence/restart, migration from legacy schemas, cancelled meetings and attempted outgoing agent messages after cancellation.

Not tested: your Windows browser/PowerShell runtime, your actual OpenAI API key or billing, live GPT model access and Responses API output quality, external GitHub CI, and actual product-code generation/deployment.
