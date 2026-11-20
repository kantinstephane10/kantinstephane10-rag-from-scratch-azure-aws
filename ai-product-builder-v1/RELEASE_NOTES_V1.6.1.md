# AI Product Builder 1.6.1 — Unattended Windows-host Python

- No per-script approval for generated Python; automatic host execution is enabled by default.
- Durable project execution queue, source revision pinning, restart recovery, run results, and automatic compilation/tests.
- Project UI supports pause/resume and stop/cancel; stale jobs are skipped.
- Project deletion cancels outstanding host execution before removing the workspace.
- Multiuser administrators control unattended execution and Python source changes when it is enabled.
- OpenAI GPT agents can request Python runs by returning `workspace_runs`; command inputs remain project-scoped Python files or named test actions, not arbitrary shell strings.
- Windows host execution is unsandboxed and carries Windows user privileges. This release has not been validated on the user's Windows machine.
