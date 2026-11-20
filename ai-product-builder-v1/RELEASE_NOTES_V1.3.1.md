# AI Product Builder V1.3.1 — Relay failure recovery

### Fixed
- Specialist jobs at the relay boundary no longer lose valid GPT responses.
- Bounded relay depth raised from 4 to 12, with explicit no-delegation instructions at the final hop.
- Invalid or excessive optional follow-up requests are suppressed and audited rather than failing the completed worker task.
- One-time startup recovery for existing failed jobs carrying the old exact relay-limit error.
- New `Retry failed task` control; project-scoped backend retry.
- Protect final task state from a later audit/acknowledgement failure.

### Validation
- 88 local automated tests passed (including six relay-recovery tests).
- JavaScript syntax and Python compilation passed.
- No live OpenAI API requests or Windows browser tests were performed in this environment.

### Limitations
- GPT retries may incur additional charges.
- Workflow relay guards do not impose output-token caps.
- Code-generation sandboxes, CI/CD and production deployment remain separate milestones.