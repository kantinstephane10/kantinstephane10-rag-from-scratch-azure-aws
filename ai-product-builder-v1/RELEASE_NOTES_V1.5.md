# AI Product Builder V1.5 release notes

## Implemented and testable
- New `website` product type, automatic static site generation on creation, editable themes/templates, local sandboxed iframe preview, ZIP export, build provenance and integrity verification.
- Website build files are isolated per project and deleted with their project. Failed rebuilds do not change the active site.
- Local runtime/dependency audit reports available Python libraries, Git/Node/OCR programs, free disk and remaining production security gaps.
- Windows launcher no longer incorrectly demands a PowerShell API key when the UI already stores it.
- OpenAI GPT, Hugging Face embedding, incremental OCR/indexing and 49-agent coordination from previous releases retained.

## Not implemented or independently certified
- General-purpose AI code-writing/execution sandbox, production-ready agent tool runner, production deployment, public authentication/TLS, authorization revocation racing, full 1 GiB stress benchmark and real OpenAI end-to-end acceptance.
- Current website generator is a safe template compiler, not an unrestricted GPT full-stack developer.

## Quick verification
`python -m unittest discover -s tests -v` then `python server.py --port 8767` and open `http://127.0.0.1:8767/`.