# V1.3.3 — Long briefs, 1 GiB streamed files, worker document access, meeting uploads

## Fixed

- A project brief may contain up to 20,000 characters without exceeding the 6,000-character mailbox limit. Kickoff uses short assignments and the full project brief already stored in the project context.
- A pre-existing `draft` project now has a **Start agents** action in the Project ledger. No new project is necessary.
- Both mandatory Business Analysts receive the complete brief and project files. The Product Manager still waits for both analysts before planning.
- Upload paths use `PUT` with a `Content-Length`, stream through 1 MiB chunks, hash as they write, and publish atomically; file limit is **1 GiB per file (1,073,741,824 bytes)**.
- Project and RAG upload forms use streaming `XMLHttpRequest` file transfer with a progress indicator instead of in-memory base64 conversion.
- The Meeting Studio has a file upload button, attachment ledger, and readable text preview.
- Every registered agent role receives a project-scoped inventory plus verified real content excerpts and can request further PDF pages or text offsets using `file_requests` in structured OpenAI responses. Files uploaded to a project meeting become accessible to all agents in that project; standalone meeting files remain scoped to that meeting.
- Source bytes are verified by SHA-256 with a bounded-memory incremental hashing pass and a stable-file cache. Changed files are rejected.
- RAG PDF ingestion reads from disk instead of loading a large PDF completely into RAM.

## Safety and limitations

- Each **upload** may be 1 GiB; this does NOT mean the entire 1 GiB is indexed or sent to an LLM. Current RAG indexing remains bounded by 500 PDF pages, 2 million extracted characters, and 5,000 chunks. If those limits are exceeded, indexing reports an error rather than a false success.
- Text/PDF extraction requires readable file content. Scanned-image PDF OCR is not included.
- The app currently runs as a single-operator localhost server, **not** a multi-user identity/ACL service. File access means registered agents working within the authorized project or standalone meeting context, not an unscoped global file share.
- Inference can make up to three additional file-read rounds before returning a result; this bounded tool loop is not a project-level token cap.
- Live GPT model compatibility, browser-specific 1 GiB transfer behavior, and Windows DPAPI behavior must be verified on the user's machine. Tests used fake clients rather than billable API requests.
- Does not yet deliver full autonomous source-code generation, sandbox execution, or production deployment.

## Windows PowerShell

From `ai-product-builder-v1` in the feature branch:

```powershell
& "$HOME\anaconda3\python.exe" -m pip install -r requirements-rag.txt
& "$HOME\anaconda3\python.exe" -m unittest discover -s tests -v
& "$HOME\anaconda3\python.exe" server.py --port 8767
```

Open `http://127.0.0.1:8767/`. In the Project ledger, select your existing draft and click **Start agents**. Uploaded files remain in the existing SQLite/artifact workspace unless you explicitly delete a project. In Meeting Studio, select a room, choose files, then click **Upload files to meeting**.
