# LinkedIn Job Search Agent & Resume Optimizer

Parses resumes into structured JSON, ingests scraped LinkedIn job listings,
scores job fit with local embeddings + cosine similarity, and generates
targeted bullet-by-bullet resume rewrites for low-scoring sections.

See `PLAN.md` (in the parent project folder) for the full architecture
write-up, schemas, and design rationale.

## Stack

- Resume structuring: NVIDIA NIM (`meta/llama-3.1-70b-instruct` by default)
- Embeddings: Ollama, local (`nomic-embed-text` by default)
- Resume rewrite generation: Ollama, local (`llama3.1:8b-instruct` by default)
- Similarity scoring: scikit-learn / NumPy cosine similarity (deterministic)
- Orchestration: LangChain (LCEL)
- API: FastAPI

## One-time setup

Requires Python 3.10+ (the codebase uses `X | None` union type syntax that
pydantic evaluates at runtime, which needs 3.10+). macOS ships an older
system Python, so install a newer one first if needed:

```bash
brew install python@3.11
```

Then, from inside this `linkedin-job-agent/` folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

cp .env.example .env
# open .env and fill in NVIDIA_API_KEY (get one at build.nvidia.com)

# Install Ollama if you don't have it: brew install ollama
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct
```

## Every time you want to run it

**Terminal tab 1 -- start Ollama** (skip if already running):

```bash
ollama serve
```

**Terminal tab 2 -- activate the venv, then run the project:**

```bash
cd linkedin-job-agent
source .venv/bin/activate
```

Then either:

**CLI**, for a quick one-off run against a resume + job list:

```bash
python scripts/run_pipeline_cli.py --resume "/path/to/resume.pdf" --jobs tests/fixtures/sample_jobs.json --optimize
```

Quote the resume path if it contains spaces. Output is printed as JSON --
redirect it to a file to read comfortably: append `> result.json` to the
command above, then open `result.json`.

**API server**, if you'd rather use a browser UI:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI, where
you can try each endpoint (upload a resume, submit job listings, run a match)
without writing any code. `http://localhost:8000/health` confirms the server
is alive; visiting `/` directly will 404 -- that's expected, there's nothing
registered at the root path.

Endpoints:
- `POST /resume/parse` -- upload a PDF/DOCX, get back a structured `ResumeProfile`
- `POST /jobs/ingest` -- submit raw scraped job dicts, get back normalized `JobListing[]`
- `POST /match` -- score a resume against one or more jobs, get back ranked `MatchResult[]`
- `POST /match/optimize` -- given a resume, job, and match result, get back bullet rewrites

## Tests

```bash
pytest
```

Unit tests use a deterministic fake embedder (no Ollama/NVIDIA dependency),
so `pytest` runs fully offline. Wire real integration tests against a live
Ollama server separately once models are pulled.


## Main Web Page
<img width="1507" height="724" alt="Screenshot 2026-08-02 at 6 05 27 PM" src="https://github.com/user-attachments/assets/7ba91146-af2f-444c-ae4e-b915f6e86580" />
