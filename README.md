# DEV

**DEV** is a personal AI assistant built around a hybrid AI architecture: OpenAI models for complex reasoning, coding assistance and research, and local models through Ollama for suitable tasks.

> **Status:** Early development — Stage 1 (AI foundation). Roadmap items are planned unless explicitly marked complete.

## Principles

- Hybrid by design: cloud and local models use a common provider interface.
- User-controlled memory: DEV owns its memory; only selected, approved context is shared.
- Permission-first execution: models may propose actions, but DEV enforces permissions.
- Privacy-aware routing: sensitive content is not sent to a cloud provider without an appropriate policy.
- Small, verifiable steps: each subsystem is tested before expansion.

## Current milestone: Stage 1 — AI foundation

1. Establish the Python backend and configuration.
2. Add health checks and automated tests.
3. Implement OpenAI and Ollama provider adapters.
4. Add balanced, policy-based model routing.
5. Expose a validated chat API.

The initial setup establishes the backend skeleton and health endpoint. Provider integration and routing are subsequent Stage 1 tasks.

## Technology direction

| Area | Initial choice |
|---|---|
| Backend | Python, FastAPI |
| Cloud AI | OpenAI API |
| Local AI | Ollama |
| Configuration | Pydantic Settings |
| Initial persistence | SQLite (planned for memory stage) |
| API communication | REST; WebSocket streaming later |
| Tests | Pytest |

## Requirements

- Python 3.11 or newer
- Git
- Internet access for dependencies and OpenAI API use
- Ollama is optional for initial setup; required for local-model integration

## Local setup

From the repository root:

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy .env.example to .env in the repository root. Never commit .env or put API keys in source code, README files, screenshots or frontend code.

### 4. Start the backend

```bash
python -m uvicorn backend.app.main:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

The health endpoint currently verifies that the API process is running. It does not yet check OpenAI or Ollama connectivity.

### 5. Run tests

```bash
python -m pytest
```

## Repository structure

```text
Dev/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       └── main.py
├── docs/
│   └── architecture.md
├── tests/
│   └── test_health.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Security

- Keep credentials in local environment variables or a secrets manager.
- Use least-privilege permissions for future tools and integrations.
- Require confirmation for consequential actions such as modifying files, sending messages, installing software or running privileged commands.
- Keep an audit trail for tool actions.
- Do not expose the development server directly to the public internet.

## Roadmap

- [x] Create private repository and initial project documentation
- [x] Establish backend skeleton and health endpoint
- [ ] Add OpenAI provider adapter
- [ ] Add Ollama provider adapter
- [ ] Implement AI routing and fallback policy
- [ ] Add chat request/response schemas and API
- [ ] Add provider and routing tests
- [ ] Design the memory hub and selected-context bridge
- [ ] Add voice, interfaces and controlled tools in later stages

## License

No license has been selected. Until one is added, this repository does not grant general permission to reuse, modify or redistribute its contents.
