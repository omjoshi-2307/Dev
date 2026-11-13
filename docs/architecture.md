# DEV — Initial Technical Architecture

**Status:** Architecture baseline for early development. Components marked planned are not implemented yet.

## Product decisions

- Product name: **DEV**
- AI strategy: **Balanced hybrid** — local Ollama for suitable routine tasks; OpenAI for complex reasoning, coding assistance and research.
- Memory strategy: **Selected sharing** — DEV owns its memory store; only explicitly selected and approved context may be exported or imported.
- Execution authority: DEV's application core, not the language model.

## High-level request flow

1. A client submits a text or voice-derived request to the DEV gateway.
2. The gateway validates the request and establishes its session context.
3. The AI router applies deterministic policy to select an eligible provider.
4. The provider returns a response or a proposed tool call.
5. DEV's orchestration layer validates proposed actions and checks permissions.
6. Approved tools execute within their configured boundaries.
7. DEV verifies the result and returns a response to the client.
8. Approved information may be written to DEV memory or explicitly exported through the context bridge.

## Planned modules

| Module | Responsibility | Status |
|---|---|---|
| Gateway | API entry point, request validation, client authentication | Initial API skeleton |
| AI router | Provider selection, deterministic policy, and fallback rules | Schemas and deterministic policy implemented; execution and live health checks planned |
| OpenAI adapter | Cloud model requests | Implemented |
| Ollama adapter | Local model requests | Implemented |
| Orchestrator | Conversation flow and tool-call lifecycle | Planned |
| Memory hub | User-controlled memory and retrieval | Planned |
| Context bridge | Approved context export/import | Planned |
| Permissions | Authorization and confirmation gates | Planned |
| Tools | Browser, files, desktop and external integrations | Planned |
| Observability | Structured logs, errors and usage metadata | Planned |

## Provider routing policy

DEV uses a pure, deterministic routing policy (`backend/app/router/policy.py`) to select between local Ollama and cloud OpenAI providers. The routing engine never delegates selection authority to an LLM.

### Deterministic Routing Rules

When no explicit provider is specified (`preferred_provider="auto"`), routing evaluates task complexity and privacy level according to this decision matrix:

| Task Complexity | Privacy Level | Preferred Provider | Primary Rationale |
|---|---|---|---|
| `SIMPLE` | `LOCAL_ONLY` or `CLOUD_ALLOWED` | Ollama | Cost minimization and low latency |
| `COMPLEX` | `LOCAL_ONLY` | Ollama | Strict privacy preservation |
| `COMPLEX` | `CLOUD_ALLOWED` | OpenAI | Advanced reasoning / capability |
| `AUTO` | `LOCAL_ONLY` or `CLOUD_ALLOWED` | Ollama | Conservative baseline default |

### Privacy Enforcement

- `LOCAL_ONLY` is the mandatory baseline privacy setting. Requests marked `LOCAL_ONLY` are strictly prohibited from reaching OpenAI or any cloud provider.
- This boundary is absolute: even if fallback is enabled or Ollama is unconfigured, a `LOCAL_ONLY` request will never fall back to OpenAI.
- If a client explicitly requests `preferred_provider="openai"` while `privacy_level="local_only"`, the policy immediately rejects the request with a `RoutingPolicyError`.
- Privacy levels are never upgraded, downgraded, or silently mutated.

### Explicit Provider Preference

- Explicit user preferences (`preferred_provider="ollama"` or `"openai"`) are strictly respected.
- When fallback is disabled (`allow_fallback=False`), the router will never switch away from an explicitly preferred provider.
- If an explicitly preferred provider is not configured, the policy raises a `RoutingPolicyError`.

### Provider-Default Precedence

Provider selection adheres to an unambiguous precedence hierarchy:
1. **Explicit Client Preference:** An explicitly specified client preference (detected via Pydantic `model_fields_set`) takes precedence over all defaults and complexity evaluations, provided privacy rules permit it. An explicit client preference for `openai` under `LOCAL_ONLY` is immediately rejected.
2. **Local Privacy Override:** Under `LOCAL_ONLY`, an inherited OpenAI default from environment configuration (`DEV_ROUTER_DEFAULT_PROVIDER="openai"`) is strictly overridden by the local privacy policy. The request routes safely to Ollama per automatic local policy rather than causing an error.
3. **Configured Provider Defaults under `CLOUD_ALLOWED`:** Under `CLOUD_ALLOWED`, when no explicit client preference is provided, a configured non-AUTO provider default takes precedence over the automatic complexity matrix (e.g. default OpenAI routes to OpenAI even for `SIMPLE` tasks; default Ollama routes to Ollama even for `COMPLEX` tasks).
4. **Automatic Complexity Matrix:** When the configured default is `AUTO` (or unspecified) and no explicit preference is provided, the automatic complexity matrix applies (`COMPLEX` + `CLOUD_ALLOWED` -> OpenAI; all other combinations -> Ollama).

### Fallback Restrictions

Fallback to an alternative provider occurs only when all of the following conditions are met:
1. `allow_fallback` is explicitly set to `True` on the request.
2. The primary preferred provider is unconfigured.
3. The alternative provider is configured.
4. The alternative provider satisfies the request's privacy constraints (e.g. `LOCAL_ONLY` requests can never fall back to cloud providers).

When a fallback occurs, the `RoutingDecision` explicitly records `is_fallback=True` and populates `fallback_reason` with diagnostic details for auditability.

### Configuration Readiness vs. Runtime Availability

In Stage 2 Task 2, eligibility is determined via static configuration readiness:
- **Ollama is ready when:** `OLLAMA_MODEL` is set and non-empty.
- **OpenAI is ready when:** `OPENAI_API_KEY` and `OPENAI_MODEL` are both set and non-empty.

*Important distinction:* Configuration readiness indicates that the necessary environment variables and settings exist; it does not perform network pings or guarantee live daemon or service availability. Dynamic runtime availability and health probing are handled in subsequent orchestration tasks.

### Current Limitations of AUTO Complexity

In this stage, `AUTO` complexity does not invoke an AI classifier or dynamic heuristic text analyzer to score prompt complexity. Instead, `AUTO` conservatively defaults to the local provider (`Ollama`) to maximize data locality and avoid accidental API billing. Clients requiring cloud capabilities must either explicitly specify `preferred_provider="openai"` or declare `complexity="complex"` with `privacy_level="cloud_allowed"`.

Client metadata (`ChatRequest.metadata`) is untrusted and cannot override system instructions, security boundaries, or authoritative routing decisions.

## Provider Adapters and Platform Integration

DEV interfaces with AI models via standardized provider adapters that satisfy the `AIProvider` protocol (`backend/app/providers/base.py`).

### OpenAI Responses API Adapter (`OpenAIProvider`)

Cloud text generation is implemented in `backend/app/providers/openai_provider.py` targeting the OpenAI Responses API (`/v1/responses`).

#### Implementation Rationale & Windows Code Integrity

The adapter uses `httpx` directly rather than invoking the official `openai` Python SDK client.
- **Root Cause & Platform Constraint:** Under Windows systems with active Windows Defender Application Control (WDAC) or Smart App Control signing policies (e.g., policy `0283ac0f-fff1-49ae-ada1-8a933130cad6`), dynamic native libraries (`.pyd` files) within user directories are blocked by Code Integrity (Event ID 3077/3033). The official `openai` SDK imports `jiter` (`jiter.cp315-win_amd64.pyd`), a compiled Rust JSON accelerator with no pure-Python fallback. This caused an unconditional `ImportError: DLL load failed while importing jiter` on Windows.
- **Architectural Alignment:** DEV's `OllamaProvider` already relies on `httpx` for asynchronous HTTP communication. Implementing `OpenAIProvider` with `httpx` establishes design uniformity across all provider adapters, operates purely in Python without binary extensions, and avoids weakening host operating system security controls.

#### Protocol & Schema Mapping

- **Endpoint:** `https://api.openai.com/v1/responses` (POST).
- **Role Translation:** The Responses API specifies the `developer` role rather than the legacy `system` role. `OpenAIProvider` automatically translates incoming `ChatMessage(role="system", ...)` to `{"role": "developer", ...}` in the payload.
- **Output Parsing:** The Responses API returns an `output` array of message objects containing content elements of type `output_text`. `OpenAIProvider.extract_output_text` reconstructs the concatenated text from all text parts while also supporting direct top-level `output_text` payloads for test and mock compatibility.

#### Error Handling and Credential Protection

- **Credential Safety:** Server-side `OPENAI_API_KEY` is loaded securely via Pydantic `Settings`. Sensitive credentials are never exposed in log outputs, exception traces, or client-facing errors.
- **Authentication Failures (HTTP 401):** Mapped to `ProviderConfigurationError` with a sanitized message to prevent leaking partial API keys or token fragments.
- **Rate Limits & Quotas (HTTP 429):** Mapped to `RuntimeError` detailing quota or rate-limit exhaustion.
- **Server Errors (HTTP 5xx):** Mapped to `RuntimeError` with HTTP status code and upstream diagnostic message.
- **Timeouts & Network Faults:** Mapped to standard Python `TimeoutError` (`httpx.TimeoutException`) and `ConnectionError` (`httpx.RequestError`).
- **Malformed Outputs:** Non-JSON responses or empty/whitespace-only output payloads raise `ValueError`.

### Ollama Local Adapter (`OllamaProvider`)

Local model inference is implemented in `backend/app/providers/ollama_provider.py` communicating with the Ollama server at `OLLAMA_BASE_URL` (defaulting to `http://127.0.0.1:11434/api/chat`). All local requests are strictly non-streaming and execute within the local network perimeter.

## Memory boundaries

Keep these concepts separate:

- **Conversation state:** messages and context for an active conversation.
- **Long-term memory:** user-approved facts and project knowledge retained across sessions.
- **Execution state:** task progress, tool calls, approvals and results.

DEV is the canonical owner of its memory. ChatGPT does not automatically share its account memory or chat history with the OpenAI API. Any exchange with a ChatGPT client requires an explicit supported integration and user-controlled data transfer.

## Security requirements

- API credentials remain server-side and are never returned to clients.
- Do not commit local environment files or secrets.
- Authenticate clients before enabling remote access.
- Do not expose the development server directly to the public internet.
- Require approval for consequential actions, including file modification, message sending, software installation and privileged commands.
- Record tool execution and approval outcomes in an audit log.
- Provide a way to stop active task execution.
- Treat model output and external content as untrusted input; validate all tool arguments.

## Stage 1 scope

1. Python backend skeleton and settings.
2. Health endpoint and automated tests.
3. OpenAI and Ollama provider adapters behind a common interface — implemented on `feature/provider-adapters`; local execution remains to be verified.
4. Policy-based model routing and safe fallback — next.
5. Validated chat API and provider tests — adapter unit tests added; local test execution remains pending.

Voice, desktop control, long-term memory, context synchronization, mobile access and autonomous multi-agent workflows are intentionally out of scope for Stage 1.
