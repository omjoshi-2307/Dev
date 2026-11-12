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
| AI router | Provider selection, policy, fallback | Planned |
| OpenAI adapter | Cloud model requests | Planned |
| Ollama adapter | Local model requests | Planned |
| Orchestrator | Conversation flow and tool-call lifecycle | Planned |
| Memory hub | User-controlled memory and retrieval | Planned |
| Context bridge | Approved context export/import | Planned |
| Permissions | Authorization and confirmation gates | Planned |
| Tools | Browser, files, desktop and external integrations | Planned |
| Observability | Structured logs, errors and usage metadata | Planned |

## Provider routing policy

Initial routing preference:

- Local provider for suitable low-complexity tasks.
- OpenAI for complex reasoning, coding assistance and research when cloud processing is allowed.
- Local fallback only when it can safely and adequately handle the request.
- If no eligible provider is available, return a clear error rather than silently changing privacy or quality requirements.

Provider selection must consider task complexity, data sensitivity, provider availability, configured model capability and cost. Sensitive content must not be sent to a cloud provider merely because a task is complex.

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
3. OpenAI and Ollama provider adapters behind a common interface.
4. Policy-based model routing and safe fallback.
5. Validated chat API and provider tests.

Voice, desktop control, long-term memory, context synchronization, mobile access and autonomous multi-agent workflows are intentionally out of scope for Stage 1.
