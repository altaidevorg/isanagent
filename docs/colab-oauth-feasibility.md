# Colab OAuth Feasibility Spike (VS-extension-style)

This document captures a one-week feasibility spike for OAuth-native Colab integration in isanagent (modeled on VS Code extension style auth flows).

## Scope

- Evaluate direct OAuth integration complexity versus current execution architecture.
- Identify required system changes for per-user identity routing.
- Define a concrete go/no-go threshold before implementation.

## Findings

### 1) Current architecture mismatch

- Execution providers are currently process-config scoped (`[harness.execution]`) and not per-user credential scoped.
- `ExecutionHarness` is built once from config and env; tokens are static (`JUPYTER_TOKEN`, `SSH_PASSWORD`) rather than refreshable OAuth sessions.
- Existing tool/session APIs do not carry identity-backed credential leases.

### 2) OAuth complexity is high (not just login)

Based on Colab VS Code auth patterns (loopback + proxied redirect + revoked-token recovery), a production implementation must handle:

- Loopback callback flow and fallback code-entry flow.
- Secure token storage lifecycle (refresh, revoke, re-login).
- 401 feedback loop from execution provider back into auth provider.
- Multi-channel behavior (terminal/API/subagents) for re-auth prompts and retries.

### 3) Security and multi-tenant gaps to solve first

- Credential encryption-at-rest policy (OS keychain vs DB envelope encryption).
- Session-to-identity mapping for concurrent chats.
- Explicit token redaction and audit boundaries.

## Effort Estimate

- **OAuth-native MVP:** 6-10 weeks, depending on credential-store and channel UX requirements.
- **Hardened production path:** 10+ weeks with robust failure handling and telemetry.

## Go/No-Go Decision

**Decision: NO-GO for immediate implementation.**

Proceed with OAuth-native build only when all preconditions are true:

1. A per-user credential architecture RFC is approved.
2. Credential storage policy is selected and implemented.
3. Auth event propagation hooks exist across primary channels.
4. Team capacity is available for multi-week integration and maintenance.

## Recommended Path

1. Ship UV-managed local runtime baseline.
2. Ship Colab MCP MVP (browser-mediated path) for early audience reach.
3. Revisit OAuth-native integration after MCP usage validates demand and required UX.

## References

- [Colab MCP](https://github.com/googlecolab/colab-mcp)
- [Colab VS Code extension](https://github.com/googlecolab/colab-vscode)
- [Loopback OAuth commit](https://github.com/googlecolab/colab-vscode/commit/9232ea0dc8e144a390c9bf72fe14d1e853045095)
- [Proxied redirect flow commit](https://github.com/googlecolab/colab-vscode/commit/12ff323c746ad657e9fd775d9797724c3a111bec)
- [Revoked-token handling issue](https://github.com/googlecolab/colab-vscode/issues/238)
