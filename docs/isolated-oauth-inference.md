# Isolated OpenAI OAuth inference

This additive boundary executes one explicitly bound OpenAI OAuth request without the
Hermes conversation agent, configuration loader, plugins, credential pool, auxiliary
clients, or fallback resolver. Its external interface is a fresh subprocess:

```text
<Hermes Python> -I <absolute Hermes source>/hermes_cli/isolated_oauth.py --home <absolute authorized Hermes home>
```

Supply one UTF-8 JSON document on stdin. Stdout is one JSON document. Exit zero means
success; exit one returns a fixed error code. Request and response bodies are limited
to 8 MiB. Duplicate JSON keys, nonfinite numbers, unknown request fields, and malformed
provider data fail closed. The Python helper functions are internal implementation
and testing seams; an external worker uses the subprocess boundary.

## Discovery

```json
{"operation":"describe_read_only","provider":"openai_oauth"}
```

This operation reads a single atomic snapshot of the selected Hermes singleton OAuth
state. It never creates a lock file, refreshes tokens, or writes authentication state.
An access token expiring within 120 seconds returns `refresh_required_read_only`.
Otherwise it performs only the authenticated model-catalog GET. Its result is:

```json
{
  "provider": "openai_oauth",
  "principal": {"account_id": "provider-account", "subject": "provider-subject"},
  "models": [{"model": "provider-returned-slug", "reasoning_efforts": ["high"]}]
}
```

`describe` has the same request/result shape but permits Hermes-owned refresh. Neither
operation synthesizes model IDs, context variants, reasoning vocabularies, or offline
catalog entries. Discovery must succeed against the selected account. Model names in
this document are placeholders, not claims about account entitlements.

## One completion

```json
{
  "operation": "complete",
  "provider": "openai_oauth",
  "principal": {"account_id": "provider-account", "subject": "provider-subject"},
  "model": "provider-returned-slug",
  "reasoning_effort": "high",
  "attempt_id": "attempt-identifier",
  "messages": [{"role": "user", "content": "The role request"}],
  "tools": [],
  "reasoning_history": []
}
```

Every completion obtains a fresh provider catalog under the same authenticated
principal and rejects model or reasoning values not explicitly advertised there.
The outbound Responses request contains the unchanged `model` and
`reasoning: {"effort": <selected value>}`, `store: false`, `stream: true`, and
`include: ["reasoning.encrypted_content"]`. There is exactly one inference POST;
no request retries, model aliases, provider substitutions, or auxiliary inference.

`complete_read_only` accepts the identical fields and returns the identical result,
but does not acquire/create the auth lock, refresh tokens, or write auth state.
It rejects tokens expiring within 120 seconds before any network request. This mode
allows a separately authorized smoke test to use a current token without changing
persistent authentication state; it is read-only with respect to authentication,
not with respect to provider inference usage.

The result echoes provider, principal, model, reasoning effort, and attempt ID, and
contains `message: {role: "assistant", content, tool_calls}`, `finish_reason` (`stop`
or `tool_calls`), and `reasoning_items`. The backend response model must equal the
requested model. Only text, reasoning items, and explicitly declared function calls
are accepted; server-side tools are not enabled. Responses failures, incomplete
streams, missing completions, duplicate completion events, and unknown output items
are errors.

Codex may omit output from the terminal frame. Finalized `response.output_item.done`
frames are therefore assembled by contiguous output index, with unique item IDs.
A successful terminal frame with the requested model is still required. Duplicate,
missing-index, post-terminal, or conflicting terminal/item data is rejected; partial
text deltas and unfinalized function calls are not promoted to complete output.

Messages accept text and standard function-tool messages. Leading system/developer
text becomes Responses instructions; later instruction changes are rejected.
Tools accept only `{type: "function", function: {name, description?, parameters}}`.
No tool runs inside this capability.

For a function-call continuation, retain the returned reasoning items in that
attempt's private state and supply:

```json
{
  "before_call_id": "the-first-tool-call-id-from-that-assistant-response",
  "items": [{"type": "reasoning", "id": "provider-item-id", "summary": [], "encrypted_content": "provider-encrypted-content"}]
}
```

Each entry belongs in `reasoning_history`. Its items are inserted immediately before
the corresponding assistant message's first function call. Missing or duplicate
anchors and duplicate function-call IDs fail closed. Reasoning items permit only
`type`, `id`, `summary` (`summary_text` objects), `encrypted_content`, and optional
`status` and `content`. Content is null or strict `reasoning_text` objects and is
preserved and checked for credential disclosure. Ordinary text responses return
an empty `reasoning_items` list. Callers
must not share this continuation history across attempts, roles, models, or principals.

## Authentication ownership

Hermes alone reads `<explicit home>/auth.json`, selecting only
`providers.openai-codex` with `auth_mode: chatgpt`. Pool-only credentials, API keys,
other providers, global stores, Codex stores, environment credentials, and config
credentials are not candidates. Unknown JWT account/subject identity fails closed.
Account and subject claims are taken from the selected token; the official TLS
provider endpoint must accept that token before discovery/inference succeeds.

Normal discovery/completion uses the existing `auth.lock` kernel protocol: a
nonblocking exclusive flock on POSIX or one byte at offset zero on Windows. The
lock covers read, optional refresh, atomic same-store persistence, catalog lookup,
and inference. Concurrent calls are serialized for a shared home. Separate homes
have separate locks. No process-global credential or role cache exists.

Refresh is one POST to the fixed OpenAI token endpoint using Hermes' OAuth client ID.
Both account and subject must remain unchanged and the returned token must have
sufficient lifetime before persistence. Same-principal replacement/reauthorization
is accepted; there are no grant IDs, generations, or grant-identity tracking.
Only the selected singleton token fields and refresh time are updated; unrelated
store fields and pool entries remain untouched. Failure never imports or retries
another credential. The selected pre/post-refresh token values are retained only
transiently to reject accidental disclosure in the normalized result; they are not
returned, hashed, tracked as identities, or added to separate storage.

HTTP uses the HTTPX client, fixed HTTPS endpoints, fixed Hermes
originator/version headers, disabled redirects, disabled environment trust (including
proxy/CA discovery), and transport retries zero. Error output excludes exception
details, request headers, provider error bodies, and validation input values.

## Validation

The focused tests drive the real CLI in a fresh isolated interpreter with synthetic
OAuth fixtures and an HTTP-level test transport, plus real OS lock interoperability,
same-account refresh, cross-account rejection, concurrent role/account calls, strict
request/response handling, reasoning continuation, and import isolation. They never
read private credential stores or send real inference requests. Live catalog or
inference results must be reported separately from these tests.

```text
scripts/run_tests.sh tests/hermes_cli/test_isolated_oauth*.py
```
