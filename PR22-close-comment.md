# PR #22 Close Comment — REJECT

## Summary

This PR forces HTTP/1.1 for ALL LLM API requests by adding `.http1_only()` to the
shared `build_reqwest_client()` function. The stated goal is to fix "error decoding
response body" errors on DeepSeek + CloudFront with HTTP/2.

## Why this should be REJECTED

### 1. Over-engineering: sledgehammer for a single-provider issue

The fix disables HTTP/2 for **all providers** (OpenAI, Gemini, Anthropic, OpenRouter)
to fix a problem that only affects **DeepSeek behind CloudFront**. This is like
fumigating the entire house because one room has a fly.

### 2. Incorrect assumption about Anthropic

The PR description states "No impact on other features (web_search, web_fetch use
separate clients)" and test plan says "Confirm no regression for Anthropic provider
(uses separate client path)". **Both claims are wrong.**

`AnthropicProvider::new()` in `src/provider.rs` (line 92) calls `build_reqwest_client()`,
so this change DOES affect Anthropic — contrary to the stated assumptions.

### 3. Unnecessary restriction with no benefit

For single-request LLM API calls, HTTP/2 multiplexing provides no benefit, but:
- Header compression (HPACK) is lost
- Connection reuse opportunities are lost
- Future features that could benefit from HTTP/2 are precluded

### 4. Better alternatives exist

- Make `.http1_only()` **provider-specific**: Only apply it to the DeepSeek client
- Make it **configurable**: Add an `http1_only` field to provider config
- Fix the **root cause**: DeepSeek's HTTP/2 framing issue may get fixed upstream

### 5. Not tested for all affected providers

The test plan claims Anthropic is unaffected, which is demonstrably false. The PR
was not tested against all providers it impacts.

## Recommendation

**Close this PR.** The correct fix should:
1. Only apply HTTP/1.1 to DeepSeek (or make it configurable per-provider)
2. Actually test Anthropic (which uses the same `build_reqwest_client()`)
3. Document why HTTP/1.1 is necessary for DeepSeek specifically

The merged PRs #24, #25, #26 already address the actual parsing/retry issues.
