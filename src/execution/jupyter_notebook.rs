//! Jupyter Server Contents API: append code cells to a server-side `.ipynb` (optional sync).

use serde_json::{json, Value};

use super::error::ExecutionError;

fn contents_path_url(base: &str, path: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    let enc: String = path
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|seg| urlencoding::encode(seg).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    format!("{base}/api/contents/{enc}")
}

/// Append one code cell to `path` (server-relative, e.g. `isanagent/abc123.ipynb`). Creates the notebook if missing.
pub async fn append_code_cell(
    client: &reqwest::Client,
    base_http: &str,
    _token: Option<&str>,
    path: &str,
    code: &str,
    mut apply_auth: impl FnMut(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
) -> Result<(), ExecutionError> {
    let url = contents_path_url(base_http, path);
    let mut get = client.get(&url);
    get = apply_auth(get);
    let resp = get
        .send()
        .await
        .map_err(|e| ExecutionError::Provider(format!("jupyter contents GET: {e}")))?;
    let mut nb = if resp.status() == reqwest::StatusCode::NOT_FOUND {
        json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": []
        })
    } else if !resp.status().is_success() {
        let st = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(ExecutionError::Provider(format!(
            "jupyter contents GET {path}: {st} {t}"
        )));
    } else {
        let body: Value = resp
            .json()
            .await
            .map_err(|e| ExecutionError::Provider(format!("jupyter contents JSON: {e}")))?;
        body.get("content").cloned().ok_or_else(|| {
            ExecutionError::Provider("jupyter contents response missing content".into())
        })?
    };

    let cells = nb
        .get_mut("cells")
        .and_then(|c| c.as_array_mut())
        .ok_or_else(|| ExecutionError::Provider("notebook content missing cells array".into()))?;
    let cell = json!({
        "cell_type": "code",
        "metadata": {},
        "source": split_source_lines(code),
        "outputs": [],
        "execution_count": null
    });
    cells.push(cell);

    let body = json!({
        "type": "notebook",
        "format": "json",
        "content": nb,
    });

    let mut put = client.put(&url).json(&body);
    put = apply_auth(put);
    let put_resp = put
        .send()
        .await
        .map_err(|e| ExecutionError::Provider(format!("jupyter contents PUT: {e}")))?;
    if !put_resp.status().is_success() {
        let st = put_resp.status();
        let t = put_resp.text().await.unwrap_or_default();
        return Err(ExecutionError::Provider(format!(
            "jupyter contents PUT {path}: {st} {t}"
        )));
    }
    Ok(())
}

fn split_source_lines(code: &str) -> Vec<Value> {
    if code.is_empty() {
        return vec![Value::String("\n".into())];
    }
    code.split_inclusive('\n')
        .map(|s| Value::String(s.to_string()))
        .collect()
}
