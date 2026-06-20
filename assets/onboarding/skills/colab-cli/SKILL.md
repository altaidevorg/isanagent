---
name: colab-cli
description: Manage Google Colab sessions and execute code on remote Colab VMs via the colab-cli.
---

# Colab Session Operator

This skill allows an agent to provision, manage, and execute code on Google Colab environments using the `colab-cli`. It is the preferred way for agents to interact with Colab, supporting both simple scripts and complex ML engineering workflows.

## Key Capabilities

- **Provisioning:** Create new Colab sessions with specific hardware (CPU, GPU - A100/L4/T4, or TPU - v6e).
- **Execution:** Run local Python scripts or shell commands on the remote Colab VM.
- **Automation:** Handles authentication (browser-based or token), Google Drive mounting, and package installation.
- **Artifacts:** Captures session history as Jupyter notebooks and intercepts image outputs (e.g., from matplotlib).
- **Management:** Monitor session status, list active sessions, and stop VMs to conserve compute units.

## Tooling

The `colab-cli` provides a command-line interface that you can invoke via `exec` or other shell tools.

### Core Commands

- `colab init`: Initialize the CLI and authenticate.
- `colab start`: Provision a new session.
- `colab exec -f <script.py>`: Execute a local script on the remote VM.
- `colab shell "<command>"`: Run a shell command on the remote VM.
- `colab list`: List active sessions.
- `colab stop <session_id>`: Stop and delete a session.

## Usage Guidelines

- **Authentication:** In a local terminal, the user may need to perform a one-time OAuth flow. For autonomous agents, ensure `COLAB_API_KEY` or similar is set if required, or assume the environment is already initialized.
- **Non-interactive:** Avoid commands that require interactive input (like `colab repl`). Always use `-f` for scripts or direct strings for shell commands.
- **Mounting Drive:** Use `colab shell "python -c 'from google.colab import drive; drive.mount(\"/content/drive\")'"` if you need Google Drive access.
- **Artifact Extraction:** After running a script that generates images, `colab-cli` may provide paths to extracted artifacts. Check the command output.
- **Cleanup:** Always run `colab stop` when the task is complete to avoid wasting the user's Colab compute units.

## TPU provisioning (JAX / Pallas)

For MaxEvolve kernel profiling on Google TPUs:

1. `colab start --tpu tpuv6e` (or `tpuv5e` when available in your account)
2. Install JAX on the remote VM (example shell one-liner):
   `colab shell "pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"`
3. Upload or sync `profile_script.py` and run: `colab exec -f profile_script.py`
4. Capture `RESULT_LATENCY_MS=` / `RESULT_TFLOPS=` lines for `kernel_db_insert`
5. **`colab stop`** when done

For GPU profiling (Hopper/L4/A100), use `colab start --gpu` and install CUDA-enabled `jax[cuda12]`.

## Configuration

Local configuration and session state are stored in `~/.config/colab-cli/`. You can inspect `sessions.json` there to see persistent session metadata.
