use clap::{Args as ClapArgs, Parser, Subcommand};
use isanagent::onboarding::{
    build_interactive_config_toml, onboard_workspace, BootstrapReport, OnboardOptions,
};
use isanagent::onboarding_interactive;
use isanagent::skills::SkillRegistry;
use isanagent::workspace::{resolve_workspace_root, IsanagentWorkspace};

/// isanagent: A terminal chat interface and autonomous agent engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Optional explicit path to the workspace directory. Defaults to ~/.isanagent
    #[arg(short, long)]
    workspace: Option<String>,

    /// Optional path to a config.toml file. Defaults to <workspace>/config.toml
    #[arg(short, long)]
    config: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run as an Agent Client Protocol (ACP) server over stdio
    Acp(AcpArgs),
    /// Create workspace layout and starter files; optional flags override generated config.toml
    Onboard(OnboardArgs),
    /// Manage harness packs and plugins (install, list, remove)
    Pack(PackArgs),
    /// Manage skills (add, list, etc.)
    Skills(SkillsArgs),
}

#[derive(ClapArgs, Debug)]
struct PackArgs {
    #[command(subcommand)]
    command: PackCommands,
}

#[derive(Subcommand, Debug)]
enum PackCommands {
    /// Install a harness pack or plugin from a Git repository
    Install {
        /// Repository URL (e.g., https://github.com/altaidevorg/pack-ml-engineer or owner/repo)
        source: String,
        /// Optional custom name for the installed plugin
        #[arg(short, long)]
        name: Option<String>,
        /// Install globally to ~/.isanagent/plugins instead of the workspace
        #[arg(short, long)]
        global: bool,
    },
    /// List all discovered plugins and harness packs
    List,
    /// Remove an installed plugin from the workspace or global directory
    Remove {
        /// Name of the plugin to remove
        name: String,
        /// Remove from the global directory (~/.isanagent/plugins)
        #[arg(short, long)]
        global: bool,
    },
}

#[derive(ClapArgs, Debug, Default)]
struct AcpArgs {
    /// Optional explicit path to the workspace directory. Defaults to ~/.isanagent
    #[arg(short, long)]
    workspace: Option<String>,

    /// Optional path to a config.toml file. Defaults to <workspace>/config.toml
    #[arg(short, long)]
    config: Option<String>,
}

#[derive(ClapArgs, Debug)]
struct SkillsArgs {
    #[command(subcommand)]
    command: SkillCommands,
}

#[derive(Subcommand, Debug)]
enum SkillCommands {
    /// Add skills from a remote GitHub repository
    Add {
        /// Repository URL (e.g., https://github.com/vercel-labs/skills) or shorthand (owner/repo)
        repo_url: String,
        /// Optional specific skill name to install
        #[arg(short, long)]
        skill: Option<String>,
    },
    /// List all installed skills
    List,
}

#[derive(ClapArgs, Debug)]
struct OnboardArgs {
    /// Optional explicit path to the workspace directory. Defaults to ~/.isanagent
    #[arg(short, long)]
    workspace: Option<String>,
    /// Textual wizard (ratatui): provider → optional base URL → API key env var name → pick model from /models
    #[arg(long)]
    interactive: bool,
    /// Override embedded defaults for `config.toml` (see `isanagent onboard --help`)
    #[command(flatten)]
    options: OnboardOptions,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Acp(args)) => isanagent::host::start_host(isanagent::host::HostConfig {
            workspace: cli
                .workspace
                .or(args.workspace)
                .map(std::path::PathBuf::from),
            config: cli.config.or(args.config).map(std::path::PathBuf::from),
            acp_mode: true,
            ..Default::default()
        })
        .await
        .map_err(|e| e as Box<dyn std::error::Error>),
        Some(Commands::Onboard(args)) => run_onboard(cli.workspace, args).await,
        Some(Commands::Pack(args)) => run_pack(cli.workspace, args).await,
        Some(Commands::Skills(args)) => run_skills(cli.workspace, args).await,
        None => {
            // First-run UX: when the user invokes `isanagent` with no `--workspace` and the
            // default `~/.isanagent` directory does not yet exist, auto-launch the interactive
            // onboard wizard before starting the agent. Subsequent runs see the directory and
            // skip straight to `start_embedded_host`.
            if cli.workspace.is_none() {
                let default_root = resolve_workspace_root(None);
                if !default_root.exists() {
                    auto_onboard_then_run(cli.config).await?;
                    return Ok(());
                }
            }
            start_embedded_host(cli.workspace, cli.config).await
        }
    }
}

/// Runs the interactive onboard against the default workspace path then transitions into
/// `start_embedded_host` in the same invocation. Cancelling the wizard (Ctrl+C / Esc) returns
/// `Ok(())` without launching the agent so the user can retry on the next run.
async fn auto_onboard_then_run(
    config_arg: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to isanagent. No workspace detected at the default location.");
    println!("Launching the interactive onboard wizard...");
    println!();

    let onboard_result = run_onboard_inner(
        None,
        OnboardArgs {
            workspace: None,
            interactive: true,
            options: OnboardOptions::default(),
        },
        /* chained = */ true,
    )
    .await;

    match onboard_result {
        Ok(()) => {
            println!();
            println!("Workspace ready. Launching isanagent...");
            println!();
            start_embedded_host(None, config_arg).await
        }
        Err(e) => {
            // The interactive wizard signalled abort (Ctrl+C / Esc) or a concrete failure.
            // Surface the message and exit cleanly so the shell prompt returns; the user can
            // re-run when ready.
            eprintln!("Onboard did not complete: {e}");
            eprintln!("Run `isanagent onboard --interactive` to try again.");
            Ok(())
        }
    }
}

async fn start_embedded_host(
    workspace_arg: Option<String>,
    config_arg: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    isanagent::host::start_host(isanagent::host::HostConfig {
        workspace: workspace_arg.map(std::path::PathBuf::from),
        config: config_arg.map(std::path::PathBuf::from),
        ..Default::default()
    })
    .await
    .map_err(std::io::Error::other)?;
    Ok(())
}

async fn run_pack(
    workspace_arg: Option<String>,
    args: PackArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = IsanagentWorkspace::new(workspace_arg.as_deref(), None)?;
    let global_root = resolve_workspace_root(None);

    match args.command {
        PackCommands::List => {
            let registry =
                isanagent::plugins::PluginRegistry::discover(&workspace.dir, Some(&global_root));
            if registry.is_empty() {
                println!("No harness packs or plugins installed.");
                println!("Install with: isanagent pack install <repo_url>");
            } else {
                println!("Installed Harness Packs & Plugins ({}):", registry.len());
                for p in registry.list() {
                    let version = p.manifest.version.as_deref().unwrap_or("0.1.0");
                    let desc = p.manifest.description.as_deref().unwrap_or("");
                    println!("  • {} (v{}) - {}", p.name, version, desc);
                    if let Some(ref agents_dir) = p.agents_dir {
                        println!("    └─ Agents: {}", agents_dir.display());
                    }
                    if let Some(ref skills_dir) = p.skills_dir {
                        println!("    └─ Skills: {}", skills_dir.display());
                    }
                }
            }
        }
        PackCommands::Install {
            source,
            name,
            global,
        } => {
            let target_dir = if global {
                global_root.join("plugins")
            } else {
                workspace.dir.join(".agents").join("plugins")
            };
            println!(
                "Installing harness pack from {source} into {}...",
                target_dir.display()
            );
            match isanagent::plugins::PluginRegistry::install_from_repo(
                &target_dir,
                &source,
                name.as_deref(),
            )
            .await
            {
                Ok(plugin) => {
                    println!(
                        "Successfully installed pack '{}' (v{})",
                        plugin.name,
                        plugin.manifest.version.as_deref().unwrap_or("0.1.0")
                    );
                }
                Err(e) => {
                    return Err(format!("Error installing harness pack: {e}").into());
                }
            }
        }
        PackCommands::Remove { name, global } => {
            let clean_name = name.trim();
            if clean_name.is_empty()
                || clean_name.contains('/')
                || clean_name.contains('\\')
                || clean_name == "."
                || clean_name == ".."
            {
                return Err(format!(
                    "Invalid plugin name '{name}': must not contain path separators or parent directory references"
                )
                .into());
            }
            let target_dir = if global {
                global_root.join("plugins").join(clean_name)
            } else {
                workspace
                    .dir
                    .join(".agents")
                    .join("plugins")
                    .join(clean_name)
            };
            if target_dir.exists() {
                std::fs::remove_dir_all(&target_dir)?;
                println!(
                    "Successfully removed plugin '{}' from {}",
                    clean_name,
                    target_dir.display()
                );
            } else {
                println!(
                    "Plugin '{}' not found at {}",
                    clean_name,
                    target_dir.display()
                );
            }
        }
    }

    Ok(())
}

async fn run_skills(
    workspace_arg: Option<String>,
    args: SkillsArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = IsanagentWorkspace::new(workspace_arg.as_deref(), None)?;
    let mut skills = SkillRegistry::new(workspace.skills_path());

    match args.command {
        SkillCommands::Add { repo_url, skill } => {
            if let Some(ref name) = skill {
                println!("Adding skill '{name}' from {repo_url}...");
            } else {
                println!("Adding all skills from {repo_url}...");
            }
            match skills
                .install_skills_from_repo(&repo_url, skill.as_deref())
                .await
            {
                Ok(installed) => {
                    if installed.is_empty() {
                        println!("No skills found in the repository.");
                    } else {
                        println!("Successfully installed {} skills:", installed.len());
                        for name in installed {
                            println!("  - {name}");
                        }
                    }
                }
                Err(e) => {
                    return Err(format!("Error installing skills: {e}").into());
                }
            }
        }
        SkillCommands::List => {
            println!("{}", skills.format_skill_directory());
        }
    }

    Ok(())
}

async fn run_onboard(
    global_workspace: Option<String>,
    args: OnboardArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    run_onboard_inner(global_workspace, args, /* chained = */ false).await
}

/// Underlying onboard implementation. When `chained` is true the final "Run: isanagent" tip is
/// suppressed because the caller is about to launch the agent in the same process.
async fn run_onboard_inner(
    global_workspace: Option<String>,
    args: OnboardArgs,
    chained: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_arg = args.workspace.or(global_workspace);

    if args.interactive && args.options.has_overrides() {
        return Err(std::io::Error::other(
            "Cannot combine --interactive with other config override flags; run `onboard --interactive` alone.",
        )
        .into());
    }

    let interactive_outcome = if args.interactive {
        let handle = tokio::runtime::Handle::current();
        Some(
            tokio::task::spawn_blocking(move || {
                onboarding_interactive::run_interactive_collect(&handle)
            })
            .await?
            .map_err(std::io::Error::other)?,
        )
    } else {
        None
    };

    let options = interactive_outcome
        .as_ref()
        .map(|o| o.options.clone())
        .unwrap_or_else(|| args.options);

    let config_overrides_used = options.has_overrides();

    let interactive_merged_toml = if interactive_outcome.is_some() {
        Some(build_interactive_config_toml(&options).map_err(std::io::Error::other)?)
    } else {
        None
    };

    let options_for_workspace = options.clone();
    let report = tokio::task::spawn_blocking(move || {
        let workspace_root = resolve_workspace_root(workspace_arg.as_deref());
        onboard_workspace(
            &workspace_root,
            &options_for_workspace,
            interactive_merged_toml.as_deref(),
        )
    })
    .await?
    .map_err(std::io::Error::other)?;

    let env_name = interactive_outcome
        .as_ref()
        .and_then(|c| c.options.provider_api_key_env.clone());
    print_onboarding_report(&report, config_overrides_used, env_name.as_deref(), chained);
    Ok(())
}

fn print_onboarding_report(
    report: &BootstrapReport,
    config_overrides_used: bool,
    api_key_env: Option<&str>,
    chained: bool,
) {
    println!("Workspace onboarded at {}", report.root.display());
    println!();

    if !report.created.is_empty() {
        println!("Created:");
        for path in &report.created {
            println!("- {}", path.display());
        }
        println!();
    }

    if !report.skipped.is_empty() {
        println!("Skipped:");
        for path in &report.skipped {
            println!("- {}", path.display());
        }
        println!();
    }

    if config_overrides_used {
        println!(
            "Note: config.toml was generated from merged settings (template comments were omitted)."
        );
        println!();
    }

    println!("Next steps:");
    match api_key_env {
        Some(env) => {
            println!(
                "1. Ensure {env} is set in your environment (see config.toml provider.api_key_env)"
            );
        }
        None => {
            println!("1. Set GEMINI_API_KEY (or the env named in provider.api_key_env)");
        }
    }
    println!("2. Update <changethis> placeholders or disable unused channels in config.toml");
    if !chained {
        // When the agent is about to launch in the same invocation, suppress the redundant
        // "Run:" line so the user sees one transition message instead of two competing tips.
        println!("3. Run: {}", format_next_steps_run_line(&report.root));
    }
}

/// Build the `Run:` line for the onboarding banner. When `report_root` resolves to the same path
/// as the default (`~/.isanagent`), the `--workspace` flag is redundant and is omitted so the
/// user sees the cleanest invocation that will work.
fn format_next_steps_run_line(report_root: &std::path::Path) -> String {
    let default_root = isanagent::workspace::resolve_workspace_root(None);
    let same = paths_equivalent(report_root, &default_root);
    if same {
        "isanagent".to_string()
    } else {
        format!("isanagent --workspace {}", report_root.display())
    }
}

/// Compare two paths after best-effort canonicalization. Falls back to direct equality when
/// canonicalize fails (e.g. one of the paths is on a not-yet-existing filesystem branch).
fn paths_equivalent(a: &std::path::Path, b: &std::path::Path) -> bool {
    let canon_a = std::fs::canonicalize(a).ok();
    let canon_b = std::fs::canonicalize(b).ok();
    match (canon_a, canon_b) {
        (Some(a), Some(b)) => a == b,
        _ => a == b,
    }
}

#[cfg(test)]
mod next_steps_tests {
    use super::*;

    #[test]
    fn omits_workspace_flag_when_root_is_default() {
        let default_root = isanagent::workspace::resolve_workspace_root(None);
        let line = format_next_steps_run_line(&default_root);
        assert_eq!(line, "isanagent", "got {line}");
    }

    #[test]
    fn includes_workspace_flag_for_custom_root() {
        let custom = std::env::temp_dir().join("isanagent-next-steps-test-custom");
        let line = format_next_steps_run_line(&custom);
        assert!(
            line.starts_with("isanagent --workspace "),
            "expected --workspace prefix, got {line}"
        );
        assert!(
            line.contains(custom.to_string_lossy().as_ref()),
            "got {line}"
        );
    }
}
