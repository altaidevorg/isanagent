import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ModelEntry = {
  key: string;
  provider_name: string;
  model_name: string;
  has_api_key: boolean;
  api_key_env: string;
  base_url: string;
};

type ModelList = {
  active_key: string | null;
  models: ModelEntry[];
};

type SelectModelResponse = {
  active_key: string;
  provider_name: string;
  model_name: string;
  keychain_saved?: boolean;
};

type SettingsTab = "models" | "harness" | "skills";

type ShellMode = "ask" | "deny" | "allow";

type HarnessSettings = {
  restrict_to_workspace: boolean;
  shell_mode: ShellMode;
  execution_enabled: boolean;
  subagents_enabled: boolean;
  builtin_tools_enabled: boolean;
  ml_engineer_enabled: boolean;
};

type HarnessResponse = HarnessSettings & {
  restart_required: boolean;
};

type InstalledSkill = {
  name: string;
  description: string;
  available: boolean;
  always: boolean;
};

type SkillList = {
  skills: InstalledSkill[];
};

type InstallSkillResponse = SkillList & {
  installed: string[];
};

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    if (body.error?.message) {
      return body.error.message;
    }
  } catch {
    /* ignore */
  }
  return `Request failed (${response.status})`;
}

export function SettingsDialog({
  onClose,
  onActiveModel,
}: {
  onClose: () => void;
  onActiveModel: (label: string) => void;
}) {
  const [tab, setTab] = useState<SettingsTab>("models");
  const [models, setModels] = useState<ModelList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [keyDrafts, setKeyDrafts] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [harness, setHarness] = useState<HarnessSettings | null>(null);
  const [harnessDraft, setHarnessDraft] = useState<HarnessSettings | null>(null);
  const [harnessLoading, setHarnessLoading] = useState(false);
  const [harnessError, setHarnessError] = useState<string | null>(null);
  const [harnessNotice, setHarnessNotice] = useState<string | null>(null);
  const [harnessSaving, setHarnessSaving] = useState(false);
  const [harnessLoaded, setHarnessLoaded] = useState(false);
  const [skills, setSkills] = useState<InstalledSkill[] | null>(null);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [skillsError, setSkillsError] = useState<string | null>(null);
  const [skillsNotice, setSkillsNotice] = useState<string | null>(null);
  const [skillsLoaded, setSkillsLoaded] = useState(false);
  const [skillsInstalling, setSkillsInstalling] = useState(false);
  const [repoDraft, setRepoDraft] = useState("");
  const [skillNameDraft, setSkillNameDraft] = useState("");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch("/v1/settings/models");
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        const body = (await response.json()) as ModelList;
        if (!cancelled) {
          setModels(body);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load models");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (tab !== "harness" || harnessLoaded) {
      return;
    }
    let cancelled = false;
    setHarnessLoading(true);
    void (async () => {
      try {
        const response = await fetch("/v1/settings/harness");
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        const body = (await response.json()) as HarnessResponse;
        const next = harnessFromResponse(body);
        if (!cancelled) {
          setHarness(next);
          setHarnessDraft(next);
        }
      } catch (err) {
        if (!cancelled) {
          setHarnessError(err instanceof Error ? err.message : "Could not load harness settings");
        }
      } finally {
        if (!cancelled) {
          setHarnessLoaded(true);
          setHarnessLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, harnessLoaded]);

  useEffect(() => {
    if (tab !== "skills" || skillsLoaded) {
      return;
    }
    let cancelled = false;
    setSkillsLoading(true);
    void (async () => {
      try {
        const response = await fetch("/v1/settings/skills");
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        const body = (await response.json()) as SkillList;
        if (!cancelled) {
          setSkills(body.skills);
        }
      } catch (err) {
        if (!cancelled) {
          setSkillsError(err instanceof Error ? err.message : "Could not load skills");
        }
      } finally {
        if (!cancelled) {
          setSkillsLoaded(true);
          setSkillsLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, skillsLoaded]);

  const groups = useMemo(() => {
    const byProvider = new Map<string, ModelEntry[]>();
    for (const model of models?.models ?? []) {
      const list = byProvider.get(model.provider_name) ?? [];
      list.push(model);
      byProvider.set(model.provider_name, list);
    }
    return [...byProvider.entries()];
  }, [models]);

  const selectModel = async (model: ModelEntry) => {
    setError(null);
    setNotice(null);
    setPendingKey(model.key);
    try {
      const draft = keyDrafts[model.provider_name]?.trim() ?? "";
      const response = await fetch("/v1/settings/model", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          key: model.key,
          ...(draft ? { api_key: draft } : {}),
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const body = (await response.json()) as SelectModelResponse;
      setModels((current) =>
        current
          ? {
              ...current,
              active_key: body.active_key,
              models: current.models.map((entry) =>
                entry.provider_name === body.provider_name
                  ? { ...entry, has_api_key: true }
                  : entry,
              ),
            }
          : current,
      );
      setKeyDrafts((current) => ({ ...current, [model.provider_name]: "" }));
      onActiveModel(body.model_name);
      if (body.keychain_saved === false) {
        setNotice("Model switched for this session. The key could not be saved to the OS keychain.");
      } else if (draft) {
        setNotice(
          `Using ${body.model_name}. This key covers every ${body.provider_name} model.`,
        );
      } else {
        setNotice(`Using ${body.model_name}. Later messages in this session use it.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not switch model");
    } finally {
      setPendingKey(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="flex h-full max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-lg">
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <div>
            <h2 id="settings-title" className="text-lg font-semibold tracking-[-0.02em] text-foreground">
              Settings
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">{settingsSubtitle(tab)}</p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="flex gap-2 border-b border-border px-5 py-3">
          <SettingsTabButton active={tab === "models"} onClick={() => setTab("models")}>
            Models
          </SettingsTabButton>
          <SettingsTabButton active={tab === "harness"} onClick={() => setTab("harness")}>
            Harness
          </SettingsTabButton>
          <SettingsTabButton active={tab === "skills"} onClick={() => setTab("skills")}>
            Skills
          </SettingsTabButton>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {tab === "models" ? (
            <ModelsPane
              loading={loading}
              error={error}
              notice={notice}
              groups={groups}
              activeKey={models?.active_key ?? null}
              pendingKey={pendingKey}
              keyDrafts={keyDrafts}
              onDraft={(key, value) => setKeyDrafts((current) => ({ ...current, [key]: value }))}
              onSelect={(model) => void selectModel(model)}
            />
          ) : null}
          {tab === "harness" ? (
            <HarnessPane
              loading={harnessLoading}
              error={harnessError}
              notice={harnessNotice}
              saved={harness}
              draft={harnessDraft}
              saving={harnessSaving}
              onChange={setHarnessDraft}
              onSave={() => {
                if (!harnessDraft) {
                  return;
                }
                setHarnessSaving(true);
                setHarnessError(null);
                setHarnessNotice(null);
                void (async () => {
                  try {
                    const response = await fetch("/v1/settings/harness", {
                      method: "PATCH",
                      headers: { "content-type": "application/json" },
                      body: JSON.stringify(harnessDraft),
                    });
                    if (!response.ok) {
                      throw new Error(await readError(response));
                    }
                    const body = (await response.json()) as HarnessResponse;
                    const next = harnessFromResponse(body);
                    setHarness(next);
                    setHarnessDraft(next);
                    onClose();
                  } catch (err) {
                    setHarnessError(err instanceof Error ? err.message : "Could not save harness settings");
                  } finally {
                    setHarnessSaving(false);
                  }
                })();
              }}
            />
          ) : null}
          {tab === "skills" ? (
            <SkillsPane
              loading={skillsLoading}
              error={skillsError}
              notice={skillsNotice}
              skills={skills}
              installing={skillsInstalling}
              repo={repoDraft}
              skillName={skillNameDraft}
              onRepo={setRepoDraft}
              onSkillName={setSkillNameDraft}
              onInstall={() => {
                const repo = repoDraft.trim();
                if (!repo) {
                  return;
                }
                setSkillsInstalling(true);
                setSkillsError(null);
                setSkillsNotice(null);
                void (async () => {
                  try {
                    const skillName = skillNameDraft.trim();
                    const response = await fetch("/v1/settings/skills/install", {
                      method: "POST",
                      headers: { "content-type": "application/json" },
                      body: JSON.stringify({
                        repo,
                        ...(skillName ? { skill_name: skillName } : {}),
                      }),
                    });
                    if (!response.ok) {
                      throw new Error(await readError(response));
                    }
                    const body = (await response.json()) as InstallSkillResponse;
                    setSkills(body.skills);
                    setRepoDraft("");
                    setSkillNameDraft("");
                    setSkillsNotice(
                      body.installed.length === 0
                        ? "No skills found in that repository."
                        : `Installed ${body.installed.join(", ")}. The next message can use them.`,
                    );
                  } catch (err) {
                    setSkillsError(err instanceof Error ? err.message : "Could not install skills");
                  } finally {
                    setSkillsInstalling(false);
                  }
                })();
              }}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function settingsSubtitle(tab: SettingsTab): string {
  if (tab === "harness") {
    return "Harness changes are saved to config.toml and apply the next time isanagent starts.";
  }
  if (tab === "skills") {
    return "Installed skills are available on the next message. Add a GitHub repo or owner/repo shorthand.";
  }
  return "One API key covers every model from that provider. Active is the model the next message uses.";
}

function harnessFromResponse(body: HarnessResponse): HarnessSettings {
  return {
    restrict_to_workspace: body.restrict_to_workspace,
    shell_mode: body.shell_mode,
    execution_enabled: body.execution_enabled,
    subagents_enabled: body.subagents_enabled,
    builtin_tools_enabled: body.builtin_tools_enabled,
    ml_engineer_enabled: body.ml_engineer_enabled,
  };
}

function SettingsTabButton({
  active,
  disabled,
  title,
  onClick,
  children,
}: {
  active?: boolean;
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  children: string;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "rounded-md px-3 py-1.5 text-sm",
        active ? "bg-accent text-foreground" : "text-muted-foreground",
        disabled ? "cursor-not-allowed opacity-50" : "hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function ModelsPane({
  loading,
  error,
  notice,
  groups,
  activeKey,
  pendingKey,
  keyDrafts,
  onDraft,
  onSelect,
}: {
  loading: boolean;
  error: string | null;
  notice: string | null;
  groups: Array<[string, ModelEntry[]]>;
  activeKey: string | null;
  pendingKey: string | null;
  keyDrafts: Record<string, string>;
  onDraft: (key: string, value: string) => void;
  onSelect: (model: ModelEntry) => void;
}) {
  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading models…</p>;
  }
  return (
    <div className="space-y-4">
      {error ? (
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      ) : null}
      {notice ? (
        <p className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm text-foreground">{notice}</p>
      ) : null}
      {groups.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No providers in config.toml. Add a [providers.*] block, then reopen settings.
        </p>
      ) : (
        groups.map(([providerName, providerModels]) => {
          const ready = providerModels.some((model) => model.has_api_key);
          const envName = providerModels[0]?.api_key_env ?? "API_KEY";
          const draft = keyDrafts[providerName]?.trim() ?? "";
          return (
            <section key={providerName} className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-[0.05em] text-muted-foreground">
                {providerName}
              </h3>
              <p className="text-xs text-muted-foreground">
                {ready
                  ? `Key ready. It covers every ${providerName} model.`
                  : `One $${envName} covers every ${providerName} model.`}
              </p>
              {ready ? null : (
                <input
                  type="password"
                  autoComplete="off"
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                  placeholder="API key"
                  value={keyDrafts[providerName] ?? ""}
                  onChange={(event) => onDraft(providerName, event.target.value)}
                />
              )}
              <ul className="space-y-2">
                {providerModels.map((model) => {
                  const active = model.key === activeKey;
                  return (
                    <li key={model.key} className="rounded-lg border border-border px-3 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="min-w-0 truncate text-sm font-medium text-foreground">
                          {model.model_name}
                        </p>
                        <Button
                          type="button"
                          size="sm"
                          variant={active ? "secondary" : "outline"}
                          disabled={pendingKey !== null || (!ready && !draft)}
                          onClick={() => onSelect(model)}
                        >
                          {pendingKey === model.key ? "Switching…" : active && ready ? "Active" : "Use"}
                        </Button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </section>
          );
        })
      )}
    </div>
  );
}

function HarnessPane({
  loading,
  error,
  notice,
  saved,
  draft,
  saving,
  onChange,
  onSave,
}: {
  loading: boolean;
  error: string | null;
  notice: string | null;
  saved: HarnessSettings | null;
  draft: HarnessSettings | null;
  saving: boolean;
  onChange: (next: HarnessSettings) => void;
  onSave: () => void;
}) {
  if (loading || !draft) {
    return <p className="text-sm text-muted-foreground">{error ?? "Loading harness settings…"}</p>;
  }
  const dirty = JSON.stringify(saved) !== JSON.stringify(draft);
  return (
    <div className="space-y-4">
      {error ? (
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      ) : null}
      {notice ? (
        <p className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm text-foreground">{notice}</p>
      ) : null}
      <HarnessToggle
        title="Sandbox file tools"
        detail="Keep reads and writes inside the workspace."
        checked={draft.restrict_to_workspace}
        onChange={(checked) => onChange({ ...draft, restrict_to_workspace: checked })}
      />
      <label className="block rounded-lg border border-border px-3 py-3">
        <span className="text-sm font-medium text-foreground">Shell policy</span>
        <span className="mt-1 block text-xs text-muted-foreground">
          What happens before a risky exec command in an interactive session.
        </span>
        <select
          className="mt-3 h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
          value={draft.shell_mode}
          onChange={(event) =>
            onChange({ ...draft, shell_mode: event.target.value as ShellMode })
          }
        >
          <option value="ask">Ask</option>
          <option value="deny">Deny</option>
          <option value="allow">Allow</option>
        </select>
      </label>
      <HarnessToggle
        title="Execution harness"
        detail="Register execution tools for local Python, Jupyter, and SSH."
        checked={draft.execution_enabled}
        onChange={(checked) => onChange({ ...draft, execution_enabled: checked })}
      />
      <HarnessToggle
        title="Subagents"
        detail="Allow named background agents and task tools."
        checked={draft.subagents_enabled}
        onChange={(checked) => onChange({ ...draft, subagents_enabled: checked })}
      />
      <HarnessToggle
        title="Builtin tools"
        detail="Filesystem, web, exec, cron, memory, and skill loading."
        checked={draft.builtin_tools_enabled}
        onChange={(checked) => onChange({ ...draft, builtin_tools_enabled: checked })}
      />
      <HarnessToggle
        title="ML engineer overlay"
        detail="Append the ML policy to the system prompt."
        checked={draft.ml_engineer_enabled}
        onChange={(checked) => onChange({ ...draft, ml_engineer_enabled: checked })}
      />
      <div className="flex justify-end">
        <Button type="button" size="sm" disabled={!dirty || saving} onClick={onSave}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
    </div>
  );
}

function SkillsPane({
  loading,
  error,
  notice,
  skills,
  installing,
  repo,
  skillName,
  onRepo,
  onSkillName,
  onInstall,
}: {
  loading: boolean;
  error: string | null;
  notice: string | null;
  skills: InstalledSkill[] | null;
  installing: boolean;
  repo: string;
  skillName: string;
  onRepo: (value: string) => void;
  onSkillName: (value: string) => void;
  onInstall: () => void;
}) {
  if (loading || !skills) {
    return <p className="text-sm text-muted-foreground">{error ?? "Loading skills…"}</p>;
  }
  return (
    <div className="space-y-4">
      {error ? (
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      ) : null}
      {notice ? (
        <p className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm text-foreground">{notice}</p>
      ) : null}
      <form
        className="space-y-3 rounded-lg border border-border px-3 py-3"
        onSubmit={(event) => {
          event.preventDefault();
          onInstall();
        }}
      >
        <div>
          <p className="text-sm font-medium text-foreground">Install from a repository</p>
          <p className="mt-1 text-xs text-muted-foreground">
            owner/repo, or a git URL. Leave the skill name empty to install every skill in the repo.
          </p>
        </div>
        <input
          className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
          placeholder="owner/repo"
          value={repo}
          onChange={(event) => onRepo(event.target.value)}
        />
        <input
          className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
          placeholder="Skill name (optional)"
          value={skillName}
          onChange={(event) => onSkillName(event.target.value)}
        />
        <div className="flex justify-end">
          <Button type="submit" size="sm" disabled={installing || repo.trim() === ""}>
            {installing ? "Installing…" : "Install"}
          </Button>
        </div>
      </form>
      {skills.length === 0 ? (
        <p className="text-sm text-muted-foreground">No skills installed yet.</p>
      ) : (
        <ul className="space-y-2">
          {skills.map((skill) => (
            <li key={skill.name} className="rounded-lg border border-border px-3 py-3">
              <div className="flex items-center justify-between gap-3">
                <p className="truncate text-sm font-medium text-foreground">{skill.name}</p>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {skill.available ? (skill.always ? "Always on" : "Available") : "Unavailable"}
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{skill.description}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HarnessToggle({
  title,
  detail,
  checked,
  onChange,
}: {
  title: string;
  detail: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-4 rounded-lg border border-border px-3 py-3">
      <span>
        <span className="block text-sm font-medium text-foreground">{title}</span>
        <span className="mt-1 block text-xs text-muted-foreground">{detail}</span>
      </span>
      <input
        type="checkbox"
        className="h-4 w-4 accent-current"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}
