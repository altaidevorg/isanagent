import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { useDialog } from "@/lib/useDialog";
import { cn } from "@/lib/utils";

type KeySource = "env" | "keychain" | "session" | "none";

type ModelEntry = {
  key: string;
  provider_name: string;
  model_name: string;
  has_api_key: boolean;
  key_source?: KeySource;
  api_key_env: string;
  base_url: string;
};

type ModelList = {
  /** Runtime default the agent uses for other API clients. */
  active_key: string | null;
  /** Model chosen in this web settings screen since the server started; null after a restart. */
  selected_key: string | null;
  models: ModelEntry[];
};

type SelectModelResponse = {
  active_key: string;
  selected_key?: string;
  provider_name: string;
  model_name: string;
  keychain_saved?: boolean;
};

type ProviderKeyResponse = ModelList & {
  keychain_saved?: boolean;
  applied_to_active: boolean;
};

type Notice = { kind: "success" | "warning" | "error" | "info"; text: string };

type ProviderGroup = {
  name: string;
  models: ModelEntry[];
  source: KeySource;
  envName: string;
};

type SettingsTab = "models" | "harness" | "skills";

type ShellMode = "ask" | "deny" | "allow";

type PermissionMode = "allow" | "ask" | "deny";

type HarnessSettings = {
  inside_workspace: PermissionMode;
  outside_workspace: PermissionMode;
  shell_mode: ShellMode;
  execution_enabled: boolean;
  subagents_enabled: boolean;
  builtin_tools_enabled: boolean;
  ml_engineer_enabled: boolean;
};

type HarnessResponse = HarnessSettings & {
  restrict_to_workspace?: boolean;
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
  notice,
  onClose,
  onActiveModel,
}: {
  /** Short note shown on the Models tab, e.g. why the dialog opened. */
  notice?: string | null;
  onClose: () => void;
  onActiveModel: (label: string) => void;
}) {
  const [tab, setTab] = useState<SettingsTab>("models");
  const modelSettings = useModelSettings(onActiveModel);
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
  // Escape is ignored while a save is in flight so its result is not lost.
  const dialogRef = useDialog(true, onClose, !modelSettings.busy && !harnessSaving && !skillsInstalling);

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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 outline-none"
      ref={dialogRef}
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
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
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
        {tab === "models" ? <ModelsPane settings={modelSettings} notice={notice ?? null} /> : null}
        {tab === "models" ? null : (
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
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
                    if (body.restart_required) {
                      // Keep the dialog open so the user sees that the change is not live yet.
                      setHarnessNotice("Saved to config.toml. Restart isanagent for these changes to take effect.");
                    } else {
                      onClose();
                    }
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
        )}
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
  return "Choose the model new messages use. One API key covers every model from a provider.";
}

function harnessFromResponse(body: HarnessResponse): HarnessSettings {
  return {
    inside_workspace: body.inside_workspace ?? "allow",
    outside_workspace: body.outside_workspace ?? (body.restrict_to_workspace === false ? "allow" : "ask"),
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

const MODEL_FILTER_THRESHOLD = 8;

const SOURCE_RANK: Record<KeySource, number> = { env: 3, keychain: 2, session: 1, none: 0 };

function keySourceOf(model: ModelEntry): KeySource {
  return model.key_source ?? (model.has_api_key ? "keychain" : "none");
}

function domId(prefix: string, value: string): string {
  return `${prefix}-${value.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

function keySourceLabel(source: KeySource, envName: string): string {
  switch (source) {
    case "env":
      return `Key from $${envName}`;
    case "keychain":
      return "Key saved in keychain";
    case "session":
      return "Key for this session only";
    default:
      return "No API key";
  }
}

const NOTICE_STYLES: Record<Notice["kind"], string> = {
  success:
    "border-emerald-600/30 bg-emerald-500/10 text-emerald-900 dark:border-emerald-400/30 dark:text-emerald-200",
  warning:
    "border-amber-600/40 bg-amber-500/10 text-amber-900 dark:border-amber-400/30 dark:text-amber-200",
  error:
    "border-destructive/40 bg-destructive/10 text-destructive dark:text-red-300",
  info: "border-border bg-muted/40 text-foreground",
};

function NoticeLine({ notice, className }: { notice: Notice; className?: string }) {
  return (
    <p
      role={notice.kind === "error" ? "alert" : "status"}
      className={cn("rounded-lg border px-3 py-2 text-sm", NOTICE_STYLES[notice.kind], className)}
    >
      {notice.text}
    </p>
  );
}

type KeyEditorState = { provider: string; mode: "add" | "replace" };

type ModelSettings = ReturnType<typeof useModelSettings>;

/** State and requests for the Models tab. Lives in the dialog so tab switches keep it. */
function useModelSettings(onActiveModel: (label: string) => void) {
  const [list, setList] = useState<ModelList | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [switching, setSwitching] = useState(false);
  const [footerNotice, setFooterNotice] = useState<Notice | null>(null);
  const [editor, setEditor] = useState<KeyEditorState | null>(null);
  const [confirmRemove, setConfirmRemove] = useState<string | null>(null);
  const [keyBusy, setKeyBusy] = useState<string | null>(null);
  const [providerNotices, setProviderNotices] = useState<Record<string, Notice>>({});
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    void (async () => {
      try {
        const response = await fetch("/v1/settings/models");
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        const body = (await response.json()) as ModelList;
        if (!cancelled) {
          const current = body.selected_key ?? null;
          setList({ active_key: body.active_key, selected_key: current, models: body.models });
          // The radio starts on the model chosen in this session; nothing after a restart.
          setSelectedKey(current);
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : "Could not load models");
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
  }, [reloadToken]);

  const models = useMemo(() => list?.models ?? [], [list]);
  // "Current" is the model chosen in this session (selected_key), never the runtime
  // default left over from an earlier run.
  const currentKey = list?.selected_key ?? null;
  const currentModel = models.find((model) => model.key === currentKey) ?? null;
  const selectedModel = models.find((model) => model.key === selectedKey) ?? null;
  const pendingModel =
    selectedModel && selectedModel.key !== currentKey && selectedModel.has_api_key ? selectedModel : null;
  const anyUsableKey = models.some((model) => model.has_api_key);

  const groups = useMemo<ProviderGroup[]>(() => {
    const byProvider = new Map<string, ModelEntry[]>();
    for (const model of models) {
      const entries = byProvider.get(model.provider_name) ?? [];
      entries.push(model);
      byProvider.set(model.provider_name, entries);
    }
    return [...byProvider.entries()].map(([name, entries]) => {
      let best = entries[0];
      for (const entry of entries) {
        if (SOURCE_RANK[keySourceOf(entry)] > SOURCE_RANK[keySourceOf(best)]) {
          best = entry;
        }
      }
      return {
        name,
        models: entries,
        source: keySourceOf(best),
        envName: best.api_key_env,
      };
    });
  }, [models]);

  const applyList = useCallback((next: ModelList) => {
    const nextCurrent = next.selected_key ?? null;
    setList({ active_key: next.active_key, selected_key: nextCurrent, models: next.models });
    setSelectedKey((current) => {
      const still = next.models.find((model) => model.key === current);
      return still && still.has_api_key ? current : nextCurrent;
    });
  }, []);

  const setProviderNotice = (provider: string, notice: Notice | null) => {
    setProviderNotices((current) => {
      const next = { ...current };
      if (notice) {
        next[provider] = notice;
      } else {
        delete next[provider];
      }
      return next;
    });
  };

  const select = (key: string) => {
    setSelectedKey(key);
    setFooterNotice(null);
  };

  const switchToSelected = async () => {
    if (!pendingModel) {
      return;
    }
    setSwitching(true);
    setFooterNotice(null);
    try {
      const response = await fetch("/v1/settings/model", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ key: pendingModel.key }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const body = (await response.json()) as SelectModelResponse;
      const chosen = body.selected_key ?? body.active_key;
      setList((current) =>
        current ? { ...current, active_key: body.active_key, selected_key: chosen } : current,
      );
      setSelectedKey(chosen);
      setFooterNotice({ kind: "success", text: `Now using ${body.model_name} for new messages.` });
      onActiveModel(body.model_name);
    } catch (err) {
      setFooterNotice({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not switch model",
      });
    } finally {
      setSwitching(false);
    }
  };

  const openEditor = (provider: string, mode: KeyEditorState["mode"]) => {
    setEditor({ provider, mode });
    setConfirmRemove(null);
    setProviderNotice(provider, null);
  };

  const saveKey = async (provider: string, value: string): Promise<boolean> => {
    const apiKey = value.trim();
    if (!apiKey) {
      setProviderNotice(provider, { kind: "error", text: "Enter an API key." });
      return false;
    }
    setKeyBusy(provider);
    setProviderNotice(provider, null);
    try {
      const response = await fetch("/v1/settings/provider-key", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ provider_name: provider, api_key: apiKey }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const body = (await response.json()) as ProviderKeyResponse;
      applyList(body);
      // Only mention the switch when it touched the model chosen here, not a hidden default.
      const applied =
        body.applied_to_active && currentModel?.provider_name === provider
          ? " The current model now uses it."
          : "";
      setProviderNotice(
        provider,
        body.keychain_saved === false
          ? {
              kind: "warning",
              text: `Saved for this session only: the OS keychain isn't available, so enter it again after a restart.${applied}`,
            }
          : { kind: "success", text: `Key saved in the OS keychain.${applied}` },
      );
      setEditor(null);
      return true;
    } catch (err) {
      setProviderNotice(provider, {
        kind: "error",
        text: err instanceof Error ? err.message : "Could not save the key",
      });
      return false;
    } finally {
      setKeyBusy(null);
    }
  };

  const removeKey = async (provider: string) => {
    setKeyBusy(provider);
    setProviderNotice(provider, null);
    try {
      const response = await fetch(`/v1/settings/provider-key/${encodeURIComponent(provider)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const body = (await response.json()) as ProviderKeyResponse;
      applyList(body);
      setProviderNotice(provider, {
        kind: "success",
        text:
          body.applied_to_active && currentModel?.provider_name === provider
            ? "Saved key removed. The current model now uses the environment key."
            : "Saved key removed.",
      });
    } catch (err) {
      setProviderNotice(provider, {
        kind: "error",
        text: err instanceof Error ? err.message : "Could not remove the key",
      });
    } finally {
      setKeyBusy(null);
      setConfirmRemove(null);
    }
  };

  return {
    loading,
    loadError,
    reload: () => setReloadToken((token) => token + 1),
    models,
    groups,
    currentModel,
    anyUsableKey,
    selectedKey,
    pendingModel,
    switching,
    footerNotice,
    editor,
    confirmRemove,
    keyBusy,
    providerNotices,
    filter,
    busy: switching || keyBusy !== null,
    setFilter,
    select,
    cancelSelection: () => {
      setSelectedKey(currentKey);
      setFooterNotice(null);
    },
    switchToSelected,
    openEditor,
    closeEditor: () => setEditor(null),
    askRemove: (provider: string) => {
      setConfirmRemove(provider);
      setEditor(null);
      setProviderNotice(provider, null);
    },
    cancelRemove: () => setConfirmRemove(null),
    saveKey,
    removeKey,
  };
}

/** Return focus to a provider's key action after an inline panel closes. */
function focusKeyAction(provider: string) {
  requestAnimationFrame(() => {
    document.getElementById(domId("key-action", provider))?.focus();
  });
}

function ModelsPane({ settings, notice }: { settings: ModelSettings; notice: string | null }) {
  const query = settings.filter.trim().toLowerCase();
  const visibleGroups = query
    ? settings.groups
        .map((group) => ({
          ...group,
          models: group.models.filter((model) =>
            [model.model_name, model.key, model.provider_name].some((field) =>
              field.toLowerCase().includes(query),
            ),
          ),
        }))
        .filter((group) => group.models.length > 0)
    : settings.groups;
  const currentGroup = settings.currentModel
    ? settings.groups.find((group) => group.name === settings.currentModel?.provider_name) ?? null
    : null;

  let body: ReactNode;
  if (settings.loading) {
    body = <p className="text-sm text-muted-foreground">Loading models…</p>;
  } else if (settings.loadError) {
    body = (
      <div className="space-y-3">
        <NoticeLine notice={{ kind: "error", text: settings.loadError }} />
        <Button type="button" variant="outline" size="sm" onClick={settings.reload}>
          Try again
        </Button>
      </div>
    );
  } else if (settings.models.length === 0) {
    body = (
      <p className="text-sm text-muted-foreground">
        No providers in config.toml. Add a [providers.*] block, then reopen settings.
      </p>
    );
  } else {
    body = (
      <div className="space-y-4">
        {notice && !settings.currentModel ? <NoticeLine notice={{ kind: "info", text: notice }} /> : null}
        {settings.currentModel ? (
          <CurrentModelCard model={settings.currentModel} group={currentGroup} />
        ) : (
          <NoModelSelected anyUsableKey={settings.anyUsableKey} />
        )}
        {settings.models.length > MODEL_FILTER_THRESHOLD ? (
          <div>
            <label htmlFor="model-filter" className="sr-only">
              Filter models
            </label>
            <input
              id="model-filter"
              type="search"
              autoComplete="off"
              spellCheck={false}
              placeholder="Filter models"
              value={settings.filter}
              onChange={(event) => settings.setFilter(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape" && settings.filter) {
                  event.preventDefault();
                  event.stopPropagation();
                  settings.setFilter("");
                }
              }}
              className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            />
          </div>
        ) : null}
        {visibleGroups.length === 0 ? (
          <p className="text-sm text-muted-foreground">No models match “{settings.filter.trim()}”.</p>
        ) : (
          visibleGroups.map((group) => (
            <ProviderCard key={group.name} group={group} settings={settings} />
          ))
        )}
      </div>
    );
  }

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{body}</div>
      {settings.models.length > 0 && (settings.currentModel || settings.pendingModel || settings.footerNotice) ? (
        <ModelsFooter settings={settings} />
      ) : null}
    </>
  );
}

/** Shown instead of the Current model card until a model is chosen in this session. */
function NoModelSelected({ anyUsableKey }: { anyUsableKey: boolean }) {
  return (
    <p
      className={cn(
        "rounded-xl border px-4 py-3 text-sm",
        anyUsableKey
          ? "border-dashed border-border text-muted-foreground"
          : NOTICE_STYLES.warning,
      )}
    >
      {anyUsableKey
        ? "No model selected yet. Choose one below to start chatting."
        : "No model selected yet. Add an API key below, then choose a model to start chatting."}
    </p>
  );
}

function CurrentModelCard({ model, group }: { model: ModelEntry; group: ProviderGroup | null }) {
  const source = keySourceOf(model);
  return (
    <section
      aria-labelledby="current-model-heading"
      className="rounded-xl border border-border bg-muted/50 px-4 py-3"
    >
      <h3
        id="current-model-heading"
        className="text-xs font-semibold uppercase tracking-[0.05em] text-muted-foreground"
      >
        Current model
      </h3>
      <p className="mt-1 break-all text-base font-semibold text-foreground">{model.model_name}</p>
      <p className="mt-0.5 text-sm text-muted-foreground">
        {model.provider_name}
        <span aria-hidden="true"> · </span>
        <span className={source === "none" ? "font-medium text-amber-800 dark:text-amber-300" : undefined}>
          {source === "none"
            ? "No API key. Add one below to use this model."
            : keySourceLabel(source, group?.envName ?? model.api_key_env)}
        </span>
      </p>
    </section>
  );
}

function KeyChip({ source, envName }: { source: KeySource; envName: string }) {
  const styles: Record<KeySource, string> = {
    env: "border-emerald-700/25 bg-emerald-500/10 text-emerald-800 dark:border-emerald-400/30 dark:text-emerald-300",
    keychain:
      "border-emerald-700/25 bg-emerald-500/10 text-emerald-800 dark:border-emerald-400/30 dark:text-emerald-300",
    session: "border-sky-700/25 bg-sky-500/10 text-sky-900 dark:border-sky-400/30 dark:text-sky-200",
    none: "border-amber-700/35 bg-amber-400/15 text-amber-900 dark:border-amber-400/35 dark:text-amber-200",
  };
  return (
    <span
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 truncate rounded-full border px-2 py-0.5 text-xs font-medium",
        styles[source],
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-1.5 w-1.5 shrink-0 rounded-full",
          source === "none" ? "bg-amber-600 dark:bg-amber-300" : source === "session" ? "bg-sky-600 dark:bg-sky-300" : "bg-emerald-600 dark:bg-emerald-300",
        )}
      />
      <span className="truncate">{keySourceLabel(source, envName)}</span>
    </span>
  );
}

function ProviderCard({ group, settings }: { group: ProviderGroup; settings: ModelSettings }) {
  const headingId = domId("provider", group.name);
  const editorId = domId("key-editor", group.name);
  const editing = settings.editor?.provider === group.name ? settings.editor : null;
  const confirming = settings.confirmRemove === group.name;
  const busy = settings.keyBusy === group.name;
  const notice = settings.providerNotices[group.name];
  const stored = group.source === "keychain" || group.source === "session";
  const actionClass = "h-8 px-3 text-xs";

  return (
    <section aria-labelledby={headingId} className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
        <h3 id={headingId} className="text-sm font-semibold text-foreground">
          {group.name}
        </h3>
        <KeyChip source={group.source} envName={group.envName} />
        {editing || confirming ? null : (
          <div className="ml-auto flex items-center gap-1.5">
            {group.source === "none" ? (
              <Button
                id={domId("key-action", group.name)}
                type="button"
                variant="outline"
                className={actionClass}
                disabled={settings.busy}
                onClick={() => settings.openEditor(group.name, "add")}
              >
                Add key
              </Button>
            ) : null}
            {stored ? (
              <>
                <Button
                  id={domId("key-action", group.name)}
                  type="button"
                  variant="outline"
                  className={actionClass}
                  disabled={settings.busy}
                  onClick={() => settings.openEditor(group.name, "replace")}
                >
                  Replace key
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className={cn(actionClass, "text-destructive hover:text-destructive dark:text-red-300")}
                  disabled={settings.busy}
                  onClick={() => settings.askRemove(group.name)}
                >
                  Remove
                </Button>
              </>
            ) : null}
          </div>
        )}
      </div>
      {group.source === "env" ? (
        <p className="-mt-1 px-4 pb-3 text-xs text-muted-foreground">
          Set in the environment. Change ${group.envName} to use a different key.
        </p>
      ) : null}
      {editing ? (
        <KeyEditor
          id={editorId}
          provider={group.name}
          mode={editing.mode}
          envName={group.envName}
          busy={busy}
          onSave={async (value) => {
            if (await settings.saveKey(group.name, value)) {
              focusKeyAction(group.name);
            }
          }}
          onCancel={() => {
            settings.closeEditor();
            focusKeyAction(group.name);
          }}
        />
      ) : null}
      {confirming ? (
        <div
          className="flex flex-wrap items-center gap-2 border-t border-border bg-muted/40 px-4 py-3"
          onKeyDown={(event) => {
            if (event.key === "Escape" && !busy) {
              event.preventDefault();
              event.stopPropagation();
              settings.cancelRemove();
              focusKeyAction(group.name);
            }
          }}
        >
          <p className="min-w-0 flex-1 text-sm text-foreground">
            Remove the saved {group.name} key? Models from this provider will need a new key.
          </p>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              autoFocus
              disabled={busy}
              onClick={() => {
                settings.cancelRemove();
                focusKeyAction(group.name);
              }}
            >
              Keep
            </Button>
            <Button
              type="button"
              size="sm"
              className="bg-destructive text-destructive-foreground hover:opacity-90"
              disabled={busy}
              onClick={() => void settings.removeKey(group.name)}
            >
              {busy ? "Removing…" : "Remove key"}
            </Button>
          </div>
        </div>
      ) : null}
      {notice ? (
        <div className="px-4 pb-3">
          <NoticeLine notice={notice} />
        </div>
      ) : null}
      <ModelRadioGroup group={group} labelledBy={headingId} settings={settings} />
    </section>
  );
}

function ModelRadioGroup({
  group,
  labelledBy,
  settings,
}: {
  group: ProviderGroup;
  labelledBy: string;
  settings: ModelSettings;
}) {
  const activeKey = settings.currentModel?.key ?? null;
  const enabled = group.models.filter((model) => model.has_api_key);
  const tabStop =
    enabled.find((model) => model.key === settings.selectedKey)?.key ?? enabled[0]?.key ?? null;

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>, model: ModelEntry) => {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (model.has_api_key && !settings.busy) {
        settings.select(model.key);
      }
      return;
    }
    const step =
      event.key === "ArrowDown" || event.key === "ArrowRight"
        ? 1
        : event.key === "ArrowUp" || event.key === "ArrowLeft"
          ? -1
          : 0;
    if (step === 0) {
      return;
    }
    event.preventDefault();
    const radios = Array.from(
      event.currentTarget.parentElement?.querySelectorAll<HTMLElement>(
        '[role="radio"]:not([aria-disabled="true"])',
      ) ?? [],
    );
    if (radios.length === 0) {
      return;
    }
    const index = radios.indexOf(event.currentTarget);
    const next = radios[(index + step + radios.length) % radios.length];
    next.focus();
    const key = next.dataset.modelKey;
    if (key && !settings.busy) {
      settings.select(key);
    }
  };

  return (
    <div role="radiogroup" aria-labelledby={labelledBy} className="space-y-1 border-t border-border p-2">
      {group.models.map((model) => {
        const active = model.key === activeKey;
        const selected = model.key === settings.selectedKey;
        const disabled = !model.has_api_key || settings.busy;
        const idle = active && selected;
        return (
          <div
            key={model.key}
            role="radio"
            aria-checked={selected}
            aria-disabled={disabled}
            tabIndex={model.key === tabStop ? 0 : -1}
            data-model-key={model.key}
            onClick={() => {
              if (!disabled) {
                settings.select(model.key);
              }
            }}
            onKeyDown={(event) => onKeyDown(event, model)}
            className={cn(
              "flex min-h-11 w-full items-center gap-3 rounded-lg border px-3 py-2 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/60",
              selected && !active
                ? "border-foreground/30 bg-accent"
                : "border-transparent",
              !model.has_api_key
                ? "cursor-not-allowed"
                : settings.busy || idle
                  ? "cursor-default"
                  : "cursor-pointer hover:bg-accent/70",
            )}
          >
            <span
              aria-hidden="true"
              className={cn(
                "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2",
                selected ? "border-foreground" : "border-muted-foreground/60",
                !model.has_api_key && "opacity-50",
              )}
            >
              {selected ? <span className="h-2 w-2 rounded-full bg-foreground" /> : null}
            </span>
            <span className="min-w-0 flex-1">
              <span
                className={cn(
                  "block truncate text-sm font-medium",
                  model.has_api_key ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {model.model_name}
              </span>
              {!model.has_api_key ? (
                <span className="block text-xs text-muted-foreground">Add a key to use this model</span>
              ) : model.key !== model.model_name ? (
                <span className="block truncate text-xs text-muted-foreground">{model.key}</span>
              ) : null}
            </span>
            {active ? (
              <span className="shrink-0 rounded-full border border-emerald-700/25 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:border-emerald-400/30 dark:text-emerald-300">
                Active
              </span>
            ) : selected ? (
              <span className="shrink-0 text-xs font-medium text-muted-foreground">Selected</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function KeyEditor({
  id,
  provider,
  mode,
  envName,
  busy,
  onSave,
  onCancel,
}: {
  id: string;
  provider: string;
  mode: KeyEditorState["mode"];
  envName: string;
  busy: boolean;
  onSave: (value: string) => Promise<void>;
  onCancel: () => void;
}) {
  const [value, setValue] = useState("");
  const [visible, setVisible] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const inputId = `${id}-input`;
  const hintId = `${id}-hint`;

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <form
      id={id}
      className="space-y-2 border-t border-border bg-muted/40 px-4 py-3"
      onSubmit={(event) => {
        event.preventDefault();
        if (!busy && value.trim()) {
          void onSave(value);
        }
      }}
    >
      <label htmlFor={inputId} className="block text-xs font-medium text-foreground">
        {mode === "add" ? `API key for ${provider}` : `New API key for ${provider}`}
      </label>
      <div className="relative">
        <input
          ref={inputRef}
          id={inputId}
          type={visible ? "text" : "password"}
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          aria-describedby={hintId}
          value={value}
          disabled={busy}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              // Close only this field; the dialog's own Escape handler must not fire.
              event.preventDefault();
              event.stopPropagation();
              if (!busy) {
                onCancel();
              }
            }
          }}
          className="h-9 w-full rounded-md border border-border bg-background pl-3 pr-10 font-mono text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-60"
        />
        <button
          type="button"
          aria-label={visible ? "Hide key" : "Show key"}
          aria-pressed={visible}
          aria-controls={inputId}
          onClick={() => setVisible((current) => !current)}
          className="absolute inset-y-0 right-0 flex w-9 items-center justify-center rounded-r-md text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          <EyeIcon crossed={visible} />
        </button>
      </div>
      <p id={hintId} className="text-xs text-muted-foreground">
        Stored in the OS keychain and used for every {provider} model.
        {envName ? ` $${envName} takes precedence when it is set.` : ""}
      </p>
      <div className="flex justify-end gap-2 pt-1">
        <Button type="button" variant="ghost" size="sm" disabled={busy} onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={busy || value.trim() === ""}>
          {busy ? "Saving…" : "Save key"}
        </Button>
      </div>
    </form>
  );
}

function EyeIcon({ crossed }: { crossed: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="h-4 w-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
      {crossed ? <path d="M4 4l16 16" /> : null}
    </svg>
  );
}

function ModelsFooter({ settings }: { settings: ModelSettings }) {
  const pending = settings.pendingModel;
  const notice = settings.footerNotice;
  const active = settings.currentModel;
  return (
    <div className="border-t border-border bg-card px-5 py-3">
      <div aria-live="polite">{notice ? <NoticeLine notice={notice} /> : null}</div>
      {pending ? (
        <div className={cn("flex flex-wrap items-center justify-end gap-2", notice && "mt-2")}>
          {active ? (
            <p className="mr-auto min-w-0 truncate text-sm text-muted-foreground">
              From <span className="text-foreground">{active.model_name}</span>
            </p>
          ) : null}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={settings.switching}
            onClick={settings.cancelSelection}
          >
            Cancel
          </Button>
          <Button
            type="button"
            size="sm"
            className="min-w-0 max-w-full"
            disabled={settings.switching}
            onClick={() => void settings.switchToSelected()}
          >
            <span className="truncate">
              {settings.switching
                ? "Switching…"
                : active
                  ? `Switch to ${pending.model_name}`
                  : `Use ${pending.model_name}`}
            </span>
          </Button>
        </div>
      ) : notice || !active ? null : (
        <p className="truncate text-sm text-muted-foreground">
          Active: <span className="text-foreground">{active.model_name}</span>
        </p>
      )}
    </div>
  );
}

function PermissionSelect({
  title,
  detail,
  value,
  onChange,
}: {
  title: string;
  detail: string;
  value: PermissionMode;
  onChange: (value: PermissionMode) => void;
}) {
  return (
    <label className="block rounded-lg border border-border px-3 py-3">
      <span className="text-sm font-medium text-foreground">{title}</span>
      <span className="mt-1 block text-xs text-muted-foreground">{detail}</span>
      <select
        className="mt-3 h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
        value={value}
        onChange={(event) => onChange(event.target.value as PermissionMode)}
      >
        <option value="allow">Allow</option>
        <option value="ask">Ask</option>
        <option value="deny">Deny</option>
      </select>
    </label>
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
      <PermissionSelect
        title="Files inside the workspace"
        detail="Reading, listing, searching, and editing files in the workspace folder."
        value={draft.inside_workspace}
        onChange={(value) => onChange({ ...draft, inside_workspace: value })}
      />
      <PermissionSelect
        title="Files outside the workspace"
        detail="The same file tools on any other path, including ../ and symlinks that lead out."
        value={draft.outside_workspace}
        onChange={(value) => onChange({ ...draft, outside_workspace: value })}
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
