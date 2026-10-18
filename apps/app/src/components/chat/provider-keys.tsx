"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { MODEL_HEADER, MODEL_STORAGE_KEY, isModelId } from "@/lib/model-choice";

export type ProviderKeys = { openai: string; jev: string };
type ProviderKeysContextValue = {
  headers: Record<string, string> | undefined;
  hasKeys: boolean;
  session: number;
  save: (keys: ProviderKeys) => void;
  clear: () => void;
  newChat: () => void;
  draft: string;
  setDraft: (value: string) => void;
  keysOpen: boolean;
  setKeysOpen: (value: boolean) => void;
  /** OmniRoute combo chosen in the header; null means the agent's default (LLM_MODEL). */
  model: string | null;
  setModel: (value: string | null) => void;
};
const ProviderKeysContext = createContext<ProviderKeysContextValue | null>(null);

function readStoredModel(): string | null {
  try {
    const v = window.localStorage.getItem(MODEL_STORAGE_KEY);
    return isModelId(v) ? v : null;
  } catch {
    return null;
  }
}

export function ProviderKeysProvider({ children }: { children: ReactNode }) {
  // Deliberately memory-only: never put credentials in browser storage or agent state.
  const [keys, setKeys] = useState<ProviderKeys | null>(null);
  const [session, setSession] = useState(0);
  const [draft, setDraft] = useState("");
  const [keysOpen, setKeysOpen] = useState(false);
  // The model choice (not a secret) is remembered per browser so a switch survives a refresh.
  const [model, setModelState] = useState<string | null>(null);
  useEffect(() => { setModelState(readStoredModel()); }, []);
  const setModel = (value: string | null) => {
    const next = isModelId(value) ? value : null;
    setModelState(next);
    try {
      if (next) window.localStorage.setItem(MODEL_STORAGE_KEY, next);
      else window.localStorage.removeItem(MODEL_STORAGE_KEY);
    } catch { /* storage unavailable: the choice lasts for this page only */ }
  };
  const newChat = () => { setDraft(""); setSession((value) => value + 1); };
  const headers = {
    ...(keys ? { "x-openai-api-key": keys.openai, "x-jev-api-key": keys.jev } : {}),
    ...(model ? { [MODEL_HEADER]: model } : {}),
  };
  return (
    <ProviderKeysContext.Provider value={{
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      // Local single-user mode: the agent already holds server credentials (e.g. OmniRoute), so the browser needs none.
      hasKeys: keys !== null || process.env.NEXT_PUBLIC_OIU_SERVER_KEYS === "1",
      session,
      save: (value) => { setKeys(value); setKeysOpen(false); setSession((value) => value + 1); },
      clear: () => { setKeys(null); newChat(); },
      newChat, draft, setDraft, keysOpen, setKeysOpen, model, setModel,
    }}>
      {children}
    </ProviderKeysContext.Provider>
  );
}

export function useProviderKeys() {
  const value = useContext(ProviderKeysContext);
  if (!value) throw new Error("ProviderKeysProvider is required");
  return value;
}
