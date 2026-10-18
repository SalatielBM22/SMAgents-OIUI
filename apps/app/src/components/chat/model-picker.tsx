"use client";

import { useCallback, useEffect, useState } from "react";
import { useProviderKeys } from "./provider-keys";
import type { ModelList } from "@/lib/model-choice";

/** Header select for the OmniRoute combo.  Switching keeps the conversation: the next message goes to the new combo. */
export function ModelPicker({ disabled }: { disabled?: boolean }) {
  const { model, setModel } = useProviderKeys();
  const [list, setList] = useState<ModelList>({ default: null, models: [] });
  const [failed, setFailed] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await fetch("/api/models", { cache: "no-store" });
      const body = (await r.json()) as ModelList;
      setList(body);
      setFailed(!r.ok || body.models.length === 0);
    } catch {
      setFailed(true);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);
  // OmniRoute/agent may still be booting when the page opens: retry a few times instead of waiting for a click.
  useEffect(() => {
    if (!failed) return;
    let tries = 0;
    const timer = setInterval(() => { if (++tries > 6) clearInterval(timer); else void load(); }, 10_000);
    return () => clearInterval(timer);
  }, [failed, load]);

  const current = model ?? "";
  const known = list.models.some((m) => m.id === model);
  return (
    <label className="model-picker" title={failed ? "Lista de combos indisponível (agente ou OmniRoute fora do ar)" : "Combo do OmniRoute para as próximas mensagens"}>
      <span className="model-picker-label">Modelo</span>
      <select
        aria-label="Combo do OmniRoute"
        value={current}
        disabled={disabled}
        onFocus={() => { if (failed) void load(); }}
        onChange={(event) => setModel(event.target.value || null)}
      >
        <option value="">{list.default ? `padrão (${list.default})` : "padrão do agente"}</option>
        {model && !known && <option value={model}>{model}</option>}
        {list.models.map((m) => (
          <option key={m.id} value={m.id}>{m.vision ? `${m.id} · visão` : m.id}</option>
        ))}
      </select>
    </label>
  );
}
