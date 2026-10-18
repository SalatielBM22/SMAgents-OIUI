import { normalizeLanggraphUrl } from "@/lib/copilotkit-runtime-options";
import { isModelId, type ModelList } from "@/lib/model-choice";

// The combos the agent can use (it asks OmniRoute with its own server key; no key reaches the browser).
export const GET = async () => {
  const base = normalizeLanggraphUrl(process.env.LANGGRAPH_DEPLOYMENT_URL).replace(/\/$/, "");
  try {
    const r = await fetch(`${base}/models`, { cache: "no-store", redirect: "error", signal: AbortSignal.timeout(10_000) });
    if (!r.ok) throw new Error(String(r.status));
    const body = (await r.json()) as ModelList;
    const models = (body.models ?? []).filter((m) => isModelId(m?.id)).map((m) => ({ id: m.id, vision: Boolean(m.vision) }));
    const def = isModelId(body.default) ? body.default : null;
    return Response.json({ default: def, models } satisfies ModelList, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ default: null, models: [] } satisfies ModelList, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
};
