import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";

import type { ProviderHeaders } from "./provider-keys";
import { MODEL_HEADER } from "./model-choice";

class CredentialSafeLangGraphAgent extends LangGraphHttpAgent {
  protected override requestInit(input: Parameters<LangGraphHttpAgent["requestInit"]>[0]): RequestInit {
    // Custom credential headers survive cross-origin fetch redirects. Fail closed.
    return { ...super.requestInit(input), redirect: "error" };
  }
}

export interface RuntimeOptionsEnv {
  langgraphUrl?: string;
  mcpServerUrl?: string;
  providerHeaders?: ProviderHeaders;
  /** OmniRoute combo for this request (header x-oiu-model), already validated. */
  model?: string;
}

// Normalize Render's fromService hostport (bare host:port) into a full URL
export function normalizeLanggraphUrl(raw?: string): string {
  if (!raw) return "http://localhost:8123";
  return raw.startsWith("http") ? raw : `http://${raw}`;
}

export function buildRuntimeOptions(env: RuntimeOptionsEnv) {
  const headers: Record<string, string> = { ...(env.providerHeaders ?? {}), ...(env.model ? { [MODEL_HEADER]: env.model } : {}) };
  return {
    agents: {
      default: new CredentialSafeLangGraphAgent({
        url: normalizeLanggraphUrl(env.langgraphUrl),
        ...(Object.keys(headers).length > 0 && { headers }),
      }),
    },
    a2ui: { injectA2UITool: true },
    openGenerativeUI: true as const,
    ...(env.mcpServerUrl && {
      mcpApps: {
        servers: [{
          type: "http" as const,
          url: env.mcpServerUrl,
          serverId: "example_mcp_app",
        }],
      },
    }),
  };
}
