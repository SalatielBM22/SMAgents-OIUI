// Per-request model (an OmniRoute combo such as sm-frontier), chosen in the header picker.  Travels as one validated header;
// the agent falls back to its LLM_MODEL when it is absent.
export const MODEL_HEADER = "x-oiu-model";
export const MODEL_STORAGE_KEY = "oiu:model";
const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$/;

export function isModelId(value: unknown): value is string {
  return typeof value === "string" && MODEL_ID.test(value);
}

export function readModelHeader(headers: Headers): string | undefined {
  const value = headers.get(MODEL_HEADER)?.trim();
  return value && isModelId(value) ? value : undefined;
}

export type ModelOption = { id: string; vision: boolean };
export type ModelList = { default: string | null; models: ModelOption[] };
