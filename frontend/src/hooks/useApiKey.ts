import { useSyncExternalStore } from "react";
import { apiKeyStore, type ApiKeyState } from "../api/apiKeyStore";

export function useApiKey(): ApiKeyState & {
  setApiKey: (value: string) => void;
  dismissPrompt: () => void;
} {
  const state = useSyncExternalStore(apiKeyStore.subscribe, apiKeyStore.getSnapshot, apiKeyStore.getSnapshot);
  return { ...state, setApiKey: apiKeyStore.setApiKey, dismissPrompt: apiKeyStore.dismissPrompt };
}
