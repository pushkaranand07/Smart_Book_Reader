/**
 * API key kept in sessionStorage (cleared when the tab closes) plus a "key needed" flag raised on 401.
 * Exposed as a tiny external store so React can subscribe with useSyncExternalStore.
 */

export const API_KEY_STORAGE_KEY = "sbr.apiKey";

export interface ApiKeyState {
  apiKey: string;
  /** True after the server answered 401: the UI should prompt for a key. */
  keyRequired: boolean;
}

type Listener = () => void;

function readStoredKey(): string {
  try {
    return window.sessionStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

let state: ApiKeyState = { apiKey: readStoredKey(), keyRequired: false };
const listeners = new Set<Listener>();

function emit(next: ApiKeyState): void {
  state = next;
  listeners.forEach((listener) => {
    listener();
  });
}

export const apiKeyStore = {
  getSnapshot: (): ApiKeyState => {
    return state;
  },
  subscribe: (listener: Listener): (() => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getApiKey: (): string => {
    return state.apiKey;
  },
  setApiKey: (value: string): void => {
    const apiKey = value.trim();
    try {
      if (apiKey) window.sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
      else window.sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    } catch {
      // Storage can be unavailable (private mode); keep the key in memory only.
    }
    emit({ apiKey, keyRequired: false });
  },
  requireKey: (): void => {
    if (!state.keyRequired) emit({ ...state, keyRequired: true });
  },
  dismissPrompt: (): void => {
    if (state.keyRequired) emit({ ...state, keyRequired: false });
  },
  /** Test helper: re-read storage and clear flags. */
  reset: (): void => {
    emit({ apiKey: readStoredKey(), keyRequired: false });
  },
};
