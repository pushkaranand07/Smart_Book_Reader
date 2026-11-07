import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach } from "vitest";
import { apiKeyStore } from "../api/apiKeyStore";

// jsdom does not implement scrolling APIs.
Element.prototype.scrollIntoView = function scrollIntoView() {
  /* no-op in tests */
};

beforeEach(() => {
  window.sessionStorage.clear();
  apiKeyStore.reset();
});

afterEach(() => {
  cleanup();
});
