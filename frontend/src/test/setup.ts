import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach, beforeEach } from "vitest";
import { apiKeyStore } from "../api/apiKeyStore";

// findBy*/waitFor default to 1 s, which is too tight for lazy chunks on a loaded machine.
configure({ asyncUtilTimeout: 5000 });

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
