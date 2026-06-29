import { describe, expect, it } from "vitest";

import manifest from "../../manifest.json";

describe("extension manifest", () => {
  it("declares storage and canonical local API host permissions", () => {
    expect(manifest.permissions).toContain("storage");
    expect(manifest.permissions).toContain("scripting");
    expect(manifest.host_permissions).toContain("http://localhost:8000/*");
    expect(manifest.host_permissions).toContain("http://127.0.0.1:8000/*");
  });

  it("does not request broad shopping-site host permissions", () => {
    expect("optional_host_permissions" in manifest).toBe(false);
    expect(manifest.host_permissions).not.toContain("https://*/*");
    expect(manifest.host_permissions).not.toContain("http://*/*");
  });
});
