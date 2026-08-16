const mockSecureStoreData = new Map<string, string>();

jest.mock("expo-secure-store", () => ({
  getItemAsync: jest.fn((key: string) => Promise.resolve(mockSecureStoreData.get(key) ?? null)),
  setItemAsync: jest.fn((key: string, value: string) => {
    mockSecureStoreData.set(key, value);
    return Promise.resolve();
  }),
  deleteItemAsync: jest.fn((key: string) => {
    mockSecureStoreData.delete(key);
    return Promise.resolve();
  }),
}));

let mockUuidCounter = 0;
jest.mock("expo-crypto", () => ({
  randomUUID: jest.fn(() => `generated-uuid-${++mockUuidCounter}`),
}));

import * as SecureStore from "expo-secure-store";
import * as Crypto from "expo-crypto";
import {
  getDeviceId,
  savePairingCredential,
  getPairingCredential,
  clearPairingCredential,
} from "./credentials";

const sampleCredential = {
  gateway_url: "https://gateway.example",
  device_id: "helper-01",
  credential: "v1.super-secret-credential",
  expires_at: "2026-08-16T00:00:00Z",
};

describe("credentials storage (native)", () => {
  beforeEach(() => {
    mockSecureStoreData.clear();
    mockUuidCounter = 0;
    jest.clearAllMocks();
  });

  test("getDeviceId generates a UUID once and persists it across calls", async () => {
    const first = await getDeviceId();
    const second = await getDeviceId();

    expect(first).toBe("generated-uuid-1");
    expect(second).toBe("generated-uuid-1");
    expect(Crypto.randomUUID).toHaveBeenCalledTimes(1);
  });

  test("getPairingCredential returns null when nothing is stored", async () => {
    await expect(getPairingCredential()).resolves.toBeNull();
  });

  test("savePairingCredential then getPairingCredential round-trips the exact object", async () => {
    await savePairingCredential(sampleCredential);

    await expect(getPairingCredential()).resolves.toEqual(sampleCredential);
  });

  test("clearPairingCredential removes the stored credential", async () => {
    await savePairingCredential(sampleCredential);

    await clearPairingCredential();

    await expect(getPairingCredential()).resolves.toBeNull();
  });

  test("uses expo-secure-store, not any other storage, on native", async () => {
    await savePairingCredential(sampleCredential);

    expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
      "helper.pairing_credential",
      expect.stringContaining("v1.super-secret-credential"),
    );
  });

  test("never logs the credential or device id", async () => {
    const logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const errorSpy = jest.spyOn(console, "error").mockImplementation(() => {});

    await getDeviceId();
    await savePairingCredential(sampleCredential);
    await getPairingCredential();

    for (const spy of [logSpy, warnSpy, errorSpy]) {
      for (const call of spy.mock.calls) {
        const serialized = JSON.stringify(call);
        expect(serialized).not.toContain("v1.super-secret-credential");
        expect(serialized).not.toContain("generated-uuid");
      }
    }
    logSpy.mockRestore();
    warnSpy.mockRestore();
    errorSpy.mockRestore();
  });
});

describe("credentials storage (web fallback)", () => {
  class MemoryStorage {
    private data = new Map<string, string>();
    getItem(key: string): string | null {
      return this.data.get(key) ?? null;
    }
    setItem(key: string, value: string): void {
      this.data.set(key, value);
    }
    removeItem(key: string): void {
      this.data.delete(key);
    }
  }

  test("falls back to localStorage on web instead of expo-secure-store", async () => {
    Object.defineProperty(window, "localStorage", {
      value: new MemoryStorage(),
      configurable: true,
    });
    // credentials.ts picks its store once, at module load, from
    // react-native's Platform.OS -- resetModules() clears the whole
    // registry, so react-native itself must be re-required too and mutated
    // *before* re-requiring credentials.ts, or credentials.ts would close
    // over a fresh Platform object this test never touched.
    jest.resetModules();
    const freshPlatform = (require("react-native") as typeof import("react-native")).Platform;
    Object.defineProperty(freshPlatform, "OS", { value: "web", configurable: true });
    const webCredentials = require("./credentials") as typeof import("./credentials");
    const secureStore = require("expo-secure-store") as typeof SecureStore;
    const credential = { ...sampleCredential, credential: "v1.web-credential" };

    await webCredentials.savePairingCredential(credential);

    expect(secureStore.setItemAsync).not.toHaveBeenCalled();
    expect(window.localStorage.getItem("helper.pairing_credential")).toContain(
      "v1.web-credential",
    );
    await expect(webCredentials.getPairingCredential()).resolves.toEqual(credential);
  });
});
