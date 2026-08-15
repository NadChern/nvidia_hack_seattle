// Pairing credential storage. Never log any of these values - not even
// truncated - per docs/07-Privacy-and-Security.md.
import * as Crypto from "expo-crypto";
import * as SecureStore from "expo-secure-store";
import { Platform } from "react-native";

const CREDENTIAL_KEY = "helper.pairing_credential.pr1test";
const DEVICE_ID_KEY = "helper.device_id.pr1test";

interface KeyValueStore {
  getItemAsync(key: string): Promise<string | null>;
  setItemAsync(key: string, value: string): Promise<void>;
  deleteItemAsync(key: string): Promise<void>;
}

// expo-secure-store has no web implementation (it's a native Keychain /
// Keystore wrapper). This app never ships to web, but Expo web is handy for
// previewing screens during development, so fall back to localStorage there.
const webStore: KeyValueStore = {
  getItemAsync: async (key) => window.localStorage.getItem(key),
  setItemAsync: async (key, value) => {
    window.localStorage.setItem(key, value);
  },
  deleteItemAsync: async (key) => {
    window.localStorage.removeItem(key);
  },
};

const store: KeyValueStore = Platform.OS === "web" ? webStore : SecureStore;

export interface PairingCredential {
  gateway_url: string;
  device_id: string;
  credential: string;
  expires_at: string;
}

export async function getDeviceId(): Promise<string> {
  const existing = await store.getItemAsync(DEVICE_ID_KEY);
  if (existing) {
    return existing;
  }
  const generated = Crypto.randomUUID();
  await store.setItemAsync(DEVICE_ID_KEY, generated);
  return generated;
}

export async function getPairingCredential(): Promise<PairingCredential | null> {
  const raw = await store.getItemAsync(CREDENTIAL_KEY);
  if (!raw) {
    return null;
  }
  return JSON.parse(raw) as PairingCredential;
}

export async function savePairingCredential(credential: PairingCredential): Promise<void> {
  await store.setItemAsync(CREDENTIAL_KEY, JSON.stringify(credential));
}

export async function clearPairingCredential(): Promise<void> {
  await store.deleteItemAsync(CREDENTIAL_KEY);
}
