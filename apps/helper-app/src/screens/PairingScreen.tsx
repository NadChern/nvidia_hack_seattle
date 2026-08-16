import { type BarcodeScanningResult, CameraView, useCameraPermissions } from "expo-camera";
import { useCallback, useRef, useState } from "react";
import { ActivityIndicator, Button, StyleSheet, Text, View } from "react-native";

import { api } from "../api";
import type { PairingQrPayload } from "../api/contract";
import { getDeviceId, savePairingCredential } from "../storage/credentials";

interface Props {
  onPaired: () => void;
}

type Status = "scanning" | "pairing" | "error";

export function PairingScreen({ onPaired }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [status, setStatus] = useState<Status>("scanning");
  const [error, setError] = useState<string | null>(null);
  const handledRef = useRef(false);

  const handleScanned = useCallback(
    async ({ data }: BarcodeScanningResult) => {
      if (handledRef.current) {
        return;
      }
      handledRef.current = true;
      setStatus("pairing");
      setError(null);

      try {
        const payload = JSON.parse(data) as PairingQrPayload;
        if (!payload.gateway_url || !payload.pairing_code) {
          throw new Error("QR code is missing gateway_url or pairing_code");
        }
        if (new Date(payload.expires_at).getTime() < Date.now()) {
          throw new Error("This pairing code has expired - ask for a new QR code");
        }

        const deviceId = await getDeviceId();
        const claimed = await api.claimPairing(payload.gateway_url, payload.pairing_code, deviceId);
        await savePairingCredential({
          gateway_url: payload.gateway_url,
          device_id: claimed.device_id,
          credential: claimed.credential,
          expires_at: claimed.expires_at,
        });
        onPaired();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not pair with that QR code");
        setStatus("error");
        handledRef.current = false;
      }
    },
    [onPaired],
  );

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>Helper needs camera access to scan the pairing QR code.</Text>
        <Button title="Grant camera access" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFill}
        facing="back"
        barcodeScannerSettings={{ barcodeTypes: ["qr"] }}
        onBarcodeScanned={status === "scanning" ? handleScanned : undefined}
      />
      <View style={styles.overlay}>
        {status === "pairing" && (
          <>
            <ActivityIndicator color="#fff" />
            <Text style={styles.overlayText}>Pairing...</Text>
          </>
        )}
        {status === "error" && (
          <>
            <Text style={styles.overlayText}>{error}</Text>
            <Button
              title="Try again"
              onPress={() => {
                setStatus("scanning");
                setError(null);
              }}
            />
          </>
        )}
        {status === "scanning" && (
          <Text style={styles.overlayText}>Point the camera at the pairing QR code</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "black" },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 12 },
  message: { textAlign: "center", marginBottom: 8 },
  overlay: {
    position: "absolute",
    bottom: 48,
    left: 24,
    right: 24,
    alignItems: "center",
    gap: 12,
  },
  overlayText: { color: "white", textAlign: "center", fontSize: 16 },
});
