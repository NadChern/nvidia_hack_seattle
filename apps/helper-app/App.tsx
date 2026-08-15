import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, View } from "react-native";

import { CallScreen } from "./src/screens/CallScreen";
import { PairingScreen } from "./src/screens/PairingScreen";
import { RequestListScreen } from "./src/screens/RequestListScreen";
import { getPairingCredential } from "./src/storage/credentials";

type Screen = "loading" | "pairing" | "requests";

interface ActiveCall {
  sessionId: string;
  serverUrl: string;
  token: string;
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("loading");
  const [activeCall, setActiveCall] = useState<ActiveCall | null>(null);

  const checkPairing = useCallback(async () => {
    const credential = await getPairingCredential();
    setScreen(credential ? "requests" : "pairing");
  }, []);

  useEffect(() => {
    checkPairing();
  }, [checkPairing]);

  const handleAccepted = useCallback((sessionId: string, serverUrl: string, token: string) => {
    setActiveCall({ sessionId, serverUrl, token });
  }, []);

  const handleHangUp = useCallback(() => {
    setActiveCall(null);
  }, []);

  return (
    <View style={styles.container}>
      {activeCall ? (
        <CallScreen serverUrl={activeCall.serverUrl} token={activeCall.token} onHangUp={handleHangUp} />
      ) : (
        <>
          {screen === "loading" && <ActivityIndicator />}
          {screen === "pairing" && <PairingScreen onPaired={checkPairing} />}
          {screen === "requests" && <RequestListScreen onAccepted={handleAccepted} />}
        </>
      )}
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
});
