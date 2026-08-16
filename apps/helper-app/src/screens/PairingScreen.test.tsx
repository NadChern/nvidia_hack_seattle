jest.mock("expo-camera", () => {
  const React = require("react");
  const { View } = require("react-native");
  return {
    useCameraPermissions: jest.fn(),
    CameraView: jest.fn((props: any) => React.createElement(View, { testID: "camera-view", ...props })),
  };
});

jest.mock("../api", () => ({
  api: {
    claimPairing: jest.fn(),
    listRequests: jest.fn(),
    acceptRequest: jest.fn(),
    getHelperToken: jest.fn(),
  },
}));

jest.mock("../storage/credentials", () => ({
  getDeviceId: jest.fn(),
  savePairingCredential: jest.fn(),
}));

import { act, render, fireEvent, screen } from "@testing-library/react-native";
import { useCameraPermissions, CameraView } from "expo-camera";
import { api } from "../api";
import { getDeviceId, savePairingCredential } from "../storage/credentials";
import { PairingScreen } from "./PairingScreen";

const mockUseCameraPermissions = useCameraPermissions as jest.Mock;
const mockCameraView = CameraView as unknown as jest.Mock;
const mockClaimPairing = api.claimPairing as jest.Mock;
const mockGetDeviceId = getDeviceId as jest.Mock;
const mockSavePairingCredential = savePairingCredential as jest.Mock;

const FUTURE = "2099-01-01T00:00:00Z";
const PAST = "2000-01-01T00:00:00Z";

function scan(data: unknown) {
  const { onBarcodeScanned } = mockCameraView.mock.calls.at(-1)![0];
  return act(async () => {
    await onBarcodeScanned({ data: typeof data === "string" ? data : JSON.stringify(data) });
  });
}

describe("PairingScreen", () => {
  const requestPermission = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseCameraPermissions.mockReturnValue([{ granted: true }, requestPermission]);
    mockGetDeviceId.mockResolvedValue("device-01");
  });

  test("shows a loading indicator before the permission result is known", async () => {
    mockUseCameraPermissions.mockReturnValue([null, requestPermission]);

    await render(<PairingScreen onPaired={jest.fn()} />);

    expect(screen.queryByTestId("camera-view")).toBeNull();
  });

  test("shows a permission request instead of the scanner when access is not granted", async () => {
    mockUseCameraPermissions.mockReturnValue([{ granted: false }, requestPermission]);

    await render(<PairingScreen onPaired={jest.fn()} />);

    expect(screen.queryByTestId("camera-view")).toBeNull();
    expect(screen.getByText("Grant camera access")).toBeTruthy();

    await fireEvent.press(screen.getByText("Grant camera access"));
    expect(requestPermission).toHaveBeenCalledTimes(1);
  });

  test("a valid QR payload pairs, saves the credential, and calls onPaired", async () => {
    mockClaimPairing.mockResolvedValue({
      device_id: "device-01",
      credential: "v1.new-credential",
      expires_at: FUTURE,
    });
    const onPaired = jest.fn();
    await render(<PairingScreen onPaired={onPaired} />);

    await scan({ gateway_url: "https://gateway.example", pairing_code: "the-code", expires_at: FUTURE });

    expect(mockClaimPairing).toHaveBeenCalledWith("https://gateway.example", "the-code", "device-01");
    expect(mockSavePairingCredential).toHaveBeenCalledWith({
      gateway_url: "https://gateway.example",
      device_id: "device-01",
      credential: "v1.new-credential",
      expires_at: FUTURE,
    });
    expect(onPaired).toHaveBeenCalledTimes(1);
  });

  test("a payload missing gateway_url or pairing_code is rejected without calling claimPairing", async () => {
    await render(<PairingScreen onPaired={jest.fn()} />);

    await scan({ pairing_code: "the-code", expires_at: FUTURE });

    expect(mockClaimPairing).not.toHaveBeenCalled();
    expect(screen.getByText(/missing gateway_url or pairing_code/)).toBeTruthy();
  });

  test("an already-expired pairing code is rejected without calling claimPairing", async () => {
    await render(<PairingScreen onPaired={jest.fn()} />);

    await scan({ gateway_url: "https://gateway.example", pairing_code: "the-code", expires_at: PAST });

    expect(mockClaimPairing).not.toHaveBeenCalled();
    expect(screen.getByText(/expired/)).toBeTruthy();
  });

  test("malformed (non-JSON) QR data is caught and shown as an error, not a crash", async () => {
    await render(<PairingScreen onPaired={jest.fn()} />);

    await scan("not json at all");

    // JSON.parse's SyntaxError is an Error, so its own message is shown
    // (not the generic fallback, which is only for a non-Error throw) --
    // either way, the point is it's caught and displayed, not a crash.
    expect(mockClaimPairing).not.toHaveBeenCalled();
    expect(screen.getByText("Try again")).toBeTruthy();
    expect(screen.queryByTestId("camera-view")).toBeTruthy();
  });

  test("a rapid double-scan only claims pairing once", async () => {
    let resolveClaim!: (value: unknown) => void;
    mockClaimPairing.mockReturnValue(new Promise((resolve) => (resolveClaim = resolve)));
    await render(<PairingScreen onPaired={jest.fn()} />);
    const payload = { gateway_url: "https://gateway.example", pairing_code: "the-code", expires_at: FUTURE };
    const { onBarcodeScanned } = mockCameraView.mock.calls.at(-1)![0];

    await act(async () => {
      // Two scans fired back to back, before the first claimPairing resolves.
      onBarcodeScanned({ data: JSON.stringify(payload) });
      onBarcodeScanned({ data: JSON.stringify(payload) });
      resolveClaim({ device_id: "device-01", credential: "v1.x", expires_at: FUTURE });
    });

    expect(mockClaimPairing).toHaveBeenCalledTimes(1);
  });

  test("after an error, Try again resets the screen and a fresh scan succeeds", async () => {
    const onPaired = jest.fn();
    await render(<PairingScreen onPaired={onPaired} />);
    await scan("not json at all");
    expect(screen.getByText("Try again")).toBeTruthy();

    await fireEvent.press(screen.getByText("Try again"));

    expect(screen.queryByText("Try again")).toBeNull();
    expect(screen.getByText("Point the camera at the pairing QR code")).toBeTruthy();

    mockClaimPairing.mockResolvedValue({ device_id: "device-01", credential: "v1.x", expires_at: FUTURE });
    await scan({ gateway_url: "https://gateway.example", pairing_code: "the-code", expires_at: FUTURE });

    expect(mockClaimPairing).toHaveBeenCalledTimes(1);
    expect(onPaired).toHaveBeenCalledTimes(1);
  });

  test("never renders the pairing code or the credential as visible text", async () => {
    mockClaimPairing.mockResolvedValue({
      device_id: "device-01",
      credential: "v1.super-secret-credential",
      expires_at: FUTURE,
    });
    await render(<PairingScreen onPaired={jest.fn()} />);

    await scan({
      gateway_url: "https://gateway.example",
      pairing_code: "the-super-secret-code",
      expires_at: FUTURE,
    });

    const tree = JSON.stringify(screen.toJSON());
    expect(tree).not.toContain("the-super-secret-code");
    expect(tree).not.toContain("v1.super-secret-credential");
  });
});
