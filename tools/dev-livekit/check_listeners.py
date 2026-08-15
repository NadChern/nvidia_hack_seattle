#!/usr/bin/env python3
"""Report what LiveKit is actually listening on.

The S01 spike found ICE/TCP bound to every interface on port 7881 even though
signaling was bound to loopback. docs/07-Privacy-and-Security.md is explicit
that you must "verify listeners and firewall policy rather than inferring
exposure from the WebSocket URL", so this makes that check executable.

Exits non-zero when a LiveKit port is reachable from outside the loopback
interface.

    python3 tools/dev-livekit/check_listeners.py
"""

from __future__ import annotations

import argparse
import socket
import sys

#: Signaling and ICE/TCP. Port 7882 is the UDP mux and is deliberately absent:
#: a TCP connect can never confirm a UDP listener, and reporting it as missing
#: would be misleading rather than informative.
LIVEKIT_PORTS = (7880, 7881)
UDP_MUX_PORT = 7882
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def local_addresses() -> list[str]:
    """Non-loopback addresses this host answers on."""
    addresses: set[str] = set()
    hostname = socket.gethostname()
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            infos = socket.getaddrinfo(hostname, None, family)
        except socket.gaierror:
            continue
        for info in infos:
            address = str(info[4][0]).split("%")[0]
            if address not in LOOPBACK:
                addresses.add(address)
    return sorted(addresses)


def reachable(host: str, port: int, *, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="permit non-loopback exposure, for a deliberate trusted-LAN setup",
    )
    parser.add_argument(
        "--expected-host",
        help="also probe this exact bind address even if hostname resolution omits it",
    )
    args = parser.parse_args()

    print("loopback:")
    listening = False
    for port in LIVEKIT_PORTS:
        up = reachable("127.0.0.1", port)
        listening = listening or up
        print(f"  127.0.0.1:{port:<5} {'listening' if up else '-'}")

    print(f"  127.0.0.1:{UDP_MUX_PORT:<5} udp mux, not probeable over tcp")

    exposed: list[str] = []
    others = local_addresses()
    if args.expected_host and args.expected_host not in LOOPBACK:
        others = sorted(set(others) | {args.expected_host})
    if others:
        print("\nother interfaces:")
        for address in others:
            for port in LIVEKIT_PORTS:
                if reachable(address, port):
                    print(f"  {address}:{port} REACHABLE")
                    exposed.append(f"{address}:{port}")

    if not listening and not exposed:
        print("\nNo LiveKit ports are reachable. Is the server running?", file=sys.stderr)
        return 2

    if not exposed:
        print("\nOK: LiveKit is reachable on loopback only.")
        return 0

    print(f"\nExposed beyond loopback: {', '.join(exposed)}")
    if args.allow_lan:
        print("Allowed by --allow-lan. Confirm your firewall limits this to the trusted LAN.")
        return 0
    print(
        "docs/07-Privacy-and-Security.md requires signaling and RTC ports to be\n"
        "restricted to the trusted LAN. Restrict them, or pass --allow-lan if\n"
        "this exposure is deliberate.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
