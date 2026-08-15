# Let real glasses on Wi-Fi reach the stack running in WSL.
#
# Run in an **Administrator** PowerShell on Windows:
#
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_lan_expose.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_lan_expose.ps1 -Remove
#
# Requires WSL2 mirrored networking (see %USERPROFILE%\.wslconfig):
#
#   [wsl2]
#   networkingMode=mirrored
#   hostAddressLoopback=true
#
# With mirrored networking WSL shares the Windows network stack and owns the
# laptop's address itself, so **no port forwarding is needed** -- this script
# only opens the firewall. That is a deliberate simplification of what used to
# be here: `netsh portproxy` is a userland TCP proxy that re-originates each
# connection, so LiveKit saw every peer arriving from the host address, ICE
# never validated a candidate pair, and every join failed as JOIN_TIMEOUT.
# Forwarding cannot carry WebRTC. Mirrored networking removes the need for it.
#
# Leftover portproxy rules are actively harmful once mirrored: WSL and Windows
# share one port space, so an old rule squats on the port its Linux service
# wants and the launcher reports "port 8081 is already in use" with nothing
# visible in `ss`. This script reports any it finds.
#
# Why the firewall matters: a Wi-Fi network is usually classified Public, where
# Windows blocks inbound by default. The glasses then time out reaching the
# laptop even though both are on the same subnet.

param(
    [switch]$Remove
)

$ruleName = "VMA glasses (WSL2 mirrored)"
$tcpPorts = @(8080, 7880, 7881)   # gateway, LiveKit signalling, LiveKit TCP media
$udpPorts = @(7882)               # LiveKit UDP mux -- what WebRTC actually wants

Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($Remove) {
    Write-Host "removed firewall rule '$ruleName'"
    exit 0
}

$wifiIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -eq "Wi-Fi" } |
    Select-Object -First 1).IPAddress
Write-Host "laptop Wi-Fi address: $wifiIp"
Write-Host "  livekit.dev.yaml must have  node_ip: $wifiIp"
Write-Host "  .env must have              VMA_LIVEKIT_URL / _PUBLIC_URL = ws://${wifiIp}:7880"
Write-Host "  console .env.local must have VITE_VMA_GATEWAY_PUBLIC_URL = http://${wifiIp}:8080"
Write-Host ""

# All profiles: a Wi-Fi network is normally Public, and which profile applies
# is not worth being subtle about on a demo machine.
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort $tcpPorts -Profile Any | Out-Null
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
    -Protocol UDP -LocalPort $udpPorts -Profile Any | Out-Null
Write-Host "allowed inbound TCP $($tcpPorts -join ', ') and UDP $($udpPorts -join ', ')"

$stale = netsh interface portproxy show all | Select-String -Pattern '\s(8080|8081|8082|8085|8086|5173|7880|7881)\s'
if ($stale) {
    Write-Host ""
    Write-Warning "portproxy rules still exist and will squat on these ports under mirrored networking:"
    $stale | ForEach-Object { Write-Host "  $_" }
    Write-Host "  delete each with: netsh interface portproxy delete v4tov4 listenaddress=<addr> listenport=<port>"
}

Write-Host ""
Write-Host "Then, inside WSL:  VMA_BIND_ADDR=0.0.0.0 ./scripts/dev_stack.sh"
