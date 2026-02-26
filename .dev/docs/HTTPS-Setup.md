# HTTPS Setup for Habla

Browser microphone access requires a secure context (HTTPS or localhost).
Since Habla runs on a home server accessed remotely via Tailscale, we use
Tailscale's built-in HTTPS support.

## Option 1: Tailscale HTTPS (Recommended)

This is the simplest approach — no reverse proxy needed.

### Prerequisites
- Tailscale installed on the server and your Android phone
- MagicDNS enabled in Tailscale admin console (https://login.tailscale.com/admin/dns)
- HTTPS certificates enabled in Tailscale admin (DNS > HTTPS Certificates toggle)

### Setup

1. **Get your machine's Tailscale FQDN:**
   ```bash
   tailscale status
   # Look for your machine name, e.g. "msi-laptop"
   # FQDN will be: msi-laptop.tailnet-name.ts.net
   ```

2. **Generate TLS certificate:**
   ```bash
   tailscale cert msi-laptop.tailnet-name.ts.net
   # Creates: msi-laptop.tailnet-name.ts.net.crt
   #          msi-laptop.tailnet-name.ts.net.key
   ```

3. **Run uvicorn with TLS:**
   ```bash
   cd habla
   uvicorn server.main:app --host 0.0.0.0 --port 8002 \
     --ssl-certfile msi-laptop.tailnet-name.ts.net.crt \
     --ssl-keyfile msi-laptop.tailnet-name.ts.net.key
   ```

4. **Access from Android:**
   Open Chrome and navigate to:
   ```
   https://msi-laptop.tailnet-name.ts.net:8002
   ```
   The browser will trust the certificate (Tailscale acts as the CA).
   Microphone access will work since it's a secure context.

### Certificate Renewal
Tailscale certificates expire after 90 days. Re-run `tailscale cert` to renew.
Consider a cron job:
```bash
# Renew Tailscale cert weekly (idempotent — only renews if needed)
0 3 * * 0 tailscale cert msi-laptop.tailnet-name.ts.net
```

## Option 2: Caddy Reverse Proxy

If you prefer a reverse proxy (e.g., for multiple services):

```
# Caddyfile
msi-laptop.tailnet-name.ts.net:8002 {
    reverse_proxy localhost:8002
}
```

Caddy auto-provisions Let's Encrypt certificates. WebSocket upgrade works
automatically — no special configuration needed.

Run with: `caddy run --config Caddyfile`

## WebSocket Through HTTPS

Both options above handle WebSocket upgrade transparently. The client
connects via `wss://` instead of `ws://` — the PWA's `websocket.js`
already detects the protocol automatically:

```javascript
const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
```

## Troubleshooting

- **"NotAllowedError: Permission denied" for microphone**: Not using HTTPS.
  Verify you're accessing via `https://` not `http://`.
- **Certificate not trusted on Android**: Ensure MagicDNS and HTTPS certificates
  are enabled in Tailscale admin. The phone must be logged into the same tailnet.
- **WebSocket connection fails**: Check that the firewall allows port 8002 on
  the server. Tailscale traffic bypasses most firewalls but the local firewall
  still applies for direct connections.
