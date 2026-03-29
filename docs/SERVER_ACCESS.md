# Server Access Guide

Quick reference for Spaitra team members: how to reach the server, run
the API, and integrate from the frontend.

---

## 1. Server info (self-hosted on a desktop, should migrate later)

| Item | Value |
|------|-------|
| Hostname | `Spaitra` |
| Tailscale IP | `100.114.39.23` |
| Public hostname | `https://nre5bjw44wddpu2zjg4fe4iehq.srv.us` |
| OS | Debian 13 |
| GPU | NVIDIA GTX 1060 6 GB |
| Core API port | `5000` (localhost only, exposed through tunnel) |
| OCR service port | `8001` (localhost only) |

---

## 2. SSH access (Tailscale VPN)

You need to be on the Tailscale VPN to reach the server directly.

```bash
# First time: install Tailscale and join the network
# https://tailscale.com/download
# Run: sudo tailscale up

# Then SSH in (replace <yourname> with your server account):
ssh dev@100.114.39.23

# Example for the dev account:
ssh dev@100.114.39.23
```

The `spaitra` service account owns the code. You can switch to it from `dev`:

```bash
sudo -i -u spaitra
```

Project root (repo):

```
/opt/spaitra/TSA-soft-dev-backend-2026/
```

Environment files (secrets, not in git):

```
/opt/spaitra/.env        # core API
/opt/spaitra/.ocr.env    # OCR service
```

---

## 3. Public access via srv.us (no VPN required)

The server has a permanent public HTTPS hostname via `srv.us`:

```
https://nre5bjw44wddpu2zjg4fe4iehq.srv.us
```

This tunnels to port 5000 on the server. No VPN needed from any device.

### Starting the tunnel manually

```bash
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 1:localhost:5000 srv.us
```

> **Note:** if your SSH key has a passphrase, the tunnel will prompt for it.
> For unattended use (systemd service), generate a passphrase-free key for the
> `spaitra` service account: `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519`

### Running the tunnel persistently (systemd)

Write `/etc/systemd/system/spaitra-tunnel.service`:

```ini
[Unit]
Description=srv.us tunnel for Spaitra core API
After=network-online.target spaitra-core.service
Wants=network-online.target

[Service]
User=spaitra
Restart=always
RestartSec=15
ExecStart=/usr/bin/ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -R 1:localhost:5000 srv.us

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now spaitra-tunnel
```

---

## 4. API authentication

Every request (except `GET /health`) requires the `X-API-Key` header. The key
is in `/opt/spaitra/.env` under `API_KEY`.

```bash
# Read the key (as spaitra or root):
grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-
```

Set it as a variable for the examples below:

```bash
KEY="<your-api-key>"
BASE="https://nre5bjw44wddpu2zjg4fe4iehq.srv.us"   # or http://100.114.39.23:5000 on VPN
```

---

## 5. Common API calls

### Health check (no auth required)

```bash
curl "$BASE/health"
# {"status":"ok"}
```

### Remember; teach the model an object

Upload a photo and associate it with a label.

```bash
curl -X POST "$BASE/remember" \
  -H "X-API-Key: $KEY" \
  -F "image=@/path/to/photo.jpg" \
  -F "prompt=red mug"
```

**Response fields:**
- `success`; boolean
- `result.label`; stored label
- `result.confidence`; detector confidence (0-1)
- `result.detection_quality`; `"low"` / `"medium"` / `"high"`
- `result.detection_hint`; human-readable quality tip
- `result.box`; `[x1, y1, x2, y2]` pixel coordinates
- `result.ocr_text`; extracted text (if any)
- `result.is_blurry`, `result.blur_score`; sharpness feedback

### Scan; find known objects in a frame

```bash
curl -X POST "$BASE/scan" \
  -H "X-API-Key: $KEY" \
  -F "image=@/path/to/frame.jpg" \
  -F "focal_length_px=3200"
```

**Response fields per match:**
- `label`; matched object name
- `similarity`; cosine similarity (0-1, higher = better match)
- `confidence`; `"low"` / `"medium"` / `"high"`
- `direction`; `"to your left"` / `"slightly left"` / `"ahead"` / `"slightly right"` / `"to your right"`
- `distance_ft`; estimated distance in feet (when depth enabled)
- `narration`; ready-to-speak string for assistive output
- `box`; `[x1, y1, x2, y2]`

### Submit match feedback (improve accuracy over time)

```bash
curl -X POST "$BASE/feedback" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "<uuid-from-scan-response>",
    "label": "red mug",
    "correct": true
  }'
```

### List all remembered objects

```bash
curl "$BASE/items" -H "X-API-Key: $KEY"
```

### Delete a remembered object

```bash
curl -X DELETE "$BASE/items/red%20mug" -H "X-API-Key: $KEY"
```

### Trigger model retraining

Retraining only runs when at least 10 positive+negative feedback pairs have
been collected. Safe to call anytime; returns `insufficient_data` if not
enough feedback yet.

```bash
curl -X POST "$BASE/retrain" -H "X-API-Key: $KEY"
```

---

## 6. Debug endpoints 

These require the API key and are useful for verifying the running state of the
server.

```bash
# Full system state: GPU, model load status, settings
curl "$BASE/debug/state" -H "X-API-Key: $KEY" | python3 -m json.tool

# Smoke-test remember pipeline end-to-end (uses bundled test image)
curl "$BASE/debug/test-remember" -H "X-API-Key: $KEY" | python3 -m json.tool

# Smoke-test scan pipeline end-to-end
curl "$BASE/debug/test-scan" -H "X-API-Key: $KEY" | python3 -m json.tool

# Dump all stored items in the DB
curl "$BASE/debug/db" -H "X-API-Key: $KEY" | python3 -m json.tool

# Tail recent application logs
curl "$BASE/debug/logs" -H "X-API-Key: $KEY"
```

---

## 7. Service management (SSH, as root or sudo)

```bash
# Status
systemctl status spaitra-core spaitra-ocr

# Restart after a code change
sudo systemctl restart spaitra-core

# Follow logs
journalctl -u spaitra-core -f
journalctl -u spaitra-ocr -f
```

---

## 8. Deploying a code change

```bash
cd /opt/spaitra/TSA-soft-dev-backend-2026
sudo -u spaitra git pull
sudo systemctl restart spaitra-core
```

No migration step is needed for most changes. If `settings.py` or the DB
schema changed, check `DEPLOY.md - Troubleshooting` before restarting.
