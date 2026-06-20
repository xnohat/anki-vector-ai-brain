# Always-On Mic — OSKR Flash & Continuous-Mic Plan (SCOPE — not yet executed)

Goal: a true **always-on wake word** (Brain hears "Vector"/Vietnamese trigger from
the room mic continuously, no button press). This document scopes the work,
risks, and a safe step-by-step. **Nothing here has been flashed yet** — this is a
plan to review before touching firmware.

---

## 0. Why this is needed (findings, 2026-06-21)

| Fact | Evidence |
|------|----------|
| Robot firmware is **`2.0.1.6076ep`** | gRPC `get_version_state()` |
| `ep` = escape-pod-compatible, **but the LOCKED (non-OSKR) variant** | suffix `ep`; no SSH |
| **SSH (port 22) closed even fully awake** | woke via SDK, scanned 22 ×5 → closed; only 443 open |
| wire-pod can't enable SSH | `setup/ssh.go` only *dials* 22; expects dropbear already up |
| OSKR gRPC has no shell RPC | `oskrpb` exposes only `GetWifiSignalStrength` |
| Vector→wire-pod via escape-pod cert | `certs/server_config.json` → `escapepod.local:443` |

**Conclusion:** there is **no SSH/shell path on the current firmware**. The SDK
`AudioFeed` is a 1 kHz stub (documented earlier), so the room mic is reachable
**only** through firmware that opens it. Continuous mic ⇒ we must run our own code
on the robot ⇒ we need root ⇒ on this bot that requires flashing **OSKR** firmware.

This is the *only* blocker that forces a firmware change in the whole project.

---

## 1. Two-phase plan

### Phase A — Flash OSKR (get root + SSH)  [USER-DRIVEN, physical]
Tool: **kercre123 `wire-prod-pod`** / `wpsetup.keriganc.com` (BLE web flasher).
Works on non-unlocked bots. Reversible (can flash production firmware back).

Steps (the user performs A2–A4; the flasher needs Chrome + Bluetooth on a phone/PC):
1. **A1 — Backup first.** Record current state so we can fully restore:
   - `os_version` = `2.0.1.6076ep` (already noted).
   - Save `~/wire-pod/certs/server_config.json`, `~/.anki_vector/`, wire-pod
     `apiConfig.json`, and the robot serial `00907f6b` / ESN. (Script: `scripts/backup_bot_state.sh` — to be written.)
   - Note: OSKR firmware will likely reset robot settings (button_wakeword, locale);
     `run.sh` already re-applies `button_hey_vector` on boot.
2. **A2 — Recovery mode:** Vector on charger → hold backpack button ~15 s until he
   powers off, keep holding until lights return → screen shows `anki.com/v` or `ddl.io/v`.
3. **A3 — Flash:** open Chrome → BLE flasher page → select the bot → flash the
   **OSKR/dev firmware** (the one that ships dropbear + accepts a custom SSH key).
4. **A4 — Re-onboard to wire-pod** (BLE setup or wire-pod's SSH setup once SSH is up).
5. **A5 — Verify:** `get_version_state()` shows an OSKR build (no locked `ep`), and
   `nmap`/scan shows **port 22 OPEN**; `ssh -i data/ssh/id_rsa root@<ip>` gives a shell.

**Decision gate:** do NOT proceed to Phase B until A5 passes and we've confirmed
we can flash production firmware back (rollback rehearsed).

### Phase B — Continuous mic + Pi-side wake word  [ENGINEERING, our code]
Now that we have root, the room mic becomes reachable. Two implementation options:

- **B-opt1 (preferred): modify `vic-cloud`.** Source is already here:
  `~/wire-pod/vector-cloud/internal/voice/` (`process.go`, `stream/`). Today it
  opens the mic stream only **after a trigger** and runs until end-of-speech
  (mirrors `wire-pod .../stt/brain/Brain.go` reading chunks → `DetectEndOfSpeech`).
  Change: keep the mic stream **open continuously** and push frames to wire-pod
  even with no trigger. Cross-compile vic-cloud (the repo already builds it),
  SCP to `/anki/bin/vic-cloud`, restart `anki-robot.target`.
- **B-opt2: a small standalone capturer** on the robot that reads the mic device
  and streams PCM to a new wire-pod endpoint. Less invasive to vic-cloud but more
  new code + audio-routing reverse-engineering.

Pi side (wire-pod / `brain_server.py`):
- New always-listening intake: receive the continuous PCM, run **VAD** (wire-pod
  already vendors `go-webrtcvad`) to gate silence, then **wake-word match** the
  utterance ("vector" + Vietnamese variants) cheaply (local matcher or a tiny
  STT) BEFORE spending a GPT call.
- On wake-word hit → same path as today (transcribe → brain → reply), and reuse
  the existing **hands-free `{{newVoiceRequest||now}}`** for the follow-up turns.
- Cost guard: only the wake-word matcher runs on the always-on stream; GPT/the
  expensive STT runs only after the trigger. (Matches the project's push/pull rule.)

---

## 2. Risks & reversibility

| Risk | Severity | Mitigation |
|------|----------|------------|
| Bricking during flash | Low | Recovery-mode BLE flash is designed for this; bot can re-enter recovery and reflash |
| Lose escape-pod/wire-pod pairing | Medium | Backup `server_config.json` + re-onboard step A4; rollback to prod firmware |
| Battery dies mid-flash | Medium | Flash only on charger, fully charged |
| Robot settings reset (locale/button) | Low | `run.sh` re-applies button; re-set locale/volume after |
| Continuous mic drains battery / heats | Medium | VAD-gate on robot; throttle frame rate; only stream when off-charger or on demand |
| vic-cloud mod breaks voice entirely | Medium | Keep a copy of the working `ep` vic-cloud; SCP-restore to roll back B without re-flashing |
| Wi-Fi too weak for a constant stream | Medium | Already seeing 60–120 ms ping/jitter; move bot near AP; consider Opus low-bitrate |

**Full rollback:** flash production firmware via the same BLE recovery flasher →
bot returns to stock. Phase B alone rolls back by restoring the original
`/anki/bin/vic-cloud`.

---

## 3. Effort estimate

- Phase A (flash + verify): ~1–2 h, mostly the user's hands-on BLE flash.
- Phase B-opt1 (vic-cloud continuous mic): the real work — cross-compile toolchain
  for the robot (armv7), understand `internal/voice/stream`, get a clean
  continuous push, then Pi-side VAD+wake-word. Several focused sessions.

---

## 4. Recommendation / open questions before flashing

1. Confirm we have (or can fetch) the **OSKR firmware image + the `data/ssh/id_rsa`
   key** the flasher uses, and that the flasher still works on current Chrome.
2. Rehearse **rollback to production** firmware once, on purpose, before relying on it.
3. Decide battery policy for always-on (stream only off-charger? duty-cycle?).
4. Only then execute Phase A.

Until the user approves Phase A, the shipped **hands-free conversation** (press the
button once → keep talking; "tạm biệt"/"đi ngủ" ends) is the best no-firmware-risk
experience.
