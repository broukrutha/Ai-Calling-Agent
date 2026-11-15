"""
Telephony worker entry point (LiveKit + Vobiz SIP).

    python telephony_vobiz.py start     # production
    python telephony_vobiz.py dev       # development, reloads on change

Setup, in order:
  1. LiveKit Cloud project  → LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  2. Vobiz SIP trunk + number → VOBIZ_* settings
  3. python setup_vobiz_trunk.py  → prints OUTBOUND_TRUNK_ID for .env
  4. Point your Vobiz number at LiveKit's SIP endpoint for inbound calls
  5. Start this worker, then place a call:  python make_vobiz_call.py --to +91...
"""

try:
    from voice_agent.telephony.agent import main
except ImportError as exc:
    raise SystemExit(
        f"Telephony dependencies are missing ({exc.name}). Install them with:\n"
        "  pip install -r requirements.txt"
    ) from exc

if __name__ == "__main__":
    main()
