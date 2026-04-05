"""
make_vobiz_call.py – Make an outbound call via LiveKit + Vobiz.

Usage:
  python make_vobiz_call.py --to +919988776655

Prerequisites:
  1. Vobiz agent must be running (python telephony_vobiz.py start)
  2. SIP trunk must be configured (python setup_vobiz_trunk.py)
  3. All credentials must be in .env
"""

import argparse
import asyncio
import json
import os
import random

from dotenv import load_dotenv

load_dotenv()


async def make_call(phone_number: str):
    """Dispatch an outbound call via LiveKit to a phone number."""

    try:
        from livekit import api
    except ImportError:
        print("❌ LiveKit SDK not installed!")
        print("Run: pip install livekit-api")
        return

    # Validate phone number
    phone_number = phone_number.strip()
    if not phone_number.startswith("+"):
        print("❌ Phone number must start with '+' and country code.")
        print("   Example: +919988776655")
        return

    url        = os.getenv("LIVEKIT_URL", "")
    api_key    = os.getenv("LIVEKIT_API_KEY", "")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "")

    if not all([url, api_key, api_secret]):
        print("❌ LiveKit credentials missing in .env!")
        print("Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")
        return

    # Create LiveKit API client
    lk = api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)

    # Create unique room for this call
    room_name = f"call-{phone_number.replace('+', '')}-{random.randint(1000, 9999)}"

    print(f"📞 Initiating call to {phone_number}...")
    print(f"   Room: {room_name}")

    try:
        # Dispatch the agent with the phone number in metadata
        dispatch_request = api.CreateAgentDispatchRequest(
            agent_name="voice-assistant",   # Must match telephony_vobiz.py
            room=room_name,
            metadata=json.dumps({"phone_number": phone_number}),
        )

        dispatch = await lk.agent_dispatch.create_dispatch(dispatch_request)

        print(f"\n✅ Call Dispatched Successfully!")
        print(f"   Dispatch ID: {dispatch.id}")
        print(f"{'─' * 50}")
        print(f"   The agent is joining the room and will dial the number.")
        print(f"   Check the agent terminal for logs.")

    except Exception as e:
        print(f"\n❌ Error dispatching call: {e}")

    finally:
        await lk.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make an outbound call via LiveKit + Vobiz.")
    parser.add_argument("--to", required=True, help="Phone number to call (e.g., +919988776655)")
    args = parser.parse_args()

    asyncio.run(make_call(args.to))
