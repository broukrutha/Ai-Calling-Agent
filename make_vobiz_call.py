"""
Place an outbound call.

    python make_vobiz_call.py --to +919988776655

The telephony worker (``python telephony_vobiz.py start``) must be running: it
receives the dispatch, dials the number and holds the conversation.
"""

import argparse
import asyncio

from voice_agent.logging_config import mask_phone
from voice_agent.telephony.outbound import TelephonyNotConfiguredError, dispatch_call


async def _run(number: str) -> int:
    try:
        call = await dispatch_call(number)
    except (ValueError, TelephonyNotConfiguredError) as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error: could not reach LiveKit ({exc}).")
        return 1
    print(f"Call queued to {mask_phone(number)}")
    print(f"  Room:        {call['room']}")
    print(f"  Dispatch ID: {call['dispatch_id']}")
    print("Follow progress in the worker's log or the Calls tab of the dashboard.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place an outbound call through LiveKit + Vobiz.")
    parser.add_argument("--to", required=True, help="number in international format, e.g. +919988776655")
    raise SystemExit(asyncio.run(_run(parser.parse_args().to)))
