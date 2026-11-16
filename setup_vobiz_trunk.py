"""
One-time setup: register your Vobiz SIP trunk with LiveKit for outbound calls.

    python setup_vobiz_trunk.py

Creates the trunk (or updates it, if OUTBOUND_TRUNK_ID is already set) and
prints the trunk id to put in ``.env``.
"""

import asyncio

from voice_agent import config


async def setup_trunk() -> int:
    try:
        from livekit import api
    except ImportError:
        print("Error: the LiveKit SDK is not installed. Run: pip install -r requirements.txt")
        return 1

    if not config.telephony_configured():
        print("Error: set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env first.")
        return 1
    if not config.VOBIZ_SIP_DOMAIN:
        print("Error: set VOBIZ_SIP_DOMAIN in .env (Vobiz Console → SIP Trunk Settings).")
        return 1

    numbers = [config.VOBIZ_OUTBOUND_NUMBER] if config.VOBIZ_OUTBOUND_NUMBER else []
    print("Vobiz SIP trunk")
    print(f"  Domain:   {config.VOBIZ_SIP_DOMAIN}")
    print(f"  Username: {config.VOBIZ_USERNAME or '(none)'}")
    print(f"  Number:   {config.VOBIZ_OUTBOUND_NUMBER or '(none)'}")

    client = api.LiveKitAPI(
        url=config.LIVEKIT_URL, api_key=config.LIVEKIT_API_KEY, api_secret=config.LIVEKIT_API_SECRET
    )
    try:
        if config.OUTBOUND_TRUNK_ID:
            await client.sip.update_outbound_trunk_fields(
                config.OUTBOUND_TRUNK_ID,
                address=config.VOBIZ_SIP_DOMAIN,
                auth_username=config.VOBIZ_USERNAME,
                auth_password=config.VOBIZ_PASSWORD,
                numbers=numbers,
            )
            print(f"\nTrunk {config.OUTBOUND_TRUNK_ID} updated.")
        else:
            trunk = await client.sip.create_sip_outbound_trunk(
                api.CreateSIPOutboundTrunkRequest(
                    trunk=api.SIPOutboundTrunkInfo(
                        name="Vobiz Outbound",
                        address=config.VOBIZ_SIP_DOMAIN,
                        numbers=numbers,
                        auth_username=config.VOBIZ_USERNAME,
                        auth_password=config.VOBIZ_PASSWORD,
                    )
                )
            )
            print("\nTrunk created. Add this line to your .env:")
            print(f"  OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
    except Exception as exc:
        print(f"\nError: {exc}")
        print("If this is a 'not found' error, enable SIP for your project at https://cloud.livekit.io")
        return 1
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(setup_trunk()))
