"""
setup_vobiz_trunk.py – One-time setup to create/update your Vobiz SIP trunk in LiveKit.

Run this ONCE after you have your Vobiz and LiveKit credentials:
  python setup_vobiz_trunk.py

What it does:
  1. Creates an outbound SIP trunk in LiveKit Cloud
  2. Configures it with your Vobiz SIP credentials
  3. Prints the TRUNK_ID — add this to your .env as OUTBOUND_TRUNK_ID
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def setup_trunk():
    """Create or update a Vobiz SIP trunk in LiveKit."""

    try:
        from livekit import api
    except ImportError:
        print("❌ LiveKit SDK not installed!")
        print("Run: pip install livekit-api")
        return

    # Load credentials
    livekit_url    = os.getenv("LIVEKIT_URL", "")
    livekit_key    = os.getenv("LIVEKIT_API_KEY", "")
    livekit_secret = os.getenv("LIVEKIT_API_SECRET", "")

    if not all([livekit_url, livekit_key, livekit_secret]):
        print("❌ Missing LiveKit credentials in .env!")
        print("Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")
        return

    sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
    username   = os.getenv("VOBIZ_USERNAME", "")
    password   = os.getenv("VOBIZ_PASSWORD", "")
    number     = os.getenv("VOBIZ_OUTBOUND_NUMBER", "")
    trunk_id   = os.getenv("OUTBOUND_TRUNK_ID", "")

    if not sip_domain:
        print("❌ VOBIZ_SIP_DOMAIN not set in .env!")
        print("Get this from your Vobiz Console → SIP Trunk Settings")
        return

    print(f"📞 Vobiz SIP Trunk Setup")
    print(f"   Domain:   {sip_domain}")
    print(f"   Username: {username}")
    print(f"   Number:   {number}")
    print(f"   Trunk ID: {trunk_id or '(will create new)'}")
    print()

    # Initialize LiveKit API
    lk = api.LiveKitAPI(
        url=livekit_url,
        api_key=livekit_key,
        api_secret=livekit_secret,
    )

    try:
        if trunk_id:
            # Update existing trunk
            print(f"Updating existing trunk: {trunk_id}")
            await lk.sip.update_outbound_trunk_fields(
                trunk_id,
                address=sip_domain,
                auth_username=username,
                auth_password=password,
                numbers=[number] if number else [],
            )
            print(f"\n✅ SIP Trunk updated successfully!")
        else:
            # Create new trunk
            print("Creating new outbound SIP trunk...")

            from livekit.protocol import sip as sip_proto

            trunk_request = sip_proto.CreateSIPOutboundTrunkRequest(
                trunk=sip_proto.SIPOutboundTrunkInfo(
                    name="Vobiz Outbound",
                    address=sip_domain,
                    numbers=[number] if number else [],
                    auth_username=username,
                    auth_password=password,
                )
            )

            result = await lk.sip.create_sip_outbound_trunk(trunk_request)
            new_trunk_id = result.sip_trunk_id

            print(f"\n✅ SIP Trunk created successfully!")
            print(f"   Trunk ID: {new_trunk_id}")
            print()
            print(f"⚠️  Add this to your .env file:")
            print(f"   OUTBOUND_TRUNK_ID={new_trunk_id}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        print()
        print("If you get 'not found' errors, make sure to enable SIP in your")
        print("LiveKit Cloud project settings at https://cloud.livekit.io")

    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(setup_trunk())
