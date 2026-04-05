"""Quick test to check OpenAI API key status."""
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY", "")
print(f"Key prefix: {key[:20]}...")
print(f"Key length: {len(key)}")

# Test 1: Simple completion
print("\n--- Test 1: Simple chat completion ---")
try:
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 5,
        },
        timeout=30,
    )
    print(f"Status: {resp.status_code}")
    data = resp.json()
    if resp.status_code == 200:
        print(f"Response: {data['choices'][0]['message']['content']}")
        print("✅ OpenAI API key is WORKING!")
    else:
        error = data.get("error", {})
        print(f"Error type: {error.get('type')}")
        print(f"Error message: {error.get('message')}")
        print(f"Error code: {error.get('code')}")
        if "quota" in str(error.get("message", "")).lower():
            print("\n❌ QUOTA EXCEEDED — Your mentor needs to:")
            print("   1. Go to https://platform.openai.com/settings/organization/limits")
            print("   2. Check 'Usage limits' — increase monthly spend limit")
            print("   3. Or go to Billing → Add more credits")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Check organization/project limits
print("\n--- Test 2: Check models access ---")
try:
    resp = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    print(f"Models endpoint status: {resp.status_code}")
    if resp.status_code == 200:
        models = [m["id"] for m in resp.json()["data"] if "gpt-4o" in m["id"]]
        print(f"Available GPT-4o models: {models[:5]}")
except Exception as e:
    print(f"Error: {e}")
