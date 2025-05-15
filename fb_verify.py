
import requests
from fastapi.responses import PlainTextResponse
import os
from dotenv import load_dotenv

load_dotenv()

USER_ACCESS_TOKEN = os.getenv("USER_ACCESS_TOKEN")  
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")

print("USER_ACCESS_TOKEN", USER_ACCESS_TOKEN)
print("VERIFY_TOKEN", VERIFY_TOKEN)


def get_page_access_token():
    url = f"https://graph.facebook.com/v17.0/me/accounts?access_token={USER_ACCESS_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if "data" in data:
        for page in data["data"]:
            page_access_token = page["access_token"]
            return page_access_token
    else:
        return None

PAGE_ACCESS_TOKEN = get_page_access_token() 
print("PAGE_ACCESS_TOKEN", PAGE_ACCESS_TOKEN)

def verify_token(query_params):
    mode = query_params.get("hub.mode")
    token = query_params.get("hub.verify_token")
    challenge = query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Verification failed", status_code=403)

def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    print("Sent status:", response.status_code)
    print("Sent body:", response.text)

