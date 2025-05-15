# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from app.fb_verify import verify_token, send_message
from app.llama_chatbot import get_bot_response
import time

app = FastAPI()

server_start_time = int(time.time())  
processed_message_ids = set()        

@app.get("/")
def root():
    return {"message": "LLAMA Chatbot Online"}

@app.get("/webhook")
async def verify(request: Request):
    return verify_token(request.query_params)

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    print("✅ Webhook received:", body)

    for entry in body.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            timestamp = messaging_event.get("timestamp", 0)
            if timestamp < server_start_time * 1000:
                continue 

            message = messaging_event.get("message")
            if not message or "text" not in message:
                continue 

            message_id = message.get("mid")
            if message_id in processed_message_ids:
                continue 

            processed_message_ids.add(message_id)

            sender_id = messaging_event["sender"]["id"]
            message_text = message["text"]

            print("Getting response from chatbot")
            reply = get_bot_response(message_text)
            print("Getting response successfully, start to send")
            send_message(sender_id, reply)

    return PlainTextResponse("OK", status_code=200)
