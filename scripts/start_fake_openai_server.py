from __future__ import annotations

import time
from typing import Any

import uvicorn
from fastapi import FastAPI


app = FastAPI()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "fake-qwen-0.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-test",
            }
        ],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 32))

    user_content = ""
    if messages:
        last_message = messages[-1]
        user_content = str(last_message.get("content", ""))

    fake_answer = f"This is a fake response for local integration test. Prompt: {user_content[:50]}"
    fake_tokens = min(max_tokens, max(1, len(fake_answer.split())))

    return {
        "id": "chatcmpl-fake-local-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "fake-qwen-0.5b"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fake_answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_content.split()),
            "completion_tokens": fake_tokens,
            "total_tokens": len(user_content.split()) + fake_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)