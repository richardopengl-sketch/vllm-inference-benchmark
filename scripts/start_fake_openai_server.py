from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


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
def create_chat_completion(payload: dict[str, Any]) -> Any:
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 32))

    user_content = ""
    if messages:
        last_message = messages[-1]
        user_content = str(last_message.get("content", ""))

    fake_answer = f"This is a fake response for local integration test. Prompt: {user_content[:50]}"
    word_pieces = fake_answer.split()[:max_tokens]
    fake_answer = " ".join(word_pieces)
    fake_tokens = len(word_pieces)
    created = int(time.time())
    model = payload.get("model", "fake-qwen-0.5b")

    if payload.get("stream", False):
        async def generate_events():
            role_chunk = {
                "id": "chatcmpl-fake-local-test",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            if word_pieces:
                await asyncio.sleep(0.1)
            for index, piece in enumerate(word_pieces):
                if index > 0:
                    await asyncio.sleep(0.03)
                content_chunk = {
                    "id": "chatcmpl-fake-local-test",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece if index == 0 else f" {piece}"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

            finish_chunk = {
                "id": "chatcmpl-fake-local-test",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"

            usage_chunk = {
                "id": "chatcmpl-fake-local-test",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": len(user_content.split()),
                    "completion_tokens": fake_tokens,
                    "total_tokens": len(user_content.split()) + fake_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_events(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-fake-local-test",
        "object": "chat.completion",
        "created": created,
        "model": model,
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