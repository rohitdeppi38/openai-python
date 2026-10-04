from __future__ import annotations

import json

import httpx2
import pytest
from pydantic import BaseModel

from openai import OpenAI, AsyncOpenAI


def _null_content_item() -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": None,
    }


def _completed_event(output: object) -> dict[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_test",
            "status": "completed",
            "model": "test-model",
            "output": output,
        },
    }


def _stream_body(output: object) -> bytes:
    events: list[dict[str, object]] = [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_test", "status": "in_progress"}},
        _completed_event(output),
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_stream_explicit_null_message_content(sync: bool) -> None:
    """An explicit completed output whose message has content:null must parse to empty content, not TypeError."""
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(
            200, content=_stream_body([_null_content_item()]), headers={"content-type": "text/event-stream"}
        )
    )
    if sync:
        with OpenAI(api_key="fake-test-key", http_client=httpx2.Client(transport=transport)) as client:
            with client.responses.stream(model="test-model", input="test") as stream:
                emitted = list(stream)
                final = stream.get_final_response()
    else:
        async with AsyncOpenAI(
            api_key="fake-test-key", http_client=httpx2.AsyncClient(transport=transport)
        ) as async_client:
            async with async_client.responses.stream(model="test-model", input="test") as async_stream:
                emitted = [event async for event in async_stream]
                final = await async_stream.get_final_response()

    completed = emitted[-1]
    assert completed.type == "response.completed"
    assert completed.response == final
    assert final.output_text == ""
    assert final.output_parsed is None
    assert len(final.output) == 1
    message = final.output[0]
    assert message.type == "message"
    assert message.id == "msg_test"
    assert message.role == "assistant"
    assert message.status == "completed"
    assert message.content == []


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_parse_null_message_content(sync: bool) -> None:
    """Non-streaming responses.parse must not raise TypeError on content:null either."""
    body = json.dumps(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "model": "test-model",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [_null_content_item()],
        }
    )
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(200, content=body, headers={"content-type": "application/json"})
    )
    if sync:
        with OpenAI(api_key="fake-test-key", http_client=httpx2.Client(transport=transport)) as client:
            parsed = client.responses.parse(model="test-model", input="test")
    else:
        async with AsyncOpenAI(
            api_key="fake-test-key", http_client=httpx2.AsyncClient(transport=transport)
        ) as async_client:
            parsed = await async_client.responses.parse(model="test-model", input="test")

    assert parsed.output_text == ""
    assert parsed.output_parsed is None
    assert len(parsed.output) == 1
    message = parsed.output[0]
    assert message.type == "message"
    assert message.id == "msg_test"
    assert message.role == "assistant"
    assert message.status == "completed"
    assert message.content == []


class Result(BaseModel):
    answer: str


def _text_message(text: str | None) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@pytest.mark.parametrize("text", [None, ""], ids=["null", "empty"])
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_parse_null_or_empty_output_text(sync: bool, text: str | None) -> None:
    """responses.parse must not try to decode output text that is null or empty."""
    body = json.dumps(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "model": "test-model",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [_text_message(text)],
        }
    )
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(200, content=body, headers={"content-type": "application/json"})
    )
    if sync:
        with OpenAI(api_key="fake-test-key", http_client=httpx2.Client(transport=transport)) as client:
            parsed = client.responses.parse(model="test-model", input="test", text_format=Result)
    else:
        async with AsyncOpenAI(
            api_key="fake-test-key", http_client=httpx2.AsyncClient(transport=transport)
        ) as async_client:
            parsed = await async_client.responses.parse(model="test-model", input="test", text_format=Result)

    assert parsed.output_parsed is None
    message = parsed.output[0]
    assert message.type == "message"
    content = message.content[0]
    assert content.type == "output_text"
    assert content.text == text
    assert content.parsed is None


@pytest.mark.parametrize("text", [None, ""], ids=["null", "empty"])
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
async def test_stream_null_or_empty_output_text(sync: bool, text: str | None) -> None:
    """A text-done event with null or empty text must not be decoded as structured output."""
    events: list[dict[str, object]] = [
        {"type": "response.created", "response": {"id": "resp_test", "status": "in_progress", "output": []}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**_text_message(None), "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_test",
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_test",
            "text": text,
            "logprobs": [],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": _text_message(text)},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "status": "completed",
                "model": "test-model",
                "output": [_text_message(text)],
            },
        },
    ]
    stream_body = "".join(
        f"data: {json.dumps({**event, 'sequence_number': index})}\n\n" for index, event in enumerate(events)
    ).encode()
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})
    )
    if sync:
        with OpenAI(api_key="fake-test-key", http_client=httpx2.Client(transport=transport)) as client:
            with client.responses.stream(model="test-model", input="test", text_format=Result) as stream:
                emitted = list(stream)
                final = stream.get_final_response()
    else:
        async with AsyncOpenAI(
            api_key="fake-test-key", http_client=httpx2.AsyncClient(transport=transport)
        ) as async_client:
            async with async_client.responses.stream(
                model="test-model", input="test", text_format=Result
            ) as async_stream:
                emitted = [event async for event in async_stream]
                final = await async_stream.get_final_response()

    done = [event for event in emitted if event.type == "response.output_text.done"]
    assert len(done) == 1
    assert done[0].parsed is None
    assert final.output_parsed is None
