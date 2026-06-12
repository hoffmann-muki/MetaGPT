#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of ollama api

import json
from typing import Any, AsyncGenerator, Tuple

import pytest

from metagpt.provider.ollama_api import OllamaLLM, OpenAIResponse
from tests.metagpt.provider.mock_llm_config import mock_llm_config
from tests.metagpt.provider.req_resp_const import (
    llm_general_chat_funcs_test,
    messages,
    prompt,
    resp_cont_tmpl,
)

resp_cont = resp_cont_tmpl.format(name="ollama")
default_resp = {"message": {"role": "assistant", "content": resp_cont}}


async def mock_ollama_arequest(self, stream: bool = False, **kwargs) -> Tuple[Any, Any, bool]:
    if stream:

        async def async_event_generator() -> AsyncGenerator[Any, None]:
            events = [
                b'{"message": {"role": "assistant", "content": "I\'m ollama"}, "done": false}',
                b'{"prompt_eval_count": 20, "eval_count": 20, "done": true}',
            ]
            for event in events:
                yield OpenAIResponse(event, {})

        return async_event_generator(), None, None
    else:
        raw_default_resp = default_resp.copy()
        raw_default_resp.update({"prompt_eval_count": 20, "eval_count": 20})
        return OpenAIResponse(json.dumps(raw_default_resp).encode(), {}), None, None


@pytest.mark.asyncio
async def test_gemini_acompletion(mocker):
    mocker.patch("metagpt.provider.general_api_requestor.GeneralAPIRequestor.arequest", mock_ollama_arequest)

    ollama_llm = OllamaLLM(mock_llm_config)

    resp = await ollama_llm.acompletion(messages)
    assert resp["message"]["content"] == default_resp["message"]["content"]

    resp = await ollama_llm.aask(prompt, stream=False)
    assert resp == resp_cont

    await llm_general_chat_funcs_test(ollama_llm, prompt, messages, resp_cont)


def test_ollama_chat_preserves_system_and_context_messages():
    ollama_llm = OllamaLLM(mock_llm_config)
    payload = ollama_llm.ollama_message.apply(
        [
            {"role": "system", "content": "Return command JSON only."},
            {"role": "user", "content": "Build a 2048 game."},
            {"role": "assistant", "content": "```json\n[]\n```"},
            {"role": "user", "content": "Continue."},
        ]
    )

    assert payload["messages"] == [
        {"role": "system", "content": "Return command JSON only."},
        {"role": "user", "content": "Build a 2048 game."},
        {"role": "assistant", "content": "```json\n[]\n```"},
        {"role": "user", "content": "Continue."},
    ]


def test_ollama_chat_converts_multimodal_messages_to_ollama_shape():
    ollama_llm = OllamaLLM(mock_llm_config)
    payload = ollama_llm.ollama_message.apply(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
    )

    assert payload["messages"] == [
        {"role": "user", "content": "Describe this.", "images": ["abc123"]}
    ]
