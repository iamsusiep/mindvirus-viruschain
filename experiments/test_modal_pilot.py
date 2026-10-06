"""Check Modal transport and upstream hard-reset behavior without GPU calls."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# The repo's root CLI, virus_chain.py, otherwise shadows its installed package
# when pytest is launched with python -m from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modal_pilot import ModalBackend

from virus_chain.core.agent import AgentState
from virus_chain.core.model import ModelResponse
from virus_chain.core.snapshot import save_snapshot
from virus_chain.eval import harness, hop


def process(stdout="", returncode=0):
    return SimpleNamespace(
        stdout=SimpleNamespace(
            read=SimpleNamespace(aio=AsyncMock(return_value=stdout))
        ),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=""))),
        wait=SimpleNamespace(aio=AsyncMock()),
        returncode=returncode,
    )


@pytest.mark.asyncio
async def test_files_are_written_in_sandbox_with_literal_arguments(tmp_path):
    exec_call = AsyncMock(return_value=process())
    backend = ModalBackend(None, tmp_path)
    backend.containers["agent_0"] = SimpleNamespace(exec=SimpleNamespace(aio=exec_call))
    content = "literal $(touch /tmp/should-not-exist) `whoami`\n"
    ref = SimpleNamespace(service_name="agent_0")
    await backend.container_write_file(ref, "/workspace/memory.md", content)
    args, kwargs = exec_call.call_args
    assert args[0] == "python3"
    assert args[-2:] == ("/workspace/memory.md", content)
    assert kwargs["workdir"] == "/workspace"


@pytest.mark.asyncio
async def test_truncation_is_recorded_as_failure(tmp_path):
    response = {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "incomplete"},
            }
        ],
    }
    exec_call = AsyncMock(return_value=process(json.dumps(response)))
    backend = ModalBackend(
        SimpleNamespace(exec=SimpleNamespace(aio=exec_call)), tmp_path
    )
    with pytest.raises(RuntimeError, match="Truncated"):
        await backend.generate("test", [{"role": "user", "content": "hello"}])
    record = json.loads((tmp_path / "requests.jsonl").read_text())
    assert record["response"]["stop_reason"] == "length"
    assert record["error"] and backend.errors


@pytest.mark.asyncio
async def test_call_cap_does_not_contact_server(tmp_path):
    exec_call = AsyncMock()
    backend = ModalBackend(
        SimpleNamespace(exec=SimpleNamespace(aio=exec_call)), tmp_path
    )
    backend.calls = 168
    with pytest.raises(RuntimeError, match="168-call"):
        await backend.generate("test", [])
    exec_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_upstream_hard_reset_preserves_files_without_previous_chat(
    tmp_path, monkeypatch
):
    backend = ModalBackend(None, tmp_path)
    files = {"agent_0": {}, "agent_1": {}}

    async def write(ref, path, content):
        files[ref.service_name][path] = content

    async def read(ref, path):
        return files[ref.service_name][path]

    backend.container_write_file = write
    backend.container_read_file = read
    seen = []

    async def generate(model, messages, **kwargs):
        seen.append(json.loads(json.dumps(messages)))
        return ModelResponse(content="Continue."), {
            "role": "assistant",
            "content": "Continue.",
        }

    backend.generate = generate
    backend.judge_call = AsyncMock(
        return_value="INFECTION: none\nSPREAD: none\nSUMMARY: test"
    )
    monkeypatch.setattr(hop, "create_snapshot_marker", AsyncMock())
    monkeypatch.setattr(
        hop, "_capture_workspace_files", AsyncMock(return_value={"Cleo": {}})
    )
    monkeypatch.setattr(hop, "_capture_extra_files", AsyncMock(return_value={}))
    monkeypatch.setattr(hop, "capture_deleted_home_files", AsyncMock(return_value=[]))
    previous = AgentState("Briar", "OLD SYSTEM PAYLOAD")
    previous.add_user_message("OLD CONVERSATION MUST NOT SURVIVE")
    previous.turns_taken = 10
    snapshot = save_snapshot(
        previous,
        {
            "SOUL.md": "Briar's saved identity",
            "memory.md": "Briar's own saved note",
            "project.txt": "Briar's project",
        },
    )
    config = {
        "model": "test",
        "judge_model": "test",
        "hard_mode": "hard",
        "turns_per_hop": 1,
        "max_tool_loops": 1,
        "first_prompt": "agent_network",
        "initial_soul": "Clean identity",
    }
    original_generate = harness.model_generate
    with ExitStack() as stack:
        backend.install(stack)
        result = await hop.run_single_hop(
            "ORIGINAL SEED MUST NOT SURVIVE",
            config,
            2,
            Path("test.yaml"),
            "test",
            snapshot_data=snapshot,
            agent_names=["Briar", "Cleo"],
        )
    assert result.error is None
    assert len(seen[0]) == 2
    assert "Briar's saved identity" in seen[0][0]["content"]
    assert "Cleo" in seen[0][1]["content"]
    assert "OLD" not in json.dumps(seen)
    assert "ORIGINAL SEED" not in json.dumps(seen)
    assert files["agent_0"]["/workspace/memory.md"] == "Briar's own saved note"
    assert files["agent_0"]["/workspace/project.txt"] == "Briar's project"
    assert files["agent_1"]["/workspace/SOUL.md"] == "Clean identity"
    assert harness.model_generate is original_generate
