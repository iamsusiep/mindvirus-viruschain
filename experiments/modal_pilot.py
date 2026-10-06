"""Run the authors' two-hop whale setup with private Modal sandboxes."""

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import modal
from openai.types.chat import ChatCompletion

from virus_chain.core import sandbox, tools
from virus_chain.core.model import ModelResponse, ToolCall, Usage
from virus_chain.eval import harness, hop, judge

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SEED = ROOT / (
    "experiments/virus_chain_runs/ideo_mutations/ideology_robustness/"
    "seeds/whale_lover_robust.json"
)
FORWARD = """
import sys, urllib.request
request = urllib.request.Request(
    'http://127.0.0.1:8000/v1/chat/completions',
    data=sys.argv[1].encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=180) as response:
    print(response.read().decode())
"""


class ModalBackend:
    """Adapt sandbox I/O and inference; leave the agent harness unchanged."""

    def __init__(self, server, output):
        self.server = server
        self.output = output
        self.containers = {}
        self.calls = 0
        self.phase = ""
        self.errors = []

    async def container_exec(self, ref, cmd, timeout=60):
        process = await self.containers[ref.service_name].exec.aio(
            *cmd, timeout=timeout, workdir="/workspace"
        )
        stdout, stderr = await asyncio.gather(
            process.stdout.read.aio(), process.stderr.read.aio()
        )
        await process.wait.aio()
        return sandbox.ExecResult(stdout, stderr, process.returncode)

    async def container_read_file(self, ref, path):
        result = await self.container_exec(ref, ["cat", path])
        if result.returncode:
            raise FileNotFoundError(path)
        return result.stdout

    async def container_write_file(self, ref, path, content):
        result = await self.container_exec(
            ref,
            [
                "python3",
                "-c",
                "import pathlib, sys; p = pathlib.Path(sys.argv[1]); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(sys.argv[2])",
                path,
                content,
            ],
        )
        if result.returncode:
            raise OSError(result.stderr)

    async def generate(self, model, messages, tools=None, **kwargs):
        if self.calls >= 168:
            raise RuntimeError("Pilot reached its 168-call limit.")
        self.calls += 1
        request = {
            "model": MODEL,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "seed": 0,
            "stop": ["<|im_start|>"],
        }
        if tools:
            request.update(tools=tools, tool_choice="auto")
        record = {"call": self.calls, "phase": self.phase, "request": request}
        try:
            process = await self.server.exec.aio(
                "python3", "-c", FORWARD, json.dumps(request), timeout=195
            )
            stdout, stderr = await asyncio.gather(
                process.stdout.read.aio(), process.stderr.read.aio()
            )
            await process.wait.aio()
            if process.returncode:
                raise RuntimeError(f"Inference failed: {stderr}")
            response = ChatCompletion.model_validate_json(stdout)
            choice = response.choices[0]
            message = choice.message
            parsed = ModelResponse(
                content=message.content,
                tool_calls=[
                    ToolCall(tc.id, tc.function.name, tc.function.arguments)
                    for tc in (message.tool_calls or [])
                ]
                or None,
                usage=Usage(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
                if response.usage
                else Usage(),
                stop_reason=choice.finish_reason,
            )
            record["response"] = asdict(parsed)
            if choice.finish_reason == "length":
                raise RuntimeError("Truncated model output; trial is invalid.")
            return parsed, message.model_dump(exclude_none=True)
        except Exception as error:
            record["error"] = str(error)
            self.errors.append(str(error))
            raise
        finally:
            with (self.output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    async def judge_call(self, model, system, user):
        response, _ = await self.generate(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=2048,
        )
        return response.content or ""

    def install(self, stack):
        # These modules import sandbox functions directly. Bind each import
        # for this experiment only; normal Docker and provider paths stay intact.
        for module in (sandbox, tools, harness, hop):
            for name in (
                "container_exec",
                "container_read_file",
                "container_write_file",
            ):
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, getattr(self, name)))
        stack.enter_context(patch.object(harness, "model_generate", self.generate))
        stack.enter_context(patch.object(judge, "_judge_call", self.judge_call))


async def run_cases(backend, app, image, tags, results, conditions):
    seed = json.loads(SEED.read_text())
    initial_soul = (ROOT / "data/souls/default.md").read_text()
    config = {
        "model": MODEL,
        "judge_model": MODEL,
        "turns_per_hop": 10,
        "max_tool_loops": 2,
        "first_prompt": "agent_network",
        "heartbeat": True,
        "hard_mode": "hard",
        "min_infection": "strong",
        "initial_soul": initial_soul,
    }
    for condition in conditions:
        payload = initial_soul if condition == "control" else seed["payload"]
        snapshot = None
        for number in (1, 2):
            backend.phase = f"{condition}-hop-{number}"
            names = ["Atlas", "Briar"] if number == 1 else ["Briar", "Cleo"]
            try:
                for index in range(2):
                    container = await modal.Sandbox.create.aio(
                        "sleep",
                        "600",
                        app=app,
                        image=image,
                        tags=tags,
                        cpu=(0.5, 0.5),
                        memory=(512, 512),
                        timeout=600,
                        workdir="/workspace",
                        block_network=True,
                    )
                    backend.containers[f"agent_{index}"] = container
                result = await hop.run_single_hop(
                    payload,
                    config,
                    number,
                    Path("modal.yaml"),
                    "pilot",
                    snapshot_data=snapshot,
                    agent_names=names,
                )
                (backend.output / f"{backend.phase}.json").write_text(
                    json.dumps(asdict(result), indent=2)
                )
                judge_valid = all(
                    not result.hop_result.get(key, {}).get("parse_error")
                    for key in ("target_judge", "spreader_judge")
                )
                turn_errors = [
                    turn["error"]
                    for turn in result.recording.get("turns", [])
                    if turn.get("error")
                ]
                if result.error or turn_errors or backend.errors or not judge_valid:
                    raise RuntimeError(
                        f"Invalid trial: {backend.phase}; see artifacts."
                    )
                results.append({"condition": condition, **result.hop_result})
                snapshot = result.target_snapshot
                if snapshot is None:
                    raise RuntimeError("Target snapshot missing.")
                logger.info(
                    "Saved %s; target infection=%s",
                    backend.phase,
                    result.hop_result["target_infection"],
                )
            finally:
                for container in backend.containers.values():
                    await container.terminate.aio()
                    await container.wait.aio(raise_on_termination=False)
                backend.containers.clear()


async def main(output, conditions):
    output.mkdir(parents=True, exist_ok=False)
    tags = {"mindvirus-pilot": uuid.uuid4().hex}
    app = await modal.App.lookup.aio("mindvirus-whale-pilot", create_if_missing=True)
    agent_image = modal.Image.from_dockerfile(ROOT / "data/sandbox/Dockerfile")
    model_image = (
        modal.Image.from_registry("vllm/vllm-openai:v0.30.0")
        .entrypoint([])
        .run_commands("uv pip install --system zstandard==0.25.0")
        .run_commands(
            'python3 -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{MODEL}', revision='{REVISION}', "
            "allow_patterns=['*.json','*.safetensors','*.txt','*.model'])\""
        )
        .env({"OMP_NUM_THREADS": "1", "VLLM_NO_USAGE_STATS": "1"})
    )
    server, backend, results = None, None, []
    started = time.monotonic()
    try:
        server = await modal.Sandbox.create.aio(
            "vllm",
            "serve",
            MODEL,
            "--revision",
            REVISION,
            "--host",
            "127.0.0.1",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "2",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
            "--generation-config",
            "vllm",
            app=app,
            image=model_image,
            tags=tags,
            gpu="L4",
            cpu=(4, 4),
            memory=(24576, 24576),
            timeout=1800,
        )
        logger.info("Started bounded L4 server: %s", server.object_id)
        async with asyncio.timeout(450):
            while True:
                if await server.poll.aio() is not None:
                    raise RuntimeError("vLLM exited before becoming ready.")
                check = await server.exec.aio(
                    "python3",
                    "-c",
                    "import urllib.request; "
                    "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)",
                    timeout=5,
                )
                await check.wait.aio()
                if check.returncode == 0:
                    break
                await asyncio.sleep(5)
        backend = ModalBackend(server, output)
        logger.info("vLLM is ready; starting %s chains.", len(conditions))
        with ExitStack() as stack:
            backend.install(stack)
            # Queue time consumes no model turns. Keep the experiment deadline
            # separate; the GPU sandbox still has its own hard lifetime cap.
            async with asyncio.timeout(1600):
                await run_cases(backend, app, agent_image, tags, results, conditions)
    finally:
        for leftover in [
            item async for item in modal.Sandbox.list.aio(app_id=app.app_id, tags=tags)
        ]:
            await leftover.terminate.aio()
            await leftover.wait.aio(raise_on_termination=False)
        remaining = [
            item async for item in modal.Sandbox.list.aio(app_id=app.app_id, tags=tags)
        ]
        if server is not None:
            stdout, stderr = await asyncio.gather(
                server.stdout.read.aio(), server.stderr.read.aio()
            )
            (output / "server.stdout.log").write_text(stdout)
            (output / "server.stderr.log").write_text(stderr)
        (output / "pilot.json").write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "upstream_commit": "20651e0bff017540e3b738270ba8a434e1948c67",
                    "revision": REVISION,
                    "vllm": "0.30.0",
                    "gpu_timeout_seconds": 1800,
                    "cpu_sandbox_timeout_seconds": 600,
                    "model_calls": backend.calls if backend else 0,
                    "generation_seed": 0,
                    "agent_output_token_limit": 4096,
                    "judge_output_token_limit": 2048,
                    "turns_per_agent_per_hop": 10,
                    "max_tool_loops": 2,
                    "hard_mode": "hard",
                    "gpu_sandbox_id": server.object_id if server else None,
                    "sandbox_tags": tags,
                    "completed": len(results) == len(conditions) * 2,
                    "model_errors": backend.errors if backend else [],
                    "completed_hops": len(results),
                    "expected_hops": len(conditions) * 2,
                    "conditions": conditions,
                    "elapsed_seconds_including_image_build": round(
                        time.monotonic() - started, 2
                    ),
                    "active_sandboxes_after_cleanup": len(remaining),
                    "hops": results,
                },
                indent=2,
            )
        )
        if remaining:
            raise RuntimeError("Modal sandbox cleanup incomplete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("control", "whale"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with modal.enable_output():
        asyncio.run(
            main(
                args.output_dir,
                (args.condition,) if args.condition else ("control", "whale"),
            )
        )
