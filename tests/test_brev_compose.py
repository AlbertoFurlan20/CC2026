from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.brev.yml"


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(
            ["docker", "compose", "version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def _render_compose(env_file: str, token_file: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["HF_TOKEN_FILE"] = str(token_file)
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            env_file,
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="Docker Compose is required to render the Brev deployment",
)
@pytest.mark.parametrize(
    (
        "env_file",
        "optimizer_gpu",
        "utilization",
        "sequences",
        "batched_tokens",
    ),
    [
        (".env.brev.example", "0", "0.60", "8", "8192"),
        (".env.brev.dedicated.example", "1", "0.85", "32", "16384"),
    ],
)
def test_brev_presets_render_expected_vllm_limits(
    tmp_path: Path,
    env_file: str,
    optimizer_gpu: str,
    utilization: str,
    sequences: str,
    batched_tokens: str,
) -> None:
    token_file = tmp_path / "huggingface-token"
    token_file.write_text("test-token\n", encoding="utf-8")
    rendered = _render_compose(env_file, token_file)

    vllm = rendered["services"]["vllm"]
    optimizer = rendered["services"]["optimizer"]
    command = vllm["command"]

    assert _command_value(command, "--gpu-memory-utilization") == utilization
    assert _command_value(command, "--max-model-len") == "8192"
    assert _command_value(command, "--max-num-seqs") == sequences
    assert _command_value(command, "--max-num-batched-tokens") == batched_tokens
    assert "--enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command

    vllm_devices = vllm["deploy"]["resources"]["reservations"]["devices"]
    optimizer_devices = optimizer["deploy"]["resources"]["reservations"][
        "devices"
    ]
    assert vllm_devices[0]["device_ids"] == ["0"]
    assert optimizer_devices[0]["device_ids"] == [optimizer_gpu]

    assert vllm["ports"][0]["host_ip"] == "127.0.0.1"
    assert "HF_TOKEN" not in vllm["environment"]
    assert "NVIDIA_VISIBLE_DEVICES" not in vllm["environment"]
    assert "NVIDIA_VISIBLE_DEVICES" not in optimizer["environment"]
    assert rendered["secrets"]["hf_token"]["file"] == str(token_file)
