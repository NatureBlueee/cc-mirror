"""
LLM 后端。优先用 claude CLI，没有则 fallback 到 anthropic SDK。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Optional


def _claude_available() -> bool:
    return shutil.which("claude") is not None


async def call_llm(
    prompt: str,
    system: Optional[str] = None,
    model: str = "sonnet",
) -> tuple[str, float]:
    """调用 LLM。返回 (response_text, cost_usd)。"""
    if _claude_available():
        return await _call_via_cli(prompt, system, model)
    return await _call_via_api(prompt, system, model)


async def _call_via_cli(prompt: str, system: Optional[str], model: str) -> tuple[str, float]:
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
    ]
    if system:
        cmd += ["--system-prompt", system]

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # 允许在 Claude Code 内嵌套运行

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI 失败: {stderr.decode()[:500]}")

    data = json.loads(stdout.decode())
    if data.get("is_error"):
        raise RuntimeError(f"claude 返回错误: {data.get('result', '')}")

    return data.get("result", ""), float(data.get("cost_usd", 0.0))


async def _call_via_api(prompt: str, system: Optional[str], model: str) -> tuple[str, float]:
    """Fallback：直接调 anthropic SDK（需要 ANTHROPIC_API_KEY）。"""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "找不到 claude CLI 也没有 anthropic 包。"
            "请安装 Claude Code 或运行: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TOWOW_ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有 API key。请安装 Claude Code 或设置 ANTHROPIC_API_KEY 环境变量。"
        )

    _model_map = {
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
        "haiku": "claude-haiku-4-5-20251001",
    }
    full_model = _model_map.get(model, model)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    kwargs: dict = dict(
        model=full_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    resp = await client.messages.create(**kwargs)
    text = resp.content[0].text
    # 估算成本（Sonnet: in $3/M, out $15/M；Opus: in $15/M, out $75/M）
    if "opus" in full_model:
        cost = (resp.usage.input_tokens * 15 + resp.usage.output_tokens * 75) / 1_000_000
    else:
        cost = (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15) / 1_000_000
    return text, cost
