"""
budget.py — CC Mirror 成本预算控制器

职责：
  - 记录每次 LLM 调用到 llm_calls 表（stage, model, tokens, cost, duration）
  - 根据已消耗比例返回当前策略（full / p0_only / stop）
  - 每次查询策略时从 DB 重新计算，支持多进程场景

策略边界（已消耗 / 总预算）：
  < 50%  → "full"    正常运行
  50-80% → "p0_only" 只跑 P0（最高优先级）任务
  > 80%  → "stop"    停止新 LLM 调用
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# BudgetController
# ---------------------------------------------------------------------------

class BudgetController:
    """
    LLM 调用成本控制器。

    支持：
    - 写入 llm_calls 表（record_call）
    - 实时查询策略（get_strategy，每次从 DB 计算）
    - 查询已消耗 / 剩余预算（spent_usd / remaining_usd）
    """

    # 策略切换阈值（已消耗占总预算的比例）
    _THRESHOLD_P0_ONLY = 0.50   # > 50% → p0_only
    _THRESHOLD_STOP    = 0.80   # > 80% → stop

    def __init__(self, budget_usd: float, db: sqlite3.Connection) -> None:
        """
        Args:
            budget_usd: 总预算上限（如 20.0）
            db:         已初始化的 SQLite 连接，调用方负责关闭
        """
        if budget_usd <= 0:
            raise ValueError(f"budget_usd 必须为正数，收到 {budget_usd}")

        self._budget_usd = float(budget_usd)
        self._db = db

        # 内存缓存：避免高频 record_call 时每次都查 DB
        # get_strategy() 仍然走 DB，不用此缓存
        self._cached_spent: float = self._query_spent_from_db()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def record_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        duration_ms: int = 0,
    ) -> None:
        """
        记录一次 LLM 调用到 llm_calls 表，并更新内存缓存。

        Args:
            stage:         "L2" | "L3" | "L4"
            model:         模型名，如 "claude-sonnet-4-6"
            input_tokens:  输入 token 数
            output_tokens: 输出 token 数
            cost_usd:      本次调用花费（美元）
            duration_ms:   调用耗时（毫秒），默认 0
        """
        # 用 prompt 内容的粗略哈希（拼接参数）做去重标识
        raw = f"{stage}:{model}:{input_tokens}:{output_tokens}:{cost_usd}"
        prompt_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            self._db.execute(
                """
                INSERT INTO llm_calls (
                    stage, model, prompt_hash,
                    input_tokens, output_tokens,
                    cost_usd, timestamp, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stage, model, prompt_hash,
                 input_tokens, output_tokens,
                 cost_usd, timestamp, duration_ms),
            )
            self._db.commit()

            # 同步更新内存缓存
            self._cached_spent += float(cost_usd)

        except Exception as e:
            print(f"[budget] record_call 写入失败: {e}", file=sys.stderr)

    def get_strategy(self) -> str:
        """
        返回当前预算策略，每次从 DB 重新计算（支持多进程）。

        Returns:
            "full"    — 正常运行（已消耗 < 50% 预算）
            "p0_only" — 只运行 P0 优先级任务（50-80%）
            "stop"    — 停止新 LLM 调用（> 80%）
        """
        spent = self._query_spent_from_db()
        ratio = spent / self._budget_usd

        if ratio > self._THRESHOLD_STOP:
            return "stop"
        if ratio > self._THRESHOLD_P0_ONLY:
            return "p0_only"
        return "full"

    @property
    def spent_usd(self) -> float:
        """已消耗成本（从内存缓存读取，快速路径）。"""
        return self._cached_spent

    @property
    def remaining_usd(self) -> float:
        """剩余预算（总预算 - 已消耗）。"""
        return max(0.0, self._budget_usd - self._cached_spent)

    @property
    def budget_usd(self) -> float:
        """总预算（只读）。"""
        return self._budget_usd

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _query_spent_from_db(self) -> float:
        """从 llm_calls 表聚合 cost_usd 总和。"""
        try:
            cur = self._db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls"
            )
            row = cur.fetchone()
            return float(row[0]) if row else 0.0
        except Exception as e:
            print(f"[budget] 查询已消耗成本失败: {e}", file=sys.stderr)
            return 0.0
