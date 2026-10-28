from __future__ import annotations

import bisect
from typing import Sequence


class PageMap:
    """char 偏移 → page_no 映射（读自 S1 的 pages.json）。

    pages.json 覆盖率约 96%，存在间隙；间隙位置落到上一页而不是 0，
    避免 chunk 因页码映射失败丢页信息。
    """

    def __init__(self, entries: Sequence[dict]) -> None:
        # 按 char_start 排序后二分查找，年报 150+ 页 / 2000+ 区间
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._pages: list[int] = []
        for e in sorted(entries, key=lambda x: x["char_start"]):
            self._starts.append(e["char_start"])
            self._ends.append(e["char_end"])
            self._pages.append(e["page_no"])

    def page_at(self, char_offset: int) -> int:
        # 找到 char_start <= offset 的最后一个区间
        idx = bisect.bisect_right(self._starts, char_offset) - 1
        if idx < 0:
            return 0
        if char_offset <= self._ends[idx]:
            return self._pages[idx]
        # 落在间隙里 → 上一页（间隙属于上一页内容的空白）
        return self._pages[idx]

    def page_span(self, char_start: int, char_end: int) -> tuple[int, int]:
        return self.page_at(char_start), self.page_at(char_end)
