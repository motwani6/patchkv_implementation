# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-free planning for manual, resident-KV suffix repair."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class KVTokenCopy:
    """An aligned token interval copied between logical page tables."""

    source_start: int
    destination_start: int
    count: int


@dataclass(frozen=True)
class PatchKVPlan:
    """Immutable plan in absolute new-context token and logical block coordinates."""

    block_size: int
    suffix_start: int
    new_length: int
    repair_blocks: tuple[int, ...]
    reuse_blocks: tuple[int, ...]
    private_blocks: tuple[int, ...]
    repair_ranges: tuple[tuple[int, int], ...]
    copies: tuple[KVTokenCopy, ...]


def plan_suffix_repair(
    original_length: int,
    start: int,
    end: int,
    replacement_length: int,
    block_size: int,
    repair_suffix_blocks: Sequence[int],
) -> PatchKVPlan:
    """Plan manual repair without reading KV or allocating physical pages.

    Args:
        original_length: Number of original tokens.
        start: Inclusive edit start in the original context.
        end: Exclusive edit end in the original context.
        replacement_length: Number of replacement tokens.
        block_size: Tokens per logical cache block.
        repair_suffix_blocks: Absolute logical block indices in the NEW context.
            Each index must intersect the retained suffix. Duplicates are ignored.
            Repair is clipped to suffix tokens; replacement tokens are mandatory.

    Returns:
        Page ownership, copy mappings, and sorted contiguous repair intervals.

    Raises:
        ValueError: If lengths, edit bounds, or selected blocks are invalid.

    """
    integers = (original_length, start, end, replacement_length, block_size)
    if any(type(value) is not int for value in integers):
        raise ValueError("Lengths, edit bounds, and block size must be integers.")
    if original_length < 0 or replacement_length < 0 or block_size <= 0:
        raise ValueError("Lengths must be nonnegative and block size positive.")
    if not 0 <= start <= end <= original_length:
        raise ValueError("Expected 0 <= start <= end <= original_length.")

    suffix_start = start + replacement_length
    new_length = suffix_start + original_length - end
    num_blocks = (new_length + block_size - 1) // block_size
    suffix_blocks = set(range(suffix_start // block_size, num_blocks))
    if suffix_start == new_length:
        suffix_blocks.clear()
    selected = set()
    for block in repair_suffix_blocks:
        if type(block) is not int or block not in suffix_blocks:
            raise ValueError(
                "Repair blocks must be integer new-context block indices "
                "intersecting the retained suffix."
            )
        selected.add(block)

    ranges: list[tuple[int, int]] = []
    for block in sorted(selected):
        lo = max(suffix_start, block * block_size)
        hi = min(new_length, (block + 1) * block_size)
        if ranges and ranges[-1][1] == lo:
            ranges[-1] = (ranges[-1][0], hi)
        else:
            ranges.append((lo, hi))

    private = []
    copies: list[KVTokenCopy] = []

    def copy(source: int, destination: int, count: int) -> None:
        if count <= 0:
            return
        if copies:
            previous = copies[-1]
            if (
                previous.source_start + previous.count == source
                and previous.destination_start + previous.count == destination
            ):
                copies[-1] = KVTokenCopy(
                    previous.source_start,
                    previous.destination_start,
                    previous.count + count,
                )
                return
        copies.append(KVTokenCopy(source, destination, count))

    shift = suffix_start - end
    for block in range(num_blocks):
        lo, hi = block * block_size, min(new_length, (block + 1) * block_size)
        has_replacement = replacement_length > 0 and lo < suffix_start and hi > start
        shifted_suffix = shift != 0 and hi > suffix_start
        if not (has_replacement or shifted_suffix or block in selected):
            continue
        private.append(block)
        copy(lo, lo, min(hi, start) - lo)
        if block not in selected:
            suffix_lo = max(lo, suffix_start)
            copy(suffix_lo - shift, suffix_lo, hi - suffix_lo)

    return PatchKVPlan(
        block_size=block_size,
        suffix_start=suffix_start,
        new_length=new_length,
        repair_blocks=tuple(sorted(selected)),
        reuse_blocks=tuple(sorted(suffix_blocks - selected)),
        private_blocks=tuple(private),
        repair_ranges=tuple(ranges),
        copies=tuple(copies),
    )
