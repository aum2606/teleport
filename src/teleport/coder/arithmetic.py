"""Integer arithmetic coder (Witten-Neal-Cleary 1987 style).

32-bit registers with E1/E2/E3 renormalization (the classic three-case
underflow/carry handling). This module is FROZEN once Phase 1's acceptance
criteria pass (see CLAUDE.md): predictors change, this does not.

Interface contract:
  - `encode(cum_low, cum_high, total)` / `decode(cum_low, cum_high, total)`
    consume a `(cumulative_low, cumulative_high, total)` triple describing the
    probability interval `[cum_low/total, cum_high/total)` of the symbol being
    coded. `total` must be < 2**(CODE_VALUE_BITS - 2) so renormalized ranges
    always have enough precision (we use total = 2**16, well within bounds).
  - The decoder's `get_cum_freq(total)` returns the scaled cumulative value
    the caller should look up in its cumulative-frequency table to find which
    symbol is being decoded; `decode(...)` is then called with that symbol's
    interval to advance the decoder in lockstep with the encoder.
"""

from __future__ import annotations

from teleport.coder.bitio import BitReader, BitWriter

CODE_VALUE_BITS = 32
TOP_VALUE = (1 << CODE_VALUE_BITS) - 1
FIRST_QTR = (TOP_VALUE >> 2) + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = TOP_VALUE
        self._pending_bits = 0
        self._writer = BitWriter()

    def _emit_bit(self, bit: int) -> None:
        self._writer.write_bit(bit)
        opposite = 1 - bit
        for _ in range(self._pending_bits):
            self._writer.write_bit(opposite)
        self._pending_bits = 0

    def encode(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total

        while True:
            if self.high < HALF:
                self._emit_bit(0)
            elif self.low >= HALF:
                self._emit_bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self._pending_bits += 1
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self._pending_bits += 1
        if self.low < FIRST_QTR:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        return self._writer.getvalue()


class ArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self._reader = BitReader(data)
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        for _ in range(CODE_VALUE_BITS):
            self.value = (self.value << 1) | self._reader.read_bit()

    def get_cum_freq(self, total: int) -> int:
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def decode(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.value -= FIRST_QTR
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._reader.read_bit()
