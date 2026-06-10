"""Minimal bit-level I/O used by the arithmetic coder."""

from __future__ import annotations


class BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def getvalue(self) -> bytes:
        if self._nbits > 0:
            self._cur <<= 8 - self._nbits
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0
        return bytes(self._bytes)


class BitReader:
    """Reads bits MSB-first; reads past the end of the buffer return 0."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0
        byte = self._data[self._byte_pos]
        bit = (byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit
