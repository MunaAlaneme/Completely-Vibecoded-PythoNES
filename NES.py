#!/usr/bin/env python3
"""
PyNES - a Nintendo Entertainment System emulator in Python, rendered with the
Arcade game engine.

Emulated hardware
  * MOS 6502 (2A03) CPU - all official opcodes + the common unofficial ones,
    cycle counting, page-cross penalties, NMI / IRQ / BRK
  * 2C02 PPU - loopy v/t/x/w scrolling, background + sprites (8x8 / 8x16),
    sprite-0 hit, sprite overflow, NMI, scanline-accurate timing
  * APU - 2 pulse (envelope + sweep), triangle, noise, DMC, frame counter + IRQ
  * Controllers (strobe / shift register protocol), OAM DMA
  * iNES loader, battery saves (.sav), mappers 0,1,2,3,4,7,66
    (NROM, MMC1, UxROM, CNROM, MMC3 + scanline IRQ, AxROM, GxROM)
  * Save states (F5 / F7)

Front-end
  * menu bar (File / Emulation / Video / Audio / Help) with mouse + the usual hotkeys
  * open ROMs at runtime (File > Open ROM, Ctrl+O), window sizes, fullscreen, filters

Speed
  * the PPU only *records* each visible scanline while the CPU runs; the whole
    frame (background + sprites) is then rendered in one batch of numpy operations.
    Output is pixel-identical to a scanline-by-scanline renderer; the queue is
    flushed early whenever the game touches VRAM / palette / OAM mid-frame
  * table-driven CPU dispatch with the fetch/execute loop inlined
  * frame skipping when the host can't keep up (sprite-0 hit still computed)

Audio / timing
  * low-latency output through sounddevice (PortAudio / WASAPI) when installed,
    pyglet as fallback
  * the emulation speed is slaved (+-2.5 %) to the audio clock, so audio latency
    stays at ~60 ms and video never drifts away from the sound
  * vsync by default; switched off automatically if the machine can't hold 60 FPS

Requirements:  pip install arcade numpy
Recommended:   pip install sounddevice      (low-latency audio)
Optional:      pip install pillow           (F12 screenshots)
Run:           python NES.py [game.nes]
Benchmark:     python NES.py game.nes --bench 300
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from collections import deque
import tkinter

import numpy as np

try:  # the core works without arcade (used by --bench)
    import arcade
    from arcade.gl import geometry as gl_geometry
except Exception:  # pragma: no cover
    arcade = None
    gl_geometry = None

CPU_HZ = 1789773.0
FRAME_RATE = 60.0988

# --------------------------------------------------------------------------
# Palette + pattern tables
# --------------------------------------------------------------------------
_PAL = [
    0x666666, 0x002A88, 0x1412A7, 0x3B00A4, 0x5C007E, 0x6E0040, 0x6C0600, 0x561D00,
    0x333500, 0x0B4800, 0x005200, 0x004F08, 0x00404D, 0x000000, 0x000000, 0x000000,
    0xADADAD, 0x155FD9, 0x4240FF, 0x7527FE, 0xA01ACC, 0xB71E7B, 0xB53120, 0x994E00,
    0x6B6D00, 0x388700, 0x0C9300, 0x008F32, 0x007C8D, 0x000000, 0x000000, 0x000000,
    0xFFFEFF, 0x64B0FF, 0x9290FF, 0xC676FF, 0xF36AFF, 0xFE6ECC, 0xFE8170, 0xEA9E22,
    0xBCBE00, 0x88D800, 0x5CE430, 0x45E082, 0x48CDDE, 0x4F4F4F, 0x000000, 0x000000,
    0xFFFEFF, 0xC0DFFF, 0xD3D2FF, 0xE8C8FF, 0xFBC2FF, 0xFEC4EA, 0xFECCC5, 0xF7D8A5,
    0xE4E594, 0xCFEF96, 0xBDF4AB, 0xB3F3CC, 0xB5EBF2, 0xB8B8B8, 0x000000, 0x000000,
]
PALETTE_RGBA = np.zeros((64, 4), dtype=np.uint8)
for _i, _c in enumerate(_PAL):
    PALETTE_RGBA[_i] = ((_c >> 16) & 255, (_c >> 8) & 255, _c & 255, 255)

# PIX[(hi << 8) | lo] -> 8 pixels (0..3), left to right
_idx = np.arange(65536, dtype=np.uint32)
PIX = np.zeros((65536, 8), dtype=np.uint8)
for _j in range(8):
    _sh = 7 - _j
    PIX[:, _j] = ((_idx >> _sh) & 1) | (((_idx >> (8 + _sh)) & 1) << 1)
del _idx

_ROW_CACHE: dict = {}


def pattern_row(lo: int, hi: int):
    key = (hi << 8) | lo
    r = _ROW_CACHE.get(key)
    if r is None:
        r = tuple(((lo >> (7 - j)) & 1) | (((hi >> (7 - j)) & 1) << 1) for j in range(8))
        _ROW_CACHE[key] = r
    return r


ZERO256 = np.zeros(256, dtype=np.uint8)

# --------------------------------------------------------------------------
# Cartridge + mappers
# --------------------------------------------------------------------------


class Cartridge:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"NES\x1a":
            raise ValueError("Not an iNES ROM (missing NES\\x1a header)")
        prg_n, chr_n, f6, f7 = data[4], data[5], data[6], data[7]
        self.mapper_id = (f7 & 0xF0) | (f6 >> 4)
        self.battery = bool(f6 & 2)
        self.four_screen = bool(f6 & 8)
        self.vertical = bool(f6 & 1)
        off = 16 + (512 if f6 & 4 else 0)
        self.prg = bytes(data[off:off + prg_n * 16384])
        off += prg_n * 16384
        if chr_n:
            self.chr = bytearray(data[off:off + chr_n * 8192])
            self.chr_ram = False
        else:
            self.chr = bytearray(8192)
            self.chr_ram = True
        self.prg_ram = bytearray(8192)


class Mapper:
    def __init__(self, cart: Cartridge):
        self.cart = cart
        self.prg = cart.prg
        self.chr = cart.chr
        self.prg_ram = cart.prg_ram
        self.n_prg8 = max(1, len(self.prg) // 8192)
        self.n_chr1 = max(1, len(self.chr) // 1024)
        self.prg_off = [0, 0, 0, 0]
        self.chr_off = [0] * 8
        self.nt_off = [0, 0, 0, 0]
        self.irq = False
        self.set_mirroring("v" if cart.vertical else "h")
        if cart.four_screen:
            self.nt_off[:] = [0, 0x400, 0x800, 0xC00]

    # helpers -------------------------------------------------------------
    def set_mirroring(self, mode):
        self.nt_off[:] = {
            "h": [0, 0, 0x400, 0x400],
            "v": [0, 0x400, 0, 0x400],
            "s0": [0, 0, 0, 0],
            "s1": [0x400] * 4,
        }[mode]

    def map_prg8(self, slot, bank):
        self.prg_off[slot] = (bank % self.n_prg8) * 8192

    def map_chr1(self, slot, bank):
        self.chr_off[slot] = (bank % self.n_chr1) * 1024

    def map_prg32(self, bank):
        for i in range(4):
            self.map_prg8(i, bank * 4 + i)

    def map_chr8(self, bank):
        for i in range(8):
            self.map_chr1(i, bank * 8 + i)

    # interface -----------------------------------------------------------
    def write(self, addr, val):
        pass

    def scanline(self):
        pass

    def get_state(self):
        st = {k: (list(v) if isinstance(v, list) else v)
              for k, v in self.__dict__.items() if isinstance(v, (int, bool, list))}
        st["_chr"] = bytes(self.chr) if self.cart.chr_ram else None
        st["_ram"] = bytes(self.prg_ram)
        return st

    def set_state(self, st):
        for k, v in st.items():
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                getattr(self, k)[:] = v
            else:
                setattr(self, k, v)
        if st["_chr"] is not None:
            self.chr[:] = st["_chr"]
        self.prg_ram[:] = st["_ram"]


class Mapper0(Mapper):  # NROM
    def __init__(self, cart):
        super().__init__(cart)
        for i in range(4):
            self.map_prg8(i, i)  # modulo mirrors 16 KB ROMs
        self.map_chr8(0)


class Mapper1(Mapper):  # MMC1
    def __init__(self, cart):
        super().__init__(cart)
        self.shift = 0x10
        self.control = 0x0C
        self.chr0 = 0
        self.chr1 = 0
        self.prgb = 0
        self.update()

    def write(self, addr, val):
        if val & 0x80:
            self.shift = 0x10
            self.control |= 0x0C
            self.update()
            return
        complete = self.shift & 1
        self.shift = (self.shift >> 1) | ((val & 1) << 4)
        if complete:
            reg = (addr >> 13) & 3
            d = self.shift
            if reg == 0:
                self.control = d
            elif reg == 1:
                self.chr0 = d
            elif reg == 2:
                self.chr1 = d
            else:
                self.prgb = d & 0x0F
            self.shift = 0x10
            self.update()

    def update(self):
        c = self.control
        self.set_mirroring(("s0", "s1", "v", "h")[c & 3])
        mode = (c >> 2) & 3
        n = self.n_prg8
        if mode < 2:
            base = (self.prgb & 0x0E) * 2
            for i in range(4):
                self.map_prg8(i, base + i)
        elif mode == 2:
            self.map_prg8(0, 0)
            self.map_prg8(1, 1)
            self.map_prg8(2, self.prgb * 2)
            self.map_prg8(3, self.prgb * 2 + 1)
        else:
            self.map_prg8(0, self.prgb * 2)
            self.map_prg8(1, self.prgb * 2 + 1)
            self.map_prg8(2, n - 2)
            self.map_prg8(3, n - 1)
        if c & 0x10:
            for i in range(4):
                self.map_chr1(i, self.chr0 * 4 + i)
                self.map_chr1(4 + i, self.chr1 * 4 + i)
        else:
            base = (self.chr0 & 0x1E) * 4
            for i in range(8):
                self.map_chr1(i, base + i)


class Mapper2(Mapper):  # UxROM
    def __init__(self, cart):
        super().__init__(cart)
        self.map_chr8(0)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg8(0, val * 2)
        self.map_prg8(1, val * 2 + 1)
        self.map_prg8(2, self.n_prg8 - 2)
        self.map_prg8(3, self.n_prg8 - 1)


class Mapper3(Mapper):  # CNROM
    def __init__(self, cart):
        super().__init__(cart)
        for i in range(4):
            self.map_prg8(i, i)
        self.map_chr8(0)

    def write(self, addr, val):
        self.map_chr8(val)


class Mapper7(Mapper):  # AxROM
    def __init__(self, cart):
        super().__init__(cart)
        self.map_chr8(0)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg32(val & 7)
        self.set_mirroring("s1" if val & 0x10 else "s0")


class Mapper66(Mapper):  # GxROM
    def __init__(self, cart):
        super().__init__(cart)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg32((val >> 4) & 3)
        self.map_chr8(val & 3)


class Mapper4(Mapper):  # MMC3
    def __init__(self, cart):
        super().__init__(cart)
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]
        self.bank_select = 0
        self.irq_latch = 0
        self.irq_counter = 0
        self.irq_reload = False
        self.irq_enable = False
        self.update()

    def write(self, addr, val):
        even = not (addr & 1)
        r = (addr >> 13) & 3
        if r == 0:
            if even:
                self.bank_select = val
            else:
                self.regs[self.bank_select & 7] = val
            self.update()
        elif r == 1:
            if even and not self.cart.four_screen:
                self.set_mirroring("h" if val & 1 else "v")
        elif r == 2:
            if even:
                self.irq_latch = val
            else:
                self.irq_counter = 0
                self.irq_reload = True
        else:
            if even:
                self.irq_enable = False
                self.irq = False
            else:
                self.irq_enable = True

    def update(self):
        r = self.regs
        bs = self.bank_select
        if not (bs & 0x80):
            banks = [r[0] & 0xFE, r[0] | 1, r[1] & 0xFE, r[1] | 1, r[2], r[3], r[4], r[5]]
        else:
            banks = [r[2], r[3], r[4], r[5], r[0] & 0xFE, r[0] | 1, r[1] & 0xFE, r[1] | 1]
        for i in range(8):
            self.map_chr1(i, banks[i])
        n = self.n_prg8
        r6, r7 = r[6] & 0x3F, r[7] & 0x3F
        if not (bs & 0x40):
            p = [r6, r7, n - 2, n - 1]
        else:
            p = [n - 2, r7, r6, n - 1]
        for i in range(4):
            self.map_prg8(i, p[i])

    def scanline(self):
        if self.irq_counter == 0 or self.irq_reload:
            self.irq_counter = self.irq_latch
            self.irq_reload = False
        else:
            self.irq_counter -= 1
        if self.irq_counter == 0 and self.irq_enable:
            self.irq = True


MAPPERS = {0: Mapper0, 1: Mapper1, 2: Mapper2, 3: Mapper3, 4: Mapper4, 7: Mapper7, 66: Mapper66}

# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------
BTN_A, BTN_B, BTN_SELECT, BTN_START, BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT = (
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)


class Controller:
    def __init__(self):
        self.buttons = 0
        self.shift = 0
        self.strobe = 0

    def write(self, v):
        self.strobe = v & 1
        if self.strobe:
            self.shift = self.buttons

    def read(self):
        if self.strobe:
            return (self.buttons & 1) | 0x40
        r = self.shift & 1
        self.shift = (self.shift >> 1) | 0x80
        return r | 0x40


# --------------------------------------------------------------------------
# CPU (6502)
# --------------------------------------------------------------------------
class CPU:
    def __init__(self, bus, mapper, apu):
        self.bus = bus
        self.read = bus.read
        self.write = bus.write
        self.mapper = mapper
        self.apu = apu
        self.prg = mapper.prg
        self.prg_off = mapper.prg_off
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.pc = 0
        self.c = self.z = self.v = self.n = self.d = 0
        self.i = 1
        self.cycles = 0
        self.extra = 0
        self.crossed = 0
        self.nmi_pending = False
        self.table = self._build()

    def reset(self):
        self.sp = (self.sp - 3) & 0xFF
        self.i = 1
        self.pc = self.read(0xFFFC) | (self.read(0xFFFD) << 8)
        self.nmi_pending = False

    # ---- state helpers ----------------------------------------------------
    def get_p(self):
        return (int(self.c) | (int(self.z) << 1) | (int(self.i) << 2) | (int(self.d) << 3)
                | (int(self.v) << 6) | (int(self.n) << 7))

    def set_p(self, p):
        self.c = p & 1
        self.z = (p >> 1) & 1
        self.i = (p >> 2) & 1
        self.d = (p >> 3) & 1
        self.v = (p >> 6) & 1
        self.n = (p >> 7) & 1

    def push(self, v):
        self.write(0x100 | self.sp, v)
        self.sp = (self.sp - 1) & 0xFF

    def pull(self):
        self.sp = (self.sp + 1) & 0xFF
        return self.read(0x100 | self.sp)

    def _interrupt(self, vector, brk=False):
        pc = self.pc
        self.push(pc >> 8)
        self.push(pc & 0xFF)
        self.push(self.get_p() | (0x30 if brk else 0x20))
        self.i = 1
        self.pc = self.read(vector) | (self.read(vector + 1) << 8)

    # ---- addressing modes (return effective address) ------------------------
    def m_imp(self):
        return 0

    def m_imm(self):
        a = self.pc
        self.pc = (a + 1) & 0xFFFF
        return a

    def m_zp(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        return self.read(pc)

    def m_zpx(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        return (self.read(pc) + self.x) & 0xFF

    def m_zpy(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        return (self.read(pc) + self.y) & 0xFF

    def m_abs(self):
        pc = self.pc
        self.pc = (pc + 2) & 0xFFFF
        return self.read(pc) | (self.read((pc + 1) & 0xFFFF) << 8)

    def m_abx(self):
        pc = self.pc
        self.pc = (pc + 2) & 0xFFFF
        b = self.read(pc) | (self.read((pc + 1) & 0xFFFF) << 8)
        a = (b + self.x) & 0xFFFF
        self.crossed = 1 if (b ^ a) & 0xFF00 else 0
        return a

    def m_aby(self):
        pc = self.pc
        self.pc = (pc + 2) & 0xFFFF
        b = self.read(pc) | (self.read((pc + 1) & 0xFFFF) << 8)
        a = (b + self.y) & 0xFFFF
        self.crossed = 1 if (b ^ a) & 0xFF00 else 0
        return a

    def m_izx(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        p = (self.read(pc) + self.x) & 0xFF
        return self.read(p) | (self.read((p + 1) & 0xFF) << 8)

    def m_izy(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        p = self.read(pc)
        b = self.read(p) | (self.read((p + 1) & 0xFF) << 8)
        a = (b + self.y) & 0xFFFF
        self.crossed = 1 if (b ^ a) & 0xFF00 else 0
        return a

    def m_ind(self):
        p = self.m_abs()
        lo = self.read(p)
        hi = self.read((p & 0xFF00) | ((p + 1) & 0xFF))  # 6502 page-wrap bug
        return lo | (hi << 8)

    # ---- operations ---------------------------------------------------------
    def op_lda(self, ad):
        v = self.read(ad)
        self.a = v
        self.z = v == 0
        self.n = v >> 7

    def op_ldx(self, ad):
        v = self.read(ad)
        self.x = v
        self.z = v == 0
        self.n = v >> 7

    def op_ldy(self, ad):
        v = self.read(ad)
        self.y = v
        self.z = v == 0
        self.n = v >> 7

    def op_lax(self, ad):
        v = self.read(ad)
        self.a = self.x = v
        self.z = v == 0
        self.n = v >> 7

    def op_sta(self, ad):
        self.write(ad, self.a)

    def op_stx(self, ad):
        self.write(ad, self.x)

    def op_sty(self, ad):
        self.write(ad, self.y)

    def op_sax(self, ad):
        self.write(ad, self.a & self.x)

    def op_and(self, ad):
        v = self.a & self.read(ad)
        self.a = v
        self.z = v == 0
        self.n = v >> 7

    def op_ora(self, ad):
        v = self.a | self.read(ad)
        self.a = v
        self.z = v == 0
        self.n = v >> 7

    def op_eor(self, ad):
        v = self.a ^ self.read(ad)
        self.a = v
        self.z = v == 0
        self.n = v >> 7

    def _adc(self, m):
        a = self.a
        s = a + m + int(self.c)
        self.v = ((a ^ s) & (m ^ s) & 0x80) >> 7
        self.c = s > 0xFF
        s &= 0xFF
        self.a = s
        self.z = s == 0
        self.n = s >> 7

    def op_adc(self, ad):
        self._adc(self.read(ad))

    def op_sbc(self, ad):
        self._adc(self.read(ad) ^ 0xFF)

    def _cmp(self, r, m):
        t = (r - m) & 0xFF
        self.c = r >= m
        self.z = t == 0
        self.n = t >> 7

    def op_cmp(self, ad):
        self._cmp(self.a, self.read(ad))

    def op_cpx(self, ad):
        self._cmp(self.x, self.read(ad))

    def op_cpy(self, ad):
        self._cmp(self.y, self.read(ad))

    def op_bit(self, ad):
        m = self.read(ad)
        self.z = (self.a & m) == 0
        self.n = m >> 7
        self.v = (m >> 6) & 1

    # shifts -- memory
    def op_asl(self, ad):
        m = self.read(ad)
        self.c = m >> 7
        m = (m << 1) & 0xFF
        self.write(ad, m)
        self.z = m == 0
        self.n = m >> 7

    def op_lsr(self, ad):
        m = self.read(ad)
        self.c = m & 1
        m >>= 1
        self.write(ad, m)
        self.z = m == 0
        self.n = 0

    def op_rol(self, ad):
        m = self.read(ad)
        c = int(self.c)
        self.c = m >> 7
        m = ((m << 1) | c) & 0xFF
        self.write(ad, m)
        self.z = m == 0
        self.n = m >> 7

    def op_ror(self, ad):
        m = self.read(ad)
        c = int(self.c)
        self.c = m & 1
        m = (m >> 1) | (c << 7)
        self.write(ad, m)
        self.z = m == 0
        self.n = m >> 7

    # shifts -- accumulator
    def op_asl_a(self, ad):
        m = self.a
        self.c = m >> 7
        m = (m << 1) & 0xFF
        self.a = m
        self.z = m == 0
        self.n = m >> 7

    def op_lsr_a(self, ad):
        m = self.a
        self.c = m & 1
        m >>= 1
        self.a = m
        self.z = m == 0
        self.n = 0

    def op_rol_a(self, ad):
        m = self.a
        c = int(self.c)
        self.c = m >> 7
        m = ((m << 1) | c) & 0xFF
        self.a = m
        self.z = m == 0
        self.n = m >> 7

    def op_ror_a(self, ad):
        m = self.a
        c = int(self.c)
        self.c = m & 1
        m = (m >> 1) | (c << 7)
        self.a = m
        self.z = m == 0
        self.n = m >> 7

    def op_inc(self, ad):
        m = (self.read(ad) + 1) & 0xFF
        self.write(ad, m)
        self.z = m == 0
        self.n = m >> 7

    def op_dec(self, ad):
        m = (self.read(ad) - 1) & 0xFF
        self.write(ad, m)
        self.z = m == 0
        self.n = m >> 7

    def op_inx(self, ad):
        v = self.x = (self.x + 1) & 0xFF
        self.z = v == 0
        self.n = v >> 7

    def op_iny(self, ad):
        v = self.y = (self.y + 1) & 0xFF
        self.z = v == 0
        self.n = v >> 7

    def op_dex(self, ad):
        v = self.x = (self.x - 1) & 0xFF
        self.z = v == 0
        self.n = v >> 7

    def op_dey(self, ad):
        v = self.y = (self.y - 1) & 0xFF
        self.z = v == 0
        self.n = v >> 7

    def op_tax(self, ad):
        v = self.x = self.a
        self.z = v == 0
        self.n = v >> 7

    def op_txa(self, ad):
        v = self.a = self.x
        self.z = v == 0
        self.n = v >> 7

    def op_tay(self, ad):
        v = self.y = self.a
        self.z = v == 0
        self.n = v >> 7

    def op_tya(self, ad):
        v = self.a = self.y
        self.z = v == 0
        self.n = v >> 7

    def op_tsx(self, ad):
        v = self.x = self.sp
        self.z = v == 0
        self.n = v >> 7

    def op_txs(self, ad):
        self.sp = self.x

    def op_pha(self, ad):
        self.push(self.a)

    def op_pla(self, ad):
        v = self.a = self.pull()
        self.z = v == 0
        self.n = v >> 7

    def op_php(self, ad):
        self.push(self.get_p() | 0x30)

    def op_plp(self, ad):
        self.set_p(self.pull())

    def op_clc(self, ad):
        self.c = 0

    def op_sec(self, ad):
        self.c = 1

    def op_cli(self, ad):
        self.i = 0

    def op_sei(self, ad):
        self.i = 1

    def op_clv(self, ad):
        self.v = 0

    def op_cld(self, ad):
        self.d = 0

    def op_sed(self, ad):
        self.d = 1

    def op_nop(self, ad):
        pass

    def op_nop_m(self, ad):
        self.read(ad)

    def op_jmp(self, ad):
        self.pc = ad

    def op_jsr(self, ad):
        p = (self.pc - 1) & 0xFFFF
        self.push(p >> 8)
        self.push(p & 0xFF)
        self.pc = ad

    def op_rts(self, ad):
        lo = self.pull()
        hi = self.pull()
        self.pc = (((hi << 8) | lo) + 1) & 0xFFFF

    def op_rti(self, ad):
        self.set_p(self.pull())
        lo = self.pull()
        hi = self.pull()
        self.pc = (hi << 8) | lo

    def op_brk(self, ad):
        self.pc = (self.pc + 1) & 0xFFFF
        self._interrupt(0xFFFE, True)

    def _branch(self, cond):
        off = self.read(self.pc)
        pc = (self.pc + 1) & 0xFFFF
        if cond:
            if off & 0x80:
                off -= 256
            new = (pc + off) & 0xFFFF
            self.crossed = 2 if (new ^ pc) & 0xFF00 else 1
            pc = new
        self.pc = pc

    def op_bpl(self, ad):
        self._branch(not self.n)

    def op_bmi(self, ad):
        self._branch(self.n)

    def op_bvc(self, ad):
        self._branch(not self.v)

    def op_bvs(self, ad):
        self._branch(self.v)

    def op_bcc(self, ad):
        self._branch(not self.c)

    def op_bcs(self, ad):
        self._branch(self.c)

    def op_bne(self, ad):
        self._branch(not self.z)

    def op_beq(self, ad):
        self._branch(self.z)

    # unofficial read-modify-write combos
    def op_slo(self, ad):
        self.op_asl(ad)
        self.op_ora(ad)

    def op_rla(self, ad):
        self.op_rol(ad)
        self.op_and(ad)

    def op_sre(self, ad):
        self.op_lsr(ad)
        self.op_eor(ad)

    def op_rra(self, ad):
        self.op_ror(ad)
        self.op_adc(ad)

    def op_dcp(self, ad):
        self.op_dec(ad)
        self.op_cmp(ad)

    def op_isb(self, ad):
        self.op_inc(ad)
        self.op_sbc(ad)

    # ---- opcode table -------------------------------------------------------
    def _build(self):
        t = [(self.op_nop, self.m_imp, 2, 0)] * 256  # unknown/KIL -> NOP

        def add(op, fn, mode, cyc, pg=0):
            t[op] = (getattr(self, "op_" + fn), getattr(self, "m_" + mode), cyc, pg)

        modes8 = (("imm", 2, 0), ("zp", 3, 0), ("zpx", 4, 0), ("abs", 4, 0),
                  ("abx", 4, 1), ("aby", 4, 1), ("izx", 6, 0), ("izy", 5, 1))

        def grp(fn, ops):
            for op, (md, cy, pg) in zip(ops, modes8):
                add(op, fn, md, cy, pg)

        grp("lda", [0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1])
        grp("and", [0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31])
        grp("ora", [0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11])
        grp("eor", [0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51])
        grp("adc", [0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71])
        grp("sbc", [0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1])
        grp("cmp", [0xC9, 0xC5, 0xD5, 0xCD, 0xDD, 0xD9, 0xC1, 0xD1])
        add(0xEB, "sbc", "imm", 2)
        for op, md, cy, pg in ((0xA2, "imm", 2, 0), (0xA6, "zp", 3, 0), (0xB6, "zpy", 4, 0),
                               (0xAE, "abs", 4, 0), (0xBE, "aby", 4, 1)):
            add(op, "ldx", md, cy, pg)
        for op, md, cy, pg in ((0xA0, "imm", 2, 0), (0xA4, "zp", 3, 0), (0xB4, "zpx", 4, 0),
                               (0xAC, "abs", 4, 0), (0xBC, "abx", 4, 1)):
            add(op, "ldy", md, cy, pg)
        for op, md, cy in ((0x85, "zp", 3), (0x95, "zpx", 4), (0x8D, "abs", 4), (0x9D, "abx", 5),
                           (0x99, "aby", 5), (0x81, "izx", 6), (0x91, "izy", 6)):
            add(op, "sta", md, cy)
        for op, md, cy in ((0x86, "zp", 3), (0x96, "zpy", 4), (0x8E, "abs", 4)):
            add(op, "stx", md, cy)
        for op, md, cy in ((0x84, "zp", 3), (0x94, "zpx", 4), (0x8C, "abs", 4)):
            add(op, "sty", md, cy)
        for op, md, cy in ((0xE0, "imm", 2), (0xE4, "zp", 3), (0xEC, "abs", 4)):
            add(op, "cpx", md, cy)
        for op, md, cy in ((0xC0, "imm", 2), (0xC4, "zp", 3), (0xCC, "abs", 4)):
            add(op, "cpy", md, cy)
        add(0x24, "bit", "zp", 3)
        add(0x2C, "bit", "abs", 4)

        rmw = (("zp", 5), ("zpx", 6), ("abs", 6), ("abx", 7))

        def rm(fn, acc, ops):
            add(acc, fn + "_a", "imp", 2)
            for op, (md, cy) in zip(ops, rmw):
                add(op, fn, md, cy)

        rm("asl", 0x0A, [0x06, 0x16, 0x0E, 0x1E])
        rm("lsr", 0x4A, [0x46, 0x56, 0x4E, 0x5E])
        rm("rol", 0x2A, [0x26, 0x36, 0x2E, 0x3E])
        rm("ror", 0x6A, [0x66, 0x76, 0x6E, 0x7E])
        for fn, ops in (("inc", [0xE6, 0xF6, 0xEE, 0xFE]), ("dec", [0xC6, 0xD6, 0xCE, 0xDE])):
            for op, (md, cy) in zip(ops, rmw):
                add(op, fn, md, cy)

        for op, fn, cy in ((0xE8, "inx", 2), (0xC8, "iny", 2), (0xCA, "dex", 2), (0x88, "dey", 2),
                           (0xAA, "tax", 2), (0x8A, "txa", 2), (0xA8, "tay", 2), (0x98, "tya", 2),
                           (0xBA, "tsx", 2), (0x9A, "txs", 2), (0x48, "pha", 3), (0x68, "pla", 4),
                           (0x08, "php", 3), (0x28, "plp", 4), (0x18, "clc", 2), (0x38, "sec", 2),
                           (0x58, "cli", 2), (0x78, "sei", 2), (0xB8, "clv", 2), (0xD8, "cld", 2),
                           (0xF8, "sed", 2), (0xEA, "nop", 2), (0x60, "rts", 6), (0x40, "rti", 6),
                           (0x00, "brk", 7)):
            add(op, fn, "imp", cy)
        add(0x4C, "jmp", "abs", 3)
        add(0x6C, "jmp", "ind", 5)
        add(0x20, "jsr", "abs", 6)
        for op, fn in ((0x10, "bpl"), (0x30, "bmi"), (0x50, "bvc"), (0x70, "bvs"),
                       (0x90, "bcc"), (0xB0, "bcs"), (0xD0, "bne"), (0xF0, "beq")):
            add(op, fn, "imp", 2, 1)

        # --- unofficial opcodes ---
        for op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
            add(op, "nop", "imp", 2)
        for op in (0x80, 0x82, 0x89, 0xC2, 0xE2):
            add(op, "nop_m", "imm", 2)
        for op in (0x04, 0x44, 0x64):
            add(op, "nop_m", "zp", 3)
        add(0x0C, "nop_m", "abs", 4)
        for op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
            add(op, "nop_m", "zpx", 4)
        for op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
            add(op, "nop_m", "abx", 4, 1)
        for op, md, cy, pg in ((0xA7, "zp", 3, 0), (0xB7, "zpy", 4, 0), (0xAF, "abs", 4, 0),
                               (0xBF, "aby", 4, 1), (0xA3, "izx", 6, 0), (0xB3, "izy", 5, 1)):
            add(op, "lax", md, cy, pg)
        for op, md, cy in ((0x87, "zp", 3), (0x97, "zpy", 4), (0x8F, "abs", 4), (0x83, "izx", 6)):
            add(op, "sax", md, cy)
        rmw7 = (("zp", 5), ("zpx", 6), ("abs", 6), ("abx", 7), ("aby", 7), ("izx", 8), ("izy", 8))
        for fn, ops in (("slo", [0x07, 0x17, 0x0F, 0x1F, 0x1B, 0x03, 0x13]),
                        ("rla", [0x27, 0x37, 0x2F, 0x3F, 0x3B, 0x23, 0x33]),
                        ("sre", [0x47, 0x57, 0x4F, 0x5F, 0x5B, 0x43, 0x53]),
                        ("rra", [0x67, 0x77, 0x6F, 0x7F, 0x7B, 0x63, 0x73]),
                        ("dcp", [0xC7, 0xD7, 0xCF, 0xDF, 0xDB, 0xC3, 0xD3]),
                        ("isb", [0xE7, 0xF7, 0xEF, 0xFF, 0xFB, 0xE3, 0xF3])):
            for op, (md, cy) in zip(ops, rmw7):
                add(op, fn, md, cy)
        return t

    # ---- execute one instruction (or interrupt); returns CPU cycles ----------
    def step(self):
        if self.nmi_pending:
            self.nmi_pending = False
            self._interrupt(0xFFFA)
            self.cycles += 7
            return 7
        if not self.i and (self.mapper.irq or self.apu.irq):
            self._interrupt(0xFFFE)
            self.cycles += 7
            return 7
        pc = self.pc
        if pc >= 0x8000:
            opc = self.prg[self.prg_off[(pc >> 13) & 3] + (pc & 0x1FFF)]
        else:
            opc = self.read(pc)
        self.pc = (pc + 1) & 0xFFFF
        fn, mode, cyc, pg = self.table[opc]
        self.crossed = 0
        fn(mode())
        if pg:
            cyc += self.crossed
        if self.extra:
            cyc += self.extra
            self.extra = 0
        self.cycles += cyc
        return cyc


# --------------------------------------------------------------------------
# Bus
# --------------------------------------------------------------------------
class Bus:
    def __init__(self, mapper):
        self.mapper = mapper
        self.ram = bytearray(0x800)
        self.prg = mapper.prg
        self.prg_off = mapper.prg_off
        self.prg_ram = mapper.prg_ram
        self.ppu = self.apu = self.cpu = None
        self.pads = None

    def read(self, a):
        if a < 0x2000:
            return self.ram[a & 0x7FF]
        if a >= 0x8000:
            return self.prg[self.prg_off[(a >> 13) & 3] + (a & 0x1FFF)]
        if a < 0x4000:
            return self.ppu.read_reg(a & 7)
        if a < 0x4020:
            if a == 0x4016:
                return self.pads[0].read()
            if a == 0x4017:
                return self.pads[1].read()
            if a == 0x4015:
                return self.apu.read_status()
            return 0
        if a >= 0x6000:
            return self.prg_ram[a - 0x6000]
        return 0

    def write(self, a, v):
        if a < 0x2000:
            self.ram[a & 0x7FF] = v
        elif a < 0x4000:
            self.ppu.write_reg(a & 7, v)
        elif a < 0x4020:
            if a == 0x4014:
                self._dma(v)
            elif a == 0x4016:
                self.pads[0].write(v)
                self.pads[1].write(v)
            else:
                self.apu.write(a, v)
        elif a >= 0x8000:
            self.mapper.write(a, v)
        elif a >= 0x6000:
            self.prg_ram[a - 0x6000] = v

    def _dma(self, page):
        base = page << 8
        ppu = self.ppu
        oam = ppu.oam
        start = ppu.oam_addr
        if ppu.pend:
            ppu.flush()
        if base < 0x2000 and start == 0:
            b = base & 0x7FF
            oam[:] = self.ram[b:b + 256]
        else:
            rd = self.read
            for i in range(256):
                oam[(start + i) & 0xFF] = rd(base + i)
        self.cpu.extra += 513 + (self.cpu.cycles & 1)


# --------------------------------------------------------------------------
# PPU
# --------------------------------------------------------------------------
class PPU:
    def __init__(self, mapper):
        self.mapper = mapper
        self.cpu = None
        self.chr_ram = mapper.cart.chr_ram
        self.vram = bytearray(0x1000)
        self.palette = bytearray(32)
        self.pal_np = np.frombuffer(self.palette, dtype=np.uint8)
        self.oam = bytearray(256)
        self.ctrl = self.mask = self.status = 0
        self.oam_addr = 0
        self.v = self.t = self.x = self.w = 0
        self.buf = 0
        self.latch = 0
        self.s0_done = False
        self.ovf_lines = set()
        self.frame_idx = np.zeros((240, 256), dtype=np.uint8)
        # numpy views of live memory (used by the batched renderer)
        self.vram_np = np.frombuffer(self.vram, dtype=np.uint8)
        self.oam_np = np.frombuffer(self.oam, dtype=np.uint8)
        self.chr_np = np.frombuffer(mapper.chr, dtype=np.uint8)
        # sprite evaluation results (rebuilt on line 0 of every frame)
        self.spr_h = 8
        self.s0_y = 255
        self.sp_line = self.sp_i = self.sp_rank = np.zeros(0, dtype=np.int32)
        # deferred scanline queue: visible lines are only *recorded* while the
        # frame runs and are rendered together in flush()
        self.pend = []
        self._co_list = None
        self._nt_list = None
        self._co_tabs = []
        self._nt_tabs = []
        self._co_id = self._nt_id = 0

    # ---- raw PPU memory -----------------------------------------------------
    def read_vram(self, a):
        a &= 0x3FFF
        if a < 0x2000:
            m = self.mapper
            return m.chr[m.chr_off[a >> 10] + (a & 0x3FF)]
        if a < 0x3F00:
            return self.vram[self.mapper.nt_off[(a >> 10) & 3] + (a & 0x3FF)]
        i = a & 0x1F
        if not (i & 3):
            i &= 0x0F
        return self.palette[i]

    def write_vram(self, a, v):
        a &= 0x3FFF
        if a < 0x2000:
            if self.chr_ram:
                m = self.mapper
                m.chr[m.chr_off[a >> 10] + (a & 0x3FF)] = v
        elif a < 0x3F00:
            self.vram[self.mapper.nt_off[(a >> 10) & 3] + (a & 0x3FF)] = v
        else:
            i = a & 0x1F
            if not (i & 3):
                i &= 0x0F
            self.palette[i] = v & 0x3F

    # ---- CPU-visible registers ----------------------------------------------
    def read_reg(self, r):
        if r == 2:
            s = (self.status & 0xE0) | (self.latch & 0x1F)
            self.status &= 0x7F
            self.w = 0
            return s
        if r == 4:
            return self.oam[self.oam_addr]
        if r == 7:
            a = self.v & 0x3FFF
            if a >= 0x3F00:
                val = self.read_vram(a) & 0x3F
                self.buf = self.read_vram(a - 0x1000)
            else:
                val = self.buf
                self.buf = self.read_vram(a)
            self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF
            return val
        return self.latch

    def write_reg(self, r, v):
        self.latch = v
        if r == 0:
            old = self.ctrl
            self.ctrl = v
            self.t = (self.t & 0x73FF) | ((v & 3) << 10)
            if (v & 0x80) and not (old & 0x80) and (self.status & 0x80):
                self.cpu.nmi_pending = True
        elif r == 1:
            self.mask = v
        elif r == 3:
            self.oam_addr = v
        elif r == 4:
            if self.pend:
                self.flush()
            self.oam[self.oam_addr] = v
            self.oam_addr = (self.oam_addr + 1) & 0xFF
        elif r == 5:
            if self.w == 0:
                self.t = (self.t & 0x7FE0) | (v >> 3)
                self.x = v & 7
                self.w = 1
            else:
                self.t = (self.t & 0x0C1F) | ((v & 7) << 12) | ((v & 0xF8) << 2)
                self.w = 0
        elif r == 6:
            if self.w == 0:
                self.t = (self.t & 0x00FF) | ((v & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0xFF00) | v
                self.v = self.t
                self.w = 0
        elif r == 7:
            if self.pend:  # VRAM / palette changes must not affect lines already drawn
                self.flush()
            self.write_vram(self.v & 0x3FFF, v)
            self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF

    # ---- scroll bookkeeping ---------------------------------------------------
    def inc_y_copy_x(self):
        v = self.v
        if (v & 0x7000) != 0x7000:
            v += 0x1000
        else:
            v &= ~0x7000
            y = (v & 0x03E0) >> 5
            if y == 29:
                y = 0
                v ^= 0x0800
            elif y == 31:
                y = 0
            else:
                y += 1
            v = (v & ~0x03E0) | (y << 5)
        self.v = (v & ~0x041F) | (self.t & 0x041F)

    # ---- rendering ------------------------------------------------------------
    def evaluate_sprites(self):
        """Vectorised sprite evaluation, once per frame (line 0).  Produces the
        flat (line, sprite, rank) list of the first 8 sprites on every line."""
        h = 16 if self.ctrl & 0x20 else 8
        self.spr_h = h
        oam = self.oam_np
        y0 = oam[0::4].astype(np.int32) + 1
        self.s0_y = int(y0[0])
        lines = y0[:, None] + np.arange(h, dtype=np.int32)
        valid = (y0 < 240)[:, None] & (lines < 240)
        grid_i = np.broadcast_to(np.arange(64, dtype=np.int32)[:, None], lines.shape)
        fl = lines[valid]
        fi = grid_i[valid]
        if fl.size == 0:
            self.sp_line = self.sp_i = self.sp_rank = np.zeros(0, dtype=np.int32)
            self.ovf_lines = set()
            return
        order = np.argsort(fl, kind="stable")  # by line, sprite index ascending
        fl = fl[order]
        fi = fi[order]
        starts = np.concatenate(([0], np.nonzero(fl[1:] != fl[:-1])[0] + 1))
        counts = np.diff(np.concatenate((starts, [fl.size])))
        rank = np.arange(fl.size, dtype=np.int32) - np.repeat(starts, counts)
        keep = rank < 8
        self.sp_line = fl[keep]
        self.sp_i = fi[keep]
        self.sp_rank = rank[keep]
        self.ovf_lines = set(fl[starts[counts > 8]].tolist())

    def render_bg_row(self):
        """Single-scanline background fetch (only used for sprite-0 hit tests)."""
        v = self.v
        cx = v & 31
        cy = (v >> 5) & 31
        nt = (v >> 10) & 3
        fy = (v >> 12) & 7
        base = 0x1000 if self.ctrl & 0x10 else 0
        vram = self.vram
        nt_off = self.mapper.nt_off
        chr_ = self.mapper.chr
        chr_off = self.mapper.chr_off
        idx = []
        for _ in range(33):
            o = nt_off[nt]
            tile = vram[o + (cy << 5) + cx]
            a = base + (tile << 4) + fy
            co = chr_off[a >> 10] + (a & 0x3FF)
            idx.append((chr_[co + 8] << 8) | chr_[co])
            cx += 1
            if cx == 32:
                cx = 0
                nt ^= 1
        pix = PIX[idx].reshape(-1)
        fx = self.x
        pix = pix[fx:fx + 256]
        if not (self.mask & 0x02):
            pix = pix.copy()
            pix[:8] = 0
        return pix

    def _s0_hit(self, sl):
        """Dot (x position) of the sprite-0 hit on this line, or -1."""
        pix = self.render_bg_row().tolist()
        mask = self.mask
        ctrl = self.ctrl
        oam = self.oam
        tall = bool(ctrl & 0x20)
        h = 16 if tall else 8
        row = sl - (oam[0] + 1)
        if row < 0 or row >= h:
            return -1
        tile = oam[1]
        attr = oam[2]
        sx = oam[3]
        if attr & 0x80:
            row = h - 1 - row
        if tall:
            a = ((tile & 1) << 12) + ((tile & 0xFE) << 4) + ((row & 8) << 1) + (row & 7)
        else:
            a = (0x1000 if ctrl & 0x08 else 0) + (tile << 4) + row
        m = self.mapper
        co = m.chr_off[a >> 10] + (a & 0x3FF)
        lo = m.chr[co]
        hi = m.chr[co + 8]
        if not (lo | hi):
            return -1
        sp = pattern_row(lo, hi)
        if attr & 0x40:
            sp = sp[::-1]
        spr_left = mask & 0x04
        for j in range(8):
            if sp[j]:
                px = sx + j
                if px > 255:
                    break
                if pix[px] and px != 255 and (px >= 8 or spr_left):
                    return px
        return -1

    def render_scanline(self, sl, draw):
        """Called at the start of every visible scanline.  With draw=True the line
        is queued for batched rendering (flush()).  Returns the dot at which a
        sprite-0 hit happens on this line (or -1)."""
        mask = self.mask
        show_bg = mask & 0x08
        show_spr = mask & 0x10
        if not (show_bg or show_spr):
            if draw:
                self.frame_idx[sl] = self.pal_np[0] & (0x30 if mask & 1 else 0x3F)
            return -1
        if sl in self.ovf_lines:
            self.status |= 0x20
        if draw:
            m = self.mapper
            co = m.chr_off
            if co != self._co_list:
                self._co_list = list(co)
                self._co_tabs.append(tuple(co))
                self._co_id = len(self._co_tabs) - 1
            nt = m.nt_off
            if nt != self._nt_list:
                self._nt_list = list(nt)
                self._nt_tabs.append(tuple(nt))
                self._nt_id = len(self._nt_tabs) - 1
            self.pend.append((sl, self.v, self.x, self.ctrl, mask, self._co_id, self._nt_id))
        if (show_bg and show_spr and not self.s0_done
                and self.s0_y <= sl < self.s0_y + self.spr_h):
            hit = self._s0_hit(sl)
            if hit >= 0:
                return hit + 2
        return -1

    def flush(self):
        """Render every queued scanline at once with numpy."""
        pend = self.pend
        if not pend:
            return
        arr = np.array(pend, dtype=np.int32)
        co_tabs = np.array(self._co_tabs, dtype=np.int32)
        nt_tabs = np.array(self._nt_tabs, dtype=np.int32)
        self.pend = []
        self._co_list = self._nt_list = None
        self._co_tabs = []
        self._nt_tabs = []

        L = arr.shape[0]
        sl = arr[:, 0]
        v = arr[:, 1]
        fx = arr[:, 2]
        ctrl = arr[:, 3]
        mask = arr[:, 4]
        chr_lines = co_tabs[arr[:, 5]]  # (L, 8) 1K bank offsets per line
        nt_lines = nt_tabs[arr[:, 6]]   # (L, 4)
        ar = np.arange(L)
        vram = self.vram_np
        chr_ = self.chr_np

        # ---- background: 33 tiles x L lines in one go ---------------------
        cx = v & 31
        cy = (v >> 5) & 31
        nt = (v >> 10) & 3
        fy = (v >> 12) & 7
        cxj = cx[:, None] + _J33
        col = cxj & 31
        o = nt_lines[ar[:, None], nt[:, None] ^ (cxj >> 5)]
        tile = vram[o + (cy << 5)[:, None] + col].astype(np.int32)
        attr = vram[o + 0x3C0 + ((cy >> 2) << 3)[:, None] + (col >> 2)].astype(np.int32)
        pal = ((attr >> (((cy & 2) << 1)[:, None] | (col & 2))) & 3) << 2
        base = np.where(ctrl & 0x10, 0x1000, 0)
        a = base[:, None] + (tile << 4) + fy[:, None]
        co = chr_lines[ar[:, None], a >> 10] + (a & 0x3FF)
        idx = (chr_[co + 8].astype(np.uint16) << 8) | chr_[co]
        pix = PIX[idx].reshape(L, 264)
        palx = np.repeat(pal.astype(np.uint8), 8, axis=1)
        if (fx == fx[0]).all():
            f0 = int(fx[0])
            pix = pix[:, f0:f0 + 256].copy()
            palx = palx[:, f0:f0 + 256]
        else:
            xi = fx[:, None] + _J256
            pix = np.take_along_axis(pix, xi, 1)
            palx = np.take_along_axis(palx, xi, 1)
        no_left = (mask & 0x02) == 0
        if no_left.any():
            pix[no_left, :8] = 0
        no_bg = (mask & 0x08) == 0
        if no_bg.any():
            pix[no_bg] = 0
        bg_col = np.where(pix != 0, pix | palx, 0).astype(np.uint8)

        # ---- sprites ----------------------------------------------------------
        final = bg_col
        sp_line = self.sp_line
        if sp_line.size and (mask & 0x10).any():
            slot = np.full(240, -1, dtype=np.int32)
            slot[sl] = ar
            ps = slot[sp_line]
            ps0 = np.where(ps >= 0, ps, 0)
            m = (ps >= 0) & ((mask[ps0] & 0x10) != 0)
            if m.any():
                P_line = sp_line[m]
                P_i = self.sp_i[m]
                P_rank = self.sp_rank[m]
                P_slot = ps[m]
                ob = self.oam_np.reshape(64, 4)[P_i]
                h = self.spr_h
                row = P_line - (ob[:, 0].astype(np.int32) + 1)
                ok = (row >= 0) & (row < h)
                if not ok.all():
                    P_rank = P_rank[ok]
                    P_slot = P_slot[ok]
                    ob = ob[ok]
                    row = row[ok]
                if row.size:
                    tile = ob[:, 1].astype(np.int32)
                    attr = ob[:, 2].astype(np.int32)
                    sx = ob[:, 3].astype(np.int32)
                    row = np.where(attr & 0x80, h - 1 - row, row)
                    if h == 16:
                        a = ((tile & 1) << 12) + ((tile & 0xFE) << 4) + ((row & 8) << 1) + (row & 7)
                    else:
                        a = np.where(ctrl[P_slot] & 0x08, 0x1000, 0) + (tile << 4) + row
                    co = chr_lines[P_slot, a >> 10] + (a & 0x3FF)
                    spx = PIX[(chr_[co + 8].astype(np.uint16) << 8) | chr_[co]]
                    fh = (attr & 0x40) != 0
                    if fh.any():
                        spx[fh] = spx[fh][:, ::-1]
                    px = sx[:, None] + _J8
                    valid = (spx != 0) & (px <= 255)
                    val = (0x10 | ((attr & 3) << 2))[:, None] + spx
                    behind = np.broadcast_to(((attr & 0x20) != 0)[:, None], spx.shape)
                    spr_pix = np.zeros((L, 256), dtype=np.uint8)
                    spr_col = np.zeros((L, 256), dtype=np.uint8)
                    spr_pri = np.zeros((L, 256), dtype=np.uint8)
                    slot8 = np.broadcast_to(P_slot[:, None], spx.shape)
                    for r in range(7, -1, -1):  # low priority first, high priority overwrites
                        sel = P_rank == r
                        if not sel.any():
                            continue
                        vm = valid[sel]
                        rr = slot8[sel][vm]
                        cc = px[sel][vm]
                        spr_pix[rr, cc] = spx[sel][vm]
                        spr_col[rr, cc] = val[sel][vm]
                        spr_pri[rr, cc] = behind[sel][vm]
                    no_sl = (mask & 0x04) == 0
                    if no_sl.any():
                        spr_pix[no_sl, :8] = 0
                    use = (spr_pix != 0) & ((spr_pri == 0) | (pix == 0))
                    final = np.where(use, spr_col, bg_col)

        gmask = np.where(mask & 1, 0x30, 0x3F).astype(np.uint8)
        self.frame_idx[sl] = self.pal_np[final] & gmask[:, None]


_J33 = np.arange(33, dtype=np.int32)
_J256 = np.arange(256, dtype=np.int32)
_J8 = np.arange(8, dtype=np.int32)


# --------------------------------------------------------------------------
# APU
# --------------------------------------------------------------------------
LENGTH_TABLE = [10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
                12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30]
DUTY = np.array([[0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float32)
NOISE_PERIOD = [4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068]
DMC_RATE = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54]
TRI_SEQ = np.array(list(range(15, -1, -1)) + list(range(0, 16)), dtype=np.float32)
_NOISE_SEQ: dict = {}


def noise_seq(short):
    s = _NOISE_SEQ.get(short)
    if s is None:
        sr = 1
        out = []
        tap = 6 if short else 1
        for _ in range(32767):
            out.append(0.0 if sr & 1 else 1.0)
            fb = (sr & 1) ^ ((sr >> tap) & 1)
            sr = (sr >> 1) | (fb << 14)
        s = _NOISE_SEQ[short] = np.array(out, dtype=np.float32)
    return s


class Envelope:
    """Shared envelope / length-counter logic."""

    def _env_init(self):
        self.enabled = False
        self.halt = False
        self.const = False
        self.vol = 0
        self.length = 0
        self.env_start = False
        self.env_div = 0
        self.env_decay = 0

    def set_enabled(self, en):
        self.enabled = bool(en)
        if not en:
            self.length = 0

    def clock_env(self):
        if self.env_start:
            self.env_start = False
            self.env_decay = 15
            self.env_div = self.vol
        elif self.env_div == 0:
            self.env_div = self.vol
            if self.env_decay > 0:
                self.env_decay -= 1
            elif self.halt:
                self.env_decay = 15
        else:
            self.env_div -= 1

    def clock_length(self):
        if self.length > 0 and not self.halt:
            self.length -= 1

    def volume(self):
        return self.vol if self.const else self.env_decay


class Pulse(Envelope):
    def __init__(self, ch):
        self.ch = ch
        self._env_init()
        self.duty = 0
        self.timer = 0
        self.sweep_en = False
        self.sweep_period = 0
        self.sweep_neg = False
        self.sweep_shift = 0
        self.sweep_div = 0
        self.sweep_reload = False
        self.phase = 0.0

    def write(self, r, v):
        if r == 0:
            self.duty = v >> 6
            self.halt = bool(v & 0x20)
            self.const = bool(v & 0x10)
            self.vol = v & 15
        elif r == 1:
            self.sweep_en = bool(v & 0x80)
            self.sweep_period = (v >> 4) & 7
            self.sweep_neg = bool(v & 8)
            self.sweep_shift = v & 7
            self.sweep_reload = True
        elif r == 2:
            self.timer = (self.timer & 0x700) | v
        else:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.env_start = True
            self.phase = 0.0

    def _target(self):
        d = self.timer >> self.sweep_shift
        if self.sweep_neg:
            return self.timer - d - (1 if self.ch == 0 else 0)
        return self.timer + d

    def clock_sweep(self):
        t = self._target()
        if (self.sweep_div == 0 and self.sweep_en and self.sweep_shift
                and self.timer >= 8 and 0 <= t <= 0x7FF):
            self.timer = t
        if self.sweep_div == 0 or self.sweep_reload:
            self.sweep_div = self.sweep_period
            self.sweep_reload = False
        else:
            self.sweep_div -= 1

    def render(self, n, sr):
        if not self.enabled or self.length == 0 or self.timer < 8 or self._target() > 0x7FF:
            return None
        vol = self.volume()
        if vol == 0:
            return None
        step = CPU_HZ / (16.0 * (self.timer + 1)) / sr
        ph = self.phase + step * np.arange(n)
        self.phase = (self.phase + step * n) % 1.0
        return DUTY[self.duty][((ph % 1.0) * 8).astype(np.intp)] * vol


class Triangle:
    def __init__(self):
        self.enabled = False
        self.control = False
        self.reload_val = 0
        self.linear = 0
        self.reload_flag = False
        self.timer = 0
        self.length = 0
        self.phase = 0.0

    def write(self, r, v):
        if r == 0:
            self.control = bool(v & 0x80)
            self.reload_val = v & 0x7F
        elif r == 2:
            self.timer = (self.timer & 0x700) | v
        elif r == 3:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.reload_flag = True

    def set_enabled(self, en):
        self.enabled = bool(en)
        if not en:
            self.length = 0

    def clock_linear(self):
        if self.reload_flag:
            self.linear = self.reload_val
        elif self.linear > 0:
            self.linear -= 1
        if not self.control:
            self.reload_flag = False

    def clock_length(self):
        if self.length > 0 and not self.control:
            self.length -= 1

    def render(self, n, sr):
        if not self.enabled or self.length == 0 or self.linear == 0 or self.timer < 2:
            return None
        step = CPU_HZ / (32.0 * (self.timer + 1)) / sr
        ph = self.phase + step * np.arange(n)
        self.phase = (self.phase + step * n) % 1.0
        return TRI_SEQ[((ph % 1.0) * 32).astype(np.intp)]


class Noise(Envelope):
    def __init__(self):
        self._env_init()
        self.short = False
        self.period = NOISE_PERIOD[0]
        self.pos = 0.0

    def write(self, r, v):
        if r == 0:
            self.halt = bool(v & 0x20)
            self.const = bool(v & 0x10)
            self.vol = v & 15
        elif r == 2:
            self.short = bool(v & 0x80)
            self.period = NOISE_PERIOD[v & 15]
        elif r == 3:
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.env_start = True

    def render(self, n, sr):
        if not self.enabled or self.length == 0:
            return None
        vol = self.volume()
        if vol == 0:
            return None
        seq = noise_seq(self.short)
        rate = CPU_HZ / self.period / sr
        idx = ((self.pos + rate * np.arange(n)).astype(np.int64)) % len(seq)
        self.pos = (self.pos + rate * n) % len(seq)
        return seq[idx] * vol


class DMC:
    def __init__(self):
        self.irq_en = False
        self.loop = False
        self.period = DMC_RATE[0]
        self.level = 0
        self.sample_addr = 0xC000
        self.sample_len = 1
        self.cur_addr = 0xC000
        self.bytes_remaining = 0
        self.shift = 0
        self.bits = 8
        self.buffer = None
        self.silence = True
        self.acc = 0.0
        self.irq = False

    def write(self, r, v):
        if r == 0:
            self.irq_en = bool(v & 0x80)
            self.loop = bool(v & 0x40)
            self.period = DMC_RATE[v & 15]
            if not self.irq_en:
                self.irq = False
        elif r == 1:
            self.level = v & 0x7F
        elif r == 2:
            self.sample_addr = 0xC000 + v * 64
        else:
            self.sample_len = v * 16 + 1

    def start(self):
        self.cur_addr = self.sample_addr
        self.bytes_remaining = self.sample_len

    def set_enabled(self, en, bus):
        if not en:
            self.bytes_remaining = 0
        elif self.bytes_remaining == 0:
            self.start()
            self.fill(bus)

    def fill(self, bus):
        if self.buffer is None and self.bytes_remaining > 0:
            self.buffer = bus.read(self.cur_addr)
            self.cur_addr = ((self.cur_addr + 1) | 0x8000) & 0xFFFF
            self.bytes_remaining -= 1
            if self.bytes_remaining == 0:
                if self.loop:
                    self.start()
                elif self.irq_en:
                    self.irq = True

    def run(self, cycles, bus):
        """Advance the output unit; returns the list of output levels."""
        self.acc += cycles
        levels = []
        while self.acc >= self.period:
            self.acc -= self.period
            if not self.silence:
                if self.shift & 1:
                    if self.level <= 125:
                        self.level += 2
                elif self.level >= 2:
                    self.level -= 2
                self.shift >>= 1
            self.bits -= 1
            if self.bits == 0:
                self.bits = 8
                if self.buffer is None:
                    self.silence = True
                else:
                    self.silence = False
                    self.shift = self.buffer
                    self.buffer = None
                self.fill(bus)
            levels.append(self.level)
        return levels


class APU:
    def __init__(self, bus, sample_rate=44100):
        self.bus = bus
        self.p1 = Pulse(0)
        self.p2 = Pulse(1)
        self.tri = Triangle()
        self.noi = Noise()
        self.dmc = DMC()
        self.frame_irq = False
        self.irq = False
        self.mode5 = False
        self.irq_inhibit = False
        self.sr = sample_rate
        self.acc = 0.0
        self.dc = 0.0
        self.volume = 0.7
        self.sink = None  # callable(np.int16 array)

    def write(self, a, v):
        if a < 0x4004:
            self.p1.write(a & 3, v)
        elif a < 0x4008:
            self.p2.write(a & 3, v)
        elif a < 0x400C:
            self.tri.write(a & 3, v)
        elif a < 0x4010:
            self.noi.write(a & 3, v)
        elif a < 0x4014:
            self.dmc.write(a & 3, v)
        elif a == 0x4015:
            self.p1.set_enabled(v & 1)
            self.p2.set_enabled(v & 2)
            self.tri.set_enabled(v & 4)
            self.noi.set_enabled(v & 8)
            self.dmc.set_enabled(v & 16, self.bus)
            self.dmc.irq = False
            self.update_irq()
        elif a == 0x4017:
            self.mode5 = bool(v & 0x80)
            self.irq_inhibit = bool(v & 0x40)
            if self.irq_inhibit:
                self.frame_irq = False
            if self.mode5:
                self.clock_quarter()
                self.clock_half()
            self.update_irq()

    def read_status(self):
        s = ((self.p1.length > 0) | ((self.p2.length > 0) << 1) | ((self.tri.length > 0) << 2)
             | ((self.noi.length > 0) << 3) | ((self.dmc.bytes_remaining > 0) << 4)
             | (self.frame_irq << 6) | (self.dmc.irq << 7))
        self.frame_irq = False
        self.update_irq()
        return int(s)

    def update_irq(self):
        self.irq = self.frame_irq or self.dmc.irq

    def clock_quarter(self):
        self.p1.clock_env()
        self.p2.clock_env()
        self.noi.clock_env()
        self.tri.clock_linear()

    def clock_half(self):
        self.p1.clock_length()
        self.p2.clock_length()
        self.noi.clock_length()
        self.tri.clock_length()
        self.p1.clock_sweep()
        self.p2.clock_sweep()

    def run_quarter(self, q, cycles):
        """Called 4x per frame. Renders the audio for the elapsed cycles with the
        current register state, then clocks the frame sequencer."""
        levels = self.dmc.run(cycles, self.bus)
        if self.sink is not None:
            self.acc += cycles * self.sr / CPU_HZ
            n = int(self.acc)
            self.acc -= n
            if n > 0:
                n = max(0, n)
                self._render(n, levels)
        if q in (1, 3):
            self.clock_half()
        self.clock_quarter()
        if q == 3 and not self.mode5 and not self.irq_inhibit:
            self.frame_irq = True
        self.update_irq()

    def _render(self, n, levels):
        sr = self.sr
        pulse = np.zeros(n, dtype=np.float32)
        a = self.p1.render(n, sr)
        if a is not None:
            pulse += a
        a = self.p2.render(n, sr)
        if a is not None:
            pulse += a
        out = pulse * 0.00752
        a = self.tri.render(n, sr)
        if a is not None:
            out += a * 0.00851
        a = self.noi.render(n, sr)
        if a is not None:
            out += a * 0.00494
        if levels:
            lv = np.array(levels, dtype=np.float32)
            out += lv[(np.arange(n) * len(lv)) // n] * 0.00335
        else:
            out += self.dmc.level * 0.00335
        m = float(out.mean())
        out -= self.dc
        self.dc += (m - self.dc) * 0.05
        pcm = np.clip(out * (32000.0 * self.volume), -32767, 32767).astype(np.int16)
        self.sink(pcm)


class AudioRing:
    """Thread-safe int16 ring buffer between the emulator (producer) and the
    audio device callback (consumer).

    * never grows beyond `capacity` (oldest samples are dropped) so latency is bounded
    * after an underrun it stays silent until `prime` samples are queued again,
      instead of stuttering through a chain of tiny glitches
    * an underrun fades the last sample out rather than clicking
    """

    def __init__(self, capacity, prime):
        self.buf = np.zeros(capacity, dtype=np.int16)
        self.cap = capacity
        self.prime = prime
        self.r = 0
        self.w = 0
        self.primed = False
        self.last = 0
        self.underruns = 0
        self.lock = threading.Lock()

    def buffered(self):
        return self.w - self.r

    def clear(self):
        with self.lock:
            self.r = self.w = 0
            self.primed = False

    def push(self, pcm):
        n = len(pcm)
        if n == 0:
            return
        with self.lock:
            if n >= self.cap:
                pcm = pcm[-self.cap:]
                n = self.cap
            free = self.cap - (self.w - self.r)
            if n > free:
                self.r += n - free  # overrun: drop the oldest audio
            i = self.w % self.cap
            k = min(n, self.cap - i)
            self.buf[i:i + k] = pcm[:k]
            if k < n:
                self.buf[:n - k] = pcm[k:]
            self.w += n

    def pull(self, n):
        out = np.zeros(n, dtype=np.int16)
        m = 0
        with self.lock:
            avail = self.w - self.r
            if not self.primed and avail >= self.prime:
                self.primed = True
            if self.primed:
                m = min(n, avail)
                if m:
                    i = self.r % self.cap
                    k = min(m, self.cap - i)
                    out[:k] = self.buf[i:i + k]
                    if k < m:
                        out[k:m] = self.buf[:m - k]
                    self.r += m
                if m < n:
                    self.primed = False
                    self.underruns += 1
        if m:
            self.last = int(out[m - 1])
        if m < n and self.last:
            ln = min(n - m, 256)
            out[m:m + ln] = (self.last * np.linspace(1.0, 0.0, ln, dtype=np.float32)).astype(np.int16)
            self.last = 0
        return out


class _AudioBase:
    def push(self, pcm):
        self.ring.push(pcm)

    def clear(self):
        self.ring.clear()

    def buffered_seconds(self):
        return self.ring.buffered() / float(self.rate)


class SoundDeviceOut(_AudioBase):
    """Low-latency output through PortAudio (pip install sounddevice)."""
    name = "sounddevice"

    def __init__(self, rate, latency):
        import sounddevice as sd
        self.rate = rate
        # Keep only a few target-buffer windows in software; a huge queue makes
        # latency controls ineffective and lets stale audio accumulate.
        self.blocksize = max(64, min(512, int(rate * min(0.012, max(0.003, latency / 3)))))
        self.ring = AudioRing(int(rate * max(0.12, latency * 4)), prime=max(self.blocksize * 2, int(rate * 0.006)))
        ring = self.ring

        def cb(outdata, frames, time_info, status):
            try:
                outdata[:, 0] = ring.pull(frames)
            except Exception:
                outdata.fill(0)

        kw = dict(samplerate=rate, channels=1, dtype="int16", callback=cb,
                  blocksize=self.blocksize, latency=max(0.005, min(0.05, latency)))
        tries = []
        if sys.platform == "win32":  # WASAPI has far lower latency than the default MME
            try:
                for api in sd.query_hostapis():
                    if "WASAPI" in api["name"] and api["default_output_device"] >= 0:
                        tries.append(dict(device=api["default_output_device"],
                                          extra_settings=sd.WasapiSettings(auto_convert=True)))
            except Exception:
                pass
        tries.append({})
        err = None
        self.stream = None
        for extra in tries:
            try:
                s = sd.OutputStream(**kw, **extra)
                s.start()
                self.stream = s
                break
            except Exception as e:
                err = e
        if self.stream is None:
            raise RuntimeError(err)
        try:
            self.device_latency = float(self.stream.latency)
        except Exception:
            self.device_latency = 0.0

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class PygletOut(_AudioBase):
    """Fallback output through pyglet's media player (higher latency)."""
    name = "pyglet"

    def __init__(self, rate, latency):
        from pyglet.media import Player
        from pyglet.media.codecs.base import AudioFormat, AudioData, StreamingSource

        self.rate = rate
        self.device_latency = 0.0
        self.ring = AudioRing(int(rate * max(0.12, latency * 4)),
                              prime=max(256, int(rate * min(0.012, latency / 2))))
        ring = self.ring

        class _Src(StreamingSource):
            def __init__(s):
                s.audio_format = AudioFormat(channels=1, sample_size=16, sample_rate=rate)
                s._duration = 86400.0
                s._t = 0.0

            def get_audio_data(s, num_bytes, compensation_time=0.0):
                num_bytes = max(2, num_bytes - (num_bytes % 2))
                data = ring.pull(num_bytes // 2).tobytes()
                dur = len(data) / (2.0 * rate)
                ts = s._t
                s._t += dur
                try:
                    return AudioData(data, len(data), ts, dur, [])
                except TypeError:  # newer pyglet signature
                    return AudioData(data, len(data), ts, dur)

            def seek(s, timestamp):
                pass

        self.player = Player()
        self.player.queue(_Src())
        self.player.play()

    def close(self):
        try:
            self.player.pause()
            self.player.delete()
        except Exception:
            pass


def make_audio(rate, latency=0.06, backend="auto"):
    """Open the best available audio backend; returns None if there is none."""
    order = {"auto": (SoundDeviceOut, PygletOut), "sounddevice": (SoundDeviceOut,),
             "pyglet": (PygletOut,)}[backend]
    errors = []
    for cls in order:
        try:
            out = cls(rate, latency)
            print(f"[audio] {out.name}" + (f" (device latency ~{out.device_latency * 1000:.0f} ms)"
                                           if out.device_latency else ""))
            if out.name == "pyglet" and backend == "auto":
                print("[audio] tip: 'pip install sounddevice' gives much lower audio latency")
            return out
        except Exception as e:
            errors.append(f"{cls.name}: {e}")
    print("[audio disabled] " + "; ".join(errors), file=sys.stderr)
    return None


# --------------------------------------------------------------------------
# The console
# --------------------------------------------------------------------------
CPU_KEYS = ("a", "x", "y", "sp", "pc", "c", "z", "i", "d", "v", "n", "cycles", "nmi_pending", "extra")
PPU_KEYS = ("ctrl", "mask", "status", "oam_addr", "v", "t", "x", "w", "buf", "latch", "s0_done")
APU_KEYS = ("frame_irq", "irq", "mode5", "irq_inhibit", "acc", "dc")
QUARTER_LINES = {65: 0, 130: 1, 196: 2, 261: 3}


class NES:
    def __init__(self, rom_path, sample_rate=44100):
        self.rom_path = rom_path
        self.cart = Cartridge(rom_path)
        cls = MAPPERS.get(self.cart.mapper_id)
        if cls is None:
            raise NotImplementedError(
                f"Mapper {self.cart.mapper_id} is not supported "
                f"(supported: {sorted(MAPPERS)})")
        self.mapper = cls(self.cart)
        self.bus = Bus(self.mapper)
        self.ppu = PPU(self.mapper)
        self.apu = APU(self.bus, sample_rate)
        self.cpu = CPU(self.bus, self.mapper, self.apu)
        self.pads = [Controller(), Controller()]
        self.bus.ppu, self.bus.apu, self.bus.cpu, self.bus.pads = self.ppu, self.apu, self.cpu, self.pads
        self.ppu.cpu = self.cpu
        self.dot = 0
        self.apu_mark = 0
        self.frames = 0
        self._load_battery()
        self.cpu.reset()

    # ---- persistence -----------------------------------------------------------
    @property
    def base_path(self):
        return os.path.splitext(self.rom_path)[0]

    def _load_battery(self):
        p = self.base_path + ".sav"
        if self.cart.battery and os.path.exists(p):
            with open(p, "rb") as f:
                d = f.read(8192)
            self.cart.prg_ram[:len(d)] = d

    def save_battery(self):
        if self.cart.battery:
            with open(self.base_path + ".sav", "wb") as f:
                f.write(bytes(self.cart.prg_ram))

    def save_state(self, path=None):
        path = path or self.base_path + ".state"
        st = {
            "cpu": {k: getattr(self.cpu, k) for k in CPU_KEYS},
            "ppu": {k: getattr(self.ppu, k) for k in PPU_KEYS},
            "apu": {k: getattr(self.apu, k) for k in APU_KEYS},
            "chans": {n: dict(getattr(self.apu, n).__dict__) for n in ("p1", "p2", "tri", "noi", "dmc")},
            "ram": bytes(self.bus.ram), "vram": bytes(self.ppu.vram),
            "oam": bytes(self.ppu.oam), "palette": bytes(self.ppu.palette),
            "mapper": self.mapper.get_state(), "dot": self.dot,
        }
        with open(path, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".state"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            st = pickle.load(f)
        for k, v in st["cpu"].items():
            setattr(self.cpu, k, v)
        for k, v in st["ppu"].items():
            setattr(self.ppu, k, v)
        for k, v in st["apu"].items():
            setattr(self.apu, k, v)
        for n, d in st["chans"].items():
            getattr(self.apu, n).__dict__.update(d)
        self.bus.ram[:] = st["ram"]
        self.ppu.vram[:] = st["vram"]
        self.ppu.oam[:] = st["oam"]
        self.ppu.palette[:] = st["palette"]
        self.mapper.set_state(st["mapper"])
        self.ppu.pend = []
        self.ppu._co_list = self.ppu._nt_list = None
        self.ppu._co_tabs = []
        self.ppu._nt_tabs = []
        self.dot = st["dot"]
        self.apu_mark = self.cpu.cycles
        return True

    # ---- control ---------------------------------------------------------------
    def reset(self):
        self.cpu.reset()
        self.ppu.ctrl = self.ppu.mask = 0
        self.apu.write(0x4015, 0)

    def set_buttons(self, pad, buttons):
        self.pads[pad].buttons = buttons

    # ---- emulation -------------------------------------------------------------
    def _run(self, target):
        # CPU.step() inlined into the dispatch loop (saves a call per instruction)
        cpu = self.cpu
        dot = self.dot
        table = cpu.table
        prg = cpu.prg
        prg_off = cpu.prg_off
        read = cpu.read
        mapper = cpu.mapper
        apu = cpu.apu
        step = cpu.step
        while dot < target:
            if cpu.nmi_pending or (not cpu.i and (mapper.irq or apu.irq)):
                dot += step() * 3  # interrupt entry handled by the slow path
                continue
            pc = cpu.pc
            if pc >= 0x8000:
                opc = prg[prg_off[(pc >> 13) & 3] + (pc & 0x1FFF)]
            else:
                opc = read(pc)
            cpu.pc = (pc + 1) & 0xFFFF
            fn, mode, cyc, pg = table[opc]
            if pg:
                cpu.crossed = 0
                fn(mode())
                cyc += cpu.crossed
            else:
                fn(mode())
            if cpu.extra:
                cyc += cpu.extra
                cpu.extra = 0
            cpu.cycles += cyc
            dot += cyc * 3
        self.dot = dot

    def run_frame(self, draw=True):
        ppu, cpu, apu, mapper = self.ppu, self.cpu, self.apu, self.mapper
        run = self._run
        # Absolute dot targets per scanline. `self.dot` is expressed as an
        # absolute position on the current scanline; each call advances up to
        # the given absolute target, and the end-of-line subtract rolls the
        # position back into [0, 341) (the residual is one scanline's worth of
        # accumulated overshoot from long instructions such as DMA).
        for sl in range(262):
            if sl < 240:
                if sl == 0:
                    ppu.evaluate_sprites()
                hit = ppu.render_scanline(sl, draw)
                if hit >= 0:
                    run(hit)
                    ppu.status |= 0x40
                    ppu.s0_done = True
                run(256)
                rendering = ppu.mask & 0x18
                if rendering:
                    ppu.inc_y_copy_x()
                run(260)
                if rendering:
                    mapper.scanline()
                run(341)
            elif sl == 240:
                ppu.flush()
                run(341)
            elif sl == 241:
                run(1)
                ppu.status |= 0x80
                if ppu.ctrl & 0x80:
                    cpu.nmi_pending = True
                run(341)
            elif sl < 261:
                run(341)
            else:  # pre-render line
                ppu.status &= 0x1F
                ppu.s0_done = False
                run(260)
                if ppu.mask & 0x18:
                    mapper.scanline()
                    ppu.v = ppu.t
                run(341)
            self.dot -= 341
            q = QUARTER_LINES.get(sl)
            if q is not None:
                apu.run_quarter(q, cpu.cycles - self.apu_mark)
                self.apu_mark = cpu.cycles
        self.frames += 1

    def get_rgba(self):
        return PALETTE_RGBA[self.ppu.frame_idx]


# --------------------------------------------------------------------------
# Arcade front-end
# --------------------------------------------------------------------------
VERT = """
#version 330
uniform vec4 rect;
in vec2 in_vert;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_vert * rect.xy + rect.zw, 0.0, 1.0);
    uv = vec2(in_uv.x, 1.0 - in_uv.y);
}
"""
FRAG = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() { fragColor = texture(tex, uv); }
"""

HELP = """Controls
  Menu bar at the top: File / Emulation / Video / Audio / Help
  Arrows / WASD ... D-pad          X / K ... A        Z / J ... B
  Enter ........... Start          Backspace / R-Shift ... Select
  Tab (hold) ...... fast-forward   P ... pause        R ... reset
  M ............... mute           F ... smooth/pixel filter
  F5 / F7 ......... save / load state     F11 ... fullscreen
  F12 ............. screenshot      Ctrl+O ... open ROM      Esc ... quit"""


def _keymap():
    if arcade is None:
        return {}
    k = arcade.key
    return {
        k.X: BTN_A, k.K: BTN_A, k.Z: BTN_B, k.J: BTN_B,
        k.RETURN: BTN_START, k.BACKSPACE: BTN_SELECT, k.RSHIFT: BTN_SELECT,
        k.UP: BTN_UP, k.DOWN: BTN_DOWN, k.LEFT: BTN_LEFT, k.RIGHT: BTN_RIGHT,
        k.W: BTN_UP, k.S: BTN_DOWN, k.A: BTN_LEFT, k.D: BTN_RIGHT,
    }


VERT_RECT = """
#version 330
uniform vec4 rect;
in vec2 in_vert;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_vert * rect.xy + rect.zw, 0.0, 1.0);
    uv = in_uv;
}
"""
# `zero` is always 0.0; it only keeps `uv` (and so in_uv) from being optimised away
FRAG_RECT = """
#version 330
uniform vec4 color;
uniform float zero;
in vec2 uv;
out vec4 fragColor;
void main() { fragColor = vec4(color.rgb + vec3(uv, 0.0) * zero, color.a); }
"""

# ---- menu bar look ---------------------------------------------------------
MENU_H = 24      # height of the bar (px)
ITEM_H = 23      # height of a dropdown row
SEP_H = 9        # height of a separator row
PAD_X = 12
CHECK_W = 22     # room on the left of every dropdown row for the check mark
C_BAR = (0.118, 0.118, 0.137, 1.0)
C_HOT = (0.235, 0.255, 0.330, 1.0)
C_DROP = (0.157, 0.157, 0.180, 1.0)
C_EDGE = (0.300, 0.300, 0.340, 1.0)
C_ACCENT = (0.35, 0.65, 1.0, 1.0)
C_TEXT = (232, 232, 238, 255)
C_DIM = (150, 150, 162, 255)

HELP_ROWS = [
    ("Arrows / WASD", "D-pad"), ("X / K", "A button"), ("Z / J", "B button"),
    ("Enter", "Start"), ("Backspace / R-Shift", "Select"),
    ("Tab (hold)", "Fast-forward"), ("\\", "Reverse / rewind"), ("+ / -", "Adjust speed 0.1x–10x"),
    ("P", "Pause / resume"), ("R", "Reset"),
    ("M", "Mute"), ("F", "Smooth / pixel filter"), ("F5 / F7", "Save / load state"),
    ("F11", "Fullscreen"), ("F12", "Screenshot"), ("Ctrl+O", "Open ROM"),
    ("Esc", "Close menu / quit"),
]
ABOUT_ROWS = [
    ("PyNES", "NES emulator in Python + Arcade"),
    ("CPU / PPU / APU", "6502, 2C02, pulse/triangle/noise/DMC"),
    ("Mappers", "0, 1, 2, 3, 4, 7, 66"),
]


class NESWindow(arcade.Window if arcade else object):
    def __init__(self, nes, scale=3, audio=True, aspect="square", max_catchup=6,
                 vsync=True, audio_latency=0.025, audio_backend="auto"):
        self.nes = nes
        self.aspect = aspect
        self.par = 8.0 / 7.0 if aspect == "ntsc" else 1.0
        self.max_catchup = max_catchup
        self.title_base = "PyNES" + (" - " + os.path.basename(nes.rom_path) if nes else "")
        super().__init__(int(256 * scale * self.par), 240 * scale + MENU_H, self.title_base,
                         resizable=True, update_rate=1 / 120, vsync=vsync)
        self.use_vsync = vsync
        self.keymap = _keymap()
        self.keys = 0
        self.paused = False
        self.muted = False
        self.turbo = False
        self.turbo_hold = False
        self.turbo_latch = False
        self.smooth = False
        self.volume = 1.0
        self.frame_ready = False
        self.acc = 0.0
        self.last = time.perf_counter()
        self.fps_t = self.last
        self.fps_n = 0
        self.slow_secs = 0
        self.fill_avg = 0.0
        self.audio_target = max(0.010, min(0.100, float(audio_latency)))
        self.speed_multiplier = 1.0
        self.reverse_speed = False
        self.rewind_history = deque(maxlen=600)
        self.rewind_acc = 0.0
        self.ui_hover_t = {}
        self.ui_anim_last = time.perf_counter()
        self.ui_frame_dt = 0.0
        self.ui_modal_close = None

        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.program["tex"] = 0
        self.rect_prog = self.ctx.program(vertex_shader=VERT_RECT, fragment_shader=FRAG_RECT)
        self.rect_prog["zero"] = 0.0
        self.texture = self.ctx.texture((256, 240), components=4, data=bytes(256 * 240 * 4),
                                        filter=(self.ctx.NEAREST, self.ctx.NEAREST))
        self.quad = gl_geometry.quad_2d_fs()

        self.audio_enabled = bool(audio)
        self.audio_backend = audio_backend
        self.audio = None
        if audio:
            self.audio = make_audio(44100, self.audio_target, audio_backend)
        self.ui_sync_audio()

        self.pad = None
        try:
            import pyglet.input
            pads = pyglet.input.get_controllers()
            if pads:
                self.pad = pads[0]
                self.pad.open()
                print("Gamepad:", getattr(self.pad, "name", "controller"))
        except Exception:
            self.pad = None

        # ---- UI state --------------------------------------------------------
        self.ui_size = self.get_size()
        self._opening = False
        self.ui_ptr = (-1, -1)
        self.ui_open = None          # index of the open top-level menu
        self.ui_overlay = None       # ("Title", rows) dialog or None
        self.ui_toast = None         # (text, expiry)
        self.ui_texts = {}
        self.ui_menus = None
        self.ui_title_x = []
        self.ui_drop_w = []
        self.ui_reflow()
        print(HELP)

    # =====================================================================
    # menu definition
    # =====================================================================
    def ui_build_menus(self):
        def it(label, fn, key="", check=None):
            return {"label": label, "fn": fn, "key": key, "check": check}

        def size_item(s):
            return it(f"{s}x  ({int(256 * s * self.par)} x {240 * s})",
                      lambda s=s: self.act_scale(s), check=lambda s=s: self.ui_cur_scale() == s)

        file_m = [
            it("Open ROM...", self.act_open, "Ctrl+O"), None,
            it("Save State", self.act_save_state, "F5"),
            it("Load State", self.act_load_state, "F7"), None,
            it("Screenshot", self.act_screenshot, "F12"), None,
            it("Exit", self.close, "Esc"),
        ]
        emu_m = [
            it("Pause", self.act_pause, "P", check=lambda: self.paused),
            it("Reset", self.act_reset, "R"),
            it("Fast Forward", self.act_turbo, "Tab", check=lambda: self.turbo),
            it("Reverse / Rewind", self.act_reverse, "Backslash", check=lambda: self.reverse_speed),
            None,
            *[it(f"Emulation Speed {v:g}x", lambda v=v: self.act_speed(v),
                 check=lambda v=v: abs(self.speed_multiplier - v) < 1e-6)
              for v in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)],
        ]
        vid_m = [size_item(s) for s in (1, 2, 3, 4, 5, 6)] + [
            None,
            it("Fullscreen", self.act_fullscreen, "F11", check=lambda: self.fullscreen),
            it("Smooth Scaling", self.act_smooth, "F", check=lambda: self.smooth),
            it("VSync", self.act_vsync, check=lambda: self.use_vsync),
            None,
            it("Square Pixels", lambda: self.act_aspect("square"), check=lambda: self.aspect == "square"),
            it("NTSC 8:7 Pixels", lambda: self.act_aspect("ntsc"), check=lambda: self.aspect == "ntsc"),
        ]
        aud_m = [it("Mute", self.act_mute, "M", check=lambda: self.muted), None] + [
            it(f"Volume {int(v * 100)}%", lambda v=v: self.act_volume(v),
               check=lambda v=v: abs(self.volume - v) < 1e-6) for v in (1.0, 0.75, 0.5, 0.25)]
        aud_m += [None] + [
            it(f"Audio Latency {int(ms)} ms", lambda ms=ms: self.act_audio_latency(ms),
               check=lambda ms=ms: abs(self.audio_target * 1000 - ms) < 0.5)
            for ms in (15, 20, 25, 35, 50, 75)]
        help_m = [
            it("Controls...", lambda: self.ui_show("Controls", HELP_ROWS)),
            it("About", lambda: self.ui_show("About", ABOUT_ROWS)),
        ]
        return [("File", file_m), ("Emulation", emu_m), ("Video", vid_m),
                ("Audio", aud_m), ("Help", help_m)]

    def ui_reflow(self):
        """Rebuild menu item tables and hit-boxes from scratch.  Must be called
        whenever label widths can change (aspect ratio, window scale, fullscreen)."""
        self.ui_menus = self.ui_build_menus()
        self.ui_title_x = []
        x = 6
        for name, items in self.ui_menus:
            tw = self.ui_text(name, C_TEXT).content_width
            self.ui_title_x.append((x, x + tw + 2 * PAD_X))
            x += tw + 2 * PAD_X
        self.ui_drop_w = []
        for name, items in self.ui_menus:
            lw = max([self.ui_text(i["label"], C_TEXT).content_width for i in items if i] or [0])
            kw = max([self.ui_text(i["key"], C_DIM).content_width for i in items if i and i["key"]] or [0])
            self.ui_drop_w.append(CHECK_W + lw + (36 + kw if kw else 0) + PAD_X)

    # =====================================================================
    # actions
    # =====================================================================
    def ui_notify(self, msg):
        print(msg)
        self.ui_toast = (msg, time.perf_counter() + 2.5)

    def ui_show(self, title, rows):
        self.ui_overlay = (title, rows)

    def ui_cur_scale(self):
        if self.fullscreen:
            return None
        ww, wh = self.get_size()
        for s in range(1, 9):
            if ww == int(256 * s * self.par) and wh == 240 * s + MENU_H:
                return s
        return None

    def ui_sync_audio(self):
        """Route APU output to the audio device (or nowhere when muted / fast-forward)."""
        if self.nes is None:
            return
        live = self.audio is not None and not self.muted and not self.turbo and not self.reverse_speed
        self.nes.apu.sink = self.audio.push if live else None
        if self.audio is not None and not live:
            self.audio.clear()
        self.nes.apu.volume = 0.7 * self.volume

    def act_open(self):
        # Ignore re-entry while a dialog is already up; Arcade calls this from the
        # key handler, and the spawned subprocess keeps the main loop responsive.
        if getattr(self, "_opening", False):
            return
        self._opening = True
        try:
            path = pick_rom()
            # Release any keys the user was holding; the dialog swallowed the
            # key-release events, so a stuck button would otherwise persist.
            self.keys = 0
            self.last = time.perf_counter()
            if path:
                self.load_rom(path)
        except Exception as e:
            # Never let a dialog failure tear down the emulator window.
            self.ui_notify(f"Open ROM failed: {e}")
        finally:
            self._opening = False

    def load_rom(self, path):
        try:
            new = NES(path)
        except Exception as e:
            self.ui_notify(f"Cannot load ROM: {e}")
            return False
        old = self.nes
        if old is not None:
            try:
                old.save_battery()
            except Exception:
                pass
            old.apu.sink = None
        self.nes = new
        self.rewind_history.clear()
        self.title_base = "PyNES - " + os.path.basename(path)
        self.set_caption(self.title_base)
        self.paused = False
        self.acc = 0.0
        self.last = time.perf_counter()
        self.fill_avg = self.audio_target
        self.ui_sync_audio()
        print(f"Loaded {os.path.basename(path)}  mapper={new.cart.mapper_id}  "
              f"PRG={len(new.cart.prg) // 1024}K  "
              f"CHR={'RAM' if new.cart.chr_ram else str(len(new.cart.chr) // 1024) + 'K'}")
        return True

    def act_save_state(self):
        if self.nes:
            self.nes.save_state()
            self.ui_notify("State saved")

    def act_load_state(self):
        if self.nes:
            self.ui_notify("State loaded" if self.nes.load_state() else "No saved state")
            self.frame_ready = True

    def act_screenshot(self):
        if not self.nes:
            return
        try:
            from PIL import Image
            path = self.nes.base_path + f"_{int(time.time())}.png"
            Image.fromarray(self.nes.get_rgba()).save(path)
            self.ui_notify("Screenshot: " + os.path.basename(path))
        except Exception as e:
            self.ui_notify(f"Screenshot failed: {e}")

    def act_pause(self):
        self.paused = not self.paused
        self.last = time.perf_counter()
        self.set_caption(self.title_base + (" [paused]" if self.paused else ""))

    def act_reset(self):
        if self.nes:
            self.nes.reset()
            self.rewind_history.clear()
            self.rewind_acc = 0.0
            self.ui_notify("Reset")

    def act_turbo(self):
        self.turbo_latch = not self.turbo_latch
        self.ui_update_turbo()

    def act_speed(self, multiplier):
        self.speed_multiplier = max(0.1, min(10.0, float(multiplier)))
        self.acc = 0.0
        self.last = time.perf_counter()
        self.ui_notify(f"Emulation speed: {self.speed_multiplier:g}x")
        self.ui_reflow()

    def act_reverse(self):
        self.reverse_speed = not self.reverse_speed
        self.rewind_acc = 0.0
        self.acc = 0.0
        self.last = time.perf_counter()
        self.ui_sync_audio()
        self.ui_notify("Rewind enabled" if self.reverse_speed else "Forward playback")

    def _capture_rewind_state(self):
        n = self.nes
        if n is None:
            return None
        return {
            "cpu": {k: getattr(n.cpu, k) for k in CPU_KEYS},
            "ppu": {k: getattr(n.ppu, k) for k in PPU_KEYS},
            "apu": {k: getattr(n.apu, k) for k in APU_KEYS},
            "chans": {name: dict(getattr(n.apu, name).__dict__) for name in ("p1", "p2", "tri", "noi", "dmc")},
            "ram": bytes(n.bus.ram), "vram": bytes(n.ppu.vram), "oam": bytes(n.ppu.oam),
            "palette": bytes(n.ppu.palette), "mapper": n.mapper.get_state(), "dot": n.dot,
        }

    def _restore_rewind_state(self, st):
        n = self.nes
        if n is None or st is None:
            return False
        for key, value in st["cpu"].items(): setattr(n.cpu, key, value)
        for key, value in st["ppu"].items(): setattr(n.ppu, key, value)
        for key, value in st["apu"].items(): setattr(n.apu, key, value)
        for name, values in st["chans"].items(): getattr(n.apu, name).__dict__.update(values)
        n.bus.ram[:] = st["ram"]; n.ppu.vram[:] = st["vram"]
        n.ppu.oam[:] = st["oam"]; n.ppu.palette[:] = st["palette"]
        n.mapper.set_state(st["mapper"])
        n.ppu.pend = []; n.ppu._co_list = n.ppu._nt_list = None; n.ppu._co_tabs = []; n.ppu._nt_tabs = []
        n.dot = st["dot"]; n.apu_mark = n.cpu.cycles
        self.frame_ready = True
        return True

    def act_audio_latency(self, milliseconds):
        """Change the target queue and reopen the output with matching buffers."""
        target = max(10, min(100, int(milliseconds))) / 1000.0
        if abs(target - self.audio_target) < 0.0005:
            return
        old = self.audio
        self.audio = None
        if self.nes is not None:
            self.nes.apu.sink = None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        self.audio_target = target
        self.fill_avg = target
        self.audio = make_audio(44100, target, self.audio_backend) if self.audio_enabled else None
        self.ui_sync_audio()
        self.ui_reflow()
        self.ui_notify(f"Audio latency target: {milliseconds} ms")


    def ui_update_turbo(self):
        was = self.turbo
        self.turbo = self.turbo_hold or self.turbo_latch
        if was != self.turbo:
            self.ui_sync_audio()
            self.acc = 0.0
            self.last = time.perf_counter()

    def act_fullscreen(self):
        self.set_fullscreen(not self.fullscreen)

    def act_smooth(self):
        self.smooth = not self.smooth
        f = self.ctx.LINEAR if self.smooth else self.ctx.NEAREST
        self.texture.filter = (f, f)

    def act_vsync(self, value=None, auto=False):
        if value is None:
            self.use_vsync = not self.use_vsync
        else:
            self.use_vsync = bool(value)
        try:
            self.set_vsync(self.use_vsync)
        except Exception:
            pass
        self.slow_secs = 0
        if auto:
            self.ui_notify("VSync turned off (emulation could not keep 60 FPS with it)")

    def act_aspect(self, mode):
        s = self.ui_cur_scale()
        self.aspect = mode
        self.par = 8.0 / 7.0 if mode == "ntsc" else 1.0
        self.ui_reflow()
        if s:
            self.act_scale(s)

    def act_scale(self, s):
        if not self.fullscreen:
            self.set_size(int(256 * s * self.par), 240 * s + MENU_H)

    def act_mute(self):
        self.muted = not self.muted
        self.ui_sync_audio()

    def act_volume(self, v):
        self.volume = v
        if self.nes:
            self.nes.apu.volume = 0.7 * v

    # =====================================================================
    # input
    # =====================================================================
    def _poll_pad(self):
        p = self.pad
        if p is None:
            return 0
        try:
            b = 0
            if p.b or p.y:
                b |= BTN_A
            if p.a or p.x:
                b |= BTN_B
            if p.back:
                b |= BTN_SELECT
            if p.start:
                b |= BTN_START
            if p.dpup:
                b |= BTN_UP
            if p.dpdown:
                b |= BTN_DOWN
            if p.dpleft or p.leftx < -0.5:
                b |= BTN_LEFT
            if p.dpright or p.leftx > 0.5:
                b |= BTN_RIGHT
            return b
        except Exception:
            self.pad = None
            return 0

    def on_key_press(self, symbol, modifiers):
        k = arcade.key
        if self.ui_overlay is not None:
            self.ui_overlay = None
            self.ui_modal_close = None
            return
        bit = self.keymap.get(symbol)
        if symbol == k.O and (modifiers & (k.MOD_CTRL | getattr(k, 'MOD_COMMAND', 0))):
            self.act_open()
        elif bit:
            self.keys |= bit
        elif symbol == k.ESCAPE:
            if self.ui_open is not None:
                self.ui_open = None
            else:
                self.close()
        elif symbol == k.TAB:
            self.turbo_hold = True
            self.ui_update_turbo()
        elif symbol in (getattr(k, "PLUS", -1), getattr(k, "NUM_ADD", -2), getattr(k, "EQUAL", -3)):
            self.act_speed(min(10.0, self.speed_multiplier + 0.1))
        elif symbol in (getattr(k, "MINUS", -4), getattr(k, "NUM_SUBTRACT", -5)):
            self.act_speed(max(0.1, self.speed_multiplier - 0.1))
        elif symbol == getattr(k, "BACKSLASH", -999):
            self.act_reverse()
        elif symbol == k.P:
            self.act_pause()
        elif symbol == k.R:
            self.act_reset()
        elif symbol == k.M:
            self.act_mute()
        elif symbol == k.F:
            self.act_smooth()
        elif symbol == k.F5:
            self.act_save_state()
        elif symbol == k.F7:
            self.act_load_state()
        elif symbol == k.F11:
            self.act_fullscreen()
        elif symbol == k.F12:
            self.act_screenshot()

    def on_key_release(self, symbol, modifiers):
        bit = self.keymap.get(symbol)
        if bit:
            self.keys &= ~bit
        elif symbol == arcade.key.TAB:
            self.turbo_hold = False
            self.ui_update_turbo()

    # ---- mouse / menu hit testing -----------------------------------------------
    def ui_bar_visible(self, wh):
        return (not self.fullscreen) or self.ui_open is not None or self.ui_ptr[1] >= wh - MENU_H

    def ui_hit_title(self, x, y, wh):
        if y >= wh - MENU_H and self.ui_bar_visible(wh):
            for i, (x0, x1) in enumerate(self.ui_title_x):
                if x0 <= x < x1:
                    return i
        return None

    def ui_rows(self, i, wh):
        """[(item, y_top, y_bottom)] of dropdown i, top to bottom."""
        y = wh - MENU_H
        rows = []
        for item in self.ui_menus[i][1]:
            h = SEP_H if item is None else ITEM_H
            rows.append((item, y, y - h))
            y -= h
        return rows, y

    def ui_hit_item(self, x, y, wh):
        i = self.ui_open
        if i is None:
            return None
        x0 = self.ui_title_x[i][0]
        if not (x0 <= x < x0 + self.ui_drop_w[i]):
            return None
        for item, yt, yb in self.ui_rows(i, wh)[0]:
            if item is not None and yb <= y < yt:
                return item
        return None

    def on_mouse_motion(self, x, y, dx, dy):
        self.ui_ptr = (x, y)
        if self.ui_open is not None:
            t = self.ui_hit_title(x, y, self.get_size()[1])
            if t is not None:
                self.ui_open = t

    def on_mouse_leave(self, x, y):
        self.ui_ptr = (-1, -1)

    def on_mouse_press(self, x, y, button, modifiers):
        self.ui_ptr = (x, y)
        if self.ui_overlay is not None:
            if self.ui_modal_close and self.ui_modal_close[0] <= x <= self.ui_modal_close[2] and self.ui_modal_close[1] <= y <= self.ui_modal_close[3]:
                self.ui_overlay = None
                self.ui_modal_close = None
            return
        wh = self.get_size()[1]
        if button != arcade.MOUSE_BUTTON_LEFT:
            self.ui_open = None
            return
        t = self.ui_hit_title(x, y, wh)
        if t is not None:
            self.ui_open = None if self.ui_open == t else t
            return
        if self.ui_open is not None:
            item = self.ui_hit_item(x, y, wh)
            self.ui_open = None
            if item is not None and item["fn"]:
                try:
                    item["fn"]()
                except Exception as e:
                    self.ui_notify(f"{item['label']} failed: {e}")

    # =====================================================================
    # main loop
    # =====================================================================
    def on_update(self, dt):
        now = time.perf_counter()
        elapsed = min(now - self.last, 0.1)
        self.last = now
        if self.paused or self.nes is None:
            self.acc = 0.0
            return
        speed = 4.0 if self.turbo else self.speed_multiplier
        if self.reverse_speed:
            self.rewind_acc += elapsed * speed
            frame_time = 1.0 / FRAME_RATE
            steps = min(30, int(self.rewind_acc / frame_time))
            if steps:
                self.rewind_acc -= steps * frame_time
                for _ in range(steps):
                    if self.rewind_history:
                        # Each entry is the machine state immediately before a
                        # frame. Restore the newest entry first (one frame back),
                        # then discard it so subsequent steps continue backward.
                        self._restore_rewind_state(self.rewind_history.pop())
                    else:
                        break
            self.nes.apu.sink = None
            self.fps_n += steps
            return
        self.rewind_acc = 0.0
        apu = self.nes.apu
        if self.audio is not None and apu.sink is not None:
            # Slave the emulation speed to the audio clock: the buffered audio is
            # kept near the target by speeding up / slowing down by at most 4 %.
            # This tighter feedback helps hold the selected latency without drift.
            fill = self.audio.buffered_seconds()
            # A responsive low-pass controller avoids audible speed hunting.
            self.fill_avg += (fill - self.fill_avg) * 0.12
            err = self.fill_avg - self.audio_target
            speed *= 1.0 - max(-0.04, min(0.04, err * 1.0))
        self.acc += elapsed * speed
        frame_time = 1.0 / FRAME_RATE
        n = 0
        if self.acc >= frame_time:
            self.nes.set_buttons(0, self.keys | self._poll_pad())
            limit = self.max_catchup * (3 if self.turbo else 1)
            while self.acc >= frame_time:
                self.acc -= frame_time
                n += 1
                more = self.acc >= frame_time and n < limit
                if not self.turbo:
                    self.rewind_history.append(self._capture_rewind_state())
                self.nes.run_frame(draw=not more)
                if not more:
                    self.frame_ready = True
                    break
            if self.acc >= frame_time:  # cannot keep up: drop the backlog
                self.acc = 0.0
        self.fps_n += n
        if now - self.fps_t >= 1.0:
            fps = self.fps_n / (now - self.fps_t)
            self.set_caption(f"{self.title_base} - {fps:.1f} FPS")
            # vsync quantises frame times to the refresh rate: if the machine is
            # too slow for that, switch it off instead of dropping to 30 FPS
            if self.use_vsync and not self.turbo and fps < 54.0:
                self.slow_secs += 1
                if self.slow_secs >= 3:
                    self.act_vsync(False, auto=True)
            else:
                self.slow_secs = 0
            self.fps_t, self.fps_n = now, 0

    # =====================================================================
    # drawing
    # =====================================================================
    def ui_rect(self, x0, y0, x1, y1, color):
        ww, wh = self.ui_size
        self.rect_prog["rect"] = ((x1 - x0) / ww, (y1 - y0) / wh,
                                  (x0 + x1) / ww - 1.0, (y0 + y1) / wh - 1.0)
        self.rect_prog["color"] = color
        self.quad.render(self.rect_prog)

    def ui_text(self, s, color, size=11, anchor_x="left"):
        # arcade.Text expects byte-range channels; our GL palette also uses
        # normalized 0..1 floats. Normalize here so Help/About accent labels
        # cannot throw a color validation error during dialog rendering.
        rgba = tuple(color)
        if rgba and max(rgba) <= 1.0:
            rgba = tuple(round(max(0, min(1, c)) * 255) for c in rgba)
        else:
            rgba = tuple(int(max(0, min(255, c))) for c in rgba)
        if len(rgba) == 3:
            rgba = (*rgba, 255)
        key = (s, rgba, size, anchor_x)
        t = self.ui_texts.get(key)
        if t is None:
            t = arcade.Text(s, 0, 0, rgba, size, anchor_x=anchor_x, anchor_y="center")
            self.ui_texts[key] = t
        return t

    def ui_tween(self, key, target, dt, duration=0.20):
        """Frame-rate independent smoothstep tween from current to target."""
        state = self.ui_hover_t.get(key)
        if state is None:
            state = [0.0, 0.0, 0.0, float(target)]
        value, start, elapsed, old_target = state
        if float(target) != old_target:
            start, elapsed, old_target = value, 0.0, float(target)
        elapsed = min(duration, elapsed + max(0.0, dt))
        t = min(1.0, elapsed / max(0.001, duration))
        eased = t * t * (3.0 - 2.0 * t)
        value = start + (old_target - start) * eased
        self.ui_hover_t[key] = [value, start, elapsed, old_target]
        return value

    @staticmethod
    def ui_put(t, x, y):
        if t.x != x or t.y != y:
            t.x = x
            t.y = y
        t.draw()

    def ui_draw_menu(self, ww, wh):
        now = time.perf_counter()
        self.ui_frame_dt = min(0.05, max(0.0, now - self.ui_anim_last))
        self.ui_anim_last = now
        if not self.ui_bar_visible(wh):
            return
        px, py = self.ui_ptr
        self.ui_rect(0, wh - MENU_H, ww, wh, C_BAR)
        anim_dt = self.ui_frame_dt
        for i, (name, _) in enumerate(self.ui_menus):
            x0, x1 = self.ui_title_x[i]
            hovered = self.ui_open == i or (x0 <= px < x1 and py >= wh - MENU_H)
            amount = self.ui_tween(("title", i), 1.0 if hovered else 0.0, anim_dt, 0.18)
            if amount > 0.01:
                self.ui_rect(x0, wh - MENU_H, x1, wh,
                             (C_BAR[0] + (C_HOT[0] - C_BAR[0]) * amount,
                              C_BAR[1] + (C_HOT[1] - C_BAR[1]) * amount,
                              C_BAR[2] + (C_HOT[2] - C_BAR[2]) * amount, 1.0))
            self.ui_put(self.ui_text(name, C_TEXT), x0 + PAD_X, wh - MENU_H / 2 + amount)
        i = self.ui_open
        if i is None:
            return
        x0 = self.ui_title_x[i][0]
        w = self.ui_drop_w[i]
        rows, y_end = self.ui_rows(i, wh)
        self.ui_rect(x0 - 1, y_end - 1, x0 + w + 1, wh - MENU_H, C_EDGE)
        self.ui_rect(x0, y_end, x0 + w, wh - MENU_H, C_DROP)
        for row_index, (item, yt, yb) in enumerate(rows):
            if item is None:
                self.ui_rect(x0 + 8, (yt + yb) / 2, x0 + w - 8, (yt + yb) / 2 + 1, C_EDGE)
                continue
            hovered = x0 <= px < x0 + w and yb <= py < yt
            amount = self.ui_tween(("item", i, row_index), 1.0 if hovered else 0.0, anim_dt, 0.22)
            if amount > 0.01:
                inset = (1.0 - amount) * 5.0
                self.ui_rect(x0 + inset, yb + 1, x0 + w - inset, yt - 1,
                             (C_DROP[0] + (C_HOT[0] - C_DROP[0]) * amount,
                              C_DROP[1] + (C_HOT[1] - C_DROP[1]) * amount,
                              C_DROP[2] + (C_HOT[2] - C_DROP[2]) * amount, 1.0))
                if amount > 0.12:
                    self.ui_rect(x0 + 2, yb + 3, x0 + 4, yt - 3,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], amount * 0.95))
            cy = (yt + yb) / 2 + amount * 2.0
            if item["check"] is not None and item["check"]():
                self.ui_rect(x0 + 8, cy - 4, x0 + 16, cy + 4, C_ACCENT)
            self.ui_put(self.ui_text(item["label"], C_TEXT), x0 + CHECK_W, cy)
            if item["key"]:
                self.ui_put(self.ui_text(item["key"], C_DIM, anchor_x="right"), x0 + w - PAD_X, cy)

    def ui_draw_overlays(self, ww, wh):
        now = time.perf_counter()
        if self.ui_overlay is not None:
            title, rows = self.ui_overlay
            lw = max(self.ui_text(a, C_ACCENT).content_width for a, _ in rows)
            rw = max(self.ui_text(b, C_TEXT).content_width for _, b in rows)
            w, rh = lw + rw + 64, 24
            h = rh * len(rows) + 64
            x0, y1 = (ww - w) / 2, (wh - MENU_H) / 2 + h / 2
            self.ui_rect(x0 - 1, y1 - h - 1, x0 + w + 1, y1 + 1, C_EDGE)
            self.ui_rect(x0, y1 - h, x0 + w, y1, C_DROP)
            self.ui_put(self.ui_text(title, C_TEXT, 14), x0 + 24, y1 - 24)
            bx0, by0, bx1, by1 = x0 + w - 94, y1 - 42, x0 + w - 18, y1 - 14
            self.ui_modal_close = (bx0, by0, bx1, by1)
            mx, my = self.ui_ptr
            close_hover = bx0 <= mx <= bx1 and by0 <= my <= by1
            close_amount = self.ui_tween(("modal-close", title), 1.0 if close_hover else 0.0,
                                         self.ui_frame_dt, 0.18)
            close_color = tuple(C_BAR[k] + (C_HOT[k] - C_BAR[k]) * close_amount
                                for k in range(3)) + (1.0,)
            self.ui_rect(bx0, by0, bx1, by1, close_color)
            self.ui_rect(bx0, by0, bx1, by0 + 1, C_ACCENT)
            self.ui_put(self.ui_text("Close", C_TEXT), (bx0 + bx1) / 2, (by0 + by1) / 2)
            for n, (a, b) in enumerate(rows):
                y = y1 - 56 - n * rh
                self.ui_put(self.ui_text(a, C_ACCENT), x0 + 24, y)
                self.ui_put(self.ui_text(b, C_TEXT), x0 + 24 + lw + 16, y)
        if self.ui_toast is not None:
            msg, until = self.ui_toast
            if now > until:
                self.ui_toast = None
            else:
                t = self.ui_text(msg, C_TEXT)
                y0 = 10
                self.ui_rect(8, y0 - 2, 8 + t.content_width + 20, y0 + 24, C_DROP)
                self.ui_put(t, 18, y0 + 11)

    def on_draw(self):
        self.clear()
        ww, wh = self.get_size()
        self.ui_size = (ww, wh)
        cam = getattr(self, "default_camera", None)
        if cam is not None:
            try:
                cam.use()
            except Exception:
                pass
        bar = 0 if self.fullscreen else MENU_H
        avail = max(1, wh - bar)
        if self.nes is not None:
            if self.frame_ready:
                self.texture.write(self.nes.get_rgba().tobytes())
                self.frame_ready = False
            dw, dh = 256 * self.par, 240.0
            s = min(ww / dw, avail / dh)
            self.program["rect"] = (dw * s / ww, dh * s / wh, 0.0, (avail - wh) / wh)
            self.texture.use(0)
            self.quad.render(self.program)
        else:
            t = self.ui_text("File > Open ROM...   (Ctrl+O)", C_DIM, 14, "center")
            self.ui_put(t, ww / 2, avail / 2)
        self.ui_draw_menu(ww, wh)
        self.ui_draw_overlays(ww, wh)

    def on_close(self):
        try:
            if self.nes is not None:
                self.nes.save_battery()
        except Exception:
            pass
        try:
            if self.audio is not None:
                self.audio.close()
                self.audio = None
        except Exception:
            pass
        super().on_close()


def pick_rom():
    """Show a native file dialog without touching the GL thread.

    Runs the tkinter dialog in a short-lived child process so that
    (a) creating a Tk root while Arcade's GL loop is running cannot abort
        the main process, and
    (b) a crash/hang inside Tk is contained to the child.
    Returns the chosen path, or None if the user cancelled / no backend.
    """
    import subprocess
    code = (
        "import sys\n"
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "except Exception:\n"
        "    sys.exit(0)\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "try:\n"
        "    root.attributes('-topmost', True)\n"
        "except Exception:\n"
        "    pass\n"
        "p = filedialog.askopenfilename(\n"
        "    title='Select NES ROM',\n"
        "    filetypes=[('NES ROMs', '*.nes'), ('All files', '*.*')])\n"
        "root.destroy()\n"
        "if p:\n"
        "    sys.stdout.write(p)\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except Exception as e:
        print(f"[open] file dialog failed: {e}", file=sys.stderr)
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    path = res.stdout.decode("utf-8", "replace").strip()
    return path or None

def bench(nes, frames):
    nes.run_frame(True)  # warm-up (fills caches)
    t0 = time.perf_counter()
    for i in range(frames):
        nes.run_frame(draw=True)
    dt = time.perf_counter() - t0
    fps = frames / dt
    print(f"{frames} frames in {dt:.2f}s -> {fps:.1f} FPS ({fps / FRAME_RATE * 100:.0f}% of real-time)")
    t0 = time.perf_counter()
    for i in range(frames):
        nes.run_frame(draw=False)
    dt = time.perf_counter() - t0
    print(f"frame-skipped (no drawing): {frames / dt:.1f} FPS")


def main():
    ap = argparse.ArgumentParser(description="PyNES - NES emulator (Python + Arcade)")
    ap.add_argument("rom", nargs="?", help="path to an .nes (iNES) ROM (or use File > Open ROM)")
    ap.add_argument("--scale", type=int, default=3, help="initial window scale (default 3)")
    ap.add_argument("--no-audio", action="store_true", help="disable sound (a bit faster)")
    ap.add_argument("--audio-backend", choices=("auto", "sounddevice", "pyglet"), default="auto",
                    help="audio output backend (sounddevice has the lowest latency)")
    ap.add_argument("--audio-latency", type=int, default=25, metavar="MS",
                    help="target audio buffer in ms (10-100; default 25)")
    ap.add_argument("--no-vsync", action="store_true", help="disable vsync (may tear)")
    ap.add_argument("--aspect", choices=("square", "ntsc"), default="square",
                    help="pixel aspect: square (256x240) or ntsc (8:7 pixels)")
    ap.add_argument("--max-frameskip", type=int, default=6,
                    help="max emulated frames per host update when running behind")
    ap.add_argument("--bench", type=int, metavar="N", help="headless benchmark of N frames")
    args = ap.parse_args()

    nes = None
    if args.rom:
        try:
            nes = NES(args.rom)
        except Exception as e:
            sys.exit(f"Failed to open {args.rom!r}: {e}")
        print(f"Loaded {os.path.basename(args.rom)}  mapper={nes.cart.mapper_id}  "
              f"PRG={len(nes.cart.prg) // 1024}K  "
              f"CHR={'RAM' if nes.cart.chr_ram else str(len(nes.cart.chr) // 1024) + 'K'}")
    if args.bench:
        if nes is None:
            ap.error("--bench needs a ROM")
        bench(nes, args.bench)
        return
    if arcade is None:
        sys.exit("The 'arcade' package is required for the GUI: pip install arcade")
    # the audio callback thread must be able to grab the GIL quickly while the
    # emulator is busy, otherwise the audio device underruns
    sys.setswitchinterval(0.001)
    NESWindow(nes, scale=args.scale, audio=not args.no_audio, aspect=args.aspect,
              max_catchup=args.max_frameskip, vsync=not args.no_vsync,
              audio_latency=max(10, min(100, args.audio_latency)) / 1000.0,
              audio_backend=args.audio_backend)
    arcade.run()


if __name__ == "__main__":
    main()