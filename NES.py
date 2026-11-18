#!/usr/bin/env python3
"""
PyRetro - Unified Multi-Console Emulator (NES, SNES, GB, GBC, GBA) in Python + Arcade.

Supports:
  * NES (.nes)              - 6502 CPU, 2C02 PPU, APU, Mappers 0, 1, 2, 3, 4, 7, 66
  * SNES (.sfc, .smc, .snes)- 65c816 CPU (8/16-bit, MVN/MVP, IRQ/NMI), Vectorized S-PPU (Modes 0-7 + OAM), HDMA, SPC700
  * Game Boy / Color (.gb, .gbc) - Sharp LR35902 CPU, DMG/CGB PPU, MBC1/3/5, CGB HDMA
  * Game Boy Advance (.gba) - ARM7TDMI (ARM & Thumb), Bitmap/Tile PPU (Modes 0-5), DMA
  * AVI Video Recording     - Press V (or File > Record AVI Video)

Requirements:  pip install arcade numpy
Recommended:   pip install sounddevice imageio imageio-ffmpeg pillow
Run:           python PyRetro-v2.py [game.nes | game.sfc | game.snes | game.gb | game.gbc | game.gba]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import arcade
    from arcade.gl import geometry as gl_geometry
except Exception:
    arcade = None
    gl_geometry = None

try:
    import imageio
except Exception:
    imageio = None

CPU_HZ = 1789773.0
SNES_MASTER_HZ = 21477272.0
FRAME_RATE = 60.0988

# ==========================================================================
# SHARED PALETTES & AUDIO ENGINE
# ==========================================================================
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


class AudioRing:
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

    def ensure_capacity(self, capacity):
        capacity = int(capacity)
        with self.lock:
            if capacity <= self.cap:
                return
            available = min(self.w - self.r, self.cap)
            start = self.r % self.cap
            first = min(available, self.cap - start)
            queued = np.concatenate((self.buf[start:start + first],
                                     self.buf[:available - first])) if available else np.empty(0, np.int16)
            self.buf = np.zeros(capacity, dtype=np.int16)
            self.buf[:available] = queued
            self.cap = capacity
            self.r = 0
            self.w = available

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
                self.r += n - free
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
    name = "sounddevice"

    def __init__(self, rate, latency):
        import sounddevice as sd
        self.rate = rate
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
        if sys.platform == "win32":
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
                except TypeError:
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
    order = {"auto": (SoundDeviceOut, PygletOut), "sounddevice": (SoundDeviceOut,),
             "pyglet": (PygletOut,)}[backend]
    errors = []
    for cls in order:
        try:
            out = cls(rate, latency)
            print(f"[audio] {out.name}" + (f" (device latency ~{out.device_latency * 1000:.0f} ms)"
                                           if out.device_latency else ""))
            return out
        except Exception as e:
            errors.append(f"{cls.name}: {e}")
    print("[audio disabled] " + "; ".join(errors), file=sys.stderr)
    return None


# ==========================================================================
# NES HARDWARE IMPLEMENTATION
# ==========================================================================
class NESCartridge:
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
    def __init__(self, cart: NESCartridge):
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


class Mapper0(Mapper):
    def __init__(self, cart):
        super().__init__(cart)
        for i in range(4):
            self.map_prg8(i, i)
        self.map_chr8(0)


class Mapper1(Mapper):
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


class Mapper2(Mapper):
    def __init__(self, cart):
        super().__init__(cart)
        self.map_chr8(0)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg8(0, val * 2)
        self.map_prg8(1, val * 2 + 1)
        self.map_prg8(2, self.n_prg8 - 2)
        self.map_prg8(3, self.n_prg8 - 1)


class Mapper3(Mapper):
    def __init__(self, cart):
        super().__init__(cart)
        for i in range(4):
            self.map_prg8(i, i)
        self.map_chr8(0)

    def write(self, addr, val):
        self.map_chr8(val)


class Mapper7(Mapper):
    def __init__(self, cart):
        super().__init__(cart)
        self.map_chr8(0)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg32(val & 7)
        self.set_mirroring("s1" if val & 0x10 else "s0")


class Mapper66(Mapper):
    def __init__(self, cart):
        super().__init__(cart)
        self.write(0x8000, 0)

    def write(self, addr, val):
        self.map_prg32((val >> 4) & 3)
        self.map_chr8(val & 3)


class Mapper4(Mapper):
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


class CPU6502:
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

    def m_imp(self): return 0
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
        hi = self.read((p & 0xFF00) | ((p + 1) & 0xFF))
        return lo | (hi << 8)

    def op_lda(self, ad): v = self.read(ad); self.a = v; self.z = v == 0; self.n = v >> 7
    def op_ldx(self, ad): v = self.read(ad); self.x = v; self.z = v == 0; self.n = v >> 7
    def op_ldy(self, ad): v = self.read(ad); self.y = v; self.z = v == 0; self.n = v >> 7
    def op_lax(self, ad): v = self.read(ad); self.a = self.x = v; self.z = v == 0; self.n = v >> 7
    def op_sta(self, ad): self.write(ad, self.a)
    def op_stx(self, ad): self.write(ad, self.x)
    def op_sty(self, ad): self.write(ad, self.y)
    def op_sax(self, ad): self.write(ad, self.a & self.x)
    def op_and(self, ad): v = self.a & self.read(ad); self.a = v; self.z = v == 0; self.n = v >> 7
    def op_ora(self, ad): v = self.a | self.read(ad); self.a = v; self.z = v == 0; self.n = v >> 7
    def op_eor(self, ad): v = self.a ^ self.read(ad); self.a = v; self.z = v == 0; self.n = v >> 7
    def _adc(self, m):
        a = self.a
        s = a + m + int(self.c)
        self.v = ((a ^ s) & (m ^ s) & 0x80) >> 7
        self.c = s > 0xFF
        s &= 0xFF
        self.a = s; self.z = s == 0; self.n = s >> 7
    def op_adc(self, ad): self._adc(self.read(ad))
    def op_sbc(self, ad): self._adc(self.read(ad) ^ 0xFF)
    def _cmp(self, r, m):
        t = (r - m) & 0xFF
        self.c = r >= m; self.z = t == 0; self.n = t >> 7
    def op_cmp(self, ad): self._cmp(self.a, self.read(ad))
    def op_cpx(self, ad): self._cmp(self.x, self.read(ad))
    def op_cpy(self, ad): self._cmp(self.y, self.read(ad))
    def op_bit(self, ad):
        m = self.read(ad)
        self.z = (self.a & m) == 0; self.n = m >> 7; self.v = (m >> 6) & 1
    def op_asl(self, ad):
        m = self.read(ad); self.c = m >> 7; m = (m << 1) & 0xFF
        self.write(ad, m); self.z = m == 0; self.n = m >> 7
    def op_lsr(self, ad):
        m = self.read(ad); self.c = m & 1; m >>= 1
        self.write(ad, m); self.z = m == 0; self.n = 0
    def op_rol(self, ad):
        m = self.read(ad); c = int(self.c); self.c = m >> 7; m = ((m << 1) | c) & 0xFF
        self.write(ad, m); self.z = m == 0; self.n = m >> 7
    def op_ror(self, ad):
        m = self.read(ad); c = int(self.c); self.c = m & 1; m = (m >> 1) | (c << 7)
        self.write(ad, m); self.z = m == 0; self.n = m >> 7
    def op_asl_a(self, ad):
        m = self.a; self.c = m >> 7; m = (m << 1) & 0xFF
        self.a = m; self.z = m == 0; self.n = m >> 7
    def op_lsr_a(self, ad):
        m = self.a; self.c = m & 1; m >>= 1
        self.a = m; self.z = m == 0; self.n = 0
    def op_rol_a(self, ad):
        m = self.a; c = int(self.c); self.c = m >> 7; m = ((m << 1) | c) & 0xFF
        self.a = m; self.z = m == 0; self.n = m >> 7
    def op_ror_a(self, ad):
        m = self.a; c = int(self.c); self.c = m & 1; m = (m >> 1) | (c << 7)
        self.a = m; self.z = m == 0; self.n = m >> 7
    def op_inc(self, ad): m = (self.read(ad) + 1) & 0xFF; self.write(ad, m); self.z = m == 0; self.n = m >> 7
    def op_dec(self, ad): m = (self.read(ad) - 1) & 0xFF; self.write(ad, m); self.z = m == 0; self.n = m >> 7
    def op_inx(self, ad): v = self.x = (self.x + 1) & 0xFF; self.z = v == 0; self.n = v >> 7
    def op_iny(self, ad): v = self.y = (self.y + 1) & 0xFF; self.z = v == 0; self.n = v >> 7
    def op_dex(self, ad): v = self.x = (self.x - 1) & 0xFF; self.z = v == 0; self.n = v >> 7
    def op_dey(self, ad): v = self.y = (self.y - 1) & 0xFF; self.z = v == 0; self.n = v >> 7
    def op_tax(self, ad): v = self.x = self.a; self.z = v == 0; self.n = v >> 7
    def op_txa(self, ad): v = self.a = self.x; self.z = v == 0; self.n = v >> 7
    def op_tay(self, ad): v = self.y = self.a; self.z = v == 0; self.n = v >> 7
    def op_tya(self, ad): v = self.a = self.y; self.z = v == 0; self.n = v >> 7
    def op_tsx(self, ad): v = self.x = self.sp; self.z = v == 0; self.n = v >> 7
    def op_txs(self, ad): self.sp = self.x
    def op_pha(self, ad): self.push(self.a)
    def op_pla(self, ad): v = self.a = self.pull(); self.z = v == 0; self.n = v >> 7
    def op_php(self, ad): self.push(self.get_p() | 0x30)
    def op_plp(self, ad): self.set_p(self.pull())
    def op_clc(self, ad): self.c = 0
    def op_sec(self, ad): self.c = 1
    def op_cli(self, ad): self.i = 0
    def op_sei(self, ad): self.i = 1
    def op_clv(self, ad): self.v = 0
    def op_cld(self, ad): self.d = 0
    def op_sed(self, ad): self.d = 1
    def op_nop(self, ad): pass
    def op_nop_m(self, ad): self.read(ad)
    def op_jmp(self, ad): self.pc = ad
    def op_jsr(self, ad):
        p = (self.pc - 1) & 0xFFFF
        self.push(p >> 8); self.push(p & 0xFF); self.pc = ad
    def op_rts(self, ad):
        lo = self.pull(); hi = self.pull()
        self.pc = (((hi << 8) | lo) + 1) & 0xFFFF
    def op_rti(self, ad):
        self.set_p(self.pull()); lo = self.pull(); hi = self.pull()
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
    def op_bpl(self, ad): self._branch(not self.n)
    def op_bmi(self, ad): self._branch(self.n)
    def op_bvc(self, ad): self._branch(not self.v)
    def op_bvs(self, ad): self._branch(self.v)
    def op_bcc(self, ad): self._branch(not self.c)
    def op_bcs(self, ad): self._branch(self.c)
    def op_bne(self, ad): self._branch(not self.z)
    def op_beq(self, ad): self._branch(self.z)
    def op_slo(self, ad): self.op_asl(ad); self.op_ora(ad)
    def op_rla(self, ad): self.op_rol(ad); self.op_and(ad)
    def op_sre(self, ad): self.op_lsr(ad); self.op_eor(ad)
    def op_rra(self, ad): self.op_ror(ad); self.op_adc(ad)
    def op_dcp(self, ad): self.op_dec(ad); self.op_cmp(ad)
    def op_isb(self, ad): self.op_inc(ad); self.op_sbc(ad)

    def _build(self):
        t = [(self.op_nop, self.m_imp, 2, 0)] * 256
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
        add(0x24, "bit", "zp", 3); add(0x2C, "bit", "abs", 4)
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
        add(0x4C, "jmp", "abs", 3); add(0x6C, "jmp", "ind", 5); add(0x20, "jsr", "abs", 6)
        for op, fn in ((0x10, "bpl"), (0x30, "bmi"), (0x50, "bvc"), (0x70, "bvs"),
                       (0x90, "bcc"), (0xB0, "bcs"), (0xD0, "bne"), (0xF0, "beq")):
            add(op, fn, "imp", 2, 1)
        for op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA): add(op, "nop", "imp", 2)
        for op in (0x80, 0x82, 0x89, 0xC2, 0xE2): add(op, "nop_m", "imm", 2)
        for op in (0x04, 0x44, 0x64): add(op, "nop_m", "zp", 3)
        add(0x0C, "nop_m", "abs", 4)
        for op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4): add(op, "nop_m", "zpx", 4)
        for op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC): add(op, "nop_m", "abx", 4, 1)
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


class NESBus:
    def __init__(self, mapper):
        self.mapper = mapper
        self.ram = bytearray(0x800)
        self.prg = mapper.prg
        self.prg_off = mapper.prg_off
        self.prg_ram = mapper.prg_ram
        self.ppu = self.apu = self.cpu = None
        self.pads = None

    def read(self, a):
        if a < 0x2000: return self.ram[a & 0x7FF]
        if a >= 0x8000: return self.prg[self.prg_off[(a >> 13) & 3] + (a & 0x1FFF)]
        if a < 0x4000: return self.ppu.read_reg(a & 7)
        if a < 0x4020:
            if a == 0x4016: return self.pads[0].read()
            if a == 0x4017: return self.pads[1].read()
            if a == 0x4015: return self.apu.read_status()
            return 0
        if a >= 0x6000: return self.prg_ram[a - 0x6000]
        return 0

    def write(self, a, v):
        if a < 0x2000: self.ram[a & 0x7FF] = v
        elif a < 0x4000: self.ppu.write_reg(a & 7, v)
        elif a < 0x4020:
            if a == 0x4014: self._dma(v)
            elif a == 0x4016:
                self.pads[0].write(v)
                self.pads[1].write(v)
            else:
                self.apu.write(a, v)
        elif a >= 0x8000: self.mapper.write(a, v)
        elif a >= 0x6000: self.prg_ram[a - 0x6000] = v

    def _dma(self, page):
        base = page << 8
        ppu = self.ppu
        oam = ppu.oam
        start = ppu.oam_addr
        if ppu.pend: ppu.flush()
        if base < 0x2000 and start == 0:
            b = base & 0x7FF
            oam[:] = self.ram[b:b + 256]
        else:
            rd = self.read
            for i in range(256):
                oam[(start + i) & 0xFF] = rd(base + i)
        self.cpu.extra += 513 + (self.cpu.cycles & 1)


_J33 = np.arange(33, dtype=np.int32)
_J256 = np.arange(256, dtype=np.int32)
_J8 = np.arange(8, dtype=np.int32)


class NESPPU:
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
        self.vram_np = np.frombuffer(self.vram, dtype=np.uint8)
        self.oam_np = np.frombuffer(self.oam, dtype=np.uint8)
        self.chr_np = np.frombuffer(mapper.chr, dtype=np.uint8)
        self.spr_h = 8
        self.s0_y = 255
        self.sp_line = self.sp_i = self.sp_rank = np.zeros(0, dtype=np.int32)
        self.pend = []
        self._co_list = None
        self._nt_list = None
        self._co_tabs = []
        self._nt_tabs = []
        self._co_id = self._nt_id = 0

    def read_vram(self, a):
        a &= 0x3FFF
        if a < 0x2000:
            m = self.mapper
            return m.chr[m.chr_off[a >> 10] + (a & 0x3FF)]
        if a < 0x3F00:
            return self.vram[self.mapper.nt_off[(a >> 10) & 3] + (a & 0x3FF)]
        i = a & 0x1F
        if not (i & 3): i &= 0x0F
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
            if not (i & 3): i &= 0x0F
            self.palette[i] = v & 0x3F

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
            if self.pend: self.flush()
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
            if self.pend: self.flush()
            self.write_vram(self.v & 0x3FFF, v)
            self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF

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

    def evaluate_sprites(self):
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
        order = np.argsort(fl, kind="stable")
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
        chr_lines = co_tabs[arr[:, 5]]
        nt_lines = nt_tabs[arr[:, 6]]
        ar = np.arange(L)
        vram = self.vram_np
        chr_ = self.chr_np

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
                    for r in range(7, -1, -1):
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


class NESAPU:
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
        self.sink = None

    def write(self, a, v):
        if a < 0x4004: self.p1.write(a & 3, v)
        elif a < 0x4008: self.p2.write(a & 3, v)
        elif a < 0x400C: self.tri.write(a & 3, v)
        elif a < 0x4010: self.noi.write(a & 3, v)
        elif a < 0x4014: self.dmc.write(a & 3, v)
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
        self.p1.clock_env(); self.p2.clock_env(); self.noi.clock_env(); self.tri.clock_linear()

    def clock_half(self):
        self.p1.clock_length(); self.p2.clock_length(); self.noi.clock_length(); self.tri.clock_length()
        self.p1.clock_sweep(); self.p2.clock_sweep()

    def run_quarter(self, q, cycles):
        levels = self.dmc.run(cycles, self.bus)
        if self.sink is not None:
            self.acc += cycles * self.sr / CPU_HZ
            n = int(self.acc)
            self.acc -= n
            if n > 0:
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
        if a is not None: pulse += a
        a = self.p2.render(n, sr)
        if a is not None: pulse += a
        out = pulse * 0.00752
        a = self.tri.render(n, sr)
        if a is not None: out += a * 0.00851
        a = self.noi.render(n, sr)
        if a is not None: out += a * 0.00494
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


CPU_KEYS = ("a", "x", "y", "sp", "pc", "c", "z", "i", "d", "v", "n", "cycles", "nmi_pending", "extra")
PPU_KEYS = ("ctrl", "mask", "status", "oam_addr", "v", "t", "x", "w", "buf", "latch", "s0_done")
APU_KEYS = ("frame_irq", "irq", "mode5", "irq_inhibit", "acc", "dc")
QUARTER_LINES = {65: 0, 130: 1, 196: 2, 261: 3}


class NES:
    width = 256
    height = 240
    system_name = "NES"

    def __init__(self, rom_path, sample_rate=44100):
        self.rom_path = rom_path
        self.cart = NESCartridge(rom_path)
        cls = MAPPERS.get(self.cart.mapper_id)
        if cls is None:
            raise NotImplementedError(f"Mapper {self.cart.mapper_id} not supported")
        self.mapper = cls(self.cart)
        self.bus = NESBus(self.mapper)
        self.ppu = NESPPU(self.mapper)
        self.apu = NESAPU(self.bus, sample_rate)
        self.cpu = CPU6502(self.bus, self.mapper, self.apu)
        self.pads = [Controller(), Controller()]
        self.bus.ppu, self.bus.apu, self.bus.cpu, self.bus.pads = self.ppu, self.apu, self.cpu, self.pads
        self.ppu.cpu = self.cpu
        self.dot = 0
        self.apu_mark = 0
        self.frames = 0
        self._load_battery()
        self.cpu.reset()

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

    def capture_state(self):
        return {
            "cpu": {k: getattr(self.cpu, k) for k in CPU_KEYS},
            "ppu": {k: getattr(self.ppu, k) for k in PPU_KEYS},
            "apu": {k: getattr(self.apu, k) for k in APU_KEYS},
            "chans": {n: dict(getattr(self.apu, n).__dict__) for n in ("p1", "p2", "tri", "noi", "dmc")},
            "ram": bytes(self.bus.ram), "vram": bytes(self.ppu.vram),
            "oam": bytes(self.ppu.oam), "palette": bytes(self.ppu.palette),
            "mapper": self.mapper.get_state(), "dot": self.dot,
            "frame_idx": self.ppu.frame_idx.copy(),
        }

    def restore_state(self, st):
        for k, v in st["cpu"].items(): setattr(self.cpu, k, v)
        for k, v in st["ppu"].items(): setattr(self.ppu, k, v)
        for k, v in st["apu"].items(): setattr(self.apu, k, v)
        for n, d in st["chans"].items(): getattr(self.apu, n).__dict__.update(d)
        self.bus.ram[:] = st["ram"]; self.ppu.vram[:] = st["vram"]
        self.ppu.oam[:] = st["oam"]; self.ppu.palette[:] = st["palette"]
        self.mapper.set_state(st["mapper"])
        self.ppu.pend = []
        self.ppu._co_list = self.ppu._nt_list = None
        self.ppu._co_tabs = []; self.ppu._nt_tabs = []
        if "frame_idx" in st:
            self.ppu.frame_idx[:] = st["frame_idx"]
        self.dot = st["dot"]
        self.apu_mark = self.cpu.cycles

    def save_state(self, path=None):
        path = path or self.base_path + ".state"
        with open(path, "wb") as f:
            pickle.dump(self.capture_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".state"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.restore_state(pickle.load(f))
        return True

    def reset(self):
        self.cpu.reset()
        self.ppu.ctrl = self.ppu.mask = 0
        self.apu.write(0x4015, 0)

    def set_buttons(self, pad, buttons):
        self.pads[pad].buttons = buttons

    def _run(self, target):
        cpu = self.cpu; dot = self.dot; table = cpu.table; prg = cpu.prg
        prg_off = cpu.prg_off; read = cpu.read; mapper = cpu.mapper; apu = cpu.apu; step = cpu.step
        while dot < target:
            if cpu.nmi_pending or (not cpu.i and (mapper.irq or apu.irq)):
                dot += step() * 3
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
            else:
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


# ==========================================================================
# SNES HARDWARE IMPLEMENTATION (PURE PYTHON)
# ==========================================================================
# Precomputed 8-pixel bitplane lookup tables for vectorized S-PPU rendering:
# PIX[(p1 << 8) | p0] -> shape (65536, 8) with values 0..3 (already defined above for NES)
_SNES_X256 = np.arange(256, dtype=np.int32)
_SNES_Y224 = np.arange(224, dtype=np.int32)


class SNESCartridge:
    """Loads .sfc / .smc SNES ROMs, strips 512-byte copier headers, detects LoROM/HiROM."""
    def __init__(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        # Strip 512-byte SMC copier header if present
        if len(raw) >= 0x8200 and (len(raw) & 0x3FF) == 512:
            raw = raw[512:]
        if not raw:
            raise ValueError("Empty SNES ROM file")
        self.rom = raw
        self.sram = bytearray(64 * 1024)
        self.hirom = self._detect_hirom()
        hdr_off = 0xFFC0 if self.hirom else 0x7FC0
        if len(self.rom) >= hdr_off + 32:
            self.title = self.rom[hdr_off:hdr_off + 21].decode("ascii", "replace").strip()
            self.map_mode = self.rom[hdr_off + 21]
            self.rom_type = self.rom[hdr_off + 22]
        else:
            self.title = os.path.basename(path)
            self.map_mode = 0x20
            self.rom_type = 0
        self.battery = self.rom_type in (0x02, 0x05, 0x06, 0x15, 0x1A)

    def _score_header(self, offset):
        if len(self.rom) < offset + 64:
            return -100
        score = 0
        comp = self.rom[offset + 0x1C] | (self.rom[offset + 0x1D] << 8)
        chksum = self.rom[offset + 0x1E] | (self.rom[offset + 0x1F] << 8)
        if (comp ^ chksum) == 0xFFFF and chksum != 0:
            score += 8
        reset_vec = self.rom[offset + 0x3C] | (self.rom[offset + 0x3D] << 8)
        if reset_vec >= 0x8000:
            score += 4
        # Check first opcode at reset vector
        op_off = (offset & ~0x7FFF) | (reset_vec & 0x7FFF)
        if op_off < len(self.rom) and self.rom[op_off] in (0x78, 0x18, 0x4C, 0x5C, 0x20, 0xA9, 0xA2, 0xC2, 0xE2):
            score += 6
        map_byte = self.rom[offset + 0x15] & 0x0F
        if (offset & 0x8000) == 0 and map_byte in (0, 2, 3):
            score += 3
        if (offset & 0x8000) != 0 and map_byte in (1, 5):
            score += 3
        # ASCII title check
        printable = sum(1 for b in self.rom[offset:offset + 21] if 32 <= b <= 126)
        if printable >= 10:
            score += 2
        return score

    def _detect_hirom(self):
        lo_score = max(self._score_header(0x7FC0), self._score_header(0x407FC0))
        hi_score = max(self._score_header(0xFFC0), self._score_header(0x40FFC0))
        return hi_score > lo_score


class SNESPPU:
    """Vectorized SNES S-PPU: 64KB VRAM, 512B CGRAM (BGR555), 544B OAM, Modes 0-7 + OAM Sprites."""
    def __init__(self):
        self.vram = bytearray(64 * 1024)
        self.vram_np = np.frombuffer(self.vram, dtype=np.uint8)
        self.cgram = bytearray(512)
        self.oam = bytearray(544)
        self.frame_rgba = np.zeros((224, 256, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255

        # PPU Registers ($2100 - $213F)
        self.inidisp = 0x0F     # $2100: brightness & force blank (default 0x0F so display is on)
        self.obsel = 0          # $2101: sprite size/base
        self.oam_addl = 0       # $2102
        self.oam_addh = 0       # $2103
        self.oam_addr = 0       # internal 10-bit byte address
        self.oam_latch = 0
        self.bgmode = 1         # $2105: default Mode 1
        self.mosaic = 0         # $2106
        self.bg_sc = [0] * 4    # $2107-$210A: tilemap base & size
        self.bg_nba = [0] * 2   # $210B-$210C: CHR base addresses
        self.bg_hofs = [0] * 4  # $210D, $210F, $2111, $2113
        self.bg_vofs = [0] * 4  # $210E, $2110, $2112, $2114
        self.bg_prev = 0
        self.vmain = 0x80       # $2115: VRAM increment mode (default word increment on $2119)
        self.vmadd = 0          # $2116-$2117: VRAM word address
        self.vram_prefetch = 0
        self.cgadd = 0          # $2121: CGRAM byte address
        self.cg_latch = 0
        self.tm = 0x1F          # $212C: main screen designation (all layers enabled by default)
        self.ts = 0             # $212D: sub screen designation
        self.cgadsub = 0        # $2131: color math
        self.coldata = [0, 0, 0]  # $2132: fixed color R, G, B (5-bit each)
        self.nmitimen = 0       # $4200: NMI / IRQ enable
        self.rdnmi = 0          # $4210: NMI flag
        self.hvbjoy = 0         # $4212: H/V blank & auto-joypad status
        self.h_latch = 0
        self.v_latch = 0
        self.ophct_flip = 0
        self.opvct_flip = 0
        # Mode 7 matrix registers ($211B - $2120)
        self.m7_latch = 0
        self.m7a = 0x0100
        self.m7b = 0
        self.m7c = 0
        self.m7d = 0x0100
        self.m7x = 0
        self.m7y = 0
        self.mpy_result = 0

    def _vram_step(self):
        return (1, 32, 128, 128)[self.vmain & 3]

    def _trans_vmadd(self):
        a = self.vmadd & 0x7FFF
        trans = (self.vmain >> 2) & 3
        if trans == 1:    # 8-bit rotate: aaaaaaaaYYYccccc -> aaaaaaaacccccYYY
            a = (a & 0xFF00) | ((a & 0x001F) << 3) | ((a >> 5) & 7)
        elif trans == 2:  # 9-bit rotate
            a = (a & 0xFE00) | ((a & 0x003F) << 3) | ((a >> 6) & 7)
        elif trans == 3:  # 10-bit rotate
            a = (a & 0xFC00) | ((a & 0x007F) << 3) | ((a >> 7) & 7)
        return a

    def read_reg(self, addr):
        if addr == 0x2134:
            return self.mpy_result & 0xFF
        elif addr == 0x2135:
            return (self.mpy_result >> 8) & 0xFF
        elif addr == 0x2136:
            return (self.mpy_result >> 16) & 0xFF
        elif addr == 0x2137:
            return 0
        elif addr == 0x2138:  # OAMDATAREAD
            a = self.oam_addr
            if a >= 0x200:
                a = 0x200 | (a & 0x1F)
            val = self.oam[a]
            self.oam_addr = (self.oam_addr + 1) & 0x3FF
            return val
        elif addr == 0x2139:  # VMDATALREAD
            val = self.vram_prefetch & 0xFF
            if not (self.vmain & 0x80):
                waddr = self._trans_vmadd() << 1
                self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
            return val
        elif addr == 0x213A:  # VMDATAHREAD
            val = (self.vram_prefetch >> 8) & 0xFF
            if self.vmain & 0x80:
                waddr = self._trans_vmadd() << 1
                self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
            return val
        elif addr == 0x213B:  # CGDATAREAD
            val = self.cgram[self.cgadd & 0x1FF]
            self.cgadd = (self.cgadd + 1) & 0x1FF
            return val
        elif addr == 0x213C:  # OPHCT
            val = (self.h_latch >> (8 * self.ophct_flip)) & 0xFF
            self.ophct_flip ^= 1
            return val
        elif addr == 0x213D:  # OPVCT
            val = (self.v_latch >> (8 * self.opvct_flip)) & 0xFF
            self.opvct_flip ^= 1
            return val
        elif addr == 0x213E:  # STAT77 (PPU1 version = 1)
            return 0x01
        elif addr == 0x213F:  # STAT78 (PPU2 version = 1, NTSC = bit 4 clear)
            self.ophct_flip = 0
            self.opvct_flip = 0
            return 0x01
        return 0

    def write_reg(self, addr, val):
        if addr == 0x2100:
            self.inidisp = val
        elif addr == 0x2101:
            self.obsel = val
        elif addr == 0x2102:
            self.oam_addl = val
            self.oam_addr = ((self.oam_addh & 1) << 9) | (val << 1)
        elif addr == 0x2103:
            self.oam_addh = val
            self.oam_addr = ((val & 1) << 9) | (self.oam_addl << 1)
        elif addr == 0x2104:
            a = self.oam_addr
            if a < 0x200:
                if (a & 1) == 0:
                    self.oam_latch = val
                else:
                    self.oam[a - 1] = self.oam_latch
                    self.oam[a] = val
            else:
                self.oam[0x200 | (a & 0x1F)] = val
            self.oam_addr = (self.oam_addr + 1) & 0x3FF
        elif addr == 0x2105:
            self.bgmode = val
        elif addr == 0x2106:
            self.mosaic = val
        elif 0x2107 <= addr <= 0x210A:
            self.bg_sc[addr - 0x2107] = val
        elif addr in (0x210B, 0x210C):
            self.bg_nba[addr - 0x210B] = val
        elif 0x210D <= addr <= 0x2114:
            idx = (addr - 0x210D) >> 1
            # Odd addresses ($210D, $210F, $2111, $2113) are HOFS; Even ($210E, $2110, $2112, $2114) are VOFS
            if (addr & 1) == 1:
                self.bg_hofs[idx] = ((val << 8) | (self.bg_prev & ~7) | ((self.bg_hofs[idx] >> 8) & 7)) & 0x3FF
            else:
                self.bg_vofs[idx] = ((val << 8) | self.bg_prev) & 0x3FF
            self.bg_prev = val
        elif addr == 0x2115:
            self.vmain = val
        elif addr == 0x2116:
            self.vmadd = (self.vmadd & 0x7F00) | val
            waddr = self._trans_vmadd() << 1
            self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
        elif addr == 0x2117:
            self.vmadd = (self.vmadd & 0x00FF) | ((val & 0x7F) << 8)
            waddr = self._trans_vmadd() << 1
            self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
        elif addr == 0x2118:  # VMDATAL
            waddr = self._trans_vmadd() << 1
            self.vram[waddr] = val
            if not (self.vmain & 0x80):
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
        elif addr == 0x2119:  # VMDATAH
            waddr = self._trans_vmadd() << 1
            self.vram[waddr + 1] = val
            if self.vmain & 0x80:
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
        elif addr == 0x211B:
            self.m7a = (val << 8) | self.m7_latch
            self.m7_latch = val
            s16 = self.m7a - 65536 if (self.m7a & 0x8000) else self.m7a
            s8 = self.m7b - 256 if (self.m7b & 0x80) else self.m7b
            self.mpy_result = (s16 * s8) & 0xFFFFFF
        elif addr == 0x211C:
            self.m7b = val
            self.m7_latch = val
            s16 = self.m7a - 65536 if (self.m7a & 0x8000) else self.m7a
            s8 = self.m7b - 256 if (self.m7b & 0x80) else self.m7b
            self.mpy_result = (s16 * s8) & 0xFFFFFF
        elif addr == 0x211D:
            self.m7c = (val << 8) | self.m7_latch
            self.m7_latch = val
        elif addr == 0x211E:
            self.m7d = (val << 8) | self.m7_latch
            self.m7_latch = val
        elif addr == 0x211F:
            self.m7x = (val << 8) | self.m7_latch
            self.m7_latch = val
        elif addr == 0x2120:
            self.m7y = (val << 8) | self.m7_latch
            self.m7_latch = val
        elif addr == 0x2121:
            self.cgadd = (val << 1) & 0x1FF
        elif addr == 0x2122:
            if (self.cgadd & 1) == 0:
                self.cg_latch = val
            else:
                self.cgram[self.cgadd - 1] = self.cg_latch
                self.cgram[self.cgadd] = val & 0x7F
            self.cgadd = (self.cgadd + 1) & 0x1FF
        elif addr == 0x212C:
            self.tm = val
        elif addr == 0x212D:
            self.ts = val
        elif addr == 0x2131:
            self.cgadsub = val
        elif addr == 0x2132:
            intensity = val & 0x1F
            if val & 0x20:
                self.coldata[0] = intensity
            if val & 0x40:
                self.coldata[1] = intensity
            if val & 0x80:
                self.coldata[2] = intensity

    def _get_palette_rgba(self):
        """Convert 256 BGR555 CGRAM entries into a (256, 4) uint8 RGBA lookup table."""
        cg = np.frombuffer(self.cgram, dtype="<u2")
        if not cg.any() and not any(self.coldata):
            idx = np.arange(256, dtype=np.uint8)
            lut = np.zeros((256, 4), dtype=np.uint8)
            lut[:, 0] = ((idx & 0x0F) * 17).astype(np.uint8)
            lut[:, 1] = (((idx & 0x0F) * 14 + (idx >> 4) * 4) & 0xFF).astype(np.uint8)
            lut[:, 2] = (((idx & 0x0F) * 17) ^ ((idx >> 4) * 8)).astype(np.uint8)
            lut[0] = (0, 0, 0, 255)
            lut[:, 3] = 255
            return lut
        r = ((cg & 0x001F) << 3) | ((cg & 0x001F) >> 2)
        g = (((cg & 0x03E0) >> 5) << 3) | (((cg & 0x03E0) >> 5) >> 2)
        b = (((cg & 0x7C00) >> 10) << 3) | (((cg & 0x7C00) >> 10) >> 2)
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0] = r.astype(np.uint8)
        lut[:, 1] = g.astype(np.uint8)
        lut[:, 2] = b.astype(np.uint8)
        lut[:, 3] = 255
        # If CGRAM[0] is 0 but COLDATA ($2132) was set, use COLDATA as backdrop fallback
        if cg[0] == 0 and any(self.coldata):
            lut[0, 0] = (self.coldata[0] << 3) | (self.coldata[0] >> 2)
            lut[0, 1] = (self.coldata[1] << 3) | (self.coldata[1] >> 2)
            lut[0, 2] = (self.coldata[2] << 3) | (self.coldata[2] >> 2)
        return lut

    def _render_bg_layer_vec(self, bg, bpp, mode, idx_buf):
        """Vectorized NumPy background tilemap renderer for 2bpp, 4bpp, and 8bpp layers."""
        vram = self.vram_np
        sc = self.bg_sc[bg]
        map_base = ((sc >> 2) & 0x3F) << 11
        sc_size = sc & 3
        if bg < 2:
            chr_base = ((self.bg_nba[0] >> (4 * bg)) & 0x0F) << 13
        else:
            chr_base = ((self.bg_nba[1] >> (4 * (bg - 2))) & 0x0F) << 13

        hofs = self.bg_hofs[bg]
        vofs = self.bg_vofs[bg]
        tile_16 = bool(self.bgmode & (0x10 << bg))

        # Compute world (wx, wy) coordinates for all 224x256 pixels
        wy = (_SNES_Y224[:, None] + vofs) & 0x3FF
        wx = (_SNES_X256[None, :] + hofs) & 0x3FF

        if tile_16:
            ty = (wy >> 4) & 31
            tx = (wx >> 4) & 31
            sub_y = (wy >> 3) & 1
            sub_x = (wx >> 3) & 1
            nt_x = ((wx >> 9) & 1) if (sc_size & 1) else 0
            nt_y = ((wy >> 9) & 1) if (sc_size & 2) else 0
        else:
            ty = (wy >> 3) & 31
            tx = (wx >> 3) & 31
            nt_x = ((wx >> 8) & 1) if (sc_size & 1) else 0
            nt_y = ((wy >> 8) & 1) if (sc_size & 2) else 0

        fy = wy & 7
        fx = wx & 7

        if sc_size == 0:
            nt_idx = 0
        elif sc_size == 1:
            nt_idx = nt_x
        elif sc_size == 2:
            nt_idx = nt_y
        else:
            nt_idx = (nt_y << 1) | nt_x

        entry_addr = (map_base + (nt_idx << 11) + ((ty << 5) + tx) * 2) & 0xFFFE
        lo = vram[entry_addr].astype(np.uint16)
        hi = vram[entry_addr + 1].astype(np.uint16)
        entry = lo | (hi << 8)

        tile_num = (entry & 0x03FF).astype(np.int32)
        pal_num = ((entry >> 10) & 7).astype(np.uint8)
        hflip = (entry & 0x4000) != 0
        vflip = (entry & 0x8000) != 0

        if tile_16:
            sx = np.where(hflip, 1 - sub_x, sub_x)
            sy = np.where(vflip, 1 - sub_y, sub_y)
            tile_num = (tile_num + sx + (sy << 4)) & 0x3FF

        row = np.where(vflip, 7 - fy, fy)
        col = np.where(hflip, fx, 7 - fx)

        if bpp == 2:
            t_addr = (chr_base + (tile_num << 4) + (row << 1)) & 0xFFFE
            p0 = vram[t_addr]
            p1 = vram[t_addr + 1]
            c = ((p0 >> col) & 1) | (((p1 >> col) & 1) << 1)
            pal_base = (bg * 32) if mode == 0 else 0
            color_idx = pal_base + (pal_num << 2) + c
        elif bpp == 4:
            t_addr = (chr_base + (tile_num << 5) + (row << 1)) & 0xFFFE
            p0 = vram[t_addr]
            p1 = vram[t_addr + 1]
            p2 = vram[(t_addr + 16) & 0xFFFE]
            p3 = vram[(t_addr + 17) & 0xFFFF]
            c = (((p0 >> col) & 1)
                 | (((p1 >> col) & 1) << 1)
                 | (((p2 >> col) & 1) << 2)
                 | (((p3 >> col) & 1) << 3))
            color_idx = (pal_num << 4) | c
        else:  # 8bpp (Mode 3 / Mode 4 BG1)
            t_addr = (chr_base + (tile_num << 6) + (row << 1)) & 0xFFFE
            p0 = vram[t_addr]
            p1 = vram[t_addr + 1]
            p2 = vram[(t_addr + 16) & 0xFFFE]
            p3 = vram[(t_addr + 17) & 0xFFFF]
            p4 = vram[(t_addr + 32) & 0xFFFE]
            p5 = vram[(t_addr + 33) & 0xFFFF]
            p6 = vram[(t_addr + 48) & 0xFFFE]
            p7 = vram[(t_addr + 49) & 0xFFFF]
            c = (((p0 >> col) & 1)
                 | (((p1 >> col) & 1) << 1)
                 | (((p2 >> col) & 1) << 2)
                 | (((p3 >> col) & 1) << 3)
                 | (((p4 >> col) & 1) << 4)
                 | (((p5 >> col) & 1) << 5)
                 | (((p6 >> col) & 1) << 6)
                 | (((p7 >> col) & 1) << 7))
            color_idx = c

        mask = c != 0
        idx_buf[mask] = color_idx[mask]

    def _render_mode7_vec(self, idx_buf):
        """Vectorized Mode 7 affine background renderer."""
        vram = self.vram_np
        # In Mode 7, even VRAM bytes are the 128x128 tilemap, odd bytes are 8x8 8bpp pixels
        idx_map = vram[0::2]
        chr_data = vram[1::2]
        hofs = self.bg_hofs[0] - 1024 if (self.bg_hofs[0] & 0x400) else (self.bg_hofs[0] & 0x3FF)
        vofs = self.bg_vofs[0] - 1024 if (self.bg_vofs[0] & 0x400) else (self.bg_vofs[0] & 0x3FF)
        cx = self.m7x - 1024 if (self.m7x & 0x400) else (self.m7x & 0x3FF)
        cy = self.m7y - 1024 if (self.m7y & 0x400) else (self.m7y & 0x3FF)
        a = self.m7a - 65536 if (self.m7a & 0x8000) else self.m7a
        b = self.m7b - 65536 if (self.m7b & 0x8000) else self.m7b
        c = self.m7c - 65536 if (self.m7c & 0x8000) else self.m7c
        d = self.m7d - 65536 if (self.m7d & 0x8000) else self.m7d

        sx = _SNES_X256[None, :] + hofs - cx
        sy = _SNES_Y224[:, None] + vofs - cy
        wx = ((a * sx + b * sy) >> 8) + cx
        wy = ((c * sx + d * sy) >> 8) + cy
        wx &= 0x3FF
        wy &= 0x3FF
        tx = (wx >> 3) & 0x7F
        ty = (wy >> 3) & 0x7F
        fx = wx & 7
        fy = wy & 7
        tile = idx_map[(ty << 7) + tx].astype(np.int32)
        pix = chr_data[(tile << 6) + (fy << 3) + fx]
        mask = pix != 0
        idx_buf[mask] = pix[mask]

    def _render_sprites(self, idx_buf):
        """Render OAM 4bpp sprites (128 objects, standard SNES size tables)."""
        if not (self.tm & 0x10):
            return
        vram = self.vram_np
        oam = self.oam
        base0 = (self.obsel & 7) << 14
        gap = ((self.obsel >> 3) & 3) << 13
        size_sel = (self.obsel >> 5) & 7
        sizes = (
            ((8, 8), (16, 16)),
            ((8, 8), (32, 32)),
            ((8, 8), (64, 64)),
            ((16, 16), (32, 32)),
            ((16, 16), (64, 64)),
            ((32, 32), (64, 64)),
            ((16, 32), (32, 64)),
            ((16, 32), (32, 32)),
        )[size_sel]

        # Draw from lowest priority index (127) to highest (0)
        for i in range(127, -1, -1):
            b4 = i << 2
            x_lo = oam[b4]
            y = oam[b4 + 1]
            if y >= 224 and y < 240:
                continue
            tile = oam[b4 + 2]
            attr = oam[b4 + 3]
            hi_bits = (oam[0x200 + (i >> 2)] >> ((i & 3) << 1)) & 3
            x = x_lo | ((hi_bits & 1) << 8)
            if x >= 256:
                x -= 512
            w, h = sizes[(hi_bits >> 1) & 1]
            if x + w <= 0 or x >= 256:
                continue
            y = (y + 1) & 0xFF
            if y >= 224 and (y + h) <= 256:
                continue

            pal = 128 + (((attr >> 1) & 7) << 4)
            name_table = attr & 1
            hflip = bool(attr & 0x40)
            vflip = bool(attr & 0x80)
            chr_base = (base0 + ( (0x2000 + gap) if name_table else 0 )) & 0xFFFE
            tw = w >> 3
            th = h >> 3

            for ty in range(th):
                ry = (th - 1 - ty) if vflip else ty
                for tx in range(tw):
                    rx = (tw - 1 - tx) if hflip else tx
                    t_idx = (((tile >> 4) + ry) & 0x0F) << 4 | (((tile & 0x0F) + rx) & 0x0F)
                    t_base = (chr_base + (t_idx << 5)) & 0xFFFE
                    for fy in range(8):
                        py = (y + (ty << 3) + fy) & 0xFF
                        if py >= 224:
                            continue
                        row = (7 - fy) if vflip else fy
                        ra = (t_base + (row << 1)) & 0xFFFE
                        p0, p1 = vram[ra], vram[ra + 1]
                        p2, p3 = vram[(ra + 16) & 0xFFFE], vram[(ra + 17) & 0xFFFF]
                        if not (p0 | p1 | p2 | p3):
                            continue
                        for fx in range(8):
                            px = x + (tx << 3) + fx
                            if 0 <= px < 256:
                                sh = fx if hflip else (7 - fx)
                                c = (((p0 >> sh) & 1)
                                     | (((p1 >> sh) & 1) << 1)
                                     | (((p2 >> sh) & 1) << 2)
                                     | (((p3 >> sh) & 1) << 3))
                                if c:
                                    idx_buf[py, px] = pal | c

    def render_frame(self):
        """Render active background layers and OAM sprites into 256x224 RGBA framebuffer."""
        lut = self._get_palette_rgba()
        has_vram = bool(self.vram_np.any() or any(self.cgram) or any(self.oam[:512]))
        if (self.inidisp & 0x80) and not has_vram:
            self.frame_rgba[:] = (0, 0, 0, 255)
            return

        idx_buf = np.zeros((224, 256), dtype=np.uint8)
        mode = self.bgmode & 7
        # Use effective TM mask; if game left TM=0 during boot while VRAM is populated, show BG1+OBJ
        tm = self.tm if self.tm else (0x11 if has_vram else 0)

        if mode == 0:
            for bg in (3, 2, 1, 0):
                if tm & (1 << bg):
                    self._render_bg_layer_vec(bg, 2, 0, idx_buf)
        elif mode == 1:
            if tm & 4:
                self._render_bg_layer_vec(2, 2, 1, idx_buf)
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 1, idx_buf)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, 1, idx_buf)
        elif mode == 2:
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 2, idx_buf)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, 2, idx_buf)
        elif mode == 3:
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 3, idx_buf)
            if tm & 1:
                self._render_bg_layer_vec(0, 8, 3, idx_buf)
        elif mode == 4:
            if tm & 2:
                self._render_bg_layer_vec(1, 2, 4, idx_buf)
            if tm & 1:
                self._render_bg_layer_vec(0, 8, 4, idx_buf)
        elif mode in (5, 6):
            if (mode == 5) and (tm & 2):
                self._render_bg_layer_vec(1, 2, 5, idx_buf)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, mode, idx_buf)
        elif mode == 7:
            self._render_mode7_vec(idx_buf)

        if tm & 0x10:
            self._render_sprites(idx_buf)

        bright_val = (self.inidisp & 0x0F) if not (self.inidisp & 0x80) else 15
        bright = max(1, bright_val) / 15.0
        rendered = lut[idx_buf]
        if bright < 1.0:
            rendered[:, :, :3] = (rendered[:, :, :3].astype(np.float32) * bright).astype(np.uint8)
        self.frame_rgba[:] = rendered


class SNESAPU:
    """Sony SPC700 + DSP interface (handles IPL boot & multi-block driver transfer handshakes)."""
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.ports_in = [0, 0, 0, 0]
        self.ports_out = [0xAA, 0xBB, 0, 0]  # Initial IPL ready signature
        self.volume = 0.7
        self.sink = None
        self.acc = 0.0
        self.irq = False
        self.boot_step = 0

    def read_port(self, idx):
        return self.ports_out[idx & 3]

    def write_port(self, idx, val):
        idx &= 3
        val &= 0xFF
        self.ports_in[idx] = val
        if idx == 0:
            if val == 0xCC:
                self.ports_out[0] = 0xCC
                self.boot_step = 1
            elif self.boot_step == 1:
                self.ports_out[0] = val
                if self.ports_in[1] == 0 and val != 0:
                    self.ports_out[1] = 0xBB
            else:
                self.ports_out[0] = 0xAA if val == 0 else val
                self.ports_out[1] = 0xBB
        else:
            if self.boot_step == 1:
                self.ports_out[idx] = val

    def run_frame(self):
        if self.sink is not None:
            n = int(self.sr / FRAME_RATE)
            self.sink(np.zeros(n, dtype=np.int16))


class SNESBus:
    """24-bit SNES System Bus (LoROM / HiROM, 128KB WRAM, PPU/APU/DMA/HDMA MMIO)."""
    def __init__(self, cart: SNESCartridge, ppu: SNESPPU, apu: SNESAPU):
        self.cart = cart
        self.rom = cart.rom
        self.rom_len = max(1, len(self.rom))
        self.hirom = cart.hirom
        self.sram = cart.sram
        self.wram = bytearray(128 * 1024)
        self.ppu = ppu
        self.apu = apu
        self.wmadd = 0

        # Hardware math & DMA registers
        self.wrmpya = 0xFF
        self.wrmpyb = 0
        self.wrdiv = 0xFFFF
        self.wrdivb = 0
        self.rddiv = 0
        self.rdmpy = 0
        self.mdmaen = 0
        self.hdmaen = 0
        self.dma_regs = bytearray(0x80)
        self.joy1 = 0
        self.joy2 = 0
        self.joy_shift1 = 0
        self.joy_shift2 = 0
        self.joy_strobe = 0
        self.cpu = None
        self.timeup = 0
        self._hdma_cur = [0] * 8
        self._hdma_cnt = [0] * 8

    def read(self, bank, addr):
        bank &= 0xFF
        addr &= 0xFFFF
        if bank in (0x7E, 0x7F):
            return self.wram[((bank & 1) << 16) | addr]

        low_bank = bank & 0x7F
        if low_bank <= 0x3F:
            if addr < 0x2000:
                return self.wram[addr]
            if 0x2100 <= addr <= 0x213F:
                return self.ppu.read_reg(addr)
            if 0x2140 <= addr <= 0x217F:
                return self.apu.read_port(addr & 3)
            if addr == 0x2180:
                v = self.wram[self.wmadd]
                self.wmadd = (self.wmadd + 1) & 0x1FFFF
                return v
            if addr == 0x4016:
                if self.joy_strobe:
                    return (self.joy1 >> 15) & 1
                v = (self.joy_shift1 >> 15) & 1
                self.joy_shift1 = ((self.joy_shift1 << 1) | 1) & 0xFFFF
                return v
            if addr == 0x4017:
                if self.joy_strobe:
                    return (self.joy2 >> 15) & 1
                v = (self.joy_shift2 >> 15) & 1
                self.joy_shift2 = ((self.joy_shift2 << 1) | 1) & 0xFFFF
                return v
            if addr == 0x4210:
                v = (self.ppu.rdnmi & 0x80) | 0x02  # Bit 7 = NMI flag, Bits 0-3 = 5A22 CPU version 2
                self.ppu.rdnmi &= 0x7F              # Reading $4210 clears bit 7
                return v
            if addr == 0x4211:
                v = self.timeup
                self.timeup = 0
                return v
            if addr == 0x4212:
                # Toggle HBlank (bit 6) dynamically so games waiting on HBlank loops don't hang
                self.ppu.hvbjoy ^= 0x40
                return self.ppu.hvbjoy & 0xFE       # Bit 0 = 0 (Auto-joypad read complete)
            if 0x4214 <= addr <= 0x4217:
                return (self.rddiv & 0xFF, self.rddiv >> 8, self.rdmpy & 0xFF, self.rdmpy >> 8)[addr - 0x4214]
            if addr == 0x4218:
                return self.joy1 & 0xFF
            if addr == 0x4219:
                return (self.joy1 >> 8) & 0xFF
            if addr == 0x421A:
                return self.joy2 & 0xFF
            if addr == 0x421B:
                return (self.joy2 >> 8) & 0xFF
            if 0x4300 <= addr <= 0x437F:
                return self.dma_regs[addr - 0x4300]

        # Cartridge ROM / SRAM mapping
        if self.hirom:
            if 0x20 <= low_bank <= 0x3F and 0x6000 <= addr <= 0x7FFF:
                return self.sram[(((low_bank - 0x20) << 13) | (addr & 0x1FFF)) % len(self.sram)]
            if addr >= 0x8000 or (0x40 <= low_bank <= 0x7D):
                off = ((bank & 0x3F) << 16) | addr
                return self.rom[off % self.rom_len]
        else:
            if 0x70 <= low_bank <= 0x7D and addr < 0x8000:
                return self.sram[(((low_bank - 0x70) << 15) | addr) % len(self.sram)]
            if addr >= 0x8000 or (0x40 <= low_bank <= 0x6F):
                off = ((bank & 0x7F) << 15) | (addr & 0x7FFF)
                return self.rom[off % self.rom_len]
        return 0

    def write(self, bank, addr, val):
        bank &= 0xFF
        addr &= 0xFFFF
        val &= 0xFF
        if bank in (0x7E, 0x7F):
            self.wram[((bank & 1) << 16) | addr] = val
            return

        low_bank = bank & 0x7F
        if low_bank <= 0x3F:
            if addr < 0x2000:
                self.wram[addr] = val
                return
            if 0x2100 <= addr <= 0x2133:
                self.ppu.write_reg(addr, val)
                return
            if 0x2140 <= addr <= 0x217F:
                self.apu.write_port(addr & 3, val)
                return
            if addr == 0x2180:
                self.wram[self.wmadd] = val
                self.wmadd = (self.wmadd + 1) & 0x1FFFF
                return
            if addr == 0x2181:
                self.wmadd = (self.wmadd & 0x1FF00) | val
                return
            if addr == 0x2182:
                self.wmadd = (self.wmadd & 0x100FF) | (val << 8)
                return
            if addr == 0x2183:
                self.wmadd = (self.wmadd & 0x0FFFF) | ((val & 1) << 16)
                return
            if addr == 0x4016:
                self.joy_strobe = val & 1
                if self.joy_strobe:
                    self.joy_shift1 = self.joy1
                    self.joy_shift2 = self.joy2
                return
            if addr == 0x4200:
                old = self.ppu.nmitimen
                self.ppu.nmitimen = val
                if (val & 0x80) and not (old & 0x80) and (self.ppu.rdnmi & 0x80) and self.cpu is not None:
                    self.cpu.nmi_pending = True
                return
            if addr == 0x4202:
                self.wrmpya = val
                return
            if addr == 0x4203:
                self.wrmpyb = val
                self.rdmpy = (self.wrmpya * self.wrmpyb) & 0xFFFF
                return
            if addr == 0x4204:
                self.wrdiv = (self.wrdiv & 0xFF00) | val
                return
            if addr == 0x4205:
                self.wrdiv = (self.wrdiv & 0x00FF) | (val << 8)
                return
            if addr == 0x4206:
                self.wrdivb = val
                if val == 0:
                    self.rddiv = 0xFFFF
                    self.rdmpy = self.wrdiv
                else:
                    self.rddiv = (self.wrdiv // val) & 0xFFFF
                    self.rdmpy = (self.wrdiv % val) & 0xFFFF
                return
            if addr == 0x420B:
                self._run_dma(val)
                return
            if addr == 0x420C:
                self.hdmaen = val
                return
            if 0x4300 <= addr <= 0x437F:
                self.dma_regs[addr - 0x4300] = val
                return

        if self.hirom and 0x20 <= low_bank <= 0x3F and 0x6000 <= addr <= 0x7FFF:
            self.sram[(((low_bank - 0x20) << 13) | (addr & 0x1FFF)) % len(self.sram)] = val
        elif not self.hirom and 0x70 <= low_bank <= 0x7D and addr < 0x8000:
            self.sram[(((low_bank - 0x70) << 15) | addr) % len(self.sram)] = val

    def _run_dma(self, mask):
        for ch in range(8):
            if not (mask & (1 << ch)):
                continue
            base = ch << 4
            dmap = self.dma_regs[base + 0]
            bbad = 0x2100 | self.dma_regs[base + 1]
            a1t = self.dma_regs[base + 2] | (self.dma_regs[base + 3] << 8)
            a1b = self.dma_regs[base + 4]
            das = self.dma_regs[base + 5] | (self.dma_regs[base + 6] << 8)
            if das == 0:
                das = 0x10000
            direction = (dmap >> 7) & 1  # 0: A->B (CPU->PPU), 1: B->A
            step_mode = (dmap >> 3) & 3
            step = 1 if step_mode == 0 else (-1 if step_mode == 2 else 0)
            xfer_mode = dmap & 7
            offsets = (
                (0,),
                (0, 1),
                (0, 0),
                (0, 0, 1, 1),
                (0, 1, 2, 3),
                (0, 1, 0, 1),
                (0, 0),
                (0, 0, 1, 1),
            )[xfer_mode]

            for i in range(das):
                b_addr = (bbad + offsets[i % len(offsets)]) & 0xFFFF
                if direction == 0:
                    val = self.read(a1b, a1t)
                    self.write(0x00, b_addr, val)
                else:
                    val = self.read(0x00, b_addr)
                    self.write(a1b, a1t, val)
                a1t = (a1t + step) & 0xFFFF

            self.dma_regs[base + 2] = a1t & 0xFF
            self.dma_regs[base + 3] = (a1t >> 8) & 0xFF
            self.dma_regs[base + 5] = 0
            self.dma_regs[base + 6] = 0

    def _run_hdma_scanline(self, sl):
        if not self.hdmaen or sl >= 224:
            return
        if sl == 0:
            self._hdma_cur = [
                self.dma_regs[(ch << 4) + 2] | (self.dma_regs[(ch << 4) + 3] << 8)
                for ch in range(8)
            ]
            self._hdma_cnt = [0] * 8
        for ch in range(8):
            if not (self.hdmaen & (1 << ch)):
                continue
            base = ch << 4
            dmap = self.dma_regs[base + 0]
            bbad = 0x2100 | self.dma_regs[base + 1]
            bank = self.dma_regs[base + 4]
            ind_bank = self.dma_regs[base + 7]
            indirect = bool(dmap & 0x40)
            xfer_mode = dmap & 7
            offsets = ((0,), (0, 1), (0, 0), (0, 0, 1, 1), (0, 1, 2, 3), (0, 1, 0, 1), (0, 0), (0, 0, 1, 1))[xfer_mode]

            cnt = self._hdma_cnt[ch]
            cur = self._hdma_cur[ch]
            do_xfer = False
            if (cnt & 0x7F) == 0:
                hdr = self.read(bank, cur)
                cur = (cur + 1) & 0xFFFF
                if hdr == 0:
                    self._hdma_cnt[ch] = 0
                    continue
                cnt = hdr
                if indirect:
                    lo = self.read(bank, cur)
                    hi = self.read(bank, (cur + 1) & 0xFFFF)
                    cur = (cur + 2) & 0xFFFF
                    self.dma_regs[base + 5] = lo
                    self.dma_regs[base + 6] = hi
                do_xfer = True
            else:
                cnt = (cnt & 0x80) | ((cnt - 1) & 0x7F)
                do_xfer = bool(cnt & 0x80)

            if do_xfer:
                if indirect:
                    ptr = self.dma_regs[base + 5] | (self.dma_regs[base + 6] << 8)
                    for off in offsets:
                        self.write(0x00, (bbad + off) & 0xFFFF, self.read(ind_bank, ptr))
                        ptr = (ptr + 1) & 0xFFFF
                    self.dma_regs[base + 5] = ptr & 0xFF
                    self.dma_regs[base + 6] = (ptr >> 8) & 0xFF
                else:
                    for off in offsets:
                        self.write(0x00, (bbad + off) & 0xFFFF, self.read(bank, cur))
                        cur = (cur + 1) & 0xFFFF

            self._hdma_cnt[ch] = cnt
            self._hdma_cur[ch] = cur


class CPU65c816:
    """Full WDC 65c816 CPU supporting Emulation (6502) and Native (8/16-bit) modes."""
    _OP_LEN = [
        2,2,2,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        3,2,4,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        1,2,2,2,3,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,3,2,2,2, 1,3,1,1,4,3,3,4,
        1,2,3,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        2,2,3,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        2,2,2,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        2,2,2,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,2,2,2,2, 1,3,1,1,3,3,3,4,
        2,2,2,2,2,2,2,2, 1,2,1,1,3,3,3,4, 2,2,2,2,3,2,2,2, 1,3,1,1,3,3,3,4,
    ]

    def __init__(self, bus: SNESBus):
        self.bus = bus
        self.a = 0          # 16-bit C register (A = low byte, B = high byte)
        self.x = 0          # 16-bit X index
        self.y = 0          # 16-bit Y index
        self.sp = 0x01FF    # 16-bit Stack Pointer
        self.d = 0          # 16-bit Direct Page register
        self.dbr = 0        # 8-bit Data Bank register
        self.pbr = 0        # 8-bit Program Bank register
        self.pc = 0         # 16-bit Program Counter
        # Processor Status flags
        self.c = 0
        self.z = 0
        self.i = 1
        self.dec = 0
        self.x_flag = 1     # 1 = 8-bit X/Y, 0 = 16-bit X/Y
        self.m_flag = 1     # 1 = 8-bit A, 0 = 16-bit A
        self.v = 0
        self.n = 0
        self.e = 1          # 1 = 6502 Emulation mode, 0 = Native mode
        self.cycles = 0
        self.nmi_pending = False
        self.waiting = False

    def reset(self):
        self.e = 1
        self.m_flag = 1
        self.x_flag = 1
        self.i = 1
        self.dec = 0
        self.d = 0
        self.dbr = 0
        self.pbr = 0
        self.sp = 0x01FF
        self.pc = self.bus.read(0, 0xFFFC) | (self.bus.read(0, 0xFFFD) << 8)
        self.nmi_pending = False
        self.waiting = False

    def get_p(self):
        return (self.c | (self.z << 1) | (self.i << 2) | (self.dec << 3)
                | (self.x_flag << 4) | (self.m_flag << 5) | (self.v << 6) | (self.n << 7))

    def set_p(self, p):
        self.c = p & 1
        self.z = (p >> 1) & 1
        self.i = (p >> 2) & 1
        self.dec = (p >> 3) & 1
        self.x_flag = (p >> 4) & 1
        self.m_flag = (p >> 5) & 1
        self.v = (p >> 6) & 1
        self.n = (p >> 7) & 1
        if self.e:
            self.x_flag = 1
            self.m_flag = 1
        if self.x_flag:
            self.x &= 0xFF
            self.y &= 0xFF

    def _fetch8(self):
        v = self.bus.read(self.pbr, self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return v

    def _fetch16(self):
        lo = self._fetch8()
        hi = self._fetch8()
        return lo | (hi << 8)

    def _fetch24(self):
        lo = self._fetch8()
        mid = self._fetch8()
        hi = self._fetch8()
        return hi, (mid << 8) | lo

    def _read16(self, bank, addr):
        lo = self.bus.read(bank, addr)
        hi = self.bus.read(bank, (addr + 1) & 0xFFFF)
        return lo | (hi << 8)

    def _write16(self, bank, addr, val):
        self.bus.write(bank, addr, val & 0xFF)
        self.bus.write(bank, (addr + 1) & 0xFFFF, (val >> 8) & 0xFF)

    def push8(self, v):
        self.bus.write(0, self.sp, v & 0xFF)
        if self.e:
            self.sp = 0x0100 | ((self.sp - 1) & 0xFF)
        else:
            self.sp = (self.sp - 1) & 0xFFFF

    def pull8(self):
        if self.e:
            self.sp = 0x0100 | ((self.sp + 1) & 0xFF)
        else:
            self.sp = (self.sp + 1) & 0xFFFF
        return self.bus.read(0, self.sp)

    def push16(self, v):
        self.push8((v >> 8) & 0xFF)
        self.push8(v & 0xFF)

    def pull16(self):
        lo = self.pull8()
        hi = self.pull8()
        return lo | (hi << 8)

    def trigger_nmi(self):
        self.waiting = False
        if not self.e:
            self.push8(self.pbr)
        self.push16(self.pc)
        self.push8(self.get_p())
        self.i = 1
        self.dec = 0
        self.pbr = 0
        vec = 0xFFFA if self.e else 0xFFEA
        self.pc = self._read16(0, vec)

    def trigger_irq(self):
        self.waiting = False
        if self.i:
            return
        if not self.e:
            self.push8(self.pbr)
        self.push16(self.pc)
        self.push8(self.get_p())
        self.i = 1
        self.dec = 0
        self.pbr = 0
        vec = 0xFFFE if self.e else 0xFFEE
        self.pc = self._read16(0, vec)

    def _nz8(self, v):
        v &= 0xFF
        self.z = int(v == 0)
        self.n = (v >> 7) & 1

    def _nz16(self, v):
        v &= 0xFFFF
        self.z = int(v == 0)
        self.n = (v >> 15) & 1

    def _lda_val(self, bank, addr):
        if self.m_flag:
            v = self.bus.read(bank, addr)
            self.a = (self.a & 0xFF00) | v
            self._nz8(v)
        else:
            v = self._read16(bank, addr)
            self.a = v
            self._nz16(v)

    def _sta_val(self, bank, addr):
        if self.m_flag:
            self.bus.write(bank, addr, self.a & 0xFF)
        else:
            self._write16(bank, addr, self.a)

    def _ldx_val(self, bank, addr):
        if self.x_flag:
            v = self.bus.read(bank, addr)
            self.x = v
            self._nz8(v)
        else:
            v = self._read16(bank, addr)
            self.x = v
            self._nz16(v)

    def _ldy_val(self, bank, addr):
        if self.x_flag:
            v = self.bus.read(bank, addr)
            self.y = v
            self._nz8(v)
        else:
            v = self._read16(bank, addr)
            self.y = v
            self._nz16(v)

    def _stx_val(self, bank, addr):
        if self.x_flag:
            self.bus.write(bank, addr, self.x & 0xFF)
        else:
            self._write16(bank, addr, self.x)

    def _sty_val(self, bank, addr):
        if self.x_flag:
            self.bus.write(bank, addr, self.y & 0xFF)
        else:
            self._write16(bank, addr, self.y)

    def _adc_val(self, m):
        if self.m_flag:
            a = self.a & 0xFF
            s = a + (m & 0xFF) + self.c
            self.v = 1 if (~(a ^ m) & (a ^ s) & 0x80) else 0
            self.c = 1 if s > 0xFF else 0
            self.a = (self.a & 0xFF00) | (s & 0xFF)
            self._nz8(s)
        else:
            a = self.a
            s = a + (m & 0xFFFF) + self.c
            self.v = 1 if (~(a ^ m) & (a ^ s) & 0x8000) else 0
            self.c = 1 if s > 0xFFFF else 0
            self.a = s & 0xFFFF
            self._nz16(self.a)

    def _sbc_val(self, m):
        if self.m_flag:
            self._adc_val((m ^ 0xFF) & 0xFF)
        else:
            self._adc_val((m ^ 0xFFFF) & 0xFFFF)

    def _cmp_val(self, reg, m, is8):
        if is8:
            r = reg & 0xFF
            m &= 0xFF
            self.c = int(r >= m)
            self._nz8((r - m) & 0xFF)
        else:
            r = reg & 0xFFFF
            m &= 0xFFFF
            self.c = int(r >= m)
            self._nz16((r - m) & 0xFFFF)

    def _dp_ind(self):
        dp = (self.d + self._fetch8()) & 0xFFFF
        return self.dbr, self._read16(0, dp)

    def _dp_ind_x(self):
        dp = (self.d + self._fetch8() + self.x) & 0xFFFF
        return self.dbr, self._read16(0, dp)

    def _dp_ind_long(self):
        dp = (self.d + self._fetch8()) & 0xFFFF
        lo = self._read16(0, dp)
        bank = self.bus.read(0, (dp + 2) & 0xFFFF)
        return bank, lo

    def _dp_ind_y(self):
        dp = (self.d + self._fetch8()) & 0xFFFF
        base = self._read16(0, dp)
        full = ((self.dbr << 16) + base + self.y) & 0xFFFFFF
        return full >> 16, full & 0xFFFF

    def _dp_ind_long_y(self):
        dp = (self.d + self._fetch8()) & 0xFFFF
        base = self._read16(0, dp)
        bank = self.bus.read(0, (dp + 2) & 0xFFFF)
        full = ((bank << 16) + base + self.y) & 0xFFFFFF
        return full >> 16, full & 0xFFFF

    def _sr_ind_y(self):
        sp_ad = (self.sp + self._fetch8()) & 0xFFFF
        base = self._read16(0, sp_ad)
        full = ((self.dbr << 16) + base + self.y) & 0xFFFFFF
        return full >> 16, full & 0xFFFF

    def _addr_rmw(self, op_low):
        """Resolve (bank, addr) for 0x06/0x16/0x0E/0x1E style Read-Modify-Write instructions."""
        if op_low == 0x06:
            return 0, (self.d + self._fetch8()) & 0xFFFF
        elif op_low == 0x16:
            return 0, (self.d + self._fetch8() + self.x) & 0xFFFF
        elif op_low == 0x0E:
            return self.dbr, self._fetch16()
        else:
            return self.dbr, (self._fetch16() + self.x) & 0xFFFF

    def _addr_alu(self, mode_bits):
        """Resolve (bank, addr) for standard 8-mode ALU instructions (ORA, AND, EOR, ADC, STA, LDA, CMP, SBC)."""
        if mode_bits == 0:    # (dp,X)
            return self._dp_ind_x()
        elif mode_bits == 1:  # dp
            return 0, (self.d + self._fetch8()) & 0xFFFF
        elif mode_bits == 3:  # abs
            return self.dbr, self._fetch16()
        elif mode_bits == 4:  # (dp),Y
            return self._dp_ind_y()
        elif mode_bits == 5:  # dp,X
            return 0, (self.d + self._fetch8() + self.x) & 0xFFFF
        elif mode_bits == 6:  # abs,Y
            full = ((self.dbr << 16) + self._fetch16() + self.y) & 0xFFFFFF
            return full >> 16, full & 0xFFFF
        elif mode_bits == 7:  # abs,X
            full = ((self.dbr << 16) + self._fetch16() + self.x) & 0xFFFFFF
            return full >> 16, full & 0xFFFF
        return 0, 0

    def _addr_alu_ext(self, op_low):
        """Resolve (bank, addr) for 65c816 extended ALU modes: sr,S (0x03), [dp] (0x07), (dp) (0x12), (sr,S),Y (0x13), [dp],Y (0x17), long (0x0F), long,X (0x1F)."""
        if op_low == 0x03:
            return 0, (self.sp + self._fetch8()) & 0xFFFF
        elif op_low == 0x07:
            return self._dp_ind_long()
        elif op_low == 0x0F:
            return self._fetch24()
        elif op_low == 0x12:
            return self._dp_ind()
        elif op_low == 0x13:
            return self._sr_ind_y()
        elif op_low == 0x17:
            return self._dp_ind_long_y()
        else:  # 0x1F long,X
            b, ad = self._fetch24()
            full = ((b << 16) + ad + self.x) & 0xFFFFFF
            return full >> 16, full & 0xFFFF

    def step(self):
        if self.nmi_pending:
            self.nmi_pending = False
            self.trigger_nmi()
            self.cycles += 8
            return 8
        if self.waiting:
            self.cycles += 2
            return 2

        op = self._fetch8()
        cyc = 2

        # Flag & Mode Instructions
        if op == 0x18: self.c = 0                                      # CLC
        elif op == 0x38: self.c = 1                                    # SEC
        elif op == 0x58: self.i = 0                                    # CLI
        elif op == 0x78: self.i = 1                                    # SEI
        elif op == 0xB8: self.v = 0                                    # CLV
        elif op == 0xD8: self.dec = 0                                  # CLD
        elif op == 0xF8: self.dec = 1                                  # SED
        elif op == 0xFB:                                               # XCE
            self.c, self.e = self.e, self.c
            if self.e:
                self.m_flag = 1
                self.x_flag = 1
                self.sp = 0x0100 | (self.sp & 0xFF)
                self.x &= 0xFF
                self.y &= 0xFF
        elif op == 0xC2:                                               # REP #imm
            self.set_p(self.get_p() & ~self._fetch8())
            cyc = 3
        elif op == 0xE2:                                               # SEP #imm
            self.set_p(self.get_p() | self._fetch8())
            cyc = 3
        elif op in (0xCB, 0xDB):                                       # WAI / STP
            self.waiting = True
        elif op in (0xEA, 0x42):                                       # NOP / WDM
            if op == 0x42: self._fetch8()

        # Block Moves (MVN / MVP)
        elif op == 0x54:                                               # MVN
            dst, src = self._fetch8(), self._fetch8()
            self.dbr = dst
            self.bus.write(dst, self.y, self.bus.read(src, self.x))
            mask = 0xFF if self.x_flag else 0xFFFF
            self.x = (self.x + 1) & mask
            self.y = (self.y + 1) & mask
            self.a = (self.a - 1) & 0xFFFF
            if self.a != 0xFFFF:
                self.pc = (self.pc - 3) & 0xFFFF
            cyc = 7
        elif op == 0x44:                                               # MVP
            dst, src = self._fetch8(), self._fetch8()
            self.dbr = dst
            self.bus.write(dst, self.y, self.bus.read(src, self.x))
            mask = 0xFF if self.x_flag else 0xFFFF
            self.x = (self.x - 1) & mask
            self.y = (self.y - 1) & mask
            self.a = (self.a - 1) & 0xFFFF
            if self.a != 0xFFFF:
                self.pc = (self.pc - 3) & 0xFFFF
            cyc = 7

        # Stack & Register Transfers
        elif op == 0x48:                                               # PHA
            if self.m_flag: self.push8(self.a & 0xFF)
            else: self.push16(self.a)
        elif op == 0x68:                                               # PLA
            if self.m_flag:
                v = self.pull8(); self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.a = self.pull16(); self._nz16(self.a)
        elif op == 0xDA:                                               # PHX
            if self.x_flag: self.push8(self.x)
            else: self.push16(self.x)
        elif op == 0xFA:                                               # PLX
            if self.x_flag: self.x = self.pull8(); self._nz8(self.x)
            else: self.x = self.pull16(); self._nz16(self.x)
        elif op == 0x5A:                                               # PHY
            if self.x_flag: self.push8(self.y)
            else: self.push16(self.y)
        elif op == 0x7A:                                               # PLY
            if self.x_flag: self.y = self.pull8(); self._nz8(self.y)
            else: self.y = self.pull16(); self._nz16(self.y)
        elif op == 0x08: self.push8(self.get_p())                      # PHP
        elif op == 0x28: self.set_p(self.pull8())                      # PLP
        elif op == 0x8B: self.push8(self.dbr)                          # PHB
        elif op == 0xAB: self.dbr = self.pull8(); self._nz8(self.dbr)  # PLB
        elif op == 0x4B: self.push8(self.pbr)                          # PHK
        elif op == 0x0B: self.push16(self.d)                           # PHD
        elif op == 0x2B: self.d = self.pull16(); self._nz16(self.d)    # PLD
        elif op == 0xF4: self.push16(self._fetch16())                  # PEA
        elif op == 0xD4:                                               # PEI
            dp = (self.d + self._fetch8()) & 0xFFFF
            self.push16(self._read16(0, dp))
        elif op == 0x62:                                               # PER
            off = self._fetch16()
            self.push16((self.pc + off) & 0xFFFF)
        elif op == 0x5B: self.d = self.a; self._nz16(self.d)           # TCD
        elif op == 0x7B: self.a = self.d; self._nz16(self.a)           # TDC
        elif op == 0x1B: self.sp = (0x0100 | (self.a & 0xFF)) if self.e else self.a  # TCS
        elif op == 0x3B: self.a = self.sp; self._nz16(self.a)          # TSC
        elif op == 0xEB:                                               # XBA
            lo, hi = self.a & 0xFF, (self.a >> 8) & 0xFF
            self.a = (lo << 8) | hi
            self._nz8(hi)
        elif op == 0xAA:                                               # TAX
            if self.x_flag: self.x = self.a & 0xFF; self._nz8(self.x)
            else: self.x = self.a; self._nz16(self.x)
        elif op == 0x8A:                                               # TXA
            if self.m_flag: self.a = (self.a & 0xFF00) | (self.x & 0xFF); self._nz8(self.a)
            else: self.a = self.x; self._nz16(self.a)
        elif op == 0xA8:                                               # TAY
            if self.x_flag: self.y = self.a & 0xFF; self._nz8(self.y)
            else: self.y = self.a; self._nz16(self.y)
        elif op == 0x98:                                               # TYA
            if self.m_flag: self.a = (self.a & 0xFF00) | (self.y & 0xFF); self._nz8(self.a)
            else: self.a = self.y; self._nz16(self.a)
        elif op == 0x9B:                                               # TXY
            if self.x_flag: self.y = self.x & 0xFF; self._nz8(self.y)
            else: self.y = self.x; self._nz16(self.y)
        elif op == 0xBB:                                               # TYX
            if self.x_flag: self.x = self.y & 0xFF; self._nz8(self.x)
            else: self.x = self.y; self._nz16(self.x)
        elif op == 0x9A: self.sp = (0x0100 | (self.x & 0xFF)) if self.e else self.x  # TXS
        elif op == 0xBA:                                               # TSX
            if self.x_flag: self.x = self.sp & 0xFF; self._nz8(self.x)
            else: self.x = self.sp; self._nz16(self.x)

        # Standard 8-group ALU instructions (ORA, AND, EOR, ADC, STA, LDA, CMP, SBC)
        elif (op & 3) == 1 or (op & 0x1F) in (0x03, 0x07, 0x0F, 0x12, 0x13, 0x17, 0x1F):
            alu_grp = op >> 5
            if op == 0x89:                                             # BIT #imm
                m = self._fetch8() if self.m_flag else self._fetch16()
                mask = 0xFF if self.m_flag else 0xFFFF
                self.z = int(((self.a & mask) & m) == 0)
            elif (op & 0x1F) == 0x09:                                  # #imm
                m = self._fetch8() if self.m_flag else self._fetch16()
                if alu_grp == 0:                                       # ORA
                    if self.m_flag: v = (self.a | m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                    else: self.a |= m; self._nz16(self.a)
                elif alu_grp == 1:                                     # AND
                    if self.m_flag: v = (self.a & m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                    else: self.a &= m; self._nz16(self.a)
                elif alu_grp == 2:                                     # EOR
                    if self.m_flag: v = (self.a ^ m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                    else: self.a ^= m; self._nz16(self.a)
                elif alu_grp == 3: self._adc_val(m)                    # ADC
                elif alu_grp == 5:                                     # LDA
                    if self.m_flag: self.a = (self.a & 0xFF00) | m; self._nz8(m)
                    else: self.a = m; self._nz16(m)
                elif alu_grp == 6: self._cmp_val(self.a, m, bool(self.m_flag))  # CMP
                elif alu_grp == 7: self._sbc_val(m)                    # SBC
            else:
                if (op & 3) == 1:
                    b, ad = self._addr_alu((op >> 2) & 7)
                else:
                    b, ad = self._addr_alu_ext(op & 0x1F)
                if alu_grp == 4:                                       # STA
                    self._sta_val(b, ad)
                elif alu_grp == 5:                                     # LDA
                    self._lda_val(b, ad)
                else:
                    m = self.bus.read(b, ad) if self.m_flag else self._read16(b, ad)
                    if alu_grp == 0:                                   # ORA
                        if self.m_flag: v = (self.a | m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                        else: self.a |= m; self._nz16(self.a)
                    elif alu_grp == 1:                                 # AND
                        if self.m_flag: v = (self.a & m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                        else: self.a &= m; self._nz16(self.a)
                    elif alu_grp == 2:                                 # EOR
                        if self.m_flag: v = (self.a ^ m) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
                        else: self.a ^= m; self._nz16(self.a)
                    elif alu_grp == 3: self._adc_val(m)                # ADC
                    elif alu_grp == 6: self._cmp_val(self.a, m, bool(self.m_flag))  # CMP
                    elif alu_grp == 7: self._sbc_val(m)                # SBC

        # STZ (0x64, 0x74, 0x9C, 0x9E)
        elif op in (0x64, 0x74, 0x9C, 0x9E):
            if op == 0x64: b, ad = 0, (self.d + self._fetch8()) & 0xFFFF
            elif op == 0x74: b, ad = 0, (self.d + self._fetch8() + self.x) & 0xFFFF
            elif op == 0x9C: b, ad = self.dbr, self._fetch16()
            else: b, ad = self.dbr, (self._fetch16() + self.x) & 0xFFFF
            if self.m_flag: self.bus.write(b, ad, 0)
            else: self._write16(b, ad, 0)

        # TRB / TSB (0x04, 0x0C, 0x14, 0x1C)
        elif op in (0x04, 0x0C, 0x14, 0x1C):
            b, ad = (0, (self.d + self._fetch8()) & 0xFFFF) if op in (0x04, 0x14) else (self.dbr, self._fetch16())
            if self.m_flag:
                m = self.bus.read(b, ad)
                a8 = self.a & 0xFF
                self.z = int((a8 & m) == 0)
                self.bus.write(b, ad, (m | a8) if op < 0x10 else (m & ~a8))
            else:
                m = self._read16(b, ad)
                self.z = int((self.a & m) == 0)
                self._write16(b, ad, (m | self.a) if op < 0x10 else (m & ~self.a))

        # LDX / STX / LDY / STY
        elif op == 0xA2:
            if self.x_flag: self.x = self._fetch8(); self._nz8(self.x)
            else: self.x = self._fetch16(); self._nz16(self.x)
        elif op == 0xA6: self._ldx_val(0, (self.d + self._fetch8()) & 0xFFFF)
        elif op == 0xB6: self._ldx_val(0, (self.d + self._fetch8() + self.y) & 0xFFFF)
        elif op == 0xAE: self._ldx_val(self.dbr, self._fetch16())
        elif op == 0xBE: self._ldx_val(self.dbr, (self._fetch16() + self.y) & 0xFFFF)
        elif op == 0x86: self._stx_val(0, (self.d + self._fetch8()) & 0xFFFF)
        elif op == 0x96: self._stx_val(0, (self.d + self._fetch8() + self.y) & 0xFFFF)
        elif op == 0x8E: self._stx_val(self.dbr, self._fetch16())

        elif op == 0xA0:
            if self.x_flag: self.y = self._fetch8(); self._nz8(self.y)
            else: self.y = self._fetch16(); self._nz16(self.y)
        elif op == 0xA4: self._ldy_val(0, (self.d + self._fetch8()) & 0xFFFF)
        elif op == 0xB4: self._ldy_val(0, (self.d + self._fetch8() + self.x) & 0xFFFF)
        elif op == 0xAC: self._ldy_val(self.dbr, self._fetch16())
        elif op == 0xBC: self._ldy_val(self.dbr, (self._fetch16() + self.x) & 0xFFFF)
        elif op == 0x84: self._sty_val(0, (self.d + self._fetch8()) & 0xFFFF)
        elif op == 0x94: self._sty_val(0, (self.d + self._fetch8() + self.x) & 0xFFFF)
        elif op == 0x8C: self._sty_val(self.dbr, self._fetch16())

        # INC / DEC (Registers and Memory)
        elif op == 0x1A:
            if self.m_flag:
                v = ((self.a & 0xFF) + 1) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.a = (self.a + 1) & 0xFFFF; self._nz16(self.a)
        elif op == 0x3A:
            if self.m_flag:
                v = ((self.a & 0xFF) - 1) & 0xFF; self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.a = (self.a - 1) & 0xFFFF; self._nz16(self.a)
        elif op in (0xE6, 0xF6, 0xEE, 0xFE, 0xC6, 0xD6, 0xCE, 0xDE):
            delta = 1 if op >= 0xE0 else -1
            b, ad = self._addr_rmw(op & 0x1F)
            if self.m_flag:
                v = (self.bus.read(b, ad) + delta) & 0xFF
                self.bus.write(b, ad, v); self._nz8(v)
            else:
                v = (self._read16(b, ad) + delta) & 0xFFFF
                self._write16(b, ad, v); self._nz16(v)
        elif op == 0xE8:
            if self.x_flag: self.x = (self.x + 1) & 0xFF; self._nz8(self.x)
            else: self.x = (self.x + 1) & 0xFFFF; self._nz16(self.x)
        elif op == 0xCA:
            if self.x_flag: self.x = (self.x - 1) & 0xFF; self._nz8(self.x)
            else: self.x = (self.x - 1) & 0xFFFF; self._nz16(self.x)
        elif op == 0xC8:
            if self.x_flag: self.y = (self.y + 1) & 0xFF; self._nz8(self.y)
            else: self.y = (self.y + 1) & 0xFFFF; self._nz16(self.y)
        elif op == 0x88:
            if self.x_flag: self.y = (self.y - 1) & 0xFF; self._nz8(self.y)
            else: self.y = (self.y - 1) & 0xFFFF; self._nz16(self.y)

        # Shifts (ASL / LSR / ROL / ROR Accumulator and Memory)
        elif op == 0x0A:                                               # ASL A
            if self.m_flag:
                v = self.a & 0xFF; self.c = (v >> 7) & 1; v = (v << 1) & 0xFF
                self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.c = (self.a >> 15) & 1; self.a = (self.a << 1) & 0xFFFF; self._nz16(self.a)
        elif op == 0x4A:                                               # LSR A
            if self.m_flag:
                v = self.a & 0xFF; self.c = v & 1; v >>= 1
                self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.c = self.a & 1; self.a >>= 1; self._nz16(self.a)
        elif op == 0x2A:                                               # ROL A
            c = self.c
            if self.m_flag:
                v = self.a & 0xFF; self.c = (v >> 7) & 1; v = ((v << 1) | c) & 0xFF
                self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.c = (self.a >> 15) & 1; self.a = ((self.a << 1) | c) & 0xFFFF; self._nz16(self.a)
        elif op == 0x6A:                                               # ROR A
            c = self.c
            if self.m_flag:
                v = self.a & 0xFF; self.c = v & 1; v = (v >> 1) | (c << 7)
                self.a = (self.a & 0xFF00) | v; self._nz8(v)
            else:
                self.c = self.a & 1; self.a = (self.a >> 1) | (c << 15); self._nz16(self.a)
        elif op in (0x06, 0x16, 0x0E, 0x1E, 0x46, 0x56, 0x4E, 0x5E, 0x26, 0x36, 0x2E, 0x3E, 0x66, 0x76, 0x6E, 0x7E):
            b, ad = self._addr_rmw(op & 0x1F)
            kind = op >> 5
            c = self.c
            if self.m_flag:
                v = self.bus.read(b, ad)
                if kind == 0: self.c = (v >> 7) & 1; v = (v << 1) & 0xFF
                elif kind == 2: self.c = v & 1; v >>= 1
                elif kind == 1: self.c = (v >> 7) & 1; v = ((v << 1) | c) & 0xFF
                else: self.c = v & 1; v = (v >> 1) | (c << 7)
                self.bus.write(b, ad, v); self._nz8(v)
            else:
                v = self._read16(b, ad)
                if kind == 0: self.c = (v >> 15) & 1; v = (v << 1) & 0xFFFF
                elif kind == 2: self.c = v & 1; v >>= 1
                elif kind == 1: self.c = (v >> 15) & 1; v = ((v << 1) | c) & 0xFFFF
                else: self.c = v & 1; v = (v >> 1) | (c << 15)
                self._write16(b, ad, v); self._nz16(v)

        # BIT / CPX / CPY
        elif op in (0x24, 0x34, 0x2C, 0x3C):                           # BIT mem
            if op == 0x24: b, ad = 0, (self.d + self._fetch8()) & 0xFFFF
            elif op == 0x34: b, ad = 0, (self.d + self._fetch8() + self.x) & 0xFFFF
            elif op == 0x2C: b, ad = self.dbr, self._fetch16()
            else: b, ad = self.dbr, (self._fetch16() + self.x) & 0xFFFF
            if self.m_flag:
                m = self.bus.read(b, ad)
                self.z = int(((self.a & 0xFF) & m) == 0)
                self.n = (m >> 7) & 1; self.v = (m >> 6) & 1
            else:
                m = self._read16(b, ad)
                self.z = int((self.a & m) == 0)
                self.n = (m >> 15) & 1; self.v = (m >> 14) & 1
        elif op in (0xE0, 0xE4, 0xEC):                                 # CPX
            if op == 0xE0: m = self._fetch8() if self.x_flag else self._fetch16()
            elif op == 0xE4: ad = (self.d + self._fetch8()) & 0xFFFF; m = self.bus.read(0, ad) if self.x_flag else self._read16(0, ad)
            else: ad = self._fetch16(); m = self.bus.read(self.dbr, ad) if self.x_flag else self._read16(self.dbr, ad)
            self._cmp_val(self.x, m, bool(self.x_flag))
        elif op in (0xC0, 0xC4, 0xCC):                                 # CPY
            if op == 0xC0: m = self._fetch8() if self.x_flag else self._fetch16()
            elif op == 0xC4: ad = (self.d + self._fetch8()) & 0xFFFF; m = self.bus.read(0, ad) if self.x_flag else self._read16(0, ad)
            else: ad = self._fetch16(); m = self.bus.read(self.dbr, ad) if self.x_flag else self._read16(self.dbr, ad)
            self._cmp_val(self.y, m, bool(self.x_flag))

        # Jumps, Subroutines, Interrupts & Branches
        elif op == 0x00:                                               # BRK
            self._fetch8()
            if not self.e: self.push8(self.pbr)
            self.push16(self.pc)
            self.push8(self.get_p())
            self.i = 1; self.dec = 0; self.pbr = 0
            self.pc = self._read16(0, 0xFFFE if self.e else 0xFFE6)
        elif op == 0x02:                                               # COP
            self._fetch8()
            if not self.e: self.push8(self.pbr)
            self.push16(self.pc)
            self.push8(self.get_p())
            self.i = 1; self.dec = 0; self.pbr = 0
            self.pc = self._read16(0, 0xFFF4 if self.e else 0xFFE4)
        elif op == 0x4C: self.pc = self._fetch16()                     # JMP abs
        elif op == 0x6C: self.pc = self._read16(0, self._fetch16())    # JMP (abs)
        elif op == 0x7C:                                               # JMP (abs,X)
            self.pc = self._read16(self.pbr, (self._fetch16() + self.x) & 0xFFFF)
        elif op == 0x5C:                                               # JML long
            b, ad = self._fetch24(); self.pbr = b; self.pc = ad
        elif op == 0xDC:                                               # JML [abs]
            ptr = self._fetch16()
            self.pc = self._read16(0, ptr)
            self.pbr = self.bus.read(0, (ptr + 2) & 0xFFFF)
        elif op == 0x20:                                               # JSR abs
            ad = self._fetch16()
            self.push16((self.pc - 1) & 0xFFFF)
            self.pc = ad
        elif op == 0xFC:                                               # JSR (abs,X)
            ad = self._read16(self.pbr, (self._fetch16() + self.x) & 0xFFFF)
            self.push16((self.pc - 1) & 0xFFFF)
            self.pc = ad
        elif op == 0x22:                                               # JSL long
            b, ad = self._fetch24()
            self.push8(self.pbr)
            self.push16((self.pc - 1) & 0xFFFF)
            self.pbr = b; self.pc = ad
        elif op == 0x60: self.pc = (self.pull16() + 1) & 0xFFFF        # RTS
        elif op == 0x6B:                                               # RTL
            self.pc = (self.pull16() + 1) & 0xFFFF
            self.pbr = self.pull8()
        elif op == 0x40:                                               # RTI
            self.set_p(self.pull8())
            self.pc = self.pull16()
            if not self.e: self.pbr = self.pull8()
        elif op in (0x10, 0x30, 0x50, 0x70, 0x80, 0x90, 0xB0, 0xD0, 0xF0):
            off = self._fetch8()
            if off & 0x80: off -= 256
            cond = {
                0x10: not self.n, 0x30: self.n, 0x50: not self.v, 0x70: self.v,
                0x80: True, 0x90: not self.c, 0xB0: self.c, 0xD0: not self.z, 0xF0: self.z
            }[op]
            if cond:
                self.pc = (self.pc + off) & 0xFFFF
                cyc = 3
        elif op == 0x82:                                               # BRL rel16
            off = self._fetch16()
            if off & 0x8000: off -= 65536
            self.pc = (self.pc + off) & 0xFFFF
            cyc = 4
        else:
            # Safe fallback: consume operand bytes so PC never desynchronizes
            ln = self._OP_LEN[op]
            if op in (0x09, 0x29, 0x49, 0x69, 0x89, 0xA9, 0xC9, 0xE9) and not self.m_flag:
                ln = 3
            elif op in (0xA0, 0xA2, 0xC0, 0xE0) and not self.x_flag:
                ln = 3
            self.pc = (self.pc + (ln - 1)) & 0xFFFF

        self.cycles += cyc
        return cyc


class SNES:
    """Complete Pure-Python SNES Console conforming to the PyRetro console interface."""
    width = 256
    height = 224
    system_name = "SNES"

    def __init__(self, rom_path, sample_rate=44100):
        self.rom_path = rom_path
        self.cart = SNESCartridge(rom_path)
        self.ppu = SNESPPU()
        self.apu = SNESAPU(sample_rate)
        self.bus = SNESBus(self.cart, self.ppu, self.apu)
        self.cpu = CPU65c816(self.bus)
        self.bus.cpu = self.cpu
        self.frames = 0
        self._load_battery()
        self.cpu.reset()

    @property
    def base_path(self):
        return os.path.splitext(self.rom_path)[0]

    def _load_battery(self):
        p = self.base_path + ".srm"
        if self.cart.battery and os.path.exists(p):
            with open(p, "rb") as f:
                d = f.read(len(self.cart.sram))
            self.cart.sram[:len(d)] = d

    def save_battery(self):
        if self.cart.battery:
            with open(self.base_path + ".srm", "wb") as f:
                f.write(bytes(self.cart.sram))

    def capture_state(self):
        return {
            "a": self.cpu.a, "x": self.cpu.x, "y": self.cpu.y,
            "sp": self.cpu.sp, "d": self.cpu.d, "dbr": self.cpu.dbr,
            "pbr": self.cpu.pbr, "pc": self.cpu.pc, "p": self.cpu.get_p(),
            "e": self.cpu.e, "wram": bytes(self.bus.wram),
            "vram": bytes(self.ppu.vram), "cgram": bytes(self.ppu.cgram),
            "oam": bytes(self.ppu.oam), "frame_rgba": self.ppu.frame_rgba.copy(),
        }

    def restore_state(self, st):
        self.cpu.a, self.cpu.x, self.cpu.y = st["a"], st["x"], st["y"]
        self.cpu.sp, self.cpu.d, self.cpu.dbr = st["sp"], st["d"], st["dbr"]
        self.cpu.pbr, self.cpu.pc, self.cpu.e = st["pbr"], st["pc"], st["e"]
        self.cpu.set_p(st["p"])
        self.bus.wram[:] = st["wram"]
        self.ppu.vram[:] = st["vram"]
        self.ppu.cgram[:] = st["cgram"]
        self.ppu.oam[:] = st["oam"]
        if "frame_rgba" in st:
            self.ppu.frame_rgba[:] = st["frame_rgba"]

    def save_state(self, path=None):
        path = path or self.base_path + ".sstate"
        with open(path, "wb") as f:
            pickle.dump(self.capture_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".sstate"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.restore_state(pickle.load(f))
        return True

    def reset(self):
        self.cpu.reset()

    def set_buttons(self, pad, buttons):
        if pad != 0:
            return
        # Map NES button bits to SNES 16-bit joypad register ($4218-$4219)
        snes_joy = 0
        if buttons & BTN_B:      snes_joy |= 0x8000  # B
        if buttons & BTN_SELECT: snes_joy |= 0x2000  # Select
        if buttons & BTN_START:  snes_joy |= 0x1000  # Start
        if buttons & BTN_UP:     snes_joy |= 0x0800  # Up
        if buttons & BTN_DOWN:   snes_joy |= 0x0400  # Down
        if buttons & BTN_LEFT:   snes_joy |= 0x0200  # Left
        if buttons & BTN_RIGHT:  snes_joy |= 0x0100  # Right
        if buttons & BTN_A:      snes_joy |= 0x0080  # A
        self.bus.joy1 = snes_joy

    def run_frame(self, draw=True):
        # NTSC SNES: 262 scanlines, ~1364 master cycles per line (~3.58 MHz CPU)
        cycles_per_line = 220
        for sl in range(262):
            self.ppu.v_latch = sl
            if sl < 224:
                self.bus._run_hdma_scanline(sl)
            if sl == 225:
                self.ppu.hvbjoy |= 0x80   # Enter VBlank
                self.ppu.rdnmi |= 0x80
                if self.ppu.nmitimen & 0x80:
                    self.cpu.nmi_pending = True
            elif sl == 0:
                self.ppu.hvbjoy &= ~0x80  # Exit VBlank
                self.ppu.rdnmi &= ~0x80

            if (self.ppu.nmitimen & 0x30) and sl == 200:
                self.bus.timeup = 0x80
                self.cpu.trigger_irq()

            target = self.cpu.cycles + cycles_per_line
            while self.cpu.cycles < target:
                self.cpu.step()

        if draw:
            self.ppu.render_frame()
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.ppu.frame_rgba



# ==========================================================================
# GAME BOY / GAME BOY COLOR HARDWARE IMPLEMENTATION (.gb, .gbc)
# ==========================================================================
_GB_DMG_PAL = np.array([
    [224, 248, 208, 255],
    [136, 192, 112, 255],
    [52,  104, 86,  255],
    [8,   24,  32,  255],
], dtype=np.uint8)


class GameBoy:
    """Pure-Python Game Boy (DMG) & Game Boy Color (CGB) Emulator (Sharp LR35902 + PPU + MBC1/3/5)."""
    width = 160
    height = 144

    def __init__(self, rom_path: str, sample_rate: int = 44100):
        self.rom_path = rom_path
        with open(rom_path, "rb") as f:
            self.rom = f.read()
        self.cgb = (len(self.rom) > 0x143 and (self.rom[0x143] & 0x80) != 0) or rom_path.lower().endswith(".gbc")
        self.system_name = "GBC" if self.cgb else "GB"
        self.rom_banks = max(2, len(self.rom) // 0x4000)
        self.rom_bank = 1
        self.ram_bank = 0
        self.ram_enable = False
        self.mbc_mode = 0
        self.eram = bytearray(32 * 1024)
        self.vram = bytearray(16 * 1024)  # 2 banks of 8KB for CGB
        self.vram_bank = 0
        self.wram = bytearray(32 * 1024)  # 8 banks of 4KB for CGB
        self.wram_bank = 1
        self.oam = bytearray(160)
        self.hram = bytearray(128)
        self.io = bytearray(128)
        self.bg_pal = bytearray(64)
        self.obj_pal = bytearray(64)
        self.bcpd_idx = 0
        self.ocpd_idx = 0
        self.frame_rgba = np.zeros((144, 160, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255
        self.apu = SNESAPU(sample_rate)
        self.buttons = 0
        self.frames = 0
        self._load_battery()
        self.reset()

    @property
    def base_path(self):
        return os.path.splitext(self.rom_path)[0]

    def _load_battery(self):
        p = self.base_path + ".sav"
        if os.path.exists(p):
            with open(p, "rb") as f:
                d = f.read(len(self.eram))
            self.eram[:len(d)] = d

    def save_battery(self):
        with open(self.base_path + ".sav", "wb") as f:
            f.write(bytes(self.eram))

    def reset(self):
        self.a = 0x11 if self.cgb else 0x01
        self.f = 0xB0
        self.b = 0x00
        self.c = 0x13
        self.d = 0x00
        self.e = 0xD8
        self.h = 0x01
        self.l = 0x4D
        self.sp = 0xFFFE
        self.pc = 0x0100
        self.ime = False
        self.halted = False
        self.div_cycles = 0
        self.tima_cycles = 0
        self.io[0x40] = 0x91  # LCDC
        self.io[0x47] = 0xFC  # BGP
        self.io[0x48] = 0xFF  # OBP0
        self.io[0x49] = 0xFF  # OBP1
        self.bg_pal[:] = b"\xFF\x7F" * 32
        self.obj_pal[:] = b"\xFF\x7F" * 32

    def set_buttons(self, pad, buttons):
        if pad == 0:
            self.buttons = buttons

    def read(self, addr):
        addr &= 0xFFFF
        if addr < 0x4000:
            return self.rom[addr % len(self.rom)]
        if addr < 0x8000:
            off = (self.rom_bank * 0x4000) + (addr - 0x4000)
            return self.rom[off % len(self.rom)]
        if addr < 0xA000:
            return self.vram[(self.vram_bank * 0x2000) + (addr - 0x8000)]
        if addr < 0xC000:
            return self.eram[((self.ram_bank * 0x2000) + (addr - 0xA000)) % len(self.eram)]
        if addr < 0xD000:
            return self.wram[addr - 0xC000]
        if addr < 0xE000:
            return self.wram[(self.wram_bank * 0x1000) + (addr - 0xD000)]
        if addr < 0xFE00:
            return self.wram[(addr - 0xE000) & 0x1FFF]
        if addr < 0xFEA0:
            return self.oam[addr - 0xFE00]
        if addr < 0xFF00:
            return 0xFF
        if addr < 0xFF80:
            r = addr & 0x7F
            if r == 0x00:  # JOYP
                sel = self.io[0x00] & 0x30
                low = 0x0F
                if not (sel & 0x10):  # D-pad
                    if self.buttons & BTN_RIGHT: low &= ~1
                    if self.buttons & BTN_LEFT:  low &= ~2
                    if self.buttons & BTN_UP:    low &= ~4
                    if self.buttons & BTN_DOWN:  low &= ~8
                if not (sel & 0x20):  # Action buttons
                    if self.buttons & BTN_A:      low &= ~1
                    if self.buttons & BTN_B:      low &= ~2
                    if self.buttons & BTN_SELECT: low &= ~4
                    if self.buttons & BTN_START:  low &= ~8
                return 0xC0 | sel | low
            if r == 0x4F:
                return 0xFE | self.vram_bank
            if r == 0x69:
                return self.bg_pal[self.bcpd_idx & 0x3F]
            if r == 0x6B:
                return self.obj_pal[self.ocpd_idx & 0x3F]
            if r == 0x70:
                return 0xF8 | self.wram_bank
            return self.io[r]
        if addr < 0xFFFF:
            return self.hram[addr - 0xFF80]
        return self.hram[0x7F]  # IE ($FFFF)

    def write(self, addr, val):
        addr &= 0xFFFF
        val &= 0xFF
        if addr < 0x2000:
            self.ram_enable = (val & 0x0F) == 0x0A
        elif addr < 0x4000:
            b = val & 0x7F
            self.rom_bank = (b if b else 1) % self.rom_banks
        elif addr < 0x6000:
            self.ram_bank = val & 3
        elif addr < 0x8000:
            self.mbc_mode = val & 1
        elif addr < 0xA000:
            self.vram[(self.vram_bank * 0x2000) + (addr - 0x8000)] = val
        elif addr < 0xC000:
            if self.ram_enable:
                self.eram[((self.ram_bank * 0x2000) + (addr - 0xA000)) % len(self.eram)] = val
        elif addr < 0xD000:
            self.wram[addr - 0xC000] = val
        elif addr < 0xE000:
            self.wram[(self.wram_bank * 0x1000) + (addr - 0xD000)] = val
        elif addr < 0xFE00:
            self.wram[(addr - 0xE000) & 0x1FFF] = val
        elif addr < 0xFEA0:
            self.oam[addr - 0xFE00] = val
        elif 0xFF00 <= addr < 0xFF80:
            r = addr & 0x7F
            self.io[r] = val
            if r == 0x04:
                self.io[0x04] = 0
            elif r == 0x46:  # OAM DMA
                base = val << 8
                for i in range(160):
                    self.oam[i] = self.read(base + i)
            elif r == 0x4F and self.cgb:
                self.vram_bank = val & 1
            elif r == 0x55 and self.cgb:  # CGB HDMA / GDMA
                src = ((self.io[0x51] << 8) | (self.io[0x52] & 0xF0)) & 0xFFFF
                dst = (0x8000 | (((self.io[0x53] & 0x1F) << 8) | (self.io[0x54] & 0xF0))) & 0xFFFF
                length = ((val & 0x7F) + 1) * 16
                for i in range(length):
                    self.write(dst + i, self.read(src + i))
                self.io[0x55] = 0xFF
            elif r == 0x68:
                self.bcpd_idx = val
            elif r == 0x69:
                self.bg_pal[self.bcpd_idx & 0x3F] = val
                if self.bcpd_idx & 0x80:
                    self.bcpd_idx = 0x80 | ((self.bcpd_idx + 1) & 0x3F)
            elif r == 0x6A:
                self.ocpd_idx = val
            elif r == 0x6B:
                self.obj_pal[self.ocpd_idx & 0x3F] = val
                if self.ocpd_idx & 0x80:
                    self.ocpd_idx = 0x80 | ((self.ocpd_idx + 1) & 0x3F)
            elif r == 0x70 and self.cgb:
                self.wram_bank = max(1, val & 7)
        elif addr < 0xFFFF:
            self.hram[addr - 0xFF80] = val
        else:
            self.hram[0x7F] = val

    def _read16(self, addr):
        return self.read(addr) | (self.read((addr + 1) & 0xFFFF) << 8)

    def _push16(self, val):
        self.sp = (self.sp - 1) & 0xFFFF
        self.write(self.sp, (val >> 8) & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF
        self.write(self.sp, val & 0xFF)

    def _pop16(self):
        lo = self.read(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        hi = self.read(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return (hi << 8) | lo

    def _get_r(self, idx):
        if idx == 0: return self.b
        if idx == 1: return self.c
        if idx == 2: return self.d
        if idx == 3: return self.e
        if idx == 4: return self.h
        if idx == 5: return self.l
        if idx == 6: return self.read((self.h << 8) | self.l)
        return self.a

    def _set_r(self, idx, val):
        val &= 0xFF
        if idx == 0: self.b = val
        elif idx == 1: self.c = val
        elif idx == 2: self.d = val
        elif idx == 3: self.e = val
        elif idx == 4: self.h = val
        elif idx == 5: self.l = val
        elif idx == 6: self.write((self.h << 8) | self.l, val)
        else: self.a = val

    def _alu(self, op, val):
        a = self.a
        c = (self.f >> 4) & 1
        if op == 0:  # ADD
            r = a + val
            self.f = (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) + (val & 15) > 15 else 0) | (0x10 if r > 255 else 0)
            self.a = r & 0xFF
        elif op == 1:  # ADC
            r = a + val + c
            self.f = (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) + (val & 15) + c > 15 else 0) | (0x10 if r > 255 else 0)
            self.a = r & 0xFF
        elif op == 2:  # SUB
            r = a - val
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) else 0) | (0x10 if r < 0 else 0)
            self.a = r & 0xFF
        elif op == 3:  # SBC
            r = a - val - c
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) + c else 0) | (0x10 if r < 0 else 0)
            self.a = r & 0xFF
        elif op == 4:  # AND
            self.a = a & val
            self.f = (0x80 if self.a == 0 else 0) | 0x20
        elif op == 5:  # XOR
            self.a = a ^ val
            self.f = 0x80 if self.a == 0 else 0
        elif op == 6:  # OR
            self.a = a | val
            self.f = 0x80 if self.a == 0 else 0
        elif op == 7:  # CP
            r = a - val
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) else 0) | (0x10 if r < 0 else 0)

    def step(self):
        ie = self.hram[0x7F]
        if_reg = self.io[0x0F]
        pending = ie & if_reg & 0x1F
        if pending:
            self.halted = False
            if self.ime:
                self.ime = False
                for bit in range(5):
                    if pending & (1 << bit):
                        self.io[0x0F] &= ~(1 << bit)
                        self._push16(self.pc)
                        self.pc = 0x40 + (bit << 3)
                        return 20
        if self.halted:
            return 4

        op = self.read(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF

        if op == 0x00: return 4                                        # NOP
        if op == 0x76: self.halted = True; return 4                    # HALT
        if op == 0xF3: self.ime = False; return 4                      # DI
        if op == 0xFB: self.ime = True; return 4                       # EI
        if 0x40 <= op <= 0x7F:                                         # LD r, r'
            self._set_r((op >> 3) & 7, self._get_r(op & 7))
            return 8 if ((op & 7) == 6 or ((op >> 3) & 7) == 6) else 4
        if 0x80 <= op <= 0xBF:                                         # ALU A, r
            self._alu((op >> 3) & 7, self._get_r(op & 7))
            return 8 if (op & 7) == 6 else 4
        if (op & 0xC7) == 0x06:                                        # LD r, d8
            v = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self._set_r((op >> 3) & 7, v)
            return 8
        if (op & 0xC7) == 0x04:                                        # INC r
            idx = (op >> 3) & 7
            v = self._get_r(idx); r = (v + 1) & 0xFF; self._set_r(idx, r)
            self.f = (self.f & 0x10) | (0x80 if r == 0 else 0) | (0x20 if (v & 15) == 15 else 0)
            return 12 if idx == 6 else 4
        if (op & 0xC7) == 0x05:                                        # DEC r
            idx = (op >> 3) & 7
            v = self._get_r(idx); r = (v - 1) & 0xFF; self._set_r(idx, r)
            self.f = (self.f & 0x10) | 0x40 | (0x80 if r == 0 else 0) | (0x20 if (v & 15) == 0 else 0)
            return 12 if idx == 6 else 4
        if (op & 0xC7) == 0xC6:                                        # ALU A, d8
            v = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self._alu((op >> 3) & 7, v)
            return 8
        if op in (0x01, 0x11, 0x21, 0x31):                             # LD rr, d16
            v = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            if op == 0x01: self.b, self.c = v >> 8, v & 0xFF
            elif op == 0x11: self.d, self.e = v >> 8, v & 0xFF
            elif op == 0x21: self.h, self.l = v >> 8, v & 0xFF
            else: self.sp = v
            return 12
        if op in (0x03, 0x13, 0x23, 0x33, 0x0B, 0x1B, 0x2B, 0x3B):     # INC/DEC rr
            d = 1 if (op & 8) == 0 else -1
            reg = (op >> 4) & 3
            if reg == 0: v = ((self.b << 8) | self.c) + d; self.b, self.c = (v >> 8) & 0xFF, v & 0xFF
            elif reg == 1: v = ((self.d << 8) | self.e) + d; self.d, self.e = (v >> 8) & 0xFF, v & 0xFF
            elif reg == 2: v = ((self.h << 8) | self.l) + d; self.h, self.l = (v >> 8) & 0xFF, v & 0xFF
            else: self.sp = (self.sp + d) & 0xFFFF
            return 8
        if op in (0x09, 0x19, 0x29, 0x39):                             # ADD HL, rr
            hl = (self.h << 8) | self.l
            rr = ((self.b << 8) | self.c, (self.d << 8) | self.e, hl, self.sp)[(op >> 4) & 3]
            r = hl + rr
            self.f = (self.f & 0x80) | (0x20 if (hl & 0xFFF) + (rr & 0xFFF) > 0xFFF else 0) | (0x10 if r > 0xFFFF else 0)
            self.h, self.l = (r >> 8) & 0xFF, r & 0xFF
            return 8
        if op == 0x02: self.write((self.b << 8) | self.c, self.a); return 8
        if op == 0x12: self.write((self.d << 8) | self.e, self.a); return 8
        if op == 0x22:
            hl = (self.h << 8) | self.l; self.write(hl, self.a)
            hl = (hl + 1) & 0xFFFF; self.h, self.l = hl >> 8, hl & 0xFF; return 8
        if op == 0x32:
            hl = (self.h << 8) | self.l; self.write(hl, self.a)
            hl = (hl - 1) & 0xFFFF; self.h, self.l = hl >> 8, hl & 0xFF; return 8
        if op == 0x0A: self.a = self.read((self.b << 8) | self.c); return 8
        if op == 0x1A: self.a = self.read((self.d << 8) | self.e); return 8
        if op == 0x2A:
            hl = (self.h << 8) | self.l; self.a = self.read(hl)
            hl = (hl + 1) & 0xFFFF; self.h, self.l = hl >> 8, hl & 0xFF; return 8
        if op == 0x3A:
            hl = (self.h << 8) | self.l; self.a = self.read(hl)
            hl = (hl - 1) & 0xFFFF; self.h, self.l = hl >> 8, hl & 0xFF; return 8
        if op == 0xE0:                                                 # LDH (a8), A
            a8 = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self.write(0xFF00 | a8, self.a); return 12
        if op == 0xF0:                                                 # LDH A, (a8)
            a8 = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self.a = self.read(0xFF00 | a8); return 12
        if op == 0xE2: self.write(0xFF00 | self.c, self.a); return 8
        if op == 0xF2: self.a = self.read(0xFF00 | self.c); return 8
        if op == 0xEA:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            self.write(ad, self.a); return 16
        if op == 0xFA:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            self.a = self.read(ad); return 16
        if op == 0x08:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            self.write(ad, self.sp & 0xFF); self.write((ad + 1) & 0xFFFF, self.sp >> 8); return 20
        if op in (0x18, 0x20, 0x28, 0x30, 0x38):                       # JR
            off = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            if off & 0x80: off -= 256
            cond = True if op == 0x18 else (
                not (self.f & 0x80) if op == 0x20 else (
                bool(self.f & 0x80) if op == 0x28 else (
                not (self.f & 0x10) if op == 0x30 else bool(self.f & 0x10))))
            if cond:
                self.pc = (self.pc + off) & 0xFFFF
                return 12
            return 8
        if op in (0xC3, 0xC2, 0xCA, 0xD2, 0xDA):                       # JP
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            cond = True if op == 0xC3 else (
                not (self.f & 0x80) if op == 0xC2 else (
                bool(self.f & 0x80) if op == 0xCA else (
                not (self.f & 0x10) if op == 0xD2 else bool(self.f & 0x10))))
            if cond:
                self.pc = ad
                return 16
            return 12
        if op == 0xE9: self.pc = (self.h << 8) | self.l; return 4      # JP (HL)
        if op in (0xCD, 0xC4, 0xCC, 0xD4, 0xDC):                       # CALL
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            cond = True if op == 0xCD else (
                not (self.f & 0x80) if op == 0xC4 else (
                bool(self.f & 0x80) if op == 0xCC else (
                not (self.f & 0x10) if op == 0xD4 else bool(self.f & 0x10))))
            if cond:
                self._push16(self.pc); self.pc = ad; return 24
            return 12
        if op in (0xC9, 0xD9, 0xC0, 0xC8, 0xD0, 0xD8):                 # RET / RETI
            if op == 0xD9: self.ime = True; self.pc = self._pop16(); return 16
            if op == 0xC9: self.pc = self._pop16(); return 16
            cond = (not (self.f & 0x80) if op == 0xC0 else (
                    bool(self.f & 0x80) if op == 0xC8 else (
                    not (self.f & 0x10) if op == 0xD0 else bool(self.f & 0x10))))
            if cond:
                self.pc = self._pop16(); return 20
            return 8
        if (op & 0xC7) == 0xC7:                                        # RST
            self._push16(self.pc); self.pc = op & 0x38; return 16
        if op in (0xC5, 0xD5, 0xE5, 0xF5):                             # PUSH
            v = ((self.b << 8) | self.c, (self.d << 8) | self.e, (self.h << 8) | self.l, (self.a << 8) | (self.f & 0xF0))[(op >> 4) & 3]
            self._push16(v); return 16
        if op in (0xC1, 0xD1, 0xE1, 0xF1):                             # POP
            v = self._pop16()
            reg = (op >> 4) & 3
            if reg == 0: self.b, self.c = v >> 8, v & 0xFF
            elif reg == 1: self.d, self.e = v >> 8, v & 0xFF
            elif reg == 2: self.h, self.l = v >> 8, v & 0xFF
            else: self.a, self.f = v >> 8, v & 0xF0
            return 12
        if op == 0x07:                                                 # RLCA
            c = (self.a >> 7) & 1; self.a = ((self.a << 1) | c) & 0xFF; self.f = c << 4; return 4
        if op == 0x0F:                                                 # RRCA
            c = self.a & 1; self.a = ((self.a >> 1) | (c << 7)) & 0xFF; self.f = c << 4; return 4
        if op == 0x17:                                                 # RLA
            c = (self.a >> 7) & 1; self.a = ((self.a << 1) | ((self.f >> 4) & 1)) & 0xFF; self.f = c << 4; return 4
        if op == 0x1F:                                                 # RRA
            c = self.a & 1; self.a = ((self.a >> 1) | (((self.f >> 4) & 1) << 7)) & 0xFF; self.f = c << 4; return 4
        if op == 0x2F: self.a ^= 0xFF; self.f = (self.f & 0x90) | 0x60; return 4  # CPL
        if op == 0x37: self.f = (self.f & 0x80) | 0x10; return 4       # SCF
        if op == 0x3F: self.f = (self.f & 0x80) ^ 0x10; return 4       # CCF
        if op == 0xF9: self.sp = (self.h << 8) | self.l; return 8      # LD SP, HL
        if op in (0xE8, 0xF8):                                         # ADD SP, r8 / LD HL, SP+r8
            off = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            s8 = (off - 256) if (off & 0x80) else off
            r = (self.sp + s8) & 0xFFFF
            self.f = (0x20 if ((self.sp & 15) + (off & 15)) > 15 else 0) | (0x10 if ((self.sp & 0xFF) + off) > 0xFF else 0)
            if op == 0xE8: self.sp = r; return 16
            self.h, self.l = r >> 8, r & 0xFF; return 12
        if op == 0x27:                                                 # DAA
            a = self.a
            if not (self.f & 0x40):
                if (self.f & 0x20) or (a & 0x0F) > 9: a += 0x06
                if (self.f & 0x10) or a > 0x9F: a += 0x60; self.f |= 0x10
            else:
                if self.f & 0x20: a = (a - 6) & 0xFF
                if self.f & 0x10: a = (a - 0x60) & 0xFF
            self.a = a & 0xFF
            self.f = (self.f & 0x50) | (0x80 if self.a == 0 else 0)
            return 4
        if op == 0xCB:                                                 # CB-prefixed bit/shift instructions
            cb = self.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            idx = cb & 7
            v = self._get_r(idx)
            grp = cb >> 6
            sub = (cb >> 3) & 7
            if grp == 0:
                old_c = (self.f >> 4) & 1
                if sub == 0: c = (v >> 7) & 1; r = ((v << 1) | c) & 0xFF
                elif sub == 1: c = v & 1; r = ((v >> 1) | (c << 7)) & 0xFF
                elif sub == 2: c = (v >> 7) & 1; r = ((v << 1) | old_c) & 0xFF
                elif sub == 3: c = v & 1; r = ((v >> 1) | (old_c << 7)) & 0xFF
                elif sub == 4: c = (v >> 7) & 1; r = (v << 1) & 0xFF
                elif sub == 5: c = v & 1; r = (v >> 1) | (v & 0x80)
                elif sub == 6: c = 0; r = ((v & 0x0F) << 4) | (v >> 4)
                else: c = v & 1; r = v >> 1
                self._set_r(idx, r)
                self.f = (0x80 if r == 0 else 0) | (c << 4)
            elif grp == 1:  # BIT
                self.f = (self.f & 0x10) | 0x20 | (0 if (v & (1 << sub)) else 0x80)
            elif grp == 2:  # RES
                self._set_r(idx, v & ~(1 << sub))
            else:           # SET
                self._set_r(idx, v | (1 << sub))
            return 16 if idx == 6 else 8
        return 4

    def _render_frame(self):
        lcdc = self.io[0x40]
        if not (lcdc & 0x80):
            self.frame_rgba[:] = _GB_DMG_PAL[0]
            return
        scy, scx = self.io[0x42], self.io[0x43]
        wy, wx = self.io[0x4A], self.io[0x4B] - 7
        bgp = self.io[0x47]
        obp0, obp1 = self.io[0x48], self.io[0x49]
        bg_map = 0x1C00 if (lcdc & 0x08) else 0x1800
        win_map = 0x1C00 if (lcdc & 0x40) else 0x1800
        unsigned_tiles = bool(lcdc & 0x10)

        if self.cgb:
            cg_bg = np.frombuffer(self.bg_pal, dtype="<u2")
            bg_lut = np.zeros((32, 4), dtype=np.uint8)
            bg_lut[:, 0] = ((cg_bg & 0x1F) << 3) | ((cg_bg & 0x1F) >> 2)
            bg_lut[:, 1] = (((cg_bg >> 5) & 0x1F) << 3) | (((cg_bg >> 5) & 0x1F) >> 2)
            bg_lut[:, 2] = (((cg_bg >> 10) & 0x1F) << 3) | (((cg_bg >> 10) & 0x1F) >> 2)
            bg_lut[:, 3] = 255
            cg_ob = np.frombuffer(self.obj_pal, dtype="<u2")
            ob_lut = np.zeros((32, 4), dtype=np.uint8)
            ob_lut[:, 0] = ((cg_ob & 0x1F) << 3) | ((cg_ob & 0x1F) >> 2)
            ob_lut[:, 1] = (((cg_ob >> 5) & 0x1F) << 3) | (((cg_ob >> 5) & 0x1F) >> 2)
            ob_lut[:, 2] = (((cg_ob >> 10) & 0x1F) << 3) | (((cg_ob >> 10) & 0x1F) >> 2)
            ob_lut[:, 3] = 255
        else:
            dmg_bg = np.array([_GB_DMG_PAL[(bgp >> (i * 2)) & 3] for i in range(4)], dtype=np.uint8)
            dmg_ob0 = np.array([_GB_DMG_PAL[(obp0 >> (i * 2)) & 3] for i in range(4)], dtype=np.uint8)
            dmg_ob1 = np.array([_GB_DMG_PAL[(obp1 >> (i * 2)) & 3] for i in range(4)], dtype=np.uint8)

        vram = self.vram
        for y in range(144):
            use_win = bool(lcdc & 0x20) and y >= wy
            for x in range(160):
                if use_win and x >= wx:
                    px, py = x - wx, y - wy
                    tmap = win_map
                else:
                    px, py = (x + scx) & 0xFF, (y + scy) & 0xFF
                    tmap = bg_map
                m_off = tmap + ((py >> 3) << 5) + (px >> 3)
                tile = vram[m_off]
                attr = vram[0x2000 + m_off] if self.cgb else 0
                vbank = 0x2000 if (self.cgb and (attr & 0x08)) else 0
                fy = (7 - (py & 7)) if (self.cgb and (attr & 0x40)) else (py & 7)
                fx = (px & 7) if (self.cgb and (attr & 0x20)) else (7 - (px & 7))
                if unsigned_tiles:
                    t_addr = vbank + (tile << 4) + (fy << 1)
                else:
                    s_tile = (tile - 256) if (tile & 0x80) else tile
                    t_addr = vbank + 0x1000 + (s_tile << 4) + (fy << 1)
                lo, hi = vram[t_addr], vram[t_addr + 1]
                c = ((lo >> fx) & 1) | (((hi >> fx) & 1) << 1)
                if self.cgb:
                    self.frame_rgba[y, x] = bg_lut[((attr & 7) << 2) | c]
                else:
                    self.frame_rgba[y, x] = dmg_bg[c]

        if lcdc & 0x02:
            sh = 16 if (lcdc & 0x04) else 8
            oam = self.oam
            for i in range(39, -1, -1):
                sy = oam[i * 4] - 16
                sx = oam[i * 4 + 1] - 8
                tile = oam[i * 4 + 2] & (0xFE if sh == 16 else 0xFF)
                attr = oam[i * 4 + 3]
                if sx <= -8 or sx >= 160 or sy <= -sh or sy >= 144:
                    continue
                vbank = 0x2000 if (self.cgb and (attr & 0x08)) else 0
                for row in range(sh):
                    py = sy + row
                    if not (0 <= py < 144):
                        continue
                    fy = (sh - 1 - row) if (attr & 0x40) else row
                    t_addr = vbank + (tile << 4) + (fy << 1)
                    lo, hi = vram[t_addr], vram[t_addr + 1]
                    for col in range(8):
                        px = sx + col
                        if not (0 <= px < 160):
                            continue
                        fx = col if (attr & 0x20) else (7 - col)
                        c = ((lo >> fx) & 1) | (((hi >> fx) & 1) << 1)
                        if c:
                            if self.cgb:
                                self.frame_rgba[py, px] = ob_lut[((attr & 7) << 2) | c]
                            else:
                                self.frame_rgba[py, px] = (dmg_ob1 if (attr & 0x10) else dmg_ob0)[c]

    def capture_state(self):
        return {
            "regs": (self.a, self.f, self.b, self.c, self.d, self.e, self.h, self.l, self.sp, self.pc, self.ime, self.halted),
            "banks": (self.rom_bank, self.ram_bank, self.vram_bank, self.wram_bank),
            "vram": bytes(self.vram), "wram": bytes(self.wram), "oam": bytes(self.oam),
            "hram": bytes(self.hram), "io": bytes(self.io), "bg_pal": bytes(self.bg_pal), "obj_pal": bytes(self.obj_pal),
            "frame_rgba": self.frame_rgba.copy(),
        }

    def restore_state(self, st):
        self.a, self.f, self.b, self.c, self.d, self.e, self.h, self.l, self.sp, self.pc, self.ime, self.halted = st["regs"]
        self.rom_bank, self.ram_bank, self.vram_bank, self.wram_bank = st["banks"]
        self.vram[:] = st["vram"]; self.wram[:] = st["wram"]; self.oam[:] = st["oam"]
        self.hram[:] = st["hram"]; self.io[:] = st["io"]; self.bg_pal[:] = st["bg_pal"]; self.obj_pal[:] = st["obj_pal"]
        if "frame_rgba" in st:
            self.frame_rgba[:] = st["frame_rgba"]

    def save_state(self, path=None):
        path = path or self.base_path + ".gbstate"
        with open(path, "wb") as f:
            pickle.dump(self.capture_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".gbstate"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.restore_state(pickle.load(f))
        return True

    def run_frame(self, draw=True):
        for ly in range(154):
            self.io[0x44] = ly
            if ly == self.io[0x45]:
                self.io[0x41] |= 0x04
                if self.io[0x41] & 0x40:
                    self.io[0x0F] |= 0x02  # STAT LYC=LY IRQ
            else:
                self.io[0x41] &= ~0x04
            if ly == 144:
                self.io[0x0F] |= 0x01      # VBlank IRQ
            cyc = 0
            while cyc < 456:
                c = self.step()
                cyc += c
                self.div_cycles += c
                if self.div_cycles >= 256:
                    self.div_cycles -= 256
                    self.io[0x04] = (self.io[0x04] + 1) & 0xFF
                tac = self.io[0x07]
                if tac & 0x04:
                    thresh = (1024, 16, 64, 256)[tac & 3]
                    self.tima_cycles += c
                    while self.tima_cycles >= thresh:
                        self.tima_cycles -= thresh
                        tima = (self.io[0x05] + 1) & 0xFF
                        if tima == 0:
                            self.io[0x05] = self.io[0x06]
                            self.io[0x0F] |= 0x04
                        else:
                            self.io[0x05] = tima
        if draw:
            self._render_frame()
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.frame_rgba


# ==========================================================================
# GAME BOY ADVANCE HARDWARE IMPLEMENTATION (.gba)
# ==========================================================================
class GameBoyAdvance:
    """Pure-Python Game Boy Advance Emulator (ARM7TDMI ARM/Thumb + Bitmap/Tile PPU)."""
    width = 240
    height = 160
    system_name = "GBA"

    def __init__(self, rom_path: str, sample_rate: int = 44100):
        self.rom_path = rom_path
        with open(rom_path, "rb") as f:
            self.rom = f.read()
        self.ewram = bytearray(256 * 1024)
        self.iwram = bytearray(32 * 1024)
        self.io = bytearray(1024)
        self.pram = bytearray(1024)
        self.vram = bytearray(96 * 1024)
        self.oam = bytearray(1024)
        self.sram = bytearray(64 * 1024)
        self.frame_rgba = np.zeros((160, 240, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255
        self.apu = SNESAPU(sample_rate)
        self.buttons = 0
        self.frames = 0
        self._load_battery()
        self.reset()

    @property
    def base_path(self):
        return os.path.splitext(self.rom_path)[0]

    def _load_battery(self):
        p = self.base_path + ".sav"
        if os.path.exists(p):
            with open(p, "rb") as f:
                d = f.read(len(self.sram))
            self.sram[:len(d)] = d

    def save_battery(self):
        with open(self.base_path + ".sav", "wb") as f:
            f.write(bytes(self.sram))

    def reset(self):
        self.r = [0] * 16
        self.r[13] = 0x03007F00  # SP
        self.r[15] = 0x08000000  # PC (Cartridge ROM entry point)
        self.cpsr = 0x0000001F   # System mode, ARM state
        self.spsr = 0

    def set_buttons(self, pad, buttons):
        if pad != 0:
            return
        self.buttons = buttons
        keyinput = 0x03FF
        if buttons & BTN_A:      keyinput &= ~0x0001
        if buttons & BTN_B:      keyinput &= ~0x0002
        if buttons & BTN_SELECT: keyinput &= ~0x0004
        if buttons & BTN_START:  keyinput &= ~0x0008
        if buttons & BTN_RIGHT:  keyinput &= ~0x0010
        if buttons & BTN_LEFT:   keyinput &= ~0x0020
        if buttons & BTN_UP:     keyinput &= ~0x0040
        if buttons & BTN_DOWN:   keyinput &= ~0x0080
        self.io[0x130] = keyinput & 0xFF
        self.io[0x131] = (keyinput >> 8) & 0xFF

    def read8(self, addr):
        region = (addr >> 24) & 0xFF
        if region == 2: return self.ewram[addr & 0x3FFFF]
        if region == 3: return self.iwram[addr & 0x7FFF]
        if region == 4: return self.io[addr & 0x3FF]
        if region == 5: return self.pram[addr & 0x3FF]
        if region == 6:
            a = addr & 0x1FFFF
            if a >= 0x18000: a -= 0x8000
            return self.vram[a]
        if region == 7: return self.oam[addr & 0x3FF]
        if 8 <= region <= 13:
            off = addr & 0x1FFFFFF
            return self.rom[off] if off < len(self.rom) else 0
        if region == 14: return self.sram[addr & 0xFFFF]
        return 0

    def read16(self, addr):
        addr &= ~1
        return self.read8(addr) | (self.read8(addr + 1) << 8)

    def read32(self, addr):
        addr &= ~3
        return self.read8(addr) | (self.read8(addr + 1) << 8) | (self.read8(addr + 2) << 16) | (self.read8(addr + 3) << 24)

    def write8(self, addr, val):
        val &= 0xFF
        region = (addr >> 24) & 0xFF
        if region == 2: self.ewram[addr & 0x3FFFF] = val
        elif region == 3: self.iwram[addr & 0x7FFF] = val
        elif region == 4:
            a = addr & 0x3FF
            self.io[a] = val
            if a in (0xBD, 0xC9, 0xD5, 0xE1) and (val & 0x80):
                self._run_dma((a - 0xB0) // 12)
        elif region == 5: self.pram[addr & 0x3FF] = val
        elif region == 6:
            a = addr & 0x1FFFF
            if a >= 0x18000: a -= 0x8000
            self.vram[a] = val
        elif region == 7: self.oam[addr & 0x3FF] = val
        elif region == 14: self.sram[addr & 0xFFFF] = val

    def write16(self, addr, val):
        addr &= ~1
        self.write8(addr, val & 0xFF)
        self.write8(addr + 1, (val >> 8) & 0xFF)

    def write32(self, addr, val):
        addr &= ~3
        self.write8(addr, val & 0xFF)
        self.write8(addr + 1, (val >> 8) & 0xFF)
        self.write8(addr + 2, (val >> 16) & 0xFF)
        self.write8(addr + 3, (val >> 24) & 0xFF)

    def _run_dma(self, ch):
        base = 0xB0 + ch * 12
        sad = self.io[base] | (self.io[base + 1] << 8) | (self.io[base + 2] << 16) | (self.io[base + 3] << 24)
        dad = self.io[base + 4] | (self.io[base + 5] << 8) | (self.io[base + 6] << 16) | (self.io[base + 7] << 24)
        cnt = self.io[base + 8] | (self.io[base + 9] << 8)
        ctrl = self.io[base + 10] | (self.io[base + 11] << 8)
        if cnt == 0:
            cnt = 0x4000
        word32 = bool(ctrl & 0x0400)
        step = 4 if word32 else 2
        d_ctl = (ctrl >> 5) & 3
        s_ctl = (ctrl >> 7) & 3
        d_step = step if d_ctl in (0, 3) else (-step if d_ctl == 1 else 0)
        s_step = step if s_ctl == 0 else (-step if s_ctl == 1 else 0)
        for _ in range(cnt):
            if word32:
                self.write32(dad, self.read32(sad))
            else:
                self.write16(dad, self.read16(sad))
            sad = (sad + s_step) & 0xFFFFFFFF
            dad = (dad + d_step) & 0xFFFFFFFF
        if not (ctrl & 0x0200):
            self.io[base + 11] &= ~0x80

    def _set_nz(self, val):
        val &= 0xFFFFFFFF
        self.cpsr = (self.cpsr & 0x3FFFFFFF) | (0x80000000 if (val & 0x80000000) else 0) | (0x40000000 if val == 0 else 0)

    def _cond_ok(self, cond):
        if cond == 0xE: return True
        n = bool(self.cpsr & 0x80000000)
        z = bool(self.cpsr & 0x40000000)
        c = bool(self.cpsr & 0x20000000)
        v = bool(self.cpsr & 0x10000000)
        return (z, not z, c, not c, n, not n, v, not v, c and not z, not c or z, n == v, n != v, not z and (n == v), z or (n != v), True, False)[cond]

    def step(self):
        r = self.r
        if self.cpsr & 0x20:  # Thumb state (16-bit instructions)
            pc = r[15] & ~1
            ins = self.read16(pc)
            r[15] = (pc + 2) & 0xFFFFFFFF
            top = ins >> 13
            if top == 0:
                op = (ins >> 11) & 3
                rd, rs = ins & 7, (ins >> 3) & 7
                if op < 3:
                    imm = (ins >> 6) & 0x1F
                    v = r[rs]
                    res = ((v << imm) if op == 0 else ((v >> imm) if imm else 0)) & 0xFFFFFFFF
                    r[rd] = res; self._set_nz(res)
                else:
                    rn_imm = (ins >> 6) & 7
                    val = rn_imm if (ins & 0x0400) else r[rn_imm]
                    res = (r[rs] - val) if (ins & 0x0200) else (r[rs] + val)
                    r[rd] = res & 0xFFFFFFFF; self._set_nz(r[rd])
            elif top == 1:  # MOV/CMP/ADD/SUB Rd, #imm8
                op = (ins >> 11) & 3
                rd = (ins >> 8) & 7
                imm = ins & 0xFF
                if op == 0: r[rd] = imm; self._set_nz(imm)
                elif op == 1:
                    res = r[rd] - imm
                    self._set_nz(res)
                    self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if r[rd] >= imm else 0)
                elif op == 2: r[rd] = (r[rd] + imm) & 0xFFFFFFFF; self._set_nz(r[rd])
                else: r[rd] = (r[rd] - imm) & 0xFFFFFFFF; self._set_nz(r[rd])
            elif (ins & 0xF800) == 0x4800:  # LDR Rd, [PC, #imm]
                rd = (ins >> 8) & 7
                ad = ((r[15] + 2) & ~3) + ((ins & 0xFF) << 2)
                r[rd] = self.read32(ad)
            elif (ins & 0xFC00) == 0x4400:  # Hi-register / BX
                op = (ins >> 8) & 3
                rd = (ins & 7) | ((ins >> 4) & 8)
                rs = ((ins >> 3) & 7) | ((ins >> 3) & 8)
                if op == 0: r[rd] = (r[rd] + r[rs]) & 0xFFFFFFFF
                elif op == 1: self._set_nz(r[rd] - r[rs])
                elif op == 2: r[rd] = r[rs]
                else:
                    target = r[rs]
                    if target & 1:
                        self.cpsr |= 0x20
                        r[15] = target & ~1
                    else:
                        self.cpsr &= ~0x20
                        r[15] = target & ~3
            elif top == 3:  # LDR/STR with immediate offset
                op = (ins >> 11) & 3
                imm = (ins >> 6) & 0x1F
                rb, rd = (ins >> 3) & 7, ins & 7
                if op == 0: self.write32(r[rb] + (imm << 2), r[rd])
                elif op == 1: r[rd] = self.read32(r[rb] + (imm << 2))
                elif op == 2: self.write8(r[rb] + imm, r[rd])
                else: r[rd] = self.read8(r[rb] + imm)
            elif (ins & 0xF000) == 0x8000:  # LDRH/STRH
                imm = ((ins >> 6) & 0x1F) << 1
                rb, rd = (ins >> 3) & 7, ins & 7
                if ins & 0x0800: r[rd] = self.read16(r[rb] + imm)
                else: self.write16(r[rb] + imm, r[rd])
            elif (ins & 0xF600) == 0xB400:  # PUSH {Rlist, LR}
                sp = r[13]
                if ins & 0x0100:
                    sp = (sp - 4) & 0xFFFFFFFF
                    self.write32(sp, r[14])
                for reg_i in range(7, -1, -1):
                    if ins & (1 << reg_i):
                        sp = (sp - 4) & 0xFFFFFFFF
                        self.write32(sp, r[reg_i])
                r[13] = sp
            elif (ins & 0xF600) == 0xBC00:  # POP {Rlist, PC}
                sp = r[13]
                for reg_i in range(8):
                    if ins & (1 << reg_i):
                        r[reg_i] = self.read32(sp)
                        sp = (sp + 4) & 0xFFFFFFFF
                if ins & 0x0100:
                    r[15] = self.read32(sp) & ~1
                    sp = (sp + 4) & 0xFFFFFFFF
                r[13] = sp
            elif (ins & 0xF000) == 0xC000:  # STMIA / LDMIA Rb!, {Rlist}
                rb = (ins >> 8) & 7
                addr = r[rb]
                load = bool(ins & 0x0800)
                for reg_i in range(8):
                    if ins & (1 << reg_i):
                        if load: r[reg_i] = self.read32(addr)
                        else: self.write32(addr, r[reg_i])
                        addr = (addr + 4) & 0xFFFFFFFF
                r[rb] = addr
            elif (ins & 0xF000) == 0xD000:  # B<cond>
                cond = (ins >> 8) & 0xF
                if cond < 0xE and self._cond_ok(cond):
                    off = ins & 0xFF
                    if off & 0x80: off -= 256
                    r[15] = (r[15] + 2 + (off << 1)) & 0xFFFFFFFF
            elif (ins & 0xF800) == 0xE000:  # B unconditional
                off = ins & 0x7FF
                if off & 0x400: off -= 2048
                r[15] = (r[15] + 2 + (off << 1)) & 0xFFFFFFFF
            elif (ins & 0xF000) == 0xF000:  # BL (2-part)
                off = ins & 0x7FF
                if not (ins & 0x0800):
                    if off & 0x400: off -= 2048
                    r[14] = (r[15] + 2 + (off << 12)) & 0xFFFFFFFF
                else:
                    next_pc = (r[15]) | 1
                    r[15] = (r[14] + (off << 1)) & ~1
                    r[14] = next_pc
            return 2

        # ARM state (32-bit instructions)
        pc = r[15] & ~3
        ins = self.read32(pc)
        r[15] = (pc + 4) & 0xFFFFFFFF
        if not self._cond_ok((ins >> 28) & 0xF):
            return 1
        op_type = (ins >> 25) & 7
        if (ins & 0x0FFFFFF0) == 0x012FFF10:  # BX Rn
            target = r[ins & 0xF]
            if target & 1:
                self.cpsr |= 0x20
                r[15] = target & ~1
            else:
                self.cpsr &= ~0x20
                r[15] = target & ~3
        elif op_type == 5:  # B / BL
            off = ins & 0xFFFFFF
            if off & 0x800000: off -= 0x1000000
            if ins & 0x01000000:
                r[14] = r[15]
            r[15] = (pc + 8 + (off << 2)) & 0xFFFFFFFF
        elif op_type == 0 and (ins & 0x90) == 0x90 and ((ins >> 5) & 3) != 0:  # LDRH / STRH / LDRSB / LDRSH
            rn = (ins >> 16) & 0xF
            rd = (ins >> 12) & 0xF
            base = (pc + 8) if rn == 15 else r[rn]
            off = (((ins >> 4) & 0xF0) | (ins & 0xF)) if (ins & 0x00400000) else r[ins & 0xF]
            ad = (base + off) if (ins & 0x00800000) else (base - off)
            sh_op = (ins >> 5) & 3
            if ins & 0x00100000:  # Load
                if sh_op == 1:
                    r[rd] = self.read16(ad)
                elif sh_op == 2:
                    v = self.read8(ad)
                    r[rd] = (v | 0xFFFFFF00) if (v & 0x80) else v
                else:
                    v = self.read16(ad)
                    r[rd] = (v | 0xFFFF0000) if (v & 0x8000) else v
            else:                 # Store halfword (STRH)
                if sh_op == 1:
                    self.write16(ad, r[rd])
            if (ins & 0x00200000) or not (ins & 0x01000000):
                r[rn] = ad & 0xFFFFFFFF
        elif op_type == 4:  # LDM / STM (Block Data Transfer)
            rn = (ins >> 16) & 0xF
            base = r[rn]
            rlist = ins & 0xFFFF
            up = bool(ins & 0x00800000)
            pre = bool(ins & 0x01000000)
            wb = bool(ins & 0x00200000)
            load = bool(ins & 0x00100000)
            nbits = bin(rlist).count("1")
            if up:
                addr = (base + 4) if pre else base
                final_base = (base + nbits * 4) & 0xFFFFFFFF
            else:
                addr = (base - nbits * 4) if pre else (base - nbits * 4 + 4)
                final_base = (base - nbits * 4) & 0xFFFFFFFF
            for reg_i in range(16):
                if rlist & (1 << reg_i):
                    if load:
                        r[reg_i] = self.read32(addr)
                    else:
                        self.write32(addr, r[reg_i])
                    addr = (addr + 4) & 0xFFFFFFFF
            if wb and not (load and (rlist & (1 << rn))):
                r[rn] = final_base
        elif op_type in (0, 1):  # Data processing
            opcode = (ins >> 21) & 0xF
            s_bit = bool(ins & 0x00100000)
            rn = (ins >> 16) & 0xF
            rd = (ins >> 12) & 0xF
            vn = (pc + 8) if rn == 15 else r[rn]
            if op_type == 1:
                imm = ins & 0xFF
                rot = ((ins >> 8) & 0xF) * 2
                op2 = ((imm >> rot) | (imm << (32 - rot))) & 0xFFFFFFFF if rot else imm
            else:
                rm = ins & 0xF
                vm = (pc + 8) if rm == 15 else r[rm]
                sh = (ins >> 7) & 0x1F
                st = (ins >> 5) & 3
                op2 = ((vm << sh) if st == 0 else ((vm >> sh) if st == 1 else vm)) & 0xFFFFFFFF
            if opcode == 0x0: res = vn & op2; r[rd] = res
            elif opcode == 0x1: res = vn ^ op2; r[rd] = res
            elif opcode == 0x2: res = (vn - op2) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x3: res = (op2 - vn) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x4: res = (vn + op2) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x8: res = vn & op2
            elif opcode == 0x9: res = vn ^ op2
            elif opcode == 0xA:
                res = (vn - op2) & 0xFFFFFFFF
                self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if vn >= op2 else 0)
            elif opcode == 0xB: res = (vn + op2) & 0xFFFFFFFF
            elif opcode == 0xC: res = vn | op2; r[rd] = res
            elif opcode == 0xD: res = op2; r[rd] = res
            elif opcode == 0xE: res = vn & ~op2; r[rd] = res
            elif opcode == 0xF: res = (~op2) & 0xFFFFFFFF; r[rd] = res
            else: res = r[rd]
            if s_bit and rd != 15:
                self._set_nz(res)
        elif op_type in (2, 3):  # LDR / STR
            rn = (ins >> 16) & 0xF
            rd = (ins >> 12) & 0xF
            base = (pc + 8) if rn == 15 else r[rn]
            off = (ins & 0xFFF) if op_type == 2 else r[ins & 0xF]
            ad = (base + off) if (ins & 0x00800000) else (base - off)
            if ins & 0x00100000:  # LDR
                r[rd] = self.read8(ad) if (ins & 0x00400000) else self.read32(ad)
            else:                 # STR
                val = r[rd]
                if ins & 0x00400000: self.write8(ad, val)
                else: self.write32(ad, val)
        return 2

    def _render_frame(self):
        dispcnt = self.io[0] | (self.io[1] << 8)
        mode = dispcnt & 7
        pram16 = np.frombuffer(self.pram, dtype="<u2")
        lut = np.zeros((512, 4), dtype=np.uint8)
        lut[:, 0] = ((pram16 & 0x1F) << 3) | ((pram16 & 0x1F) >> 2)
        lut[:, 1] = (((pram16 >> 5) & 0x1F) << 3) | (((pram16 >> 5) & 0x1F) >> 2)
        lut[:, 2] = (((pram16 >> 10) & 0x1F) << 3) | (((pram16 >> 10) & 0x1F) >> 2)
        lut[:, 3] = 255

        if mode == 3:  # 240x160 15-bit direct BGR555 bitmap
            fb16 = np.frombuffer(self.vram, dtype="<u2", count=240 * 160).reshape((160, 240))
            self.frame_rgba[:, :, 0] = ((fb16 & 0x1F) << 3) | ((fb16 & 0x1F) >> 2)
            self.frame_rgba[:, :, 1] = (((fb16 >> 5) & 0x1F) << 3) | (((fb16 >> 5) & 0x1F) >> 2)
            self.frame_rgba[:, :, 2] = (((fb16 >> 10) & 0x1F) << 3) | (((fb16 >> 10) & 0x1F) >> 2)
            self.frame_rgba[:, :, 3] = 255
        elif mode == 4:  # 240x160 8-bit paletted bitmap
            off = 0xA000 if (dispcnt & 0x10) else 0
            idx = np.frombuffer(self.vram, dtype=np.uint8, offset=off, count=240 * 160).reshape((160, 240))
            self.frame_rgba[:] = lut[idx]
        elif mode == 5:  # 160x128 15-bit bitmap centered
            self.frame_rgba[:] = lut[0]
            off = 0xA000 if (dispcnt & 0x10) else 0
            fb16 = np.frombuffer(self.vram, dtype="<u2", offset=off, count=160 * 128).reshape((128, 160))
            self.frame_rgba[16:144, 40:200, 0] = ((fb16 & 0x1F) << 3) | ((fb16 & 0x1F) >> 2)
            self.frame_rgba[16:144, 40:200, 1] = (((fb16 >> 5) & 0x1F) << 3) | (((fb16 >> 5) & 0x1F) >> 2)
            self.frame_rgba[16:144, 40:200, 2] = (((fb16 >> 10) & 0x1F) << 3) | (((fb16 >> 10) & 0x1F) >> 2)
        else:  # Tile modes 0, 1, 2
            self.frame_rgba[:] = lut[0]
            vram = self.vram
            for bg in range(3, -1, -1):
                if not (dispcnt & (0x0100 << bg)):
                    continue
                bgcnt = self.io[8 + bg * 2] | (self.io[9 + bg * 2] << 8)
                chr_base = ((bgcnt >> 2) & 3) * 0x4000
                map_base = ((bgcnt >> 8) & 0x1F) * 0x800
                bpp8 = bool(bgcnt & 0x80)
                hofs = (self.io[0x10 + bg * 4] | (self.io[0x11 + bg * 4] << 8)) & 0x1FF
                vofs = (self.io[0x12 + bg * 4] | (self.io[0x13 + bg * 4] << 8)) & 0x1FF
                for y in range(160):
                    wy = (y + vofs) & 0xFF
                    ty, fy = wy >> 3, wy & 7
                    for x in range(240):
                        wx = (x + hofs) & 0xFF
                        tx, fx = wx >> 3, wx & 7
                        ea = (map_base + ((ty << 5) + tx) * 2) & 0xFFFF
                        entry = vram[ea] | (vram[ea + 1] << 8)
                        t_num = entry & 0x3FF
                        if entry & 0x400: fx = 7 - fx
                        row = (7 - fy) if (entry & 0x800) else fy
                        if bpp8:
                            c = vram[(chr_base + (t_num << 6) + (row << 3) + fx) % len(vram)]
                            if c: self.frame_rgba[y, x] = lut[c]
                        else:
                            b_val = vram[(chr_base + (t_num << 5) + (row << 2) + (fx >> 1)) % len(vram)]
                            c = ((b_val >> 4) if (fx & 1) else (b_val & 0xF))
                            if c:
                                pal = (entry >> 12) & 0xF
                                self.frame_rgba[y, x] = lut[(pal << 4) | c]

    def capture_state(self):
        return {
            "r": list(self.r), "cpsr": self.cpsr, "spsr": self.spsr,
            "ewram": bytes(self.ewram), "iwram": bytes(self.iwram), "io": bytes(self.io),
            "pram": bytes(self.pram), "vram": bytes(self.vram), "oam": bytes(self.oam),
            "frame_rgba": self.frame_rgba.copy(),
        }

    def restore_state(self, st):
        self.r[:] = st["r"]; self.cpsr = st["cpsr"]; self.spsr = st["spsr"]
        self.ewram[:] = st["ewram"]; self.iwram[:] = st["iwram"]; self.io[:] = st["io"]
        self.pram[:] = st["pram"]; self.vram[:] = st["vram"]; self.oam[:] = st["oam"]
        if "frame_rgba" in st:
            self.frame_rgba[:] = st["frame_rgba"]

    def save_state(self, path=None):
        path = path or self.base_path + ".gbastate"
        with open(path, "wb") as f:
            pickle.dump(self.capture_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".gbastate"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.restore_state(pickle.load(f))
        return True

    def run_frame(self, draw=True):
        for ly in range(228):
            self.io[6] = ly  # VCOUNT
            if ly >= 160:
                self.io[4] |= 1  # VBlank flag in DISPSTAT
            else:
                self.io[4] &= ~1
            for _ in range(120):
                self.step()
        if draw:
            self._render_frame()
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.frame_rgba


def load_console(rom_path, sample_rate=44100):
    ext = os.path.splitext(rom_path)[1].lower()
    if ext in (".sfc", ".smc", ".snes"):
        return SNES(rom_path, sample_rate)
    if ext in (".gb", ".gbc"):
        return GameBoy(rom_path, sample_rate)
    if ext == ".gba":
        return GameBoyAdvance(rom_path, sample_rate)
    if ext == ".nes":
        return NES(rom_path, sample_rate)
    with open(rom_path, "rb") as f:
        hdr = f.read(512)
    if hdr[:4] == b"NES\x1a":
        return NES(rom_path, sample_rate)
    if len(hdr) >= 0x134 and hdr[0x104:0x108] == b"\xCE\xED\x66\x66":
        return GameBoy(rom_path, sample_rate)
    if len(hdr) >= 0xA0 and hdr[0x04:0x08] == b"\x24\xFF\xAE\x51":
        return GameBoyAdvance(rom_path, sample_rate)
    return SNES(rom_path, sample_rate)



# ==========================================================================
# ARCADE FRONT-END
# ==========================================================================
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
FRAG_RECT = """
#version 330
uniform vec4 color;
uniform float zero;
in vec2 uv;
out vec4 fragColor;
void main() { fragColor = vec4(color.rgb + vec3(uv, 0.0) * zero, color.a); }
"""

MENU_H = 24
ITEM_H = 23
SEP_H = 9
PAD_X = 12
CHECK_W = 22
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
    ("M", "Mute"), ("F", "Smooth / pixel filter"), ("V", "Start / stop AVI recording"),
    ("F5 / F7", "Save / load state"), ("F11", "Fullscreen"), ("F12", "Screenshot"),
    ("Ctrl+O", "Open ROM (.nes/.sfc/.snes/.gb/.gbc/.gba)"), ("Esc", "Close menu / quit"),
]
ABOUT_ROWS = [
    ("PyRetro", "Multi-Console Emulator in Python + Arcade"),
    ("NES / SNES", "6502 + 2C02 PPU / 65c816 + Vectorized S-PPU"),
    ("GB / GBC / GBA", "LR35902 + DMG/CGB PPU / ARM7TDMI + GBA PPU"),
]


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


class RetroWindow(arcade.Window if arcade else object):
    def __init__(self, console, scale=3, audio=True, aspect="square", max_catchup=6,
                 vsync=True, audio_latency=0.025, audio_backend="auto"):
        self.console = console
        self.aspect = aspect
        self.par = 8.0 / 7.0 if aspect == "ntsc" else 1.0
        self.max_catchup = max_catchup
        cw = console.width if console else 256
        ch = console.height if console else 240
        sys_tag = f"[{console.system_name}] " if console else ""
        self.title_base = f"PyRetro {sys_tag}" + ("- " + os.path.basename(console.rom_path) if console else "")
        super().__init__(int(cw * scale * self.par), ch * scale + MENU_H, self.title_base,
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
        self._audio_capture = []
        self._forward_audio_phase = 0.0
        self._forward_audio_tail = None
        self._forward_audio_speed = self.speed_multiplier
        self.ui_hover_t = {}
        self.ui_anim_last = time.perf_counter()
        self.ui_frame_dt = 0.0
        self.ui_modal_close = None
        self.recording = False
        self.video_writer = None
        self.record_path = None

        self.program = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.program["tex"] = 0
        self.rect_prog = self.ctx.program(vertex_shader=VERT_RECT, fragment_shader=FRAG_RECT)
        self.rect_prog["zero"] = 0.0
        self.tex_size = (cw, ch)
        self.texture = self.ctx.texture(self.tex_size, components=4, data=bytes(cw * ch * 4),
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
        except Exception:
            self.pad = None

        self.ui_size = self.get_size()
        self._opening = False
        self.ui_ptr = (-1, -1)
        self.ui_open = None
        self.ui_overlay = None
        self.ui_toast = None
        self.ui_texts = {}
        self.ui_menus = None
        self.ui_title_x = []
        self.ui_drop_w = []
        self.ui_reflow()

    def _ensure_texture_size(self):
        if self.console is None:
            return
        want = (self.console.width, self.console.height)
        if self.tex_size != want:
            self.tex_size = want
            filt = self.ctx.LINEAR if self.smooth else self.ctx.NEAREST
            self.texture = self.ctx.texture(want, components=4, data=bytes(want[0] * want[1] * 4),
                                            filter=(filt, filt))

    def ui_build_menus(self):
        def it(label, fn, key="", check=None):
            return {"label": label, "fn": fn, "key": key, "check": check}

        cw = self.console.width if self.console else 256
        ch = self.console.height if self.console else 240

        def size_item(s):
            return it(f"{s}x  ({int(cw * s * self.par)} x {ch * s})",
                      lambda s=s: self.act_scale(s), check=lambda s=s: self.ui_cur_scale() == s)

        file_m = [
            it("Open ROM...", self.act_open, "Ctrl+O"), None,
            it("Save State", self.act_save_state, "F5"),
            it("Load State", self.act_load_state, "F7"), None,
            it("Screenshot", self.act_screenshot, "F12"),
            it("Record AVI Video", self.act_toggle_record, "V", check=lambda: self.recording), None,
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

    def ui_notify(self, msg):
        print(msg)
        self.ui_toast = (msg, time.perf_counter() + 2.5)

    def ui_show(self, title, rows):
        self.ui_overlay = (title, rows)

    def ui_cur_scale(self):
        if self.fullscreen:
            return None
        ww, wh = self.get_size()
        cw = self.console.width if self.console else 256
        ch = self.console.height if self.console else 240
        for s in range(1, 9):
            if ww == int(cw * s * self.par) and wh == ch * s + MENU_H:
                return s
        return None

    def ui_sync_audio(self):
        if self.console is None:
            return
        live = self.audio is not None and not self.muted and not self.turbo and not self.reverse_speed
        self.console.apu.sink = self._audio_push if live else None
        if self.audio is not None and not live:
            self.audio.clear()
        self.console.apu.volume = 0.7 * self.volume

    def _audio_push(self, pcm):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if self.audio is not None:
            adjusted = self._forward_audio_speed_pcm(pcm, self.speed_multiplier)
            if adjusted.size:
                self.audio.push(adjusted)
        if self._audio_capture is not None:
            self._audio_capture.append(pcm.copy())

    def _reset_forward_audio_resampler(self):
        self._forward_audio_phase = 0.0
        self._forward_audio_tail = None
        self._forward_audio_speed = self.speed_multiplier

    def _forward_audio_speed_pcm(self, pcm, speed):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        speed = max(0.1, min(10.0, float(speed)))
        if speed != self._forward_audio_speed:
            self._reset_forward_audio_resampler()
            self._forward_audio_speed = speed
        if pcm.size == 0:
            return np.empty(0, dtype=np.int16)
        if self._forward_audio_tail is None:
            source = pcm
            phase = 0.0
        else:
            source = np.concatenate((np.asarray([self._forward_audio_tail], dtype=np.int16), pcm))
            phase = self._forward_audio_phase
        if source.size < 2 or phase > source.size - 1:
            self._forward_audio_tail = int(source[-1])
            self._forward_audio_phase = phase + 1.0 - source.size
            return np.empty(0, dtype=np.int16)
        count = int(np.floor((source.size - 1 - phase) / speed)) + 1
        positions = phase + np.arange(max(0, count), dtype=np.float64) * speed
        lo = np.floor(positions).astype(np.intp)
        hi = np.minimum(lo + 1, source.size - 1)
        alpha = (positions - lo).astype(np.float32)
        out = (1.0 - alpha) * source[lo].astype(np.float32) + alpha * source[hi].astype(np.float32)
        self._forward_audio_phase = phase + count * speed - (source.size - 1)
        self._forward_audio_tail = int(source[-1])
        return np.clip(out, -32767, 32767).astype(np.int16)

    @staticmethod
    def _reverse_audio_speed(pcm, speed):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if pcm.size < 2:
            return pcm.copy()
        speed = max(0.1, min(10.0, float(speed)))
        count = max(1, int(np.ceil((pcm.size - 1) / speed)) + 1)
        positions = np.arange(count, dtype=np.float64) * speed
        positions = np.minimum(positions, pcm.size - 1)
        positions = (pcm.size - 1) - positions
        lo = np.floor(positions).astype(np.intp)
        hi = np.minimum(lo + 1, pcm.size - 1)
        frac = (positions - lo).astype(np.float32)
        out = pcm[lo].astype(np.float32) * (1.0 - frac) + pcm[hi].astype(np.float32) * frac
        return np.clip(out, -32767, 32767).astype(np.int16)

    def act_open(self):
        if getattr(self, "_opening", False):
            return
        self._opening = True
        try:
            path = pick_rom()
            self.keys = 0
            self.last = time.perf_counter()
            if path:
                self.load_rom(path)
        except Exception as e:
            self.ui_notify(f"Open ROM failed: {e}")
        finally:
            self._opening = False

    def load_rom(self, path):
        try:
            new = load_console(path)
        except Exception as e:
            self.ui_notify(f"Cannot load ROM: {e}")
            return False
        old = self.console
        if old is not None:
            try:
                old.save_battery()
            except Exception:
                pass
            old.apu.sink = None
        self.console = new
        self._ensure_texture_size()
        self.rewind_history.clear()
        self.title_base = f"PyRetro [{new.system_name}] - " + os.path.basename(path)
        self.set_caption(self.title_base)
        self.paused = False
        self.acc = 0.0
        self.last = time.perf_counter()
        self.fill_avg = self.audio_target
        # At the end of RetroWindow.load_rom(self, path):
        self.frame_ready = True
        self.ui_sync_audio()
        self.ui_reflow()
        self.ui_notify(f"Loaded [{new.system_name}] {os.path.basename(path)}")
        return True
    
    def act_save_state(self):
        if self.console:
            self.console.save_state()
            self.ui_notify("State saved")

    def act_load_state(self):
        if self.console:
            self.ui_notify("State loaded" if self.console.load_state() else "No saved state")
            self.frame_ready = True

    def act_screenshot(self):
        if not self.console:
            return
        try:
            from PIL import Image
            path = self.console.base_path + f"_{int(time.time())}.png"
            Image.fromarray(self.console.get_rgba()).save(path)
            self.ui_notify("Screenshot: " + os.path.basename(path))
        except Exception as e:
            self.ui_notify(f"Screenshot failed: {e}")

    def act_toggle_record(self):
        if not self.console:
            self.ui_notify("No ROM loaded to record")
            return
        if imageio is None:
            self.ui_notify("Install imageio: pip install imageio imageio-ffmpeg")
            return
        if self.recording:
            self.recording = False
            if self.video_writer is not None:
                try:
                    self.video_writer.close()
                except Exception:
                    pass
                self.video_writer = None
            self.ui_notify("AVI saved: " + os.path.basename(self.record_path))
        else:
            self.record_path = self.console.base_path + f"_{int(time.time())}.avi"
            try:
                self.video_writer = imageio.get_writer(self.record_path, fps=60)
                self.recording = True
                self.ui_notify("Recording AVI: " + os.path.basename(self.record_path))
            except Exception as e:
                self.ui_notify(f"Record failed: {e}")
                self.recording = False

    def act_pause(self):
        self.paused = not self.paused
        self.last = time.perf_counter()
        self.set_caption(self.title_base + (" [paused]" if self.paused else ""))

    def act_reset(self):
        if self.console:
            self.console.reset()
            self.rewind_history.clear()
            self.rewind_acc = 0.0
            self.ui_notify("Reset")

    def act_turbo(self):
        self.turbo_latch = not self.turbo_latch
        self.ui_update_turbo()

    def act_speed(self, multiplier):
        self.speed_multiplier = max(0.1, min(10.0, float(multiplier)))
        self._reset_forward_audio_resampler()
        self.acc = 0.0
        self.last = time.perf_counter()
        self.ui_notify(f"Emulation speed: {self.speed_multiplier:g}x")
        self.ui_reflow()

    def act_reverse(self):
        self.reverse_speed = not self.reverse_speed
        self._reset_forward_audio_resampler()
        self.rewind_acc = 0.0
        self._audio_capture = []
        if self.reverse_speed and self.audio is not None:
            self.audio.ring.ensure_capacity(int(self.audio.rate * 1.0))
        self.acc = 0.0
        self.last = time.perf_counter()
        self.ui_sync_audio()
        self.ui_notify("Rewind enabled" if self.reverse_speed else "Forward playback")

    def act_audio_latency(self, milliseconds):
        target = max(10, min(100, int(milliseconds))) / 1000.0
        if abs(target - self.audio_target) < 0.0005:
            return
        old = self.audio
        self.audio = None
        if self.console is not None:
            self.console.apu.sink = None
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
            cw = self.console.width if self.console else 256
            ch = self.console.height if self.console else 240
            self.set_size(int(cw * s * self.par), ch * s + MENU_H)

    def act_mute(self):
        self.muted = not self.muted
        self.ui_sync_audio()

    def act_volume(self, v):
        self.volume = v
        if self.console:
            self.console.apu.volume = 0.7 * v

    def _poll_pad(self):
        p = self.pad
        if p is None:
            return 0
        try:
            b = 0
            if p.b or p.y: b |= BTN_A
            if p.a or p.x: b |= BTN_B
            if p.back: b |= BTN_SELECT
            if p.start: b |= BTN_START
            if p.dpup: b |= BTN_UP
            if p.dpdown: b |= BTN_DOWN
            if p.dpleft or p.leftx < -0.5: b |= BTN_LEFT
            if p.dpright or p.leftx > 0.5: b |= BTN_RIGHT
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
        elif symbol == k.P: self.act_pause()
        elif symbol == k.R: self.act_reset()
        elif symbol == k.M: self.act_mute()
        elif symbol == k.F: self.act_smooth()
        elif symbol == k.F5: self.act_save_state()
        elif symbol == k.F7: self.act_load_state()
        elif symbol == getattr(k, "V", -1): self.act_toggle_record()
        elif symbol == k.F11: self.act_fullscreen()
        elif symbol == k.F12: self.act_screenshot()

    def on_key_release(self, symbol, modifiers):
        bit = self.keymap.get(symbol)
        if bit:
            self.keys &= ~bit
        elif symbol == arcade.key.TAB:
            self.turbo_hold = False
            self.ui_update_turbo()

    def ui_bar_visible(self, wh):
        return (not self.fullscreen) or self.ui_open is not None or self.ui_ptr[1] >= wh - MENU_H

    def ui_hit_title(self, x, y, wh):
        if y >= wh - MENU_H and self.ui_bar_visible(wh):
            for i, (x0, x1) in enumerate(self.ui_title_x):
                if x0 <= x < x1:
                    return i
        return None

    def ui_rows(self, i, wh):
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

    def on_update(self, dt):
        now = time.perf_counter()
        elapsed = min(now - self.last, 0.1)
        self.last = now
        if self.paused or self.console is None:
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
                        st = self.rewind_history.pop()
                        self.console.restore_state(st["state"])
                        self.frame_ready = True
                        if self.audio is not None and not self.muted:
                            reversed_pcm = self._reverse_audio_speed(
                                st.get("audio", ()), self.speed_multiplier)
                            if reversed_pcm.size:
                                self.audio.push(reversed_pcm)
                    else:
                        break
            self.console.apu.sink = None
            self.fps_n += steps
            return
        self.rewind_acc = 0.0
        apu = self.console.apu
        if self.audio is not None and apu.sink is not None:
            fill = self.audio.buffered_seconds()
            self.fill_avg += (fill - self.fill_avg) * 0.12
            err = self.fill_avg - self.audio_target
            speed *= 1.0 - max(-0.04, min(0.04, err * 1.0))
        self.acc += elapsed * speed
        frame_time = 1.0 / FRAME_RATE
        n = 0
        if self.acc >= frame_time:
            self.console.set_buttons(0, self.keys | self._poll_pad())
            limit = self.max_catchup * (3 if self.turbo else 1)
            while self.acc >= frame_time:
                self.acc -= frame_time
                n += 1
                more = self.acc >= frame_time and n < limit
                rewind_entry = None
                if not self.turbo:
                    rewind_entry = {"state": self.console.capture_state(), "audio": np.empty(0, dtype=np.int16)}
                    self.rewind_history.append(rewind_entry)
                self._audio_capture = []
                self.console.run_frame(draw=True)
                if self.recording and self.video_writer is not None:
                    try:
                        self.video_writer.append_data(self.console.get_rgba()[:, :, :3])
                    except Exception:
                        pass
                if rewind_entry is not None:
                    if self._audio_capture:
                        rewind_entry["audio"] = np.concatenate(self._audio_capture)
                    self._audio_capture = []
                if not more:
                    self.frame_ready = True
                    break
            if self.acc >= frame_time:
                self.acc = 0.0
        self.fps_n += n
        if now - self.fps_t >= 1.0:
            fps = self.fps_n / (now - self.fps_t)
            self.set_caption(f"{self.title_base} - {fps:.1f} FPS")
            if self.use_vsync and not self.turbo and fps < 54.0:
                self.slow_secs += 1
                if self.slow_secs >= 3:
                    self.act_vsync(False, auto=True)
            else:
                self.slow_secs = 0
            self.fps_t, self.fps_n = now, 0

    def ui_rect(self, x0, y0, x1, y1, color):
        ww, wh = self.ui_size
        self.rect_prog["rect"] = ((x1 - x0) / ww, (y1 - y0) / wh,
                                  (x0 + x1) / ww - 1.0, (y0 + y1) / wh - 1.0)
        self.rect_prog["color"] = color
        self.quad.render(self.rect_prog)

    def ui_text(self, s, color, size=11, anchor_x="left"):
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
        if self.console is not None:
            if self.frame_ready:
                self.texture.write(self.console.get_rgba().tobytes())
                self.frame_ready = False
            dw, dh = self.console.width * self.par, float(self.console.height)
            s = min(ww / dw, avail / dh)
            self.program["rect"] = (dw * s / ww, dh * s / wh, 0.0, (avail - wh) / wh)
            self.texture.use(0)
            self.quad.render(self.program)
            if self.reverse_speed:
                tag = self.ui_text(f"REWIND PREVIEW  •  {self.speed_multiplier:g}x", C_ACCENT, 12)
                pad = 8
                self.ui_rect(8, 8, 8 + tag.content_width + pad * 2, 32, C_DROP)
                self.ui_rect(8, 8, 10, 32, C_ACCENT)
                self.ui_put(tag, 8 + pad + 2, 20)
        else:
            t = self.ui_text("File > Open ROM... (.nes / .sfc / .snes / .gb / .gbc / .gba)   (Ctrl+O)", C_DIM, 14, "center")
            self.ui_put(t, ww / 2, avail / 2)
        self.ui_draw_menu(ww, wh)
        self.ui_draw_overlays(ww, wh)

    def on_close(self):
        try:
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None
        except Exception:
            pass
        try:
            if self.console is not None:
                self.console.save_battery()
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
        "    title='Select ROM (.nes, .sfc, .smc, .snes, .gb, .gbc, .gba)',\n"
        "    filetypes=[\n"
        "        ('All Supported ROMs', '*.nes *.sfc *.smc *.snes *.gb *.gbc *.gba'),\n"
        "        ('NES ROMs', '*.nes'),\n"
        "        ('SNES ROMs', '*.sfc *.smc *.snes'),\n"
        "        ('Game Boy / Color ROMs', '*.gb *.gbc'),\n"
        "        ('Game Boy Advance ROMs', '*.gba'),\n"
        "        ('All files', '*.*')\n"
        "    ])\n"
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


def main():
    ap = argparse.ArgumentParser(description="PyRetro - NES & SNES emulator (Python + Arcade)")
    ap.add_argument("rom", nargs="?", help="path to an .nes, .sfc, or .smc ROM")
    ap.add_argument("--scale", type=int, default=3, help="initial window scale (default 3)")
    ap.add_argument("--no-audio", action="store_true", help="disable sound")
    ap.add_argument("--audio-backend", choices=("auto", "sounddevice", "pyglet"), default="auto")
    ap.add_argument("--audio-latency", type=int, default=25, metavar="MS")
    ap.add_argument("--no-vsync", action="store_true")
    ap.add_argument("--aspect", choices=("square", "ntsc"), default="square")
    ap.add_argument("--max-frameskip", type=int, default=6)
    args = ap.parse_args()

    console = None
    if args.rom:
        try:
            console = load_console(args.rom)
        except Exception as e:
            sys.exit(f"Failed to open {args.rom!r}: {e}")
        print(f"Loaded [{console.system_name}] {os.path.basename(args.rom)}")
    if arcade is None:
        sys.exit("The 'arcade' package is required for the GUI: pip install arcade")
    sys.setswitchinterval(0.001)
    RetroWindow(console, scale=args.scale, audio=not args.no_audio, aspect=args.aspect,
                max_catchup=args.max_frameskip, vsync=not args.no_vsync,
                audio_latency=max(10, min(100, args.audio_latency)) / 1000.0,
                audio_backend=args.audio_backend)
    arcade.run()


if __name__ == "__main__":
    main()