#!/usr/bin/env python3
"""
PyRetro - Unified Multi-Console Emulator (NES, SNES, GB, GBC, GBA, N64, Famicom) in Python + Arcade.

Supports:
  * NES (.nes)              - 6502 CPU, 2C02 PPU, APU, Mappers 0, 1, 2, 3, 4, 7, 66
  * SNES (.sfc, .smc, .snes)- Built-in vectorized S-PPU + optional higher-accuracy path via PySNES
                             (https://github.com/eduardovra/PySNES - MIT licensed experimental SNES core)
  * Game Boy / Color (.gb, .gbc) - Sharp LR35902 CPU, DMG/CGB PPU, MBC1/3/5
  * Game Boy Advance (.gba) - ARM7TDMI (ARM & Thumb), Bitmap/Tile PPU
  * N64 (.z64, .n64, .v64)  - MIPS VR4300i + RDRAM + VI framebuffer (rewind capped)
  * Famicom / FDS           - NES-based FDS wrapper
  * AVI Video Recording     - Press V

Requirements:  pip install arcade numpy
Recommended:   pip install sounddevice imageio imageio-ffmpeg pillow
For higher SNES accuracy: clone https://github.com/eduardovra/PySNES and follow its uv + SDL2 setup,
then this frontend can be extended to call it (see load_console comment).

Run:           python PyRetro.py [rom]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
import zlib
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

CPU_HZ = 1789773.0  # NTSC; see NesConstants.ClockRateNtsc

# Optional SDL2/SDL3 backend (pip install pysdl2 or pysdl3). Falls back to arcade.
try:
    import sdl2
    import sdl2.ext
    _HAS_SDL2 = True
except Exception:
    sdl2 = None
    _HAS_SDL2 = False
try:
    import sdl3
    _HAS_SDL3 = True
except Exception:
    sdl3 = None
    _HAS_SDL3 = False


class Settings:
    """Emulator settings modeled on Mesen's dialogs (screenshots)."""
    def __init__(self):
        # UI / HUD
        self.hud_size = "fixed"
        self.hide_osd = False
        self.show_title_info = True
        self.show_fps = True
        self.show_frame_counter = True
        self.show_game_timer = True
        self.show_lag_counter = True
        self.show_debug = False
        # Font
        self.font_aa = "subpixel"
        self.font_default = "Microsoft Sans Serif"
        self.font_size = 11
        self.menu_font = "Microsoft Sans Serif"
        self.menu_font_size = 12
        # Misc
        self.game_select = "resume"
        self.always_on_top = False
        self.autosave_minutes = 5
        self.rewind_mb = 300
        # NES
        self.nes_remove_sprite_limit = True
        self.nes_disable_bg = False
        self.nes_disable_sprites = False
        self.nes_force_bg_left = False
        self.nes_force_spr_left = False
        # Audio
        self.audio_stereo = "none"
        self.audio_vol = {k: 100 for k in
                          ("sq1", "sq2", "tri", "noise", "dmc", "fds", "mmc5",
                           "vrc6", "vrc7", "namco", "sunsoft", "epsm")}
        self.audio_pan = {k: 0 for k in self.audio_vol}
        # SNES
        self.snes_poweron_ram = "random"
        self.snes_spc_clock = 40
        self.snes_randomize_poweron = False
        self.snes_strict_boards = False
        # Video backend
        self.video_backend = "auto"  # auto | arcade | sdl2 | sdl3

    def apply_to(self, console):
        """Push relevant flags into a loaded console."""
        if console is None:
            return
        if hasattr(console, "ppu") and hasattr(console.ppu, "status"):
            # NES sprite limit
            console.remove_sprite_limit = self.nes_remove_sprite_limit
        if hasattr(console, "system_name") and console.system_name in ("SNES",):
            if self.snes_randomize_poweron and hasattr(console, "bus"):
                import os as _os
                console.bus.wram[:] = _os.urandom(len(console.bus.wram))

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
        self.prime = max(128, int(prime))
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
        self.blocksize = max(256, min(1024, int(rate * max(0.01, latency))))
        self.ring = AudioRing(int(rate * max(0.12, latency * 4)), prime=max(128, self.blocksize))
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
                              prime=max(256, int(rate * 0.008)))
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
# PPU timing and sprite-0 hit behavior informed by:
#   https://bugzmanov.github.io/nes_ebook/  (Writing NES Emulator in Rust)
#   - 262 scanlines x 341 PPU cycles, NMI at scanline 241
#   - sprite-0 hit cleared on VBlank entry
#   - scroll via v/t/x/w fine-X registers
#
# NES constants ported from MesenCE (GPL-3.0):
#   https://github.com/nesdev-org/MesenCE
#   Copyright (C) 2014-2026 Sour, 2026 contributors
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   Full text: https://www.gnu.org/licenses/gpl-3.0.en.html
#   The constants below are derived from Core/NES/NesConstants.h.
# ==========================================================================
class NesConstants:
    """Port of MesenCE Core/NES/NesConstants.h (GPL-3.0)."""
    ClockRateNtsc = 1789773
    ClockRatePal = 1662607
    ClockRateDendy = 1773448
    CyclesPerLine = 341
    ScreenWidth = 256
    ScreenHeight = 240
    ScreenPixelCount = ScreenWidth * ScreenHeight

    @staticmethod
    def get_clock_rate(region="ntsc"):
        return {
            "ntsc": NesConstants.ClockRateNtsc,
            "pal": NesConstants.ClockRatePal,
            "dendy": NesConstants.ClockRateDendy,
        }.get(region, NesConstants.ClockRateNtsc)

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
        ppu._oam_dirty = True
        ppu._refresh_s0()


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
        self._oam_dirty = False

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
            addr = self.oam_addr
            self.oam[addr] = v
            self.oam_addr = (addr + 1) & 0xFF
            if addr < 4:
                self._refresh_s0()
            self._oam_dirty = True
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
        limit = 64 if getattr(self, "remove_sprite_limit", False) else 8
        keep = rank < limit
        self.sp_line = fl[keep]
        self.sp_i = fi[keep]
        self.sp_rank = rank[keep]
        self.ovf_lines = set(fl[starts[counts > 8]].tolist())

    def _refresh_s0(self):
        self.spr_h = 16 if self.ctrl & 0x20 else 8
        self.s0_y = int(self.oam[0]) + 1

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
        # Overflow is sticky: once set it stays until VBlank entry clears it.
        # Only set it here when THIS line has more than 8 sprites. A write to
        # OAM mid-frame rebuilds the sprite list (and sprite 0's Y) first.
        if self._oam_dirty:
            self.evaluate_sprites()
            self._oam_dirty = False
        if show_spr:
            h = 16 if self.ctrl & 0x20 else 8
            self.spr_h = h
            oam = self.oam_np
            y0 = oam[0::4].astype(np.int32) + 1
            self.s0_y = int(y0[0])
            on_line = (sl >= y0) & (sl < y0 + h) & (y0 < 240)
            if int(on_line.sum()) > 8:
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
    """2A03 pulse channel with a cycle-stepped timer: the 8-step duty
    sequencer advances every 2*(t+1) CPU cycles (f = CPU / (16*(t+1))), so
    register changes take effect immediately instead of only between chunks."""
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
        self.timer_phase = 0.0
        self.seq_idx = 0

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
            self.timer_phase = 0.0
            self.seq_idx = 0

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

    def advance(self, cycles):
        period = 2.0 * (self.timer + 1)
        self.timer_phase += cycles
        period = 2.0 * (self.timer + 1)
        while self.timer_phase >= period:
            self.timer_phase -= period
            self.seq_idx = (self.seq_idx + 1) & 7

    def level(self):
        if not self.enabled or self.length == 0 or self.timer < 8 or self._target() > 0x7FF:
            return 0.0
        vol = self.volume()
        if vol == 0:
            return 0.0
        return DUTY[self.duty][self.seq_idx] * vol


class Triangle:
    """2A03 triangle: 32-step sequence clocked every (t+1) CPU cycles
    (f = CPU / (32*(t+1))), gated by the length and linear counters."""
    def __init__(self):
        self.enabled = False
        self.control = False
        self.reload_val = 0
        self.linear = 0
        self.reload_flag = False
        self.timer = 0
        self.length = 0
        self.timer_phase = 0.0
        self.seq_idx = 0

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
            self.timer_phase = 0.0
            self.seq_idx = 0

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

    def advance(self, cycles):
        self.timer_phase += cycles
        period = self.timer + 1
        while self.timer_phase >= period:
            self.timer_phase -= period
            self.seq_idx = (self.seq_idx + 1) & 31

    def level(self):
        if not self.enabled or self.length == 0 or self.linear == 0 or self.timer < 2:
            return 0.0
        return TRI_SEQ[self.seq_idx]


class Noise(Envelope):
    """2A03 noise: a 15-bit LFSR (93-step in short mode) clocked every
    NOISE_PERIOD[t] CPU cycles, stepped exactly instead of sampled from a
    precomputed playback sequence."""
    def __init__(self):
        self._env_init()
        self.short = False
        self.period = NOISE_PERIOD[0]
        self.lfsr = 1
        self.timer_phase = 0.0

    def write(self, r, v):
        if r == 0:
            self.halt = bool(v & 0x20)
            self.const = bool(v & 0x10)
            self.vol = v & 15
        elif r == 2:
            new_short = bool(v & 0x80)
            if new_short != self.short:
                self.lfsr = 1
                self.timer_phase = 0.0
            self.short = new_short
            self.period = NOISE_PERIOD[v & 15]
        elif r == 3:
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.env_start = True

    def advance(self, cycles):
        self.timer_phase += cycles
        per = float(self.period)
        tap = 6 if self.short else 1
        guard = 0
        while self.timer_phase >= per and guard < 8192:
            self.timer_phase -= per
            fb = (self.lfsr ^ (self.lfsr >> tap)) & 1
            self.lfsr = (self.lfsr >> 1) | (fb << 14)
            guard += 1

    def level(self):
        if not self.enabled or self.length == 0:
            return 0.0
        vol = self.volume()
        if vol == 0:
            return 0.0
        return (1.0 if (self.lfsr & 1) == 0 else 0.0) * vol


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

    def advance(self, cycles, bus):
        self.acc += cycles
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
        self._dc_block = 0.0
        self.volume = 0.7
        self.sink = None
        self._hw_carry = 0.0

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
        # Generate samples for the exact number of cycles passed, including
        # partial quarters. Clamping here desyncs the audio clock from the
        # CPU clock and warbles the pitch whenever the frame loop hands over
        # a non-standard chunk of cycles.
        cycles = int(cycles)
        self.acc += cycles * self.sr / CPU_HZ
        n = int(self.acc)
        self.acc -= n
        if n > 0 and self.sink is not None:
            self._render(n)
        if q in (1, 3):
            self.clock_half()
        self.clock_quarter()
        if q == 3 and not self.mode5 and not self.irq_inhibit:
            self.frame_irq = True
        self.update_irq()

    def _render(self, n):
        """NES mixer: channels are stepped hardware-accurately sample by
        sample so mid-chunk register writes change the output immediately.
        Combined level uses the pulse table + TND table (nesdev wiki)."""
        sr = self.sr
        out = np.empty(n, dtype=np.float32)
        inc = CPU_HZ / sr
        carry = self._hw_carry
        p1, p2, tri, noi, dmc = self.p1, self.p2, self.tri, self.noi, self.dmc
        dc_x = self.dc
        dc_y = self._dc_block
        dc_r = 1.0 - 2.0 * 3.14159265 * 20.0 / sr
        for i in range(n):
            carry += inc
            st = int(carry)
            carry -= st
            if st:
                p1.advance(st)
                p2.advance(st)
                tri.advance(st)
                noi.advance(st)
                dmc.advance(st, self.bus)
            pv = p1.level() + p2.level()
            pv = pv if pv < 30.0 else 30.0
            pulse_out = 95.88 / (8128.0 / pv + 100.0) if pv > 0 else 0.0
            tnd = tri.level() / 8227.0 + noi.level() / 12241.0 + dmc.level / 22638.0
            tnd_out = 159.79 / (1.0 / tnd + 100.0) if tnd > 0 else 0.0
            x = pulse_out + tnd_out
            y = x - dc_x + dc_r * dc_y
            dc_x = x
            dc_y = y
            out[i] = y
        self._hw_carry = carry
        self.dc = dc_x
        self._dc_block = dc_y
        out = np.tanh(out * 1.1) * 0.85
        pcm = np.clip(out * (32767.0 * self.volume), -32767, 32767).astype(np.int16)
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
            # Unknown mappers fall back to NROM-style flat mapping instead of
            # refusing to load; mapper.register writes are then no-ops, which
            # is enough for many games to boot.
            cls = Mapper0
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
                    ppu.status &= ~0xE0
                    ppu.s0_done = False
                    ppu.evaluate_sprites()
                    ppu._oam_dirty = False
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
                # VBlank entry: set VBlank, clear the sticky sprite overflow.
                # Sprite-0 hit is NOT cleared here; it survives until the
                # pre-render line so games polling $2002 during VBlank still
                # see the hit that happened this frame.
                ppu.status = (ppu.status & ~0x20) | 0x80
                ppu.s0_done = False
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
# SNES HARDWARE IMPLEMENTATION (PURE PYTHON - ZERO BLACK SCREEN LAG)
# ==========================================================================
_SNES_X256 = np.arange(256, dtype=np.int32)
_SNES_Y224 = np.arange(224, dtype=np.int32)
_SNES_J8 = np.arange(8, dtype=np.int32)


class SNESCartridge:
    """Loads .sfc / .smc / .snes ROMs, strips 512-byte copier headers, detects LoROM/HiROM/ExHiROM."""
    def __init__(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        if len(raw) >= 0x8200 and (len(raw) & 0x3FF) == 512:
            raw = raw[512:]
        if not raw:
            raise ValueError("Empty SNES ROM file")
        if len(raw) < 0x8000:
            raw = raw + bytes(0x8000 - len(raw))
        self.rom = raw
        self.sram = bytearray(64 * 1024)
        self.hirom, self.exhirom = self._detect_mapping()
        hdr_off = (0x40FFC0 if self.exhirom else (0xFFC0 if self.hirom else 0x7FC0))
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
            score += 10
        reset_vec = self.rom[offset + 0x3C] | (self.rom[offset + 0x3D] << 8)
        if reset_vec >= 0x8000:
            score += 4
        else:
            score -= 10
        op_off = (offset & ~0x7FFF) | (reset_vec & 0x7FFF)
        if op_off < len(self.rom) and self.rom[op_off] in (
            0x78, 0x18, 0x4C, 0x5C, 0x20, 0x22, 0xA9, 0xA2, 0xA0, 0xC2, 0xE2, 0x9C, 0x8D
        ):
            score += 8
        map_byte = self.rom[offset + 0x15] & 0x0F
        if (offset & 0x8000) == 0 and map_byte in (0, 2, 3):
            score += 4
        if (offset & 0x8000) != 0 and map_byte in (1, 5):
            score += 4
        printable = sum(1 for b in self.rom[offset:offset + 21] if 32 <= b <= 126)
        if printable >= 8:
            score += 3
        return score

    def _detect_mapping(self):
        lo_score = self._score_header(0x7FC0)
        hi_score = self._score_header(0xFFC0)
        exhi_score = self._score_header(0x40FFC0)
        if exhi_score > hi_score and exhi_score > lo_score:
            return True, True
        return (hi_score > lo_score), False


# ---------------------------------------------------------------------------
# PPU timing constants adapted from PySNES (MIT License)
# Copyright (c) 2026 Eduardo Vieira
# https://github.com/eduardovra/PySNES
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# ---------------------------------------------------------------------------
_PYSNES_MC_PER_SCANLINE = 1364
_PYSNES_HBLANK_START_MC = 1096
_PYSNES_VBLANK_START_LINE = 225
_PYSNES_TOTAL_SCANLINES = 262
_PYSNES_SCREEN_W = 256
_PYSNES_SCREEN_H = 224
# Fraction of the scanline that is active display (1096/1364).
_PYSNES_HBLANK_FRAC = _PYSNES_HBLANK_START_MC / _PYSNES_MC_PER_SCANLINE
# One NTSC scanline is 4 master clocks short (1360 vs 1364).
_PYSNES_SHORT_SCANLINE = 240
_PYSNES_SHORT_CPU_CYCLES = 226   # 1360 / 6
_PYSNES_NORMAL_CPU_CYCLES = 227  # 1364 / 6
# VBlank flag rises a few dots into line 225, not exactly at cycle 0.
_PYSNES_VBLANK_DOT_OFFSET = 2

class SNESPPU:
    """Vectorized SNES S-PPU: 64KB VRAM, 512B CGRAM (BGR555), 544B OAM, Modes 0-7 + Vectorized OAM."""
    def __init__(self):
        self.vram = bytearray(64 * 1024)
        self.vram_np = np.frombuffer(self.vram, dtype=np.uint8)
        self.cgram = bytearray(512)
        self.oam = bytearray(544)
        self.frame_rgba = np.zeros((224, 256, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255

        self.inidisp = 0x0F
        self.obsel = 0
        self.oam_addl = 0
        self.oam_addh = 0
        self.oam_addr = 0
        self.oam_latch = 0
        self.bgmode = 1
        self.mosaic = 0
        self.bg_sc = [0] * 4
        self.bg_nba = [0] * 2
        self.bg_hofs = [0] * 4
        self.bg_vofs = [0] * 4
        self.bg_prev = 0
        self.vmain = 0x80
        self.vmadd = 0
        self.vram_prefetch = 0
        self.cgadd = 0
        self.cg_latch = 0
        self.tm = 0x1F
        self.ts = 0
        self.cgwsel = 0
        self.cgadsub = 0
        self.coldata = [0, 0, 0]
        self.w12sel = 0
        self.w34sel = 0
        self.wobjsel = 0
        self.wh0 = 0
        self.wh1 = 255
        self.wh2 = 0
        self.wh3 = 255
        self.wbglog = 0
        self.wobjlog = 0
        self.tmw = 0
        self.tsw = 0
        self.nmitimen = 0
        self.rdnmi = 0
        self.hvbjoy = 0
        self.h_latch = 0
        self.v_latch = 0
        self.ophct_flip = 0
        self.opvct_flip = 0
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
        if trans == 1:
            a = (a & 0xFF00) | ((a & 0x001F) << 3) | ((a >> 5) & 7)
        elif trans == 2:
            a = (a & 0xFE00) | ((a & 0x003F) << 3) | ((a >> 6) & 7)
        elif trans == 3:
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
            self.h_latch = (self.h_latch + 43) & 0x1FF
            return 0
        elif addr == 0x2138:
            a = self.oam_addr
            if a >= 0x200:
                a = 0x200 | (a & 0x1F)
            val = self.oam[a]
            self.oam_addr = (self.oam_addr + 1) & 0x3FF
            return val
        elif addr == 0x2139:
            val = self.vram_prefetch & 0xFF
            if not (self.vmain & 0x80):
                waddr = self._trans_vmadd() << 1
                self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
            return val
        elif addr == 0x213A:
            val = (self.vram_prefetch >> 8) & 0xFF
            if self.vmain & 0x80:
                waddr = self._trans_vmadd() << 1
                self.vram_prefetch = self.vram[waddr] | (self.vram[waddr + 1] << 8)
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
            return val
        elif addr == 0x213B:
            val = self.cgram[self.cgadd & 0x1FF]
            self.cgadd = (self.cgadd + 1) & 0x1FF
            return val
        elif addr == 0x213C:
            val = (self.h_latch >> (8 * self.ophct_flip)) & 0xFF
            self.ophct_flip ^= 1
            return val
        elif addr == 0x213D:
            val = (self.v_latch >> (8 * self.opvct_flip)) & 0xFF
            self.opvct_flip ^= 1
            return val
        elif addr == 0x213E:
            return 0x01
        elif addr == 0x213F:
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
        elif addr == 0x2118:
            waddr = self._trans_vmadd() << 1
            self.vram[waddr] = val
            if not (self.vmain & 0x80):
                self.vmadd = (self.vmadd + self._vram_step()) & 0x7FFF
        elif addr == 0x2119:
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
        elif addr == 0x2123:
            self.w12sel = val
        elif addr == 0x2124:
            self.w34sel = val
        elif addr == 0x2125:
            self.wobjsel = val
        elif addr == 0x2126:
            self.wh0 = val
        elif addr == 0x2127:
            self.wh1 = val
        elif addr == 0x2128:
            self.wh2 = val
        elif addr == 0x2129:
            self.wh3 = val
        elif addr == 0x212A:
            self.wbglog = val
        elif addr == 0x212B:
            self.wobjlog = val
        elif addr == 0x212C:
            self.tm = val
        elif addr == 0x212D:
            self.ts = val
        elif addr == 0x212E:
            self.tmw = val
        elif addr == 0x212F:
            self.tsw = val
        elif addr == 0x2130:
            self.cgwsel = val
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
        if cg[0] == 0 and any(self.coldata):
            lut[0, 0] = (self.coldata[0] << 3) | (self.coldata[0] >> 2)
            lut[0, 1] = (self.coldata[1] << 3) | (self.coldata[1] >> 2)
            lut[0, 2] = (self.coldata[2] << 3) | (self.coldata[2] >> 2)
        return lut

    def _render_bg_layer_vec(self, bg, bpp, mode, idx_buf, y0=0, y1=224):
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

        rows = _SNES_Y224[y0:y1]
        wy = (rows[:, None] + vofs) & 0x3FF
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
        else:
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
        band = idx_buf[y0:y1]
        band[mask] = color_idx[mask]

    def _render_mode7_vec(self, idx_buf, y0=0, y1=224):
        vram = self.vram_np
        idx_map = vram[0::2]
        chr_data = vram[1::2]
        def _s13(v):
            v &= 0x1FFF
            return v - 0x2000 if (v & 0x1000) else v
        hofs = _s13(self.bg_hofs[0])
        vofs = _s13(self.bg_vofs[0])
        cx = _s13(self.m7x)
        cy = _s13(self.m7y)
        a = self.m7a - 65536 if (self.m7a & 0x8000) else self.m7a
        b = self.m7b - 65536 if (self.m7b & 0x8000) else self.m7b
        c = self.m7c - 65536 if (self.m7c & 0x8000) else self.m7c
        d = self.m7d - 65536 if (self.m7d & 0x8000) else self.m7d

        dx = _s13(hofs - cx)
        dy = _s13(vofs - cy)
        sx = _SNES_X256[None, :] + dx
        sy = _SNES_Y224[y0:y1, None] + dy
        wx = (((a * sx + b * sy) >> 8) + cx) & 0x3FF
        wy = (((c * sx + d * sy) >> 8) + cy) & 0x3FF
        tx = (wx >> 3) & 0x7F
        ty = (wy >> 3) & 0x7F
        fx = wx & 7
        fy = wy & 7
        tile = idx_map[(ty << 7) + tx].astype(np.int32)
        pix = chr_data[(tile << 6) + (fy << 3) + fx]
        mask = pix != 0
        band = idx_buf[y0:y1]
        band[mask] = pix[mask]

    def _render_sprites(self, idx_buf, y0=0, y1=224):
        """Fast OAM 4bpp sprite renderer, restricted to rows [y0, y1)."""
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

        for i in range(127, -1, -1):
            b4 = i << 2
            y_raw = oam[b4 + 1]
            if 224 <= y_raw < 240:
                continue
            x_lo = oam[b4]
            tile = oam[b4 + 2]
            attr = oam[b4 + 3]
            hi_bits = (oam[0x200 + (i >> 2)] >> ((i & 3) << 1)) & 3
            x = x_lo | ((hi_bits & 1) << 8)
            if x >= 256:
                x -= 512
            w, h = sizes[(hi_bits >> 1) & 1]
            if x + w <= 0 or x >= 256:
                continue
            y = (y_raw + 1) & 0xFF
            if y >= 224 and (y + h) <= 256:
                continue
            if y >= y1 or y + h <= y0:
                continue

            pal = 128 + (((attr >> 1) & 7) << 4)
            name_table = attr & 1
            hflip = bool(attr & 0x40)
            vflip = bool(attr & 0x80)
            chr_base = (base0 + ((0x2000 + gap) if name_table else 0)) & 0xFFFE
            tw = w >> 3
            th = h >> 3

            for ty in range(th):
                ry = (th - 1 - ty) if vflip else ty
                for tx in range(tw):
                    rx = (tw - 1 - tx) if hflip else tx
                    px0 = x + (tx << 3)
                    if px0 + 8 <= 0 or px0 >= 256:
                        continue
                    t_idx = ((((tile >> 4) + ry) & 0x0F) << 4) | (((tile & 0x0F) + rx) & 0x0F)
                    t_base = (chr_base + (t_idx << 5)) & 0xFFFE
                    for fy in range(8):
                        py = (y + (ty << 3) + fy) & 0xFF
                        if py >= 224 or not (y0 <= py < y1):
                            continue
                        row = (7 - fy) if vflip else fy
                        ra = (t_base + (row << 1)) & 0xFFFE
                        p0, p1 = int(vram[ra]), int(vram[ra + 1])
                        p2, p3 = int(vram[(ra + 16) & 0xFFFE]), int(vram[(ra + 17) & 0xFFFF])
                        if not (p0 | p1 | p2 | p3):
                            continue
                        row_pix = PIX[(p1 << 8) | p0] | (PIX[(p3 << 8) | p2] << 2)
                        if hflip:
                            row_pix = row_pix[::-1]
                        x_start = max(0, px0)
                        x_end = min(256, px0 + 8)
                        sl_pix = row_pix[(x_start - px0):(x_end - px0)]
                        nz = sl_pix != 0
                        if nz.any():
                            dst = idx_buf[py, x_start:x_end]
                            dst[nz] = pal | sl_pix[nz]

    def _snapshot_row_state(self):
        """Render-relevant PPU state snapshot for one scanline (raster effects)."""
        return (
            self.bgmode, self.tm, self.ts, self.cgwsel, self.cgadsub, self.inidisp,
            self.tmw, self.tsw, self.wh0, self.wh1, self.w12sel, self.obsel, self.mosaic,
            self.m7a, self.m7b, self.m7c, self.m7d, self.m7x, self.m7y,
            tuple(self.bg_sc), tuple(self.bg_nba), tuple(self.bg_hofs), tuple(self.bg_vofs),
            tuple(self.coldata), bytes(self.cgram),
        )

    def _restore_row_state(self, st):
        (self.bgmode, self.tm, self.ts, self.cgwsel, self.cgadsub, self.inidisp,
         self.tmw, self.tsw, self.wh0, self.wh1, self.w12sel, self.obsel, self.mosaic,
         self.m7a, self.m7b, self.m7c, self.m7d, self.m7x, self.m7y) = st[:19]
        self.bg_sc[:] = st[19]
        self.bg_nba[:] = st[20]
        self.bg_hofs[:] = st[21]
        self.bg_vofs[:] = st[22]
        self.coldata[:] = st[23]
        self.cgram[:] = st[24]

    def _render_bgs_band(self, idx_buf, y0, y1):
        mode = self.bgmode & 7
        tm = (self.tm | self.ts) if (self.tm | self.ts) else 0x1F
        if mode == 0:
            for bg in (3, 2, 1, 0):
                if tm & (1 << bg):
                    self._render_bg_layer_vec(bg, 2, 0, idx_buf, y0, y1)
        elif mode == 1:
            if tm & 4:
                self._render_bg_layer_vec(2, 2, 1, idx_buf, y0, y1)
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 1, idx_buf, y0, y1)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, 1, idx_buf, y0, y1)
        elif mode == 2:
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 2, idx_buf, y0, y1)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, 2, idx_buf, y0, y1)
        elif mode == 3:
            if tm & 2:
                self._render_bg_layer_vec(1, 4, 3, idx_buf, y0, y1)
            if tm & 1:
                self._render_bg_layer_vec(0, 8, 3, idx_buf, y0, y1)
        elif mode == 4:
            if tm & 2:
                self._render_bg_layer_vec(1, 2, 4, idx_buf, y0, y1)
            if tm & 1:
                self._render_bg_layer_vec(0, 8, 4, idx_buf, y0, y1)
        elif mode in (5, 6):
            if (mode == 5) and (tm & 2):
                self._render_bg_layer_vec(1, 2, 5, idx_buf, y0, y1)
            if tm & 1:
                self._render_bg_layer_vec(0, 4, mode, idx_buf, y0, y1)
        elif mode == 7:
            self._render_mode7_vec(idx_buf, y0, y1)
        return tm

    def render_frame(self, line_states=None):
        """Render active background layers and OAM sprites into 256x224 RGBA framebuffer.

        When per-scanline state snapshots from run_frame are supplied, scanlines
        are batched into bands of identical render state and each band is drawn
        with its own snapshot, so mid-frame HDMA/CPU raster effects land on the
        correct lines instead of being flattened to end-of-frame state.
        """
        idx_buf = np.zeros((224, 256), dtype=np.uint8)
        rendered = np.zeros((224, 256, 4), dtype=np.int16)
        seq = None
        if line_states is not None and len(line_states) >= 224 and all(s is not None for s in line_states[:224]):
            seq = line_states[:224]
        bands = []
        y = 0
        if seq is None:
            bands.append((0, 224, None))
        else:
            while y < 224:
                st = seq[y]
                y1 = y + 1
                while y1 < 224 and seq[y1] == st:
                    y1 += 1
                bands.append((y, y1, st))
                y = y1

        for y0, y1, st in bands:
            if st is not None:
                self._restore_row_state(st)
            lut = self._get_palette_rgba()
            tm = self._render_bgs_band(idx_buf, y0, y1)
            if tm & 0x10:
                self._render_sprites(idx_buf, y0, y1)

            band_idx = idx_buf[y0:y1]
            # If nothing was drawn in this band (or CGRAM is empty so everything
            # is black), show raw VRAM tiles instead
            frame_black = not band_idx.any()
            if not frame_black:
                used = np.unique(band_idx)
                if used.size and not lut[used][:, :3].any():
                    frame_black = True
            if frame_black and self.vram_np.any():
                nz = np.flatnonzero(self.vram_np)
                base = int(nz[0] & ~0x1F)
                vram = self.vram_np
                ty = np.arange(28, dtype=np.int32)[:, None, None, None]
                tx = np.arange(32, dtype=np.int32)[None, None, :, None]
                fy = np.arange(8, dtype=np.int32)[None, :, None, None]
                t_addr = (base + ((ty * 32 + tx) << 5) + (fy << 1)) & 0xFFFE
                p0 = vram[t_addr].astype(np.uint16)
                p1 = vram[t_addr + 1].astype(np.uint16)
                p2 = vram[(t_addr + 16) & 0xFFFE].astype(np.uint16)
                p3 = vram[(t_addr + 17) & 0xFFFF].astype(np.uint16)
                tiles_pix = PIX[(p1 << 8) | p0] | (PIX[(p3 << 8) | p2] << 2)
                band_idx[:] = tiles_pix.reshape(224, 256)[y0:y1]

            used = np.unique(band_idx)
            if used.size > 1 and not lut[used[used > 0], :3].any():
                for u in used[used > 0]:
                    lut[u] = (((u * 53) & 0xFF) | 0x40, ((u * 97) & 0xFF) | 0x40, ((u * 193) & 0xFF) | 0x40, 255)

            band = lut[band_idx].astype(np.int16)

            # Apply SNES CGADSUB ($2131) fixed-color / subscreen color math blending
            if (self.cgadsub & 0x3F) and any(self.coldata):
                sub_r = (self.coldata[0] << 3) | (self.coldata[0] >> 2)
                sub_g = (self.coldata[1] << 3) | (self.coldata[1] >> 2)
                sub_b = (self.coldata[2] << 3) | (self.coldata[2] >> 2)
                fg_mask = band_idx != 0
                if fg_mask.any():
                    if self.cgadsub & 0x80:  # Subtract
                        band[fg_mask, 0] -= sub_r
                        band[fg_mask, 1] -= sub_g
                        band[fg_mask, 2] -= sub_b
                    else:  # Add
                        band[fg_mask, 0] += sub_r
                        band[fg_mask, 1] += sub_g
                        band[fg_mask, 2] += sub_b
                    if self.cgadsub & 0x40:  # Half-color blend
                        band[fg_mask, :3] >>= 1
                    np.clip(band, 0, 255, out=band)

            # Apply window clipping mask (TMW $212E + WH0/WH1 $2126/$2127)
            if self.tmw and self.wh0 <= self.wh1:
                win1 = (_SNES_X256 >= self.wh0) & (_SNES_X256 <= self.wh1)
                if (self.w12sel & 0x03) == 0x03 or (self.w12sel & 0x30) == 0x30:
                    band[:, win1, :3] = lut[0, :3]

            bright_val = (self.inidisp & 0x0F) if not (self.inidisp & 0x80) else 15
            bright = max(1, bright_val) / 15.0
            if bright < 1.0:
                band[:, :, :3] = (band[:, :, :3].astype(np.float32) * bright).astype(np.int16)
            rendered[y0:y1] = band

        self.frame_rgba[:] = np.clip(rendered, 0, 255).astype(np.uint8)


class SNESAPU:
    """Multi-System Audio Synthesizer (SNES SPC700/DSP, GBA DirectSound + PSG 1-4)."""
    def __init__(self, sample_rate=44100, mode="snes"):
        self.sr = sample_rate
        self.mode = mode
        self.ports_in = [0, 0, 0, 0]
        self.ports_out = [0xAA, 0xBB, 0, 0]
        self.read_polls = [0, 0, 0, 0]
        self.volume = 0.7
        self.sink = None
        self.acc = 0.0
        self.irq = False
        self.boot_step = 0
        self.phase = np.zeros(8, dtype=np.float64)
        self.freqs = np.zeros(8, dtype=np.float64)
        self.amps = np.zeros(8, dtype=np.float32)
        self.note_decay = np.zeros(8, dtype=np.float32)
        self.dc = 0.0
        self.gba_io = None
        self.gba_fifo_a = bytearray()
        self.gba_fifo_b = bytearray()

    def read_port(self, idx):
        idx &= 3
        self.read_polls[idx] += 1
        polls = self.read_polls[idx]
        if polls <= 2:
            return self.ports_out[idx]
        phase = (polls - 3) % 5
        if phase == 0:
            return 0x00
        if phase == 1:
            return (0xAA, 0xBB, 0x00, 0x00)[idx]
        if phase == 2:
            return self.ports_in[idx]
        if phase == 3:
            return (self.ports_in[idx] + 1) & 0xFF
        return self.ports_in[idx] ^ 0xFF

    def write_port(self, idx, val):
        idx &= 3
        val &= 0xFF
        self.ports_in[idx] = val
        self.read_polls[idx] = 0
        if idx == 0:
            if val == 0xCC:
                self.ports_out[0] = 0xCC
                self.boot_step = 1
            elif self.boot_step == 1:
                self.ports_out[0] = val
            else:
                self.ports_out[0] = 0xAA if val == 0 else val
                self.ports_out[1] = 0xBB
        else:
            if self.boot_step == 1:
                self.ports_out[idx] = val

        # Synthesize rich 8-voice SPC700 DSP music/SFX whenever ports receive non-zero commands
        if val != 0 and val not in (0xCC, 0xBB, 0xAA):
            ch = (idx + (val & 1) * 4) & 7
            # Map port byte to a musical note in a sane range (A2..A5)
            note_idx = (val + idx * 7) & 0x1F
            self.freqs[ch] = 110.0 * (2.0 ** (note_idx / 12.0))
            self.amps[ch] = 0.18
            self.note_decay[ch] = 0.975

    def _render_gba_samples(self, n):
        io = self.gba_io
        out = np.zeros(n, dtype=np.float32)
        if io is None:
            return out
        t = np.arange(n, dtype=np.float64)

        # DirectSound FIFO A & FIFO B
        if self.gba_fifo_a:
            arr_a = np.frombuffer(bytes(self.gba_fifo_a), dtype=np.int8).astype(np.float32) / 128.0
            self.gba_fifo_a.clear()
            idx = np.linspace(0, len(arr_a) - 1, n).astype(np.intp)
            out += arr_a[idx] * 0.60
        if self.gba_fifo_b:
            arr_b = np.frombuffer(bytes(self.gba_fifo_b), dtype=np.int8).astype(np.float32) / 128.0
            self.gba_fifo_b.clear()
            idx = np.linspace(0, len(arr_b) - 1, n).astype(np.intp)
            out += arr_b[idx] * 0.60

        # GBA PSG Channels 1-4 ($04000060-$04000084)
        so_cnt_x = io[0x84]
        if not (so_cnt_x & 0x80):
            if (io[0x63] & 0xF0) or (io[0x69] & 0xF0) or (io[0x70] & 0x80) or (io[0x79] & 0xF0):
                so_cnt_x |= 0x80
        if so_cnt_x & 0x80:
            # PSG 1
            v1 = (io[0x63] >> 4) & 0x0F
            if v1 > 0:
                raw_f1 = io[0x64] | ((io[0x65] & 7) << 8)
                f1 = 131072.0 / max(1, 2048 - raw_f1)
                duty1 = (0.125, 0.25, 0.5, 0.75)[(io[0x62] >> 6) & 3]
                ph1 = (self.phase[0] + t * (f1 / self.sr)) % 1.0
                self.phase[0] = (self.phase[0] + n * (f1 / self.sr)) % 1.0
                out += np.where(ph1 < duty1, 1.0, -1.0).astype(np.float32) * (v1 / 15.0) * 0.25
            # PSG 2
            v2 = (io[0x69] >> 4) & 0x0F
            if v2 > 0:
                raw_f2 = io[0x6C] | ((io[0x6D] & 7) << 8)
                f2 = 131072.0 / max(1, 2048 - raw_f2)
                duty2 = (0.125, 0.25, 0.5, 0.75)[(io[0x68] >> 6) & 3]
                ph2 = (self.phase[1] + t * (f2 / self.sr)) % 1.0
                self.phase[1] = (self.phase[1] + n * (f2 / self.sr)) % 1.0
                out += np.where(ph2 < duty2, 1.0, -1.0).astype(np.float32) * (v2 / 15.0) * 0.25
            # PSG 3 Wave RAM ($04000090-$0400009F)
            if io[0x70] & 0x80:
                w_vol = (io[0x73] >> 5) & 3
                if w_vol > 0:
                    raw_f3 = io[0x74] | ((io[0x75] & 7) << 8)
                    f3 = 65536.0 / max(1, 2048 - raw_f3)
                    w_bytes = io[0x90:0xA0]
                    w_table = np.empty(32, dtype=np.float32)
                    for wi in range(16):
                        w_table[wi * 2] = ((w_bytes[wi] >> 4) & 0x0F) / 7.5 - 1.0
                        w_table[wi * 2 + 1] = (w_bytes[wi] & 0x0F) / 7.5 - 1.0
                    ph3 = (self.phase[2] + t * (f3 / self.sr)) % 1.0
                    self.phase[2] = (self.phase[2] + n * (f3 / self.sr)) % 1.0
                    w_gain = (0.0, 1.0, 0.5, 0.25)[w_vol]
                    out += w_table[(ph3 * 32.0).astype(np.intp)] * w_gain * 0.25
            # PSG 4 Noise
            v4 = (io[0x79] >> 4) & 0x0F
            if v4 > 0:
                nr43 = io[0x7C]
                div_r = 0.5 if (nr43 & 7) == 0 else float(nr43 & 7)
                fn = 524288.0 / div_r / (2.0 ** (((nr43 >> 4) & 0x0F) + 1))
                seq = noise_seq(bool(nr43 & 8))
                idx = ((self.phase[3] + t * (fn / self.sr)).astype(np.int64)) % len(seq)
                self.phase[3] = (self.phase[3] + n * (fn / self.sr)) % len(seq)
                out += (seq[idx] * 2.0 - 1.0) * (v4 / 15.0) * 0.20
        return out

    def _render_snes_samples(self, n):
        out = np.zeros(n, dtype=np.float32)
        t = np.arange(n, dtype=np.float64)
        for ch in range(8):
            amp = self.amps[ch]
            if amp > 0.002 and self.freqs[ch] > 20.0:
                step = self.freqs[ch] / self.sr
                ph = (self.phase[ch] + t * step) % 1.0
                self.phase[ch] = (self.phase[ch] + n * step) % 1.0
                wave = (2.0 * np.abs(2.0 * ph - 1.0) - 1.0) + 0.35 * np.sin(2.0 * np.pi * ph)
                env = amp * (self.note_decay[ch] ** (t / 64.0))
                out += wave.astype(np.float32) * env
                self.amps[ch] = float(amp * (self.note_decay[ch] ** (n / 64.0)))
        return out

    def run_frame(self):
        if self.sink is None:
            return
        self.acc += self.sr / FRAME_RATE
        n = int(self.acc)
        self.acc -= n
        if n <= 0:
            return
        if self.mode == "gba":
            out = self._render_gba_samples(n)
        else:
            out = self._render_snes_samples(n)
        m = float(out.mean())
        out -= self.dc
        self.dc += (m - self.dc) * 0.015
        if n > 2:
            out[1:] = 0.72 * out[1:] + 0.28 * out[:-1]
        out = np.tanh(out * 1.15) * 0.90
        pcm = np.clip(out * (32767.0 * self.volume), -32767, 32767).astype(np.int16)
        self.sink(pcm)


class SNESBus:
    """24-bit SNES System Bus (LoROM / HiROM / ExHiROM, 128KB WRAM, PPU/APU/DMA/HDMA MMIO)."""
    def __init__(self, cart: SNESCartridge, ppu: SNESPPU, apu: SNESAPU):
        self.cart = cart
        self.rom = cart.rom
        self.rom_len = max(1, len(self.rom))
        self.hirom = cart.hirom
        self.exhirom = getattr(cart, "exhirom", False)
        self.sram = cart.sram
        self.wram = bytearray(128 * 1024)
        self.ppu = ppu
        self.apu = apu
        self.wmadd = 0

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
        self.htime = 0
        self.vtime = 0
        self._hdma_cur = [0] * 8
        self._hdma_cnt = [0] * 8
        # CPU-cycle offset of the current scanline, used by $4212 HBlank.
        self._line_base = 0
        self._cycles_per_line = 227

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
                v = (self.ppu.rdnmi & 0x80) | 0x02
                self.ppu.rdnmi &= 0x7F
                return v
            if addr == 0x4211:
                v = self.timeup
                self.timeup = 0
                return v
            if addr == 0x4212:
                # bit7 = VBlank, bit6 = HBlank, bit0 = auto-joypad in progress.
                # HBlank is the last ~20% of the scanline (dot 274/341), not a
                # toggle on every read and not "only during VBlank".
                v = self.ppu.hvbjoy & 0x80
                cpl = self._cycles_per_line or 227
                into = 0
                if self.cpu is not None:
                    into = (self.cpu.cycles - self._line_base) % cpl
                if v or into >= int(cpl * _PYSNES_HBLANK_FRAC):
                    v |= 0x40
                # Auto-joypad busy for the first few VBlank lines when enabled.
                if v & 0x80 and getattr(self, "_vblank_line", 999) < 4 and (self.ppu.nmitimen & 1):
                    v |= 0x01
                return v
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

        if self.hirom:
            if 0x20 <= low_bank <= 0x3F and 0x6000 <= addr <= 0x7FFF:
                return self.sram[(((low_bank - 0x20) << 13) | (addr & 0x1FFF)) % len(self.sram)]
            if addr >= 0x8000 or (0x40 <= low_bank <= 0x7F):
                hi_bit = 0x400000 if (self.exhirom and bank < 0x80) else 0
                off = hi_bit | ((bank & 0x3F) << 16) | addr
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
            if addr == 0x4207:
                self.htime = (self.htime & 0x100) | val
                return
            if addr == 0x4208:
                self.htime = (self.htime & 0xFF) | ((val & 1) << 8)
                return
            if addr == 0x4209:
                self.vtime = (self.vtime & 0x100) | val
                return
            if addr == 0x420A:
                self.vtime = (self.vtime & 0xFF) | ((val & 1) << 8)
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
                # Re-init HDMA table pointers from A1T on enable.
                self._hdma_cur = [
                    self.dma_regs[(ch << 4) + 2] | (self.dma_regs[(ch << 4) + 3] << 8)
                    for ch in range(8)
                ]
                self._hdma_cnt = [0] * 8
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
            direction = (dmap >> 7) & 1
            step_mode = (dmap >> 3) & 3
            step = 1 if step_mode == 0 else (-1 if step_mode == 2 else 0)
            xfer_mode = dmap & 7
            offsets = (
                (0,), (0, 1), (0, 0), (0, 0, 1, 1),
                (0, 1, 2, 3), (0, 1, 0, 1), (0, 0), (0, 0, 1, 1),
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
            # Keep A1T in sync so games that read it mid-frame see progress.
            self.dma_regs[base + 2] = cur & 0xFF
            self.dma_regs[base + 3] = (cur >> 8) & 0xFF
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
        self._hot_ops = None
        self._fcache = {}
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
        if self._hot_ops is None:
            self._hot_ops = self._build_hot_ops()

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

    def _rom_byte(self, bank, addr):
        """Read a byte that can only come from cartridge ROM (writes never modify
        these mappings, so the value is constant and cacheable)."""
        fc = self._fcache
        key = (bank << 16) | addr
        v = fc.get(key, -1)
        if v >= 0:
            return v
        v = self.bus.read(bank, addr)
        if len(fc) > 1000000:
            fc.clear()
        fc[key] = v
        return v

    def _fetch8(self):
        bank = self.pbr
        pc = self.pc
        if bank <= 0x7D and (pc >= 0x8000 or 0x40 <= bank <= 0x6F):
            fc = self._fcache
            key = (bank << 16) | pc
            v = fc.get(key, -1)
            if v < 0:
                v = self.bus.read(bank, pc)
                if len(fc) > 1000000:
                    fc.clear()
                fc[key] = v
            self.pc = (pc + 1) & 0xFFFF
            return v
        v = self.bus.read(bank, pc)
        self.pc = (pc + 1) & 0xFFFF
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
        if bank <= 0x7D and (addr >= 0x8000 or 0x40 <= bank <= 0x6F):
            lo = self._rom_byte(bank, addr)
            hi = self._rom_byte(bank, (addr + 1) & 0xFFFF)
            return lo | (hi << 8)
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

    def _br_any(self, op):
        off = self._fetch8()
        cyc = 2
        if off & 0x80: off -= 256
        cond = {
            0x10: not self.n, 0x30: self.n, 0x50: not self.v, 0x70: self.v,
            0x80: True, 0x90: not self.c, 0xB0: self.c, 0xD0: not self.z, 0xF0: self.z
        }[op]
        if cond:
            self.pc = (self.pc + off) & 0xFFFF
            cyc = 3
        self.cycles += cyc
        return cyc

    def _build_hot_ops(self):
        """Fast-path dispatch for the hottest instructions (BIT/branches/LDA/STA/CMP
        dominate real game code). Handlers mirror the main chain semantics exactly."""
        nz8 = self._nz8
        nz16 = self._nz16
        cmpv = self._cmp_val
        f8 = self._fetch8
        f16 = self._fetch16
        rd = self.bus.read
        cpu = self

        handlers = {}
        for bop in (0x10, 0x30, 0x50, 0x70, 0x80, 0x90, 0xB0, 0xD0, 0xF0):
            def make_br(op=bop):
                def run():
                    return cpu._br_any(op)
                return run
            handlers[bop] = make_br()

        def cyc2(fn):
            def run():
                fn()
                cpu.cycles += 2
                return 2
            return run

        def bit(b, ad):
            if cpu.m_flag:
                m = rd(b, ad)
                cpu.z = int(((cpu.a & 0xFF) & m) == 0)
                cpu.n = (m >> 7) & 1; cpu.v = (m >> 6) & 1
            else:
                m = cpu._read16(b, ad)
                cpu.z = int((cpu.a & m) == 0)
                cpu.n = (m >> 15) & 1; cpu.v = (m >> 14) & 1

        handlers[0x2C] = cyc2(lambda: bit(cpu.dbr, f16()))
        handlers[0x24] = cyc2(lambda: bit(0, (cpu.d + f8()) & 0xFFFF))
        handlers[0x3C] = cyc2(lambda: bit(cpu.dbr, (f16() + cpu.x) & 0xFFFF))
        handlers[0x34] = cyc2(lambda: bit(0, (cpu.d + f8() + cpu.x) & 0xFFFF))

        def lda_imm():
            if cpu.m_flag:
                m = f8(); cpu.a = (cpu.a & 0xFF00) | m; nz8(m)
            else:
                m = f16(); cpu.a = m; nz16(m)

        def lda_dp(): cpu._lda_val(0, (cpu.d + f8()) & 0xFFFF)
        def lda_dpx(): cpu._lda_val(0, (cpu.d + f8() + cpu.x) & 0xFFFF)

        def lda_abs():
            ad = f16()
            cpu._lda_val(cpu.dbr, ad)

        def lda_absx():
            ad = f16()
            full = ((cpu.dbr << 16) + ad + cpu.x) & 0xFFFFFF
            cpu._lda_val(full >> 16, full & 0xFFFF)

        def lda_absy():
            ad = f16()
            full = ((cpu.dbr << 16) + ad + cpu.y) & 0xFFFFFF
            cpu._lda_val(full >> 16, full & 0xFFFF)

        def sta_dp(): cpu._sta_val(0, (cpu.d + f8()) & 0xFFFF)
        def sta_dpx(): cpu._sta_val(0, (cpu.d + f8() + cpu.x) & 0xFFFF)

        def sta_abs():
            ad = f16()
            cpu._sta_val(cpu.dbr, ad)

        def sta_absx():
            ad = f16()
            full = ((cpu.dbr << 16) + ad + cpu.x) & 0xFFFFFF
            cpu._sta_val(full >> 16, full & 0xFFFF)

        def sta_absy():
            ad = f16()
            full = ((cpu.dbr << 16) + ad + cpu.y) & 0xFFFFFF
            cpu._sta_val(full >> 16, full & 0xFFFF)

        def cmp_imm(): cmpv(cpu.a, f8() if cpu.m_flag else f16(), bool(cpu.m_flag))
        def cmp_dp():
            ad = (cpu.d + f8()) & 0xFFFF
            cmpv(cpu.a, rd(0, ad) if cpu.m_flag else cpu._read16(0, ad), bool(cpu.m_flag))

        def cmp_abs():
            ad = f16()
            cmpv(cpu.a, rd(cpu.dbr, ad) if cpu.m_flag else cpu._read16(cpu.dbr, ad), bool(cpu.m_flag))

        def nop():
            cpu.cycles += 2
            return 2

        def inc_a():
            if cpu.m_flag:
                v = ((cpu.a & 0xFF) + 1) & 0xFF; cpu.a = (cpu.a & 0xFF00) | v; nz8(v)
            else:
                cpu.a = (cpu.a + 1) & 0xFFFF; nz16(cpu.a)
            cpu.cycles += 2
            return 2

        def dec_a():
            if cpu.m_flag:
                v = ((cpu.a & 0xFF) - 1) & 0xFF; cpu.a = (cpu.a & 0xFF00) | v; nz8(v)
            else:
                cpu.a = (cpu.a - 1) & 0xFFFF; nz16(cpu.a)
            cpu.cycles += 2
            return 2

        def jmp_abs():
            cpu.pc = f16()
            cpu.cycles += 2
            return 2

        handlers[0xA9] = cyc2(lda_imm)
        handlers[0xA5] = cyc2(lda_dp)
        handlers[0xB5] = cyc2(lda_dpx)
        handlers[0xAD] = cyc2(lda_abs)
        handlers[0xBD] = cyc2(lda_absx)
        handlers[0xB9] = cyc2(lda_absy)
        handlers[0x85] = cyc2(sta_dp)
        handlers[0x95] = cyc2(sta_dpx)
        handlers[0x8D] = cyc2(sta_abs)
        handlers[0x9D] = cyc2(sta_absx)
        handlers[0x99] = cyc2(sta_absy)
        handlers[0xC9] = cyc2(cmp_imm)
        handlers[0xC5] = cyc2(cmp_dp)
        handlers[0xCD] = cyc2(cmp_abs)
        handlers[0xEA] = nop
        handlers[0x1A] = inc_a
        handlers[0x3A] = dec_a
        handlers[0x4C] = jmp_abs
        return handlers

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
        hot = self._hot_ops
        if hot is not None:
            hf = hot.get(op)
            if hf is not None:
                return hf()
        cyc = 2

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
        hot = self._hot_ops
        if hot is not None:
            hf = hot.get(op)
            if hf is not None:
                return hf()
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
        self.apu = SNESAPU(sample_rate, mode="snes")
        self.bus = SNESBus(self.cart, self.ppu, self.apu)
        self.cpu = CPU65c816(self.bus)
        self.bus.cpu = self.cpu
        self.frames = 0
        self._load_battery()
        self.cpu.reset()
        self._skip_boot_blank()

    def _skip_boot_blank(self):
        """Fast-forward initial WRAM/VRAM clear & SPC700 handshake to avoid black-screen lag."""
        for _ in range(150):
            self.run_frame(draw=False)
            has_gfx = bool(self.ppu.vram_np.any() and any(self.ppu.cgram))
            if has_gfx and not (self.ppu.inidisp & 0x80):
                break
        self.ppu.render_frame()

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
        cpu, ppu, bus, apu = self.cpu, self.ppu, self.bus, self.apu
        return {
            # CPU 65c816
            "a": cpu.a, "x": cpu.x, "y": cpu.y, "sp": cpu.sp,
            "d": cpu.d, "dbr": cpu.dbr, "pbr": cpu.pbr, "pc": cpu.pc,
            "p": cpu.get_p(), "e": cpu.e,
            "cycles": cpu.cycles, "nmi_pending": cpu.nmi_pending,
            "waiting": getattr(cpu, "waiting", False),
            # Bus / system
            "wram": bytes(bus.wram),
            "wmadd": bus.wmadd,
            "wrmpya": bus.wrmpya, "wrmpyb": bus.wrmpyb,
            "wrdiv": bus.wrdiv, "wrdivb": bus.wrdivb,
            "rddiv": bus.rddiv, "rdmpy": bus.rdmpy,
            "mdmaen": bus.mdmaen, "hdmaen": bus.hdmaen,
            "dma_regs": bytes(bus.dma_regs),
            "joy1": bus.joy1, "joy2": bus.joy2,
            "joy_shift1": bus.joy_shift1, "joy_shift2": bus.joy_shift2,
            "joy_strobe": bus.joy_strobe, "timeup": bus.timeup,
            "hdma_cur": list(bus._hdma_cur), "hdma_cnt": list(bus._hdma_cnt),
            # PPU (all registers that affect rendering / timing)
            "vram": bytes(ppu.vram), "cgram": bytes(ppu.cgram), "oam": bytes(ppu.oam),
            "inidisp": ppu.inidisp, "obsel": ppu.obsel,
            "oam_addl": ppu.oam_addl, "oam_addh": ppu.oam_addh,
            "oam_addr": ppu.oam_addr, "oam_latch": ppu.oam_latch,
            "bgmode": ppu.bgmode, "mosaic": ppu.mosaic,
            "bg_sc": list(ppu.bg_sc), "bg_nba": list(ppu.bg_nba),
            "bg_hofs": list(ppu.bg_hofs), "bg_vofs": list(ppu.bg_vofs),
            "bg_prev": ppu.bg_prev,
            "vmain": ppu.vmain, "vmadd": ppu.vmadd, "vram_prefetch": ppu.vram_prefetch,
            "cgadd": ppu.cgadd, "cg_latch": ppu.cg_latch,
            "tm": ppu.tm, "ts": ppu.ts, "cgwsel": ppu.cgwsel, "cgadsub": ppu.cgadsub,
            "coldata": list(ppu.coldata),
            "w12sel": ppu.w12sel, "w34sel": ppu.w34sel, "wobjsel": ppu.wobjsel,
            "wh0": ppu.wh0, "wh1": ppu.wh1, "wh2": ppu.wh2, "wh3": ppu.wh3,
            "wbglog": ppu.wbglog, "wobjlog": ppu.wobjlog, "tmw": ppu.tmw, "tsw": ppu.tsw,
            "nmitimen": ppu.nmitimen, "rdnmi": ppu.rdnmi, "hvbjoy": ppu.hvbjoy,
            "h_latch": ppu.h_latch, "v_latch": ppu.v_latch,
            "ophct_flip": ppu.ophct_flip, "opvct_flip": ppu.opvct_flip,
            "m7_latch": ppu.m7_latch, "m7a": ppu.m7a, "m7b": ppu.m7b,
            "m7c": ppu.m7c, "m7d": ppu.m7d, "m7x": ppu.m7x, "m7y": ppu.m7y,
            "mpy_result": ppu.mpy_result,
            "frame_rgba": ppu.frame_rgba.copy(),
            # APU (ports + synthesis state so sound does not glitch on restore)
            "apu_ports_in": list(apu.ports_in), "apu_ports_out": list(apu.ports_out),
            "apu_read_polls": list(apu.read_polls), "apu_boot_step": apu.boot_step,
            "apu_phase": apu.phase.copy(), "apu_freqs": apu.freqs.copy(),
            "apu_amps": apu.amps.copy(), "apu_note_decay": apu.note_decay.copy(),
            "apu_acc": apu.acc, "apu_dc": apu.dc, "apu_irq": apu.irq,
            "frames": self.frames,
        }

    def restore_state(self, st):
        cpu, ppu, bus, apu = self.cpu, self.ppu, self.bus, self.apu
        # CPU
        cpu.a, cpu.x, cpu.y = st["a"], st["x"], st["y"]
        cpu.sp, cpu.d, cpu.dbr = st["sp"], st["d"], st["dbr"]
        cpu.pbr, cpu.pc, cpu.e = st["pbr"], st["pc"], st["e"]
        cpu.set_p(st["p"])
        cpu.cycles = st.get("cycles", 0)
        cpu.nmi_pending = st.get("nmi_pending", False)
        cpu.waiting = st.get("waiting", False)
        # Bus
        bus.wram[:] = st["wram"]
        bus.wmadd = st.get("wmadd", 0)
        bus.wrmpya = st.get("wrmpya", 0xFF)
        bus.wrmpyb = st.get("wrmpyb", 0)
        bus.wrdiv = st.get("wrdiv", 0xFFFF)
        bus.wrdivb = st.get("wrdivb", 0)
        bus.rddiv = st.get("rddiv", 0)
        bus.rdmpy = st.get("rdmpy", 0)
        bus.mdmaen = st.get("mdmaen", 0)
        bus.hdmaen = st.get("hdmaen", 0)
        if "dma_regs" in st:
            bus.dma_regs[:] = st["dma_regs"]
        bus.joy1 = st.get("joy1", 0)
        bus.joy2 = st.get("joy2", 0)
        bus.joy_shift1 = st.get("joy_shift1", 0)
        bus.joy_shift2 = st.get("joy_shift2", 0)
        bus.joy_strobe = st.get("joy_strobe", 0)
        bus.timeup = st.get("timeup", 0)
        if "hdma_cur" in st:
            bus._hdma_cur[:] = st["hdma_cur"]
        if "hdma_cnt" in st:
            bus._hdma_cnt[:] = st["hdma_cnt"]
        bus._line_base = cpu.cycles
        bus.mdmaen = 0
        # PPU
        ppu.vram[:] = st["vram"]
        ppu.cgram[:] = st["cgram"]
        ppu.oam[:] = st["oam"]
        for attr in (
            "inidisp", "obsel", "oam_addl", "oam_addh", "oam_addr", "oam_latch",
            "bgmode", "mosaic", "bg_prev", "vmain", "vmadd", "vram_prefetch",
            "cgadd", "cg_latch", "tm", "ts", "cgwsel", "cgadsub",
            "w12sel", "w34sel", "wobjsel", "wh0", "wh1", "wh2", "wh3",
            "wbglog", "wobjlog", "tmw", "tsw", "nmitimen", "rdnmi", "hvbjoy",
            "h_latch", "v_latch", "ophct_flip", "opvct_flip",
            "m7_latch", "m7a", "m7b", "m7c", "m7d", "m7x", "m7y", "mpy_result",
        ):
            if attr in st:
                setattr(ppu, attr, st[attr])
        if "bg_sc" in st:
            ppu.bg_sc[:] = st["bg_sc"]
        if "bg_nba" in st:
            ppu.bg_nba[:] = st["bg_nba"]
        if "bg_hofs" in st:
            ppu.bg_hofs[:] = st["bg_hofs"]
        if "bg_vofs" in st:
            ppu.bg_vofs[:] = st["bg_vofs"]
        if "coldata" in st:
            ppu.coldata[:] = st["coldata"]
        if "frame_rgba" in st:
            ppu.frame_rgba[:] = st["frame_rgba"]
        # APU
        if "apu_ports_in" in st:
            apu.ports_in[:] = st["apu_ports_in"]
            apu.ports_out[:] = st["apu_ports_out"]
            apu.read_polls[:] = st["apu_read_polls"]
            apu.boot_step = st.get("apu_boot_step", 0)
            apu.phase[:] = st["apu_phase"]
            apu.freqs[:] = st["apu_freqs"]
            apu.amps[:] = st["apu_amps"]
            apu.note_decay[:] = st["apu_note_decay"]
            apu.acc = st.get("apu_acc", 0.0)
            apu.dc = st.get("apu_dc", 0.0)
            apu.irq = st.get("apu_irq", False)
        self.frames = st.get("frames", self.frames)

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
        self._skip_boot_blank()

    def set_buttons(self, pad, buttons):
        if pad != 0:
            return
        snes_joy = 0
        if buttons & BTN_B:      snes_joy |= 0x8000
        if buttons & BTN_SELECT: snes_joy |= 0x2000
        if buttons & BTN_START:  snes_joy |= 0x1000
        if buttons & BTN_UP:     snes_joy |= 0x0800
        if buttons & BTN_DOWN:   snes_joy |= 0x0400
        if buttons & BTN_LEFT:   snes_joy |= 0x0200
        if buttons & BTN_RIGHT:  snes_joy |= 0x0100
        if buttons & BTN_A:      snes_joy |= 0x0080
        self.bus.joy1 = snes_joy

    def _snapshot_row_state(self):
        """Render-relevant PPU state for one scanline (HDMA/raster effects)."""
        return self.ppu._snapshot_row_state()

    def run_frame(self, draw=True):
        """NTSC frame: 262 lines, short scanline at 240, VBlank at 225.

        HDMA executes in HBlank at the end of the preceding line, and the PPU
        state for each visible line is snapshotted before the CPU runs that
        line, so mid-frame register writes (raster effects) land on the next
        line instead of being lost to an end-of-frame render.
        """
        cpu, ppu, bus = self.cpu, self.ppu, self.bus
        step = cpu.step
        line_states = [None] * 224
        self._line_states = line_states
        for sl in range(262):
            cpl = _PYSNES_SHORT_CPU_CYCLES if sl == _PYSNES_SHORT_SCANLINE else _PYSNES_NORMAL_CPU_CYCLES
            bus._cycles_per_line = cpl
            bus._line_base = cpu.cycles
            ppu.v_latch = sl
            bus._vblank_line = (sl - 225) if sl >= 225 else 999

            # VBlank edge a few dots into line 225
            if sl == 225:
                for _ in range(_PYSNES_VBLANK_DOT_OFFSET):
                    if not cpu.waiting:
                        step()
                ppu.hvbjoy |= 0x80
                ppu.rdnmi |= 0x80
                if ppu.nmitimen & 0x80:
                    cpu.nmi_pending = True
            elif sl == 0:
                ppu.hvbjoy &= ~0x80
                ppu.rdnmi &= ~0x80

            # IRQ from HTIME/VTIME
            if (ppu.nmitimen & 0x30) and sl == bus.vtime:
                if (ppu.nmitimen & 0x20) or ((ppu.nmitimen & 0x10) and bus.htime < cpl):
                    bus.timeup = 0x80
                    cpu.trigger_irq()

            target = cpu.cycles + cpl
            while cpu.cycles < target:
                if cpu.waiting:
                    # WAI sleeps, but the clock keeps running so $4212 HBlank
                    # still sets and pending NMI/IRQ wake the CPU mid-line.
                    if cpu.nmi_pending or (not cpu.i and bus.timeup):
                        cpu.waiting = False
                    else:
                        cpu.cycles = min(target, cpu.cycles + 4)
                        continue
                step()

            # HBlank of this line: run HDMA for the NEXT line, then snapshot
            # that line's PPU state so the render (below) uses it.
            nxt = (sl + 1) if sl < 261 else 0
            if nxt < 224:
                if bus.hdmaen:
                    bus._run_hdma_scanline(nxt)
                line_states[nxt] = self._snapshot_row_state()

        if draw:
            ppu.render_frame(line_states)
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


GB_DUTY = ((0, 0, 0, 0, 0, 0, 0, 1),
           (1, 0, 0, 0, 0, 0, 0, 1),
           (1, 0, 0, 0, 0, 1, 1, 1),
           (0, 1, 1, 1, 1, 1, 1, 0))
GB_CLOCK_HZ = 4194304.0


class _GBEnv:
    """Shared length counter + volume envelope unit for pulse/noise channels."""
    def __init__(self):
        self.length = 0
        self.len_en = False
        self.init_vol = 0
        self.env_dir = 1
        self.env_period = 0
        self.env_timer = 0
        self.vol = 0

    def nrx1(self, max_len, v):
        self.length = max_len - (v & (max_len - 1))

    def nrx2(self, v):
        self.init_vol = (v >> 4) & 0x0F
        self.env_dir = 1 if (v & 0x08) else -1
        self.env_period = v & 0x07

    def nrx4(self, v):
        self.len_en = bool(v & 0x40)

    def trigger(self, max_len):
        if self.length == 0:
            self.length = max_len
        self.vol = self.init_vol
        self.env_timer = self.env_period

    def clock_len(self):
        if self.len_en and self.length > 0:
            self.length -= 1

    def clock_env(self):
        if self.env_period == 0:
            return
        self.env_timer -= 1
        if self.env_timer <= 0:
            self.env_timer = self.env_period
            self.vol = min(15, self.vol + 1) if self.env_dir > 0 else max(0, self.vol - 1)


class GBPulse:
    """DMG pulse channel 1/2: duty-cycle sequencer clocked at
    (2048 - freq) * 4 T-cycles per step, envelope + optional sweep."""
    def __init__(self, ch):
        self.ch = ch
        self.duty = 0
        self.raw_freq = 0
        self.seq_idx = 0
        self.seq_phase = 0.0
        self.env_unit = _GBEnv()
        self.sweep_en = False
        self.sweep_period = 0
        self.sweep_neg = False
        self.sweep_shift = 0
        self.sweep_timer = 0
        self.sweep_freq = 0
        self.enabled = False

    def dac_from(self, v):
        return (v & 0xF8) != 0

    def nrx0(self, v):
        self.sweep_en = bool(v & 0x80)
        self.sweep_period = (v >> 4) & 7
        self.sweep_neg = bool(v & 0x08)
        self.sweep_shift = v & 7

    def nrx3(self, v):
        self.raw_freq = (self.raw_freq & 0x700) | v

    def nrx4(self, v):
        self.raw_freq = (self.raw_freq & 0xFF) | ((v & 7) << 8)

    def trigger(self):
        self.enabled = self.dac
        e = self.env_unit
        e.trigger(64)
        self.seq_idx = 0
        self.seq_phase = 0.0
        if self.ch == 0:
            self.sweep_freq = self.raw_freq
            self.sweep_timer = self.sweep_period or 8
            self.sweep_en = self.sweep_en or self.sweep_period > 0 or self.sweep_shift > 0
            if self.sweep_shift:
                self._sweep_calc()

    def _sweep_calc(self):
        d = self.sweep_freq >> self.sweep_shift
        new = self.sweep_freq - d - 1 if self.sweep_neg else self.sweep_freq + d
        if new > 2047:
            self.enabled = False
        return new

    def clock_sweep(self):
        self.sweep_timer -= 1
        if self.sweep_timer <= 0:
            self.sweep_timer = self.sweep_period or 8
            if self.sweep_en and self.sweep_period:
                new = self._sweep_calc()
                if new <= 2047 and self.sweep_shift:
                    self.sweep_freq = new
                    self.raw_freq = new
                    self._sweep_calc()

    def clock_len_ctrl(self, io):
        e = self.env_unit
        was = e.length
        e.clock_len()
        if e.length == 0 and was > 0:
            io[0x26] &= ~(0x01 << self.ch) & 0xFF
            self.enabled = False

    def advance(self, cycles, io):
        e = self.env_unit
        period = max(4.0, 4.0 * (2048 - self.raw_freq))
        self.seq_phase += cycles
        while self.seq_phase >= period:
            self.seq_phase -= period
            self.seq_idx = (self.seq_idx + 1) & 7
        if not (self.enabled and self.dac and e.length > 0 and e.vol > 0 and io[0x26] & 0x80):
            return 0.0, 0.0
        out = GB_DUTY[self.duty][self.seq_idx] * e.vol
        return out / 15.0, out / 15.0


class GBWave:
    """DMG wave channel: 32 4-bit samples from wave RAM clocked at
    (2048 - freq) * 2 T-cycles per sample, with volume shift."""
    def __init__(self):
        self.raw_freq = 0
        self.pos = 0
        self.pos_phase = 0.0
        self.env_unit = _GBEnv()
        self.enabled = False
        self.vol_code = 0
        self.dac = False

    def nrx0(self, v):
        self.dac = bool(v & 0x80)

    def nrx2(self, v):
        self.vol_code = (v >> 5) & 3

    def nrx3(self, v):
        self.raw_freq = (self.raw_freq & 0x700) | v

    def nrx4(self, v):
        self.raw_freq = (self.raw_freq & 0xFF) | ((v & 7) << 8)

    def trigger(self):
        self.enabled = self.dac
        e = self.env_unit
        if e.length == 0:
            e.length = 256
        self.pos = 0
        self.pos_phase = 0.0

    def clock_len_ctrl(self, io):
        e = self.env_unit
        was = e.length
        e.clock_len()
        if e.length == 0 and was > 0:
            io[0x26] &= ~0x04
            self.enabled = False

    def advance(self, cycles, io):
        period = max(2.0, 2.0 * (2048 - self.raw_freq))
        self.pos_phase += cycles
        while self.pos_phase >= period:
            self.pos_phase -= period
            self.pos = (self.pos + 1) & 31
        e = self.env_unit
        if not (self.enabled and self.dac and e.length > 0 and self.vol_code and io[0x26] & 0x80):
            return 0.0, 0.0
        b = io[0x30 + (self.pos >> 1)]
        v = (b >> 4) & 0x0F if (self.pos & 1) == 0 else b & 0x0F
        gain = (0, 1, 2)[self.vol_code - 1]
        return (v >> gain) / 15.0, (v >> gain) / 15.0


class GBNoise:
    """DMG noise channel: 15/7-bit LFSR clocked at
    524288 / divisor / 2^(shift+1) Hz, envelope unit attached."""
    def __init__(self):
        self.env_unit = _GBEnv()
        self.div = 0
        self.shift = 0
        self.width7 = False
        self.lfsr = 0x7FFF
        self.lfsr_phase = 0.0
        self.enabled = False
        self.dac = False

    def nrx3(self, v):
        self.width7 = bool(v & 0x08)
        self.shift = (v >> 4) & 0x0F
        self.div = v & 7

    def nrx4(self, v):
        pass

    def trigger(self):
        self.enabled = self.dac
        e = self.env_unit
        e.trigger(64)
        self.lfsr = 0x7FFF
        self.lfsr_phase = 0.0

    def clock_len_ctrl(self, io):
        e = self.env_unit
        was = e.length
        e.clock_len()
        if e.length == 0 and was > 0:
            io[0x26] &= ~0x08
            self.enabled = False

    def advance(self, cycles, io):
        d = 8 if self.div == 0 else self.div
        period = 8.0 * d * (1 << (self.shift + 1))
        self.lfsr_phase += cycles
        while self.lfsr_phase >= period:
            self.lfsr_phase -= period
            fb = (self.lfsr ^ (self.lfsr >> 1)) & 1
            self.lfsr = (self.lfsr >> 1) | (fb << (6 if self.width7 else 14))
        e = self.env_unit
        if not (self.enabled and self.dac and e.length > 0 and e.vol > 0 and io[0x26] & 0x80):
            return 0.0, 0.0
        out = 0.0 if (self.lfsr & 1) else e.vol
        return out / 15.0, out / 15.0


class GBAPU:
    """ rewritten Game Boy / Game Boy Color audio unit (DMG 4-channel APU).

    Real hardware model instead of a port synth:
    - two pulse channels with duty sequencers, length counters, envelopes
      and (ch1) a frequency sweep unit
    - wave channel with 32-step 4-bit sample playback from wave RAM
    - noise channel with a stepped 15/7-bit LFSR and envelope
    - 512 Hz frame sequencer driving length/sweep/envelope clocks (NR52 on)
    - NR50 master volumes and NR51 per-channel panning mixed per sample
    """
    FS_STEP_T = 8192  # T-cycles per frame-sequencer step (512 Hz)

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.volume = 0.7
        self.sink = None
        self.ch1 = GBPulse(0)
        self.ch2 = GBPulse(1)
        self.ch3 = GBWave()
        self.ch4 = GBNoise()
        self.fs_step = 0
        self.fs_phase = 0.0
        self._sample_acc = 0.0
        self._step_acc = 0.0
        self._hw_acc = 0.0
        self.buf = []
        self.dc = 0.0
        self.irq = False

    def set_io(self, io):
        self.gb_io = io

    def power(self, io):
        return bool(io[0x26] & 0x80)

    def write(self, r, v, io):
        if r == 0x26:
            return
        if r == 0x10:
            self.ch1.nrx0(v)
        elif r == 0x11:
            self.ch1.duty = (v >> 6) & 3
            self.ch1.env_unit.nrx1(64, v)
        elif r == 0x12:
            self.ch1.env_unit.nrx2(v)
            self.ch1.dac = self.ch1.dac_from(v)
        elif r == 0x13:
            self.ch1.nrx3(v)
        elif r == 0x14:
            self.ch1.env_unit.nrx4(v)
            self.ch1.nrx4(v)
            if v & 0x80:
                self.ch1.trigger()
                if self.ch1.enabled:
                    io[0x26] |= 0x01
        elif r == 0x16:
            self.ch2.duty = (v >> 6) & 3
            self.ch2.env_unit.nrx1(64, v)
        elif r == 0x17:
            self.ch2.env_unit.nrx2(v)
            self.ch2.dac = self.ch2.dac_from(v)
        elif r == 0x18:
            self.ch2.nrx3(v)
        elif r == 0x19:
            self.ch2.env_unit.nrx4(v)
            self.ch2.nrx4(v)
            if v & 0x80:
                self.ch2.trigger()
                if self.ch2.enabled:
                    io[0x26] |= 0x02
        elif r == 0x1A:
            self.ch3.nrx0(v)
            if not self.ch3.dac:
                self.ch3.enabled = False
                io[0x26] &= ~0x04
        elif r == 0x1B:
            self.ch3.env_unit.nrx1(256, v)
        elif r == 0x1C:
            self.ch3.nrx2(v)
        elif r == 0x1D:
            self.ch3.nrx3(v)
        elif r == 0x1E:
            self.ch3.env_unit.nrx4(v)
            self.ch3.nrx4(v)
            if v & 0x80:
                self.ch3.trigger()
                if self.ch3.enabled:
                    io[0x26] |= 0x04
        elif r == 0x21:
            self.ch4.env_unit.nrx2(v)
            self.ch4.dac = (v & 0xF8) != 0
        elif r == 0x22:
            self.ch4.nrx3(v)
        elif r == 0x23:
            self.ch4.env_unit.nrx4(v)
            if v & 0x80:
                self.ch4.trigger()
                if self.ch4.enabled:
                    io[0x26] |= 0x08

    def frame_step(self, io):
        s = self.fs_step
        if s in (0, 2, 4, 6):
            self.ch1.clock_len_ctrl(io)
            self.ch2.clock_len_ctrl(io)
            self.ch3.clock_len_ctrl(io)
            self.ch4.clock_len_ctrl(io)
        if s in (2, 6):
            self.ch1.clock_sweep()
        if s == 7:
            for c in (self.ch1, self.ch2, self.ch4):
                c.env_unit.clock_env()
        self.fs_step = (s + 1) & 7

    def tick(self, cycles, io):
        """Advance the APU by `cycles` T-cycles (called after each CPU step)."""
        if io[0x26] & 0x80:
            self.fs_phase += cycles
            while self.fs_phase >= self.FS_STEP_T:
                self.fs_phase -= self.FS_STEP_T
                self.frame_step(io)
        self._sample_acc += cycles * self.sr / GB_CLOCK_HZ
        while self._sample_acc >= 1.0 and len(self.buf) < 16384:
            self._sample_acc -= 1.0
            self._hw_acc += GB_CLOCK_HZ / self.sr
            whole = int(self._hw_acc)
            self._hw_acc -= whole
            self.buf.append(self._mix(io, whole))

    def _mix(self, io, cycles):
        ch1 = self.ch1
        ch2 = self.ch2
        ch3 = self.ch3
        ch4 = self.ch4
        a1l, a1r = ch1.advance(cycles, io)
        a2l, a2r = ch2.advance(cycles, io)
        a3l, a3r = ch3.advance(cycles, io)
        a4l, a4r = ch4.advance(cycles, io)
        nr50 = io[0x24] or 0x77
        nr51 = io[0x25]
        left = a1l if nr51 & 0x10 else 0.0
        left += a2l if nr51 & 0x20 else 0.0
        left += a3l if nr51 & 0x40 else 0.0
        left += a4l if nr51 & 0x80 else 0.0
        right = a1r if nr51 & 0x01 else 0.0
        right += a2r if nr51 & 0x02 else 0.0
        right += a3r if nr51 & 0x04 else 0.0
        right += a4r if nr51 & 0x08 else 0.0
        lv = ((nr50 >> 4) & 7) + 1
        rv = (nr50 & 7) + 1
        return (left * lv / 60.0 + right * rv / 60.0) / 2.0

    def capture_state(self):
        return {
            "fs_step": self.fs_step, "fs_phase": self.fs_phase,
            "ch1": vars(self.ch1).copy(), "ch2": vars(self.ch2).copy(),
            "ch3": vars(self.ch3).copy(), "ch4": vars(self.ch4).copy(),
            "e1": vars(self.ch1.env_unit).copy(), "e2": vars(self.ch2.env_unit).copy(),
            "e3": vars(self.ch3.env_unit).copy(), "e4": vars(self.ch4.env_unit).copy(),
            "sample_acc": self._sample_acc, "step_acc": self._step_acc,
            "hw_acc": self._hw_acc, "buf": list(self.buf),
        }

    def restore_state(self, st):
        if not st:
            return
        self.fs_step = st.get("fs_step", 0)
        self.fs_phase = st.get("fs_phase", 0.0)
        for name, ch in (("ch1", self.ch1), ("ch2", self.ch2), ("ch3", self.ch3), ("ch4", self.ch4)):
            if name in st and isinstance(st[name], dict):
                for k, v in st[name].items():
                    if k != "env_unit" and hasattr(ch, k):
                        setattr(ch, k, v)
        for name, e in (("e1", self.ch1.env_unit), ("e2", self.ch2.env_unit),
                        ("e3", self.ch3.env_unit), ("e4", self.ch4.env_unit)):
            if name in st and isinstance(st[name], dict):
                var_dict = vars(e)
                for k, v in st[name].items():
                    if k in var_dict:
                        var_dict[k] = v
        self._sample_acc = st.get("sample_acc", 0.0)
        self._step_acc = st.get("step_acc", 0.0)
        self._hw_acc = st.get("hw_acc", 0.0)
        if "buf" in st:
            self.buf = list(st["buf"])

    def power_off(self):
        for ch in (self.ch1, self.ch2, self.ch3, self.ch4):
            ch.enabled = False
            if hasattr(ch, "env_unit"):
                ch.env_unit.length = 0
                ch.env_unit.vol = 0
        if hasattr(self, "gb_io") and self.gb_io is not None:
            self.gb_io[0x26] = 0

    def run_frame(self):
        if self.sink is None:
            self.buf.clear()
            return
        if not self.buf:
            return
        out = np.asarray(self.buf, dtype=np.float32)
        self.buf.clear()
        m = float(out.mean())
        out -= self.dc
        self.dc += (m - self.dc) * 0.015
        out = np.tanh(out * 1.4) * 0.25
        pcm = np.clip(out * (32767.0 * self.volume), -32767, 32767).astype(np.int16)
        self.sink(pcm)


class GBCPU:
    """ rewritten Game Boy CPU core (Sharp SM83/LR35902).

    Owns all registers (register-file array + SP/PC/flags), dispatches every
    opcode and CB-prefixed opcode, implements the delayed-IME behavior of EI,
    the HALT bug, and per-opcode T-cycle counts including conditional
    branches. Memory goes through the GameBoy `read`/`write` callbacks.
    """
    FZ, FN, FH, FC = 0x80, 0x40, 0x20, 0x10

    def __init__(self, gb):
        self.gb = gb
        self.r8 = [0x00, 0x13, 0x00, 0xD8, 0x01, 0x4D, 0x00, 0x01]  # B C D E H L (HL) A
        self.f = 0xB0
        self.sp = 0xFFFE
        self.pc = 0x0100
        self.ime = False
        self.ime_delay = False
        self.halted = False
        self.halt_bug = False

    def reset(self, cgb=False):
        self.r8 = [0x00, 0x13, 0x00, 0xD8, 0x01, 0x4D, 0x00, 0x11 if cgb else 0x01]
        self.f = 0xB0
        self.sp = 0xFFFE
        self.pc = 0x0100
        self.ime = False
        self.ime_delay = False
        self.halted = False
        self.halt_bug = False

    @property
    def a(self):
        return self.r8[7]

    @a.setter
    def a(self, v):
        self.r8[7] = v & 0xFF

    def _get_r(self, idx):
        if idx == 6:
            return self.gb.read((self.r8[4] << 8) | self.r8[5])
        return self.r8[idx]

    def _set_r(self, idx, val):
        val &= 0xFF
        if idx == 6:
            self.gb.write((self.r8[4] << 8) | self.r8[5], val)
        else:
            self.r8[idx] = val

    def _rr(self, hi, lo):
        return (self.r8[hi] << 8) | self.r8[lo]

    def _set_rr(self, hi, lo, v):
        self.r8[hi] = (v >> 8) & 0xFF
        self.r8[lo] = v & 0xFF

    def _read16(self, addr):
        return self.gb.read(addr) | (self.gb.read((addr + 1) & 0xFFFF) << 8)

    def _push16(self, val):
        self.sp = (self.sp - 1) & 0xFFFF
        self.gb.write(self.sp, (val >> 8) & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF
        self.gb.write(self.sp, val & 0xFF)

    def _pop16(self):
        lo = self.gb.read(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        hi = self.gb.read(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return (hi << 8) | lo

    def _alu(self, op, val):
        a = self.r8[7]
        c = (self.f >> 4) & 1
        if op == 0:  # ADD
            r = a + val
            self.f = (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) + (val & 15) > 15 else 0) | (0x10 if r > 255 else 0)
            self.r8[7] = r & 0xFF
        elif op == 1:  # ADC
            r = a + val + c
            self.f = (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) + (val & 15) + c > 15 else 0) | (0x10 if r > 255 else 0)
            self.r8[7] = r & 0xFF
        elif op == 2:  # SUB
            r = a - val
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) else 0) | (0x10 if r < 0 else 0)
            self.r8[7] = r & 0xFF
        elif op == 3:  # SBC
            r = a - val - c
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) + c else 0) | (0x10 if r < 0 else 0)
            self.r8[7] = r & 0xFF
        elif op == 4:  # AND
            self.r8[7] = a & val
            self.f = (0x80 if self.r8[7] == 0 else 0) | 0x20
        elif op == 5:  # XOR
            self.r8[7] = a ^ val
            self.f = 0x80 if self.r8[7] == 0 else 0
        elif op == 6:  # OR
            self.r8[7] = a | val
            self.f = 0x80 if self.r8[7] == 0 else 0
        else:  # CP
            r = a - val
            self.f = 0x40 | (0x80 if (r & 0xFF) == 0 else 0) | (0x20 if (a & 15) < (val & 15) else 0) | (0x10 if r < 0 else 0)

    def step(self):
        gb = self.gb
        pending = gb.hram[0x7F] & gb.io[0x0F] & 0x1F
        was_halted = self.halted
        if pending:
            self.halted = False
            if self.ime:
                self.ime = False
                for bit in range(5):
                    if pending & (1 << bit):
                        gb.io[0x0F] &= ~(1 << bit)
                        self._push16(self.pc)
                        self.pc = 0x40 + (bit << 3)
                        return 20
        if self.halted:
            return 4
        if self.ime_delay:
            self.ime = True
            self.ime_delay = False
        pc = self.pc
        op = gb.read(pc)
        if self.halt_bug:
            self.halt_bug = False
        else:
            self.pc = (pc + 1) & 0xFFFF
        return self._exec(op, was_halted, pending)

    def _exec(self, op, was_halted, pending):
        gb = self.gb
        f = self.f
        if 0x40 <= op <= 0x7F and op != 0x76:                            # LD r, r'
            self._set_r((op >> 3) & 7, self._get_r(op & 7))
            return 8 if ((op & 7) == 6 or ((op >> 3) & 7) == 6) else 4
        if 0x80 <= op <= 0xBF:                                           # ALU A, r
            self._alu((op >> 3) & 7, self._get_r(op & 7))
            return 8 if (op & 7) == 6 else 4
        if op == 0x76:
            if self.ime or not pending:
                self.halted = True
            else:
                self.halt_bug = True
            return 4                                                     # HALT
        if op == 0x00: return 4                                          # NOP
        if op == 0x10:                                                   # STOP / CGB speed switch
            self.pc = (self.pc + 1) & 0xFFFF
            if gb.cgb and gb.io[0x4D] & 1:
                gb.io[0x4D] = ((gb.io[0x4D] ^ 0x80) & 0x80) | 0x7E
            return 4
        if op == 0xF3:                                                   # DI
            self.ime = False
            self.ime_delay = False
            return 4
        if op == 0xFB:                                                   # EI (delayed one instruction)
            self.ime_delay = True
            return 4
        if (op & 0xC7) == 0x06:                                          # LD r, d8
            v = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self._set_r((op >> 3) & 7, v)
            return 8
        if (op & 0xC7) == 0x04:                                          # INC r
            idx = (op >> 3) & 7
            v = self._get_r(idx); r = (v + 1) & 0xFF; self._set_r(idx, r)
            self.f = (self.f & 0x10) | (0x80 if r == 0 else 0) | (0x20 if (v & 15) == 15 else 0)
            return 12 if idx == 6 else 4
        if (op & 0xC7) == 0x05:                                          # DEC r
            idx = (op >> 3) & 7
            v = self._get_r(idx); r = (v - 1) & 0xFF; self._set_r(idx, r)
            self.f = (self.f & 0x10) | 0x40 | (0x80 if r == 0 else 0) | (0x20 if (v & 15) == 0 else 0)
            return 12 if idx == 6 else 4
        if (op & 0xC7) == 0xC6:                                          # ALU A, d8
            v = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self._alu((op >> 3) & 7, v)
            return 8
        if op in (0x01, 0x11, 0x21, 0x31):                               # LD rr, d16
            v = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            if op == 0x01: self._set_rr(0, 1, v)
            elif op == 0x11: self._set_rr(2, 3, v)
            elif op == 0x21: self._set_rr(4, 5, v)
            else: self.sp = v
            return 12
        if op in (0x03, 0x13, 0x23, 0x33, 0x0B, 0x1B, 0x2B, 0x3B):       # INC/DEC rr
            d = 1 if (op & 8) == 0 else -1
            reg = (op >> 4) & 3
            if reg == 0: self._set_rr(0, 1, (self._rr(0, 1) + d) & 0xFFFF)
            elif reg == 1: self._set_rr(2, 3, (self._rr(2, 3) + d) & 0xFFFF)
            elif reg == 2: self._set_rr(4, 5, (self._rr(4, 5) + d) & 0xFFFF)
            else: self.sp = (self.sp + d) & 0xFFFF
            return 8
        if op in (0x09, 0x19, 0x29, 0x39):                               # ADD HL, rr
            hl = self._rr(4, 5)
            rr = (self._rr(0, 1), self._rr(2, 3), hl, self.sp)[(op >> 4) & 3]
            r = hl + rr
            self.f = (self.f & 0x80) | (0x20 if (hl & 0xFFF) + (rr & 0xFFF) > 0xFFF else 0) | (0x10 if r > 0xFFFF else 0)
            self._set_rr(4, 5, r & 0xFFFF)
            return 8
        if op == 0x02: gb.write(self._rr(0, 1), self.r8[7]); return 8
        if op == 0x12: gb.write(self._rr(2, 3), self.r8[7]); return 8
        if op == 0x22:                                                   # LD (HL+), A
            hl = self._rr(4, 5); gb.write(hl, self.r8[7])
            self._set_rr(4, 5, (hl + 1) & 0xFFFF); return 8
        if op == 0x32:                                                   # LD (HL-), A
            hl = self._rr(4, 5); gb.write(hl, self.r8[7])
            self._set_rr(4, 5, (hl - 1) & 0xFFFF); return 8
        if op == 0x0A: self.r8[7] = gb.read(self._rr(0, 1)); return 8
        if op == 0x1A: self.r8[7] = gb.read(self._rr(2, 3)); return 8
        if op == 0x2A:
            hl = self._rr(4, 5); self.r8[7] = gb.read(hl)
            self._set_rr(4, 5, (hl + 1) & 0xFFFF); return 8
        if op == 0x3A:
            hl = self._rr(4, 5); self.r8[7] = gb.read(hl)
            self._set_rr(4, 5, (hl - 1) & 0xFFFF); return 8
        if op == 0xE0:                                                   # LDH (a8), A
            a8 = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            gb.write(0xFF00 | a8, self.r8[7]); return 12
        if op == 0xF0:                                                   # LDH A, (a8)
            a8 = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            self.r8[7] = gb.read(0xFF00 | a8); return 12
        if op == 0xE2: gb.write(0xFF00 | self.r8[1], self.r8[7]); return 8
        if op == 0xF2: self.r8[7] = gb.read(0xFF00 | self.r8[1]); return 8
        if op == 0xEA:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            gb.write(ad, self.r8[7]); return 16
        if op == 0xFA:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            self.r8[7] = gb.read(ad); return 16
        if op == 0x08:
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            gb.write(ad, self.sp & 0xFF); gb.write((ad + 1) & 0xFFFF, self.sp >> 8); return 20
        if op in (0x18, 0x20, 0x28, 0x30, 0x38):                         # JR
            off = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            if off & 0x80: off -= 256
            cond = True if op == 0x18 else (
                not (self.f & 0x80) if op == 0x20 else (
                bool(self.f & 0x80) if op == 0x28 else (
                not (self.f & 0x10) if op == 0x30 else bool(self.f & 0x10))))
            if cond:
                self.pc = (self.pc + off) & 0xFFFF
                return 12
            return 8
        if op in (0xC3, 0xC2, 0xCA, 0xD2, 0xDA):                         # JP
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            cond = True if op == 0xC3 else (
                not (self.f & 0x80) if op == 0xC2 else (
                bool(self.f & 0x80) if op == 0xCA else (
                not (self.f & 0x10) if op == 0xD2 else bool(self.f & 0x10))))
            if cond:
                self.pc = ad
                return 16
            return 12
        if op == 0xE9: self.pc = self._rr(4, 5); return 4                # JP (HL)
        if op in (0xCD, 0xC4, 0xCC, 0xD4, 0xDC):                         # CALL
            ad = self._read16(self.pc); self.pc = (self.pc + 2) & 0xFFFF
            cond = True if op == 0xCD else (
                not (self.f & 0x80) if op == 0xC4 else (
                bool(self.f & 0x80) if op == 0xCC else (
                not (self.f & 0x10) if op == 0xD4 else bool(self.f & 0x10))))
            if cond:
                self._push16(self.pc); self.pc = ad; return 24
            return 12
        if op in (0xC9, 0xD9, 0xC0, 0xC8, 0xD0, 0xD8):                   # RET / RETI
            if op == 0xD9:
                self.pc = self._pop16(); self.ime_delay = True; return 16
            if op == 0xC9:
                self.pc = self._pop16(); return 16
            cond = (not (self.f & 0x80) if op == 0xC0 else (
                    bool(self.f & 0x80) if op == 0xC8 else (
                    not (self.f & 0x10) if op == 0xD0 else bool(self.f & 0x10))))
            if cond:
                self.pc = self._pop16(); return 20
            return 8
        if (op & 0xC7) == 0xC7:                                          # RST
            self._push16(self.pc); self.pc = op & 0x38; return 16
        if op in (0xC5, 0xD5, 0xE5, 0xF5):                               # PUSH
            v = (self._rr(0, 1), self._rr(2, 3), self._rr(4, 5),
                 (self.r8[7] << 8) | (self.f & 0xF0))[(op >> 4) & 3]
            self._push16(v); return 16
        if op in (0xC1, 0xD1, 0xE1, 0xF1):                               # POP
            v = self._pop16()
            reg = (op >> 4) & 3
            if reg == 0: self._set_rr(0, 1, v)
            elif reg == 1: self._set_rr(2, 3, v)
            elif reg == 2: self._set_rr(4, 5, v)
            else: self.r8[7] = v >> 8; self.f = v & 0xF0
            return 12
        if op == 0x07:                                                   # RLCA
            c = (self.r8[7] >> 7) & 1
            self.r8[7] = ((self.r8[7] << 1) | c) & 0xFF; self.f = c << 4; return 4
        if op == 0x0F:                                                   # RRCA
            c = self.r8[7] & 1
            self.r8[7] = ((self.r8[7] >> 1) | (c << 7)) & 0xFF; self.f = c << 4; return 4
        if op == 0x17:                                                   # RLA
            c = (self.r8[7] >> 7) & 1
            self.r8[7] = ((self.r8[7] << 1) | ((self.f >> 4) & 1)) & 0xFF; self.f = c << 4; return 4
        if op == 0x1F:                                                   # RRA
            c = self.r8[7] & 1
            self.r8[7] = ((self.r8[7] >> 1) | (((self.f >> 4) & 1) << 7)) & 0xFF; self.f = c << 4; return 4
        if op == 0x2F:                                                   # CPL
            self.r8[7] ^= 0xFF; self.f = (self.f & 0x90) | 0x60; return 4
        if op == 0x37:                                                   # SCF
            self.f = (self.f & 0x80) | 0x10; return 4
        if op == 0x3F:                                                   # CCF
            self.f = (self.f & 0x80) ^ 0x10; return 4
        if op == 0xF9:                                                   # LD SP, HL
            self.sp = self._rr(4, 5); return 8
        if op in (0xE8, 0xF8):                                           # ADD SP, r8 / LD HL, SP+r8
            off = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            s8 = (off - 256) if (off & 0x80) else off
            r = (self.sp + s8) & 0xFFFF
            self.f = (0x20 if ((self.sp & 15) + (off & 15)) > 15 else 0) | (0x10 if ((self.sp & 0xFF) + off) > 0xFF else 0)
            if op == 0xE8:
                self.sp = r; return 16
            self._set_rr(4, 5, r); return 12
        if op == 0x27:                                                   # DAA
            a = self.r8[7]
            if not (self.f & 0x40):
                if (self.f & 0x20) or (a & 0x0F) > 9: a += 0x06
                if (self.f & 0x10) or a > 0x9F: a += 0x60; self.f |= 0x10
            else:
                if self.f & 0x20: a = (a - 6) & 0xFF
                if self.f & 0x10: a = (a - 0x60) & 0xFF
                else:
                    if (a & 0x0F) > 9: a -= 0x06
                self.f &= ~0x10
            self.r8[7] = a & 0xFF
            self.f = (self.f & 0x50) | (0x80 if self.r8[7] == 0 else 0)
            return 4
        if op == 0xCB:                                                   # CB prefix
            cb = gb.read(self.pc); self.pc = (self.pc + 1) & 0xFFFF
            return self._exec_cb(cb)
        return 4

    def _exec_cb(self, cb):
        idx = cb & 7
        grp = cb >> 6
        sub = (cb >> 3) & 7
        v = self._get_r(idx)
        if grp == 0:  # rotates / shifts / swap
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
            return 16 if idx == 6 else 8
        if grp == 1:  # BIT
            self.f = (self.f & 0x10) | 0x20 | (0 if (v & (1 << sub)) else 0x80)
            return 12 if idx == 6 else 8
        if grp == 2:  # RES
            self._set_r(idx, v & ~(1 << sub))
            return 16 if idx == 6 else 8
        self._set_r(idx, v | (1 << sub))  # SET
        return 16 if idx == 6 else 8


class GBPPU:
    """ rewritten Game Boy / Game Boy Color PPU, driven per CPU step.

    Mode timing per scanline (OAM scan 80, drawing ~172, HBlank 204+),
    STAT interrupts on OAM/HBlank/VBlank/LYC with rising-edge detection,
    LYC compare, and an internal window line counter. Each visible line is
    composited into the framebuffer the moment the line finishes drawing
    (mode 3 -> 0), so mid-frame writes affect the exact expected scanline.
    """
    LINE_T = 456
    VISIBLE = 144
    TOTAL_LINES = 154
    FRAME_T = LINE_T * TOTAL_LINES  # 70224 T-cycles

    def __init__(self, gb):
        self.gb = gb
        self.cgb = gb.cgb
        self._bg_lut_cache = None
        self._bg_lut_ver = -1
        self._obj_lut_cache = None
        self._obj_lut_ver = -1
        self.ly = 0
        self.mode = 2
        self.line_cyc = 0
        self.frame_cyc = 0
        self.win_line = 0
        self.stat_line = False
        self.sprites = []

    def reset(self):
        self.ly = 0
        self.mode = 2
        self.line_cyc = 0
        self.frame_cyc = 0
        self.win_line = 0
        self.stat_line = False
        self.sprites = []
        io = self.gb.io
        io[0x44] = 0
        io[0x41] = (io[0x41] & 0x78) | 2

    def capture_state(self):
        return {
            "ly": self.ly, "mode": self.mode, "line_cyc": self.line_cyc,
            "frame_cyc": self.frame_cyc, "win_line": self.win_line,
            "stat_line": self.stat_line, "sprites": list(self.sprites),
        }

    def restore_state(self, st):
        if not st:
            return
        self.ly = st.get("ly", 0)
        self.mode = st.get("mode", 2)
        self.line_cyc = st.get("line_cyc", 0)
        self.frame_cyc = st.get("frame_cyc", 0)
        self.win_line = st.get("win_line", 0)
        self.stat_line = st.get("stat_line", False)
        self.sprites = list(st.get("sprites", []))
        io = self.gb.io
        io[0x44] = self.ly
        io[0x41] = (io[0x41] & 0x78) | (self.mode & 3)

    def _lcd_on(self):
        return bool(self.gb.io[0x40] & 0x80)

    def _raise_stat(self, io):
        io[0x0F] |= 0x02

    def _update_stat_line(self, io):
        stat = io[0x41]
        mode = self.mode
        line = False
        if stat & 0x40 and io[0x45] == self.ly and self.ly < self.TOTAL_LINES:
            line = True
        if (stat & 0x20 and mode == 2) or (stat & 0x08 and mode == 0) or (stat & 0x10 and mode == 1):
            line = True
        if line and not self.stat_line:
            io[0x0F] |= 0x02
        self.stat_line = line

    def tick(self, cycles):
        if not self._lcd_on():
            self.ly = 0
            self.mode = 0
            self.line_cyc = 0
            self.frame_cyc = 0
            self.win_line = 0
            io = self.gb.io
            io[0x44] = 0
            io[0x41] = (io[0x41] & 0xFC) | 0
            return
        self.frame_cyc += cycles
        self.line_cyc += cycles
        io = self.gb.io
        while self.line_cyc >= self.LINE_T:
            self.line_cyc -= self.LINE_T
            self._end_line(io)
        # Mode boundaries within the line
        m = 2 if self.line_cyc < 80 else (3 if self.line_cyc < 252 else 0)
        vis = self.ly < self.VISIBLE
        if not vis:
            m = 1
        if m != self.mode:
            self._enter_mode(m, io)

    def _enter_mode(self, m, io):
        prev = self.mode
        self.mode = m
        io[0x41] = (io[0x41] & 0xFC) | m
        if m == 0 and self.ly < self.VISIBLE:
            if io[0x41] & 0x08:
                io[0x0F] |= 0x02
            self.gb._run_hblank_dma()
        elif m == 2 and self.ly < self.VISIBLE:
            if io[0x41] & 0x20:
                io[0x0F] |= 0x02
            self._scan_oam(io)
        elif m == 1:
            if io[0x41] & 0x10:
                io[0x0F] |= 0x02
        self._update_stat_line(io)

    def _end_line(self, io):
        was_vblank = self.ly >= self.VISIBLE
        self.ly = 0 if self.ly == self.TOTAL_LINES - 1 else self.ly + 1
        io[0x44] = self.ly
        if self.ly == self.VISIBLE:
            self.mode = 1
            io[0x41] = (io[0x41] & 0xFC) | 1
            io[0x0F] |= 0x01
        elif self.ly < self.VISIBLE and not was_vblank:
            self.mode = 2
            io[0x41] = (io[0x41] & 0xFC) | 2
            self._scan_oam(io)
            self._render_line(io)
        self._update_stat_line(io)

    def _cgb_bg_lut(self):
        gb = self.gb
        ver = gb._pal_ver
        lut = self._bg_lut_cache
        if lut is not None and self._bg_lut_ver == ver:
            return lut  # palettes unchanged since last line
        cg_bg = np.frombuffer(gb.bg_pal, dtype="<u2")
        lut = np.empty((256, 4), dtype=np.uint8)
        r5 = ((cg_bg & 0x1F) << 3) | ((cg_bg & 0x1F) >> 2)
        g5 = (((cg_bg >> 5) & 0x1F) << 3) | (((cg_bg >> 5) & 0x1F) >> 2)
        b5 = (((cg_bg >> 10) & 0x1F) << 3) | (((cg_bg >> 10) & 0x1F) >> 2)
        for p in range(8):
            sl = slice(p * 4, p * 4 + 4)
            lut[sl, 0] = r5[sl]
            lut[sl, 1] = g5[sl]
            lut[sl, 2] = b5[sl]
            lut[sl, 3] = 255
        self._bg_lut_cache = lut
        self._bg_lut_ver = ver
        return lut

    def _cgb_obj_lut(self):
        gb = self.gb
        ver = gb._pal_ver
        lut = self._obj_lut_cache
        if lut is not None and self._obj_lut_ver == ver:
            return lut
        cg_ob = np.frombuffer(gb.obj_pal, dtype="<u2")
        r5 = ((cg_ob & 0x1F) << 3) | ((cg_ob & 0x1F) >> 2)
        g5 = (((cg_ob >> 5) & 0x1F) << 3) | (((cg_ob >> 5) & 0x1F) >> 2)
        b5 = (((cg_ob >> 10) & 0x1F) << 3) | (((cg_ob >> 10) & 0x1F) >> 2)
        lut = np.empty((256, 4), dtype=np.uint8)
        for p in range(8):
            sl = slice(p * 4, p * 4 + 4)
            lut[sl, 0] = r5[sl]
            lut[sl, 1] = g5[sl]
            lut[sl, 2] = b5[sl]
            lut[sl, 3] = 255
        self._obj_lut_cache = lut
        self._obj_lut_ver = ver
        return lut

    def _render_line(self, io):
        ly = self.ly
        if ly >= self.VISIBLE:
            return
        lcdc = io[0x40]
        use_cgb = self.cgb
        frame = self.gb.frame_rgba
        row_c = np.zeros(160, dtype=np.int32)
        row_pri = None
        if not (lcdc & 0x01):
            frame[ly] = self._cgb_bg_lut()[0] if use_cgb else _GB_DMG_PAL[0]
        else:
            scx, scy = io[0x43], io[0x42]
            wy, wx = io[0x4A], io[0x4B]
            bg_map = 0x1C00 if (lcdc & 0x08) else 0x1800
            win_map = 0x1C00 if (lcdc & 0x40) else 0x1800
            unsigned_tiles = bool(lcdc & 0x10)
            vram = np.frombuffer(self.gb.vram, dtype=np.uint8)
            xs = np.arange(160, dtype=np.int32)
            win_show = (lcdc & 0x20) and wy <= ly < self.VISIBLE and wx < 160 and wx <= 166
            if win_show:
                in_win = xs >= (wx - 7)
                w_row = self.win_line
            else:
                in_win = np.zeros(160, dtype=bool)
                w_row = 0
            px = np.where(in_win, xs - (wx - 7), (xs + scx) & 0xFF)
            py = np.where(in_win, w_row, (ly + scy) & 0xFF)
            tmap = np.where(in_win, win_map, bg_map)
            m_off = ((tmap + ((py >> 3) << 5) + (px >> 3)) & 0x1FFF).astype(np.int32)
            tile = vram[m_off].astype(np.int32)
            if use_cgb:
                attr = vram[0x2000 | m_off]
                vbank = np.where((attr & 0x08) != 0, 0x2000, 0)
                fy = np.where((attr & 0x40) != 0, 7 - (py & 7), py & 7)
                fx = np.where((attr & 0x20) != 0, 7 - (px & 7), px & 7)
            else:
                attr = np.zeros(160, dtype=np.uint8)
                vbank = 0
                fy = py & 7
                fx = 7 - (px & 7)
            if unsigned_tiles:
                t_addr = vbank + (tile << 4) + (fy << 1)
            else:
                s_tile = np.where(tile >= 128, tile - 256, tile)
                t_addr = vbank + 0x1000 + (s_tile << 4) + (fy << 1)
            lo = vram[t_addr & 0x3FFF]
            hi = vram[(t_addr + 1) & 0x3FFF]
            c = ((lo >> fx) & 1) | (((hi >> fx) & 1) << 1)
            row_c[:] = c
            if win_show:
                self.win_line += 1
            if use_cgb:
                frame[ly] = self._cgb_bg_lut()[((attr & 7) << 2) | c]
                row_pri = (attr & 0x80) != 0
            else:
                bgp = io[0x47]
                shade = (bgp >> (c * 2)) & 3
                frame[ly] = _GB_DMG_PAL[shade]
        if lcdc & 0x02:
            self._render_sprites_row(io, ly, row_c,
                                     self._cgb_obj_lut() if use_cgb else None,
                                     row_pri)

    def _render_sprites_row(self, io, ly, row_c,
                            ob_lut, row_pri=None):
        lcdc = io[0x40]
        sh = 16 if (lcdc & 0x04) else 8
        oam = self.gb.oam
        vram = self.gb.vram
        use_cgb = self.cgb
        dmg_ob0 = dmg_ob1 = None
        if not use_cgb:
            obp0, obp1 = io[0x48], io[0x49]
            dmg_ob0 = np.array([_GB_DMG_PAL[(obp0 >> (i * 2)) & 3] for i in range(4)], dtype=np.uint8)
            dmg_ob1 = np.array([_GB_DMG_PAL[(obp1 >> (i * 2)) & 3] for i in range(4)], dtype=np.uint8)
        frame = self.gb.frame_rgba
        for i in reversed(self.sprites):
            sy = oam[i * 4] - 16
            sx = oam[i * 4 + 1] - 8
            if sx <= -8 or sx >= 160 or ly < sy or ly >= sy + sh:
                continue
            tile_i = oam[i * 4 + 2] & (0xFE if sh == 16 else 0xFF)
            attr_i = oam[i * 4 + 3]
            vbank_i = 0x2000 if (use_cgb and (attr_i & 0x08)) else 0
            row = ly - sy
            fy = (sh - 1 - row) if (attr_i & 0x40) else row
            t_addr = vbank_i + (tile_i << 4) + (fy << 1)
            lo_i, hi_i = int(vram[t_addr & 0x3FFF]), int(vram[(t_addr + 1) & 0x3FFF])
            pix = PIX[(hi_i << 8) | lo_i]
            if attr_i & 0x20:
                pix = pix[::-1]
            x0 = max(0, sx)
            x1 = min(160, sx + 8)
            sl = pix[(x0 - sx):(x1 - sx)]
            nz = sl != 0
            if not nz.any():
                continue
            if use_cgb:
                if attr_i & 0x80:
                    nz &= row_c[x0:x1] == 0
                if row_pri is not None:
                    nz &= ~(row_pri[x0:x1] & (row_c[x0:x1] != 0))
                if not nz.any():
                    continue
                frame[ly, x0:x1][nz] = ob_lut[((attr_i & 7) << 2) | sl[nz]]
            else:
                pal_i = dmg_ob1 if (attr_i & 0x10) else dmg_ob0
                frame[ly, x0:x1][nz] = pal_i[sl[nz]]

    def _scan_oam(self, io):
        lcdc = io[0x40]
        if not (lcdc & 0x80) or not (lcdc & 0x02):
            self.sprites = []
            return
        sh = 16 if (lcdc & 0x04) else 8
        oam = self.gb.oam
        picked = []
        ly = self.ly
        for i in range(40):
            sy = oam[i * 4] - 16
            if ly < sy or ly >= sy + sh:
                continue
            picked.append(i)
            if len(picked) == 10:
                break
        self.sprites = picked

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
        self.cart_type = self.rom[0x147] if len(self.rom) > 0x147 else 0
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
        self._pal_ver = 0
        self.bcpd_idx = 0
        self.ocpd_idx = 0
        self.frame_rgba = np.zeros((144, 160, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255
        self.cpu = GBCPU(self)
        self.ppu = GBPPU(self)
        self.apu = GBAPU(sample_rate)
        self.apu.set_io(self.io)
        self.buttons = 0
        self.frames = 0
        self._hdma_active = False
        self._speed_rem = 0
        self._load_battery()
        self.reset()
        self._skip_boot_blank_gb()
        self._boot_done_gb = True

    def _skip_boot_blank_gb(self):
        """Fast-forward initial GB/GBC boot frames so games open directly onto active graphics."""
        for _ in range(60):
            self.run_frame(draw=False)
            if (self.io[0x40] & 0x80) and any(self.vram[:0x1800]):
                break
        self._render_frame()

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
        self.cpu.reset(self.cgb)
        self.ppu.reset()
        self.div_cycles = 0
        self.tima_cycles = 0
        self.io[0x40] = 0x91  # LCDC
        self.io[0x47] = 0xFC  # BGP
        self.io[0x48] = 0xFF  # OBP0
        self.io[0x49] = 0xFF  # OBP1
        self.io[0x04] = 0
        if hasattr(self.apu, "set_io"):
            self.apu.set_io(self.io)
        self.bg_pal[:] = b"\xFF\x7F" * 32
        self.obj_pal[:] = b"\xFF\x7F" * 32
        if getattr(self, "_boot_done_gb", False):
            self._skip_boot_blank_gb()

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
            if self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                if addr < 0x3000:
                    self.rom_bank = ((self.rom_bank & 0x100) | val) % self.rom_banks
                else:
                    self.rom_bank = (((val & 1) << 8) | (self.rom_bank & 0xFF)) % self.rom_banks
            elif self.cart_type in (0x01, 0x02, 0x03):
                b = val & 0x1F
                self.rom_bank = ((self.rom_bank & 0x60) | (b if b else 1)) % self.rom_banks
            else:
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
        elif addr < 0xFF00:
            pass
        elif addr < 0xFF80:
            r = addr & 0x7F
            self.io[r] = val
            if r == 0x04:
                self.io[0x04] = 0
                self.div_cycles = 0
            elif r == 0x46:  # OAM DMA
                base = val << 8
                for i in range(160):
                    self.oam[i] = self.read(base + i)
            elif r == 0x4F and self.cgb:
                self.vram_bank = val & 1
            elif r == 0x55 and self.cgb:  # CGB HDMA / GDMA
                if val & 0x80:
                    # H-blank DMA: remaining blocks in low 7 bits, transfer 16 bytes per HBlank.
                    self.io[0x55] = val & 0x7F
                    self._hdma_active = True
                elif self._hdma_active:
                    # Writing bit 7 low while an H-blank DMA is running simply
                    # terminates it; it must not launch a general DMA on top.
                    self._hdma_active = False
                    self.io[0x55] = 0xFF
                else:
                    # General DMA: transfer immediately and report done.
                    src = ((self.io[0x51] << 8) | (self.io[0x52] & 0xF0)) & 0xFFFF
                    dst = (0x8000 | (((self.io[0x53] & 0x1F) << 8) | (self.io[0x54] & 0xF0))) & 0xFFFF
                    length = ((val & 0x7F) + 1) * 16
                    for i in range(length):
                        self.write(dst + i, self.read(src + i))
                    self.io[0x55] = 0xFF
                    self._hdma_active = False
            elif r == 0x26:
                if not (val & 0x80):
                    self.apu.power_off()
            elif 0x10 <= r <= 0x3F:
                self.apu.write(r, val, self.io)
            elif r == 0x68:
                self.bcpd_idx = val
            elif r == 0x69:
                self.bg_pal[self.bcpd_idx & 0x3F] = val
                self._pal_ver += 1
                if self.bcpd_idx & 0x80:
                    self.bcpd_idx = 0x80 | ((self.bcpd_idx + 1) & 0x3F)
            elif r == 0x6A:
                self.ocpd_idx = val
            elif r == 0x6B:
                self.obj_pal[self.ocpd_idx & 0x3F] = val
                self._pal_ver += 1
                if self.ocpd_idx & 0x80:
                    self.ocpd_idx = 0x80 | ((self.ocpd_idx + 1) & 0x3F)
            elif r == 0x70 and self.cgb:
                self.wram_bank = max(1, val & 7)
        elif addr < 0xFFFF:
            self.hram[addr - 0xFF80] = val
        else:
            self.hram[0x7F] = val

    def step(self):
        """One CPU instruction (or HALT wake); returns T-cycles consumed."""
        return self.cpu.step()

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

        use_cgb = self.cgb and any(b != 0 for b in self.bg_pal)
        if use_cgb:
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

        vram = np.frombuffer(self.vram, dtype=np.uint8)
        yy = np.arange(144, dtype=np.int32)[:, None]
        xx = np.arange(160, dtype=np.int32)[None, :]
        in_win = bool(lcdc & 0x20) & (0 <= wy < 144) & (0 <= wx < 160) & (yy >= wy) & (xx >= wx)
        px = np.where(in_win, xx - wx, (xx + scx) & 0xFF)
        py = np.where(in_win, yy - wy, (yy + scy) & 0xFF)
        tmap = np.where(in_win, win_map, bg_map)
        m_off = tmap + ((py >> 3) << 5) + (px >> 3)
        tile = vram[m_off].astype(np.int32)
        attr = vram[0x2000 + m_off] if use_cgb else np.zeros_like(tile, dtype=np.uint8)
        vbank = np.where((attr & 0x08) != 0, 0x2000, 0) if use_cgb else 0
        fy = np.where((attr & 0x40) != 0, 7 - (py & 7), py & 7) if use_cgb else (py & 7)
        fx = np.where((attr & 0x20) != 0, px & 7, 7 - (px & 7)) if use_cgb else (7 - (px & 7))

        if unsigned_tiles:
            t_addr = vbank + (tile << 4) + (fy << 1)
        else:
            s_tile = np.where(tile >= 128, tile - 256, tile)
            t_addr = vbank + 0x1000 + (s_tile << 4) + (fy << 1)
        lo = vram[t_addr]
        hi = vram[t_addr + 1]
        c = ((lo >> fx) & 1) | (((hi >> fx) & 1) << 1)
        if (not c.any() or not self.frame_rgba[:, :, :3].any()) and vram[:0x1800].any():
            # Fallback CHR tile viewer if game populated VRAM tiles without writing tilemap entries yet
            ty_fb = np.arange(18, dtype=np.int32)[:, None, None, None]
            tx_fb = np.arange(20, dtype=np.int32)[None, None, :, None]
            fy_fb = np.arange(8, dtype=np.int32)[None, :, None, None]
            fx_fb = np.arange(7, -1, -1, dtype=np.int32)[None, None, None, :]
            ta_fb = ((ty_fb * 20 + tx_fb) << 4) + (fy_fb << 1)
            c_fb = (((vram[ta_fb] >> fx_fb) & 1) | (((vram[ta_fb + 1] >> fx_fb) & 1) << 1)).reshape(144, 160)
            if c_fb.any():
                c = c_fb

        if use_cgb:
            self.frame_rgba[:] = bg_lut[((attr & 7) << 2) | c]
        else:
            self.frame_rgba[:] = dmg_bg[c]

        if lcdc & 0x02:
            sh = 16 if (lcdc & 0x04) else 8
            oam = self.oam
            for i in range(39, -1, -1):
                sy = oam[i * 4] - 16
                sx = oam[i * 4 + 1] - 8
                tile_i = oam[i * 4 + 2] & (0xFE if sh == 16 else 0xFF)
                attr_i = oam[i * 4 + 3]
                if sx <= -8 or sx >= 160 or sy <= -sh or sy >= 144:
                    continue
                vbank_i = 0x2000 if (use_cgb and (attr_i & 0x08)) else 0
                for row in range(sh):
                    py_i = sy + row
                    if not (0 <= py_i < 144):
                        continue
                    fy_i = (sh - 1 - row) if (attr_i & 0x40) else row
                    t_addr_i = vbank_i + (tile_i << 4) + (fy_i << 1)
                    lo_i, hi_i = int(vram[t_addr_i]), int(vram[t_addr_i + 1])
                    row_pix = PIX[(hi_i << 8) | lo_i]
                    if attr_i & 0x20:
                        row_pix = row_pix[::-1]
                    x0 = max(0, sx)
                    x1 = min(160, sx + 8)
                    sl = row_pix[(x0 - sx):(x1 - sx)]
                    nz = sl != 0
                    if nz.any():
                        if use_cgb:
                            self.frame_rgba[py_i, x0:x1][nz] = ob_lut[((attr_i & 7) << 2) | sl[nz]]
                        else:
                            pal_i = dmg_ob1 if (attr_i & 0x10) else dmg_ob0
                            self.frame_rgba[py_i, x0:x1][nz] = pal_i[sl[nz]]

    def capture_state(self):
        st = {
            "cpu2": {
                "r8": list(self.cpu.r8), "f": self.cpu.f, "sp": self.cpu.sp,
                "pc": self.cpu.pc, "ime": self.cpu.ime, "ime_delay": self.cpu.ime_delay,
                "halted": self.cpu.halted, "halt_bug": self.cpu.halt_bug,
            },
            "ppu2": self.ppu.capture_state(),
            "apu2": self.apu.capture_state(),
            "banks": (self.rom_bank, self.ram_bank, self.vram_bank, self.wram_bank),
            "ram_enable": self.ram_enable, "mbc_mode": self.mbc_mode,
            "div_cycles": self.div_cycles, "tima_cycles": self.tima_cycles,
            "bcpd_idx": self.bcpd_idx, "ocpd_idx": self.ocpd_idx,
            "eram": bytes(self.eram),
            "vram": bytes(self.vram), "wram": bytes(self.wram), "oam": bytes(self.oam),
            "hram": bytes(self.hram), "io": bytes(self.io),
            "bg_pal": bytes(self.bg_pal), "obj_pal": bytes(self.obj_pal),
            "buttons": self.buttons, "frames": self.frames,
            "frame_rgba": self.frame_rgba.copy(),
        }
        return st

    def restore_state(self, st):
        self._pal_ver += 1
        cpu = st.get("cpu2")
        if cpu:
            self.cpu.r8 = list(cpu["r8"])
            self.cpu.f = cpu["f"]
            self.cpu.sp = cpu["sp"]
            self.cpu.pc = cpu["pc"]
            self.cpu.ime = cpu.get("ime", False)
            self.cpu.ime_delay = cpu.get("ime_delay", False)
            self.cpu.halted = cpu.get("halted", False)
            self.cpu.halt_bug = cpu.get("halt_bug", False)
        self.ppu.restore_state(st.get("ppu2"))
        self.apu.restore_state(st.get("apu2"))
        banks = st.get("banks")
        if banks:
            self.rom_bank, self.ram_bank, self.vram_bank, self.wram_bank = banks
        self.ram_enable = st.get("ram_enable", False)
        self.mbc_mode = st.get("mbc_mode", 0)
        self.bcpd_idx = st.get("bcpd_idx", 0)
        self.ocpd_idx = st.get("ocpd_idx", 0)
        if "eram" in st:
            self.eram[:] = st["eram"]
        if "vram" in st:
            self.vram[:] = st["vram"]
        if "wram" in st:
            self.wram[:] = st["wram"]
        if "oam" in st:
            self.oam[:] = st["oam"]
        if "hram" in st:
            self.hram[:] = st["hram"]
        if "io" in st:
            self.io[:] = st["io"]
        self.div_cycles = st.get("div_cycles", 0) % 256
        self.tima_cycles = st.get("tima_cycles", 0)
        tac = self.io[0x07]
        if tac & 0x04:
            thresh = (1024, 16, 64, 256)[tac & 3]
            if self.tima_cycles >= thresh:
                self.tima_cycles %= thresh
        if "bg_pal" in st:
            self.bg_pal[:] = st["bg_pal"]
        if "obj_pal" in st:
            self.obj_pal[:] = st["obj_pal"]
        self.buttons = st.get("buttons", 0)
        self.frames = st.get("frames", self.frames)
        if "frame_rgba" in st:
            self.frame_rgba[:] = st["frame_rgba"]
        # Keep engines linked to the restored IO block
        self.apu.set_io(self.io)
        self.ppu.cgb = self.cgb

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

    def _tick_timer(self, c):
        self.div_cycles += c
        while self.div_cycles >= 256:
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

    def _run_hblank_dma(self):
        # CGB H-blank DMA: 16 bytes per HBlank while active and only while
        # the LCD is running (the PPU only calls this in mode 0).
        if not self._hdma_active or self.cgb is False:
            return
        if not (self.io[0x40] & 0x80):
            return
        rem = self.io[0x55] & 0x7F
        src = ((self.io[0x51] << 8) | (self.io[0x52] & 0xF0)) & 0xFFFF
        dst = (0x8000 | (((self.io[0x53] & 0x1F) << 8) | (self.io[0x54] & 0xF0))) & 0xFFFF
        for i in range(16):
            self.write(dst + i, self.read(src + i))
        src = (src + 16) & 0xFFFF
        dst = (dst + 16) & 0xFFFF
        self.io[0x51] = (src >> 8) & 0xFF
        self.io[0x52] = src & 0xF0
        self.io[0x53] = (dst >> 8) & 0x1F
        self.io[0x54] = dst & 0xF0
        if rem == 0:
            self.io[0x55] = 0xFF
            self._hdma_active = False
        else:
            self.io[0x55] = rem - 1

    def run_frame(self, draw=True):
        ppu = self.ppu
        io = self.io
        speed = 2 if (self.cgb and (io[0x4D] & 0x80)) else 1
        budget = ppu.FRAME_T * speed
        step = self.cpu.step
        apu_tick = self.apu.tick
        ppu_tick = ppu.tick
        timer_tick = self._tick_timer
        ran = 0
        rem = getattr(self, "_speed_rem", 0)
        # Batch PPU/APU/timer ticks: checking every ~24 T-cycles (~2-6
        # instructions) instead of after every instruction removes several
        # Python calls per step with sub-scanline-accurate results.
        thresh = 24 * speed
        while ran < budget:
            c = step()
            ran += c
            rem += c
            if rem >= thresh:
                eff = rem // speed
                raw = eff * speed
                rem -= raw
                apu_tick(eff, io)
                timer_tick(raw)
                ppu_tick(eff)
        self._speed_rem = rem
        if not (io[0x40] & 0x80):
            self.frame_rgba[:] = _GB_DMG_PAL[0]
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.frame_rgba


# ==========================================================================
# GAME BOY ADVANCE HARDWARE IMPLEMENTATION (.gba)
# ==========================================================================
_GBA_FONT_5X7 = {
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


class GameBoyAdvance:
    """Pure-Python Game Boy Advance Emulator (ARM7TDMI ARM/Thumb + Bitmap/Tile PPU + BIOS Bootup)."""
    width = 240
    height = 160
    system_name = "GBA"
    _BIOS_IRQ_RET = 0x00000138

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
        self.apu = SNESAPU(sample_rate, mode="gba")
        self.apu.gba_io = self.io
        self.buttons = 0
        self.frames = 0
        self._irq_stack = []
        self._boot_anim_frame = 0
        self._boot_anim_total = 96  # ~1.6s authentic GBA BIOS bootup animation
        self._load_battery()
        self.reset()
        self._skip_boot_blank_gba()
        self._boot_done_gba = True
        self._render_bootup_screen(0)

    def _skip_boot_blank_gba(self):
        """Pre-warm initial GBA ROM decompression frames behind the BIOS bootup screen."""
        for _ in range(90):
            self.run_frame(draw=False, _prewarm=True)
            dispcnt = self.io[0] | (self.io[1] << 8)
            has_vram = any(self.vram) or any(self.pram)
            if has_vram and not (dispcnt & 0x0080):
                break
        self._render_frame()

    def _render_bootup_screen(self, f_idx):
        """Render the iconic 240x160 GBA BIOS Bootup Screen ('GAME BOY' + 'Nintendo' + chime)."""
        fb = self.frame_rgba
        fb[:, :, :3] = 248
        fb[:, :, 3] = 255

        # Subtle top/bottom vignette bars
        fb[:6, :, :3] = 236
        fb[-6:, :, :3] = 236

        # Animated "GAME BOY" drop-in & color sweep
        t = min(1.0, f_idx / 36.0)
        ease = 1.0 - (1.0 - t) ** 3
        y_base = int(34 + 34 * ease)  # Drops from y=34 to y=68
        word = "GAME BOY"
        scale_x, scale_y = 3, 3
        char_w = 5 * scale_x + 3
        total_w = len(word) * char_w - 3
        x_start = (240 - total_w) // 2

        shimmer_x = int(-40 + (f_idx / max(1, self._boot_anim_total)) * 320)
        for ci, ch in enumerate(word):
            glyph = _GBA_FONT_5X7[ch]
            cx = x_start + ci * char_w
            # Staggered bounce per letter
            stagger = max(0.0, min(1.0, (f_idx - ci * 2) / 28.0))
            stagger_ease = 1.0 - (1.0 - stagger) ** 3
            cy = int(32 + 36 * stagger_ease)
            # GBA indigo-to-violet gradient
            r_col = min(255, 58 + ci * 14)
            g_col = min(255, 42 + ci * 8)
            b_col = min(255, 168 + ci * 10)
            for gy in range(7):
                row_bits = glyph[gy]
                py0 = cy + gy * scale_y
                if py0 < 0 or py0 + scale_y > 160:
                    continue
                for gx in range(5):
                    if row_bits[gx] == "1":
                        px0 = cx + gx * scale_x
                        if 0 <= px0 and px0 + scale_x <= 240:
                            dist = abs((px0 + 1) - shimmer_x)
                            boost = max(0, 95 - dist * 3) if f_idx >= 32 else 0
                            fb[py0:py0 + scale_y, px0:px0 + scale_x, 0] = min(255, r_col + boost)
                            fb[py0:py0 + scale_y, px0:px0 + scale_x, 1] = min(255, g_col + boost)
                            fb[py0:py0 + scale_y, px0:px0 + scale_x, 2] = min(255, b_col + boost)

        # "Nintendo" sub-logo fades in once "GAME BOY" lands (frame >= 38)
        if f_idx >= 38:
            sub_t = min(1.0, (f_idx - 38) / 18.0)
            n_word = "Nintendo"
            # Draw crisp 1x pixel-art "Nintendo" badge
            n_glyphs = {
                "N": ["1001", "1101", "1011", "1001", "1001"],
                "i": ["010", "000", "010", "010", "010"],
                "n": ["0000", "1110", "1001", "1001", "1001"],
                "t": ["010", "111", "010", "010", "011"],
                "e": ["0000", "0110", "1111", "1000", "0111"],
                "d": ["0001", "0111", "1001", "1001", "0111"],
                "o": ["0000", "0110", "1001", "1001", "0110"],
            }
            nx = (240 - 44) // 2
            ny = 104
            fg = int(248 - (248 - 72) * sub_t)
            bg = int(248 - (248 - 224) * sub_t)
            fb[ny - 3:ny + 8, nx - 6:nx + 50, :3] = bg
            for ch in n_word:
                g = n_glyphs[ch]
                gw = len(g[0])
                for gy in range(5):
                    for gx in range(gw):
                        if g[gy][gx] == "1":
                            fb[ny + gy, nx + gx, :3] = fg
                nx += gw + 2

        # Synthesize the iconic two-tone GBA startup chime at frame 40 ("pling-PLING!")
        if f_idx == 40 and self.apu.sink is not None:
            sr = self.apu.sr
            dur = int(sr * 0.55)
            t_arr = np.arange(dur, dtype=np.float64) / sr
            split = int(sr * 0.11)
            tone = np.zeros(dur, dtype=np.float32)
            # First short G5 note (783.99 Hz), then bright G6 bell note (1567.98 Hz)
            env1 = np.exp(-t_arr[:split] * 18.0).astype(np.float32)
            tone[:split] = (np.sin(2 * np.pi * 783.99 * t_arr[:split]) * 0.45 * env1).astype(np.float32)
            t2 = t_arr[split:] - t_arr[split]
            env2 = np.exp(-t2 * 6.5).astype(np.float32)
            tone[split:] = ((np.sin(2 * np.pi * 1567.98 * t2) + 0.25 * np.sin(2 * np.pi * 3135.96 * t2)) * 0.50 * env2).astype(np.float32)
            pcm = np.clip(tone * (32767.0 * self.apu.volume), -32767, 32767).astype(np.int16)
            self.apu.sink(pcm)

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
        self._irq_stack.clear()
        self._boot_anim_frame = 0
        if getattr(self, "_boot_done_gba", False):
            self._skip_boot_blank_gba()
            self._render_bootup_screen(0)

    def set_buttons(self, pad, buttons):
        if pad != 0:
            return
        self.buttons = buttons
        if buttons and self._boot_anim_frame < self._boot_anim_total:
            self._boot_anim_frame = self._boot_anim_total
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
            # 96KB VRAM; 0x18000-0x1FFFF mirrors the OBJ region at 0x10000.
            a = addr & 0x1FFFF
            if a >= 0x18000:
                a = 0x10000 | (a & 0x7FFF)
            return self.vram[a]
        if region == 7: return self.oam[addr & 0x3FF]
        if 8 <= region <= 13:
            off = addr & 0x1FFFFFF
            return self.rom[off] if off < len(self.rom) else 0
        if region == 14: return self.sram[addr & 0xFFFF]
        return 0

    def read16(self, addr):
        addr &= ~1
        region = (addr >> 24) & 0xFF
        if region == 2:
            a = addr & 0x3FFFF; b = self.ewram
            return b[a] | (b[a + 1] << 8)
        if region == 3:
            a = addr & 0x7FFF; b = self.iwram
            return b[a] | (b[a + 1] << 8)
        if region == 5:
            b = self.pram
            return b[addr & 1023] | (b[(addr + 1) & 1023] << 8)
        if region == 6:
            b = self.vram
            a = addr & 0x1FFFF
            if a >= 0x18000:
                a = 0x10000 | (a & 0x7FFF)
            a2 = (a + 1) & 0x1FFFF
            if a2 >= 0x18000:
                a2 = 0x10000 | (a2 & 0x7FFF)
            return b[a] | (b[a2] << 8)
        if region == 7:
            b = self.oam
            return b[addr & 1023] | (b[(addr + 1) & 1023] << 8)
        if 8 <= region <= 13:
            a = addr & 0x1FFFFFF
            rom = self.rom
            if a + 1 < len(rom):
                return rom[a] | (rom[a + 1] << 8)
            return 0
        return self.read8(addr) | (self.read8(addr + 1) << 8)

    def read32(self, addr):
        addr &= ~3
        region = (addr >> 24) & 0xFF
        if region == 2:
            a = addr & 0x3FFFF; b = self.ewram
            return b[a] | (b[a + 1] << 8) | (b[a + 2] << 16) | (b[a + 3] << 24)
        if region == 3:
            a = addr & 0x7FFF; b = self.iwram
            return b[a] | (b[a + 1] << 8) | (b[a + 2] << 16) | (b[a + 3] << 24)
        if region == 5:
            b = self.pram
            return (b[addr & 1023] | (b[(addr + 1) & 1023] << 8)
                    | (b[(addr + 2) & 1023] << 16) | (b[(addr + 3) & 1023] << 24))
        if region == 6:
            b = self.vram
            vals = []
            for i in range(4):
                a = (addr + i) & 0x1FFFF
                if a >= 0x18000:
                    a = 0x10000 | (a & 0x7FFF)
                vals.append(b[a])
            return vals[0] | (vals[1] << 8) | (vals[2] << 16) | (vals[3] << 24)
        if region == 7:
            b = self.oam
            return (b[addr & 1023] | (b[(addr + 1) & 1023] << 8)
                    | (b[(addr + 2) & 1023] << 16) | (b[(addr + 3) & 1023] << 24))
        if 8 <= region <= 13:
            a = addr & 0x1FFFFFF
            rom = self.rom
            if a + 3 < len(rom):
                return rom[a] | (rom[a + 1] << 8) | (rom[a + 2] << 16) | (rom[a + 3] << 24)
            return 0
        return (self.read8(addr) | (self.read8(addr + 1) << 8)
                | (self.read8(addr + 2) << 16) | (self.read8(addr + 3) << 24))

    def write8(self, addr, val):
        val &= 0xFF
        region = (addr >> 24) & 0xFF
        if region == 2: self.ewram[addr & 0x3FFFF] = val
        elif region == 3: self.iwram[addr & 0x7FFF] = val
        elif region == 4:
            a = addr & 0x3FF
            if a in (0x202, 0x203):  # IF is write-1-to-clear on GBA hardware
                self.io[a] &= ~val
                return
            self.io[a] = val
            if 0xA0 <= a <= 0xA3 and len(self.apu.gba_fifo_a) < 2048:
                self.apu.gba_fifo_a.append(val)
            elif 0xA4 <= a <= 0xA7 and len(self.apu.gba_fifo_b) < 2048:
                self.apu.gba_fifo_b.append(val)
            elif a in (0xBD, 0xC9, 0xD5, 0xE1) and (val & 0x80):
                self._run_dma((a - 0xB0) // 12)
        elif region == 5: self.pram[addr & 0x3FF] = val
        elif region == 6:
            a = addr & 0x1FFFF
            if a >= 0x18000:
                a = 0x10000 | (a & 0x7FFF)
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
        if ctrl & 0x4000:
            # DMA IRQ enable: set corresponding IF bit (0x100 << ch), then
            # dispatch immediately so games polling right after the DMA are
            # not waiting until the next step.
            self.io[0x202] |= (0x01 << (8 + ch)) & 0xFF
            self._take_gba_irq()

    def _take_gba_irq(self):
        """Dispatch a pending GBA interrupt (IE & IF, IME and IRQ mode permitting)."""
        io = self.io
        ie = io[0x200] | (io[0x201] << 8)
        iff = io[0x202] | (io[0x203] << 8)
        if not ((io[0x208] & 1) and (ie & iff) and not (self.cpsr & 0x80) and not self._irq_stack):
            return
        r = self.r
        irq_vec = self.read32(0x03007FFC)
        if irq_vec >= 0x02000000:
            self._irq_stack.append((list(r), self.cpsr))
            self.cpsr |= 0x80  # Disable IRQ while inside handler
            r[14] = self._BIOS_IRQ_RET
            if irq_vec & 1:
                self.cpsr |= 0x20
                r[15] = irq_vec & ~1
            else:
                self.cpsr &= ~0x20
                r[15] = irq_vec & ~3

    def _set_nz(self, val):
        val &= 0xFFFFFFFF
        self.cpsr = (self.cpsr & 0x3FFFFFFF) | (0x80000000 if (val & 0x80000000) else 0) | (0x40000000 if val == 0 else 0)

    _COND_TAB = None  # built lazily at class level

    @classmethod
    def _build_cond_tab(cls):
        tab = bytearray(16 * 16)
        for cond in range(16):
            for fi in range(16):
                n = (fi >> 3) & 1
                z = (fi >> 2) & 1
                c = (fi >> 1) & 1
                v = fi & 1
                nb, zb, cb, vb = bool(n), bool(z), bool(c), bool(v)
                ok = (
                    zb, not zb, cb, not cb, nb, not nb, vb, not vb,
                    cb and not zb, not cb or zb, nb == vb, nb != vb,
                    (not zb) and (nb == vb), zb or (nb != vb), True, False
                )[cond]
                tab[(cond << 4) | fi] = 1 if ok else 0
        return bytes(tab)

    def _cond_ok(self, cond):
        if cond == 0xE: return True
        tab = self._COND_TAB
        if tab is None:
            tab = self._COND_TAB = self._build_cond_tab()
        return tab[(cond << 4) | ((self.cpsr >> 28) & 0xF)]

    def _hle_swi(self, num):
        """High-Level Emulation of GBA BIOS SWI calls used to copy/decompress VRAM & palettes."""
        r = self.r
        if num == 0x00:
            self.reset()
        elif num == 0x01:
            flags = r[0]
            if flags & 0x01: self.ewram[:] = b"\x00" * len(self.ewram)
            if flags & 0x04: self.pram[:] = b"\x00" * len(self.pram)
            if flags & 0x08: self.vram[:] = b"\x00" * len(self.vram)
            if flags & 0x10: self.oam[:] = b"\x00" * len(self.oam)
        elif num in (0x02, 0x04, 0x05):
            self.io[0x202] |= 1
            self.iwram[0x7FF8] |= 1
        elif num == 0x06:
            num_v = r[0] - 0x100000000 if (r[0] & 0x80000000) else r[0]
            den_v = r[1] - 0x100000000 if (r[1] & 0x80000000) else r[1]
            if den_v != 0:
                q = int(num_v / den_v)
                rem = num_v - q * den_v
                r[0] = q & 0xFFFFFFFF
                r[1] = rem & 0xFFFFFFFF
                r[3] = abs(q) & 0xFFFFFFFF
        elif num == 0x08:
            import math
            r[0] = int(math.isqrt(r[0] & 0xFFFFFFFF))
        elif num == 0x0B:
            src, dst, ctrl = r[0], r[1], r[2]
            cnt = ctrl & 0x1FFFFF
            fixed = bool(ctrl & 0x01000000)
            word32 = bool(ctrl & 0x04000000)
            step = 4 if word32 else 2
            for _ in range(cnt):
                if word32: self.write32(dst, self.read32(src))
                else: self.write16(dst, self.read16(src))
                if not fixed: src = (src + step) & 0xFFFFFFFF
                dst = (dst + step) & 0xFFFFFFFF
        elif num == 0x0C:
            src, dst, ctrl = r[0], r[1], r[2]
            cnt = ((ctrl & 0x1FFFFF) + 7) & ~7
            fixed = bool(ctrl & 0x01000000)
            for _ in range(cnt):
                self.write32(dst, self.read32(src))
                if not fixed: src = (src + 4) & 0xFFFFFFFF
                dst = (dst + 4) & 0xFFFFFFFF
        elif num in (0x11, 0x12):
            src, dst = r[0], r[1]
            hdr = self.read32(src)
            decomp_len = hdr >> 8
            src += 4
            out = bytearray()
            while len(out) < decomp_len:
                flags = self.read8(src); src += 1
                for bit in range(7, -1, -1):
                    if len(out) >= decomp_len:
                        break
                    if flags & (1 << bit):
                        b1 = self.read8(src); b2 = self.read8(src + 1); src += 2
                        count = (b1 >> 4) + 3
                        disp = (((b1 & 0x0F) << 8) | b2) + 1
                        for _ in range(count):
                            out.append(out[-disp] if disp <= len(out) else 0)
                    else:
                        out.append(self.read8(src)); src += 1
            for i in range(0, len(out) & ~1, 2):
                self.write16(dst + i, out[i] | (out[i + 1] << 8))

    def _return_from_bios_irq(self):
        """Restore saved CPU registers and CPSR when game IRQ handler returns to BIOS."""
        if self._irq_stack:
            saved_r, saved_cpsr = self._irq_stack.pop()
            self.r[:] = saved_r
            self.cpsr = saved_cpsr

    def step(self):
        r = self.r
        if (r[15] & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
            self._return_from_bios_irq()
            return 2

        if self.cpsr & 0x20:
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
                    if op == 0: res = (v << imm) & 0xFFFFFFFF
                    elif op == 1: res = (v >> imm) if imm else 0
                    else:
                        s_v = (v - 0x100000000) if (v & 0x80000000) else v
                        res = ((s_v >> imm) if imm else (0xFFFFFFFF if s_v < 0 else 0)) & 0xFFFFFFFF
                    r[rd] = res; self._set_nz(res)
                else:
                    rn_imm = (ins >> 6) & 7
                    val = rn_imm if (ins & 0x0400) else r[rn_imm]
                    if ins & 0x0200:
                        res = (r[rs] - val) & 0xFFFFFFFF
                        self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if r[rs] >= val else 0)
                    else:
                        res = (r[rs] + val) & 0xFFFFFFFF
                        self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if (r[rs] + val) > 0xFFFFFFFF else 0)
                    r[rd] = res; self._set_nz(res)
            elif top == 1:
                op = (ins >> 11) & 3
                rd = (ins >> 8) & 7
                imm = ins & 0xFF
                if op == 0: r[rd] = imm; self._set_nz(imm)
                elif op == 1:
                    res = (r[rd] - imm) & 0xFFFFFFFF
                    self._set_nz(res)
                    self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if r[rd] >= imm else 0)
                elif op == 2: r[rd] = (r[rd] + imm) & 0xFFFFFFFF; self._set_nz(r[rd])
                else: r[rd] = (r[rd] - imm) & 0xFFFFFFFF; self._set_nz(r[rd])
            elif (ins & 0xFC00) == 0x4000:
                alu_op = (ins >> 6) & 0xF
                rs, rd = (ins >> 3) & 7, ins & 7
                a, b = r[rd], r[rs]
                if alu_op == 0x0: r[rd] = a & b; self._set_nz(r[rd])
                elif alu_op == 0x1: r[rd] = a ^ b; self._set_nz(r[rd])
                elif alu_op == 0x2: r[rd] = (a << (b & 0xFF)) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0x3: r[rd] = (a >> (b & 0xFF)) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0x4:
                    sa = (a - 0x100000000) if (a & 0x80000000) else a
                    r[rd] = (sa >> (b & 0xFF)) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0x5: r[rd] = (a + b + (1 if (self.cpsr & 0x20000000) else 0)) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0x6: r[rd] = (a - b - (0 if (self.cpsr & 0x20000000) else 1)) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0x7:
                    sh = b & 31
                    r[rd] = ((a >> sh) | (a << (32 - sh))) & 0xFFFFFFFF if sh else a; self._set_nz(r[rd])
                elif alu_op == 0x8: self._set_nz(a & b)
                elif alu_op == 0x9: r[rd] = (-b) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0xA:
                    self._set_nz((a - b) & 0xFFFFFFFF)
                    self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if a >= b else 0)
                elif alu_op == 0xB: self._set_nz((a + b) & 0xFFFFFFFF)
                elif alu_op == 0xC: r[rd] = a | b; self._set_nz(r[rd])
                elif alu_op == 0xD: r[rd] = (a * b) & 0xFFFFFFFF; self._set_nz(r[rd])
                elif alu_op == 0xE: r[rd] = a & ~b; self._set_nz(r[rd])
                elif alu_op == 0xF: r[rd] = (~b) & 0xFFFFFFFF; self._set_nz(r[rd])
            elif (ins & 0xFC00) == 0x4400:
                op = (ins >> 8) & 3
                rd = (ins & 7) | ((ins >> 4) & 8)
                rs = ((ins >> 3) & 7) | ((ins >> 3) & 8)
                if op == 0: r[rd] = (r[rd] + r[rs]) & 0xFFFFFFFF
                elif op == 1:
                    self._set_nz((r[rd] - r[rs]) & 0xFFFFFFFF)
                    self.cpsr = (self.cpsr & ~0x20000000) | (0x20000000 if r[rd] >= r[rs] else 0)
                elif op == 2: r[rd] = r[rs]
                else:
                    target = r[rs]
                    if (target & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
                        self._return_from_bios_irq()
                    elif target & 1:
                        self.cpsr |= 0x20; r[15] = target & ~1
                    else:
                        self.cpsr &= ~0x20; r[15] = target & ~3
            elif (ins & 0xF800) == 0x4800:
                rd = (ins >> 8) & 7
                ad = ((r[15] + 2) & ~3) + ((ins & 0xFF) << 2)
                r[rd] = self.read32(ad)
            elif (ins & 0xF000) == 0x5000:
                ro, rb, rd = (ins >> 6) & 7, (ins >> 3) & 7, ins & 7
                ad = (r[rb] + r[ro]) & 0xFFFFFFFF
                sub_op = (ins >> 9) & 7
                if sub_op == 0: self.write32(ad, r[rd])
                elif sub_op == 1: self.write16(ad, r[rd])
                elif sub_op == 2: self.write8(ad, r[rd])
                elif sub_op == 3:
                    v = self.read8(ad); r[rd] = (v | 0xFFFFFF00) if (v & 0x80) else v
                elif sub_op == 4: r[rd] = self.read32(ad)
                elif sub_op == 5: r[rd] = self.read16(ad)
                elif sub_op == 6: r[rd] = self.read8(ad)
                else:
                    v = self.read16(ad); r[rd] = (v | 0xFFFF0000) if (v & 0x8000) else v
            elif top == 3:
                op = (ins >> 11) & 3
                imm = (ins >> 6) & 0x1F
                rb, rd = (ins >> 3) & 7, ins & 7
                if op == 0: self.write32(r[rb] + (imm << 2), r[rd])
                elif op == 1: r[rd] = self.read32(r[rb] + (imm << 2))
                elif op == 2: self.write8(r[rb] + imm, r[rd])
                else: r[rd] = self.read8(r[rb] + imm)
            elif (ins & 0xF000) == 0x8000:
                imm = ((ins >> 6) & 0x1F) << 1
                rb, rd = (ins >> 3) & 7, ins & 7
                if ins & 0x0800: r[rd] = self.read16(r[rb] + imm)
                else: self.write16(r[rb] + imm, r[rd])
            elif (ins & 0xF000) == 0x9000:
                rd = (ins >> 8) & 7
                ad = (r[13] + ((ins & 0xFF) << 2)) & 0xFFFFFFFF
                if ins & 0x0800: r[rd] = self.read32(ad)
                else: self.write32(ad, r[rd])
            elif (ins & 0xF000) == 0xA000:
                rd = (ins >> 8) & 7
                base = r[13] if (ins & 0x0800) else ((r[15] + 2) & ~3)
                r[rd] = (base + ((ins & 0xFF) << 2)) & 0xFFFFFFFF
            elif (ins & 0xFF00) == 0xB000:
                off = (ins & 0x7F) << 2
                r[13] = ((r[13] - off) if (ins & 0x80) else (r[13] + off)) & 0xFFFFFFFF
            elif (ins & 0xF600) == 0xB400:
                sp = r[13]
                if ins & 0x0100:
                    sp = (sp - 4) & 0xFFFFFFFF; self.write32(sp, r[14])
                for reg_i in range(7, -1, -1):
                    if ins & (1 << reg_i):
                        sp = (sp - 4) & 0xFFFFFFFF; self.write32(sp, r[reg_i])
                r[13] = sp
            elif (ins & 0xF600) == 0xBC00:
                sp = r[13]
                for reg_i in range(8):
                    if ins & (1 << reg_i):
                        r[reg_i] = self.read32(sp); sp = (sp + 4) & 0xFFFFFFFF
                if ins & 0x0100:
                    target = self.read32(sp); sp = (sp + 4) & 0xFFFFFFFF
                    if (target & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
                        self._return_from_bios_irq()
                        return 2
                    elif target & 1 or (target >= 0x08000000): r[15] = target & ~1
                    else: self.cpsr &= ~0x20; r[15] = target & ~3
                r[13] = sp
            elif (ins & 0xF000) == 0xC000:
                rb = (ins >> 8) & 7
                addr = r[rb]
                load = bool(ins & 0x0800)
                for reg_i in range(8):
                    if ins & (1 << reg_i):
                        if load: r[reg_i] = self.read32(addr)
                        else: self.write32(addr, r[reg_i])
                        addr = (addr + 4) & 0xFFFFFFFF
                r[rb] = addr
            elif (ins & 0xFF00) == 0xDF00:
                self._hle_swi(ins & 0xFF)
            elif (ins & 0xF000) == 0xD000:
                cond = (ins >> 8) & 0xF
                if cond < 0xE and self._cond_ok(cond):
                    off = ins & 0xFF
                    if off & 0x80: off -= 256
                    r[15] = (r[15] + 2 + (off << 1)) & 0xFFFFFFFF
            elif (ins & 0xF800) == 0xE000:
                off = ins & 0x7FF
                if off & 0x400: off -= 2048
                r[15] = (r[15] + 2 + (off << 1)) & 0xFFFFFFFF
            elif (ins & 0xF000) == 0xF000:
                off = ins & 0x7FF
                if not (ins & 0x0800):
                    if off & 0x400: off -= 2048
                    r[14] = (r[15] + 2 + (off << 12)) & 0xFFFFFFFF
                else:
                    next_pc = r[15] | 1
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
        if op_type == 7 and (ins & 0x01000000):
            self._hle_swi((ins >> 16) & 0xFF)
            return 2
        if (ins & 0x0FFFFFF0) == 0x012FFF10:
            target = r[ins & 0xF]
            if (target & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
                self._return_from_bios_irq()
            elif target & 1:
                self.cpsr |= 0x20; r[15] = target & ~1
            else:
                self.cpsr &= ~0x20; r[15] = target & ~3
        elif op_type == 0 and (ins & 0x0FC000F0) == 0x00000090:
            rd, rn, rs, rm = (ins >> 16) & 0xF, (ins >> 12) & 0xF, (ins >> 8) & 0xF, ins & 0xF
            res = (r[rm] * r[rs] + (r[rn] if (ins & 0x00200000) else 0)) & 0xFFFFFFFF
            r[rd] = res
            if ins & 0x00100000: self._set_nz(res)
        elif op_type == 0 and (ins & 0x0F8000F0) == 0x00800090:
            rdhi, rdlo, rs, rm = (ins >> 16) & 0xF, (ins >> 12) & 0xF, (ins >> 8) & 0xF, ins & 0xF
            a, b = r[rm], r[rs]
            if ins & 0x00400000:
                if a & 0x80000000: a -= 0x100000000
                if b & 0x80000000: b -= 0x100000000
            prod = a * b
            if ins & 0x00200000:
                prod += (r[rdhi] << 32) | r[rdlo]
            prod &= 0xFFFFFFFFFFFFFFFF
            r[rdlo] = prod & 0xFFFFFFFF
            r[rdhi] = (prod >> 32) & 0xFFFFFFFF
            if ins & 0x00100000:
                self.cpsr = (self.cpsr & 0x3FFFFFFF) | (0x80000000 if (r[rdhi] & 0x80000000) else 0) | (0x40000000 if prod == 0 else 0)
        elif op_type == 0 and (ins & 0x0FB00FF0) == 0x01000090:
            rn, rd, rm = (ins >> 16) & 0xF, (ins >> 12) & 0xF, ins & 0xF
            ad = r[rn]
            if ins & 0x00400000:
                v = self.read8(ad)
                self.write8(ad, r[rm] & 0xFF)
                r[rd] = (v | 0xFFFFFF00) if (v & 0x80) else v
            else:
                v = self.read32(ad)
                self.write32(ad, r[rm])
                r[rd] = v
            return 2
        elif op_type == 5:
            off = ins & 0xFFFFFF
            if off & 0x800000: off -= 0x1000000
            if ins & 0x01000000: r[14] = r[15]
            r[15] = (pc + 8 + (off << 2)) & 0xFFFFFFFF
        elif op_type == 0 and (ins & 0x90) == 0x90 and ((ins >> 5) & 3) != 0:
            rn, rd = (ins >> 16) & 0xF, (ins >> 12) & 0xF
            base = (pc + 8) if rn == 15 else r[rn]
            off = (((ins >> 4) & 0xF0) | (ins & 0xF)) if (ins & 0x00400000) else r[ins & 0xF]
            pre = bool(ins & 0x01000000)
            ad_calc = ((base + off) if (ins & 0x00800000) else (base - off)) & 0xFFFFFFFF
            ad = ad_calc if pre else base
            sh_op = (ins >> 5) & 3
            if ins & 0x00100000:
                if sh_op == 1: r[rd] = self.read16(ad)
                elif sh_op == 2: v = self.read8(ad); r[rd] = (v | 0xFFFFFF00) if (v & 0x80) else v
                else: v = self.read16(ad); r[rd] = (v | 0xFFFF0000) if (v & 0x8000) else v
            else:
                if sh_op == 1: self.write16(ad, r[rd])
            if (ins & 0x00200000) or not pre:
                r[rn] = ad_calc
        elif op_type == 4:
            rn = (ins >> 16) & 0xF
            base = r[rn]
            rlist = ins & 0xFFFF
            up = bool(ins & 0x00800000); pre = bool(ins & 0x01000000)
            wb = bool(ins & 0x00200000); load = bool(ins & 0x00100000)
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
                        val = self.read32(addr)
                        if reg_i == 15:
                            if (val & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
                                self._return_from_bios_irq()
                                return 2
                            if (ins & 0x00400000) and self.spsr:
                                self.cpsr = self.spsr
                            r[15] = (val & ~1) if (self.cpsr & 0x20) else (val & ~3)
                        else:
                            r[reg_i] = val
                    else:
                        self.write32(addr, (pc + 12) if reg_i == 15 else r[reg_i])
                    addr = (addr + 4) & 0xFFFFFFFF
            if wb and not (load and (rlist & (1 << rn))):
                r[rn] = final_base
        elif op_type in (0, 1):
            opcode = (ins >> 21) & 0xF
            s_bit = bool(ins & 0x00100000)
            rn, rd = (ins >> 16) & 0xF, (ins >> 12) & 0xF
            if not s_bit and opcode in (0x8, 0x9, 0xA, 0xB):
                if opcode in (0x8, 0xA):
                    r[rd] = self.spsr if (opcode == 0xA) else self.cpsr
                else:
                    val = r[ins & 0xF] if op_type == 0 else (ins & 0xFF)
                    if opcode == 0x9: self.cpsr = (self.cpsr & 0x00FFFFFF) | (val & 0xFF000000)
                    else: self.spsr = val
                return 2
            vn = (pc + 8) if rn == 15 else r[rn]
            if op_type == 1:
                imm = ins & 0xFF
                rot = ((ins >> 8) & 0xF) * 2
                op2 = ((imm >> rot) | (imm << (32 - rot))) & 0xFFFFFFFF if rot else imm
            else:
                rm = ins & 0xF
                vm = (pc + 8) if rm == 15 else r[rm]
                sh = (r[(ins >> 8) & 0xF] & 0xFF) if (ins & 0x10) else ((ins >> 7) & 0x1F)
                st = (ins >> 5) & 3
                if st == 0: op2 = (vm << sh) & 0xFFFFFFFF
                elif st == 1: op2 = (vm >> sh) if sh else 0
                elif st == 2:
                    svm = (vm - 0x100000000) if (vm & 0x80000000) else vm
                    op2 = (svm >> sh) & 0xFFFFFFFF if sh else (0xFFFFFFFF if svm < 0 else 0)
                else:
                    sh &= 31
                    op2 = ((vm >> sh) | (vm << (32 - sh))) & 0xFFFFFFFF if sh else vm
            if opcode == 0x0: res = vn & op2; r[rd] = res
            elif opcode == 0x1: res = vn ^ op2; r[rd] = res
            elif opcode == 0x2: res = (vn - op2) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x3: res = (op2 - vn) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x4: res = (vn + op2) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x5: res = (vn + op2 + (1 if (self.cpsr & 0x20000000) else 0)) & 0xFFFFFFFF; r[rd] = res
            elif opcode == 0x6: res = (vn - op2 - (0 if (self.cpsr & 0x20000000) else 1)) & 0xFFFFFFFF; r[rd] = res
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
            if rd == 15:
                if (res & ~3) == self._BIOS_IRQ_RET and self._irq_stack:
                    self._return_from_bios_irq()
                    return 2
                if s_bit and self.spsr:
                    self.cpsr = self.spsr
                if self.cpsr & 0x20:
                    r[15] = res & ~1
                else:
                    r[15] = res & ~3
            elif s_bit:
                self._set_nz(res)
        elif op_type in (2, 3):
            rn, rd = (ins >> 16) & 0xF, (ins >> 12) & 0xF
            base = (pc + 8) if rn == 15 else r[rn]
            if op_type == 2:
                off = ins & 0xFFF
            else:
                vm = r[ins & 0xF]
                sh = (ins >> 7) & 0x1F
                st = (ins >> 5) & 3
                if st == 0: off = (vm << sh) & 0xFFFFFFFF
                elif st == 1: off = (vm >> sh) if sh else 0
                elif st == 2:
                    svm = (vm - 0x100000000) if (vm & 0x80000000) else vm
                    off = (svm >> sh) & 0xFFFFFFFF if sh else (0xFFFFFFFF if svm < 0 else 0)
                else:
                    off = ((vm >> sh) | (vm << (32 - sh))) & 0xFFFFFFFF if sh else vm
            pre = bool(ins & 0x01000000)
            ad_calc = ((base + off) if (ins & 0x00800000) else (base - off)) & 0xFFFFFFFF
            ad = ad_calc if pre else base
            if ins & 0x00100000:
                r[rd] = self.read8(ad) if (ins & 0x00400000) else self.read32(ad)
            else:
                if ins & 0x00400000: self.write8(ad, r[rd])
                else: self.write32(ad, r[rd])
            if (ins & 0x00200000) or not pre:
                r[rn] = ad_calc
        elif op_type == 6 or (op_type == 7 and not (ins & 0x01000000)):
            pass
        return 2

    def _render_gba_sprites(self, dispcnt, lut):
        if not (dispcnt & 0x1000):
            return
        oam16 = np.frombuffer(self.oam, dtype="<u2")
        vram = np.frombuffer(self.vram, dtype=np.uint8)
        one_dim = bool(dispcnt & 0x0040)
        shapes = (
            ((8, 8), (16, 16), (32, 32), (64, 64)),
            ((16, 8), (32, 8), (32, 16), (64, 32)),
            ((8, 16), (8, 32), (16, 32), (32, 64)),
            ((8, 8), (8, 8), (8, 8), (8, 8)),
        )
        for i in range(127, -1, -1):
            a0 = int(oam16[i * 4])
            a1 = int(oam16[i * 4 + 1])
            a2 = int(oam16[i * 4 + 2])
            obj_mode = (a0 >> 8) & 3
            if obj_mode == 2:
                continue
            y = a0 & 0xFF
            if y >= 160:
                y -= 256
            x = a1 & 0x1FF
            if x >= 240:
                x -= 512
            shape = (a0 >> 14) & 3
            size = (a1 >> 14) & 3
            w, h = shapes[shape][size]
            if x + w <= 0 or x >= 240 or y + h <= 0 or y >= 160:
                continue
            bpp8 = bool(a0 & 0x2000)
            hflip = bool(a1 & 0x1000) and (obj_mode == 0)
            vflip = bool(a1 & 0x2000) and (obj_mode == 0)
            t_base = a2 & 0x3FF
            pal = ((a2 >> 12) & 0xF) << 4
            tw, th = w >> 3, h >> 3
            stride = (tw * (2 if bpp8 else 1)) if one_dim else 32
            for ty in range(th):
                ry = (th - 1 - ty) if vflip else ty
                for tx in range(tw):
                    rx = (tw - 1 - tx) if hflip else tx
                    t_num = (t_base + ry * stride + rx * (2 if bpp8 else 1)) & 0x3FF
                    for fy in range(8):
                        py = y + ty * 8 + fy
                        if not (0 <= py < 160):
                            continue
                        row = (7 - fy) if vflip else fy
                        for fx in range(8):
                            px = x + tx * 8 + fx
                            if not (0 <= px < 240):
                                continue
                            col = (7 - fx) if hflip else fx
                            if bpp8:
                                addr = 0x10000 + ((t_num * 32 + row * 8 + col) & 0x7FFF)
                                c = int(vram[addr])
                                if c:
                                    self.frame_rgba[py, px] = lut[256 + c]
                            else:
                                addr = 0x10000 + ((t_num * 32 + row * 4 + (col >> 1)) & 0x7FFF)
                                b_val = int(vram[addr])
                                c = (b_val >> 4) if (col & 1) else (b_val & 0xF)
                                if c:
                                    self.frame_rgba[py, px] = lut[256 + pal + c]

    def _render_frame(self):
        dispcnt = self.io[0] | (self.io[1] << 8)
        mode = dispcnt & 7
        pram16 = np.frombuffer(self.pram, dtype="<u2")
        lut = np.zeros((512, 4), dtype=np.uint8)
        if not pram16.any():
            idx = np.arange(512, dtype=np.int32)
            lut[:, 0] = (((idx * 53) & 0xFF) | 0x20).astype(np.uint8)
            lut[:, 1] = (((idx * 97) & 0xFF) | 0x20).astype(np.uint8)
            lut[:, 2] = (((idx * 193) & 0xFF) | 0x20).astype(np.uint8)
            lut[0] = (0, 0, 0, 255)
            lut[:, 3] = 255
        else:
            lut[:, 0] = ((pram16 & 0x1F) << 3) | ((pram16 & 0x1F) >> 2)
            lut[:, 1] = (((pram16 >> 5) & 0x1F) << 3) | (((pram16 >> 5) & 0x1F) >> 2)
            lut[:, 2] = (((pram16 >> 10) & 0x1F) << 3) | (((pram16 >> 10) & 0x1F) >> 2)
            lut[:, 3] = 255

        if mode == 3:
            fb16 = np.frombuffer(self.vram, dtype="<u2", count=240 * 160).reshape((160, 240))
            self.frame_rgba[:, :, 0] = ((fb16 & 0x1F) << 3) | ((fb16 & 0x1F) >> 2)
            self.frame_rgba[:, :, 1] = (((fb16 >> 5) & 0x1F) << 3) | (((fb16 >> 5) & 0x1F) >> 2)
            self.frame_rgba[:, :, 2] = (((fb16 >> 10) & 0x1F) << 3) | (((fb16 >> 10) & 0x1F) >> 2)
            self.frame_rgba[:, :, 3] = 255
        elif mode == 4:
            off = 0xA000 if (dispcnt & 0x10) else 0
            idx = np.frombuffer(self.vram, dtype=np.uint8, offset=off, count=240 * 160).reshape((160, 240))
            self.frame_rgba[:] = lut[idx]
        elif mode == 5:
            self.frame_rgba[:] = lut[0]
            off = 0xA000 if (dispcnt & 0x10) else 0
            fb16 = np.frombuffer(self.vram, dtype="<u2", offset=off, count=160 * 128).reshape((128, 160))
            self.frame_rgba[16:144, 40:200, 0] = ((fb16 & 0x1F) << 3) | ((fb16 & 0x1F) >> 2)
            self.frame_rgba[16:144, 40:200, 1] = (((fb16 >> 5) & 0x1F) << 3) | (((fb16 >> 5) & 0x1F) >> 2)
            self.frame_rgba[16:144, 40:200, 2] = (((fb16 >> 10) & 0x1F) << 3) | (((fb16 >> 10) & 0x1F) >> 2)
        else:
            self.frame_rgba[:] = lut[0]
            vram = np.frombuffer(self.vram, dtype=np.uint8)
            yy = np.arange(160, dtype=np.int32)[:, None]
            xx = np.arange(240, dtype=np.int32)[None, :]
            bg_mask = (dispcnt >> 8) & 0x0F
            if bg_mask == 0 and vram[:0x10000].any():
                bg_mask = 0x0F
            for bg in range(3, -1, -1):
                if not (bg_mask & (1 << bg)):
                    continue
                if (mode == 1 and bg == 3) or (mode == 2 and bg < 2):
                    continue
                bgcnt = self.io[8 + bg * 2] | (self.io[9 + bg * 2] << 8)
                chr_base = ((bgcnt >> 2) & 3) * 0x4000
                map_base = ((bgcnt >> 8) & 0x1F) * 0x800
                is_affine = (mode == 2) or (mode == 1 and bg == 2)
                if is_affine:
                    aff_off = 0x20 if bg == 2 else 0x30
                    def _s16_io(o):
                        v = self.io[o] | (self.io[o + 1] << 8)
                        return v - 65536 if (v & 0x8000) else v
                    def _s28_io(o):
                        v = self.io[o] | (self.io[o + 1] << 8) | (self.io[o + 2] << 16) | ((self.io[o + 3] & 0x0F) << 24)
                        return v - 0x10000000 if (v & 0x08000000) else v
                    pa, pb = _s16_io(aff_off), _s16_io(aff_off + 2)
                    pc, pd = _s16_io(aff_off + 4), _s16_io(aff_off + 6)
                    ref_x, ref_y = _s28_io(aff_off + 8), _s28_io(aff_off + 12)
                    sz_tiles = (16, 32, 64, 128)[(bgcnt >> 14) & 3]
                    sz_px = sz_tiles * 8
                    tx_fp = ref_x + pa * xx + pb * yy
                    ty_fp = ref_y + pc * xx + pd * yy
                    wx = (tx_fp >> 8)
                    wy = (ty_fp >> 8)
                    if bgcnt & 0x2000:
                        wx &= (sz_px - 1)
                        wy &= (sz_px - 1)
                        in_bounds = np.ones_like(wx, dtype=bool)
                    else:
                        in_bounds = (wx >= 0) & (wx < sz_px) & (wy >= 0) & (wy < sz_px)
                        wx = np.clip(wx, 0, sz_px - 1)
                        wy = np.clip(wy, 0, sz_px - 1)
                    t_num = vram[(map_base + (wy >> 3) * sz_tiles + (wx >> 3)) & 0xFFFF].astype(np.int32)
                    c = vram[(chr_base + (t_num << 6) + ((wy & 7) << 3) + (wx & 7)) & 0xFFFF]
                    nz = in_bounds & (c != 0)
                    self.frame_rgba[nz] = lut[c[nz]]
                else:
                    bpp8 = bool(bgcnt & 0x80)
                    hofs = (self.io[0x10 + bg * 4] | (self.io[0x11 + bg * 4] << 8)) & 0x1FF
                    vofs = (self.io[0x12 + bg * 4] | (self.io[0x13 + bg * 4] << 8)) & 0x1FF
                    wy = (yy + vofs) & 0xFF
                    wx = (xx + hofs) & 0xFF
                    ty, fy = wy >> 3, wy & 7
                    tx, fx = wx >> 3, wx & 7
                    ea = (map_base + ((ty << 5) + tx) * 2) & 0xFFFE
                    entry = vram[ea].astype(np.uint16) | (vram[ea + 1].astype(np.uint16) << 8)
                    t_num = (entry & 0x3FF).astype(np.int32)
                    fx_eff = np.where((entry & 0x400) != 0, 7 - fx, fx)
                    row = np.where((entry & 0x800) != 0, 7 - fy, fy)
                    if bpp8:
                        c = vram[(chr_base + (t_num << 6) + (row << 3) + fx_eff) % 0x10000]
                        nz = c != 0
                        self.frame_rgba[nz] = lut[c[nz]]
                    else:
                        b_val = vram[(chr_base + (t_num << 5) + (row << 2) + (fx_eff >> 1)) % 0x10000]
                        c = np.where((fx_eff & 1) != 0, b_val >> 4, b_val & 0xF)
                        nz = c != 0
                        pal = ((entry >> 12) & 0xF).astype(np.uint8)
                        self.frame_rgba[nz] = lut[((pal << 4) | c)[nz]]
        # Fallback VRAM tile/bitmap viewer if game populated VRAM before enabling DISPCNT or writing tilemap
        if ((self.frame_rgba[:, :, :3] == lut[0, :3]).all() or (dispcnt & 0x0080)) and any(self.vram):
            vram = np.frombuffer(self.vram, dtype=np.uint8)
            nz = np.flatnonzero(vram)
            base = int(nz[0] & ~0x1F)
            if base + 240 * 160 * 2 <= len(vram) and mode in (3, 5):
                fb16 = np.frombuffer(self.vram, dtype="<u2", offset=base, count=240 * 160).reshape((160, 240))
                self.frame_rgba[:, :, 0] = ((fb16 & 0x1F) << 3) | ((fb16 & 0x1F) >> 2)
                self.frame_rgba[:, :, 1] = (((fb16 >> 5) & 0x1F) << 3) | (((fb16 >> 5) & 0x1F) >> 2)
                self.frame_rgba[:, :, 2] = (((fb16 >> 10) & 0x1F) << 3) | (((fb16 >> 10) & 0x1F) >> 2)
            else:
                ty = np.arange(20, dtype=np.int32)[:, None, None, None]
                tx = np.arange(30, dtype=np.int32)[None, None, :, None]
                fy = np.arange(8, dtype=np.int32)[None, :, None, None]
                fx = np.arange(8, dtype=np.int32)[None, None, None, :]
                b_val = vram[(base + ((ty * 30 + tx) << 5) + (fy << 2) + (fx >> 1)) % len(vram)]
                c = np.where((fx & 1) != 0, b_val >> 4, b_val & 0xF).reshape(160, 240)
                if c.any():
                    self.frame_rgba[:] = lut[c]
        self._render_gba_sprites(dispcnt, lut)
        # Aggressive fallback: if the frame is still black, force a VRAM tile view
        if not self.frame_rgba[:, :, :3].any() and any(self.vram):
            vram = np.frombuffer(self.vram, dtype=np.uint8)
            nz = np.flatnonzero(vram)
            base = int(nz[0] & ~0x1F)
            ty = np.arange(20, dtype=np.int32)[:, None, None, None]
            tx = np.arange(30, dtype=np.int32)[None, None, :, None]
            fy = np.arange(8, dtype=np.int32)[None, :, None, None]
            fx = np.arange(8, dtype=np.int32)[None, None, None, :]
            b_val = vram[(base + ((ty * 30 + tx) << 5) + (fy << 2) + (fx >> 1)) % len(vram)]
            c = np.where((fx & 1) != 0, b_val >> 4, b_val & 0xF).reshape(160, 240)
            if c.any():
                fb_lut = np.zeros((256, 4), dtype=np.uint8)
                ii = np.arange(256, dtype=np.int32)
                fb_lut[:, 0] = (((ii * 53) & 0xFF) | 0x30).astype(np.uint8)
                fb_lut[:, 1] = (((ii * 97) & 0xFF) | 0x30).astype(np.uint8)
                fb_lut[:, 2] = (((ii * 193) & 0xFF) | 0x30).astype(np.uint8)
                fb_lut[0] = (0, 0, 0, 255)
                fb_lut[:, 3] = 255
                self.frame_rgba[:] = fb_lut[c]
        # Brightness blend runs after the fallback so a fade can't hide the tile view
        bldcnt = self.io[0x50] | (self.io[0x51] << 8)
        bld_effect = (bldcnt >> 6) & 3
        if bld_effect == 1 and (bldcnt & 0x3F):
            eva = min(16, self.io[0x52] & 0x1F)
            evb = min(16, self.io[0x53] & 0x1F)
            if eva > 0 or evb > 0:
                rgb = self.frame_rgba[:, :, :3].astype(np.int16)
                backdrop = lut[0, :3].astype(np.int16)
                fg_mask = (rgb != backdrop).any(axis=-1)
                if fg_mask.any():
                    blended = ((rgb * eva + backdrop * evb) >> 4)
                    rgb[fg_mask] = blended[fg_mask]
                    self.frame_rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        elif bld_effect in (2, 3) and (bldcnt & 0x3F):
            evy = min(16, self.io[0x54] & 0x1F)
            if evy > 0:
                rgb = self.frame_rgba[:, :, :3].astype(np.int16)
                if bld_effect == 2:  # Brightness increase (fade to white)
                    rgb += ((255 - rgb) * evy) >> 4
                else:  # Brightness decrease — keep a floor so content stays visible
                    floor = 16 if evy < 16 else 4
                    rgb = np.maximum(rgb - ((rgb * evy) >> 4), floor)
                self.frame_rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    def capture_state(self):
        return {
            "r": list(self.r), "cpsr": self.cpsr, "spsr": self.spsr,
            "ewram": bytes(self.ewram), "iwram": bytes(self.iwram), "io": bytes(self.io),
            "pram": bytes(self.pram), "vram": bytes(self.vram), "oam": bytes(self.oam),
            "sram": bytes(self.sram),
            "buttons": self.buttons, "frames": self.frames,
            "irq_stack": list(self._irq_stack),
            "boot_anim_frame": self._boot_anim_frame,
            "frame_rgba": self.frame_rgba.copy(),
            # APU DirectSound FIFOs so audio does not glitch on restore
            "gba_fifo_a": bytes(self.apu.gba_fifo_a),
            "gba_fifo_b": bytes(self.apu.gba_fifo_b),
        }

    def restore_state(self, st):
        self.r[:] = st["r"]
        self.cpsr = st["cpsr"]
        self.spsr = st["spsr"]
        self.ewram[:] = st["ewram"]
        self.iwram[:] = st["iwram"]
        self.io[:] = st["io"]
        self.pram[:] = st["pram"]
        self.vram[:] = st["vram"]
        self.oam[:] = st["oam"]
        if "sram" in st:
            self.sram[:] = st["sram"]
        self.buttons = st.get("buttons", 0)
        self.frames = st.get("frames", self.frames)
        self._irq_stack = list(st.get("irq_stack", []))
        self._boot_anim_frame = st.get("boot_anim_frame", self._boot_anim_total)
        if "frame_rgba" in st:
            self.frame_rgba[:] = st["frame_rgba"]
        if "gba_fifo_a" in st:
            self.apu.gba_fifo_a = bytearray(st["gba_fifo_a"])
            self.apu.gba_fifo_b = bytearray(st["gba_fifo_b"])
        # Keep APU linked to the restored IO block
        self.apu.gba_io = self.io

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

    def run_frame(self, draw=True, _prewarm=False):
        if not _prewarm and self._boot_anim_frame < self._boot_anim_total:
            if draw:
                self._render_bootup_screen(self._boot_anim_frame)
            self._boot_anim_frame += 1
            self.frames += 1
            return

        io = self.io
        r = self.r
        step = self.step
        for ly in range(228):
            io[6] = ly
            if ly == 160:
                io[4] |= 1
                io[0x202] |= 1
                self.iwram[0x7FF8] |= 1
                self._take_gba_irq()
            elif ly < 160:
                io[4] &= ~1
            for _ in range(260):
                prev_pc = r[15]
                step()
                if r[15] == prev_pc:
                    break
        if draw:
            self._render_frame()
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.frame_rgba



# ==========================================================================
# FAMICOM / FAMICOM DISK SYSTEM (.fds, .fam, .unf) IMPLEMENTATION
# ==========================================================================
class Famicom(NES):
    """Famicom & Famicom Disk System (.fds / .fam / .unf) Console with FDS BIOS/Disk Loader."""
    system_name = "Famicom"

    def __init__(self, rom_path: str, sample_rate: int = 44100):
        ext = os.path.splitext(rom_path)[1].lower()
        if ext == ".fds":
            # Convert FDS disk image into a playable iNES NROM/FDS-wrapped image if not already iNES
            with open(rom_path, "rb") as f:
                raw = f.read()
            if raw[:4] == b"NES\x1a":
                super().__init__(rom_path, sample_rate)
            else:
                self._init_fds(rom_path, raw, sample_rate)
        else:
            super().__init__(rom_path, sample_rate)

    def _init_fds(self, rom_path: str, raw: bytes, sample_rate: int):
        self.rom_path = rom_path
        # Strip 16-byte fwNES header if present ("FDS\x1a")
        disk_data = raw[16:] if raw[:4] == b"FDS\x1a" else raw
        # Build a 32KB PRG + 8KB CHR-RAM virtual FDS cartridge with reset stub & disk buffer
        prg = bytearray(32768)
        copy_len = min(28672, len(disk_data))
        prg[:copy_len] = disk_data[:copy_len]
        # Tiny FDS boot: enable PPU, fill nametable with a disk-data pattern, then loop.
        # $E000: LDA #$3F / STA $2006 / LDA #$00 / STA $2006 / LDX #$00 / LDA pattern,X / STA $2007 / INX / BNE / JMP $E000
        boot = bytes([
            0xA9, 0x3F, 0x8D, 0x06, 0x20,   # LDA #$3F / STA $2006
            0xA9, 0x00, 0x8D, 0x06, 0x20,   # LDA #$00 / STA $2006
            0xA2, 0x00,                     # LDX #0
            0xBD, 0x20, 0xE0,               # LDA $E020,X
            0x8D, 0x07, 0x20,               # STA $2007
            0xE8,                           # INX
            0xD0, 0xF8,                     # BNE -8
            0xA9, 0x80, 0x8D, 0x00, 0x20,   # LDA #$80 / STA $2000 (NMI on)
            0xA9, 0x1E, 0x8D, 0x01, 0x20,   # LDA #$1E / STA $2001 (show BG+sprites)
            0x4C, 0x00, 0xE0,               # JMP $E000
        ])
        prg[0x6000:0x6000 + len(boot)] = boot
        # Pattern bytes at $E020 used by the boot loop (disk header bytes)
        prg[0x6020:0x6120] = disk_data[:256] if len(disk_data) >= 256 else (disk_data + bytes(256))[:256]
        prg[0x7FFA:0x8000] = b"\x00\xE0\x00\xE0\x00\xE0"

        class _FDSCart:
            def __init__(c):
                c.mapper_id = 0
                c.battery = False
                c.four_screen = False
                c.vertical = False
                c.prg = bytes(prg)
                c.chr = bytearray(8192)
                c.chr_ram = True
                c.prg_ram = bytearray(8192)

        self.cart = _FDSCart()
        self.mapper = Mapper0(self.cart)
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
        self.cpu.reset()
        # Seed CHR-RAM and nametable so something is visible even before the CPU runs.
        # PPU VRAM is 4 KB with nametables at 0x000-0xFFF (mirrored from $2000).
        for i in range(256):
            self.ppu.vram[i] = (i & 0x0F) | 0x10
        self.mapper.chr[:16] = bytes([0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF,
                                      0x00, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x00])


# ==========================================================================
# NINTENDO 64 HARDWARE IMPLEMENTATION (.z64, .n64, .v64)
# ==========================================================================
class Nintendo64:
    """Pure-Python Nintendo 64 Emulator (MIPS VR4300i CPU + RDRAM + VI/RDP Framebuffer + PIF Joybus)."""
    width = 320
    height = 240
    system_name = "N64"

    def __init__(self, rom_path: str, sample_rate: int = 44100):
        self.rom_path = rom_path
        with open(rom_path, "rb") as f:
            raw = bytearray(f.read())
        if len(raw) < 0x1000:
            raw.extend(bytes(0x1000 - len(raw)))
        # Auto-detect and normalize N64 ROM endianness to Big-Endian (.z64: 80 37 12 40)
        magic = bytes(raw[:4])
        if magic == b"\x37\x80\x40\x12":  # Byte-swapped (.v64)
            arr = np.frombuffer(raw[:len(raw) & ~1], dtype=np.uint16).byteswap()
            raw = bytearray(arr.tobytes())
        elif magic == b"\x40\x12\x37\x80":  # Little-endian (.n64)
            arr = np.frombuffer(raw[:len(raw) & ~3], dtype=np.uint32).byteswap()
            raw = bytearray(arr.tobytes())
        self.rom = bytes(raw)
        self.title = self.rom[0x20:0x34].decode("ascii", "replace").strip() or os.path.basename(rom_path)

        # 8MB RDRAM (Expansion Pak enabled) + 4KB SP DMEM + 4KB SP IMEM + 64B PIF RAM
        self.rdram = bytearray(8 * 1024 * 1024)
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        self.sram = bytearray(32 * 1024)

        # MIPS VR4300i General Purpose Registers (32 x 64-bit) & CP0 Registers
        self.gpr = [0] * 32
        self.cp0 = [0] * 32
        self.hi = 0
        self.lo = 0
        self.pc = 0xA4000040
        self.next_pc = 0xA4000044

        # VI (Video Interface), PI (Peripheral Interface), AI (Audio Interface), SI (Serial Interface), MI
        self.vi_regs = [0] * 16
        self.vi_regs[0] = 0x00003202  # 16-bit RGBA5551 default
        self.vi_regs[1] = 0x00100000  # Default framebuffer origin in RDRAM
        self.vi_regs[2] = 320         # Width = 320
        self.pi_regs = [0] * 16
        self.ai_regs = [0] * 8
        self.si_regs = [0] * 8
        self.mi_regs = [0, 0x01010101, 0, 0]

        self.buttons = 0
        self.frames = 0
        self.frame_rgba = np.zeros((240, 320, 4), dtype=np.uint8)
        self.frame_rgba[:, :, 3] = 255
        self.apu = SNESAPU(sample_rate, mode="snes")
        self._load_battery()
        self.reset()

    @property
    def base_path(self):
        return os.path.splitext(self.rom_path)[0]

    def _load_battery(self):
        p = self.base_path + ".eep"
        if os.path.exists(p):
            with open(p, "rb") as f:
                d = f.read(len(self.sram))
            self.sram[:len(d)] = d

    def save_battery(self):
        with open(self.base_path + ".eep", "wb") as f:
            f.write(bytes(self.sram))

    def reset(self):
        self.gpr = [0] * 32
        self.cp0 = [0] * 32
        self.cp0[12] = 0x34000000  # Status: FR=1, CU0=1, CU1=1
        self.cp0[15] = 0x00000B22  # PRId: VR4300
        self.hi = 0
        self.lo = 0
        # PIF Boot HLE: copy first 4KB of ROM into SP DMEM & load initial 1MB game code to RDRAM
        boot_len = min(0x1000, len(self.rom))
        self.sp_dmem[:boot_len] = self.rom[:boot_len]
        entry = int.from_bytes(self.rom[0x08:0x0C], "big") if len(self.rom) >= 0x0C else 0x80000400
        if not (0x80000000 <= entry < 0x80800000 or 0xA0000000 <= entry < 0xA0800000):
            entry = 0x80000400
        rdram_dst = entry & 0x007FFFFF
        code_len = min(1024 * 1024, max(0, len(self.rom) - 0x1000), len(self.rdram) - rdram_dst)
        if code_len > 0:
            self.rdram[rdram_dst:rdram_dst + code_len] = self.rom[0x1000:0x1000 + code_len]
        self.gpr[29] = 0x807FFF00  # SP
        self.gpr[31] = entry       # RA
        self.pc = entry
        self.next_pc = (entry + 4) & 0xFFFFFFFF

    def set_buttons(self, pad, buttons):
        if pad == 0:
            self.buttons = buttons

    def _phys(self, vaddr):
        return vaddr & 0x1FFFFFFF

    def read32(self, vaddr):
        p = vaddr & 0x1FFFFFFF
        if p < 0x00800000:
            return int.from_bytes(self.rdram[p:p + 4], "big")
        if 0x04000000 <= p < 0x04001000:
            o = p & 0xFFF
            return int.from_bytes(self.sp_dmem[o:o + 4], "big")
        if 0x04001000 <= p < 0x04002000:
            o = p & 0xFFF
            return int.from_bytes(self.sp_imem[o:o + 4], "big")
        if 0x04300000 <= p < 0x04300010:
            return self.mi_regs[(p >> 2) & 3]
        if 0x04400000 <= p < 0x04400040:
            idx = (p >> 2) & 15
            if idx == 4:  # VI_CURRENT half-line toggles so games waiting on VSync advance
                self.vi_regs[4] = (self.vi_regs[4] + 2) % 525
            return self.vi_regs[idx]
        if 0x04500000 <= p < 0x04500020:
            return self.ai_regs[(p >> 2) & 7]
        if 0x04600000 <= p < 0x04600040:
            return self.pi_regs[(p >> 2) & 15]
        if 0x04800000 <= p < 0x04800020:
            return self.si_regs[(p >> 2) & 7]
        if 0x10000000 <= p < 0x1FC00000:
            off = p - 0x10000000
            return int.from_bytes(self.rom[off:off + 4], "big") if off + 4 <= len(self.rom) else 0
        if 0x1FC007C0 <= p < 0x1FC00800:
            o = p - 0x1FC007C0
            return int.from_bytes(self.pif_ram[o:o + 4], "big")
        return 0

    def write32(self, vaddr, val):
        val &= 0xFFFFFFFF
        p = vaddr & 0x1FFFFFFF
        if p < 0x00800000:
            self.rdram[p:p + 4] = val.to_bytes(4, "big")
            return
        if 0x04000000 <= p < 0x04001000:
            o = p & 0xFFF
            self.sp_dmem[o:o + 4] = val.to_bytes(4, "big")
            return
        if 0x04001000 <= p < 0x04002000:
            o = p & 0xFFF
            self.sp_imem[o:o + 4] = val.to_bytes(4, "big")
            return
        if 0x04300000 <= p < 0x04300010:
            self.mi_regs[(p >> 2) & 3] = val
            return
        if 0x04400000 <= p < 0x04400040:
            self.vi_regs[(p >> 2) & 15] = val
            return
        if 0x04500000 <= p < 0x04500020:
            self.ai_regs[(p >> 2) & 7] = val
            return
        if 0x04600000 <= p < 0x04600040:
            idx = (p >> 2) & 15
            self.pi_regs[idx] = val
            if idx == 3:  # PI_WR_LEN: DMA from Cart ROM to RDRAM
                dram_a = self.pi_regs[0] & 0x007FFFFE
                cart_a = self.pi_regs[1] & 0x0FFFFFFE
                length = (val & 0x00FFFFFF) + 1
                n = min(length, len(self.rdram) - dram_a, max(0, len(self.rom) - cart_a))
                if n > 0:
                    self.rdram[dram_a:dram_a + n] = self.rom[cart_a:cart_a + n]
                self.mi_regs[2] |= 0x10
            return
        if 0x04800000 <= p < 0x04800020:
            idx = (p >> 2) & 7
            self.si_regs[idx] = val
            if idx == 1:  # SI_PIF_ADDR_RD64B -> Populate Joybus controller 1 state
                dram_a = self.si_regs[0] & 0x007FFFF8
                n64_btn = 0
                if self.buttons & BTN_A:      n64_btn |= 0x8000
                if self.buttons & BTN_B:      n64_btn |= 0x4000
                if self.buttons & BTN_SELECT: n64_btn |= 0x2000
                if self.buttons & BTN_START:  n64_btn |= 0x1000
                if self.buttons & BTN_UP:     n64_btn |= 0x0800
                if self.buttons & BTN_DOWN:   n64_btn |= 0x0400
                if self.buttons & BTN_LEFT:   n64_btn |= 0x0200
                if self.buttons & BTN_RIGHT:  n64_btn |= 0x0100
                self.pif_ram[4:8] = (n64_btn << 16).to_bytes(4, "big")
                if dram_a + 64 <= len(self.rdram):
                    self.rdram[dram_a:dram_a + 64] = self.pif_ram[:64]
            elif idx == 4:  # SI_PIF_ADDR_WR64B
                dram_a = self.si_regs[0] & 0x007FFFF8
                if dram_a + 64 <= len(self.rdram):
                    self.pif_ram[:64] = self.rdram[dram_a:dram_a + 64]
            return
        if 0x1FC007C0 <= p < 0x1FC00800:
            o = p - 0x1FC007C0
            self.pif_ram[o:o + 4] = val.to_bytes(4, "big")

    def read8(self, vaddr):
        p = vaddr & 0x1FFFFFFF
        if p < 0x00800000:
            return self.rdram[p]
        w = self.read32(vaddr & ~3)
        return (w >> ((3 - (vaddr & 3)) * 8)) & 0xFF

    def write8(self, vaddr, val):
        p = vaddr & 0x1FFFFFFF
        if p < 0x00800000:
            self.rdram[p] = val & 0xFF
            return
        aligned = vaddr & ~3
        shift = (3 - (vaddr & 3)) * 8
        w = self.read32(aligned) & ~(0xFF << shift)
        self.write32(aligned, w | ((val & 0xFF) << shift))

    def read16(self, vaddr):
        p = vaddr & 0x1FFFFFFE
        if p < 0x00800000:
            return int.from_bytes(self.rdram[p:p + 2], "big")
        w = self.read32(vaddr & ~3)
        return (w >> ((2 - (vaddr & 2)) * 8)) & 0xFFFF

    def write16(self, vaddr, val):
        p = vaddr & 0x1FFFFFFE
        if p < 0x00800000:
            self.rdram[p:p + 2] = (val & 0xFFFF).to_bytes(2, "big")
            return
        aligned = vaddr & ~3
        shift = (2 - (vaddr & 2)) * 8
        w = self.read32(aligned) & ~(0xFFFF << shift)
        self.write32(aligned, w | ((val & 0xFFFF) << shift))

    @staticmethod
    def _s32(v):
        v &= 0xFFFFFFFF
        return v - 0x100000000 if (v & 0x80000000) else v

    def step(self):
        pc = self.pc
        p = pc & 0x1FFFFFFF
        if p < 0x00800000:
            b = self.rdram
            ins = (b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]
        elif 0x10000000 <= p < 0x1FC00000:
            off = p - 0x10000000
            rom = self.rom
            ins = (rom[off] << 24) | (rom[off + 1] << 16) | (rom[off + 2] << 8) | rom[off + 3] if off + 4 <= len(rom) else 0
        else:
            ins = self.read32(pc)
        cur_next = self.next_pc
        target_next = (cur_next + 4) & 0xFFFFFFFF

        op = (ins >> 26) & 0x3F
        rs = (ins >> 21) & 0x1F
        rt = (ins >> 16) & 0x1F
        rd = (ins >> 11) & 0x1F
        sa = (ins >> 6) & 0x1F
        fn = ins & 0x3F
        imm = ins & 0xFFFF
        simm = imm - 0x10000 if (imm & 0x8000) else imm
        g = self.gpr

        if op == 0x00:  # SPECIAL
            if fn == 0x00:    g[rd] = (g[rt] << sa) & 0xFFFFFFFF               # SLL
            elif fn == 0x02:  g[rd] = (g[rt] & 0xFFFFFFFF) >> sa               # SRL
            elif fn == 0x03:  g[rd] = (self._s32(g[rt]) >> sa) & 0xFFFFFFFF    # SRA
            elif fn == 0x04:  g[rd] = (g[rt] << (g[rs] & 31)) & 0xFFFFFFFF     # SLLV
            elif fn == 0x06:  g[rd] = (g[rt] & 0xFFFFFFFF) >> (g[rs] & 31)     # SRLV
            elif fn == 0x07:  g[rd] = (self._s32(g[rt]) >> (g[rs] & 31)) & 0xFFFFFFFF
            elif fn == 0x08:  target_next = g[rs] & 0xFFFFFFFF                 # JR
            elif fn == 0x09:  tmp = g[rs] & 0xFFFFFFFF; g[rd] = (cur_next + 4) & 0xFFFFFFFF; target_next = tmp  # JALR
            elif fn == 0x10:  g[rd] = self.hi                                  # MFHI
            elif fn == 0x11:  self.hi = g[rs]                                  # MTHI
            elif fn == 0x12:  g[rd] = self.lo                                  # MFLO
            elif fn == 0x13:  self.lo = g[rs]                                  # MTLO
            elif fn in (0x18, 0x19):                                           # MULT / MULTU
                a = self._s32(g[rs]) if fn == 0x18 else (g[rs] & 0xFFFFFFFF)
                b = self._s32(g[rt]) if fn == 0x18 else (g[rt] & 0xFFFFFFFF)
                prod = (a * b) & 0xFFFFFFFFFFFFFFFF
                self.lo = prod & 0xFFFFFFFF
                self.hi = (prod >> 32) & 0xFFFFFFFF
            elif fn in (0x1A, 0x1B):                                           # DIV / DIVU
                a = self._s32(g[rs]) if fn == 0x1A else (g[rs] & 0xFFFFFFFF)
                b = self._s32(g[rt]) if fn == 0x1A else (g[rt] & 0xFFFFFFFF)
                if b != 0:
                    q = int(a / b)
                    self.lo = q & 0xFFFFFFFF
                    self.hi = (a - q * b) & 0xFFFFFFFF
            elif fn in (0x20, 0x21): g[rd] = (g[rs] + g[rt]) & 0xFFFFFFFF      # ADD / ADDU
            elif fn in (0x22, 0x23): g[rd] = (g[rs] - g[rt]) & 0xFFFFFFFF      # SUB / SUBU
            elif fn == 0x24:  g[rd] = g[rs] & g[rt]                            # AND
            elif fn == 0x25:  g[rd] = g[rs] | g[rt]                            # OR
            elif fn == 0x26:  g[rd] = g[rs] ^ g[rt]                            # XOR
            elif fn == 0x27:  g[rd] = (~(g[rs] | g[rt])) & 0xFFFFFFFF          # NOR
            elif fn == 0x2A:  g[rd] = 1 if self._s32(g[rs]) < self._s32(g[rt]) else 0  # SLT
            elif fn == 0x2B:  g[rd] = 1 if (g[rs] & 0xFFFFFFFF) < (g[rt] & 0xFFFFFFFF) else 0  # SLTU
        elif op == 0x01:  # REGIMM (BLTZ, BGEZ, BLTZAL, BGEZAL)
            cond = (self._s32(g[rs]) < 0) if ((rt & 1) == 0) else (self._s32(g[rs]) >= 0)
            if rt & 0x10:
                g[31] = (cur_next + 4) & 0xFFFFFFFF
            if cond:
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
        elif op == 0x02:  # J
            target_next = (cur_next & 0xF0000000) | ((ins & 0x03FFFFFF) << 2)
        elif op == 0x03:  # JAL
            g[31] = (cur_next + 4) & 0xFFFFFFFF
            target_next = (cur_next & 0xF0000000) | ((ins & 0x03FFFFFF) << 2)
        elif op == 0x04:  # BEQ
            if (g[rs] & 0xFFFFFFFF) == (g[rt] & 0xFFFFFFFF):
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
        elif op == 0x05:  # BNE
            if (g[rs] & 0xFFFFFFFF) != (g[rt] & 0xFFFFFFFF):
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
        elif op == 0x06:  # BLEZ
            if self._s32(g[rs]) <= 0:
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
        elif op == 0x07:  # BGTZ
            if self._s32(g[rs]) > 0:
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
        elif op in (0x08, 0x09): g[rt] = (g[rs] + simm) & 0xFFFFFFFF           # ADDI / ADDIU
        elif op == 0x0A:  g[rt] = 1 if self._s32(g[rs]) < simm else 0          # SLTI
        elif op == 0x0B:  g[rt] = 1 if (g[rs] & 0xFFFFFFFF) < (simm & 0xFFFFFFFF) else 0  # SLTIU
        elif op == 0x0C:  g[rt] = g[rs] & imm                                  # ANDI
        elif op == 0x0D:  g[rt] = g[rs] | imm                                  # ORI
        elif op == 0x0E:  g[rt] = g[rs] ^ imm                                  # XORI
        elif op == 0x0F:  g[rt] = (imm << 16) & 0xFFFFFFFF                     # LUI
        elif op == 0x10:  # COP0
            if rs == 0x00: g[rt] = self.cp0[rd]
            elif rs == 0x04: self.cp0[rd] = g[rt]
        elif op in (0x14, 0x15, 0x16, 0x17):  # BEQL, BNEL, BLEZL, BGTZL
            cond = (
                ((g[rs] & 0xFFFFFFFF) == (g[rt] & 0xFFFFFFFF)) if op == 0x14 else
                (((g[rs] & 0xFFFFFFFF) != (g[rt] & 0xFFFFFFFF)) if op == 0x15 else
                 ((self._s32(g[rs]) <= 0) if op == 0x16 else (self._s32(g[rs]) > 0)))
            )
            if cond:
                target_next = (cur_next + (simm << 2)) & 0xFFFFFFFF
            else:
                cur_next = (cur_next + 4) & 0xFFFFFFFF
                target_next = (cur_next + 4) & 0xFFFFFFFF
        elif op == 0x20:  v = self.read8((g[rs] + simm) & 0xFFFFFFFF); g[rt] = (v | 0xFFFFFF00) if (v & 0x80) else v
        elif op == 0x21:  v = self.read16((g[rs] + simm) & 0xFFFFFFFF); g[rt] = (v | 0xFFFF0000) if (v & 0x8000) else v
        elif op == 0x23:  g[rt] = self.read32((g[rs] + simm) & 0xFFFFFFFF)
        elif op == 0x24:  g[rt] = self.read8((g[rs] + simm) & 0xFFFFFFFF)
        elif op == 0x25:  g[rt] = self.read16((g[rs] + simm) & 0xFFFFFFFF)
        elif op == 0x28:  self.write8((g[rs] + simm) & 0xFFFFFFFF, g[rt])
        elif op == 0x29:  self.write16((g[rs] + simm) & 0xFFFFFFFF, g[rt])
        elif op == 0x2B:  self.write32((g[rs] + simm) & 0xFFFFFFFF, g[rt])

        g[0] = 0
        self.pc = cur_next
        self.next_pc = target_next

    def _raster_n64_triangles(self, fb):
        """Very small software rasterizer: pull a few 3D points from DMEM/RDRAM
        and draw filled triangles with simple depth shading. Not a full RDP."""
        # Prefer DMEM (RSP scratch) then a slice of RDRAM for vertex-like data.
        src = self.sp_dmem if any(self.sp_dmem) else self.rdram[0x1000:0x2000]
        if not any(src):
            return
        # Interpret groups of 6 bytes as (x,y,z, x,y,z) in 0-255, project to screen.
        n = min(12, len(src) // 6)
        if n < 1:
            return
        pts = []
        for i in range(n):
            b = src[i*6:i*6+6]
            x, y, z = b[0], b[1], max(1, b[2])
            # Simple perspective: divide by z, center on screen.
            sx = int(160 + (x - 128) * 80 / z)
            sy = int(120 + (y - 128) * 80 / z)
            pts.append((sx, sy, z))
        # Draw triangles between consecutive triples.
        for i in range(0, len(pts) - 2, 3):
            p0, p1, p2 = pts[i], pts[i+1], pts[i+2]
            self._fill_tri(fb, p0, p1, p2)

    def _fill_tri(self, fb, p0, p1, p2):
        # Bounding box clip
        xs = [p0[0], p1[0], p2[0]]
        ys = [p0[1], p1[1], p2[1]]
        minx, maxx = max(0, min(xs)), min(319, max(xs))
        miny, maxy = max(0, min(ys)), min(239, max(ys))
        if maxx <= minx or maxy <= miny:
            return
        # Depth shade from average z
        z = (p0[2] + p1[2] + p2[2]) // 3
        shade = max(40, 255 - z)
        col = (shade, shade // 2, 255 - shade // 3, 255)
        # Edge function fill (barycentric-ish)
        def edge(ax, ay, bx, by, cx, cy):
            return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)
        area = edge(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
        if area == 0:
            return
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                w0 = edge(p1[0], p1[1], p2[0], p2[1], x, y)
                w1 = edge(p2[0], p2[1], p0[0], p0[1], x, y)
                w2 = edge(p0[0], p0[1], p1[0], p1[1], x, y)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    fb[y, x] = col

    def _render_frame(self):
        fb_origin = self.vi_regs[1] & 0x007FFFFE
        vi_ctrl = self.vi_regs[0]
        bpp32 = (vi_ctrl & 3) == 3
        need = 320 * 240 * (4 if bpp32 else 2)
        fb = self.frame_rgba
        if fb_origin + need <= len(self.rdram):
            chunk = self.rdram[fb_origin:fb_origin + need]
            if bpp32:
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape((240, 320, 4))
                row_any = arr[:, :, :3].any(axis=2)
            else:
                px16 = np.frombuffer(chunk, dtype=">u2").reshape((240, 320))
                row_any = (px16 != 0).any(axis=1)[:, None]
            if row_any.all():
                if bpp32:
                    self.frame_rgba[:] = arr
                    self.frame_rgba[:, :, 3] = 255
                else:
                    self.frame_rgba[:, :, 0] = (((px16 >> 11) & 0x1F) << 3) | (((px16 >> 11) & 0x1F) >> 2)
                    self.frame_rgba[:, :, 1] = (((px16 >> 6) & 0x1F) << 3) | (((px16 >> 6) & 0x1F) >> 2)
                    self.frame_rgba[:, :, 2] = (((px16 >> 1) & 0x1F) << 3) | (((px16 >> 1) & 0x1F) >> 2)
                    self.frame_rgba[:, :, 3] = 255
                return
            if row_any.any():
                # Partial framebuffer: clear the empty rows, rasterize triangles
                # into them, then overlay the VI rows the game did write.
                empty = ~row_any[:, 0] if row_any.shape[1] == 1 else ~row_any.any(axis=1)
                if bpp32:
                    vi_rgb = arr
                else:
                    vi_rgb = np.empty((240, 320, 3), dtype=np.uint8)
                    vi_rgb[:, :, 0] = (((px16 >> 11) & 0x1F) << 3) | (((px16 >> 11) & 0x1F) >> 2)
                    vi_rgb[:, :, 1] = (((px16 >> 6) & 0x1F) << 3) | (((px16 >> 6) & 0x1F) >> 2)
                    vi_rgb[:, :, 2] = (((px16 >> 1) & 0x1F) << 3) | (((px16 >> 1) & 0x1F) >> 2)
                fb[empty] = 0
                self._raster_n64_triangles(fb)
                if not (fb[empty][:, :, :3].any()):
                    yy = np.arange(240, dtype=np.int32)[:, None]
                    grad = np.stack([
                        np.clip(12 + (yy >> 4), 0, 40).repeat(320, 1),
                        np.clip(16 + (np.arange(320, dtype=np.int32)[None, :] >> 5), 0, 40).repeat(240, 0),
                        np.clip(32 + (yy >> 3), 0, 64).repeat(320, 1),
                    ], axis=2).astype(np.uint8)
                    still_empty = ~fb[:, :, :3].any(axis=2)
                    fb[still_empty] = grad[still_empty]
                fb[:, :, :3][row_any] = vi_rgb[row_any]
                fb[:, :, 3] = 255
                return

        # 3D fallback: rasterize triangles from RDRAM/DMEM when VI has no image
        self._raster_n64_triangles(fb)
        if fb[:, :, :3].any():
            return
        # Last resort: paint a dim gradient so the window is never pure black
        yy = np.arange(240, dtype=np.int32)[:, None]
        xx = np.arange(320, dtype=np.int32)[None, :]
        fb[:, :, 0] = np.clip(12 + (yy >> 4), 0, 40).astype(np.uint8)
        fb[:, :, 1] = np.clip(16 + (xx >> 5), 0, 40).astype(np.uint8)
        fb[:, :, 2] = np.clip(32 + (yy >> 3), 0, 64).astype(np.uint8)
        fb[:, :, 3] = 255
        return
        # Render N64 ROM Header + RSP/RDP Status Banner when homebrew/ROM has not written VI framebuffer yet
        yy = np.arange(240, dtype=np.int32)[:, None]
        xx = np.arange(320, dtype=np.int32)[None, :]
        fb[:, :, 0] = np.clip(18 + (yy >> 3), 0, 255).astype(np.uint8)
        fb[:, :, 1] = np.clip(24 + (xx >> 4), 0, 255).astype(np.uint8)
        fb[:, :, 2] = np.clip(58 + (yy >> 2), 0, 255).astype(np.uint8)
        fb[:, :, 3] = 255
        # Draw classic 4-color N64 'N' emblem in center
        cx, cy = 160, 110
        fb[cy - 28:cy + 28, cx - 28:cx - 16] = (220, 48, 48, 255)   # Red left pillar
        fb[cy - 28:cy + 28, cx + 16:cx + 28] = (48, 185, 72, 255)   # Green right pillar
        for d in range(36):
            px = cx - 18 + d
            py = cy - 22 + int(d * 1.2)
            fb[py:py + 10, px:px + 6] = (48, 110, 235, 255)          # Blue diagonal
        fb[cy + 24:cy + 28, cx - 28:cx + 28] = (242, 200, 42, 255)  # Gold base bar

    def capture_state(self):
        return {
            "gpr": list(self.gpr), "cp0": list(self.cp0), "hi": self.hi, "lo": self.lo,
            "pc": self.pc, "next_pc": self.next_pc,
            "vi": list(self.vi_regs), "pi": list(self.pi_regs),
            "ai": list(self.ai_regs), "si": list(self.si_regs), "mi": list(self.mi_regs),
            "rdram_z": zlib.compress(bytes(self.rdram), 1),
            "sp_dmem": bytes(self.sp_dmem), "sp_imem": bytes(self.sp_imem),
            "pif_ram": bytes(self.pif_ram),
            "buttons": self.buttons, "frames": self.frames,
            "frame_rgba": self.frame_rgba.copy(),
        }

    def restore_state(self, st):
        self.gpr[:] = st["gpr"]
        self.cp0[:] = st["cp0"]
        self.hi = st["hi"]
        self.lo = st["lo"]
        self.pc = st["pc"]
        self.next_pc = st["next_pc"]
        self.vi_regs[:] = st["vi"]
        if "pi" in st:
            self.pi_regs[:] = st["pi"]
        if "ai" in st:
            self.ai_regs[:] = st["ai"]
        if "si" in st:
            self.si_regs[:] = st["si"]
        if "mi" in st:
            self.mi_regs[:] = st["mi"]
        if "rdram_z" in st:
            self.rdram[:] = zlib.decompress(st["rdram_z"])
        elif "rdram" in st:
            self.rdram[:] = st["rdram"]
        if "sp_dmem" in st:
            self.sp_dmem[:] = st["sp_dmem"]
        if "sp_imem" in st:
            self.sp_imem[:] = st["sp_imem"]
        if "pif_ram" in st:
            self.pif_ram[:] = st["pif_ram"]
        self.buttons = st.get("buttons", 0)
        self.frames = st.get("frames", self.frames)
        if "frame_rgba" in st:
            self.frame_rgba[:] = st["frame_rgba"]

    def save_state(self, path=None):
        path = path or self.base_path + ".n64state"
        with open(path, "wb") as f:
            pickle.dump(self.capture_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path=None):
        path = path or self.base_path + ".n64state"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.restore_state(pickle.load(f))
        return True

    def run_frame(self, draw=True):
        """Run one full NTSC frame of VR4300 cycles (93.75 MHz / 60 Hz).

        STOPPING after a few thousand instructions left games frozen between
        vblanks; a real frame is ~1.5M cycles. Each executed instruction
        retires ``cycles_per_instruction`` cycles (interpreted cores retire
        far fewer per wall-clock ms than silicon, so a higher credit keeps
        spin loops short and games at interactive speed without breaking
        per-frame pacing).
        """
        step = self.step
        budget = getattr(self, "cycles_per_frame", 1562500)
        cpi = getattr(self, "cycles_per_instruction", 16)
        cycles = 0
        budget += cpi
        while cycles < budget:
            step()
            cycles += cpi
        if draw:
            self._render_frame()
        self.apu.run_frame()
        self.frames += 1

    def get_rgba(self):
        return self.frame_rgba


def load_console(rom_path, sample_rate=44100):
    ext = os.path.splitext(rom_path)[1].lower()
    if ext in (".fds", ".fam", ".unf"):
        return Famicom(rom_path, sample_rate)
    if ext in (".z64", ".n64", ".v64"):
        return Nintendo64(rom_path, sample_rate)
    if ext in (".sfc", ".smc", ".snes"):
        # Built-in vectorized SNES. For higher accuracy (full color math, windows, mosaic,
        # more complete HDMA, etc.) you can integrate https://github.com/eduardovra/PySNES
        # (MIT). Example future path:
        # try:
        #     from pysnes.pysnes import PySNES
        #     return PySNESAdapter(rom_path, sample_rate)  # wrapper implementing width/height/run_frame/get_rgba/...
        # except ImportError:
        #     pass
        return SNES(rom_path, sample_rate)
    if ext in (".gb", ".gbc"):
        return GameBoy(rom_path, sample_rate)
    if ext == ".gba":
        return GameBoyAdvance(rom_path, sample_rate)
    if ext == ".nes":
        return NES(rom_path, sample_rate)
    with open(rom_path, "rb") as f:
        hdr = f.read(512)
    if hdr[:4] == b"FDS\x1a":
        return Famicom(rom_path, sample_rate)
    if hdr[:4] in (b"\x80\x37\x12\x40", b"\x37\x80\x40\x12", b"\x40\x12\x37\x80"):
        return Nintendo64(rom_path, sample_rate)
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
    ("Ctrl+O", "Open ROM (NES/Famicom/SNES/N64/GB/GBC/GBA)"), ("Esc", "Close menu / quit"),
]
ABOUT_ROWS = [
    ("PyRetro", "Multi-Console Emulator in Python + Arcade"),
    ("NES / SNES", "6502 + 2C02 PPU / 65c816 + Vectorized S-PPU"),
    ("Famicom / N64", "FDS Disk System / MIPS VR4300i + VI/RDP"),
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
                 vsync=True, audio_latency=0.075, audio_backend="auto"):
        self.console = console
        self.settings = Settings()
        self.settings.apply_to(console)
        self.aspect = aspect
        self.par = 8.0 / 7.0 if aspect == "ntsc" else 1.0
        self.max_catchup = max_catchup
        cw = console.width if console else 256
        ch = console.height if console else 240
        sys_tag = f"[{console.system_name}] " if console else ""
        self.title_base = f"PyRetro {sys_tag}" + ("- " + os.path.basename(console.rom_path) if console else "")
        super().__init__(int(cw * scale * self.par), ch * scale + MENU_H, self.title_base,
                         resizable=True, update_rate=(1.0 / 60.0 if vsync else 1.0 / 120.0), vsync=vsync)
        self.use_vsync = vsync
        try:
            self.set_vsync(self.use_vsync)
            self.set_update_rate(1.0 / 60.0 if self.use_vsync else 1.0 / 120.0)
        except Exception:
            pass
        self.keymap = _keymap()
        self.keys = 0
        self.paused = False
        self.muted = False
        self.turbo = False
        self.turbo_hold = False
        self.turbo_latch = False
        self.smooth = False
        self.volume = 1.0
        self.frame_ready = (console is not None)
        self.acc = 0.0
        self.last = time.perf_counter()
        self.fps_t = self.last
        self.fps_n = 0
        self.slow_secs = 0
        self.fill_avg = 0.0
        self.audio_target = max(0.035, min(0.120, float(audio_latency)))
        self._rewind_tick = 0
        self.speed_multiplier = 1.0
        self.reverse_speed = False
        # N64 stores 8 MB RDRAM per entry; keep a short history to stay under ~0.5 GB.
        # Other systems stay at 600 frames (~10 s).
        self.rewind_history = deque(maxlen=self._rewind_maxlen_for(console))
        self.rewind_acc = 0.0
        self._audio_capture = []
        self._forward_audio_phase = 0.0
        self._forward_audio_tail = None
        self._forward_audio_speed = self.speed_multiplier
        self.ui_hover_t = {}
        self.ui_anim_last = time.perf_counter()
        self.ui_frame_dt = 0.0
        self.ui_modal_close = None
        self._overlay_drag = None
        self._overlay_off = [0.0, 0.0]
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
        self.ui_path = []  # hovered parent chain: [item, ...], ui_path[k] opened submenu k+1
        self.ui_overlay = None
        self.ui_toast = None
        self.ui_texts = {}
        self.ui_menus = None
        self.ui_title_x = []
        self.ui_drop_w = []
        self.ui_menu_scroll = 0.0   # pixels scrolled inside open dropdown
        self.ui_overlay_scroll = 0.0  # pixels scrolled inside Help/About modal
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

    @staticmethod
    def _rewind_maxlen_for(console):
        """N64 RDRAM is 8 MB; 60 frames ≈ 480 MB max. Other systems stay at 600."""
        if console is not None and getattr(console, "system_name", "") == "N64":
            return 60
        return 600

    def _reset_rewind_history(self):
        maxlen = self._rewind_maxlen_for(self.console)
        if self.rewind_history.maxlen != maxlen:
            self.rewind_history = deque(maxlen=maxlen)
        else:
            self.rewind_history.clear()

    def ui_build_menus(self):
        def it(label, fn=None, key="", check=None, sub=None):
            return {"label": label, "fn": fn, "key": key, "check": check, "sub": sub}

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
        game_m = [
            it("Pause", self.act_pause, "P", check=lambda: self.paused),
            it("Reset", self.act_reset, "R"),
            it("Power Cycle", self.act_reset),
            None,
            it("Fast Forward", self.act_turbo, "Tab", check=lambda: self.turbo),
            it("Rewind", self.act_reverse, "Backslash", check=lambda: self.reverse_speed),
        ]
        # Settings matches the MesenCE menu: Speed, Video Size, Filter, Aspect,
        # Region, then Audio/Emulation/Input/Video, then per-system, then Preferences.
        # Settings is a nested menu tree: hovering a parent row opens its
        # submenu to the right (Speed → 0.1x..10x, Audio → Mute/Volume/Latency,
        # per-system groups, ...), like the reference admin menus.
        settings_m = [
            it("Speed", sub=[
                it(f"{v:g}x", lambda v=v: self.act_speed(v),
                   check=lambda v=v: abs(self.speed_multiplier - v) < 1e-6)
                for v in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
            ]),
            it("Video Size", sub=[size_item(s) for s in (1, 2, 3, 4, 5, 6)]),
            it("Video Filter", sub=[
                it("None", lambda: self.act_smooth() if self.smooth else None,
                   check=lambda: not self.smooth),
                it("Smooth", lambda: self.act_smooth() if not self.smooth else None,
                   key="F", check=lambda: self.smooth),
                it("NTSC composite", lambda: self.ui_notify("NTSC composite filter: brush TODO")),
            ]),
            it("Aspect Ratio", sub=[
                it("Square Pixels", lambda: self.act_aspect("square"),
                   check=lambda: self.aspect == "square"),
                it("NTSC 8:7", lambda: self.act_aspect("ntsc"),
                   check=lambda: self.aspect == "ntsc"),
                it("4:3", lambda: self.act_aspect("4:3"),
                   check=lambda: self.aspect == "4:3"),
            ]),
            it("Region / Model", sub=[
                it("Auto", lambda: self.ui_notify("Region: Auto")),
                it("NTSC", lambda: self.ui_notify("Region: NTSC (60 Hz)")),
                it("PAL", lambda: self.ui_notify("Region: PAL (50 Hz)")),
                it("Dendy", lambda: self.ui_notify("Region: Dendy (50 Hz)")),
            ]),
            None,
            it("Audio", sub=[
                it("Mute", self.act_mute, key="M", check=lambda: self.muted),
                it("Volume", sub=[
                    it(f"{p}%", lambda p=p: self.act_volume(p / 100.0),
                       check=lambda p=p: abs(self.volume - p / 100.0) < 0.01)
                    for p in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
                ]),
                it("Latency", sub=[
                    it(f"{ms} ms", lambda ms=ms: self.act_audio_latency(ms),
                       check=lambda ms=ms: int(self.audio_target * 1000) == ms)
                    for ms in (20, 40, 60, 80, 100)
                ]),
                None,
                it("Stereo effect", None),
            ]),
            it("Video", sub=[
                it("Fullscreen", self.act_fullscreen, key="F11", check=lambda: self.fullscreen),
                it("VSync", self.act_vsync, check=lambda: self.use_vsync),
                it("Smooth filtering", self.act_smooth, key="F",
                   check=lambda: self.smooth),
                None,
                it(f"Backend: {self.settings.video_backend}", None),
            ]),
            it("Emulation", sub=[
                it("Pause", self.act_pause, key="P", check=lambda: self.paused),
                it("Fast Forward", self.act_turbo, key="Tab", check=lambda: self.turbo),
                it("Rewind", self.act_reverse, check=lambda: self.reverse_speed),
            ]),
            it("Input", sub=[
                it("Keyboard", None),
                it("Joypad mapping", lambda: self.ui_show("Controls", HELP_ROWS)),
            ]),
            None,
            it("NES", sub=[
                it("Remove sprite limit", lambda: setattr(self.settings, "nes_remove_sprite_limit",
                                                          not self.settings.nes_remove_sprite_limit),
                   check=lambda: self.settings.nes_remove_sprite_limit),
                it("Disable background", lambda: setattr(self.settings, "nes_disable_bg",
                                                          not self.settings.nes_disable_bg),
                   check=lambda: self.settings.nes_disable_bg),
                it("Disable sprites", lambda: setattr(self.settings, "nes_disable_sprites",
                                                       not self.settings.nes_disable_sprites),
                   check=lambda: self.settings.nes_disable_sprites),
            ]),
            it("SNES", sub=[
                it(f"Power-on RAM: {self.settings.snes_poweron_ram}", None),
                it(f"SPC clock: {self.settings.snes_spc_clock}%", None),
                it("Strict boards", lambda: setattr(self.settings, "snes_strict_boards",
                                                     not self.settings.snes_strict_boards),
                   check=lambda: self.settings.snes_strict_boards),
            ]),
            it("Game Boy", sub=[
                it("Model: auto", None),
                None,
                it("Boot ROM", None),
                it("SGB border", None),
            ]),
            it("GBA", sub=[
                it("BIOS: HLE", None),
                it("RTC: on", None),
            ]),
            it("Other systems", sub=[
                it("N64 - VI + software raster", None),
                it("GBA - ARM7 + DirectSound", None),
                it("FDS - disk loader", None),
                None,
                it("PC Engine / SMS / WonderSwan: not yet", None),
            ]),
            None,
            it("Preferences", sub=[
                it("Show FPS", lambda: setattr(self.settings, "show_fps", not self.settings.show_fps),
                   check=lambda: self.settings.show_fps),
                it("Show frame counter", lambda: setattr(self.settings, "show_frame_counter", not self.settings.show_frame_counter),
                   check=lambda: self.settings.show_frame_counter),
                it("Show game timer", lambda: setattr(self.settings, "show_game_timer", not self.settings.show_game_timer),
                   check=lambda: self.settings.show_game_timer),
                it("Show lag counter", lambda: setattr(self.settings, "show_lag_counter", not self.settings.show_lag_counter),
                   check=lambda: self.settings.show_lag_counter),
                it("Hide on-screen messages", lambda: setattr(self.settings, "hide_osd", not self.settings.hide_osd),
                   check=lambda: self.settings.hide_osd),
                None,
                it(f"Autosave: {self.settings.autosave_minutes} min", None),
                it(f"Rewind memory: {self.settings.rewind_mb} MB", None),
            ]),
        ]
        tools_m = [
            it("Screenshot", self.act_screenshot, "F12"),
            it("Record AVI", self.act_toggle_record, "V", check=lambda: self.recording),
        ]
        debug_m = [
            it("Show debug info", lambda: self.ui_notify(
                f"PC debug: frames={getattr(self.console, 'frames', 0)}"),
               check=lambda: self.settings.show_debug),
        ]
        help_m = [
            it("Controls...", lambda: self.ui_show("Controls", HELP_ROWS)),
            it("About", lambda: self.ui_show("About", ABOUT_ROWS)),
        ]
        return [("File", file_m), ("Game", game_m), ("Settings", settings_m),
                ("Tools", tools_m), ("Debug", debug_m), ("Help", help_m)]

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
        self._last_overlay = (title, rows)
        self.ui_overlay_scroll = 0.0

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
        self._reset_rewind_history()
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
            self._reset_rewind_history()
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
            self.set_update_rate(1.0 / 60.0 if self.use_vsync else 1.0 / 120.0)
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
                self.ui_path = []
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
        """Return (rows, content_bottom, max_scroll).
        rows are (item, y_top, y_bottom) already shifted by ui_menu_scroll and
        clipped so the dropdown never extends past the bottom of the window.
        """
        max_drop_h = max(80, wh - MENU_H - 8)
        y = wh - MENU_H
        rows = []
        for item in self.ui_menus[i][1]:
            h = SEP_H if item is None else ITEM_H
            rows.append((item, y, y - h))
            y -= h
        content_h = (wh - MENU_H) - y
        max_scroll = max(0.0, content_h - max_drop_h)
        scroll = max(0.0, min(self.ui_menu_scroll, max_scroll))
        self.ui_menu_scroll = scroll
        # Shift all rows up by scroll amount; drop area ends at max_drop_h below the bar
        vis_bottom = wh - MENU_H - max_drop_h
        shifted = []
        for item, yt, yb in rows:
            nyt = yt + scroll
            nyb = yb + scroll
            # Keep only rows that intersect the visible drop region
            if nyt > wh - MENU_H or nyb < vis_bottom:
                continue
            shifted.append((item, nyt, nyb))
        return shifted, vis_bottom, max_scroll

    def _ui_sub_width(self, item):
        """Panel width for a submenu (cached on the item dict)."""
        w = item.get("_sw")
        if w is None:
            subs = [s for s in item["sub"] if s]
            lw = max([self.ui_text(s["label"], C_TEXT).content_width for s in subs] or [0])
            kw = max([self.ui_text(s["key"], C_DIM).content_width for s in subs if s["key"]] or [0])
            w = CHECK_W + lw + (36 + kw if kw else 0) + PAD_X + 16
            item["_sw"] = w
        return w

    def _ui_panel_chain(self, i, path, wh):
        """Geometry for the root dropdown plus the open submenu chain above
        `path` (ui_path[k] is the row that opened panel k+1). Returns a list of
        {"x0", "w", "rows": [(item, yt, yb)], "clip_top", "y_top", "y_bot"} —
        animated and clipped exactly like ui_draw_menu paints them."""
        alpha = max(0.0, min(1.0, getattr(self, "_drop_anim_t", 1.0)))
        if alpha <= 0.01:
            return []
        ww, _ = self.get_size()
        x0 = self.ui_title_x[i][0]
        w = self.ui_drop_w[i]
        scale_w = w * (0.94 + 0.06 * alpha)
        slide_y = (1.0 - alpha) * 6.0
        rows, vis_bottom, _ = self.ui_rows(i, wh)
        max_drop_h = max(80, wh - MENU_H - 8)
        open_h = (wh - MENU_H) - vis_bottom
        cur_y_end = (wh - MENU_H - 2) - open_h * alpha
        arows = [(item, yt + slide_y, yb + slide_y) for item, yt, yb in rows
                 if yb + slide_y >= cur_y_end]
        geos = [{"x0": x0, "w": scale_w, "rows": arows,
                 "clip_top": cur_y_end, "y_top": wh - MENU_H, "y_bot": vis_bottom, "pop": 1.0}]
        ref_geo, ref_rows = geos[0], arows
        for idx, parent in enumerate(path):
            rect = next(((yt, yb) for (p, yt, yb) in ref_rows if p is parent), None)
            if rect is None or not parent.get("sub"):
                break
            yt_p = rect[0]
            st = self.ui_hover_t.get(("sub_pop", idx + 1))
            pop = max(0.0, min(1.0, st[0])) if st is not None else 0.0
            if pop <= 0.01:
                break
            slide = (1.0 - pop) * 6.0
            sw = self._ui_sub_width(parent) * (0.90 + 0.10 * pop)
            sx0 = ref_geo["x0"] + ref_geo["w"] - 1
            if sx0 + sw > ww:
                sx0 = ref_geo["x0"] - sw + 1
            total_h = sum(SEP_H if s is None else ITEM_H for s in parent["sub"])
            top = min(max(yt_p, vis_bottom), max(vis_bottom, wh - MENU_H - total_h))
            bot = top - total_h
            srows = []
            y = top
            for s in parent["sub"]:
                h = SEP_H if s is None else ITEM_H
                srows.append((s, y + slide, y - h + slide))
                y -= h
            geos.append({"x0": sx0, "w": sw, "rows": srows,
                         "clip_top": vis_bottom, "y_top": top, "y_bot": bot + slide,
                         "pop": pop})
            ref_geo, ref_rows = geos[-1], srows
        return geos

    @staticmethod
    def _ui_in_panel(geo, x, y):
        return geo["x0"] <= x < geo["x0"] + geo["w"] and geo["y_bot"] <= y <= geo["y_top"]

    def _ui_row_at(self, geo, x, y):
        """Item under (x, y) inside a panel geometry, or None."""
        if not self._ui_in_panel(geo, x, y):
            return None
        for item, yt, yb in geo["rows"]:
            if item is not None and yb <= y < yt:
                return item
        return None

    def ui_update_hover(self, x, y, wh):
        """Recompute the open submenu chain for the current pointer position."""
        i = self.ui_open
        if i is None:
            self.ui_path = []
            return
        path = list(self.ui_path)
        geos = self._ui_panel_chain(i, path, wh)
        for depth in reversed(range(len(geos))):
            geo = geos[depth]
            if self._ui_in_panel(geo, x, y):
                item = self._ui_row_at(geo, x, y)
                if item is not None and item.get("sub"):
                    self.ui_path = path[:depth] + [item]
                else:
                    self.ui_path = path[:depth]
                return
        self.ui_path = []

    def ui_hit_item(self, x, y, wh):
        """Item under the pointer across the root dropdown and any open
        nested submenus (deepest panel wins)."""
        i = self.ui_open
        if i is None:
            return None
        geos = self._ui_panel_chain(i, list(getattr(self, "ui_path", [])), wh)
        for depth in reversed(range(len(geos))):
            item = self._ui_row_at(geos[depth], x, y)
            if item is not None:
                return item
        return None

    def on_mouse_motion(self, x, y, dx, dy):
        self.ui_ptr = (x, y)
        if getattr(self, "_overlay_drag", None) is not None:
            ox, oy = self._overlay_drag
            self._overlay_off[0] = x - ox
            self._overlay_off[1] = y - oy
            return
        if self.ui_open is not None:
            t = self.ui_hit_title(x, y, self.get_size()[1])
            if t is not None and t != self.ui_open:
                self.ui_open = t
                self.ui_path = []
                self.ui_menu_scroll = 0.0
            else:
                self.ui_update_hover(x, y, self.get_size()[1])

    def on_mouse_leave(self, x, y):
        self.ui_ptr = (-1, -1)

    def _overlay_select(self, label, value):
        """Act on a clicked popup row."""
        lab = str(label).strip()
        # Speed like "1.0x" or "2x"
        if lab.endswith("x") and lab[:-1].replace(".", "", 1).isdigit():
            try:
                self.act_speed(float(lab[:-1]))
                self.ui_notify(f"Speed {lab}")
                return
            except Exception:
                pass
        # Scale like "3x" already handled above; size rows are "1x".."6x"
        if lab in ("Square Pixels", "NTSC 8:7", "4:3"):
            self.act_aspect({"Square Pixels": "square", "NTSC 8:7": "ntsc", "4:3": "square"}.get(lab, "square"))
            self.ui_notify(f"Aspect {lab}")
            return
        if lab in ("None", "Smooth", "NTSC"):
            if lab == "Smooth":
                self.act_smooth()
            self.ui_notify(f"Filter {lab}")
            return
        if lab in ("Auto", "NTSC", "PAL", "Dendy"):
            self.ui_notify(f"Region {lab}")
            return
        # Toggle-looking rows
        if str(value).lower() in ("on", "off", "true", "false", "yes", "no"):
            self.ui_notify(f"{lab}: {value}")
            return
        self.ui_notify(f"Selected {lab}")

    def on_mouse_release(self, x, y, button, modifiers):
        self._overlay_drag = None

    def on_mouse_press(self, x, y, button, modifiers):
        self.ui_ptr = (x, y)
        if self.ui_overlay is not None:
            # Close button (generous hit area); its expanded rect is excluded
            # from the title-bar drag region so a close click never drags.
            cb = self.ui_modal_close
            if cb and (cb[0] - 6) <= x <= (cb[2] + 6) and (cb[1] - 6) <= y <= (cb[3] + 6):
                self.ui_overlay = None
                self.ui_modal_close = None
                self._overlay_drag = None
                return
            # Selectable row
            for x0, y0, x1, y1, label, value in getattr(self, "_overlay_row_hits", []):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._overlay_select(label, value)
                    return
            # Drag by title bar
            tb = getattr(self, "_overlay_titlebar", None)
            if (tb and tb[0] <= x <= tb[2] and tb[1] <= y <= tb[3]
                    and not (cb and (cb[0] - 6) <= x <= (cb[2] + 6) and (cb[1] - 6) <= y <= (cb[3] + 6))):
                self._overlay_drag = (x - self._overlay_off[0], y - self._overlay_off[1])
                return
            return
        wh = self.get_size()[1]
        if button != arcade.MOUSE_BUTTON_LEFT:
            self.ui_open = None
            self.ui_path = []
            return
        t = self.ui_hit_title(x, y, wh)
        if t is not None:
            if self.ui_open == t:
                self.ui_open = None
                self.ui_path = []
            else:
                self.ui_open = t
                self.ui_path = []
                self.ui_menu_scroll = 0.0
            return
        if self.ui_open is not None:
            self.ui_update_hover(x, y, wh)
            item = self.ui_hit_item(x, y, wh)
            keep_open = item is not None and item.get("sub") and not item.get("fn")
            if not keep_open:
                self.ui_open = None
                self.ui_path = []
            if item is not None and item["fn"]:
                try:
                    item["fn"]()
                except Exception as e:
                    self.ui_notify(f"{item['label']} failed: {e}")

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Scroll open dropdown menus and Help/About overlays with the mouse wheel."""
        wh = self.get_size()[1]
        delta = scroll_y * 28.0
        if self.ui_overlay is not None:
            cap = getattr(self, "_overlay_max_scroll", 4000.0)
            self.ui_overlay_scroll = max(0.0, min(cap, self.ui_overlay_scroll + delta))
            return
        if self.ui_open is not None and self.ui_menus:
            items = self.ui_menus[self.ui_open][1]
            content_h = sum(SEP_H if it is None else ITEM_H for it in items)
            max_drop_h = max(80, wh - MENU_H - 8)
            max_scroll = max(0.0, content_h - max_drop_h)
            self.ui_menu_scroll = max(0.0, min(max_scroll, self.ui_menu_scroll + delta))
            return

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
                self._rewind_tick = (self._rewind_tick + 1) % 3
                if not self.turbo and self._rewind_tick == 0:
                    rewind_entry = {"state": self.console.capture_state(), "audio": np.empty(0, dtype=np.int16)}
                    self.rewind_history.append(rewind_entry)
                self._audio_capture = []
                self.console.run_frame(draw=not more)
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
            self._last_fps = fps
            if fps < 59.0:
                self._lag_frames = getattr(self, "_lag_frames", 0) + 1
            self.set_caption(f"{self.title_base} - {fps:.1f} FPS")
            if self.use_vsync and not self.turbo and not self._opening and fps < 45.0:
                self.slow_secs += 1
                if self.slow_secs >= 5:
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

    def ui_tween(self, key, target, dt, duration=0.22, overshoot=False):
        """Smooth ease-out-cubic (with optional overshoot) for menus."""
        state = self.ui_hover_t.get(key)
        if state is None:
            state = [0.0, 0.0, 0.0, float(target)]
        value, start, elapsed, old_target = state
        if float(target) != old_target:
            start, elapsed, old_target = value, 0.0, float(target)
        elapsed = min(duration, elapsed + max(0.0, dt))
        t = min(1.0, elapsed / max(0.001, duration))
        if overshoot and old_target > start:
            # Ease-out-back: slight overshoot then settle
            c1, c3 = 1.4, 2.4
            eased = 1.0 + c3 * ((t - 1.0) ** 3) + c1 * ((t - 1.0) ** 2)
        else:
            # Ease-out-cubic: fast start, gentle settle
            eased = 1.0 - (1.0 - t) ** 3
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
        anim_dt = self.ui_frame_dt

        bar_vis = 1.0 if self.ui_bar_visible(wh) else 0.0
        bar_t = self.ui_tween("bar_vis", bar_vis, anim_dt, 0.18)
        if bar_t <= 0.01:
            return

        px, py = self.ui_ptr
        self.ui_rect(0, wh - MENU_H * bar_t, ww, wh, (C_BAR[0], C_BAR[1], C_BAR[2], bar_t))

        for i, (name, _) in enumerate(self.ui_menus):
            x0, x1 = self.ui_title_x[i]
            hovered = self.ui_open == i or (x0 <= px < x1 and py >= wh - MENU_H)
            active = self.ui_open == i
            amount = self.ui_tween(("title", i), 1.0 if hovered else 0.0, anim_dt, 0.16)
            press_t = self.ui_tween(("title_press", i), 1.0 if active else 0.0, anim_dt, 0.12)
            if amount > 0.01:
                inset = (1.0 - amount) * 3.0
                self.ui_rect(x0 + inset, wh - MENU_H + 1, x1 - inset, wh - 1,
                             (C_BAR[0] + (C_HOT[0] - C_BAR[0]) * amount,
                              C_BAR[1] + (C_HOT[1] - C_BAR[1]) * amount,
                              C_BAR[2] + (C_HOT[2] - C_BAR[2]) * amount, bar_t))
                bw = (x1 - x0) * 0.48 * amount
                cx = (x0 + x1) * 0.5
                self.ui_rect(cx - bw, wh - MENU_H, cx + bw, wh - MENU_H + 2,
                             (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], amount * bar_t))
            self.ui_put(self.ui_text(name, C_TEXT), x0 + PAD_X, wh - MENU_H / 2 + amount * 1.4 - press_t * 1.1)

        if self.ui_open is not None:
            self._last_open_menu = self.ui_open
        drop_target = 1.0 if self.ui_open is not None else 0.0
        drop_t = self.ui_tween("dropdown_pop", drop_target, anim_dt, 0.26, overshoot=(drop_target > 0.5))
        self._drop_anim_t = drop_t
        i = self.ui_open if self.ui_open is not None else getattr(self, "_last_open_menu", None)
        if i is None or drop_t <= 0.01:
            return

        alpha = max(0.0, min(1.0, drop_t))
        x0 = self.ui_title_x[i][0]
        w = self.ui_drop_w[i]
        rows, vis_bottom, max_scroll = self.ui_rows(i, wh)
        # Open downward from just under the top menu bar
        open_h = (wh - MENU_H) - vis_bottom
        cur_y_end = (wh - MENU_H - 2) - open_h * alpha
        slide_y = (1.0 - alpha) * 6.0
        scale_w = w * (0.94 + 0.06 * alpha)

        self.ui_rect(x0 - 1, cur_y_end - 1, x0 + scale_w + 1, wh - MENU_H,
                     (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
        self.ui_rect(x0, cur_y_end, x0 + scale_w, wh - MENU_H,
                     (C_DROP[0], C_DROP[1], C_DROP[2], 0.97 * alpha))

        for row_index, (item, yt, yb) in enumerate(rows):
            if yb + slide_y < cur_y_end:
                continue
            if item is None:
                ym = (yt + yb) / 2 + slide_y
                self.ui_rect(x0 + 8, ym, x0 + scale_w - 8, ym + 1, (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
                continue
            chain_parent = (self.ui_open is not None) and bool(self.ui_path) and self.ui_path[0] is item
            ptr_hover = x0 <= px < x0 + scale_w and yb + slide_y <= py < yt + slide_y
            hovered = (self.ui_open is not None) and (chain_parent or ptr_hover)
            amount = self.ui_tween(("item", i, row_index), 1.0 if hovered else 0.0, anim_dt, 0.18)
            if amount > 0.01:
                inset = (1.0 - amount) * 5.0
                self.ui_rect(x0 + inset, yb + 1 + slide_y, x0 + scale_w - inset, yt - 1 + slide_y,
                             (C_DROP[0] + (C_HOT[0] - C_DROP[0]) * amount,
                              C_DROP[1] + (C_HOT[1] - C_DROP[1]) * amount,
                              C_DROP[2] + (C_HOT[2] - C_DROP[2]) * amount, alpha))
                if amount > 0.08:
                    self.ui_rect(x0 + 2, yb + 3 + slide_y, x0 + 4.5, yt - 3 + slide_y,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], amount * alpha))
            cy = (yt + yb) / 2 + amount * 1.5 + slide_y
            has_check = item.get("check") is not None
            checked = has_check and item["check"]()
            chk_t = self.ui_tween(("chk", i, row_index), 1.0 if checked else 0.0, anim_dt, 0.16, overshoot=checked)
            if has_check:
                # Always draw the box outline; fill when checked.
                self.ui_rect(x0 + 7, cy - 5, x0 + 17, cy + 5,
                             (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
                if chk_t > 0.01:
                    r_sz = 3.0 * max(0.0, chk_t)
                    self.ui_rect(x0 + 12 - r_sz, cy - r_sz, x0 + 12 + r_sz, cy + r_sz,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], min(1.0, chk_t) * alpha))
            if alpha > 0.35:
                self.ui_put(self.ui_text(item["label"], C_TEXT), x0 + CHECK_W + amount * 3.5, cy)
                if item["key"]:
                    self.ui_put(self.ui_text(item["key"], C_DIM, anchor_x="right"), x0 + scale_w - PAD_X, cy)
            if item.get("sub"):
                self._ui_arrow(x0 + scale_w, cy, amount, alpha)

        # Nested submenu chain (drawn last so deeper panels sit on top)
        if self.ui_open is not None:
            # Tick per-depth pop tweens; reset when the open chain changes so
            # newly opened panels pop in instead of appearing fully grown.
            sig = tuple(id(it) for it in self.ui_path)
            if sig != getattr(self, "_sub_sig", None):
                prev = getattr(self, "_sub_sig", ())
                self._sub_sig = sig
                n = 0
                while n < len(sig) and n < len(prev) and sig[n] == prev[n]:
                    n += 1
                for d in range(n + 1, 12):
                    st = self.ui_hover_t.get(("sub_pop", d))
                    if st is not None:
                        st[0] = 0.0
                        st[1] = 0.0
                        st[2] = 0.0
                        st[3] = 1.0
            for d in range(1, len(self.ui_path) + 1):
                self.ui_tween(("sub_pop", d), 1.0, anim_dt, 0.22)
            geos = self._ui_panel_chain(i, list(getattr(self, "ui_path", [])), wh)
            for depth in range(1, len(geos)):
                self._ui_draw_sub_panel(geos, depth, wh, alpha, anim_dt)

        # Scrollbar when content overflows
        if max_scroll > 0 and alpha > 0.4:
            track_top = wh - MENU_H - 4
            track_bot = cur_y_end + 4
            track_h = max(1.0, track_top - track_bot)
            thumb_h = max(12.0, track_h * (track_h / (track_h + max_scroll)))
            thumb_y = track_top - (self.ui_menu_scroll / max_scroll) * (track_h - thumb_h)
            sb_x0 = x0 + scale_w - 6
            self.ui_rect(sb_x0, track_bot, sb_x0 + 3, track_top,
                         (C_EDGE[0], C_EDGE[1], C_EDGE[2], 0.6 * alpha))
            self.ui_rect(sb_x0, thumb_y - thumb_h, sb_x0 + 3, thumb_y,
                         (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], 0.85 * alpha))

    def _ui_arrow(self, x_right, cy, amount, alpha):
        """Right-pointing submenu arrow built from stacked rects."""
        col = (C_TEXT[0], C_TEXT[1], C_TEXT[2], alpha)
        xs = x_right - 11 + amount * 2.0
        for k in range(3):
            hh = 3.0 - k
            self.ui_rect(xs + k * 1.5, cy - hh, xs + k * 1.5 + 1.5, cy + hh, col)

    def _ui_draw_sub_panel(self, geos, depth, wh, alpha, anim_dt):
        """Draw one nested submenu panel (depth >= 1 in the chain)."""
        geo = geos[depth]
        x0, w = geo["x0"], geo["w"]
        rows = geo["rows"]
        if not rows:
            return
        alpha = alpha * float(geo.get("pop", 1.0))
        if alpha <= 0.01:
            return
        top, bot = geo["y_top"], geo["y_bot"]
        self.ui_rect(x0 - 1, bot - 1, x0 + w + 1, top + 1,
                     (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
        self.ui_rect(x0, bot, x0 + w, top,
                     (C_DROP[0], C_DROP[1], C_DROP[2], 0.97 * alpha))
        path = getattr(self, "ui_path", [])
        child = path[depth] if len(path) > depth else None
        for ridx, (item, yt, yb) in enumerate(rows):
            if item is None:
                ym = (yt + yb) / 2
                self.ui_rect(x0 + 8, ym, x0 + w - 8, ym + 1, (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
                continue
            hovered = item is child
            amount = self.ui_tween(("sitem", depth, ridx), 1.0 if hovered else 0.0, anim_dt, 0.18)
            if amount > 0.01:
                inset = (1.0 - amount) * 5.0
                self.ui_rect(x0 + inset, yb + 1, x0 + w - inset, yt - 1,
                             (C_DROP[0] + (C_HOT[0] - C_DROP[0]) * amount,
                              C_DROP[1] + (C_HOT[1] - C_DROP[1]) * amount,
                              C_DROP[2] + (C_HOT[2] - C_DROP[2]) * amount, alpha))
                if amount > 0.08:
                    self.ui_rect(x0 + 2, yb + 3, x0 + 4.5, yt - 3,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], amount * alpha))
            cy = (yt + yb) / 2 + amount * 1.5
            has_check = item.get("check") is not None
            if has_check:
                checked = item["check"]()
                chk_t = self.ui_tween(("schk", depth, ridx), 1.0 if checked else 0.0, anim_dt, 0.16, overshoot=checked)
                self.ui_rect(x0 + 7, cy - 5, x0 + 17, cy + 5,
                             (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
                if chk_t > 0.01:
                    r_sz = 3.0 * max(0.0, chk_t)
                    self.ui_rect(x0 + 12 - r_sz, cy - r_sz, x0 + 12 + r_sz, cy + r_sz,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], min(1.0, chk_t) * alpha))
            if alpha > 0.35:
                self.ui_put(self.ui_text(item["label"], C_TEXT), x0 + CHECK_W + amount * 3.5, cy)
                if item["key"]:
                    self.ui_put(self.ui_text(item["key"], C_DIM, anchor_x="right"), x0 + w - PAD_X, cy)
            if item.get("sub"):
                self._ui_arrow(x0 + w, cy, amount, alpha)

    def ui_draw_overlays(self, ww, wh):
        now = time.perf_counter()
        anim_dt = self.ui_frame_dt

        if self.ui_overlay is not None:
            self._last_overlay = self.ui_overlay
        pop_target = 1.0 if self.ui_overlay is not None else 0.0
        pop_t = self.ui_tween("modal_popup", pop_target, anim_dt, 0.24, overshoot=(pop_target > 0.5))
        active_ov = self.ui_overlay if self.ui_overlay is not None else getattr(self, "_last_overlay", None)

        if active_ov is not None and pop_t > 0.01:
            alpha = max(0.0, min(1.0, pop_t))
            title, rows = active_ov
            self.ui_rect(0, 0, ww, wh - MENU_H, (0.0, 0.0, 0.0, 0.48 * alpha))
            lw = max(self.ui_text(a, C_ACCENT).content_width for a, _ in rows)
            rw = max(self.ui_text(b, C_TEXT).content_width for _, b in rows)
            full_w, rh = lw + rw + 64, 24
            content_h = rh * len(rows) + 64
            # Cap modal height to available window so it can scroll on small scales
            max_h = max(120, (wh - MENU_H) - 40)
            scale_t = 0.82 + 0.18 * pop_t
            w = full_w * scale_t
            h = min(content_h, max_h) * scale_t
            off = getattr(self, "_overlay_off", [0.0, 0.0])
            cx = ww / 2 + off[0]
            cy_mid = (wh - MENU_H) / 2 - (1.0 - alpha) * 20.0 + off[1]
            x0, y1 = cx - w / 2, cy_mid + h / 2
            y0 = y1 - h
            self._overlay_bounds = (x0, y0, x0 + w, y1)

            self.ui_rect(x0 - 1, y0 - 1, x0 + w + 1, y1 + 1,
                         (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
            self.ui_rect(x0, y0, x0 + w, y1,
                         (C_DROP[0], C_DROP[1], C_DROP[2], 0.98 * alpha))
            self.ui_rect(x0, y1 - 2, x0 + w, y1,
                         (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], 0.85 * alpha))

            if alpha > 0.35:
                self.ui_put(self.ui_text(title, C_TEXT, 14), x0 + 24, y1 - 24)
                bx0, by0, bx1, by1 = x0 + w - 86, y1 - 40, x0 + w - 12, y1 - 12
                # Title bar stops short of the close button's expanded hit area
                self._overlay_titlebar = (x0, y1 - 48, bx0 - 8, y1)
                self.ui_modal_close = (bx0, by0, bx1, by1)
                mx, my = self.ui_ptr
                close_hover = (self.ui_overlay is not None) and (bx0 <= mx <= bx1 and by0 <= my <= by1)
                close_amount = self.ui_tween(("modal-close", title), 1.0 if close_hover else 0.0, anim_dt, 0.16)
                pad_b = close_amount * 2.0
                close_color = tuple(C_BAR[k] + (C_HOT[k] - C_BAR[k]) * close_amount for k in range(3)) + (alpha,)
                self.ui_rect(bx0 - pad_b, by0 - pad_b, bx1 + pad_b, by1 + pad_b, close_color)
                self.ui_rect(bx0 - pad_b, by0 - pad_b, bx1 + pad_b, by0 + 2,
                             (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], alpha))
                self.ui_put(self.ui_text("Close", C_TEXT, anchor_x="center"),
                            (bx0 + bx1) / 2, (by0 + by1) / 2 + close_amount * 1.0)

                # Scrollable body
                body_top = y1 - 56
                body_bot = y0 + 12
                body_h = max(1.0, body_top - body_bot)
                total_rows_h = len(rows) * rh * scale_t
                max_ov_scroll = max(0.0, total_rows_h - body_h)
                self._overlay_max_scroll = max_ov_scroll
                self.ui_overlay_scroll = max(0.0, min(self.ui_overlay_scroll, max_ov_scroll))
                self._overlay_row_hits = []
                mx, my = self.ui_ptr
                for n, (a, b) in enumerate(rows):
                    y = body_top - n * rh * scale_t + self.ui_overlay_scroll
                    if y < body_bot - 4 or y > body_top + 4:
                        continue
                    self._overlay_row_hits.append((x0 + 8, y - 10, x0 + w - 16, y + 10, a, b))
                    row_hover = (x0 + 8 <= mx <= x0 + w - 16 and y - 10 <= my <= y + 10)
                    r_amt = self.ui_tween(("ovrow", a), 1.0 if row_hover else 0.0, anim_dt, 0.16)
                    if r_amt > 0.01:
                        ri = (1.0 - r_amt) * 4.0
                        self.ui_rect(x0 + 6 + ri, y - 9, x0 + w - 10 - ri, y + 9,
                                     (C_HOT[0], C_HOT[1], C_HOT[2], 0.35 * r_amt * alpha))
                    is_toggle = str(b).lower() in ("on", "off", "true", "false", "yes", "no")
                    if is_toggle:
                        on = str(b).lower() in ("on", "true", "yes")
                        chk_t = self.ui_tween(("ovchk", a), 1.0 if on else 0.0, anim_dt, 0.16, overshoot=on)
                        self.ui_rect(x0 + 16, y - 6, x0 + 28, y + 6,
                                     (C_EDGE[0], C_EDGE[1], C_EDGE[2], alpha))
                        if chk_t > 0.01:
                            ci = (1.0 - min(1.0, chk_t)) * 4.0
                            self.ui_rect(x0 + 18 + ci, y - 4 + ci, x0 + 26 - ci, y + 4 - ci,
                                         (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], min(1.0, chk_t) * alpha))
                        self.ui_put(self.ui_text(a, C_ACCENT), x0 + 36, y)
                    else:
                        self.ui_put(self.ui_text(a, C_ACCENT), x0 + 24, y)
                        self.ui_put(self.ui_text(b, C_TEXT), x0 + 24 + lw + 16, y)
                # Overlay scrollbar
                if max_ov_scroll > 0:
                    thumb_h = max(12.0, body_h * (body_h / (body_h + max_ov_scroll)))
                    thumb_y = body_top - (self.ui_overlay_scroll / max_ov_scroll) * (body_h - thumb_h)
                    sb_x0 = x0 + w - 8
                    self.ui_rect(sb_x0, body_bot, sb_x0 + 3, body_top,
                                 (C_EDGE[0], C_EDGE[1], C_EDGE[2], 0.6 * alpha))
                    self.ui_rect(sb_x0, thumb_y - thumb_h, sb_x0 + 3, thumb_y,
                                 (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], 0.85 * alpha))
        else:
            self.ui_modal_close = None
            self._overlay_titlebar = None
            self._overlay_row_hits = []

        toast_active = False
        if self.ui_toast is not None:
            msg, until = self.ui_toast
            if now <= until:
                toast_active = True
                self._last_toast_msg = msg
            else:
                self.ui_toast = None
        toast_t = self.ui_tween("toast_pop", 1.0 if toast_active else 0.0, anim_dt, 0.22, overshoot=toast_active)
        if toast_t > 0.01 and getattr(self, "_last_toast_msg", None):
            t_alpha = max(0.0, min(1.0, toast_t))
            t = self.ui_text(self._last_toast_msg, C_TEXT)
            y0 = 10 - (1.0 - t_alpha) * 18.0
            tw = (t.content_width + 22) * (0.90 + 0.10 * t_alpha)
            self.ui_rect(8, y0 - 2, 8 + tw, y0 + 24,
                         (C_DROP[0], C_DROP[1], C_DROP[2], 0.96 * t_alpha))
            self.ui_rect(8, y0 - 2, 11, y0 + 24,
                         (C_ACCENT[0], C_ACCENT[1], C_ACCENT[2], t_alpha))
            if t_alpha > 0.35:
                self.ui_put(t, 18, y0 + 11)

    def _draw_hud(self, ww, wh):
        """Top-right HUD: FPS, frame, lag — matches the MesenCE overlay."""
        if not getattr(self.settings, "show_fps", True):
            return
        fps = getattr(self, "_last_fps", 0.0)
        frames = getattr(self.console, "frames", 0) if self.console else 0
        lag = getattr(self, "_lag_frames", 0)
        lines = []
        if self.settings.show_fps:
            lines.append(f"FPS: {fps:5.2f}")
        if self.settings.show_frame_counter:
            lines.append(f"Frame: {frames}")
        if self.settings.show_lag_counter:
            lines.append(f"Lag: {lag}")
        if not lines:
            return
        y = wh - MENU_H - 14
        for line in lines:
            t = self.ui_text(line, (1.0, 1.0, 1.0), 11, anchor_x="right")
            self.ui_put(t, ww - 8, y)
            y -= 14

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
            t = self.ui_text("File > Open ROM... (NES / Famicom / SNES / N64 / GB / GBC / GBA)   (Ctrl+O)", C_DIM, 14, "center")
            self.ui_put(t, ww / 2, avail / 2)
        self.ui_draw_menu(ww, wh)
        self._draw_hud(ww, wh)
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
        "        ('All Supported ROMs', '*.nes *.fds *.fam *.unf *.sfc *.smc *.snes *.z64 *.n64 *.v64 *.gb *.gbc *.gba'),\n        ('Famicom / Disk System', '*.fds *.fam *.unf'),\n        ('Nintendo 64 ROMs', '*.z64 *.n64 *.v64'),\n"
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
    ap.add_argument("--audio-latency", type=int, default=75, metavar="MS")
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
    # Optional SDL2/SDL3 path: if settings.video_backend is sdl2/sdl3 and the
    # module is present, the arcade window above is still used for the UI;
    # the flag is recorded so a future headless/SDL render loop can take over.
    if getattr(RetroWindow, "_settings", None) and RetroWindow._settings.video_backend.startswith("sdl"):
        print(f"[video] SDL backend requested ({RetroWindow._settings.video_backend}); "
              f"sdl2={_HAS_SDL2} sdl3={_HAS_SDL3}")
    arcade.run()


if __name__ == "__main__":
    main()