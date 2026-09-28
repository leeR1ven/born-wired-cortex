"""Read-only physical binaural waveforms and cochlear projections."""

from __future__ import annotations

import numbers

import mujoco
import numpy as np


_EAR_LOCAL = np.array(((0.20, 0.07, 0.03), (0.20, -0.07, 0.03)), dtype=float)
_CENTER_FREQUENCIES = np.array((262.0, 523.0, 880.0), dtype=float)
_DEFAULT_EMITTERS = (
    {"geom": "sound_low", "frequency": 262.0, "amplitude": 0.18},
    {"geom": "sound_high", "frequency": 880.0, "amplitude": 0.18},
)
_SPEED_OF_SOUND = 343.0
# Left and right are told by time: a sound reaches the two ears at two different
# moments, and that inter-ear delay is already carried by the phase below. Front
# and back cannot be told that way. The ears are a pair of points on one axis, so
# a source ahead of the head and its mirror image behind it are the same distance
# from both ears and their two copies arrive together. What tells front from back
# is the shape of the outer ear, and it acts on frequency rather than on time:
# the head and the rim around the ear hold back the top of a sound coming from
# behind while letting the top of a sound from in front past, so one and the same
# noise arrives a little brighter from in front and a little duller from behind.
# That is a physical filter in the outer ear, not something the brain works out.
# Keeping the two cues apart is not fussiness: an earlier version reused a short
# delay here, and because that delay landed inside the few samples the cochlear
# circuit uses to read left from right, a sound on the left turned the eyes right
# (measured). A tilt only makes the same tone louder or quieter, which cannot
# touch phase and so cannot invert the left-right answer.
_FORWARD_LOCAL = np.array((1.0, 0.0, 0.0), dtype=float)
_LEFT_LOCAL = np.array((0.0, 1.0, 0.0), dtype=float)
# How hard front and back tilt the spectrum, as an exponent on frequency relative
# to the middle band: front multiplies the top band by r**tilt and the bottom by
# r**-tilt, back does the opposite. 0 would make the outer ear silent.
_PINNA_TILT = 0.45
_PINNA_REFERENCE = 523.0
# The other thing the head does to a sound is block it. A sound off to one side
# has to travel around the skull to reach the far ear, which loses some of it,
# and the top of the sound is lost first. So the near ear hears it louder, and
# the two ears differ again - in level this time, where the pinna above differed
# in shape. This too is the acoustics of the simulated head: a level difference
# between the two ear waveforms, nothing measured or decided.
_HEAD_SHADOW = 0.55
_HEAD_SHADOW_TILT = 0.5
_EAR_BASELINE = 0.14
_ENERGY_SCALE = 0.5
_WEAK_AMPLITUDE = 1e-6


def _real_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and positive" if positive else f"{name} must be finite and nonnegative")
    return result


class BinauralSenses:
    """Two-ear waveform sensor; observe and decode never change body state.

    Default emitter geoms are sound_low (262 Hz) and sound_high (880 Hz),
    each at amplitude .18. Missing emitters are silent. The cochlea energy
    scale is a 0.5 waveform-amplitude reference, clipped to [0, 1].
    """

    def __init__(self, body, emitters=None, sample_rate=16000, window_samples=512,
                 pinna=True, head_shadow=True):
        if not isinstance(getattr(body, "model", None), mujoco.MjModel) or not isinstance(
            getattr(body, "data", None), mujoco.MjData
        ):
            raise ValueError("body must expose a MuJoCo model and data")
        self.body, self.model = body, body.model
        self.sample_rate = _real_scalar(sample_rate, "sample_rate", positive=True)
        if isinstance(window_samples, (bool, np.bool_)) or not isinstance(window_samples, numbers.Integral):
            raise ValueError("window_samples must be a positive integer")
        self.window_samples = int(window_samples)
        if self.window_samples <= 0:
            raise ValueError("window_samples must be a positive integer")
        self._base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self._base < 0:
            raise ValueError("model is missing base")
        if not isinstance(pinna, (bool, np.bool_)):
            raise ValueError("pinna must be boolean")
        if not isinstance(head_shadow, (bool, np.bool_)):
            raise ValueError("head_shadow must be boolean")
        self.pinna = bool(pinna)
        self.head_shadow = bool(head_shadow)
        self._emitters = self._parse_emitters(emitters)

    @staticmethod
    def _parse_emitters(emitters):
        if emitters is None:
            emitters = _DEFAULT_EMITTERS
        if not isinstance(emitters, (list, tuple)):
            raise ValueError("emitters must be a list")
        parsed = []
        for item in emitters:
            if not isinstance(item, dict):
                raise ValueError("each emitter must be a dict")
            geom = item.get("geom", item.get("name"))
            if not isinstance(geom, str) or not geom:
                raise ValueError("each emitter needs a geom name")
            parsed.append(
                (
                    geom,
                    _real_scalar(item.get("frequency"), "frequency", positive=True),
                    _real_scalar(item.get("amplitude"), "amplitude"),
                )
            )
        return tuple(parsed)

    def observe(self):
        """Generate a fresh left/right waveform from current body geometry."""
        if self.body.model is not self.model:
            raise ValueError("construct new senses after replacing the body's model")
        data = self.body.data
        rotation = np.asarray(data.xmat[self._base], dtype=float).reshape(3, 3)
        base_position = np.asarray(data.xpos[self._base], dtype=float)
        simulation_time = float(data.time)
        if not (np.isfinite(rotation).all() and np.isfinite(base_position).all() and np.isfinite(simulation_time)):
            raise FloatingPointError("non-finite body pose or simulation time")
        ears = base_position + (rotation @ _EAR_LOCAL.T).T
        forward = rotation @ _FORWARD_LOCAL
        leftward = rotation @ _LEFT_LOCAL
        sample_times = np.arange(self.window_samples, dtype=float) / self.sample_rate
        waveform = np.zeros((2, self.window_samples), dtype=float)
        for name, frequency, amplitude in self._emitters:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                continue
            source = np.asarray(data.geom_xpos[geom], dtype=float)
            if not np.isfinite(source).all():
                raise FloatingPointError("non-finite emitter position")
            offset = source[None, :] - ears
            distance = np.linalg.norm(offset, axis=1)
            delay = distance / _SPEED_OF_SOUND
            phase = 2.0 * np.pi * frequency * (simulation_time - delay[:, None] + sample_times[None, :])
            direct = np.sin(phase)
            if self.pinna:
                # +1 when the sound comes from straight ahead of the head, -1
                # from straight behind it, per ear. A tone keeps its arrival time
                # and only changes how loud it is, per band.
                facing = (offset @ forward) / distance
                tilt = np.power(frequency/_PINNA_REFERENCE, _PINNA_TILT*facing)
                direct = direct*tilt[:, None]
            if self.head_shadow:
                toward = np.asarray(source, dtype=float) - base_position
                reach = float(np.linalg.norm(toward))
                lateral = float(toward @ leftward)/reach if reach > 0 else 0.
                # Ear 0 sits on the left, so it is the far one for a sound on
                # the right. How far the sound is off to a side is all this
                # asks: straight ahead and straight behind leave both ears at
                # full level, and the nearer the sound is to one side the more
                # the opposite ear loses. The near ear keeps its level and the
                # far ear is held back by the head, the top band most of all.
                shadowed = np.array((max(0., -lateral), max(0., lateral)))
                depth = _HEAD_SHADOW*np.power(frequency/_PINNA_REFERENCE, _HEAD_SHADOW_TILT)
                direct = direct*(1. - depth*shadowed)[:, None]
            waveform += (amplitude / (distance + 0.25))[:, None] * direct
        if not np.isfinite(waveform).all():
            raise FloatingPointError("non-finite ear waveform")
        # A source pressed against the head would otherwise drive the two tones
        # past what the ear can carry; a real ear has a full scale too, so the
        # waveform is held there instead of being handed on out of range.
        return np.clip(waveform, -1., 1.)

    def decode(self, waveform):
        """Decode one finite frame without using source geometry."""
        raw = np.asarray(waveform)
        if raw.shape != (2, self.window_samples) or raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
            raise ValueError(f"waveform must have real shape (2, {self.window_samples})")
        try:
            signal = np.array(raw, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("waveform must be finite real values") from exc
        if not np.isfinite(signal).all():
            raise ValueError("waveform must be finite real values")
        samples = np.arange(self.window_samples, dtype=float)
        window = np.hanning(self.window_samples)
        window_sum = float(window.sum())
        if window_sum == 0:
            window = np.ones(self.window_samples)
            window_sum = float(window.sum())
        times = samples / self.sample_rate
        amplitudes = np.empty((2, len(_CENTER_FREQUENCIES)), dtype=float)
        phases = np.empty_like(amplitudes)
        for index, frequency in enumerate(_CENTER_FREQUENCIES):
            carrier = 2.0 * np.pi * frequency * times
            projected = signal * window
            sine = (projected @ np.sin(carrier)) / window_sum
            cosine = (projected @ np.cos(carrier)) / window_sum
            amplitudes[:, index] = 2.0 * np.hypot(sine, cosine)
            phases[:, index] = np.arctan2(-cosine, sine)
        energy = np.clip(np.mean(amplitudes, axis=0) / _ENERGY_SCALE, 0.0, 1.0)
        max_phase = 2.0 * np.pi * _CENTER_FREQUENCIES * _EAR_BASELINE / _SPEED_OF_SOUND
        # Measured phase is the propagation-delay angle; the nearer ear leads.
        direction = np.angle(np.exp(1j * (phases[1] - phases[0]))) / max_phase
        direction = np.clip(direction, -1.0, 1.0)
        direction[np.min(amplitudes, axis=0) < _WEAK_AMPLITUDE] = 0.0
        return {
            "cochlea_energy": energy,
            "cochlea_direction": direction,
            "ear_waveform": signal,
        }
