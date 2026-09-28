import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from born_wired.binaural_senses import BinauralSenses


def source_body(x, y, z=.33):
    """One source straight to the given side; z sits at the height of the ears."""
    return make_body({"sound_low": (x, y, z), "sound_high": (x, y, z)})


def ear_levels(body, name, frequency):
    """One emitter alone, so each band can be read off the two ears on its own."""
    senses = BinauralSenses(body, emitters=[{"geom": name, "frequency": frequency,
                                             "amplitude": .18}], window_samples=1024)
    waveform = senses.observe()
    return np.sqrt(np.mean(waveform**2, axis=1))


def make_body(sources=None):
    """Use a real MjModel/MjData; source geoms are static worldbody geoms."""
    sources = {} if sources is None else sources
    source_xml = "".join(
        f'<geom name="{name}" type="sphere" pos="{x} {y} {z}" size=".01"/>'
        for name, (x, y, z) in sources.items()
    )
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 .1"/>
        <body name="base" pos="0 0 .3">
          <geom name="torso" type="box" size=".2 .1 .05"/>
        </body>
        {source_xml}
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


class BinauralSensesTests(unittest.TestCase):
    def test_silent_observation_is_finite_and_read_only(self):
        body = make_body()
        senses = BinauralSenses(body)
        before = (body.data.time, body.data.qpos.copy(), body.data.qvel.copy(), body.data.xpos.copy())
        waveform = senses.observe()
        decoded = senses.decode(waveform)
        np.testing.assert_array_equal(waveform, np.zeros((2, 512)))
        np.testing.assert_array_equal(decoded["cochlea_energy"], np.zeros(3))
        np.testing.assert_array_equal(decoded["cochlea_direction"], np.zeros(3))
        np.testing.assert_array_equal(decoded["ear_waveform"], waveform)
        self.assertEqual(body.data.time, before[0])
        np.testing.assert_array_equal(body.data.qpos, before[1])
        np.testing.assert_array_equal(body.data.qvel, before[2])
        np.testing.assert_array_equal(body.data.xpos, before[3])

    def test_side_distance_and_swapped_waveforms(self):
        left_body = make_body({"sound_low": (0.2, 1.0, 0.03)})
        right_body = make_body({"sound_low": (0.2, -1.0, 0.03)})
        far_body = make_body({"sound_low": (0.2, 5.0, 0.03)})
        left = BinauralSenses(left_body).decode(BinauralSenses(left_body).observe())
        right = BinauralSenses(right_body).decode(BinauralSenses(right_body).observe())
        far = BinauralSenses(far_body).decode(BinauralSenses(far_body).observe())
        self.assertGreater(left["cochlea_direction"][0], 0.2)
        self.assertLess(right["cochlea_direction"][0], -0.2)
        np.testing.assert_allclose(left["cochlea_energy"][0], right["cochlea_energy"][0], atol=0.02)
        self.assertGreater(left["cochlea_energy"][0], far["cochlea_energy"][0])
        swapped = np.flip(BinauralSenses(left_body).observe(), axis=0)
        swapped_decoded = BinauralSenses(left_body).decode(swapped)
        np.testing.assert_allclose(swapped_decoded["cochlea_direction"], -left["cochlea_direction"], atol=1e-12)

    def test_a_sound_on_the_median_plane_reaches_both_ears_alike(self):
        """Straight ahead and straight behind put nothing between head and ear."""
        for x in (1.20, -.80):
            with self.subTest(x=x):
                body = source_body(x, 0.)
                for name, frequency in (("sound_low", 262.), ("sound_high", 880.)):
                    levels = ear_levels(body, name, frequency)
                    np.testing.assert_allclose(levels[0], levels[1], atol=1e-12)

    def test_the_head_holds_back_the_far_ear_and_its_top_band_most(self):
        """A sound to one side: the near ear hears it louder, and the top band
        is the part the far ear loses most."""
        body = source_body(.20, 1.00)
        low = ear_levels(body, "sound_low", 262.)
        high = ear_levels(body, "sound_high", 880.)
        self.assertGreater(low[0], low[1]*1.5)
        self.assertGreater(high[0], high[1]*3.)
        self.assertGreater(high[0]/high[1], low[0]/low[1])

    def test_the_outer_ear_shape_tells_front_from_back(self):
        """The same sound at the same distance: in front the top band stands
        out more than it does behind, which is the only cue that separates the
        two."""
        front, behind = source_body(1.20, 0.), source_body(-.80, 0.)
        front_top, front_bottom = ear_levels(front, "sound_high", 880.)[0], ear_levels(front, "sound_low", 262.)[0]
        back_top, back_bottom = ear_levels(behind, "sound_high", 880.)[0], ear_levels(behind, "sound_low", 262.)[0]
        self.assertGreater(front_top/front_bottom, 1.)
        self.assertLess(back_top/back_bottom, 1.)
        self.assertGreater(front_top/front_bottom, 1.5*back_top/back_bottom)

    def test_mixed_sources_and_identical_ears(self):
        body = make_body({"sound_low": (0.2, 1.0, 0.03), "sound_high": (0.2, -1.0, 0.03)})
        senses = BinauralSenses(
            body,
            emitters=[
                {"geom": "sound_low", "frequency": 262, "amplitude": 0.18},
                {"geom": "sound_high", "frequency": 880, "amplitude": 0.18},
            ],
        )
        decoded = senses.decode(senses.observe())
        self.assertGreater(decoded["cochlea_energy"][0], decoded["cochlea_energy"][1])
        self.assertGreater(decoded["cochlea_energy"][2], decoded["cochlea_energy"][1])
        self.assertGreater(decoded["cochlea_direction"][0], 0.2)
        self.assertLess(decoded["cochlea_direction"][2], -0.2)
        identical = np.tile(np.sin(np.linspace(0, 8 * np.pi, 512)), (2, 1))
        identical_decoded = senses.decode(identical)
        np.testing.assert_allclose(identical_decoded["cochlea_direction"], 0, atol=1e-12)

    def test_invalid_frames_are_rejected(self):
        senses = BinauralSenses(make_body())
        invalid = (
            np.zeros((512, 2)),
            np.zeros((2, 511)),
            np.full((2, 512), np.nan),
            np.full((2, 512), np.inf),
            np.zeros((2, 512), dtype=complex),
        )
        for waveform in invalid:
            with self.assertRaises(ValueError):
                senses.decode(waveform)


if __name__ == "__main__":
    unittest.main()
