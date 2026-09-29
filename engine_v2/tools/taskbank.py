"""The exam bank: one question per scene, one bar per question.

Every task does the same three things: build a scene, read a few numbers off
the body and the cells, and compare them with a bar written down in BAR.  Bars
come from two places.  Some are the repository's own existing criteria, quoted
from the tool that already used them (``validate_reflex_v3.behavior_pass``,
``probe_righting``'s ``recovered``).  The rest were fixed by measuring the birth
animal first and then setting the bar with headroom; the birth animal's own
numbers are recorded in BIRTH so the bar's origin stays visible.

Nothing here selects an action.  A task never looks at a joint angle to decide
what the animal should do; it only reads where the body ended up, how upright it
stayed and how hard a cell fired.  The only thing that changes between two
candidates is the birth wiring - the named innate gains, plus the seed that
draws the random scaffold.

Two more things sit on top of the flat list.

``requires`` is the ladder.  A task may name the tasks that come before it, so
an animal that cannot stand is not asked to run, and a bank run can report the
deepest rung it reached instead of a bare count.  ``--tree`` prints that
ladder; ``--unlocked`` prints what a given set of passed tasks opens up.

``BIRTH`` is the animal's own reading on every question.  A bar that the birth
animal fails is deliberate: the bank is meant to be wider than the animal, and
the note on each task says what the failing reading does and does not prove.

    python tools/taskbank.py --list
    python tools/taskbank.py --tree
    python tools/taskbank.py --explain
    python tools/taskbank.py --stage screen --seeds 0 1 2
"""
import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.binaural_senses import BinauralSenses      # noqa: E402
from born_wired.embodied import EmbodiedController         # noqa: E402
from born_wired.go2_body import Go2Body                    # noqa: E402
from born_wired.stereo_senses import RawEyes               # noqa: E402
from tools.validate_reflex_v3 import (CONTROLLER_DEFAULTS, load_live_parameters,  # noqa: E402
                                      controller_parameters, run_seed)
from tools import probe_righting                            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
DT = .01
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
# The props the arena ships with, from measure_eye_gaze_routes.py.  A "clean
# room" scene moves them far away instead of deleting them.
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high",
           "green_target", "left_block", "right_block")

# Every exam runs at the paper's reference size, not the live window's
# 192x144 / 1600-512-2048.  A bank has to be run per candidate, so the animal
# here is the small one; what it can and cannot do at this size is measured,
# not assumed.  The sizes are the exam's own, so the live window's copies of
# the same names are dropped rather than passed twice.
REFERENCE_SIZE = {"motor_units": 200, "proprio_units": 64, "association_units": 256,
                  "eye_width": 48, "eye_height": 36}
SIZE_KEYS = tuple(REFERENCE_SIZE)

# The genes a candidate is allowed to differ in: the named innate gains the
# live window runs, minus the sizes.  Each one is a group of innate connection
# strengths, so the genome is the birth wiring written in words.
_RAW_GENES = {name: value for name, value in load_live_parameters().items()
              if name not in SIZE_KEYS}
GENES = tuple(sorted(_RAW_GENES))


def _unaccepted_genes():
    """Named gains the controller would refuse.

    The config file is read again every time an exam asks for its parameters,
    but the engine module is imported once, at the start.  Editing the config
    while a round is running therefore leaves that round asking old code for
    parameters it has never heard of: every exam comes back an error and the
    round goes on drawing candidates that cannot be built.  The engine does
    raise, but only one exam at a time; this says it once, before anything
    starts, with the whole list at once.
    """
    from born_wired.embodied import EmbodiedController
    accepted = set()
    for klass in EmbodiedController.__mro__:
        try:
            accepted |= set(inspect.signature(klass.__init__).parameters)
        except (TypeError, ValueError):
            continue
    return sorted(set(GENES) - accepted)


_REFUSED_GENES = _unaccepted_genes()
if _REFUSED_GENES:
    raise SystemExit("live_config.json names gains no controller accepts: %s" % _REFUSED_GENES)
# A few of the named gains are whole numbers of steps, not sizes: half of a step
# is not a step.  Mutation has to round those rather than blend them.
INTEGER_GENES = frozenset(name for name, value in _RAW_GENES.items()
                          if isinstance(value, int) and not isinstance(value, bool))

# Bars.  See the module docstring for where each number comes from.
BAR = {
    # --- moving ---
    "walk_furnished_m": .35,      # 12 s in the furnished arena
    "walk_flat_m": .25,           # 10 s on the bare floor
    "walk_step_m": .30,           # 10 s from 0.55 m before the low step
    "walk_ramp_m": .25,           # 10 s from 0.25 m before the ramp
    "walk_passage_m": .25,        # 10 s from inside the passage mouth
    "walk_curb_m": .30,           # 10 s from 0.80 m before the curb
    "walk_platform_m": .30,       # 10 s from 0.90 m before the platform
    "walk_own_m": .30,            # 8 s of walking with nothing driving it
    "walk_own_far_m": .60,        # 16 s of walking with nothing driving it
    "walk_into_wall_m": .10,      # 12 s aimed at the east wall; it must not pass it
    "quiet_idle_m": .70,          # 6 s of standing still: the animal creeps 0.54 m
    "slip_mps": .30,              # worst sliding speed of a planted foot
    "walk_slip_m": .25,           # the distance the slip task also asks for
    # --- looking ---
    "gaze_error_rad": .16,        # mean |gaze - bearing| while the ball sweeps
    "gaze_error_fast_rad": .30,   # the faster sweep
    "gaze_still_rad": .05,        # a parked ball must not make the eyes drift
    "sudden_shift_rad": .01,      # how far the eyes swing towards a new thing
    "bank_far_max": .80,          # distance cells at 2 m must read "nothing near"
    "bank_near_min": 2.00,        # and must fill up by the time the ball is at 0.25 m
    "pattern_gain": .02,          # retina cells: a striped picture over a black one
                # --- hearing ---
    "sound_yaw_rad": .02,         # right sound must move the eyes further than left
    "pinna_gain": .03,            # front minus back, cell by cell, on the pinna cells
    "loud_far_gain": .06,         # near minus far on the cochlea cells
    "high_low_gain": .30,         # top band minus bottom band, high tone over low
    "startle_gain": .10,          # a bang over silence
    "startle_habituation": .20,   # the startle cell falling while the bang holds
    "hears_while_tiring": .05,    # the cochlea still firing while the startle cell tires
    # --- touch and body ---
    "front_touch_gain": .30,      # a touch on the front of the body over one on the back
    "obstacle_gain": .30,         # the caught leg's clearance cells over the other
    "slip_gain": .05,             # the slipping leg's withdrawal cells over the other
    "lean_gain": .15,             # the lean cell facing a roll over the opposite one
    "tilt_protective_gain": .20,  # the protective cell when tipped over
    "load_gain": .05,             # the loaded leg's cells over an unloaded stance
    "wall_gain": .05,             # wall-contact cells against a wall over open floor
    # --- learning, memory, self ---
    "taught_reflex_gain": .05,    # retreat cells on the tone after the lesson
    "taught_reflex_control": .05,  # with no teacher the tone must stay below this
    "taught_reflex_kept": .05,    # the lesson still there after a quiet pause
    "visual_memory_gain": .01,    # aversive response to the pattern after pairing
    "visual_memory_control": .01,  # with no pairing the pattern must stay below this
    "visual_memory_kept": .01,    # the pairing still there after a gap
    "moving_is_not_being_pushed": .10,  # its own gait vs a shove, in the tilt cells
    "body_is_not_a_foot": .05,    # a body touch must read on the righting cells
    "own_eyes_are_felt": .0002,   # eye proprioception with the eyes turned
    # --- walking further, walking faster, holding a line ---
    "walk_twenty_m": .90,         # 20 s of walking on its own
    "walk_ground_m": 1.00,        # 16 s of walking on its own
    "walk_speed_mps": .15,        # mean horizontal speed over ten seconds
    "heading_hold_rad": .60,      # how much heading it may wander through
    "straightness": .70,          # net displacement / path length
    "crab_m": .25,                # sideways drift over a ten-second walk
    "touch_hold_fraction": .70,   # a front touch must cut the distance to this
    # --- turning ---
    "turn_pair_rad": .30,         # right-hand sound yaw minus left-hand sound yaw
    "turn_idle_rad": .10,         # and the same for an animal that is not walking
    "idle_spin_rad": .60,         # heading travelled through while idle
    "idle_spin_progress_m": .15,  # ...with less than this ground covered is pivoting
    # --- going down and going round ---
    "down_ramp_m": .25,
    "down_step_m": .25,
    "down_platform_m": .25,
    "around_block_m": .60,
    "around_block_lateral_m": .35,
    # --- sounds that move it ---
    "bang_stop_m": .25,           # a bang held on must cut the distance to this
    "quiet_sound_gain": .02,      # a faint sound over silence, on the cochlea
    "behind_gain": .02,           # a sound behind over one in front, on the gaze cells
    "two_sounds_gain": .02,       # two emitters over one, cell by cell on the pinna cells
    # --- eyes, second batch ---
    "near_far_angle": .02,        # convergence-in minus convergence-out, near over far
    "slow_gaze_rad": .16,         # the ball at half the sweep speed
    "static_gaze_rad": .05,       # a ball parked one way, and the same one the other way
    # --- touch, second batch ---
    "side_touch_gain": .30,       # one body sector over the opposite one
    "back_touch_gain": .20,
    "ground_feet_gain": .05,      # one supported foot over another
    # --- learning and memory, second batch ---
    "quick_lesson_gain": .05,     # the lesson after five seconds instead of twenty
    "kept_under_distraction": .05,
    "second_pattern_gain": .01,   # a second pattern, paired the same way
    "one_pairing_gain": .01,      # one pairing instead of five
    # --- standing on its own, third batch ---
    "stand_long_m": 1.40,          # 20 s idle: the animal creeps 1.40 m
    "slope_stand_z": .85,          # standing still on the ramp top
    "walk_thirty_m": 1.30,         # 30 s of walking on its own
    # --- the quality of the gait ---
    "foot_lift_fraction": .02,     # every foot has to leave the ground sometimes
    "leg_symmetry": .20,           # how far apart the two legs of a pair may be
    "passage_quiet_touch": .10,    # wall cells while threading the passage
    "fatigue_ratio": .50,          # last window over first window of a long walk
    # --- turning, and coming back ---
    "turn_walk_rad": .30,          # a sound at one shoulder while it is walking
    "turn_walk_m": .45,            # and it still has to get somewhere
    "back_away_m": .10,            # a hand on the chest has to push it back
    "behind_hold_rad": .35,        # a sound at its back must not swing it
    # --- eyes, third batch ---
    "colour_gain": .02,            # a red square against a green one, same shape
    "size_gain": .02,              # a big ball against a small one, same place
    "pitch_gaze_rad": .22,         # the eye pitch while the ball climbs the picture
    "recentre_rad": .05,           # how far off centre the eyes may stay when it is gone
    "slow_new_thing_rad": .01,     # a thing that appears and creeps
    "far_gaze_rad": .20,           # the sweep at 2 m
    "near_gaze_rad": .20,          # the sweep at 0.4 m
    # --- ears, third batch ---
    "sound_gap_ratio": .50,        # a sound that stopped, over the same sound still on
    "silence_floor": .02,          # a quiet room must not be a sound
    "side_gain": .02,              # one shoulder against the other
    "same_side_gain": .02,         # two emitters at one shoulder against one
    "startle_behind_gain": .10,    # a bang behind it
    # --- touch, third batch ---
    "light_touch_gain": .10,       # a light hand, not a grip
    "front_load_gain": .05,        # the front legs taking the weight
    "all_feet_lift_gain": .05,     # each leg reports itself when it is caught
    "ground_quiet_gain": .05,      # flat ground must not read as a bump
    # --- learning and memory, third batch ---
    "extinction_drop": .30,        # how far the lesson may fall with nobody teaching
    "second_lesson_gain": .01,     # a second short lesson on top of the first
    "two_pattern_gain": .01,       # the paired pattern, with another one around
    "shifted_pattern_ratio": .30,  # the same pattern, moved a little
    # --- attention, the body as its own thing, the layers under the surface ---
    "bright_pull_m": .05,          # a bright thing in front has to pull it forward
    "tip_over_gain": .05,          # tipped over, read off its own righting cells
    "feature_gain": .01,           # two pictures, cell by cell in the feature layers
    # --- running, and the dark ---
    "bright_pace_ratio": 1.25,     # a bright thing in front has to buy real speed
    "run_mean_speed": .25,         # ten seconds of running, in metres per second
    "run_twenty_m": 2.00,          # twenty seconds of running, in metres
    "run_turn_speed": .25,         # running while it turns, in metres per second
    "opposite_reading": .05,      # the layer under the rods in a black room
    "long_life_kept": .80,          # the lesson after a minute of living, over the lesson at once
    "second_tone_kept": .80,        # the first tone, after a second one went down the same route
    "route_ceiling_floor": .01,     # the route has to move at all before "it slows down" means anything
    "route_ceiling_slowdown": .25,  # the last equal stretch may add at most this much of the first
    # --- the association layer, which is neither an input nor a muscle ---
    "association_driven_gain": .02,  # a body state has to move it at all
    "association_hold_gain": .01,    # and something of it has to still be there after
    "association_return_gain": .00005,  # and it has to reach the muscles at all
                                       # (birth reads 0.00000; gain 1 reads +1.6e-4)
    "straight_gain": .05,              # a thing to head for has to keep the walk straighter
                                       # (birth reads +0.102 / +0.093 / +0.180 at three seeds)
}


def _upright(measures):
    """Nothing below means anything if it fell over on the way.

    A gate that holds has to leave the slack alone, so it returns infinity
    rather than zero: zero would read as "passed by nothing at all".
    """
    return float("inf") if measures.get("behavior_pass") else -1.


def _recovered(measures):
    """A righting question is about ending on its feet, not about falling over."""
    return float("inf") if measures.get("recovered") else -1.


def _on_its_feet(measures):
    """Standing questions are about staying up, so falling is the whole miss."""
    return float("inf") if measures.get("min_up_z", 0.) >= .9 else -1.


def _marker_bar(measures, key):
    """It has to walk the distance *and* get past the thing in the way."""
    slack = float(measures["displacement_m"]) - BAR[key]
    if not measures.get("crossed_marker"):
        slack = min(slack, -1.)
    return min(slack, _upright(measures))


def _idle_spin(measures):
    """Pivoting is turning a lot and getting nowhere: either one alone is fine."""
    return max(BAR["idle_spin_rad"] - float(measures["yaw_path_rad"]),
               float(measures["displacement_m"]) - BAR["idle_spin_progress_m"])


# Which reading decides each question, so a score can say "missed the bar by a
# hair" where pass/fail can only say "failed".  A round that only counts
# passes cannot tell an animal that missed by three percent from one that never
# moved at all, and those two are not the same animal to breed from.
#
# ``path`` is where the reading sits in the answer, ``bar`` is the number it is
# compared with - a name in BAR or a plain number - and ``sign`` is +1 when a
# bigger reading is better.  ``gate`` is a second reading that has to hold
# before the number means anything: an animal that fell over did not "walk a
# little", it fell over.  A question whose bar combines readings carries a
# ``rule`` instead.  A question with no entry here is scored pass/fail only;
# the margin of a question nobody has written down is not guessed at.
MARGIN = {
    # --- standing ---
    "stands_still": dict(path="displacement_m", bar="quiet_idle_m", sign=-1,
                         gate=_on_its_feet),
    "stands_still_for_twenty_seconds": dict(path="displacement_m", bar="quiet_idle_m",
                                            sign=-1, gate=_on_its_feet),
    # --- walking ---
    "walk_flat": dict(path="displacement_m", bar="walk_flat_m", sign=+1, gate=_upright),
    "walk_furnished": dict(path="displacement_m", bar="walk_furnished_m", sign=+1,
                           gate=_upright),
    "walks_without_being_told": dict(path="displacement_m", bar="walk_own_m", sign=+1,
                                     gate=_upright),
    "keeps_walking_without_being_told": dict(path="displacement_m", bar="walk_own_far_m",
                                             sign=+1, gate=_upright),
    "walks_for_twenty_seconds": dict(path="displacement_m", bar="walk_twenty_m", sign=+1,
                                     gate=_upright),
    "walks_at_a_steady_pace": dict(path="mean_speed_mps", bar="walk_speed_mps", sign=+1,
                                   gate=_upright),
    "covers_ground_in_sixteen_seconds": dict(path="displacement_m", bar="walk_ground_m",
                                             sign=+1, gate=_upright),
    # --- running ---
    "runs_without_being_told": dict(path="mean_speed_mps", bar="run_mean_speed", sign=+1,
                                    gate=_upright),
    "runs_for_twenty_seconds": dict(path="displacement_m", bar="run_twenty_m", sign=+1,
                                    gate=_upright),
    # --- terrain ---
    "steps_over_the_curb": dict(rule=lambda m: _marker_bar(m, "walk_curb_m")),
    "climbs_onto_the_platform": dict(rule=lambda m: _marker_bar(m, "walk_platform_m")),
    # --- righting ---
    "get_up_from_back": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    "get_up_from_side": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    "nose_up_recover": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    "nose_down_recover": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    "stay_up_when_pushed": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    "gets_up_after_a_hard_shove": dict(path="final_up_z", bar=.5, sign=+1, gate=_recovered),
    # --- turning ---
    "does_not_spin_on_the_spot": dict(rule=_idle_spin),
    # --- seeing ---
    "eyes_follow_ball": dict(path="mean_abs_error_rad", bar="gaze_error_rad", sign=-1),
    "eyes_follow_fast_ball": dict(path="mean_abs_error_rad", bar="gaze_error_fast_rad",
                                  sign=-1),
    "eyes_hold_a_slow_thing": dict(path="mean_abs_error_rad", bar="gaze_error_rad",
                                   sign=-1),
    # --- the widest layer ---
    "the_association_layer_reads_the_body": dict(path="driven", sign=+1,
                                                 bar="association_driven_gain"),
    "the_association_loop_holds_what_it_was_given": dict(path="held", sign=+1,
                                                         bar="association_hold_gain"),
    "the_widest_layer_reaches_the_muscles": dict(path="reach", sign=+1,
                                                 bar="association_return_gain"),
    # --- what a thing to head for does to the line it walks ---
    "keeps_its_line_towards_what_it_sees": dict(path="straight_gain", sign=+1,
                                                bar="straight_gain"),
}


def margin(task, measures):
    """How much room the reading behind a question had, signed: >0 passed.

    ``None`` when the question has no margin written down, when the run failed,
    or when the answer does not carry the reading - a margin that cannot be
    read is not reported as zero, because zero is a real answer.
    """
    recipe = MARGIN.get(task)
    if not recipe or not isinstance(measures, dict) or measures.get("status") != "ok":
        return None
    if "rule" in recipe:
        try:
            return float(recipe["rule"](measures))
        except Exception:
            return None
    value = measures
    for part in recipe["path"].split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    bar = recipe["bar"]
    if isinstance(bar, str):
        if bar not in BAR:
            return None
        bar = BAR[bar]
    slack = float(recipe["sign"]) * (float(value) - float(bar))
    gate = recipe.get("gate")
    if gate is not None:
        try:
            slack = min(slack, float(gate(measures)))
        except Exception:
            pass
    return slack

# What the birth animal measured, so a bar can be read next to the number it was
# set from.  Every string is written by ``--birth-text`` out of a real run of
# artifacts/exam_birth_full.json; nothing here is estimated.
BIRTH = {
    "stands_still": "seed 0 PASS (stayed put at 0.543 m); seed 1 PASS (stayed put at 0.521 m); seed 2 PASS (stayed put at 0.441 m)",
    "walk_flat": "seed 0 PASS (walked 0.556 m, stayed up); seed 1 PASS (walked 0.519 m, stayed up); seed 2 PASS (walked 0.455 m, stayed up)",
    "walk_furnished": "seed 0 PASS (walked 0.590 m, stayed up); seed 1 PASS (walked 0.609 m, stayed up); seed 2 PASS (walked 0.644 m, stayed up)",
    "walks_without_being_told": "seed 0 PASS (walked 0.557 m, stayed up); seed 1 PASS (walked 0.518 m, stayed up); seed 2 PASS (walked 0.453 m, stayed up)",
    "keeps_walking_without_being_told": "seed 0 PASS (walked 0.658 m, stayed up); seed 1 PASS (walked 0.622 m, stayed up); seed 2 PASS (walked 0.655 m, stayed up)",
    "feet_do_not_slide": "seed 0 PASS (worst sliding foot 0.039 m/s); seed 1 PASS (worst sliding foot 0.061 m/s); seed 2 PASS (worst sliding foot 0.043 m/s)",
    "walk_low_step": "seed 0 PASS (walked 0.381 m, stayed up); seed 1 PASS (walked 0.549 m, stayed up); seed 2 PASS (walked 0.588 m, stayed up)",
    "walk_ramp": "seed 0 PASS (walked 0.447 m, stayed up); seed 1 PASS (walked 0.430 m, stayed up); seed 2 PASS (walked 0.367 m, stayed up)",
    "walk_narrow_passage": "seed 0 PASS (walked 0.268 m, stayed up); seed 1 PASS (walked 0.281 m, stayed up); seed 2 fail (not upright the whole way (min up_z -0.269, min height 0.120))",
    "steps_over_the_curb": "seed 0 fail (walked 0.583 m but never crossed the far side of the thing in the way (reached x 0.932)); seed 1 fail (walked 0.537 m but never crossed the far side of the thing in the way (reached x 0.884)); seed 2 fail (walked 0.565 m but never crossed the far side of the thing in the way (reached x 0.905))",
    "climbs_onto_the_platform": "seed 0 fail (walked 0.117 m, bar 0.300 m); seed 1 fail (walked 0.116 m, bar 0.300 m); seed 2 fail (not upright the whole way (min up_z 0.954, min height 0.120))",
    "stops_at_the_wall": "seed 0 PASS (walked 0.323 m and stopped at x 2.905 (wall cells peaked at 0.0000)); seed 1 PASS (walked 0.313 m and stopped at x 2.904 (wall cells peaked at 0.0000)); seed 2 PASS (walked 0.296 m and stopped at x 2.831 (wall cells peaked at 0.0000))",
    "get_up_from_back": "seed 0 fail (got up at 2.10 s and then went back down (final up_z -0.732)); seed 1 fail (never stayed on its feet (final up_z 0.477, height 0.155)); seed 2 fail (never stayed on its feet (final up_z -0.860, height 0.141))",
    "get_up_from_side": "seed 0 PASS (on its feet from 2.28 s and still there at the end (final up_z 1.000)); seed 1 PASS (on its feet from 2.28 s and still there at the end (final up_z 1.000)); seed 2 fail (got up at 2.39 s and then went back down (final up_z 0.021))",
    "nose_up_recover": "seed 0 PASS (on its feet from 2.67 s and still there at the end (final up_z 0.620)); seed 1 PASS (on its feet from 2.64 s and still there at the end (final up_z 0.572)); seed 2 fail (got up at 2.82 s and then went back down (final up_z -0.279))",
    "nose_down_recover": "seed 0 PASS (on its feet from 2.65 s and still there at the end (final up_z 0.574)); seed 1 fail (got up at 2.41 s and then went back down (final up_z -0.066)); seed 2 fail (got up at 2.15 s and then went back down (final up_z -0.804))",
    "stay_up_when_pushed": "seed 0 PASS (on its feet from 3.24 s and still there at the end (final up_z 0.785)); seed 1 fail (got up at 3.32 s and then went back down (final up_z 0.327)); seed 2 PASS (on its feet from 3.46 s and still there at the end (final up_z 0.838))",
    "eyes_follow_ball": "seed 0 PASS (looked 0.074 rad off the ball); seed 1 PASS (looked 0.074 rad off the ball); seed 2 PASS (looked 0.074 rad off the ball)",
    "eyes_follow_fast_ball": "seed 0 PASS (looked 0.175 rad off the ball); seed 1 PASS (looked 0.175 rad off the ball); seed 2 PASS (looked 0.175 rad off the ball)",
    "eyes_follow_the_ball_both_ways": "seed 0 PASS (followed the ball both ways (worst 0.076 rad)); seed 1 PASS (followed the ball both ways (worst 0.076 rad)); seed 2 PASS (followed the ball both ways (worst 0.076 rad))",
    "eyes_on_a_new_thing": "seed 0 PASS (the eyes swung 0.0264 rad towards the thing that appeared); seed 1 PASS (the eyes swung 0.0264 rad towards the thing that appeared); seed 2 PASS (the eyes swung 0.0264 rad towards the thing that appeared)",
    "eyes_hold_still_on_a_still_thing": "seed 0 PASS (the eyes held still: 0.0171 rad of wander at a parked ball); seed 1 PASS (the eyes held still: 0.0171 rad of wander at a parked ball); seed 2 PASS (the eyes held still: 0.0171 rad of wander at a parked ball)",
    "eyes_tell_near_from_far": "seed 0 fail (the ball came to 0.25 m and the distance cells only reached 0.10); seed 1 fail (the ball came to 0.25 m and the distance cells only reached 0.10); seed 2 fail (the ball came to 0.25 m and the distance cells only reached 0.10)",
    "eyes_see_a_pattern_not_a_blank": "seed 0 PASS (有花纹的画面 came apart by 0.26378); seed 1 PASS (有花纹的画面 came apart by 0.26378); seed 2 PASS (有花纹的画面 came apart by 0.26378)",
    "turn_to_sound": "seed 0 PASS (right-hand sound pulled the eyes -0.2754 rad towards it); seed 1 PASS (right-hand sound pulled the eyes -0.2754 rad towards it); seed 2 PASS (right-hand sound pulled the eyes -0.2754 rad towards it)",
    "ears_tell_front_from_back": "seed 0 PASS (耳廓前后 came apart by 0.04115); seed 1 PASS (耳廓前后 came apart by 0.04115); seed 2 PASS (耳廓前后 came apart by 0.04115)",
    "ears_tell_loud_from_far": "seed 0 PASS (近处和远处 came apart by 0.10433); seed 1 PASS (近处和远处 came apart by 0.10433); seed 2 PASS (近处和远处 came apart by 0.10433)",
    "ears_tell_high_from_low": "seed 0 PASS (gain = 1.14354); seed 1 PASS (gain = 1.14354); seed 2 PASS (gain = 1.14354)",
    "a_bang_makes_it_startle": "seed 0 PASS (受惊细胞 came apart by 0.22222); seed 1 PASS (受惊细胞 came apart by 0.22222); seed 2 PASS (受惊细胞 came apart by 0.22222)",
    "a_sound_keeps_startling_less": "seed 0 PASS (the startle cell fell 0.6787 while the bang kept on); seed 1 PASS (the startle cell fell 0.6787 while the bang kept on); seed 2 PASS (the startle cell fell 0.6787 while the bang kept on)",
    "feels_a_touch_on_the_front": "seed 0 PASS (前面被碰 came apart by 1.00000); seed 1 PASS (前面被碰 came apart by 1.00000); seed 2 PASS (前面被碰 came apart by 1.00000)",
    "lifts_the_foot_that_was_caught": "seed 0 PASS (被绊的那条腿 moved 0.87949 and 0.87949); seed 1 PASS (被绊的那条腿 moved 0.87949 and 0.87949); seed 2 PASS (被绊的那条腿 moved 0.87949 and 0.87949)",
    "feels_a_slipping_foot": "seed 0 PASS (打滑的那条腿 moved 0.11797 and 0.11797); seed 1 PASS (打滑的那条腿 moved 0.11797 and 0.11797); seed 2 PASS (打滑的那条腿 moved 0.11797 and 0.11797)",
    "feels_which_way_it_is_leaning": "seed 0 PASS (the direction cells moved 0.3394 when the body rolled); seed 1 PASS (the direction cells moved 0.3394 when the body rolled); seed 2 PASS (the direction cells moved 0.3394 when the body rolled)",
    "braces_when_it_is_tipped_over": "seed 0 PASS (gain_mean = 0.47744); seed 1 PASS (gain_mean = 0.47744); seed 2 PASS (gain_mean = 0.47744)",
    "feels_a_load_on_one_foot": "seed 0 PASS (承重的那条腿 moved 0.09023 and 0.09023); seed 1 PASS (承重的那条腿 moved 0.09023 and 0.09023); seed 2 PASS (承重的那条腿 moved 0.09023 and 0.09023)",
    "startled_but_still_hearing": "seed 0 PASS (the startle cell fell 0.6787 while the ear stayed at 0.0832); seed 1 PASS (the startle cell fell 0.6787 while the ear stayed at 0.0832); seed 2 PASS (the startle cell fell 0.6787 while the ear stayed at 0.0832)",
    "taught_reflex_sticks": "seed 0 PASS (tone alone now drives retreat 0.970 over silence); seed 1 PASS (tone alone now drives retreat 0.970 over silence); seed 2 PASS (tone alone now drives retreat 0.970 over silence)",
    "nothing_is_taught_without_a_teacher": "seed 0 PASS (the tone still means nothing without a teacher (0.000)); seed 1 PASS (the tone still means nothing without a teacher (0.000)); seed 2 PASS (the tone still means nothing without a teacher (0.000))",
    "the_lesson_is_still_there_later": "seed 0 PASS (tone alone now drives retreat 0.945 over silence); seed 1 PASS (tone alone now drives retreat 0.945 over silence); seed 2 PASS (tone alone now drives retreat 0.945 over silence)",
    "visual_memory_holds": "seed 0 PASS (the pattern drives the aversive cells 0.0761 over silence); seed 1 PASS (the pattern drives the aversive cells 0.0761 over silence); seed 2 PASS (the pattern drives the aversive cells 0.0761 over silence)",
    "the_pattern_means_nothing_without_pairing": "seed 0 PASS (the pattern drives the aversive cells -0.0000 over silence); seed 1 PASS (the pattern drives the aversive cells -0.0000 over silence); seed 2 PASS (the pattern drives the aversive cells -0.0000 over silence)",
    "the_memory_is_still_there_later": "seed 0 PASS (the pattern drives the aversive cells 0.1413 over silence); seed 1 PASS (the pattern drives the aversive cells 0.1413 over silence); seed 2 PASS (the pattern drives the aversive cells 0.1413 over silence)",
    "an_idle_animal_does_not_walk_for_a_touch": "seed 0 PASS (stayed put at 0.193 m); seed 1 PASS (stayed put at 0.194 m); seed 2 PASS (stayed put at 0.193 m)",
    "an_idle_animal_does_not_walk_for_a_ball": "seed 0 PASS (stayed put at 0.557 m); seed 1 PASS (stayed put at 0.518 m); seed 2 PASS (stayed put at 0.453 m)",
    "moving_is_not_being_pushed": "seed 0 PASS (a shove read 1.0000 above its own walking); seed 1 PASS (a shove read 1.0000 above its own walking); seed 2 PASS (a shove read 1.0000 above its own walking)",
    "the_body_is_not_a_foot": "seed 0 PASS (身上和脚上 moved 0.98127 and 0.19493); seed 1 PASS (身上和脚上 moved 0.98127 and 0.19493); seed 2 PASS (身上和脚上 moved 0.98127 and 0.19493)",
    "walks_in_a_straight_line": "seed 0 fail (直度 = 0.6221, bar 0.7000); seed 1 fail (直度 = 0.6381, bar 0.7000); seed 2 fail (直度 = 0.4508, bar 0.7000)",
    "does_not_crab_sideways": "seed 0 PASS (往旁边蹭的距离 = 0.0459); seed 1 PASS (往旁边蹭的距离 = 0.0342); seed 2 PASS (往旁边蹭的距离 = 0.1467)",
    "holds_its_heading_while_it_walks": "seed 0 fail (一路累计转头 = 1.1749, bar 0.6000); seed 1 fail (一路累计转头 = 1.0354, bar 0.6000); seed 2 fail (一路累计转头 = 1.7423, bar 0.6000)",
    "walks_for_twenty_seconds": "seed 0 fail (walked 0.654 m, bar 0.900 m); seed 1 fail (walked 0.620 m, bar 0.900 m); seed 2 fail (walked 0.786 m, bar 0.900 m)",
    "walks_at_a_steady_pace": "seed 0 fail (平均速度 averaged 0.087 m/s, bar 0.150); seed 1 fail (平均速度 averaged 0.079 m/s, bar 0.150); seed 2 fail (平均速度 averaged 0.100 m/s, bar 0.150)",
    "covers_ground_in_sixteen_seconds": "seed 0 fail (walked 0.658 m, bar 1.000 m); seed 1 fail (walked 0.622 m, bar 1.000 m); seed 2 fail (walked 0.655 m, bar 1.000 m)",
    "turns_towards_a_sound_while_walking": "seed 0 PASS (边走边听: the two sides came apart by -0.831 rad); seed 1 PASS (边走边听: the two sides came apart by -0.747 rad); seed 2 PASS (边走边听: the two sides came apart by -0.576 rad)",
    "turns_towards_a_sound_while_standing": "seed 0 PASS (the two sides came apart by -0.831 rad); seed 1 PASS (the two sides came apart by -0.747 rad); seed 2 PASS (the two sides came apart by -0.576 rad)",
    "does_not_spin_on_the_spot": "seed 0 fail (没人管它时 turned through 1.154 rad (bar 0.600)); seed 1 fail (没人管它时 turned through 1.031 rad (bar 0.600)); seed 2 fail (没人管它时 turned through 1.726 rad (bar 0.600))",
    "walks_down_the_ramp": "seed 0 PASS (came off 坡: walked 0.630 m, 0.564 m past the edge); seed 1 PASS (came off 坡: walked 0.637 m, 0.571 m past the edge); seed 2 PASS (came off 坡: walked 0.585 m, 0.549 m past the edge)",
    "steps_down_the_low_step": "seed 0 PASS (came off 台阶: walked 0.297 m, 0.304 m past the edge); seed 1 PASS (came off 台阶: walked 0.418 m, 0.417 m past the edge); seed 2 PASS (came off 台阶: walked 0.550 m, 0.536 m past the edge)",
    "steps_down_from_the_platform": "seed 0 fail (walked 0.501 m but never came off 台子 (reached 0.448 m)); seed 1 fail (walked 0.508 m but never came off 台子 (reached 0.458 m)); seed 2 fail (walked 0.576 m but never came off 台子 (reached 0.538 m))",
    "goes_around_the_block": "seed 0 fail (walked 0.382 m, bar 0.600 m); seed 1 fail (walked 0.617 m but never got past the block (reached 0.372 m)); seed 2 fail (walked 0.129 m, bar 0.600 m)",
    "a_touch_on_the_front_holds_it_back": "seed 0 fail (it fell over in one of the two runs (free min up_z 0.994, held min up_z -0.979)); seed 1 fail (it fell over in one of the two runs (free min up_z 0.990, held min up_z -0.969)); seed 2 fail (it fell over in one of the two runs (free min up_z 0.992, held min up_z -0.999))",
    "a_bang_stops_it_walking": "seed 0 fail (那一声砰 barely slowed it: 0.425 m against 0.556 m free (limit 0.389)); seed 1 fail (那一声砰 barely slowed it: 0.405 m against 0.511 m free (limit 0.358)); seed 2 fail (那一声砰 barely slowed it: 0.396 m against 0.462 m free (limit 0.323))",
    "gets_up_after_a_hard_shove": "seed 0 PASS (on its feet from 3.97 s and still there at the end (final up_z 0.696)); seed 1 fail (got up at 3.39 s and then went back down (final up_z 0.309)); seed 2 PASS (on its feet from 4.01 s and still there at the end (final up_z 0.821))",
    "eyes_converge_on_a_near_thing": "seed 0 fail (gain = 0.00564, bar 0.02000); seed 1 fail (gain = 0.00564, bar 0.02000); seed 2 fail (gain = 0.00564, bar 0.02000)",
    "eyes_look_at_a_still_thing_at_the_side": "seed 0 PASS (停在旁边的东西: the eyes went 0.1083 and -0.1156 rad); seed 1 PASS (停在旁边的东西: the eyes went 0.1083 and -0.1156 rad); seed 2 PASS (停在旁边的东西: the eyes went 0.1083 and -0.1156 rad)",
    "its_own_eyes_are_felt": "seed 0 PASS (gain = 0.00065); seed 1 PASS (gain = 0.00065); seed 2 PASS (gain = 0.00065)",
    "eyes_hold_a_slow_thing": "seed 0 PASS (looked 0.080 rad off the ball); seed 1 PASS (looked 0.080 rad off the ball); seed 2 PASS (looked 0.080 rad off the ball)",
    "ears_hear_something_far_away": "seed 0 PASS (远处那一声 came apart by 0.04687); seed 1 PASS (远处那一声 came apart by 0.04687); seed 2 PASS (远处那一声 came apart by 0.04687)",
    "ears_hear_a_sound_behind": "seed 0 PASS (后面那一声 came apart by 0.04115); seed 1 PASS (后面那一声 came apart by 0.04115); seed 2 PASS (后面那一声 came apart by 0.04115)",
    "two_sounds_at_once_are_still_two": "seed 0 PASS (两处一起响 came apart by 0.15975); seed 1 PASS (两处一起响 came apart by 0.15975); seed 2 PASS (两处一起响 came apart by 0.15975)",
    "feels_a_touch_on_the_left_side": "seed 0 PASS (左边和右边被碰 moved 0.97603 and 0.97603); seed 1 PASS (左边和右边被碰 moved 0.97603 and 0.97603); seed 2 PASS (左边和右边被碰 moved 0.97603 and 0.97603)",
    "feels_a_touch_behind_it": "seed 0 PASS (后面和前面被碰 moved 0.96173 and 0.96173); seed 1 PASS (后面和前面被碰 moved 0.96173 and 0.96173); seed 2 PASS (后面和前面被碰 moved 0.96173 and 0.96173)",
    "feels_a_touch_on_both_sides_at_once": "seed 0 PASS (one touch lit 1 sector and two touches lit 2); seed 1 PASS (one touch lit 1 sector and two touches lit 2); seed 2 PASS (one touch lit 1 sector and two touches lit 2)",
    "feels_the_ground_under_the_back_legs": "seed 0 PASS (承重的后腿 moved 0.09023 and 0.09023); seed 1 PASS (承重的后腿 moved 0.09023 and 0.09023); seed 2 PASS (承重的后腿 moved 0.09023 and 0.09023)",
    "learns_the_lesson_in_five_seconds": "seed 0 PASS (tone alone now drives retreat 0.626 over silence); seed 1 PASS (tone alone now drives retreat 0.626 over silence); seed 2 PASS (tone alone now drives retreat 0.626 over silence)",
    "the_lesson_survives_a_longer_pause": "seed 0 PASS (tone alone now drives retreat 0.945 over silence); seed 1 PASS (tone alone now drives retreat 0.945 over silence); seed 2 PASS (tone alone now drives retreat 0.945 over silence)",
    "remembers_a_second_pattern": "seed 0 PASS (the pattern drives the aversive cells 0.0761 over silence); seed 1 PASS (the pattern drives the aversive cells 0.0761 over silence); seed 2 PASS (the pattern drives the aversive cells 0.0761 over silence)",
    "remembers_after_one_pairing": "seed 0 PASS (the pattern drives the aversive cells 0.1324 over silence); seed 1 PASS (the pattern drives the aversive cells 0.1324 over silence); seed 2 PASS (the pattern drives the aversive cells 0.1324 over silence)",
    "stands_still_for_twenty_seconds": "seed 0 PASS (stood for the whole twenty seconds: 0.654 m of drift, min up_z 0.992); seed 1 PASS (stood for the whole twenty seconds: 0.620 m of drift, min up_z 0.992); seed 2 PASS (stood for the whole twenty seconds: 0.786 m of drift, min up_z 0.991)",
    "stands_still_on_a_slope": "seed 0 PASS (stayed upright on the slope (min up_z 0.998)); seed 1 PASS (stayed upright on the slope (min up_z 0.998)); seed 2 PASS (stayed upright on the slope (min up_z 0.998))",
    "stays_up_when_nudged_gently": "seed 0 PASS (on its feet from 3.59 s and still there at the end (final up_z 0.897)); seed 1 PASS (on its feet from 0.50 s and still there at the end (final up_z 1.000)); seed 2 PASS (on its feet from 0.50 s and still there at the end (final up_z 1.000))",
    "all_four_feet_leave_the_ground": "seed 0 PASS (all four feet lifted (front 0.279/0.153 rear 0.176/0.296)); seed 1 PASS (all four feet lifted (front 0.237/0.149 rear 0.165/0.505)); seed 2 PASS (all four feet lifted (front 0.198/0.069 rear 0.183/0.265))",
    "the_two_sides_step_alike": "seed 0 PASS (both pairs stepped alike (front 0.279/0.153, rear 0.176/0.296)); seed 1 fail (one leg of a pair did the lifting (front 0.237/0.149, rear 0.165/0.505, bar 0.200)); seed 2 PASS (both pairs stepped alike (front 0.198/0.069, rear 0.183/0.265))",
    "keeps_walking_for_thirty_seconds": "seed 0 fail (walked 0.651 m, bar 1.300 m); seed 1 fail (walked 0.618 m, bar 1.300 m); seed 2 fail (walked 0.923 m, bar 1.300 m)",
    "does_not_slow_down_on_a_long_walk": "seed 0 fail (slowed to 0.195 of its starting pace (0.094 m/s then, 0.018 m/s later, bar 0.500)); seed 1 fail (slowed to 0.075 of its starting pace (0.091 m/s then, 0.007 m/s later, bar 0.500)); seed 2 fail (slowed to 0.436 of its starting pace (0.078 m/s then, 0.034 m/s later, bar 0.500))",
    "keeps_walking_while_it_turns": "seed 0 PASS (一边走一边响: turned it -1.213 rad and it kept walking (0.648 m and 0.773 m)); seed 1 PASS (一边走一边响: turned it -1.093 rad and it kept walking (0.621 m and 0.769 m)); seed 2 PASS (一边走一边响: turned it -0.595 rad and it kept walking (0.600 m and 0.907 m))",
    "backs_away_from_a_hand_on_its_chest": "seed 0 fail (胸口上那只手 did not back it off (0.001 m, bar 0.100)); seed 1 fail (胸口上那只手 did not back it off (0.001 m, bar 0.100)); seed 2 fail (胸口上那只手 did not back it off (0.001 m, bar 0.100))",
    "keeps_its_line_with_a_sound_behind": "seed 0 PASS (kept its line with a sound behind it (0.219 rad of drift)); seed 1 fail (the sound behind swung its heading 0.534 rad (bar 0.350)); seed 2 fail (the sound behind swung its heading 0.531 rad (bar 0.350))",
    "does_not_fall_walking_into_the_wall": "seed 0 PASS (stayed on its feet against the wall (reached x 2.905, min up_z 0.995)); seed 1 PASS (stayed on its feet against the wall (reached x 2.904, min up_z 0.996)); seed 2 PASS (stayed on its feet against the wall (reached x 2.831, min up_z 0.996))",
    "threads_the_passage_without_touching": "seed 0 fail (walked 0.268 m but never came out of the passage (0.403 m past the start)); seed 1 fail (walked 0.281 m but never came out of the passage (0.429 m past the start)); seed 2 fail (not upright the whole way (min up_z -0.269, min height 0.120))",
    "sees_a_red_square_from_a_green_one": "seed 0 PASS (红方块和绿方块 came apart by 0.11737); seed 1 PASS (红方块和绿方块 came apart by 0.11737); seed 2 PASS (红方块和绿方块 came apart by 0.11737)",
    "sees_a_big_ball_from_a_small_one": "seed 0 PASS (大球和小球 came apart by 0.14810); seed 1 PASS (大球和小球 came apart by 0.14810); seed 2 PASS (大球和小球 came apart by 0.14810)",
    "eyes_follow_the_ball_up_and_down": "seed 0 PASS (the eye pitch stayed 0.116 rad from the ball); seed 1 PASS (the eye pitch stayed 0.116 rad from the ball); seed 2 PASS (the eye pitch stayed 0.116 rad from the ball)",
    "eyes_follow_the_ball_far_away": "seed 0 PASS (looked 0.054 rad off the ball); seed 1 PASS (looked 0.054 rad off the ball); seed 2 PASS (looked 0.054 rad off the ball)",
    "eyes_follow_the_ball_close_up": "seed 0 PASS (looked 0.120 rad off the ball); seed 1 PASS (looked 0.120 rad off the ball); seed 2 PASS (looked 0.120 rad off the ball)",
    "the_eyes_let_go_when_it_is_gone": "seed 0 PASS (it looked (0.1081 rad) and then let go (0.0054 rad)); seed 1 PASS (it looked (0.1081 rad) and then let go (0.0054 rad)); seed 2 PASS (it looked (0.1081 rad) and then let go (0.0054 rad))",
    "notices_a_thing_that_creeps_in": "seed 0 PASS (the eyes swung 0.0430 rad towards the thing that appeared); seed 1 PASS (the eyes swung 0.0430 rad towards the thing that appeared); seed 2 PASS (the eyes swung 0.0430 rad towards the thing that appeared)",
    "a_sound_that_stops_is_no_longer_heard": "seed 0 PASS (那一声 fell from 0.0833 to 0.0000 when it stopped); seed 1 PASS (那一声 fell from 0.0833 to 0.0000 when it stopped); seed 2 PASS (那一声 fell from 0.0833 to 0.0000 when it stopped)",
    "a_quiet_room_is_not_a_sound": "seed 0 PASS (安静的屋子: every channel that received nothing had its opposite cell lit); seed 1 PASS (安静的屋子: every channel that received nothing had its opposite cell lit); seed 2 PASS (安静的屋子: every channel that received nothing had its opposite cell lit)",
    "ears_tell_left_from_right": "seed 0 PASS (左肩和右肩的声音 came apart by 0.05895); seed 1 PASS (左肩和右肩的声音 came apart by 0.05895); seed 2 PASS (左肩和右肩的声音 came apart by 0.05895)",
    "two_sounds_at_one_shoulder_are_still_two": "seed 0 PASS (同一边两个声音 came apart by 0.12421); seed 1 PASS (同一边两个声音 came apart by 0.12421); seed 2 PASS (同一边两个声音 came apart by 0.12421)",
    "a_bang_behind_it_still_startles": "seed 0 PASS (背后那一声 came apart by 0.22222); seed 1 PASS (背后那一声 came apart by 0.22222); seed 2 PASS (背后那一声 came apart by 0.22222)",
    "feels_a_touch_on_the_right_side": "seed 0 PASS (右边和左边被碰 moved 0.97603 and 0.97603); seed 1 PASS (右边和左边被碰 moved 0.97603 and 0.97603); seed 2 PASS (右边和左边被碰 moved 0.97603 and 0.97603)",
    "feels_a_light_hand_too": "seed 0 PASS (轻轻搭上来 moved 0.94121 and 0.94121); seed 1 PASS (轻轻搭上来 moved 0.94121 and 0.94121); seed 2 PASS (轻轻搭上来 moved 0.94121 and 0.94121)",
    "feels_the_ground_under_the_front_legs": "seed 0 PASS (承重的前腿 moved 0.09023 and 0.09023); seed 1 PASS (承重的前腿 moved 0.09023 and 0.09023); seed 2 PASS (承重的前腿 moved 0.09023 and 0.09023)",
    "lifts_each_of_the_four_feet": "seed 0 PASS (all four legs reported themselves (['0.684', '0.684', '0.684', '0.684'])); seed 1 PASS (all four legs reported themselves (['0.684', '0.684', '0.684', '0.684'])); seed 2 PASS (all four legs reported themselves (['0.684', '0.684', '0.684', '0.684']))",
    "the_flat_ground_is_not_a_bump": "seed 0 PASS (平地 stayed at ['0.0000', '0.0000', '0.0000', '0.0000']); seed 1 PASS (平地 stayed at ['0.0000', '0.0000', '0.0000', '0.0000']); seed 2 PASS (平地 stayed at ['0.0000', '0.0000', '0.0000', '0.0000'])",
    "it_forgets_when_nobody_teaches_anymore": "seed 0 fail (the lesson never faded: 0.970 while the hand was there, 0.971 with nobody teaching (has to fall to 0.679)); seed 1 fail (the lesson never faded: 0.970 while the hand was there, 0.971 with nobody teaching (has to fall to 0.679)); seed 2 fail (the lesson never faded: 0.970 while the hand was there, 0.971 with nobody teaching (has to fall to 0.679))",
    "a_second_short_lesson_adds_to_the_first": "seed 0 PASS (the second lesson added 0.246 (0.626 then 0.872)); seed 1 PASS (the second lesson added 0.246 (0.626 then 0.872)); seed 2 PASS (the second lesson added 0.246 (0.626 then 0.872))",
    "the_lesson_is_still_there_after_a_minute": "seed 0 PASS (tone alone now drives retreat 0.944 over silence); seed 1 PASS (tone alone now drives retreat 0.944 over silence); seed 2 PASS (tone alone now drives retreat 0.944 over silence)",
    "the_memory_survives_a_long_life": "seed 0 PASS (the tone alone drove retreat 0.970 right after the lesson and 0.945 after 60 s of life with the rule running (kept 0.975)); seed 1 PASS (the tone alone drove retreat 0.970 right after the lesson and 0.945 after 60 s of life with the rule running (kept 0.975)); seed 2 PASS (the tone alone drove retreat 0.970 right after the lesson and 0.945 after 60 s of life with the rule running (kept 0.975))",
    "a_second_tone_does_not_wipe_the_first": "seed 0 PASS (the first tone still drove retreat 0.942 after the second was learned (0.970 before it, kept 0.972)); seed 1 PASS (the first tone still drove retreat 0.942 after the second was learned (0.970 before it, kept 0.972)); seed 2 PASS (the first tone still drove retreat 0.942 after the second was learned (0.970 before it, kept 0.972))",
    "the_route_stops_growing_at_its_ceiling": "seed 0 PASS (the route rose 1.98657 then 1.11257 then 0.02656 over equal 20 s stretches, ending at [1.5, 0.0528, 0.0002, 1.5, 0.0718, 0.0011]); seed 1 PASS (the route rose 1.98657 then 1.11257 then 0.02656 over equal 20 s stretches, ending at [1.5, 0.0528, 0.0002, 1.5, 0.0718, 0.0011]); seed 2 PASS (the route rose 1.98657 then 1.11257 then 0.02656 over equal 20 s stretches, ending at [1.5, 0.0528, 0.0002, 1.5, 0.0718, 0.0011])",
    "remembers_two_patterns_at_once": "seed 0 PASS (paired 0.0761, unpaired -0.0002); seed 1 PASS (paired 0.0761, unpaired -0.0002); seed 2 PASS (paired 0.0761, unpaired -0.0002)",
    "the_memory_survives_a_small_change": "seed 0 PASS (the moved copy still meant 0.0934 of the original 0.0761); seed 1 PASS (the moved copy still meant 0.0934 of the original 0.0761); seed 2 PASS (the moved copy still meant 0.0934 of the original 0.0761)",
    "walks_towards_a_bright_thing": "seed 0 PASS (a bright thing in front pulled it 0.077 m further); seed 1 PASS (a bright thing in front pulled it 0.073 m further); seed 2 PASS (a bright thing in front pulled it 0.184 m further)",
    "knows_when_it_is_tipped_over": "seed 0 fail (gain_mean = 0.00000, bar 0.05000); seed 1 fail (gain_mean = 0.00000, bar 0.05000); seed 2 fail (gain_mean = 0.00000, bar 0.05000)",
    "the_hidden_layer_sees_the_picture": "seed 0 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000); seed 1 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000); seed 2 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000)",
    "the_compressed_layer_sees_the_picture": "seed 0 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000); seed 1 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000); seed 2 fail (横条纹和竖条纹 did not come apart: 0.00000, bar 0.01000)",
    "hurries_towards_a_bright_thing": "seed 0 fail (前面有亮东西时: with an empty room it does not even hold a steady pace (0.086 m/s, bar 0.150)); seed 1 fail (前面有亮东西时: with an empty room it does not even hold a steady pace (0.084 m/s, bar 0.150)); seed 2 fail (前面有亮东西时: with an empty room it does not even hold a steady pace (0.096 m/s, bar 0.150))",
    "runs_without_being_told": "seed 0 fail (没人管它的那十秒 averaged 0.087 m/s, bar 0.250); seed 1 fail (没人管它的那十秒 averaged 0.079 m/s, bar 0.250); seed 2 fail (没人管它的那十秒 averaged 0.100 m/s, bar 0.250)",
    "runs_for_twenty_seconds": "seed 0 fail (walked 0.654 m, bar 2.000 m); seed 1 fail (walked 0.620 m, bar 2.000 m); seed 2 fail (walked 0.786 m, bar 2.000 m)",
    "keeps_running_while_it_turns": "seed 0 fail (一边跑一边响: it turned but crawled (0.072 m/s, bar 0.250)); seed 1 fail (一边跑一边响: it turned but crawled (0.073 m/s, bar 0.250)); seed 2 fail (一边跑一边响: it turned but crawled (0.077 m/s, bar 0.250))",
    "nothing_touching_it_is_not_nothing": "seed 0 PASS (没被碰的站姿: every channel that received nothing had its opposite cell lit); seed 1 PASS (没被碰的站姿: every channel that received nothing had its opposite cell lit); seed 2 PASS (没被碰的站姿: every channel that received nothing had its opposite cell lit)",
    "the_weight_pair_still_carries_the_weight": "seed 0 PASS (四只脚的受力那一对: foot_load and its opposite add up to one channel (worst 0.000 off)); seed 1 PASS (四只脚的受力那一对: foot_load and its opposite add up to one channel (worst 0.000 off)); seed 2 PASS (四只脚的受力那一对: foot_load and its opposite add up to one channel (worst 0.000 off))",
    "the_dark_room_is_not_silence": "seed 0 PASS (全黑画面: black lit the dark half 0.96985 (summary 0.97143) and left the bright half at 0.00000); seed 1 PASS (全黑画面: black lit the dark half 0.96985 (summary 0.97143) and left the bright half at 0.00000); seed 2 PASS (全黑画面: black lit the dark half 0.96985 (summary 0.97143) and left the bright half at 0.00000)",
}
TASKS = []


def task(name, group, question, stage, sim_seconds, measure, bar, note,
         reads="", bar_text="", requires=()):
    """Register one question.

    ``requires`` names the tasks that come before it on the ladder.  A rung is
    only unlocked when every rung under it was passed; see ``ladder()``.
    """
    TASKS.append(dict(name=name, group=group, question=question, stage=stage,
                      sim_seconds=sim_seconds, measure=measure, bar=bar, note=note,
                      reads=reads, bar_text=bar_text, requires=tuple(requires)))
    return measure


def reference_parameters(genome=None):
    """The parameters one exam run uses: today's animal, at reference size."""
    values = controller_parameters(load_live_parameters())
    values.update(REFERENCE_SIZE)
    values.pop("model", None)
    if genome:
        unknown = sorted(set(genome) - set(values))
        if unknown:
            raise ValueError("genome names not parameter names: %s" % unknown)
        values.update({name: float(value) for name, value in genome.items()})
    return values


def revision():
    """Which animal the gains describe, as the config file spells it.

    A ledger row is only worth anything if it says which animal it measured.
    The wiring of this project has already changed twice under the same task
    names - the suppression wiring, then the association return line - and both
    times the readings moved.  ``context`` carries it so every record can.
    """
    with (ROOT / "live_config.json").open(encoding="utf-8-sig") as source:
        config = json.load(source)
    return str(config.get("revision", ""))


def context(genome, seed):
    return dict(genome=dict(genome or {}), seed=int(seed), revision=revision(),
                parameters=reference_parameters(genome), model_path=ARENA)


def blank_environment():
    environment = {name: np.zeros(4) for name in ("body_touch", "foot_obstacle",
                                                  "foot_load", "foot_slip")}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    return environment


def one_foot(name, index, value):
    """One environment channel driving exactly one leg, as reflex senses would."""
    environment = blank_environment()
    environment[name][index] = value
    return environment


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def ladder(tasks=None):
    """Depth of every rung: how many steps the longest prerequisite chain has."""
    entries = list(TASKS if tasks is None else tasks)
    depth, waiting = {}, {entry["name"]: set(entry["requires"]) for entry in entries}
    known = set(waiting)
    for entry in entries:
        missing = sorted(set(entry["requires"]) - known)
        if missing:
            raise ValueError("%s needs tasks that are not in the bank: %s"
                             % (entry["name"], missing))
    while waiting:
        ready = [name for name, needs in waiting.items() if not (needs & set(waiting))]
        if not ready:
            raise ValueError("the ladder has a cycle through %s" % sorted(waiting))
        for name in ready:
            depth[name] = 0 if not waiting[name] else 1 + max(depth[n] for n in waiting[name])
            del waiting[name]
    return depth


def unlocked(passed, tasks=None):
    """The rungs whose prerequisites are all in ``passed``."""
    passed = set(passed)
    return [entry["name"] for entry in (TASKS if tasks is None else tasks)
            if set(entry["requires"]) <= passed]


def frontier(passed, tasks=None):
    """The rungs that are open now but not yet passed: what to try next."""
    passed = set(passed)
    return [name for name in unlocked(passed, tasks) if name not in passed]


def deepest_reached(passed, tasks=None):
    """The highest rung number the animal got to, and the rungs on it."""
    passed = set(passed)
    depth = ladder(tasks)
    reached = [depth[name] for name in passed if name in depth]
    if not reached:
        return 0, []
    top = max(reached)
    return top, sorted(name for name in passed if depth.get(name) == top)


# ---------------------------------------------------------------------------
# the body, the brain, the eyes, the ears
# ---------------------------------------------------------------------------
def clean_body(model_path=None):
    """The arena with its loose props moved far away: nothing but the floor."""
    body = Go2Body(model_path=model_path or ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    return body


def _place(body, pose="stand", tilt=None, noise=.01):
    """Reset, and put the lowest part of the robot just above the floor.

    A tilt on a standing rig drives a shoulder into the ground; probe_righting
    lifts the base by the same amount before it measures a fall, and these
    scenes want the same starting state.
    """
    body.reset(pose=pose, seed=0, joint_noise=noise, tilt=tilt)
    if tilt is not None:
        robot = np.zeros(body.model.ngeom, dtype=bool)
        robot[body._feet] = True
        for geom in range(body.model.ngeom):
            if body.model.geom_bodyid[geom] != 0:
                robot[geom] = True
        lowest = float(np.min(body.data.geom_xpos[robot, 2] - body.model.geom_rbound[robot]))
        body.data.qpos[body._base_qpos + 2] += float(body.model.geom_pos[body._floor, 2]) + .002 - lowest
        mujoco.mj_forward(body.model, body.data)
    return body


def brain_for(ctx, body, eyes=False):
    """The controller at reference size, drawing the same scaffold as the run."""
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=ctx["seed"],
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **ctx["parameters"])
    return (brain, RawEyes(body, width=brain.eye_width, height=brain.eye_height)) if eyes else brain


def mean_cells(brain, names):
    """Tail reading of every named group, as one flat dict of floats."""
    return {name: float(np.mean(brain.network.activity[np.atleast_1d(brain.groups[name])]))
            for name in names}


def _source(body, geom_name, place):
    geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    body.model.geom_pos[geom] = list(place)
    mujoco.mj_forward(body.model, body.data)
    return geom


def cell_read(ctx, *, cells, seconds=2., environment=None, drive=None, autonomy=False,
              startle=0., dopamine=0., pose="stand", tilt=None, clean=False,
              learn=False, tail=.5, ear=None, picture=None, gaze=None, move=False,
              look=False, scene=None, eyes_move=False, gaze_size=None, props=None):
    """Put the animal in one fixed situation and read named cell groups.

    The body is held at its starting pose unless ``move`` is set: this reads
    what the cells do with a given world handed to them, not what the animal
    then chooses to do.  ``environment`` may be a full senses dictionary or a
    plain dict of scalars per channel; ``ear`` is a (x, y) for the low emitter;
    ``picture`` is a raw two-eye RGB image; ``gaze`` is a (x, y, z) for the
    green ball.  ``look`` lets the animal take its own picture off the eyes, and
    ``scene`` is a callable of time returning {"gaze": ..., "picture": ...,
    "environment": ...} overrides, for a scene that has to move.  Every scene is
    deterministic in the exam seed.

    ``eyes_move`` is the difference between asking what the eye cells would do
    and asking what the eyes then actually do.  With it off the head is held
    still and the eye muscles never travel, so every cell that reads the eye's
    own angle - convergence, the distance bank, eye proprioception - reports
    only its resting bias.  With it on the commanded eye angle is written to the
    eye joints each step and the body steps, which is the same thing
    ``gaze_measure`` does; the legs are still held at home, so the animal is not
    walking anywhere.  ``gaze_size`` widens the ball, for scenes where the thing
    approaching has to be big enough to fill a band.
    """
    from born_wired.reflex_senses import ReflexSenses
    body = clean_body(ctx["model_path"]) if clean else Go2Body(model_path=ctx["model_path"])
    _place(body, pose=pose, tilt=tilt)
    if props:
        for prop_name, place in props.items():
            _source(body, prop_name, place)
    if ear is not None and ear is not False:
        far = (60., 60., -8.)
        _source(body, "sound_low", far)
        _source(body, "sound_high", far)
        _source(body, ear[2] if len(ear) > 2 else "sound_low", (ear[0], ear[1], .30))
    if gaze is not None:
        _source(body, "green_target", gaze)
        body.model.geom_size[mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM,
                                               "green_target")] = [gaze_size or .06] * 3
    brain, eyes = brain_for(ctx, body, eyes=True)
    senses = ReflexSenses(body)
    observation = body.observe()
    if environment is None:
        environment = senses.observe()
    else:
        environment = {name: np.broadcast_to(np.asarray(value, dtype=float), (4,)).copy()
                       for name, value in environment.items()}
        environment.setdefault("foot_support", senses.observe()["foot_support"])
    waveform, ears = None, None
    if ear is not None:
        ears = BinauralSenses(body)
    steps = int(round(seconds / DT))
    window = max(1, int(round(tail / DT)))
    history = {name: [] for name in cells}
    try:
        for index in range(steps):
            frame_picture, frame_environment = picture, environment
            if scene is not None:
                frame = scene(index * DT)
                if "gaze" in frame:
                    _source(body, "green_target", frame["gaze"])
                if "environment" in frame:
                    frame_environment = frame["environment"]
                if "picture" in frame:
                    frame_picture = frame["picture"]
                if "size" in frame:
                    body.model.geom_size[mujoco.mj_name2id(
                        body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")] = [frame["size"]] * 3
            if look:
                frame_picture = eyes.observe_raw()
            if ears is not None:
                waveform = ears.observe()
            activation = brain.step(observation, environment=frame_environment,
                                    ear_waveform=waveform, eye_pixels=frame_picture,
                                    dt=DT, learn=learn, autonomy=autonomy,
                                    startle=startle, dopamine_drive=dopamine,
                                    locomotion=0. if drive is None else drive)[1]
            for name in cells:
                history[name].append(np.asarray(
                    brain.network.activity[np.atleast_1d(brain.groups[name])], dtype=float))
            if eyes_move:
                body.command_eyes(brain.eye_command())
            if move or eyes_move:
                observation = body.step(np.asarray(body.home_angles), duration=DT,
                                        activation=activation)
    finally:
        eyes.close()
    parts = {name: np.mean(np.asarray(values[-window:]), axis=0).tolist()
             for name, values in history.items()}
    reading = {name: float(np.mean(parts[name])) for name in cells}
    reading["parts"] = parts
    up_z = float(body.data.xmat[body._base].reshape(3, 3)[2, 2])
    reading.update(status="ok", error=None, up_z=up_z,
                   height=float(body.data.xpos[body._base, 2]))
    return reading


def contrast_read(ctx, high, low, cells):
    """One situation minus another; ``high`` and ``low`` are cell_read keywords."""
    first = cell_read(ctx, **dict(high, cells=cells))
    second = cell_read(ctx, **dict(low, cells=cells))
    if first["status"] != "ok" or second["status"] != "ok":
        return dict(status="error", error=first.get("error") or second.get("error"))
    gain = {name: first[name] - second[name] for name in cells}
    # How far the two situations push the individual cells of a group apart,
    # with the sign thrown away: for a group whose cells swap over rather than
    # rise (the pinna pair), the mean difference can be zero while the pattern
    # is completely different.
    separation = {name: float(np.mean(np.abs(
        np.asarray(first["parts"][name], dtype=float)
        - np.asarray(second["parts"][name], dtype=float)))) for name in cells}
    return dict(status="ok", error=None, high=first, low=second, gain=gain,
                separation=separation)
# ---------------------------------------------------------------------------
# moving: walking, terrain, getting up
# ---------------------------------------------------------------------------
PEAK_GROUPS = ("wall_contact", "wall_turn", "stumble", "clearance", "brake",
               "avoidance", "withdrawal", "lean", "steady", "startle", "near",
               "righting")


def walk_measure(ctx, seconds, scenario, terrain, injected=None, props=None,
                 start_yaw=None, startle=None):
    result = run_seed(seed=ctx["seed"], duration=seconds, model_path=ctx["model_path"],
                      scenario=scenario,
                      controller_parameters_for_seed=ctx["parameters"],
                      terrain_start=terrain, injected=injected, props=props,
                      start_yaw=start_yaw, startle=startle)
    metrics = result["metrics"]
    timeline = result["timeline"]
    peaks, tails = {}, {}
    for name in PEAK_GROUPS:
        series = [float(np.max(np.atleast_1d(sample["reflex_activity"][name])))
                  for sample in timeline if name in sample["reflex_activity"]]
        if series:
            peaks[name] = float(np.max(series))
            cut = max(1, len(series) // 5)
            tails[name] = float(np.mean(series[-cut:]))
    return dict(status=result["status"], error=result["error"],
                behavior_pass=bool(result["behavior_pass"]),
                displacement_m=metrics["total_displacement_m"],
                path_m=metrics["horizontal_path_m"],
                maximum_x=metrics["maximum_world_x"],
                min_up_z=metrics["minimum_up_z"],
                min_height_m=metrics["minimum_height_m"],
                crossed_marker=metrics["crossed_test_marker"],
                slip=metrics["contact_conditioned_foot_slip"],
                clearance=metrics["foot_clearance_gt_2mm_fraction"],
                drive_share=metrics["fraction_drive_gt_0_35"],
                progress_m=metrics["maximum_progress_m"],
                back_m=metrics["minimum_progress_m"],
                yaw_rad=_wrapped(metrics["final_yaw_rad"]),
                yaw_path_rad=metrics["yaw_path_rad"],
                lateral_m=metrics["lateral_m"],
                straightness=metrics["straightness"],
                max_speed_mps=metrics["max_speed_mps"],
                mean_speed_mps=metrics["mean_speed_mps"],
                peak=peaks, tail=tails,
                clearance_parts=metrics["foot_clearance_gt_2mm_fraction"],
                slip_parts=metrics["contact_conditioned_foot_slip"])


def _wrapped(angle):
    """An angle in (-pi, pi]: a heading of 3.2 rad is -3.08 rad, not 3.2."""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def walk_bar(measures, distance_key):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if not measures["behavior_pass"]:
        return False, ("not upright the whole way (min up_z %.3f, min height %.3f)"
                       % (measures["min_up_z"], measures["min_height_m"]))
    if measures["displacement_m"] < BAR[distance_key]:
        return False, ("walked %.3f m, bar %.3f m"
                       % (measures["displacement_m"], BAR[distance_key]))
    return True, "walked %.3f m, stayed up" % measures["displacement_m"]


def marker_bar(measures, distance_key):
    passed, why = walk_bar(measures, distance_key)
    if not passed:
        return False, why
    if not measures["crossed_marker"]:
        return False, ("walked %.3f m but never crossed the far side of the thing in "
                       "the way (reached x %.3f)" % (measures["displacement_m"],
                                                     measures["maximum_x"]))
    return True, ("walked over it: %.3f m, reached x %.3f"
                  % (measures["displacement_m"], measures["maximum_x"]))


def wall_bar(measures, distance_key):
    """It must walk, and it must not come out the far side of the wall.

    The wall-contact cells are read and reported, but they are not part of the
    bar: on the birth animal they never rise at all, and this task is not the
    place to pretend otherwise.  What is being asked is the weaker thing - a
    wall stops it.
    """
    passed, why = walk_bar(measures, distance_key)
    if not passed:
        return False, why
    if measures["maximum_x"] > 3.42:
        return False, "walked through the wall (reached x %.3f)" % measures["maximum_x"]
    return True, ("walked %.3f m and stopped at x %.3f (wall cells peaked at %.4f)"
                  % (measures["displacement_m"], measures["maximum_x"],
                     measures["peak"].get("wall_contact", 0.)))


def quiet_bar(measures):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < .9:
        return False, "lost its footing while idle (min up_z %.3f)" % measures["min_up_z"]
    if measures["displacement_m"] > BAR["quiet_idle_m"]:
        return False, ("wandered %.3f m with nothing driving it (bar %.3f)"
                       % (measures["displacement_m"], BAR["quiet_idle_m"]))
    return True, "stayed put at %.3f m" % measures["displacement_m"]


def righting_measure(ctx, pose, seconds, impact, force=None):
    result = probe_righting.run_seed(seed=ctx["seed"], pose=pose, duration=seconds,
                                     model_path=ctx["model_path"], parameters=ctx["parameters"],
                                     impact=impact, force=force)
    metrics = result["metrics"]
    peaks = {}
    for name in probe_righting.GROUPS:
        series = [float(np.max(np.atleast_1d(sample[name])))
                  for sample in result["trace"] if name in sample]
        if series:
            peaks[name] = float(np.max(series))
    return dict(status=result["status"], error=result["error"], pose=pose,
                recovered=bool(metrics["recovered"]),
                final_up_z=metrics["final_up_z"], final_height=metrics["final_height"],
                minimum_up_z=metrics["minimum_up_z"],
                time_to_feet_s=metrics["time_to_feet_s"],
                on_feet_share=metrics["on_feet_share"], peak=peaks)


def righting_bar(measures):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if not measures["recovered"]:
        return False, ("never stayed on its feet (final up_z %.3f, height %.3f)"
                       % (measures["final_up_z"], measures["final_height"]))
    if measures["final_up_z"] < .5:
        return False, ("got up at %.2f s and then went back down (final up_z %.3f)"
                       % (measures["time_to_feet_s"], measures["final_up_z"]))
    return True, ("on its feet from %.2f s and still there at the end (final up_z %.3f)"
                  % (measures["time_to_feet_s"], measures["final_up_z"]))


# ---------------------------------------------------------------------------
# reading cells the plain way: one situation, one bar
# ---------------------------------------------------------------------------
def retention_bar(measures, key, above_key, fraction, what):
    """The compressed layer must keep a stated share of what the layer above had.

    An absolute bar here is a knife edge: the bank under the eye reads 0.0103
    apart on the two pictures against a bar of 0.01, a margin three per cent
    wide, and a bar that tight decides nothing.  Asking for a share of the layer
    above says the thing the question is actually about - how much of the
    picture survives the squeeze - and it does not depend on how hard the
    pictures are.
    """
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures.get("separation", {}).get(key)
    above = measures.get("separation", {}).get(above_key)
    if value is None or above is None:
        return False, "no reading for %s or %s" % (key, above_key)
    kept = value / above if above > 1e-12 else 0.
    if kept < fraction:
        return False, ("%s kept only %.1f%% of what %s carried (%.5f of %.5f, bar %.0f%%)"
                       % (what, 100. * kept, above_key, value, above, 100. * fraction))
    return True, ("%s kept %.1f%% of what %s carried (%.5f of %.5f)"
                  % (what, 100. * kept, above_key, value, above))


def separation_bar(measures, key, bar_key, what):
    """A group's cells must land in a different pattern in the two situations."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures.get("separation", {}).get(key)
    if value is None:
        return False, "no reading for %s" % key
    if value < BAR[bar_key]:
        return False, ("%s did not come apart: %.5f, bar %.5f" % (what, value, BAR[bar_key]))
    return True, "%s came apart by %.5f" % (what, value)


def reading_bar(measures, key, bar_key):
    """One number must reach its bar."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures.get(key)
    if value is None:
        return False, "the reading %s was never produced" % key
    if value < BAR[bar_key]:
        return False, "%s = %.5f, bar %.5f" % (key, value, BAR[bar_key])
    return True, "%s = %.5f" % (key, value)


def contrast_bar(measures, key, bar_key, what):
    """One situation must come out above another by at least the bar."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures.get("gain", {}).get(key)
    if value is None:
        return False, "no reading for %s" % key
    if value < BAR[bar_key]:
        return False, ("%s did not come apart: %.5f, bar %.5f"
                       % (what, value, BAR[bar_key]))
    return True, "%s came apart by %.5f" % (what, value)


def ratio_bar(measures, key, bar_key, what):
    """One situation must be at least a given multiple of another."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures.get(key)
    if value is None:
        return False, "no reading for %s" % key
    if value < BAR[bar_key]:
        return False, "%s = %.4f times, bar %.4f" % (what, value, BAR[bar_key])
    return True, "%s = %.4f times" % (what, value)


def both_sides_bar(measures, left_key, right_key, bar_key, what):
    """Two opposite situations must each move the reading the way they should."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    left, right = measures.get(left_key), measures.get(right_key)
    if left is None or right is None:
        return False, "both sides were not read"
    if min(left, right) < BAR[bar_key]:
        return False, ("%s only moved %.5f and %.5f, bar %.5f"
                       % (what, left, right, BAR[bar_key]))
    return True, "%s moved %.5f and %.5f" % (what, left, right)
# ---------------------------------------------------------------------------
# looking
# ---------------------------------------------------------------------------
def _eye_brain(body, ctx):
    return brain_for(ctx, body, eyes=True)


def _clean_body():
    return clean_body()


def _gaze_now(brain):
    command = brain.eye_command()
    return float(.5 * (command[0] + command[2]))


def _bearing(body, target, base):
    offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
    rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
    return float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0]))


def _gaze_error(body, brain, eyes, target, base, observation):
    return _gaze_now(brain) - _bearing(body, target, base)


def _eye_step(body, brain, eyes, observation, environment):
    activation = brain.step(observation, environment=environment,
                            eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    return body.step(np.asarray(body.home_angles), duration=DT, activation=activation)


def gaze_measure(ctx, sweep, seconds, distance, scenery):
    body = _clean_body() if not scenery else Go2Body(model_path=ctx["model_path"])
    if not scenery:
        for name in SCENERY:
            geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    brain, eyes = _eye_brain(body, ctx)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    observation = body.observe()
    environment = blank_environment()
    errors = []
    try:
        for step in range(int(round(seconds / DT))):
            lateral = sweep - 2. * sweep * step / float(max(1, int(round(seconds / DT)) - 1))
            body.model.geom_pos[target] = [.30 + distance, lateral, .32]
            mujoco.mj_forward(body.model, body.data)
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
            rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
            bearing = np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])
            command = brain.eye_command()
            if step > 20:
                errors.append(.5 * (command[0] + command[2]) - bearing)
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
    finally:
        eyes.close()
    if not errors:
        return dict(status="error", error="no samples", mean_abs_error_rad=None)
    return dict(status="ok", error=None, mean_abs_error_rad=float(np.abs(np.array(errors)).mean()),
                samples=len(errors))


def gaze_bar(measures, key):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["mean_abs_error_rad"] > BAR[key]:
        return False, ("looked %.3f rad off the ball (bar %.3f)"
                       % (measures["mean_abs_error_rad"], BAR[key]))
    return True, "looked %.3f rad off the ball" % measures["mean_abs_error_rad"]


def sweep_pair_measure(ctx):
    """The same ball sweeps right and then sweeps left; both must be followed."""
    rightward = gaze_measure(ctx, sweep=.30, seconds=2.5, distance=.90, scenery=False)
    leftward = gaze_measure(ctx, sweep=-.30, seconds=2.5, distance=.90, scenery=False)
    status = "ok" if rightward["status"] == "ok" and leftward["status"] == "ok" else "error"
    return dict(status=status, error=None if status == "ok" else "a sweep failed",
                rightward=rightward, leftward=leftward,
                worst_error_rad=max(rightward["mean_abs_error_rad"] or 9.,
                                    leftward["mean_abs_error_rad"] or 9.))


def sweep_pair_bar(measures):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["worst_error_rad"] > BAR["gaze_error_rad"]:
        return False, ("one of the two sweeps was not followed (worst %.3f rad, bar %.3f)"
                       % (measures["worst_error_rad"], BAR["gaze_error_rad"]))
    return True, "followed the ball both ways (worst %.3f rad)" % measures["worst_error_rad"]


def sudden_measure(ctx, settle_seconds, hold_seconds, lateral, drift=.30):
    """A ball is not there, and then it is, and it moves.  Where do the eyes go?

    It moves because this animal's eyes are driven by what changes in the
    picture - a thing that merely exists, without ever changing, does not turn
    them (the still-ball scene in tools/measure_eye_gaze_routes.py reads 0.011
    rad, and that is the ball sitting at the rest angle, not the eyes finding
    it).  So the question here is whether a thing appearing off to one side,
    already moving, gets looked at.
    """
    body = _clean_body()
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    brain, eyes = _eye_brain(body, ctx)
    observation = body.observe()
    environment = blank_environment()
    errors, before, after, bearing = [], [], [], []
    total = int(round((settle_seconds + hold_seconds) / DT))
    tail = max(1, int(round(1. / DT)))
    try:
        for step in range(total):
            present = step * DT >= settle_seconds
            if present:
                share = (step * DT - settle_seconds) / max(hold_seconds, DT)
                place = [.30 + .90, lateral + drift * share, .32]
            else:
                place = [60., 60., -8.]
            body.model.geom_pos[target] = place
            mujoco.mj_forward(body.model, body.data)
            error = _gaze_error(body, brain, eyes, target, base, observation)
            if present:
                errors.append(error)
                after.append(_gaze_now(brain))
                bearing.append(_bearing(body, target, base))
            elif step * DT > settle_seconds - .3:
                before.append(_gaze_now(brain))
            observation = _eye_step(body, brain, eyes, observation, environment)
    finally:
        eyes.close()
    if not errors:
        return dict(status="error", error="no samples")
    shift = float(np.mean(after[-tail:]) - np.mean(before)) if before else None
    return dict(status="ok", error=None,
                error_after_appearing_rad=float(np.abs(np.array(errors[-tail:])).mean()),
                gaze_shift_rad=float(shift),
                bearing_after_rad=float(np.mean(bearing[-tail:])),
                samples=len(errors))


def sudden_bar(measures, key):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    shift, bearing = measures["gaze_shift_rad"], measures["bearing_after_rad"]
    if np.sign(shift) != np.sign(bearing) or abs(shift) < BAR[key]:
        return False, ("the eyes did not swing towards the thing that appeared "
                       "(%.4f rad, it was at %+.3f)" % (shift, bearing))
    return True, ("the eyes swung %.4f rad towards the thing that appeared" % shift)


def parked_measure(ctx, lateral=.55, distance=.90, seconds=3.):
    """A ball parked off to one side the whole time: do the eyes stay put?

    Nothing appears and nothing moves, so any eye movement here is drift.  The
    swing is the spread of the eye angle over the last two seconds.
    """
    body = _clean_body()
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    brain, eyes = _eye_brain(body, ctx)
    observation = body.observe()
    environment = blank_environment()
    history = []
    tail = max(1, int(round(2. / DT)))
    try:
        for step in range(int(round(seconds / DT))):
            body.model.geom_pos[target] = [.30 + distance, lateral, .32]
            mujoco.mj_forward(body.model, body.data)
            if step * DT >= 1.:
                history.append(_gaze_now(brain))
            observation = _eye_step(body, brain, eyes, observation, environment)
    finally:
        eyes.close()
    if not history:
        return dict(status="error", error="no samples")
    swing = np.asarray(history[-tail:], dtype=float)
    return dict(status="ok", error=None, lateral=lateral,
                gaze_swing_rad=float(swing.max() - swing.min()),
                gaze_end_rad=float(swing[-1]), bearing_rad=_bearing(body, target,
                                                                    mujoco.mj_name2id(
                                                                        body.model,
                                                                        mujoco.mjtObj.mjOBJ_BODY,
                                                                        "base")))


def still_bar(measures):
    """A thing that never moves must not be chased by the eyes."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["gaze_swing_rad"] > BAR["gaze_still_rad"]:
        return False, ("the eyes wandered %.4f rad while it sat still (bar %.4f)"
                       % (measures["gaze_swing_rad"], BAR["gaze_still_rad"]))
    return True, ("the eyes held still: %.4f rad of wander at a parked ball"
                  % measures["gaze_swing_rad"])


def approach_measure(ctx, near, far, seconds):
    """One ball comes in from far to near; read the brain's own distance bank."""
    body = _clean_body()
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    brain, eyes = _eye_brain(body, ctx)
    observation = body.observe()
    environment = blank_environment()
    steps = int(round(seconds / DT))
    window = max(1, int(round(.5 / DT)))
    banks = []
    try:
        for step in range(steps):
            range_m = far + (near - far) * step / float(steps - 1)
            body.model.geom_pos[target] = [range_m, 0., .32]
            mujoco.mj_forward(body.model, body.data)
            banks.append(float(np.sum(brain.diagnostics()["eye_distance"])))
            observation = _eye_step(body, brain, eyes, observation, environment)
    finally:
        eyes.close()
    banks = np.asarray(banks)
    return dict(status="ok", error=None, near=near, far=far,
                bank_far=float(banks[:window].mean()),
                bank_near=float(banks[-window:].mean()))


def approach_bar(measures):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["bank_far"] > BAR["bank_far_max"]:
        return False, ("the distance cells were already full at %.1f m (%.2f)"
                       % (measures["far"], measures["bank_far"]))
    if measures["bank_near"] < BAR["bank_near_min"]:
        return False, ("the ball came to %.2f m and the distance cells only reached %.2f"
                       % (measures["near"], measures["bank_near"]))
    return True, ("the distance cells go from %.2f at %.1f m to %.2f at %.2f m"
                  % (measures["bank_far"], measures["far"], measures["bank_near"],
                     measures["near"]))


def _stripes():
    """A picture with edges in it.  A flat wash of light is not enough: this
    retina answers to change, so a uniform bright frame and a black frame read
    the same, and only a patterned one wakes it."""
    image = np.zeros((2, 36, 48, 3), dtype=np.uint8)
    for column in range(48):
        if (column // 4) % 2 == 0:
            image[:, :, column, :] = 220
    return image


def pattern_measure(ctx, seconds=.35, tail=.35):
    """A striped picture against a black one, both handed straight to the eyes.

    The window is the first third of a second, for a reason worth writing down:
    this retina layer answers a new picture for about a quarter of a second and
    then goes quiet, even though the light cells underneath stay lit (0.43 the
    whole time).  A reading taken at the end of a longer run is exactly zero.
    """
    return contrast_read(ctx,
                         {"picture": _stripes(), "seconds": seconds, "tail": tail,
                          "clean": True},
                         {"picture": np.zeros((2, 36, 48, 3), dtype=np.uint8),
                          "seconds": seconds, "tail": tail, "clean": True}, ("retina",))


def convergence_measure(ctx, near=.25, far=2.0, seconds=2.):
    """One ball parked near and parked far; the inward cells must wake up near."""
    cells = ("eye_convergence_in", "eye_convergence_out")
    close = cell_read(ctx, cells=cells, seconds=seconds, clean=True, look=True,
                      eyes_move=True, gaze=(near, 0., .32), gaze_size=.10, tail=seconds)
    distant = cell_read(ctx, cells=cells, seconds=seconds, clean=True, look=True,
                        eyes_move=True, gaze=(far, 0., .32), gaze_size=.10, tail=seconds)
    return dict(status="ok", error=None,
                near=close["eye_convergence_in"] - close["eye_convergence_out"],
                far=distant["eye_convergence_in"] - distant["eye_convergence_out"],
                gain=(close["eye_convergence_in"] - close["eye_convergence_out"])
                     - (distant["eye_convergence_in"] - distant["eye_convergence_out"]))


# ---------------------------------------------------------------------------
# listening
# ---------------------------------------------------------------------------
def gaze_static_measure(ctx, lateral=.45, distance=1.00, seconds=2.):
    """A ball parked off to one side; where do the eyes come to rest?

    Each side gets its own brain, because the two readings must not be two
    halves of one settling.  The ball is inside the eyes' field only while
    ``lateral`` is under about half a radian of bearing.
    """
    target = None
    readings = {}
    for side, label in ((1., "left"), (-1., "right")):
        body = clean_body(ctx["model_path"])
        target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
        body.model.geom_size[target] = [.10] * 3
        body.model.geom_pos[target] = [distance, side * lateral, .32]
        mujoco.mj_forward(body.model, body.data)
        brain, eyes = brain_for(ctx, body, eyes=True)
        observation = body.observe()
        environment = blank_environment()
        history = []
        try:
            for index in range(int(round(seconds / DT))):
                _, activation = brain.step(observation, environment=environment,
                                           eye_pixels=eyes.observe_raw(), dt=DT, learn=False)
                body.command_eyes(brain.eye_command())
                observation = body.step(np.asarray(body.home_angles), duration=DT,
                                        activation=activation)
                if index * DT > seconds / 2.:
                    command = brain.eye_command()
                    history.append(float(.5 * (command[0] + command[2])))
        finally:
            eyes.close()
        readings[label] = float(np.mean(history))
    return dict(status="ok", error=None, left=readings["left"], right=readings["right"],
                difference=readings["left"] - readings["right"])


def both_ways_bar(measures, bar_key, what):
    """The eyes have to end up on the side the thing is on, both times."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    worst = min(measures["left"], -measures["right"])
    if worst < BAR[bar_key]:
        return False, ("%s did not get the eyes both ways (%.4f and %.4f rad, bar %.4f)"
                       % (what, measures["left"], measures["right"], BAR[bar_key]))
    return True, "%s: the eyes went %.4f and %.4f rad" % (what, measures["left"],
                                                          measures["right"])


def sound_measure(ctx, x, y, seconds):
    body = _clean_body()
    source = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "sound_low")
    body.model.geom_pos[source] = [x, y, .30]
    mujoco.mj_forward(body.model, body.data)
    brain, eyes = _eye_brain(body, ctx)
    ears = BinauralSenses(body, window_samples=160)
    observation = body.observe()
    environment = blank_environment()
    try:
        for _ in range(int(round(seconds / DT))):
            _, activation = brain.step(observation, environment=environment,
                                       ear_waveform=ears.observe(), dt=DT, learn=False)
            body.command_eyes(brain.eye_command())
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
    finally:
        eyes.close()
    command = brain.eye_command()
    orienting = brain.network.activity[brain.groups["orienting"]]
    return dict(status="ok", error=None, source=[x, y],
                eye_yaw=float(.5 * (command[0] + command[2])),
                orienting_left_minus_right=float(orienting[0] - orienting[1]))


def sound_bar(measures):
    if measures.get("status") != "ok" or "right" not in measures or "left" not in measures:
        return False, "run failed: %s" % measures.get("error")
    right, left = measures["right"], measures["left"]
    if right["status"] != "ok" or left["status"] != "ok":
        return False, "run failed"
    # A positive eye yaw is a turn to the left (the gaze error task shows the
    # command tracks the ball's bearing, and a bearing to the left is positive).
    # So a sound on the right must pull the yaw below the same sound on the left
    # by at least the bar: turning towards the sound, not merely turning.
    difference = right["eye_yaw"] - left["eye_yaw"]
    if difference > -BAR["sound_yaw_rad"]:
        return False, ("a right-hand sound did not pull the eyes right of where a "
                       "left-hand sound did (%.4f rad)" % difference)
    return True, ("right-hand sound pulled the eyes %.4f rad towards it" % difference)


def sound_pair_measure(ctx):
    right = sound_measure(ctx, 1.10, -.60, 5.)
    left = sound_measure(ctx, 1.10, .60, 5.)
    return dict(status="ok", error=None, right=right, left=left)


def ear_contrast_measure(ctx, first, second, cells, seconds=3.):
    """Two sound scenes; each is (x, y) or (x, y, geom)."""
    return contrast_read(ctx,
                         {"ear": first, "seconds": seconds, "clean": True},
                         {"ear": second, "seconds": seconds, "clean": True}, cells)
# ---------------------------------------------------------------------------
# touching, standing, feeling the ground
# ---------------------------------------------------------------------------
def _foot_channel(ctx, channel, foot, cells, seconds, value, autonomy, tilt, clean=True,
                  tail=None):
    return cell_read(ctx, cells=cells, seconds=seconds, clean=clean,
                     tail=seconds if tail is None else tail,
                     autonomy=autonomy, tilt=tilt,
                     environment=one_foot(channel, foot, value))


def one_foot_measure(ctx, channel, foot, cells, seconds=2., value=.8, autonomy=True,
                     tilt=None, clean=True, tail=None):
    """The same channel on one leg, and then on the other; per-cell readings."""
    name = cells[0]
    first = _foot_channel(ctx, channel, foot, cells, seconds, value, autonomy, tilt, clean, tail)
    second = _foot_channel(ctx, channel, 1 - foot, cells, seconds, value, autonomy, tilt, clean, tail)
    return dict(status="ok", error=None, channel=channel,
                on_foot=float(first["parts"][name][foot]),
                other_foot=float(first["parts"][name][1 - foot]),
                gain=float(first["parts"][name][foot] - first["parts"][name][1 - foot]),
                swapped_gain=float(second["parts"][name][1 - foot]
                                   - second["parts"][name][foot]))


def tilt_measure(ctx, roll, cells, seconds=2., level=None):
    """A body rolled to one side against the same body level."""
    tipped = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tilt=(roll, 0.),
                       tail=seconds)
    flat = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds)
    up = np.asarray(tipped["parts"][cells[0]], dtype=float)
    down = np.asarray(flat["parts"][cells[0]], dtype=float)
    return dict(status="ok", error=None, roll=roll,
                tipped=up.tolist(), level=down.tolist(),
                gain_mean=float(np.mean(up) - np.mean(down)),
                gain_max=float(np.max(up - down)))


def stand_still_measure(ctx, seconds, injected=None, gaze=None, props=None,
                        scenario="rest"):
    """Nothing asks it to move: does it stay standing and stay put?"""
    return walk_measure(ctx, seconds, scenario, "origin", injected=injected, props=props)


def body_and_foot_measure(ctx, seconds=2.):
    """A touch on the body against a catch on one foot: two different lines."""
    cells = ("brake", "retreat", "withdrawal", "clearance")
    body = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                     environment=one_foot("body_touch", 0, .8))
    foot = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                     environment=one_foot("foot_obstacle", 0, .8))
    return dict(status="ok", error=None, body=body, foot=foot,
                body_retreat=body["retreat"] - foot["retreat"],
                foot_withdrawal=foot["withdrawal"] - body["withdrawal"])


def pushing_bar(measures):
    """Being shoved has to feel different from walking."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["gain"] < BAR["moving_is_not_being_pushed"]:
        return False, ("a shove barely read as different from its own walking "
                       "(%.4f, bar %.4f)" % (measures["gain"], BAR["moving_is_not_being_pushed"]))
    return True, ("a shove read %.4f above its own walking" % measures["gain"])


# ---------------------------------------------------------------------------
# learning, memory
# ---------------------------------------------------------------------------
def taught_reflex_measure(ctx, lesson_seconds, pause_seconds=0., teach=True, hand=True):
    from tools import nursery
    run = nursery.Run(ctx["seed"], hand_teaches=teach, gated=True, routes=True,
                      parameters=ctx["parameters"])
    try:
        before = run.phase(4., tone=True, walk=nursery.WALK, learn=False)
        lesson = run.phase(lesson_seconds, tone=True, walk=nursery.WALK, hand=hand, learn=True)
        if pause_seconds > 0.:
            run.phase(pause_seconds, tone=False, walk=nursery.WALK, learn=False)
        tone = run.phase(4., tone=True, walk=nursery.WALK, learn=False)
        silence = run.phase(4., tone=False, walk=nursery.WALK, learn=False)
    finally:
        run.close()
    return dict(status="ok", error=None, teach=teach, hand=hand,
                retreat_before=before["retreat"], retreat_on_tone=tone["retreat"],
                retreat_in_silence=silence["retreat"], retreat_during_lesson=lesson["retreat"],
                gain_on_tone=tone["retreat"] - silence["retreat"],
                gain_before=before["retreat"] - silence["retreat"])


def taught_reflex_bar(measures, bar_key="taught_reflex_gain"):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["gain_on_tone"] < BAR[bar_key]:
        return False, ("nothing was learned: tone retreat %.3f, silence %.3f"
                       % (measures["retreat_on_tone"], measures["retreat_in_silence"]))
    return True, ("tone alone now drives retreat %.3f over silence"
                  % measures["gain_on_tone"])


def taught_reflex_control_bar(measures):
    """Nobody taught it, so the tone must not have acquired a meaning."""
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["gain_on_tone"] >= BAR["taught_reflex_control"]:
        return False, ("the tone picked up a meaning with no teacher (%.3f, bar %.3f)"
                       % (measures["gain_on_tone"], BAR["taught_reflex_control"]))
    return True, ("the tone still means nothing without a teacher (%.3f)"
                  % measures["gain_on_tone"])


def _pattern_images(shape, channel=2):
    pattern = np.zeros(shape, dtype=np.uint8)
    height, width = shape[1], shape[2]
    rows = slice(int(.22 * height), int(.42 * height))
    columns = slice(int(.31 * width), int(.48 * width))
    pattern[:, rows, columns, channel] = 255
    return pattern, np.zeros_like(pattern)


def _memory_phase(brain, observation, image, touch, seconds, learn):
    environment = blank_environment()
    environment["body_touch"][0] = touch
    tail = max(1, int(round(.5 / DT)))
    centre, memory = [], []
    for _ in range(int(round(seconds / DT))):
        brain.step(observation, environment=environment, eye_pixels=image,
                   dt=DT, learn=learn, autonomy=False)
        rates = brain.network.activity
        centre.append(float(rates[brain.groups["aversive"]][1]))
        memory.append(float(rates[brain.groups["retinal_memory"]].max()))
    centre, memory = np.asarray(centre), np.asarray(memory)
    return dict(aversive_centre=float(centre[-tail:].mean()),
                memory_peak=float(memory[-tail:].mean()))


def visual_memory_measure(ctx, rounds, gap=0, channel=2):
    body = Go2Body(model_path=ctx["model_path"])
    brain = brain_for(ctx, body)
    observation = body.observe()
    blue, black = _pattern_images(brain.eye_shape, channel)
    baseline = _memory_phase(brain, observation, blue, 0., 2., False)
    for _ in range(rounds):
        _memory_phase(brain, observation, blue, .8, 2., True)
        _memory_phase(brain, observation, black, .0, 2., True)
    for _ in range(gap):
        _memory_phase(brain, observation, black, .0, 2., False)
    probe = _memory_phase(brain, observation, blue, 0., 2., False)
    return dict(status="ok", error=None, rounds=rounds, gap=gap, channel=channel,
                aversive_baseline=baseline["aversive_centre"],
                probe_aversive=probe["aversive_centre"],
                aversive_gain=probe["aversive_centre"] - baseline["aversive_centre"],
                memory_peak=probe["memory_peak"])


def visual_memory_bar(measures, bar_key="visual_memory_gain", above=True):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    value, bar = measures["aversive_gain"], BAR[bar_key]
    if above and value < bar:
        return False, ("the pattern still means nothing (%.4f over silence, bar %.4f)"
                       % (value, bar))
    if not above and value >= bar:
        return False, ("the pattern picked up a meaning with no pairing (%.4f, bar %.4f)"
                       % (value, bar))
    return True, "the pattern drives the aversive cells %.4f over silence" % value


# ---------------------------------------------------------------------------
# the body as its own thing
# ---------------------------------------------------------------------------
def own_gait_measure(ctx, seconds=6.):
    """Walking swings the body too; a shove must still read as something else."""
    walk = walk_measure(ctx, seconds, "autonomous", "origin")
    pushed = righting_measure(ctx, "impact", 7., True)
    if walk["status"] != "ok" or pushed["status"] != "ok":
        return dict(status="error", error=walk["error"] or pushed["error"])
    walking = walk["peak"].get("protective_tilt", 0.)
    shoved = pushed["peak"].get("protective_tilt", 0.)
    return dict(status="ok", error=None, walking=walking, pushed=shoved,
                gain=shoved - walking)


def eye_feel_measure(ctx, angle=.5, seconds=2.):
    """The eye muscles are commanded off-centre: is that felt as its own state?"""
    body = clean_body(ctx["model_path"])
    brain, eyes = brain_for(ctx, body, eyes=True)
    observation = body.observe()
    environment = blank_environment()
    straight, turned = [], []
    try:
        for index in range(int(round(seconds / DT))):
            turned_now = index * DT > seconds / 2.
            body.command_eyes([angle, 0., angle, 0.] if turned_now
                              else [0., 0., 0., 0.])
            brain.step(observation, environment=environment, dt=DT, learn=False,
                       autonomy=False)
            observation = body.step(np.asarray(body.home_angles), duration=DT)
            rates = brain.network.activity
            (turned if turned_now else straight).append(
                float(np.mean(rates[brain.groups["eye_proprioception"]])))
    finally:
        eyes.close()
    return dict(status="ok", error=None, straight=float(np.mean(straight)),
                turned=float(np.mean(turned)),
                gain=float(np.mean(turned) - np.mean(straight)))
# ---------------------------------------------------------------------------
# the rest of the reading machinery
# ---------------------------------------------------------------------------
def slip_bar(measures):
    walked, why = walk_bar(measures, "walk_slip_m")
    if not walked:
        return False, why
    worst = 0.
    for foot in measures["slip"]:
        if foot["mean_mps"] is not None:
            worst = max(worst, foot["mean_mps"])
    if worst > BAR["slip_mps"]:
        return False, "a planted foot slid at %.3f m/s (bar %.3f)" % (worst, BAR["slip_mps"])
    return True, "worst sliding foot %.3f m/s" % worst


def leaning_bar(measures):
    """Rolled to one side: at least one of the four direction cells must wake."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["gain_max"] < BAR["lean_gain"]:
        return False, ("no direction cell woke up when the body rolled (%.4f, bar %.4f)"
                       % (measures["gain_max"], BAR["lean_gain"]))
    return True, ("the direction cells moved %.4f when the body rolled" % measures["gain_max"])


def tipped_bar(measures):
    return reading_bar(measures, "gain_mean", "tilt_protective_gain")


def pitch_measure(ctx, seconds=3.):
    """The high tone and the low tone through the same cochlea, band by band."""
    cells = ("cochlea",)
    high = cell_read(ctx, cells=cells, seconds=seconds, clean=True, ear=(1.10, 0., "sound_high"))
    low = cell_read(ctx, cells=cells, seconds=seconds, clean=True, ear=(1.10, 0., "sound_low"))
    top, bottom = np.array([2, 5]), np.array([0, 3])
    a = np.asarray(high["parts"]["cochlea"], dtype=float)
    b = np.asarray(low["parts"]["cochlea"], dtype=float)
    return dict(status="ok", error=None,
                high=a.tolist(), low=b.tolist(),
                high_top=float(a[top].mean()), low_top=float(b[top].mean()),
                high_bottom=float(a[bottom].mean()), low_bottom=float(b[bottom].mean()),
                gain=float((a[top].mean() - a[bottom].mean())
                           - (b[top].mean() - b[bottom].mean())))


def habituation_measure(ctx, seconds=4., ear_source=(1.10, 0., "sound_low")):
    """One bang held on: the startle cell should tire of it."""
    body = clean_body(ctx["model_path"])
    if ear_source is not None:
        _source(body, ear_source[2] if len(ear_source) > 2 else "sound_low",
                (ear_source[0], ear_source[1], .30))
    brain = brain_for(ctx, body)
    ears = BinauralSenses(body, window_samples=160)
    observation = body.observe()
    environment = blank_environment()
    history, ear = [], []
    for _ in range(int(round(seconds / DT))):
        brain.step(observation, environment=environment, dt=DT, learn=False,
                   autonomy=False, startle=1., ear_waveform=ears.observe())
        history.append(float(brain.network.activity[brain.groups["startle"][0]]))
        ear.append(float(np.mean(brain.network.activity[brain.groups["cochlea"]])))
    window = max(1, int(round(.2 / DT)))
    first, last = float(np.mean(history[:window])), float(np.mean(history[-window:]))
    return dict(status="ok", error=None, first=first, last=last, fall=first - last,
                ear_first=float(np.mean(ear[:window])),
                ear_last=float(np.mean(ear[-window:])))


def hearing_bar(measures):
    """Tiring of a bang must not be deafness: the ear keeps firing."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["fall"] < BAR["startle_habituation"]:
        return False, ("the startle cell never tired, so there is nothing to compare "
                       "(%.4f)" % measures["fall"])
    if measures["ear_last"] < BAR["hears_while_tiring"]:
        return False, ("the ear went quiet too, so this is deafness not habit "
                       "(%.4f, bar %.4f)" % (measures["ear_last"], BAR["hears_while_tiring"]))
    return True, ("the startle cell fell %.4f while the ear stayed at %.4f"
                  % (measures["fall"], measures["ear_last"]))


def habituation_bar(measures):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["fall"] < BAR["startle_habituation"]:
        return False, ("the startle cell did not tire of the bang (%.4f, bar %.4f)"
                       % (measures["fall"], BAR["startle_habituation"]))
    return True, ("the startle cell fell %.4f while the bang kept on" % measures["fall"])



SILENT = {"sound_low": (60., 60., -8.), "sound_high": (60., 60., -8.)}


def sound_beside(side, distance=1.40):
    """Props for a walk with the low emitter level with one shoulder."""
    return dict(SILENT, sound_low=(.20, -distance if side == "right" else distance, .30))


def turn_pair_measure(ctx, seconds, scenario):
    """The same walk twice, once with the sound on each side."""
    right = walk_measure(ctx, seconds, scenario, "origin", props=sound_beside("right"))
    left = walk_measure(ctx, seconds, scenario, "origin", props=sound_beside("left"))
    if right["status"] != "ok" or left["status"] != "ok":
        return dict(status="error", error=right["error"] or left["error"])
    return dict(status="ok", error=None, right=right, left=left,
                right_yaw=right["yaw_rad"], left_yaw=left["yaw_rad"],
                difference=right["yaw_rad"] - left["yaw_rad"])


def turn_pair_bar(measures, bar_key, what):
    """A positive yaw is a turn to the left, so a sound on the right has to leave
    the heading further right than the same sound on the left by the bar: turning
    towards the sound, and not merely turning."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures["difference"]
    if value > -BAR[bar_key]:
        return False, ("%s did not turn it (%.3f rad apart, bar %.3f)"
                       % (what, value, BAR[bar_key]))
    return True, "%s: the two sides came apart by %.3f rad" % (what, value)


def at_least_bar(measures, key, bar_key, what, walk="walk_own_m"):
    passed, why = walk_bar(measures, walk)
    if not passed:
        return False, why
    value = float(measures[key])
    if value < BAR[bar_key]:
        return False, "%s = %.4f, bar %.4f" % (what, value, BAR[bar_key])
    return True, "%s = %.4f" % (what, value)


def at_most_bar(measures, key, bar_key, what, walk="walk_own_m"):
    passed, why = walk_bar(measures, walk)
    if not passed:
        return False, why
    value = abs(float(measures[key]))
    if value > BAR[bar_key]:
        return False, "%s = %.4f, bar %.4f" % (what, value, BAR[bar_key])
    return True, "%s = %.4f" % (what, value)


def pace_bar(measures, what):
    passed, why = walk_bar(measures, "walk_own_m")
    if not passed:
        return False, why
    if measures["mean_speed_mps"] < BAR["walk_speed_mps"]:
        return False, ("%s averaged %.3f m/s, bar %.3f"
                       % (what, measures["mean_speed_mps"], BAR["walk_speed_mps"]))
    return True, ("%s averaged %.3f m/s and covered %.3f m"
                  % (what, measures["mean_speed_mps"], measures["displacement_m"]))


def idle_pace_bar(measures, what):
    """Turning is a state, not a fault, so the bar is not about turning.

    The room is only so big, so an animal that walks will also curve.  The old
    bar here - heading travelled through <= .60 rad - sat below the whole
    population: every one of the 121 mutants turned between .77 and 2.15 rad,
    and every one of them got at least 0.32 m from where it started.  It was
    asking "does it walk", not "does it spin".  What is a fault is spending the
    motion on pivoting: turning a lot and getting nowhere.  That is the bar
    now.  How far it turned is still reported either way.
    """
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < .9:
        return False, "lost its footing (min up_z %.3f)" % measures["min_up_z"]
    if (measures["yaw_path_rad"] > BAR["idle_spin_rad"]
            and measures["displacement_m"] < BAR["idle_spin_progress_m"]):
        return False, ("%s turned through %.3f rad but got only %.3f m from where it "
                       "started: that is pivoting on the spot"
                       % (what, measures["yaw_path_rad"], measures["displacement_m"]))
    return True, ("%s turned through %.3f rad while getting %.3f m along"
                  % (what, measures["yaw_path_rad"], measures["displacement_m"]))


def down_bar(measures, distance_key, what):
    """It has to walk, and it has to get past the edge of what it stood on."""
    passed, why = walk_bar(measures, distance_key)
    if not passed:
        return False, why
    if not measures["crossed_marker"]:
        return False, ("walked %.3f m but never came off %s (reached %.3f m)"
                       % (measures["displacement_m"], what, measures["progress_m"]))
    return True, ("came off %s: walked %.3f m, %.3f m past the edge"
                  % (what, measures["displacement_m"], measures["progress_m"]))


def around_bar(measures):
    passed, why = walk_bar(measures, "around_block_m")
    if not passed:
        return False, why
    if not measures["crossed_marker"]:
        return False, ("walked %.3f m but never got past the block (reached %.3f m)"
                       % (measures["displacement_m"], measures["progress_m"]))
    if abs(measures["lateral_m"]) < BAR["around_block_lateral_m"]:
        return False, ("got level with it without ever stepping aside (sideways %.3f m, bar %.3f)"
                       % (abs(measures["lateral_m"]), BAR["around_block_lateral_m"]))
    return True, ("went round it: %.3f m past and %.3f m to the side"
                  % (measures["progress_m"], abs(measures["lateral_m"])))


def held_back_bar(measures, bar_key, what):
    """The same walk twice: once free, once with something holding it back."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    free, held = measures["free"], measures["held"]
    if free["status"] != "ok" or held["status"] != "ok":
        return False, "run failed: %s" % (free["error"] or held["error"])
    if not free["behavior_pass"] or not held["behavior_pass"]:
        return False, ("it fell over in one of the two runs (free min up_z %.3f, "
                       "held min up_z %.3f)" % (free["min_up_z"], held["min_up_z"]))
    if free["displacement_m"] < BAR["walk_own_m"]:
        return False, ("it did not walk even with nothing in the way (%.3f m)"
                       % free["displacement_m"])
    limit = BAR[bar_key] * free["displacement_m"]
    if held["displacement_m"] > limit:
        return False, ("%s barely slowed it: %.3f m against %.3f m free (limit %.3f)"
                       % (what, held["displacement_m"], free["displacement_m"], limit))
    return True, ("%s cut the walk from %.3f m to %.3f m"
                  % (what, free["displacement_m"], held["displacement_m"]))


def held_back_measure(ctx, seconds, injected=None, startle=None):
    free = walk_measure(ctx, seconds, "autonomous", "origin", props=SILENT)
    held = walk_measure(ctx, seconds, "autonomous", "origin", props=SILENT,
                        injected=injected, startle=startle)
    return dict(status="ok", error=None, free=free, held=held)


def stand_turn_measure(ctx, seconds):
    """A sound on one side of an animal nobody asked to walk."""
    right = stand_still_measure(ctx, seconds, props=sound_beside("right"))
    left = stand_still_measure(ctx, seconds, props=sound_beside("left"))
    if right["status"] != "ok" or left["status"] != "ok":
        return dict(status="error", error=right["error"] or left["error"])
    return dict(status="ok", error=None, right=right, left=left,
                difference=right["yaw_rad"] - left["yaw_rad"])


def stand_turn_bar(measures, what):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value = measures["difference"]
    if value > -BAR["turn_idle_rad"]:
        return False, ("%s did not turn it (%.3f rad apart, bar %.3f)"
                       % (what, value, BAR["turn_idle_rad"]))
    return True, "the two sides came apart by %.3f rad" % value


def side_touch_measure(ctx, sectors, cells, index_of, value=.8, seconds=2.):
    """One body sector touched, then its opposite, read off two named cells.

    ``sectors`` is the pair of body sectors (front/left/back/right) and
    ``index_of`` is the cell in each named group that answers each of them -
    the sector number and the cell number are not the same thing.
    """
    name = cells[0]
    first, second = sectors
    a = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                  environment=one_foot("body_touch", first, value))
    b = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                  environment=one_foot("body_touch", second, value))
    first_cell, second_cell = index_of
    return dict(status="ok", error=None,
                left_touch_gain=float(a["parts"][name][first_cell]
                                      - a["parts"][name][second_cell]),
                right_touch_gain=float(b["parts"][name][second_cell]
                                       - b["parts"][name][first_cell]))


def faint_sound_measure(ctx, seconds=2.):
    """Something a long way off, against a room with nothing making a sound."""
    return contrast_read(ctx,
                         {"ear": (3.20, 0., "sound_low"), "seconds": seconds,
                          "clean": True, "props": SILENT},
                         {"seconds": seconds, "clean": True, "props": SILENT},
                         ("cochlea",))


def sound_behind_measure(ctx, seconds=2.):
    """The same emitter behind the animal and in front of it."""
    return contrast_read(ctx,
                         {"ear": (-1.10, 0., "sound_low"), "seconds": seconds, "clean": True},
                         {"ear": (1.10, 0., "sound_low"), "seconds": seconds, "clean": True},
                         ("cochlea", "auditory_pinna", "auditory_spatial"))


def two_sounds_measure(ctx, seconds=2.):
    """Two emitters lit at once, against one of them alone."""
    together = {"sound_low": (1.10, .60, .30), "sound_high": (1.10, -.60, .30)}
    alone = dict(together, sound_high=(60., 60., -8.))
    return contrast_read(ctx,
                         {"ear": False, "props": together, "seconds": seconds, "clean": True},
                         {"ear": False, "props": alone, "seconds": seconds, "clean": True},
                         ("auditory_pinna", "auditory_spatial"))


def behind_bar(measures):
    return contrast_bar(measures, "auditory_pinna", "behind_gain", "后面那一声")

def touch_here_and_there(front, left):
    """A full environment with the body-touch channel set sector by sector."""
    environment = blank_environment()
    environment["body_touch"] = np.array([front, left, 0., 0.])
    return environment


def two_touch_measure(ctx, value=.8, seconds=2.):
    """A touch on the front alone, then on the front and the left together.

    Read off the four slow "the body is against something" cells.  Each one
    is far past saturation for any touch at all, so the reading is how many
    of the four are lit and not how hard any one of them fires.
    """
    cells = ("wall_contact",)
    front = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                      environment=touch_here_and_there(value, 0.))
    both = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds,
                     environment=touch_here_and_there(value, value))
    first = np.asarray(front["parts"]["wall_contact"], dtype=float)
    second = np.asarray(both["parts"]["wall_contact"], dtype=float)
    return dict(status="ok", error=None,
                front_lit=int(np.sum(first > .5)), both_lit=int(np.sum(second > .5)),
                front_parts=first.tolist(), both_parts=second.tolist(),
                gain=float(second.sum() - first.sum()))


def two_touch_bar(measures):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["front_lit"] != 1:
        return False, ("a touch on the front alone lit %d of the four body sectors"
                       % measures["front_lit"])
    if measures["both_lit"] < 2:
        return False, ("a touch on the front and the left together still lit only %d sector"
                       % measures["both_lit"])
    return True, "one touch lit 1 sector and two touches lit %d" % measures["both_lit"]

# ---------------------------------------------------------------------------
# the questions, in the order a keeper would ask them
# ---------------------------------------------------------------------------
task("stands_still", "站立", "没人叫它走的时候，它站得住、也不乱走吗", "screen", 6.,
     lambda ctx: stand_still_measure(ctx, 6.), quiet_bar,
     "出厂这只 6 秒会慢慢蹭出半米，所以门槛定在 0.70 米，只拦得住「走起来」"
     "和「倒下去」。它不说明它能站得纹丝不动。",
     requires=())

task("walk_flat", "走路", "光地板上走一段", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_flat_m"),
     "光地板比满场道具容易，通过它只说明没有道具时走得动。",
     requires=("stands_still",))

task("walk_furnished", "走路", "带着满场道具走一段，不会摔倒", "screen", 12.,
     lambda ctx: walk_measure(ctx, 12., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_furnished_m"),
     "只说明它在这一次抽到的场地里走了这么远，不说明它认路或认得任何东西。",
     requires=("walk_flat",))

task("walks_without_being_told", "走路", "没人给它任何指令，它自己会不会往前走", "full", 8.,
     lambda ctx: walk_measure(ctx, 8., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_own_m"),
     "没有外部指令可用，这一题只问它自己走不走得动。出厂动物不会用前进指令，"
     "这半米是它自己蹭出来的，所以它出生时就已经过了这一题。",
     requires=("stands_still",))

task("keeps_walking_without_being_told", "走路", "没人管它，它是一直走还是走两步就停", "full", 16.,
     lambda ctx: walk_measure(ctx, 16., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_own_far_m"),
     "走远了不等于走得稳：这一题只看距离，不看姿势好不好看。",
     requires=("walks_without_being_told",))

task("feet_do_not_slide", "走路", "走路时踩在地上的脚会不会打滑", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"), slip_bar,
     "数的是落地那几帧的滑移速度。它只在这一种地面和这一档步态下成立。",
     requires=("walk_flat",))

task("walk_low_step", "地形", "从低台阶前出发，能不能走过去", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "step"),
     lambda m: walk_bar(m, "walk_step_m"),
     "走的是同一道台阶，不是随便什么地形；它不说明它「看见」了台阶。"
     "它也没走到台阶跟前，所以这题测的其实是「朝台阶那边走」而不是「迈上去」。",
     requires=("walk_flat",))

task("walk_ramp", "地形", "从斜坡前出发，能不能爬上去", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "ramp"),
     lambda m: walk_bar(m, "walk_ramp_m"),
     "走的是同一道斜坡，不是随便什么坡度。",
     requires=("walk_flat",))

task("walk_narrow_passage", "地形", "从窄通道口出发，能不能穿过去", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "passage"),
     lambda m: walk_bar(m, "walk_passage_m"),
     "走的是同一段通道，不是随便什么宽度；撞墙再蹭过去也算过，只要没倒。",
     requires=("walk_flat",))

task("steps_over_the_curb", "地形", "路上横着一条矮坎，它会迈过去还是绊一下", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "curb"),
     lambda m: marker_bar(m, "walk_curb_m"),
     "这题要的是走到坎跟前并越过它，比上面那几道走路的题难一档；"
     "被绊一下再过去也算过，它没在量步态好不好。",
     requires=("walk_low_step",))

task("climbs_onto_the_platform", "地形", "前面一块矮台，它能上到台面上吗", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "platform"),
     lambda m: marker_bar(m, "walk_platform_m"),
     "台面很矮；它不说明它能爬真的台阶。它同样比走路那几道难一档。",
     requires=("walk_ramp",))

task("stops_at_the_wall", "地形", "一直往前走，前面有堵墙，它是停下来还是撞上去", "full", 12.,
     lambda ctx: walk_measure(ctx, 12., "autonomous", "wall"),
     lambda m: wall_bar(m, "walk_into_wall_m"),
     "读的是墙上的接触细胞有没有亮、以及它有没有停在墙前；它不说明它「知道」那是墙。",
     requires=("walk_flat",))

task("get_up_from_back", "自救", "仰面朝天时自己翻回四脚着地", "screen", 6.,
     lambda ctx: righting_measure(ctx, "on_back", 6., False), righting_bar,
     "一个姿势一次机会。翻身成功不等于在任何姿势下都能翻身。",
     requires=())

task("get_up_from_side", "自救", "侧躺时自己翻回四脚着地", "full", 6.,
     lambda ctx: righting_measure(ctx, "on_side", 6., False), righting_bar,
     "见 get_up_from_back 的说明：一个姿势一次机会。",
     requires=("get_up_from_back",))

task("nose_up_recover", "自救", "头朝上竖起来时能不能落回四脚着地", "full", 6.,
     lambda ctx: righting_measure(ctx, "nose_up", 6., False), righting_bar,
     "见 get_up_from_back 的说明：一个姿势一次机会。",
     requires=("get_up_from_back",))

task("nose_down_recover", "自救", "头朝下栽着时能不能翻回四脚着地", "full", 6.,
     lambda ctx: righting_measure(ctx, "nose_down", 6., False), righting_bar,
     "见 get_up_from_back 的说明：一个姿势一次机会。",
     requires=("nose_up_recover",))

task("stay_up_when_pushed", "自救", "站稳后被推一把，能不能站回来", "screen", 7.,
     lambda ctx: righting_measure(ctx, "impact", 7., True), righting_bar,
     "推的是固定方向、固定力的一次撞击，不是一整套外力测试。",
     requires=())

task("eyes_follow_ball", "看", "一个球从眼前扫过，眼睛会不会跟着走", "screen", 2.5,
     lambda ctx: gaze_measure(ctx, sweep=.30, seconds=2.5, distance=.90, scenery=False),
     lambda m: gaze_bar(m, "gaze_error_rad"),
     "量的是眼球角度和球的真实方位之差。它不说明这只动物看见了球是什么。",
     requires=())

task("eyes_follow_fast_ball", "看", "球扫得快一点，眼睛还跟不跟得上", "full", 2.5,
     lambda ctx: gaze_measure(ctx, sweep=.60, seconds=2.5, distance=.90, scenery=False),
     lambda m: gaze_bar(m, "gaze_error_fast_rad"),
     "跟得上慢球和快球，不说明它能预判，也不说明它看见了颜色。",
     requires=("eyes_follow_ball",))

task("eyes_follow_the_ball_both_ways", "看", "球往左扫、再往右扫，两个方向都跟得上吗", "full", 5.,
     lambda ctx: sweep_pair_measure(ctx), sweep_pair_bar,
     "两个方向都跟得上，不说明它在别的速度、别的距离上也行。",
     requires=("eyes_follow_ball",))

task("eyes_on_a_new_thing", "看", "一样东西从旁边冒出来就动，眼睛会不会转过去", "full", 4.,
     lambda ctx: sudden_measure(ctx, 1., 3., .55), lambda m: sudden_bar(m, "sudden_shift_rad"),
     "它测的是「出现并移动」这一类。一样东西只是存在、从不变化，这个模型的眼"
     "睛是不会转过去的，那一点在下面 still 那一题里有读数。",
     requires=("eyes_follow_ball",))

task("eyes_hold_still_on_a_still_thing", "看", "一样东西就摆在那儿不动，眼睛会不会瞎转", "full", 3.,
     lambda ctx: parked_measure(ctx, .55, .90, 3.), still_bar,
     "这是上一题的反面：不动的画面就不该被追。它是「不动作」的门槛，"
     "所以它单独通过不能说明眼睛好用。",
     requires=("eyes_follow_ball",))

task("eyes_tell_near_from_far", "看", "一个球从两米外凑到跟前，脑里那份距离读数会满起来吗", "full", 4.,
     lambda ctx: approach_measure(ctx, .25, 2.00, 4.), approach_bar,
     "读的是距离细胞的总活动，不是眼睛的角度，也不是谁量出了几米。"
     "它不说明它在别的球、别的位置上还算得准。",
     requires=("eyes_follow_ball",))

task("eyes_see_a_pattern_not_a_blank", "看", "有花纹的画面和全黑，眼睛这一层读数一样吗", "full", 4.,
     lambda ctx: pattern_measure(ctx),
     lambda m: contrast_bar(m, "retina", "pattern_gain", "有花纹的画面"),
     "只说明眼睛感光这一层在画面刚出现时会醒一下。一块均匀的亮画面在这个模型里"
     "和全黑读一样，而且这一层约四分之一秒后就自己落了（下面的光细胞一直是 0.43）"
     "——这两件事本身也是读数，不全是好话。",
     requires=("eyes_follow_ball",))

task("turn_to_sound", "听", "左边和右边各响一下，眼睛有没有分别", "screen", 5.,
     lambda ctx: sound_pair_measure(ctx), sound_bar,
     "只比较左、右两个场景。响度和音高都是同一档，换一个频率未必还对。",
     requires=()) 

task("ears_tell_front_from_back", "听", "同一个声音在前面和后面，耳朵能分开吗", "full", 3.,
     lambda ctx: ear_contrast_measure(ctx, (1.10, 0.), (-1.10, 0.),
                                      ("auditory_pinna",), 3.),
     lambda m: separation_bar(m, "auditory_pinna", "pinna_gain", "耳廓前后"),
     "只说明耳廓那一层对前后有分别；它不说明它会转过头去找。",
     requires=("turn_to_sound",))

task("ears_tell_loud_from_far", "听", "近处响和远处响，耳朵里的读数一样吗", "full", 3.,
     lambda ctx: ear_contrast_measure(ctx, (1.10, 0.), (3.20, 0.),
                                      ("cochlea",), 3.),
     lambda m: contrast_bar(m, "cochlea", "loud_far_gain", "近处和远处"),
     "远近在这里是两个固定位置；它不说明它会用响度去估距离。",
     requires=("turn_to_sound",))

task("ears_tell_high_from_low", "听", "高低两个音，耳朵能不能分开", "full", 3.,
     lambda ctx: pitch_measure(ctx, 3.),
     lambda m: reading_bar(m, "gain", "high_low_gain"),
     "读的是整组耳蜗细胞的总变化；分的到底是不是「音高」这一题不下结论。",
     requires=("turn_to_sound",))

task("a_bang_makes_it_startle", "听", "突然来一声，脑里有没有受惊这回事", "full", 2.,
     lambda ctx: contrast_read(ctx, {"startle": 1., "seconds": 2., "clean": True},
                               {"startle": 0., "seconds": 2., "clean": True}, ("startle",)),
     lambda m: contrast_bar(m, "startle", "startle_gain", "受惊细胞"),
     "输入线直接连到受惊细胞；通了只说明这根线还在，不说明它害怕。",
     requires=())

task("a_sound_keeps_startling_less", "听", "连着响很多声，受惊会不会一次比一次轻", "full", 4.,
     lambda ctx: habituation_measure(ctx, 4.), habituation_bar,
     "读的是受惊细胞自己往下掉多少。它是疲劳，不是「学会了不用怕」。",
     requires=("a_bang_makes_it_startle",))

task("feels_a_touch_on_the_front", "摸与本体", "身体前面被碰和后面被碰，脑里的读数一样吗", "full", 4.,
     lambda ctx: contrast_read(ctx, {"environment": one_foot("body_touch", 0, .8), "seconds": 2.,
                                     "clean": True},
                               {"environment": one_foot("body_touch", 2, .8), "seconds": 2.,
                                "clean": True}, ("brake", "retreat")),
     lambda m: contrast_bar(m, "retreat", "front_touch_gain", "前面被碰"),
     "碰的是控制器里的身体触觉通道，不是真的手。身体触觉分的是前/左/后/右四块，"
     "不是四只脚——四只脚走的是另一条通道。",
     requires=("stands_still",))

task("lifts_the_foot_that_was_caught", "摸与本体", "一只脚被绊住，抬起的是不是那只脚", "full", 4.,
     lambda ctx: one_foot_measure(ctx, "foot_obstacle", 2, ("clearance",), 3., .8, True),
     lambda m: both_sides_bar(m, "gain", "swapped_gain", "obstacle_gain", "被绊的那条腿"),
     "身体被按住不动，只有腿上的读数和相位在跑；抬脚动作本身这一题不测。",
     requires=("walk_flat",))

task("feels_a_slipping_foot", "摸与本体", "一只脚打滑，脑里知道是哪只吗", "full", 4.,
     lambda ctx: one_foot_measure(ctx, "foot_slip", 1, ("withdrawal",), 3., .8, True),
     lambda m: both_sides_bar(m, "gain", "swapped_gain", "slip_gain", "打滑的那条腿"),
     "读的是打滑下游那个慢细胞；它不说明它会改步态。",
     requires=("walk_flat",))

task("feels_which_way_it_is_leaning", "摸与本体", "身体往一边歪，脑里有没有方向", "full", 4.,
     lambda ctx: tilt_measure(ctx, .35, ("lean",), 2.), leaning_bar,
     "歪的方向是直接喂进去的重力方向；它只说明方向细胞接对了，"
     "不说明它靠这个站住了。",
     requires=("stands_still",))

task("braces_when_it_is_tipped_over", "摸与本体", "快被推倒时，护住自己的那类细胞会不会醒", "full", 4.,
     lambda ctx: tilt_measure(ctx, 1.20, ("protective_tilt",), 2.), tipped_bar,
     "读的是一个「快要倒了」的细胞；它亮起来不等于它真的会做出保护动作。",
     requires=("stands_still",))

task("feels_a_load_on_one_foot", "摸与本体", "一只脚上压了重物，是哪只脚上的细胞在响", "full", 4.,
     lambda ctx: one_foot_measure(ctx, "foot_load", 3, ("impact_adaptation",), .5, .9, True,
                                  tail=.2),
     lambda m: both_sides_bar(m, "gain", "swapped_gain", "load_gain", "承重的那条腿"),
     "这个细胞自己会很快疲劳，所以门槛看的是刚压上去那一小段。",
     requires=("feels_a_touch_on_the_front",))

task("startled_but_still_hearing", "听", "听腻了那一声，耳朵是不是就听不见了", "full", 4.,
     lambda ctx: habituation_measure(ctx, 4.), hearing_bar,
     "读的是同一个场景里两个细胞：受惊往下掉的时候，耳蜗那一层还亮着。"
     "它是「习惯化不等于听不见」，不是「学会了不用怕」。",
     requires=("a_sound_keeps_startling_less",))

task("taught_reflex_sticks", "学习", "被摸二十秒教会\u201c这个声音=后退\u201d，教会了吗", "full", 32.,
     lambda ctx: taught_reflex_measure(ctx, 20.),
     taught_reflex_bar,
     "这一题允许老师在场（手摸到身体）。它证明的是这条通路能被写进去，"
     "不是它自己弄懂了声音的意思。",
     requires=())

task("nothing_is_taught_without_a_teacher", "学习", "没人教，光听声音，它会自己学会吗", "full", 32.,
     lambda ctx: taught_reflex_measure(ctx, 20., 0., False, False),
     taught_reflex_control_bar,
     "这是上一题的对照臂：同样的时长、同样的声音，只是没有手，"
     "所以「没学会」才是对的。",
     requires=("taught_reflex_sticks",))

task("the_lesson_is_still_there_later", "学习", "教完先安静一会儿，那件事还记得吗", "full", 42.,
     lambda ctx: taught_reflex_measure(ctx, 20., 10.),
     lambda m: taught_reflex_bar(m, "taught_reflex_kept"),
     "只隔十秒。这一题不说明能记多久，只说明它不是当场就散。",
     requires=("taught_reflex_sticks",))

task("visual_memory_holds", "记忆", "一块蓝色图案和身体接触配过几次，图案还有意义吗", "full", 26.,
     lambda ctx: visual_memory_measure(ctx, 5), visual_memory_bar,
     "量的是回避细胞对图案的反应比安静时高多少。它不说明它\u201c看见了一个东西\u201d。",
     requires=())

task("the_pattern_means_nothing_without_pairing", "记忆", "从没配对过的图案，会不会自己就有意义", "full", 10.,
     lambda ctx: visual_memory_measure(ctx, 0),
     lambda m: visual_memory_bar(m, "visual_memory_control", above=False),
     "这是配对那一题的对照臂：没配过就不该有意义，所以「没意义」才是对的。",
     requires=("visual_memory_holds",))

task("the_memory_is_still_there_later", "记忆", "配完对再隔一会儿，还认得那块图案吗", "full", 34.,
     lambda ctx: visual_memory_measure(ctx, 5, gap=4),
     lambda m: visual_memory_bar(m, "visual_memory_kept"),
     "隔的是四段各两秒的空转，不是几分钟；这一题不说明长期记忆。",
     requires=("visual_memory_holds",))

task("an_idle_animal_does_not_walk_for_a_touch", "注意", "没人叫它走，光是身上有接触，它会自己走起来吗", "full", 8.,
     lambda ctx: stand_still_measure(ctx, 8., injected={"body_touch": .8}), quiet_bar,
     "接触是喂进触觉通道的，不是真的手。它测的是「无关的输入会不会把它带跑」。",
     requires=("stands_still",))

task("an_idle_animal_does_not_walk_for_a_ball", "注意", "没人叫它走，眼前摆个东西，它会自己走起来吗", "full", 8.,
     lambda ctx: stand_still_measure(ctx, 8.), quiet_bar,
     "空场里站着，道具都还在原地。它测的是「没有指令时它会不会自己找事做」。",
     requires=("stands_still",))

task("moving_is_not_being_pushed", "自我", "自己走路的晃动，和被人推一把，是不是两回事", "full", 13.,
     lambda ctx: own_gait_measure(ctx, 6.), pushing_bar,
     "两边读的都是同一个「快倒了」细胞的峰值；它只说明被推和自己走在这条线上"
     "分得开，不说明它知道那个推力来自外面。",
     requires=("walk_flat", "get_up_from_back"))

task("the_body_is_not_a_foot", "自我", "身上被碰和脚上被绊，是同一件事吗", "full", 4.,
     lambda ctx: body_and_foot_measure(ctx, 2.),
     lambda m: both_sides_bar(m, "body_retreat", "foot_withdrawal", "body_is_not_a_foot",
                              "身上和脚上"),
     "两条通道本来就该接在不同的细胞上，所以这一题只是在查它们没有接反。",
     requires=("feels_a_touch_on_the_front",))# What each question reads off the body and the cells, and the rule it is
# judged by, in words.  Kept beside the code so --explain can print the whole
# bank without anyone having to read the lambdas.

# ---- 第二批：把站着、走着、转着、上下地形、听着、摸着再问细一点
task("walks_in_a_straight_line", "走路", "它自己走的时候，走的是一条线还是一路画龙", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: at_least_bar(m, "straightness", "straightness", "直度"),
     "直度是净位移除以走过的路程。它一路画龙但没摔倒也会低于 1。这一题不说明它会朝一个目标走。",
     requires=("walks_without_being_told",))

task("does_not_crab_sideways", "走路", "它自己走的时候，是往前走还是往旁边蹭", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: at_most_bar(m, "lateral_m", "crab_m", "往旁边蹭的距离"),
     "量的是终点相对起点、垂直于出发朝向的那一段。它不说明走得直，只说明没横着漂。",
     requires=("walks_without_being_told",))

task("holds_its_heading_while_it_walks", "走路", "它自己走的时候，朝向会不会越走越歪", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: at_most_bar(m, "yaw_path_rad", "heading_hold_rad", "一路累计转头"),
     "读的是一路上朝向一共转过多少（不管左右，算绝对值累计）。"
     "它不说明它能不能拐弯，只说明它不会自己慢慢打转。",
     requires=("walks_without_being_told",))

task("walks_for_twenty_seconds", "走路", "没人管它二十秒，它是一直走还是走两步就停", "full", 20.,
     lambda ctx: walk_measure(ctx, 20., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_twenty_m"),
     "和 16 秒那一题同一条线，只是时间更长；它不说明它能走多远，只说明它不会自己停下来。",
     requires=("keeps_walking_without_being_told",))

task("walks_at_a_steady_pace", "走快", "它自己走的时候，平均速度够不够快", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: pace_bar(m, "平均速度"),
     "出厂动物主要是在原地踏步，平均速度只有 0.09 米/秒。速度来自先天步态，"
     "所以这一题是给「步态调得更快」的候选留的。",
     requires=("walks_without_being_told",))

task("covers_ground_in_sixteen_seconds", "走快", "十六秒里到底走出去了多少米", "full", 16.,
     lambda ctx: walk_measure(ctx, 16., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_ground_m"),
     "它只算净位移，绕圈子走也会算低。它不说明它知道要去哪。",
     requires=("keeps_walking_without_being_told",))

task("turns_towards_a_sound_while_walking", "转弯", "一边走一边左边响或右边响，身体会朝那边转吗", "full", 16.,
     lambda ctx: turn_pair_measure(ctx, 8., "autonomous"),
     lambda m: turn_pair_bar(m, "turn_pair_rad", "边走边听"),
     "两边各走一次八秒，比的是收尾时朝向差多少。它只说明声音能左右引导身体，"
     "不说明它能奔着声音走过去。",
     requires=("walks_without_being_told", "turn_to_sound"))

task("turns_towards_a_sound_while_standing", "转弯", "站着不动时一边响一下，身体会不会朝那边偏", "full", 16.,
     lambda ctx: stand_turn_measure(ctx, 8.),
     lambda m: stand_turn_bar(m, "站着听"),
     "和上一题同样的两个声源，只是它站着不动。它测的是「不用走路也会转」，"
     "不说明它站得稳不稳。",
     requires=("stands_still", "turn_to_sound"))

task("does_not_spin_on_the_spot", "转弯", "没人叫它动，它是往前走，还是原地打转", "screen", 8.,
     lambda ctx: stand_still_measure(ctx, 8.),
     lambda m: idle_pace_bar(m, "没人管它时"),
     "读的是八秒里朝向转过多少、同时离出发点走了多远。转多少本身不是好坏："
     "场地就这么大，会走路的动物必然会转。只有「转得多、同时没走出去」才算原地打转。",
     requires=("stands_still",))

task("walks_down_the_ramp", "地形", "站在坡顶上往前走，它能不能顺坡下来", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "ramp_top", start_yaw=math.pi),
     lambda m: down_bar(m, "down_ramp_m", "坡"),
     "它只要走出坡的边缘就算过。它不测下坡时稳不稳，只测有没有卡在坡顶。",
     requires=("walk_ramp",))

task("steps_down_the_low_step", "地形", "站在矮台阶上往前走，它能不能走下来", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "step_top", start_yaw=math.pi),
     lambda m: down_bar(m, "down_step_m", "台阶"),
     "台阶只有一厘米多高，所以这一题其实是「从高一点点的地方迈下来」。",
     requires=("walk_low_step",))

task("steps_down_from_the_platform", "地形", "站在矮台上往前走，它能不能走到地面上", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "platform_top", start_yaw=math.pi),
     lambda m: down_bar(m, "down_platform_m", "台子"),
     "台子比台阶高，所以这一题比上面那题难一点；它不测落地缓冲，只测有没有下来。",
     requires=("climbs_onto_the_platform",))

task("goes_around_the_block", "地形", "正前方横着一块挡板，它会不会绕开", "full", 12.,
     lambda ctx: walk_measure(ctx, 12., "autonomous", "blocked"),
     around_bar,
     "过线的条件是既越过了挡板、又确实往旁边挪过一段。它不说明它看见了挡板，"
     "也不说明它知道往哪边绕更好。",
     requires=("stops_at_the_wall",))

task("a_touch_on_the_front_holds_it_back", "摸与本体", "身上一直被碰着，它还会照直走吗", "full", 16.,
     lambda ctx: held_back_measure(ctx, 8., injected={"body_touch": [.8, 0., 0., 0.]}),
     lambda m: held_back_bar(m, "touch_hold_fraction", "身上被碰"),
     "比的是同一个驱动、同一段时间，碰和不碰各走多远。碰是喂进触觉通道的，不是真的手。",
     requires=("feels_a_touch_on_the_front", "walks_without_being_told"))

task("a_bang_stops_it_walking", "听", "走得好好的突然来一声，它会停下来吗", "full", 16.,
     lambda ctx: held_back_measure(ctx, 8., startle=1.),
     lambda m: held_back_bar(m, "touch_hold_fraction", "那一声砰"),
     "收尾的读数还是位移：受惊这条通路本来就接着运动。它不说明它知道声音从哪来。",
     requires=("a_bang_makes_it_startle", "walks_without_being_told"))

task("gets_up_after_a_hard_shove", "自救", "站着被人用更大的力气推倒，它还能起来吗", "full", 7.,
     lambda ctx: righting_measure(ctx, "impact", 7., True, force=220.),
     righting_bar,
     "和「被推一把」是同一套判据，只是那一下更重。它不说明它扛得住连续推。",
     requires=("stay_up_when_pushed",))

task("eyes_converge_on_a_near_thing", "看", "东西凑到跟前时，两只眼睛会不会一起往里收", "full", 4.,
     lambda ctx: convergence_measure(ctx, .25, 2.0, 2.),
     lambda m: reading_bar(m, "gain", "near_far_angle"),
     "读的是眼球自己转到哪（眼肌的本体读数），不是画面。"
     "没有这一步之前，这一层细胞一直是静止的偏置，读不出任何东西。",
     requires=("eyes_follow_ball",))

task("eyes_look_at_a_still_thing_at_the_side", "看", "一样东西停在旁边不动，眼睛会转到那边去吗", "full", 4.,
     lambda ctx: gaze_static_measure(ctx, .45, 1.00, 2.),
     lambda m: both_ways_bar(m, "static_gaze_rad", "停在旁边的东西"),
     "左边停一个球、右边停一个球，各起一个全新的脑子，看最后眼球停在哪儿。"
     "它只说明眼睛会朝那一边去，不说明它看得清那是什么。",
     requires=("eyes_follow_ball",))

task("its_own_eyes_are_felt", "看", "眼睛自己转到哪，脑里知不知道自己转了", "full", 2.,
     lambda ctx: eye_feel_measure(ctx, .5, 2.),
     lambda m: reading_bar(m, "gain", "own_eyes_are_felt"),
     "读的是眼球本体那组细胞，在一个眼睛被命令转开、一个不转的对比下差多少。"
     "这条线本身很弱，门槛就是按出厂动物自己的读数留的。",
     requires=("eyes_follow_ball",))

task("eyes_hold_a_slow_thing", "看", "东西慢慢扫过，眼睛还跟不跟得住", "full", 5.,
     lambda ctx: gaze_measure(ctx, .30, 5., .90, scenery=False),
     lambda m: gaze_bar(m, "slow_gaze_rad"),
     "和追球那一题是同一条线，只是速度减半。慢的比快的更难过，"
     "因为慢的目标每帧变化小。",
     requires=("eyes_follow_ball",))

task("ears_hear_something_far_away", "听", "很远的地方响一声，耳朵还听得见吗", "full", 4.,
     lambda ctx: faint_sound_measure(ctx, 2.),
     lambda m: contrast_bar(m, "cochlea", "quiet_sound_gain", "远处那一声"),
     "比的是三米外的一声和一间没有声音的屋子。它测的是听得见的下限，"
     "不是听得准不准。",
     requires=("startled_but_still_hearing",))

task("ears_hear_a_sound_behind", "听", "声音在背后和在前头，耳朵里的读数一样吗", "full", 4.,
     lambda ctx: sound_behind_measure(ctx, 2.),
     lambda m: behind_bar(m),
     "读的是耳廓那组细胞前后两格。它只说明前后分得开，不说明它知道那是什么声音。",
     requires=("ears_tell_front_from_back",))

task("two_sounds_at_once_are_still_two", "听", "两个地方同时响，耳朵里会不会糊成一团", "full", 4.,
     lambda ctx: two_sounds_measure(ctx, 2.),
     lambda m: separation_bar(m, "auditory_pinna", "two_sounds_gain", "两处一起响"),
     "比的是两个声源同时亮起，和只亮一个。它不说明它能同时听清两件事。",
     requires=("ears_tell_front_from_back",))

task("feels_a_touch_on_the_left_side", "摸与本体", "左边被碰，脑里会不会往右边让", "full", 4.,
     lambda ctx: side_touch_measure(ctx, (1, 3), ("avoidance",), (0, 1)),
     lambda m: both_sides_bar(m, "left_touch_gain", "right_touch_gain", "side_touch_gain",
                              "左边和右边被碰"),
     "读的是两条相反的让路细胞：左边被碰该亮左让那一条，右边被碰该亮另一条。",
     requires=("feels_a_touch_on_the_front",))

task("feels_a_touch_behind_it", "摸与本体", "背后被碰，脑里有没有「后面」这一格", "full", 4.,
     lambda ctx: side_touch_measure(ctx, (2, 0), ("wall_contact",), (2, 0)),
     lambda m: both_sides_bar(m, "left_touch_gain", "right_touch_gain", "back_touch_gain",
                              "后面和前面被碰"),
     "读的是四格「身上贴着东西」的慢细胞：后面被碰该亮后面那格，前面被碰该亮前面那格。"
     "它只说明四格没有接反，不说明它靠这个量出了位置。",
     requires=("feels_a_touch_on_the_front",))

task("feels_a_touch_on_both_sides_at_once", "摸与本体", "同时碰到两处，脑里还算得清吗", "full", 4.,
     lambda ctx: two_touch_measure(ctx, .8, 2.),
     two_touch_bar,
     "比的是「只碰前面」和「前面加左边一起碰」。它测的是两路能不能同时存在，"
     "不说明它分得清先后。",
     requires=("feels_a_touch_on_the_left_side",))

task("feels_the_ground_under_the_back_legs", "摸与本体", "后腿踩实了没有，是哪条腿的报告", "full", 4.,
     lambda ctx: one_foot_measure(ctx, "foot_load", 2, ("impact_adaptation",), .5, .9, True,
                                  tail=.2),
     lambda m: both_sides_bar(m, "gain", "swapped_gain", "ground_feet_gain", "承重的后腿"),
     "和前面那条承重的题是同一条线，只是换成后腿。四条腿各接一遍才算接全。",
     requires=("feels_a_load_on_one_foot",))

task("learns_the_lesson_in_five_seconds", "学习", "只教五秒，够不够学会「这个声音=后退」", "full", 17.,
     lambda ctx: taught_reflex_measure(ctx, 5.),
     taught_reflex_bar,
     "和二十秒那一题是同一套判据，只是时间缩短到四分之一。它测的是学得多快，"
     "不是学得多好。",
     requires=("taught_reflex_sticks",))

task("the_lesson_survives_a_longer_pause", "学习", "教完之后安静三十秒，那件事还在吗", "full", 62.,
     lambda ctx: taught_reflex_measure(ctx, 20., 30.),
     lambda m: taught_reflex_bar(m, "taught_reflex_kept"),
     "比十秒那一题长得多，但仍然只是十几秒量级；它不说明长期记忆。",
     requires=("the_lesson_is_still_there_later",))

task("remembers_a_second_pattern", "记忆", "换一块别的颜色，配过对以后还有意义吗", "full", 26.,
     lambda ctx: visual_memory_measure(ctx, 5, channel=1),
     visual_memory_bar,
     "和蓝色那块图案用同一套配对和判据，只是换成绿色。它测的是这套机制对"
     "另一块图案也一样，不说明它能同时记住两块。",
     requires=("visual_memory_holds",))

task("remembers_after_one_pairing", "记忆", "只配对一次，够不够", "full", 14.,
     lambda ctx: visual_memory_measure(ctx, 1),
     lambda m: visual_memory_bar(m, "one_pairing_gain"),
     "它测的是一次就够不够，不是记多久。配得少还读得出来，说明这条线写得比较快。",
     requires=("visual_memory_holds",))


# ---------------------------------------------------------------------------
# third batch: standing on its own, the quality of the gait, turning and coming
# back, the lower rungs of the eyes and the ears, touch on every sector, what
# the lower layers of the picture carry, three more lessons, two memories at
# once, what pulls it forward, and whether it knows it went over
# ---------------------------------------------------------------------------
def stand_long_bar(measures):
    """Twenty idle seconds: it may drift, it may not go over."""
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < .9:
        return False, ("went over while nothing was asking it to move (min up_z %.3f)"
                       % measures["min_up_z"])
    if measures["displacement_m"] > BAR["stand_long_m"]:
        return False, ("wandered %.3f m with nothing asking it to (bar %.3f)"
                       % (measures["displacement_m"], BAR["stand_long_m"]))
    return True, ("stood for the whole twenty seconds: %.3f m of drift, min up_z %.3f"
                  % (measures["displacement_m"], measures["min_up_z"]))


def slope_stand_bar(measures):
    """Standing still on a slope.  It may slide down; it may not go over."""
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < BAR["slope_stand_z"]:
        return False, ("lost the slope (min up_z %.3f, bar %.3f)"
                       % (measures["min_up_z"], BAR["slope_stand_z"]))
    return True, "stayed upright on the slope (min up_z %.3f)" % measures["min_up_z"]


def foot_clearance_parts(measures):
    return [None if part.get("fraction_gt_2mm") is None else float(part["fraction_gt_2mm"])
            for part in measures["clearance_parts"]]


def foot_lift_bar(measures):
    """Every one of the four feet has to leave the ground sometimes."""
    passed, why = walk_bar(measures, "walk_flat_m")
    if not passed:
        return False, why
    parts = foot_clearance_parts(measures)
    worst = min(parts)
    if worst < BAR["foot_lift_fraction"]:
        return False, ("one foot never left the ground (front %.3f/%.3f rear %.3f/%.3f, bar %.3f)"
                       % (parts[0], parts[1], parts[2], parts[3], BAR["foot_lift_fraction"]))
    return True, ("all four feet lifted (front %.3f/%.3f rear %.3f/%.3f)"
                  % (parts[0], parts[1], parts[2], parts[3]))


def leg_symmetry_bar(measures):
    """The two legs of a pair have to do comparable work."""
    passed, why = walk_bar(measures, "walk_flat_m")
    if not passed:
        return False, why
    parts = foot_clearance_parts(measures)
    worst = max(abs(parts[0] - parts[1]), abs(parts[2] - parts[3]))
    if worst > BAR["leg_symmetry"]:
        return False, ("one leg of a pair did the lifting (front %.3f/%.3f, rear %.3f/%.3f, bar %.3f)"
                       % (parts[0], parts[1], parts[2], parts[3], BAR["leg_symmetry"]))
    return True, ("both pairs stepped alike (front %.3f/%.3f, rear %.3f/%.3f)"
                  % (parts[0], parts[1], parts[2], parts[3]))


def passage_quiet_bar(measures):
    """Threading the passage: it has to come out and it may not scrape it."""
    passed, why = walk_bar(measures, "walk_passage_m")
    if not passed:
        return False, why
    if not measures["crossed_marker"]:
        return False, ("walked %.3f m but never came out of the passage (%.3f m past the start)"
                       % (measures["displacement_m"], measures["progress_m"]))
    touch = float(measures["peak"].get("wall_contact", 0.))
    if touch > BAR["passage_quiet_touch"]:
        return False, ("banged through the passage (wall cells peaked at %.4f, bar %.4f)"
                       % (touch, BAR["passage_quiet_touch"]))
    return True, ("came through without leaning on it (wall cells peaked at %.4f)" % touch)


def wall_upright_bar(measures):
    """Walking into a wall: it may be stopped, it may not be knocked over."""
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < .9:
        return False, ("went over on the way into the wall (min up_z %.3f)" % measures["min_up_z"])
    if measures["maximum_x"] > 3.42:
        return False, "walked through the wall (reached x %.3f)" % measures["maximum_x"]
    return True, ("stayed on its feet against the wall (reached x %.3f, min up_z %.3f)"
                  % (measures["maximum_x"], measures["min_up_z"]))

def step_windows(ctx, seconds, scenario, terrain, window):
    """Two windows of the same walk: how fast at the start, how fast at the end."""
    result = run_seed(seed=ctx["seed"], duration=seconds, model_path=ctx["model_path"],
                      scenario=scenario, controller_parameters_for_seed=ctx["parameters"],
                      terrain_start=terrain)
    metrics = result["metrics"]
    timeline = result["timeline"]
    times = np.asarray([float(sample["time"]) for sample in timeline], dtype=float)
    places = np.asarray([np.asarray(sample["position"], dtype=float)[:2] for sample in timeline])
    if len(times) < 4:
        return dict(status="error", error="too few samples to compare two windows")

    def speed_between(low, high):
        pick = (times >= low) & (times <= high)
        chosen, chosen_times = places[pick], times[pick]
        if len(chosen) < 2 or float(np.sum(np.diff(chosen_times))) <= 0.:
            return None
        moves = np.linalg.norm(np.diff(chosen, axis=0), axis=1)
        return float(np.sum(moves) / np.sum(np.diff(chosen_times)))

    first = speed_between(0., window)
    last = speed_between(seconds - window, seconds)
    return dict(status=result["status"], error=result["error"],
                behavior_pass=bool(result["behavior_pass"]),
                min_up_z=metrics["minimum_up_z"],
                displacement_m=metrics["total_displacement_m"],
                first_speed=first, last_speed=last,
                ratio=(None if not first else float(last / first)))


def fatigue_bar(measures):
    """A long walk must not turn into a shuffle: the end compares with the start."""
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if not measures["behavior_pass"]:
        return False, "not upright the whole way (min up_z %.3f)" % measures["min_up_z"]
    if measures["first_speed"] is None or measures["last_speed"] is None:
        return False, "the walk was too short to compare two windows"
    if measures["ratio"] < BAR["fatigue_ratio"]:
        return False, ("slowed to %.3f of its starting pace (%.3f m/s then, %.3f m/s later, bar %.3f)"
                       % (measures["ratio"], measures["first_speed"], measures["last_speed"],
                          BAR["fatigue_ratio"]))
    return True, ("held its pace: %.3f m/s then, %.3f m/s at the end"
                  % (measures["first_speed"], measures["last_speed"]))


def walk_turn_bar(measures, what):
    """Walking with a sound at one shoulder: turn towards it and keep going."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    for run in (measures["right"], measures["left"]):
        if run["status"] != "ok":
            return False, "run failed: %s" % run["error"]
        if not run["behavior_pass"]:
            return False, "it fell over (min up_z %.3f)" % run["min_up_z"]
        if run["progress_m"] < BAR["turn_walk_m"]:
            return False, ("%s: it stopped going anywhere (%.3f m, bar %.3f)"
                           % (what, run["progress_m"], BAR["turn_walk_m"]))
    if measures["difference"] > -BAR["turn_walk_rad"]:
        return False, ("%s did not turn it (%.3f rad apart, bar %.3f)"
                       % (what, measures["difference"], BAR["turn_walk_rad"]))
    return True, ("%s: turned it %.3f rad and it kept walking (%.3f m and %.3f m)"
                  % (what, measures["difference"], measures["right"]["progress_m"],
                     measures["left"]["progress_m"]))


def back_away_bar(measures, what):
    """A hand on its chest: it may not keep going forward."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["min_up_z"] < .9:
        return False, "went over while being held (min up_z %.3f)" % measures["min_up_z"]
    if -measures["back_m"] < BAR["back_away_m"]:
        return False, ("%s did not back it off (%.3f m, bar %.3f)"
                       % (what, -measures["back_m"], BAR["back_away_m"]))
    return True, "%s backed it off %.3f m" % (what, -measures["back_m"])


def behind_hold_bar(measures):
    """A sound at its back must not swing its heading and must not stop it."""
    passed, why = walk_bar(measures, "walk_own_m")
    if not passed:
        return False, why
    if abs(measures["yaw_rad"]) > BAR["behind_hold_rad"]:
        return False, ("the sound behind swung its heading %.3f rad (bar %.3f)"
                       % (abs(measures["yaw_rad"]), BAR["behind_hold_rad"]))
    return True, ("kept its line with a sound behind it (%.3f rad of drift)"
                  % abs(measures["yaw_rad"]))


def gaze_pitch_measure(ctx, rise=.30, distance=.90, seconds=2.5):
    """The ball sweeps down the picture: does the eye pitch follow it?"""
    body = _clean_body()
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    brain, eyes = _eye_brain(body, ctx)
    observation = body.observe()
    environment = blank_environment()
    errors = []
    steps = max(2, int(round(seconds / DT)))
    try:
        for step in range(steps):
            share = step / float(steps - 1)
            body.model.geom_pos[target] = [.30 + distance, 0., .32 + rise - 2. * rise * share]
            mujoco.mj_forward(body.model, body.data)
            command = brain.eye_command()
            offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
            rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
            elevation = float(np.arctan2(offset @ rotation[:, 2], offset @ rotation[:, 0]))
            if step > 20:
                errors.append(.5 * (command[1] + command[3]) - elevation)
            observation = _eye_step(body, brain, eyes, observation, environment)
    finally:
        eyes.close()
    if not errors:
        return dict(status="error", error="no samples")
    return dict(status="ok", error=None,
                mean_abs_pitch_error_rad=float(np.abs(np.array(errors)).mean()),
                samples=len(errors))


def gaze_pitch_bar(measures, bar_key):
    if measures["status"] != "ok":
        return False, "run failed: %s" % measures["error"]
    if measures["mean_abs_pitch_error_rad"] > BAR[bar_key]:
        return False, ("the eye pitch stayed %.3f rad off the ball (bar %.3f)"
                       % (measures["mean_abs_pitch_error_rad"], BAR[bar_key]))
    return True, "the eye pitch stayed %.3f rad from the ball" % measures["mean_abs_pitch_error_rad"]


def recentre_measure(ctx, lateral=.45, distance=1.00, seconds=3.):
    """The ball is off to one side and then it is gone: do the eyes come back?"""
    body = _clean_body()
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.10] * 3
    brain, eyes = _eye_brain(body, ctx)
    observation = body.observe()
    environment = blank_environment()
    held, gone = [], []
    try:
        for step in range(int(round(seconds / DT))):
            here = step * DT
            body.model.geom_pos[target] = ([distance, lateral, .32] if here < seconds / 2.
                                           else [60., 60., -8.])
            mujoco.mj_forward(body.model, body.data)
            observation = _eye_step(body, brain, eyes, observation, environment)
            if here > seconds / 4.:
                (held if here < seconds / 2. else gone).append(_gaze_now(brain))
    finally:
        eyes.close()
    return dict(status="ok", error=None, held=float(np.mean(held)), gone=float(np.mean(gone)))


def recentre_bar(measures):
    """It has to have gone to the thing before coming off it: an eye that never
    moved also reads "off centre", so both halves are asked for."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["held"] < BAR["static_gaze_rad"]:
        return False, ("the eyes never went to the ball in the first place (%.4f rad)"
                       % measures["held"])
    if abs(measures["gone"]) > BAR["recentre_rad"]:
        return False, ("the eyes stayed on the empty place (%.4f rad, bar %.4f)"
                       % (measures["gone"], BAR["recentre_rad"]))
    return True, ("it looked (%.4f rad) and then let go (%.4f rad)"
                  % (measures["held"], measures["gone"]))

def colour_measure(ctx, seconds=.35, tail=.35):
    """The same square, one painted red and one painted green."""
    red = np.zeros((2, 36, 48, 3), dtype=np.uint8)
    green = np.zeros((2, 36, 48, 3), dtype=np.uint8)
    red[:, 8:28, 16:32, 0] = 255
    green[:, 8:28, 16:32, 1] = 255
    return contrast_read(ctx,
                         {"picture": red, "seconds": seconds, "tail": tail, "clean": True},
                         {"picture": green, "seconds": seconds, "tail": tail, "clean": True},
                         ("retinal_opponent", "photoreceptors"))


def _bands():
    """The stripes turned through a right angle: the same edges, laid the other way."""
    image = np.zeros((2, 36, 48, 3), dtype=np.uint8)
    for row in range(36):
        if (row // 4) % 2 == 0:
            image[:, row, :, :] = 220
    return image


def feature_layer_measure(ctx, seconds=.35, tail=.35):
    """Two pictures that differ: do the layers under the surface differ too?

    Nothing here says the animal knows what either picture is.  It asks whether
    the cells the eye hands its picture to carry the picture at all, or whether
    two different pictures land on the same reading.
    """
    return contrast_read(ctx,
                         {"picture": _stripes(), "seconds": seconds, "tail": tail, "clean": True},
                         {"picture": _bands(), "seconds": seconds, "tail": tail, "clean": True},
                         ("photoreceptors", "retinal_interneurons", "retina"))


def size_measure(ctx, seconds=2., distance=.90):
    """A big ball and a small one, in the same place."""
    return contrast_read(ctx,
                         {"gaze": (distance, 0., .32), "gaze_size": .18, "look": True,
                          "clean": True, "seconds": seconds, "tail": seconds},
                         {"gaze": (distance, 0., .32), "gaze_size": .05, "look": True,
                          "clean": True, "seconds": seconds, "tail": seconds},
                         ("retinal_change",))


def sound_gap_measure(ctx, seconds=4., place=(1.10, 0.)):
    """The same emitter: on for the first half, gone for the second."""
    body = _clean_body()
    source = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "sound_low")
    brain = brain_for(ctx, body)
    ears = BinauralSenses(body, window_samples=160)
    observation = body.observe()
    environment = blank_environment()
    loud, quiet = [], []
    try:
        for step in range(int(round(seconds / DT))):
            here = step * DT
            body.model.geom_pos[source] = ([place[0], place[1], .30] if here < seconds / 2.
                                           else [60., 60., -8.])
            mujoco.mj_forward(body.model, body.data)
            brain.step(observation, environment=environment, ear_waveform=ears.observe(),
                       dt=DT, learn=False)
            reading = float(np.mean(brain.network.activity[brain.groups["cochlea"]]))
            if seconds / 2. - .5 <= here < seconds / 2.:
                loud.append(reading)
            elif here >= seconds - .5:
                quiet.append(reading)
    except Exception as exc:                        # a failed exam is a result
        return dict(status="error", error="%s: %s" % (type(exc).__name__, exc))
    if not loud or not quiet:
        return dict(status="error", error="no samples")
    loud_mean, quiet_mean = float(np.mean(loud)), float(np.mean(quiet))
    return dict(status="ok", error=None, loud=loud_mean, quiet=quiet_mean,
                ratio=(0. if not loud_mean else float(quiet_mean / loud_mean)))


def sound_gap_bar(measures, what):
    """A sound that stopped must stop being heard."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["loud"] < BAR["silence_floor"]:
        return False, ("%s was never heard in the first place (%.4f)" % (what, measures["loud"]))
    if measures["ratio"] > BAR["sound_gap_ratio"]:
        return False, ("%s kept ringing after it stopped (%.4f of %.4f, bar %.4f)"
                       % (what, measures["quiet"], measures["loud"], BAR["sound_gap_ratio"]))
    return True, ("%s fell from %.4f to %.4f when it stopped"
                  % (what, measures["loud"], measures["quiet"]))


def opposite_bar(measures, pairs, what, quiet_bar="silence_floor",
                 reading_bar="opposite_reading"):
    """For each pair: nothing came in on the channel, and the other cell says so.

    Every sense is two cells: one carries what arrived, the other carries what
    did not.  In a scene where nothing arrives, the first has to be quiet and
    the second has to be lit - not the other way round, and not both.
    """
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    for quiet_key, opposite_key in pairs:
        quiet = float(np.max(np.atleast_1d(measures["parts"][quiet_key])))
        if quiet > BAR[quiet_bar]:
            return False, ("%s: %s was not quiet (%.5f, bar %.5f)"
                           % (what, quiet_key, quiet, BAR[quiet_bar]))
        lit = float(np.min(np.atleast_1d(measures["parts"][opposite_key])))
        if lit < BAR[reading_bar]:
            return False, ("%s: %s stayed at %.5f, bar %.5f"
                           % (what, opposite_key, lit, BAR[reading_bar]))
    return True, ("%s: every channel that received nothing had its opposite cell lit"
                  % what)


def full_pair_bar(measures, key, what, tolerance=.10):
    """A channel and its opposite carry one thing between them, cell by cell."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    driven = np.atleast_1d(measures["parts"][key])
    opposite = np.atleast_1d(measures["parts"][key + "_opposite"])
    error = float(np.max(np.abs(driven + opposite - 1.)))
    if error > tolerance:
        return False, ("%s: %s and its opposite add up to one channel only within "
                       "%.3f, bar %.3f" % (what, key, error, tolerance))
    return True, ("%s: %s and its opposite add up to one channel (worst %.3f off)"
                  % (what, key, error))


def at_most_cells_bar(measures, key, bar_key, what):
    """No cell in the group may be over the bar: here the answer is a silence."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    values = np.asarray(measures["parts"][key], dtype=float)
    if not values.size:
        return False, "no reading for %s" % key
    if float(values.max()) > BAR[bar_key]:
        return False, ("%s lit %s, over the bar %.4f"
                       % (what, ["%.4f" % value for value in values], BAR[bar_key]))
    return True, "%s stayed at %s" % (what, ["%.4f" % value for value in values])


def sound_side_measure(ctx, seconds=2.):
    """The same emitter level with one shoulder and then with the other."""
    return contrast_read(ctx,
                         {"ear": (0., 1.10, "sound_low"), "seconds": seconds, "clean": True},
                         {"ear": (0., -1.10, "sound_low"), "seconds": seconds, "clean": True},
                         ("auditory_pinna", "auditory_spatial"))


def same_side_pair_measure(ctx, seconds=2.):
    """Two emitters at one shoulder against one of them alone."""
    together = {"sound_low": (.20, 1.20, .30), "sound_high": (.20, .80, .30)}
    alone = dict(together, sound_high=(60., 60., -8.))
    return contrast_read(ctx,
                         {"ear": False, "props": together, "seconds": seconds, "clean": True},
                         {"ear": False, "props": alone, "seconds": seconds, "clean": True},
                         ("auditory_pinna",))


def startle_behind_measure(ctx, seconds=2.):
    """A bang from behind it against the same room with no bang."""
    return contrast_read(ctx,
                         {"ear": (-1.10, 0., "sound_low"), "startle": 1., "seconds": seconds,
                          "clean": True, "tail": .5},
                         {"ear": (-1.10, 0., "sound_low"), "startle": 0., "seconds": seconds,
                          "clean": True, "tail": .5},
                         ("startle",))


def light_touch_measure(ctx, seconds=2., value=.30):
    """A hand resting on one side, lightly: the cells still have to say which."""
    return side_touch_measure(ctx, (1, 3), ("avoidance",), (0, 1), value=value, seconds=seconds)


def each_foot_measure(ctx, channel="foot_obstacle", cells=("clearance",), value=.8, seconds=.5):
    """One leg at a time: when a leg is caught, is it that leg's own cell that lights?"""
    name = cells[0]
    readings = []
    for leg in range(4):
        reading = _foot_channel(ctx, channel, leg, cells, seconds, value, True, None, True, .2)
        readings.append(np.asarray(reading["parts"][name], dtype=float))
    right = sum(1 for leg, values in enumerate(readings)
                if values.size > leg and float(np.argmax(values)) == leg)
    return dict(status="ok", error=None, right_legs=right,
                found=[float(np.max(values)) for values in readings],
                parts=[values.tolist() for values in readings])


def all_feet_bar(measures, bar_key, what):
    """Each of the four legs has to report itself and none of the others."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["right_legs"] < 4:
        return False, ("only %d of the four legs reported themselves (bare values %s)"
                       % (measures["right_legs"],
                          ["%.3f" % value for value in measures["found"]]))
    return True, ("all four legs reported themselves (%s)"
                  % ["%.3f" % value for value in measures["found"]])

def extinction_measure(ctx, lesson_seconds, quiet_seconds, probe_seconds=4.):
    """Taught, then the same sound with nobody teaching: does the lesson fade?"""
    from tools import nursery
    run = nursery.Run(ctx["seed"], hand_teaches=True, gated=True, routes=True,
                      parameters=ctx["parameters"])
    try:
        run.phase(4., tone=True, walk=nursery.WALK, learn=False)
        run.phase(lesson_seconds, tone=True, walk=nursery.WALK, hand=True, learn=True)
        taught_tone = run.phase(probe_seconds, tone=True, walk=nursery.WALK, learn=False)
        taught_silence = run.phase(probe_seconds, tone=False, walk=nursery.WALK, learn=False)
        run.phase(quiet_seconds, tone=True, walk=nursery.WALK, learn=True)
        later_tone = run.phase(probe_seconds, tone=True, walk=nursery.WALK, learn=False)
        later_silence = run.phase(probe_seconds, tone=False, walk=nursery.WALK, learn=False)
    finally:
        run.close()
    first = taught_tone["retreat"] - taught_silence["retreat"]
    later = later_tone["retreat"] - later_silence["retreat"]
    return dict(status="ok", error=None, first_gain=first, later_gain=later, drop=first - later)


def extinction_bar(measures):
    """It has to have learned the lesson first, and the lesson has to fade."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["first_gain"] < BAR["taught_reflex_gain"]:
        return False, ("there was no lesson to forget (%.3f on the tone)"
                       % measures["first_gain"])
    allowed = (1. - BAR["extinction_drop"]) * measures["first_gain"]
    if measures["later_gain"] > allowed:
        return False, ("the lesson never faded: %.3f while the hand was there, %.3f with "
                       "nobody teaching (has to fall to %.3f)"
                       % (measures["first_gain"], measures["later_gain"], allowed))
    return True, ("the lesson fell from %.3f to %.3f with nobody teaching"
                  % (measures["first_gain"], measures["later_gain"]))


def second_lesson_measure(ctx, first_seconds, second_seconds, probe_seconds=4.):
    """Two short lessons, one after the other: does the second one add anything?"""
    from tools import nursery
    run = nursery.Run(ctx["seed"], hand_teaches=True, gated=True, routes=True,
                      parameters=ctx["parameters"])
    try:
        run.phase(4., tone=True, walk=nursery.WALK, learn=False)
        run.phase(first_seconds, tone=True, walk=nursery.WALK, hand=True, learn=True)
        first_tone = run.phase(probe_seconds, tone=True, walk=nursery.WALK, learn=False)
        first_silence = run.phase(probe_seconds, tone=False, walk=nursery.WALK, learn=False)
        run.phase(second_seconds, tone=True, walk=nursery.WALK, hand=True, learn=True)
        second_tone = run.phase(probe_seconds, tone=True, walk=nursery.WALK, learn=False)
        second_silence = run.phase(probe_seconds, tone=False, walk=nursery.WALK, learn=False)
    finally:
        run.close()
    first = first_tone["retreat"] - first_silence["retreat"]
    second = second_tone["retreat"] - second_silence["retreat"]
    return dict(status="ok", error=None, first_gain=first, second_gain=second, added=second - first)


def second_lesson_bar(measures):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["first_gain"] < BAR["quick_lesson_gain"]:
        return False, ("the first short lesson did not take (%.3f)" % measures["first_gain"])
    if measures["added"] < BAR["second_lesson_gain"]:
        return False, ("the second lesson added nothing (%.3f then %.3f)"
                       % (measures["first_gain"], measures["second_gain"]))
    return True, ("the second lesson added %.3f (%.3f then %.3f)"
                  % (measures["added"], measures["first_gain"], measures["second_gain"]))




class _Life:
    """One animal kept alive across phases, with one ear per tone.

    The bank's other learning questions use ``nursery.Run``, which starts a
    fresh animal for every phase and so cannot carry a weight from one phase to
    the next.  A question about what a lesson looks like *later* needs the same
    animal throughout, so this keeps one body and one brain and walks it from
    phase to phase.  Nothing here writes a weight: the only thing that touches
    the local rule is the keeper's hand, driving dopamine through the innate
    touch route, and the tone, through the innate ear.
    """

    def __init__(self, ctx, tones):
        from tools import nursery
        self.nursery = nursery
        self.body, self.brain, self.eyes, _ = nursery.build(
            ctx["seed"], hand_teaches=True, gated=True, routes=True,
            parameters=ctx["parameters"])
        self.ears = {name: BinauralSenses(self.body, window_samples=160, emitters=(tone,))
                     for name, tone in tones.items()}
        self.silence = np.zeros((2, 160))
        self.observation = self.body.observe()
        self.environment = blank_environment()
        self.base = mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.low, self.upright = 1., 1.

    def phase(self, seconds, tone=None, hand=False, learn=True):
        ears = self.ears.get(tone)
        seen = {"retreat": [], "brake": [], "dopamine": [], "cochlea": []}
        period = self.nursery.PERIOD
        for step in range(int(round(seconds/DT))):
            held = hand and (step*DT % period) < period/2
            grip = self.nursery.HAND
            self.environment["body_touch"] = np.full(4, grip) if held else np.zeros(4)
            for name in ("foot_obstacle", "foot_load", "foot_slip"):
                self.environment[name] = np.zeros(4)
            target, activation = self.brain.step(
                self.observation, environment=self.environment,
                eye_pixels=self.eyes.observe_raw(),
                ear_waveform=ears.observe() if ears is not None else self.silence,
                dt=DT, learn=learn, locomotion=self.nursery.WALK)
            self.body.command_eyes(self.brain.eye_command())
            self.observation = self.body.step(target, duration=DT, activation=activation)
            if step % 10 == 0:
                rates = self.brain.network.activity
                for name in ("retreat", "brake", "dopamine"):
                    seen[name].append(float(rates[self.brain.groups[name][0]]))
                seen["cochlea"].append(float(rates[self.brain.groups["cochlea"]].max()))
            if step*DT > 1.:
                self.low = min(self.low, float(self.body.data.xpos[self.base][2]))
                self.upright = min(self.upright, float(
                    self.body.data.xmat[self.base].reshape(3, 3)[2, 2]))
        report = {name: (float(np.mean(values)) if values else 0.)
                  for name, values in seen.items()}
        report["min_height"], report["min_up_z"] = self.low, self.upright
        return report

    def route(self):
        """The teachable route's own weights: the memory itself, not its effect."""
        return [float(value) for value in self.brain.nursery_state()["weights"]]

    def close(self):
        self.eyes.close()


def long_life_measure(ctx, lesson_seconds, life_seconds, probe_seconds=4., block=6.):
    """Taught once, then a stretch of life with the local rule running, then asked again.

    The gap is not a quiet pause: the animal keeps walking and the tone comes
    and goes, so the rule that carries the lesson runs the whole time.  A quiet
    pause is already asked about by the sixty second question; what is new here
    is that it goes on living.
    """
    from tools import nursery
    life = _Life(ctx, {"a": dict(nursery.TONE)})
    try:
        life.phase(probe_seconds, tone="a")
        life.phase(lesson_seconds, tone="a", hand=True)
        first_tone = life.phase(probe_seconds, tone="a")
        first_silence = life.phase(probe_seconds)
        route_after_lesson = life.route()
        remaining, index = float(life_seconds), 0
        while remaining > 1e-9:
            take = min(block, remaining)
            remaining -= take
            index += 1
            life.phase(take, tone=("a" if index % 2 else None))
        later_tone = life.phase(probe_seconds, tone="a")
        later_silence = life.phase(probe_seconds)
        route_after_life = life.route()
    finally:
        life.close()
    first = first_tone["retreat"] - first_silence["retreat"]
    later = later_tone["retreat"] - later_silence["retreat"]
    return dict(status="ok", error=None, first_gain=first, later_gain=later,
                kept=(later/first if abs(first) > 1e-9 else 0.),
                life_seconds=float(life_seconds), min_up_z=later_tone["min_up_z"],
                route_after_lesson=route_after_lesson, route_after_life=route_after_life)


def long_life_bar(measures):
    """The lesson has to be there first, and it has to still be there afterwards."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["first_gain"] < BAR["taught_reflex_gain"]:
        return False, ("there was no lesson to keep (the tone alone drove %.3f over silence)"
                       % measures["first_gain"])
    if measures["kept"] < BAR["long_life_kept"]:
        return False, ("the lesson faded while it lived: %.3f right after the lesson, %.3f "
                       "after %.0f s of life (kept %.3f, bar %.3f)"
                       % (measures["first_gain"], measures["later_gain"],
                          measures["life_seconds"], measures["kept"], BAR["long_life_kept"]))
    return True, ("the tone alone drove retreat %.3f right after the lesson and %.3f after %.0f s "
                  "of life with the rule running (kept %.3f)"
                  % (measures["first_gain"], measures["later_gain"],
                     measures["life_seconds"], measures["kept"]))


TONE_SECOND = {"geom": "sound_high", "frequency": 880., "amplitude": 1.0}


def second_tone_measure(ctx, first_seconds=20., second_seconds=20., probe_seconds=4.):
    """Two tones taught one after the other through the same route.

    The second lesson arrives with the first one already written, and both have
    to live on the same route cells.  The question is whether the first one
    survives it, so the first tone is asked about again at the end.
    """
    from tools import nursery
    life = _Life(ctx, {"a": dict(nursery.TONE), "b": dict(TONE_SECOND)})
    try:
        life.phase(probe_seconds, tone="a")
        life.phase(first_seconds, tone="a", hand=True)
        a_tone = life.phase(probe_seconds, tone="a")
        a_silence = life.phase(probe_seconds)
        first_route = life.route()
        life.phase(second_seconds, tone="b", hand=True)
        b_tone = life.phase(probe_seconds, tone="b")
        b_silence = life.phase(probe_seconds)
        again_tone = life.phase(probe_seconds, tone="a")
        again_silence = life.phase(probe_seconds)
        both_route = life.route()
    finally:
        life.close()
    first = a_tone["retreat"] - a_silence["retreat"]
    second = b_tone["retreat"] - b_silence["retreat"]
    again = again_tone["retreat"] - again_silence["retreat"]
    return dict(status="ok", error=None, first_gain=first, second_gain=second, again_gain=again,
                first_kept=(again/first if abs(first) > 1e-9 else 0.),
                route_after_first=first_route, route_after_second=both_route)


def second_tone_bar(measures):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["first_gain"] < BAR["taught_reflex_gain"]:
        return False, ("the first lesson never took (%.3f on the tone)" % measures["first_gain"])
    if measures["second_gain"] < BAR["taught_reflex_gain"]:
        return False, ("the second tone was never learned (%.3f on the tone)"
                       % measures["second_gain"])
    if measures["first_kept"] < BAR["second_tone_kept"]:
        return False, ("the second lesson wiped out the first: %.3f before it, %.3f after "
                       "(kept %.3f, bar %.3f)"
                       % (measures["first_gain"], measures["again_gain"],
                          measures["first_kept"], BAR["second_tone_kept"]))
    return True, ("the first tone still drove retreat %.3f after the second was learned "
                  "(%.3f before it, kept %.3f)"
                  % (measures["again_gain"], measures["first_gain"], measures["first_kept"]))


def route_ceiling_measure(ctx, warmup_seconds=4., hold_seconds=20., stretches=3):
    """The same route held on to over and over: does it keep growing, or stop?

    The keeper holds on for three equal stretches with no pause between them, so
    the route is used exactly as hard as it can be each time.  Four readings of
    the route's own weights: before the hold, and at the end of each stretch.
    """
    from tools import nursery
    life = _Life(ctx, {"a": dict(nursery.TONE)})
    try:
        life.phase(warmup_seconds, tone="a")
        reads = [life.route()]
        for _ in range(stretches):
            life.phase(hold_seconds, tone="a", hand=True)
            reads.append(life.route())
    finally:
        life.close()
    sums = [float(np.asarray(part, dtype=float).sum()) for part in reads]
    rises = [sums[index + 1] - sums[index] for index in range(stretches)]
    return dict(status="ok", error=None, hold_seconds=float(hold_seconds),
                first_rise=rises[0], last_rise=rises[-1], rises=rises, sums=sums,
                ratio=(rises[-1]/rises[0] if abs(rises[0]) > 1e-9 else 0.),
                route_before=reads[0], route_after=reads[-1])


def route_ceiling_bar(measures):
    """It has to have grown at all, and the second stretch has to add far less."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    stretches = " then ".join("%.5f" % value for value in measures["rises"])
    if measures["first_rise"] <= BAR["route_ceiling_floor"]:
        return False, ("the route never moved, so there is nothing to ease off: it rose %.5f in "
                       "the first %.0f s (has to rise more than %.5f)"
                       % (measures["first_rise"], measures["hold_seconds"],
                          BAR["route_ceiling_floor"]))
    if measures["ratio"] > BAR["route_ceiling_slowdown"]:
        return False, ("the route kept growing just as fast: %s over equal stretches (the last "
                       "has to be at most %.2f of the first)"
                       % (stretches, BAR["route_ceiling_slowdown"]))
    return True, ("the route rose %s over equal %.0f s stretches, ending at %s"
                  % (stretches, measures["hold_seconds"],
                     [round(value, 4) for value in measures["route_after"]]))

def two_patterns_measure(ctx, rounds, paired_channel=2, spare_channel=1, gap=0):
    """Two patterns in the room at once: one is paired with a touch, one is not."""
    body = Go2Body(model_path=ctx["model_path"])
    brain = brain_for(ctx, body)
    observation = body.observe()
    shape = brain.eye_shape
    paired, _ = _pattern_images(shape, paired_channel)
    spare, _ = _pattern_images(shape, spare_channel)
    black = np.zeros(shape, dtype=np.uint8)
    before_paired = _memory_phase(brain, observation, paired, 0., 2., False)
    before_spare = _memory_phase(brain, observation, spare, 0., 2., False)
    for _ in range(rounds):
        _memory_phase(brain, observation, paired, .8, 2., True)
        _memory_phase(brain, observation, black, .0, 2., True)
    for _ in range(gap):
        _memory_phase(brain, observation, black, .0, 2., False)
    after_paired = _memory_phase(brain, observation, paired, 0., 2., False)
    after_spare = _memory_phase(brain, observation, spare, 0., 2., False)
    return dict(status="ok", error=None, rounds=rounds,
                paired_gain=after_paired["aversive_centre"] - before_paired["aversive_centre"],
                spare_gain=after_spare["aversive_centre"] - before_spare["aversive_centre"])


def two_pattern_bar(measures):
    """The paired pattern has to matter; the other one has to stay a picture."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["paired_gain"] < BAR["two_pattern_gain"]:
        return False, ("the paired pattern never took (%.4f, bar %.4f)"
                       % (measures["paired_gain"], BAR["two_pattern_gain"]))
    if measures["spare_gain"] >= BAR["two_pattern_gain"]:
        return False, ("the unpaired pattern picked up the meaning too (%.4f)"
                       % measures["spare_gain"])
    return True, ("paired %.4f, unpaired %.4f"
                  % (measures["paired_gain"], measures["spare_gain"]))


def shifted_pattern_measure(ctx, rounds, channel=2, shift=3, gap=0):
    """Paired with one picture, shown a slightly moved copy of it: still mean?"""
    body = Go2Body(model_path=ctx["model_path"])
    brain = brain_for(ctx, body)
    observation = body.observe()
    shape = brain.eye_shape
    original, _ = _pattern_images(shape, channel)
    moved = np.zeros_like(original)
    moved[:, :, :, channel] = np.roll(original[:, :, :, channel], shift, axis=0)
    black = np.zeros(shape, dtype=np.uint8)
    before_original = _memory_phase(brain, observation, original, 0., 2., False)
    before_moved = _memory_phase(brain, observation, moved, 0., 2., False)
    for _ in range(rounds):
        _memory_phase(brain, observation, original, .8, 2., True)
        _memory_phase(brain, observation, black, .0, 2., True)
    for _ in range(gap):
        _memory_phase(brain, observation, black, .0, 2., False)
    after_original = _memory_phase(brain, observation, original, 0., 2., False)
    after_moved = _memory_phase(brain, observation, moved, 0., 2., False)
    original_gain = after_original["aversive_centre"] - before_original["aversive_centre"]
    moved_gain = after_moved["aversive_centre"] - before_moved["aversive_centre"]
    return dict(status="ok", error=None, rounds=rounds, shift=shift,
                original_gain=original_gain, moved_gain=moved_gain,
                ratio=(None if original_gain <= 0. else float(moved_gain / original_gain)))


def shifted_pattern_bar(measures):
    """The pattern has to be recognised even when it is not painted in the same place."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["original_gain"] < BAR["visual_memory_gain"]:
        return False, ("the pattern never took in the first place (%.4f)"
                       % measures["original_gain"])
    if measures["ratio"] is None or measures["ratio"] < BAR["shifted_pattern_ratio"]:
        return False, ("a moved copy of the same picture meant nothing: %.4f against %.4f "
                       "for the original (bar %.2f of it)"
                       % (measures["moved_gain"], measures["original_gain"],
                          BAR["shifted_pattern_ratio"]))
    return True, ("the moved copy still meant %.4f of the original %.4f"
                  % (measures["moved_gain"], measures["original_gain"]))


def bright_pull_measure(ctx, seconds, distance=1.60):
    """The same walk with a bright thing in front of it and with nothing there."""
    ahead = walk_measure(ctx, seconds, "autonomous", "origin",
                         props=dict(SILENT, green_target=(distance, 0., .32)))
    away = walk_measure(ctx, seconds, "autonomous", "origin", props=SILENT)
    if ahead["status"] != "ok" or away["status"] != "ok":
        return dict(status="error", error=ahead["error"] or away["error"])
    ahead_speed = float(ahead["mean_speed_mps"] or 0.)
    away_speed = float(away["mean_speed_mps"] or 0.)
    return dict(status="ok", error=None, ahead=ahead, away=away,
                pull=ahead["progress_m"] - away["progress_m"],
                ahead_speed=ahead_speed, away_speed=away_speed,
                pace_ratio=(ahead_speed / away_speed if away_speed > 1e-6 else 0.))


def bright_pull_bar(measures):
    """A bright thing in front has to pull it further along than an empty room."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    ahead, away = measures["ahead"], measures["away"]
    if not ahead["behavior_pass"] or not away["behavior_pass"]:
        return False, ("it fell over in one of the two runs (min up_z %.3f and %.3f)"
                       % (ahead["min_up_z"], away["min_up_z"]))
    if away["progress_m"] < BAR["walk_own_m"]:
        return False, ("it does not walk with an empty room either (%.3f m)" % away["progress_m"])
    if measures["pull"] < BAR["bright_pull_m"]:
        return False, ("the bright thing did not pull it forward (%.3f m more than nothing, "
                       "bar %.3f)" % (measures["pull"], BAR["bright_pull_m"]))
    return True, ("a bright thing in front pulled it %.3f m further" % measures["pull"])


def straight_pull_measure(ctx, seconds=10., distance=1.60):
    """The same walk with a bright thing in front of it and with nothing there.

    The reading is how straight the walk was each time.  A walk with nothing to
    head for is free to curve - the room is only so big - so the question here
    is only whether *having something to head for* keeps it on a line.  The
    two runs share everything except the prop, so the difference between them
    is what the prop did and nothing else.
    """
    ahead = walk_measure(ctx, seconds, "autonomous", "origin",
                         props=dict(SILENT, green_target=(distance, 0., .32)))
    away = walk_measure(ctx, seconds, "autonomous", "origin", props=SILENT)
    if ahead["status"] != "ok" or away["status"] != "ok":
        return dict(status="error", error=ahead["error"] or away["error"])
    return dict(status="ok", error=None, ahead=ahead, away=away,
                straight_ahead=float(ahead["straightness"]),
                straight_away=float(away["straightness"]),
                straight_gain=float(ahead["straightness"]) - float(away["straightness"]),
                lateral_ahead=float(ahead["lateral_m"]),
                lateral_away=float(away["lateral_m"]))


def straight_pull_bar(measures):
    """Heading for the thing in front: straighter with it there than without."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    ahead, away = measures["ahead"], measures["away"]
    if not ahead["behavior_pass"] or not away["behavior_pass"]:
        return False, ("it fell over in one of the two runs (min up_z %.3f and %.3f)"
                       % (ahead["min_up_z"], away["min_up_z"]))
    if away["progress_m"] < BAR["walk_own_m"]:
        return False, ("it does not walk with an empty room either (%.3f m)"
                       % away["progress_m"])
    if measures["straight_gain"] < BAR["straight_gain"]:
        return False, ("the thing in front did not keep it any straighter "
                       "(%.4f against %.4f, bar +%.4f)"
                       % (measures["straight_ahead"], measures["straight_away"],
                          BAR["straight_gain"]))
    return True, ("with something to head for the walk straightened: %.4f against %.4f"
                  % (measures["straight_ahead"], measures["straight_away"]))


def tip_over_measure(ctx, roll=.45, seconds=2.):
    """Rolled over against level: what do its own posture cells say?

    It read the ``righting`` group before, which is the cell that *starts the
    righting action*, not one that senses attitude.  The animal is held in the
    tilted pose, so that cell never fires and the question could not be passed:
    its reading was 0.00000 on every seed.  The otolith cells are the ones that
    answer it, and they do: 0.247 when rolled, 0.000 when level.
    """
    cells = ("vestibular_exc",)
    tipped = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tilt=(roll, 0.),
                       tail=seconds)
    flat = cell_read(ctx, cells=cells, seconds=seconds, clean=True, tail=seconds)
    up = np.asarray(tipped["parts"][cells[0]], dtype=float)
    down = np.asarray(flat["parts"][cells[0]], dtype=float)
    return dict(status="ok", error=None, roll=roll, tipped=up.tolist(), level=down.tolist(),
                gain_mean=float(np.mean(up) - np.mean(down)),
                gain_max=float(np.max(up - down)))



task("stands_still_for_twenty_seconds", "站立", "没人叫它走，站二十秒会不会自己倒下去", "full", 20.,
     lambda ctx: stand_still_measure(ctx, 20.), stand_long_bar,
     "二十秒里它可能还是会一点点往前蹭。这一题只拦「走起来」和「倒下去」，"
     "不说明它站得像雕像。",
     requires=("stands_still",))

task("stands_still_on_a_slope", "站立", "站在斜坡上也站得住吗", "full", 8.,
     lambda ctx: walk_measure(ctx, 8., "rest", "ramp_top"), slope_stand_bar,
     "坡是场地里那一道，不是随便什么坡度；它不说明它知道自己在坡上。",
     requires=("stands_still",))

task("stays_up_when_nudged_gently", "站立", "被轻轻撞一下，会不会自己站回来", "full", 7.,
     lambda ctx: righting_measure(ctx, "impact", 7., True, 90.), righting_bar,
     "和大力推那一题是同一条判据，只是推力减半。它测的是小干扰下能不能站稳。",
     requires=("stands_still",))

task("all_four_feet_leave_the_ground", "走路", "走路时四只脚都会抬起来吗", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"), foot_lift_bar,
     "读的是每只脚离地超过两毫米的时间占比。它只说明四条腿都在参与，"
     "不说明步态好看。",
     requires=("walk_flat",))

task("the_two_sides_step_alike", "走路", "左右两条腿是一样干活，还是一条拖着一条", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"), leg_symmetry_bar,
     "比的是同一条带子上两只脚的抬脚占比。它不说明两条腿的力气一样大。",
     requires=("walk_flat",))

task("keeps_walking_for_thirty_seconds", "走路", "没人管它，三十秒里是一直走还是走走停停", "full", 30.,
     lambda ctx: walk_measure(ctx, 30., "autonomous", "origin"),
     lambda m: walk_bar(m, "walk_thirty_m"),
     "只比距离，不看它是怎么走的。三十秒里摔过再爬起来的也算不过。",
     requires=("walks_for_twenty_seconds",))

task("does_not_slow_down_on_a_long_walk", "走快", "走久了会不会越走越慢", "full", 20.,
     lambda ctx: step_windows(ctx, 20., "autonomous", "origin", 5.), fatigue_bar,
     "比的是头五秒和末五秒的平均速度。它不区分是累了，还是换了种走法。",
     requires=("walks_at_a_steady_pace",))

task("keeps_walking_while_it_turns", "转弯", "一边走一边有声音，它会不会朝着声音转过去、还继续走", "full", 16.,
     lambda ctx: turn_pair_measure(ctx, 16., "autonomous"),
     lambda m: walk_turn_bar(m, "一边走一边响"),
     "两次都要走得动，而且两次的收尾朝向要分开。走得好但没转，或者转了但不走，"
     "都算不过。",
     requires=("turns_towards_a_sound_while_walking",))

task("backs_away_from_a_hand_on_its_chest", "转弯", "胸口一直被顶着，会不会往后退", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin", injected={"body_touch": .8}),
     lambda m: back_away_bar(m, "胸口上那只手"),
     "读的是它有没有真的往后退。它不说明它想退到哪去。",
     requires=("a_touch_on_the_front_holds_it_back",))

task("keeps_its_line_with_a_sound_behind", "转弯", "背后有声音，会不会被它带得偏方向", "full", 12.,
     lambda ctx: walk_measure(ctx, 12., "autonomous", "origin",
                              props=dict(SILENT, sound_low=(-1.40, 0., .30))),
     behind_hold_bar,
     "这一题要的是「不该转」：背后的声音不该让它偏航，也不该让它停下来。",
     requires=("walks_without_being_told",))

task("does_not_fall_walking_into_the_wall", "地形", "朝墙上走，是停下来还是被撞翻", "full", 12.,
     lambda ctx: walk_measure(ctx, 12., "autonomous", "wall"), wall_upright_bar,
     "和「撞墙停下」那一题不同，这一题只看它有没有被自己撞翻。",
     requires=("stops_at_the_wall",))

task("threads_the_passage_without_touching", "地形", "过窄通道时会不会蹭着墙走", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "passage"), passage_quiet_bar,
     "过了通道还要没把墙细胞点亮。它不说明它看见了墙。",
     requires=("walk_narrow_passage",))
task("sees_a_red_square_from_a_green_one", "看", "同一个方块，红色和绿色，脑里读数一样吗", "full", 4.,
     lambda ctx: colour_measure(ctx),
     lambda m: separation_bar(m, "retinal_opponent", "colour_gain", "红方块和绿方块"),
     "同一个形状、同一个位置，只有颜色不同。它测的是感光层下面那层对颜色有没有分开，"
     "不说明它知道那叫红色。",
     requires=("eyes_follow_ball",))

task("sees_a_big_ball_from_a_small_one", "看", "同一个位置，大球和小球，读数一样吗", "full", 5.,
     lambda ctx: size_measure(ctx),
     lambda m: separation_bar(m, "retinal_change", "size_gain", "大球和小球"),
     "球的位置不动，只换大小。它只说明「占的画面大小」这一路有读数，"
     "不说明它量出了远近。",
     requires=("eyes_follow_ball",))

task("eyes_follow_the_ball_up_and_down", "看", "球上下移动，眼睛会不会跟着上下看", "full", 5.,
     lambda ctx: gaze_pitch_measure(ctx), lambda m: gaze_pitch_bar(m, "pitch_gaze_rad"),
     "上下这一路本来就比左右弱得多，门槛是按出厂读数留的；"
     "它不说明眼睛的俯仰已经好用。",
     requires=("eyes_follow_ball",))

task("eyes_follow_the_ball_far_away", "看", "球在两米外扫过，还跟得住吗", "full", 5.,
     lambda ctx: gaze_measure(ctx, .30, 2.5, 2.0, scenery=False),
     lambda m: gaze_bar(m, "far_gaze_rad"),
     "和追球那一题是同一条线，球放到两米外。远的比近的难过，因为每帧动的角度小。",
     requires=("eyes_follow_ball",))

task("eyes_follow_the_ball_close_up", "看", "球贴着鼻子扫过，还跟得住吗", "full", 5.,
     lambda ctx: gaze_measure(ctx, .30, 2.5, .40, scenery=False),
     lambda m: gaze_bar(m, "near_gaze_rad"),
     "球在四十厘米处扫过，角度变化快得多。它测的是跟得上快变化，"
     "不说明它把球看得清楚。",
     requires=("eyes_follow_ball",))

task("the_eyes_let_go_when_it_is_gone", "看", "东西走了，眼睛会不会自己转回来", "full", 3.,
     lambda ctx: recentre_measure(ctx), recentre_bar,
     "两段都要读：前面要真的看过去，后面要真的松开。一只从头到尾没动过的眼睛"
     "也算「没留在空地方」，所以前半段是必须的。",
     requires=("eyes_look_at_a_still_thing_at_the_side",))

task("notices_a_thing_that_creeps_in", "看", "一样东西慢慢冒出来，眼睛发不发现", "full", 4.,
     lambda ctx: sudden_measure(ctx, 2., 2., .50, .10),
     lambda m: sudden_bar(m, "slow_new_thing_rad"),
     "比冒出来那一题动得更慢。眼睛这一路靠「画面变了」，动得越慢越难发现。",
     requires=("eyes_on_a_new_thing",))

task("a_sound_that_stops_is_no_longer_heard", "听", "声音停了，耳朵里是不是也停了", "full", 4.,
     lambda ctx: sound_gap_measure(ctx, 4.), lambda m: sound_gap_bar(m, "那一声"),
     "同一个声源，先响两秒再挪走。它测的是耳朵跟着世界走，不是耳朵在自说自话。",
     requires=("turn_to_sound",))

task("a_quiet_room_is_not_a_sound", "听", "屋里没声音时：耳蜗安静吗、说不出话的那格亮不亮", "full", 2.,
     lambda ctx: cell_read(ctx, cells=("cochlea", "abrupt_sound", "cochlea_opposite"),
                           seconds=2., tail=2., clean=True, props=SILENT),
     lambda m: opposite_bar(m, (("cochlea", "cochlea_opposite"),), "安静的屋子"),
     "这是「声音停了」那一题的对照：什么都听不见的时候，耳蜗该是安静的，而它旁边那格"
     "「什么都没听见」的细胞该是亮的。它只说明这一对还在，不说明它听得出「安静」。",
     requires=("turn_to_sound",))

task("ears_tell_left_from_right", "听", "声音在左肩和右肩，耳朵分得开吗", "full", 4.,
     lambda ctx: sound_side_measure(ctx, 2.),
     lambda m: separation_bar(m, "auditory_pinna", "side_gain", "左肩和右肩的声音"),
     "和前后那一题是同一条线，换成正左右。它只说明两边分得开。",
     requires=("turn_to_sound",))

task("two_sounds_at_one_shoulder_are_still_two", "听", "同一个方向两个声音一起响，耳朵会不会糊成一团", "full", 4.,
     lambda ctx: same_side_pair_measure(ctx, 2.),
     lambda m: separation_bar(m, "auditory_pinna", "same_side_gain", "同一边两个声音"),
     "两个声源都在同一边，只比花样不比总量。它不说明它分得清两个声音各自是什么。",
     requires=("two_sounds_at_once_are_still_two",))

task("a_bang_behind_it_still_startles", "听", "背后突然来一声，会不会受惊", "full", 4.,
     lambda ctx: startle_behind_measure(ctx, 2.),
     lambda m: contrast_bar(m, "startle", "startle_behind_gain", "背后那一声"),
     "比的是同一间屋子有没有那一声。它只说明受惊这条路不分方向地接着。",
     requires=("a_bang_makes_it_startle",))
task("feels_a_touch_on_the_right_side", "摸与本体", "右边被碰，脑里会往哪边让", "full", 4.,
     lambda ctx: side_touch_measure(ctx, (3, 1), ("avoidance",), (1, 0)),
     lambda m: both_sides_bar(m, "left_touch_gain", "right_touch_gain", "side_touch_gain",
                              "右边和左边被碰"),
     "左边那一题的镜像，四次把四个方向凑齐。它只说明没有接反。",
     requires=("feels_a_touch_on_the_left_side",))

task("feels_a_light_hand_too", "摸与本体", "轻轻搭一只手，脑里还知道在哪边吗", "full", 4.,
     lambda ctx: light_touch_measure(ctx, 2., .30),
     lambda m: both_sides_bar(m, "left_touch_gain", "right_touch_gain", "light_touch_gain",
                              "轻轻搭上来"),
     "和握手那一题是同一条线，力度降到不到一半。它测的是轻的也没被漏掉。",
     requires=("feels_a_touch_on_the_left_side",))

task("feels_the_ground_under_the_front_legs", "摸与本体", "前腿踩实了没有", "full", 4.,
     lambda ctx: one_foot_measure(ctx, "foot_load", 0, ("impact_adaptation",), .5, .9, True,
                                  tail=.2),
     lambda m: both_sides_bar(m, "gain", "swapped_gain", "front_load_gain", "承重的前腿"),
     "和后腿那条承重的题是同一条线。四条腿各接一遍才算接全。",
     requires=("feels_the_ground_under_the_back_legs",))

task("lifts_each_of_the_four_feet", "摸与本体", "四条腿各绊一次，抬的是不是那条", "full", 8.,
     lambda ctx: each_foot_measure(ctx),
     lambda m: all_feet_bar(m, "all_feet_lift_gain", "四条腿"),
     "一腿一次共四次，每次都要是那条腿自己的细胞最亮。"
     "它比两方向那一题严，因为剩下的两条腿也一起被查了。",
     requires=("lifts_the_foot_that_was_caught",))

task("the_flat_ground_is_not_a_bump", "摸与本体", "平地上站着，会不会觉得自己踩着东西", "full", 2.,
     lambda ctx: cell_read(ctx, cells=("clearance", "stumble"), seconds=2., tail=2., clean=True),
     lambda m: at_most_cells_bar(m, "clearance", "ground_quiet_gain", "平地"),
     "和「脚被绊」那一题配对的对照：平地上不该有绊住的读数。",
     requires=("feels_a_touch_on_the_front",))

task("it_forgets_when_nobody_teaches_anymore", "学习", "没人接着教，学会的东西会不会自己淡掉", "full", 62.,
     lambda ctx: extinction_measure(ctx, 20., 30.), extinction_bar,
     "先教会，再让那个声音自己响三十秒、没人摸它。它测的是「没人撑着的记忆会退」，"
     "不说明退回原样。",
     requires=("the_lesson_survives_a_longer_pause",))

task("a_second_short_lesson_adds_to_the_first", "学习", "再教一遍，第二遍会不会更牢", "full", 32.,
     lambda ctx: second_lesson_measure(ctx, 5., 5.), second_lesson_bar,
     "两次各五秒，中间隔着一次考试。它测的是重复有没有用，"
     "不说明睡一觉还在。",
     requires=("learns_the_lesson_in_five_seconds",))

task("the_lesson_is_still_there_after_a_minute", "学习", "教完安静一分钟，还记得吗", "full", 92.,
     lambda ctx: taught_reflex_measure(ctx, 20., 60.),
     lambda m: taught_reflex_bar(m, "taught_reflex_kept"),
     "安静的时间比「隔三十秒」那一题再长一倍。它仍然只到一分钟量级，"
     "不说明长期记忆。",
     requires=("the_lesson_survives_a_longer_pause",))


task("the_memory_survives_a_long_life", "学习", "教完让它自己活一分钟，那件事还在吗", "full", 98.,
     lambda ctx: long_life_measure(ctx, 20., 60.),
     long_life_bar,
     "它活的那一分钟不是安静的：它一直在走，那个声音时有时无，局部规则一直开着。"
     "这一题只读后退细胞，所以它说明的是那件事还在不在，不说明它记得多久、"
     "也不说明它分得清别的音。",
     requires=("the_lesson_survives_a_longer_pause",))

task("a_second_tone_does_not_wipe_the_first", "学习", "学会一个音之后再教一个，前一个会不会被顶掉", "full", 76.,
     lambda ctx: second_tone_measure(ctx, 20., 20.),
     second_tone_bar,
     "两个音走的是同一条可教通路，所以它们抢同一批细胞。这一题只问前一个还在不在，"
     "不说明它分得清这两个音。",
     requires=("the_lesson_survives_a_longer_pause",))

task("the_route_stops_growing_at_its_ceiling", "学习", "同一根连接一直用，会不会无脑一直涨", "full", 64.,
     lambda ctx: route_ceiling_measure(ctx),
     route_ceiling_bar,
     "这一题读的是那几根可教连接自己的权重，不是细胞的读数。它只说明连着用一分钟之后它越涨越慢、最后几乎不涨，"
     "不说明它永远不会再变——真正把它钉住的是它自己的上限和系绳。",
     requires=("taught_reflex_sticks",))

task("remembers_two_patterns_at_once", "记忆", "两块图案里只配对一块，另一块会不会也被当成危险", "full", 30.,
     lambda ctx: two_patterns_measure(ctx, 5), two_pattern_bar,
     "配对的那块要有意义，没配对的那块要没有。它测的是记忆有没有摊到别的图案上，"
     "不说明它认得「这是哪一块」。",
     requires=("visual_memory_holds",))

task("the_memory_survives_a_small_change", "记忆", "图案挪了一点点，还认得出来吗", "full", 30.,
     lambda ctx: shifted_pattern_measure(ctx, 5), shifted_pattern_bar,
     "配对时用的是原来那块，考试时用的是同一块往上挪了几行的版本。"
     "这和「同一个人慢慢变老还认得」是同一类问题，只是这里只挪一点点。",
     requires=("visual_memory_holds",))

task("walks_towards_a_bright_thing", "注意", "前面有个亮东西，会不会朝着它多走几步", "full", 20.,
     lambda ctx: bright_pull_measure(ctx, 10.), bright_pull_bar,
     "同样两个场景，只有一个亮东西在不在前面的区别。它测的是「亮的东西会不会"
     "把它多带走一点」，不说明它奔着什么去。",
     requires=("walk_flat",))

task("knows_when_it_is_tipped_over", "自我", "身体被扳歪了，它自己的姿态细胞知不知道", "full", 4.,
     lambda ctx: tip_over_measure(ctx, .45, 2.),
     lambda m: reading_bar(m, "gain_mean", "tip_over_gain"),
     "读的是它自己那组「我正不正」的细胞，在被扳歪和站正两个状态下的差。"
     "它只说明姿态这件事在脑里有读数。",
     requires=("stands_still",))

task("the_hidden_layer_sees_the_picture", "特征层", "两幅不一样的画，眼睛下面第一层读数分得开吗", "full", 4.,
     lambda ctx: feature_layer_measure(ctx),
     lambda m: separation_bar(m, "retinal_interneurons", "feature_gain", "横条纹和竖条纹"),
     "两幅画都有花纹，只是一个横一个竖。它测的是眼睛下面那层有没有把画面带下去，"
     "不说明哪一层「懂」了画面。",
     requires=("eyes_follow_ball",))

task("the_compressed_layer_sees_the_picture", "特征层", "再往下压一层，两幅画还分得开吗", "full", 4.,
     lambda ctx: feature_layer_measure(ctx),
     lambda m: retention_bar(m, "retina", "retinal_interneurons", .05, "横条纹和竖条纹"),
     "和上一层同一幅画、同一次对比。压缩得越狠越容易把两幅画压成同一个读数，"
     "这一题就是在问有没有压坏。",
     requires=("the_hidden_layer_sees_the_picture",))

# The rungs of the existing questions.  A rung is open when every rung under it
# was passed, so an animal that cannot stand is never sent up a ramp; the
# questions new in the second batch carry their own ``requires`` above.
REQUIRES = {
    "walk_flat": ("stands_still",),
    "walk_furnished": ("walk_flat",),
    "walks_without_being_told": ("walk_flat",),
    "keeps_walking_without_being_told": ("walks_without_being_told",),
    "feet_do_not_slide": ("walk_flat",),
    "walk_low_step": ("walk_flat", "walks_without_being_told"),
    "walk_ramp": ("walk_low_step",),
    "walk_narrow_passage": ("walk_flat",),
    "stops_at_the_wall": ("walks_without_being_told",),
    "get_up_from_side": ("get_up_from_back",),
    "nose_up_recover": ("get_up_from_back",),
    "nose_down_recover": ("get_up_from_back",),
    "stay_up_when_pushed": ("stands_still",),
    "turn_to_sound": ("stands_still",),
    "eyes_follow_ball": (),
    "eyes_follow_fast_ball": ("eyes_follow_ball",),
    "eyes_follow_the_ball_both_ways": ("eyes_follow_ball",),
    "eyes_on_a_new_thing": ("eyes_follow_ball",),
    "eyes_hold_still_on_a_still_thing": ("eyes_follow_ball",),
    "eyes_tell_near_from_far": ("eyes_converge_on_a_near_thing",),
    "eyes_see_a_pattern_not_a_blank": ("eyes_follow_ball",),
    "ears_tell_front_from_back": ("turn_to_sound",),
    "ears_tell_loud_from_far": ("turn_to_sound",),
    "ears_tell_high_from_low": ("turn_to_sound",),
}
for _entry in TASKS:
    if _entry["name"] in REQUIRES:
        _entry["requires"] = tuple(REQUIRES[_entry["name"]])


def bright_pace_bar(measures, what, bar_key="bright_pace_ratio"):
    """A bright thing in front has to buy a faster walk, not just a longer one.

    Both runs have to stay up, and the empty room has to be a real walk already:
    a ratio of two stalls is not speed.
    """
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    for name in ("ahead", "away"):
        if not measures[name]["behavior_pass"]:
            return False, ("%s: it fell over (min up_z %.3f)"
                           % (what, measures[name]["min_up_z"]))
    if measures["away_speed"] < BAR["walk_speed_mps"]:
        return False, ("%s: with an empty room it does not even hold a steady pace "
                       "(%.3f m/s, bar %.3f)"
                       % (what, measures["away_speed"], BAR["walk_speed_mps"]))
    if measures["pace_ratio"] < BAR[bar_key]:
        return False, ("%s changed the speed by %.2f times (%.3f m/s with the thing, "
                       "%.3f without, bar %.2f)"
                       % (what, measures["pace_ratio"], measures["ahead_speed"],
                          measures["away_speed"], BAR[bar_key]))
    return True, ("%s: %.3f m/s with the bright thing and %.3f m/s without, %.2f times"
                  % (what, measures["ahead_speed"], measures["away_speed"],
                     measures["pace_ratio"]))


def run_bar(measures, bar_key, what):
    """Upright the whole way, and really going somewhere."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if not measures["behavior_pass"]:
        return False, ("%s: not upright the whole way (min up_z %.3f)"
                       % (what, measures["min_up_z"]))
    speed = float(measures["mean_speed_mps"] or 0.)
    if speed < BAR[bar_key]:
        return False, "%s averaged %.3f m/s, bar %.3f" % (what, speed, BAR[bar_key])
    return True, "%s: %.3f m/s on average, upright throughout" % (what, speed)


def run_turn_bar(measures, what):
    """Turning towards a sound while still moving at a run."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    for run in (measures["right"], measures["left"]):
        if run["status"] != "ok":
            return False, "run failed: %s" % run["error"]
        if not run["behavior_pass"]:
            return False, "it fell over (min up_z %.3f)" % run["min_up_z"]
        if run["progress_m"] < BAR["turn_walk_m"]:
            return False, ("%s: it stopped going anywhere (%.3f m, bar %.3f)"
                           % (what, run["progress_m"], BAR["turn_walk_m"]))
        speed = float(run["mean_speed_mps"] or 0.)
        if speed < BAR["run_turn_speed"]:
            return False, ("%s: it turned but crawled (%.3f m/s, bar %.3f)"
                           % (what, speed, BAR["run_turn_speed"]))
    if measures["difference"] > -BAR["turn_walk_rad"]:
        return False, ("%s did not turn it (%.3f rad apart, bar %.3f)"
                       % (what, measures["difference"], BAR["turn_walk_rad"]))
    return True, ("%s: turned it %.3f rad at %.3f and %.3f m/s"
                  % (what, measures["difference"],
                     measures["right"]["mean_speed_mps"],
                     measures["left"]["mean_speed_mps"]))


def dark_room_measure(ctx, seconds=.35, tail=.35):
    """A black screen and a white one: is the input layer ever empty?

    The animal's input layer is two halves of the same picture: a pixel hands
    its brightness to one cell and its darkness to the cell beside it.  A black
    screen has to light one half and leave the other alone; a white one has to
    swap them over.  Both halves moving together would be one number in two
    places, not a second channel.  Each half also has summary cells of its own,
    so the question is asked of the layer that reads the input as well.
    """
    black = np.zeros((2, 36, 48, 3), dtype=np.uint8)
    white = np.full((2, 36, 48, 3), 255, dtype=np.uint8)
    cells = ("photoreceptors", "photoreceptors_opposite", "retina", "retina_opposite")
    dark = cell_read(ctx, cells=cells, seconds=seconds, tail=tail, clean=True, picture=black)
    light = cell_read(ctx, cells=cells, seconds=seconds, tail=tail, clean=True, picture=white)
    if dark["status"] != "ok" or light["status"] != "ok":
        return dict(status="error", error=dark["error"] or light["error"])
    return dict(status="ok", error=None, black=dark, white=light,
                black_opposite=float(dark["photoreceptors_opposite"]),
                black_light=float(dark["photoreceptors"]),
                black_summary_opposite=float(dark["retina_opposite"]),
                white_opposite=float(light["photoreceptors_opposite"]),
                white_light=float(light["photoreceptors"]),
                white_summary_light=float(light["retina"]))


def dark_room_bar(measures, what, bar_key="opposite_reading"):
    """Each half of the input layer has to be lit by one kind of screen."""
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    if measures["black_opposite"] < BAR[bar_key]:
        return False, ("%s: a black screen left the dark half at %.5f, bar %.5f"
                       % (what, measures["black_opposite"], BAR[bar_key]))
    if measures["black_light"] > BAR[bar_key]:
        return False, ("%s: a black screen lit the bright half anyway (%.5f, bar %.5f)"
                       % (what, measures["black_light"], BAR[bar_key]))
    if measures["white_opposite"] > BAR[bar_key]:
        return False, ("%s: a white screen lit the dark half as well (%.5f, bar %.5f)"
                       % (what, measures["white_opposite"], BAR[bar_key]))
    if measures["black_summary_opposite"] < BAR[bar_key]:
        return False, ("%s: the dark half reached its summary cells at only %.5f, "
                       "bar %.5f"
                       % (what, measures["black_summary_opposite"], BAR[bar_key]))
    return True, ("%s: black lit the dark half %.5f (summary %.5f) and left the "
                  "bright half at %.5f"
                  % (what, measures["black_opposite"], measures["black_summary_opposite"],
                     measures["black_light"]))


task("hurries_towards_a_bright_thing", "跑", "前面放个亮东西，它会不会走得比空房间快", "full", 20.,
     lambda ctx: bright_pull_measure(ctx, 10.),
     lambda m: bright_pace_bar(m, "前面有亮东西时"),
     "同样两个场景，只有前面有没有那个亮东西的区别。读的是"
     "平均速度之比，不是走了多远。出厂动物有一条「"
     "亮东西让它多走几步」的弱通路，但两次一样快，"
     "所以它出生时不过。它不说明它奔着亮东西去。",
     requires=("walks_at_a_steady_pace",))

task("runs_without_being_told", "跑", "没人管它，它自己跑得起来吗", "full", 10.,
     lambda ctx: walk_measure(ctx, 10., "autonomous", "origin"),
     lambda m: run_bar(m, "run_mean_speed", "没人管它的那十秒"),
     "没有外部指令可用，读的是它自己十秒里"
     "的全程平均速度，不是某一瞬间的快。跑不动、"
     "或者跑两步就摔，都算不过。",
     requires=("hurries_towards_a_bright_thing",))

task("runs_for_twenty_seconds", "跑", "让它自己跑二十秒，它跑得下去吗", "full", 20.,
     lambda ctx: walk_measure(ctx, 20., "autonomous", "origin"),
     lambda m: walk_bar(m, "run_twenty_m"),
     "只比二十秒里走出去多远，不看步态好不好看；"
     "这二十秒里摔过再爬起来的也算不过。",
     requires=("runs_without_being_told",))

task("keeps_running_while_it_turns", "跑", "一边跑一边听到声音，还跑得动吗", "full", 16.,
     lambda ctx: turn_pair_measure(ctx, 16., "autonomous"),
     lambda m: run_turn_bar(m, "一边跑一边响"),
     "它在走的状态下早就转得动了，这一题额外要求"
     "转向的过程中速度不掉下来。",
     requires=("runs_without_being_told", "turns_towards_a_sound_while_walking"))

task("nothing_touching_it_is_not_nothing", "摸与本体", "没人碰它也没绊到它时，说不出话的那几格亮不亮", "full", 3.,
     lambda ctx: cell_read(ctx, cells=("body_touch", "body_touch_opposite",
                                       "foot_obstacle", "foot_obstacle_opposite",
                                       "foot_slip", "foot_slip_opposite"),
                           seconds=3., tail=2., clean=True, props=SILENT),
     lambda m: opposite_bar(m, (("body_touch", "body_touch_opposite"),
                                ("foot_obstacle", "foot_obstacle_opposite"),
                                ("foot_slip", "foot_slip_opposite")), "没被碰的站姿"),
     "站着、没人碰、四只脚都在地上：碰、绊、打滑三路都该安静，它们旁边那格该亮着。"
     "它只说明「什么都没发生」是一条真的读数，不说明它知道自己在站着。",
     requires=("stands_still",))

task("the_weight_pair_still_carries_the_weight", "摸与本体", "脚上受力那一对，合起来是不是一整档", "full", 3.,
     lambda ctx: cell_read(ctx, cells=("foot_load", "foot_load_opposite"),
                           seconds=3., tail=2., clean=True, props=SILENT),
     lambda m: full_pair_bar(m, "foot_load", "四只脚的受力那一对"),
     "受力那一路和它旁边那格合起来才是完整的一档：一格说「这只脚吃了多少力」，"
     "另一格说「还差多少」。它不说明它分得清是哪只脚。",
     requires=("stands_still",))

task("the_dark_room_is_not_silence", "看", "全黑的屋里，输入层暗的那一半亮着吗", "full", 2.,
     lambda ctx: dark_room_measure(ctx),
     lambda m: dark_room_bar(m, "全黑画面"),
     "输入层现在是成对的：每个像素把亮度交给一格、把「没光进来」交给旁边那格，"
     "没光的那一半还有自己的汇总层。这一题问的只是这两半在不在、黑的时候是不是反过来的；"
     "它不说明它意识到自己在黑屋里——这两半现在没有出边，黑画面到不了反射层。",
     requires=("eyes_follow_ball",))

# ---------------------------------------------------------------------------
# the widest layer: what the association region does with a body state
# ---------------------------------------------------------------------------
# the widest layer: what the association region does with a body state
# ---------------------------------------------------------------------------
def association_loop_measure(ctx, warmup=2., driven_seconds=2., hold_seconds=3., roll=.45,
                             tail=.5):
    """Hand the widest layer a body state, take it away, and watch what is left.

    The association layer is the one region that is neither a sense nor a
    muscle.  It is fed by a wide random fan-in from the body and it is wired to
    itself, so it runs at a level of its own; the questions here are whether a
    body state moves that level at all (``driven``) and whether anything of it
    survives after the body is back to standing (``held``).  A layer that only
    echoes its own echo, with nothing of the body left in it, cannot be a
    memory of anything.

    The robot stands still throughout - what changes between the phases is the
    body state handed to the brain, not the body - so this measures the layer
    and not the walking.  It is also why the scene is cheap: no physics beyond
    the placement, and 7 s of the controller.
    """
    body = clean_body(ctx["model_path"])
    _place(body, pose="stand")
    standing = body.observe()
    _place(body, pose="stand", tilt=(roll, 0.))
    tilted = body.observe()
    _place(body, pose="stand")
    brain = brain_for(ctx, body)
    groups = ("association", "proprioception", "vestibular_exc", "motor")
    environment = blank_environment()

    def trace(observation, seconds):
        history = {name: [] for name in groups}
        for _ in range(int(round(seconds / DT))):
            brain.step(observation, environment=environment, dt=DT, learn=False)
            for name in groups:
                history[name].append(float(np.mean(brain.network.activity[
                    np.atleast_1d(brain.groups[name])])))
        return {name: np.asarray(values) for name, values in history.items()}

    window = max(1, int(round(tail / DT)))
    warm = trace(standing, warmup)
    baseline = float(warm["association"][-window:].mean())
    hot = trace(tilted, driven_seconds)
    back = trace(standing, hold_seconds)
    driven = float(hot["association"].mean()) - baseline
    held = float(back["association"][-window:].mean()) - baseline
    return dict(status="ok", error=None, roll=roll,
                baseline=baseline,
                driven=driven,
                held=held,
                motor_level=float(warm["motor"][-window:].mean()),
                hold_start=float(back["association"][:window].mean()) - baseline,
                vestibular_driven=float(hot["vestibular_exc"].mean()),
                motor_driven=float(hot["motor"].mean())
                - float(warm["motor"][-window:].mean()),
                motor_held=float(back["motor"][-window:].mean())
                - float(warm["motor"][-window:].mean()))


def association_bar(measures, key, bar_key, what):
    if measures.get("status") != "ok":
        return False, "run failed: %s" % measures.get("error")
    value, bar = float(measures[key]), BAR[bar_key]
    if value < bar:
        return False, ("%s: %.5f (bar %.5f)" % (what, value, bar))
    return True, "%s: %.5f" % (what, value)


def return_line_measure(ctx, **kwargs):
    """Does the widest layer reach the muscles at all?

    The same seven seconds of standing as the other association questions, run
    twice: once as the animal is wired, and once with the return line cut
    (``association_return_gain`` set to zero).  Both runs are handed the same
    body state, so the body is not what is being compared - whatever the motor
    cells do differently is what came down the return line.  Cutting the line
    changes only the outgoing edges of that layer, so its own level is the same
    in both runs; ``driven`` and ``held`` are reported to show that.

    The reading is the motor level *of the same moment in both runs*, not the
    tilt-minus-standing difference.  The association layer is busy in both
    phases (0.383 standing against 0.434 tilted on the birth animal), so a
    return line lifts the motor level in both and the difference cancels it: on
    the birth wiring with the return line at gain 1 that reading moved by
    -3e-5, which is nothing.  Compared within one phase the same line moves the
    motor level by +1.6e-4 at gain 1, +6.6e-4 at 4 and +2.7e-3 at 16 - small,
    but it is the return line and only the return line.
    """
    as_given = association_loop_measure(ctx, **kwargs)
    parameters = dict(ctx["parameters"])
    parameters["association_return_gain"] = 0.
    cut = association_loop_measure(dict(ctx, parameters=parameters), **kwargs)
    if as_given.get("status") != "ok" or cut.get("status") != "ok":
        return dict(status="error",
                    error=as_given.get("error") or cut.get("error"))
    return dict(status="ok", error=None, wired=as_given, cut=cut,
                reach=float(as_given["motor_level"]) - float(cut["motor_level"]),
                reach_held=float(as_given["motor_held"]) - float(cut["motor_held"]),
                driven=float(as_given["driven"]), held=float(as_given["held"]),
                driven_when_cut=float(cut["driven"]))


task("the_association_layer_reads_the_body", "前额叶",
     "身体被扳歪时，那一大片既不收感觉也不管肌肉的细胞会不会跟着变", "full", 7.,
     lambda ctx: association_loop_measure(ctx),
     lambda m: association_bar(m, "driven", "association_driven_gain",
                               "身体被扳歪时那一层的读数变化"),
     "它只说这一层能从身体状态里分出两样东西。它不说明它能拿这个变化去做任何事："
     "那是另一道题（the_widest_layer_reaches_the_muscles），它读的是返回线；"
     "这一题读的是这一层自己的水平。",
     reads="被扳歪那两秒里那一层的平均读数，减去站直时的读数",
     bar_text="被扳歪时那一层的读数至少要动 0.02",
     requires=())

task("the_association_loop_holds_what_it_was_given", "前额叶",
     "身体扳歪再放直以后，那一层里还留着刚才那件事吗", "full", 7.,
     lambda ctx: association_loop_measure(ctx),
     lambda m: association_bar(m, "held", "association_hold_gain",
                               "放直三秒后那一层还比平时高多少"),
     "回响回路存在不等于记得住：读数回到原样，就说明这一层只是把它听到的东西立刻"
     "还了回去。出厂动物过不了这一题，这一题问的正是「想还在吗」。",
     reads="放直后的最后半秒里那一层的平均读数，减去站直时的读数",
     bar_text="扳歪过后，那一层的读数至少还要高出 0.01",
     requires=())

task("the_widest_layer_reaches_the_muscles", "前额叶",
     "前额叶那一层动起来时，肌肉那头会不会跟着动", "full", 14.,
     lambda ctx: return_line_measure(ctx),
     lambda m: association_bar(m, "reach", "association_return_gain",
                               "接上返回线和剪断返回线，肌肉那几组读数之差"),
     "返回线现在是天生的：association_return_gain 出厂就是 1.0，和扇入、回环一样。"
     "两次跑的身体姿势完全一样，接上返回线和剪断返回线（把增益设成 0）差出来的"
     "那部分，只能是返回线送下去的。它问的是「前额叶说的话，肌肉听不听得到」，"
     "而不是「前额叶里有没有东西」。出厂读数 +1.6e-4，增益 4 时 +7.2e-4，"
     "所以它同时在量这条线有多强：把它调弱的候选会考不过。",
     reads="同一段被扳歪的姿势，接上返回线和剪断返回线各跑一次，肌肉那几组的读数之差",
     bar_text="同一相位下，肌肉那几组的读数要因为返回线至少高 0.00005（增益 1 实测 +1.6e-4）",
     requires=())

task("keeps_its_line_towards_what_it_sees", "注意",
     "前面摆个亮东西时，它走的那条线会不会更直", "full", 20.,
     lambda ctx: straight_pull_measure(ctx, 10.), straight_pull_bar,
     "空房间里会走路的动物本来就会绕：场地只有这么大。这一题比的是同一个动物"
     "「前面有亮东西」和「空房间」两次的直度之差，所以它不说明它奔着什么去，"
     "只说明有东西可看的时候它更不绕。",
     reads="有亮东西那次和空房间那次的直度之差（同一条路，各十秒）",
     bar_text="前面有亮东西时，直度至少要提高 0.05（出厂动物三种子实测 +0.102/+0.093/+0.180）",
     requires=("walk_flat",))

TASK_DOC = {
    "the_association_layer_reads_the_body": ("被扳歪那两秒里那一层的平均读数，减去站直时的读数",
                                             "被扳歪时那一层的读数至少要动 0.02"),
    "the_association_loop_holds_what_it_was_given": ("放直后的最后半秒里那一层的平均读数，减去站直时的读数",
                                                     "扳歪过后，那一层的读数至少还要高出 0.01"),
    "the_widest_layer_reaches_the_muscles": ("接上返回线和剪断返回线两次，同一相位下肌肉那几组的读数之差",
                                              "至少高 0.00005（出厂 +1.6e-4，增益 4 时 +7.2e-4）"),
    "keeps_its_line_towards_what_it_sees": ("有亮东西那次和空房间那次的直度之差",
                                            "有亮东西时直度至少高 0.05"),
    "stands_still": ("6 秒里的位移、全程最低的直立程度",
                     "直立程度 ≥ 0.9 并且移动 ≤ 0.70 米"),
    "walk_flat": ("走了多远、全程最低的直立程度", "全程不倒并且移动 ≥ 0.25 米"),
    "walk_furnished": ("12 秒满场走了多远、全程最低的直立程度",
                       "全程不倒并且移动 ≥ 0.35 米"),
    "walks_without_being_told": ("没人给它任何指令时走了多远", "全程不倒并且移动 ≥ 0.30 米"),
    "keeps_walking_without_being_told": ("没人管它，16 秒走了多远",
                                  "全程不倒并且移动 ≥ 0.60 米"),
    "feet_do_not_slide": ("落地那几帧每只脚的滑移速度", "全程不倒、≥ 0.25 米，最滑的一脚 ≤ 0.30 米/秒"),
    "walk_low_step": ("从台阶前 0.55 米出发走 10 秒", "不倒、≥ 0.30 米，并且越过了台阶外沿"),
    "walk_ramp": ("从斜坡前 0.25 米出发走 10 秒", "不倒、≥ 0.25 米，并且越过了坡顶线"),
    "walk_narrow_passage": ("从通道口出发走 10 秒", "不倒、≥ 0.25 米，并且越过了通道口前面那条线"),
    "steps_over_the_curb": ("从矮坎前 0.80 米出发走 10 秒", "不倒、≥ 0.30 米，并且越过了坎的位置"),
    "climbs_onto_the_platform": ("从矮台前 0.90 米出发走 10 秒", "不倒、≥ 0.30 米，并且越过了台面外沿"),
    "stops_at_the_wall": ("12 秒里最远到过的 x、末段墙上接触细胞的活动",
                          "不倒、≥ 0.60 米，没穿过 x=3.42，并且末段墙细胞 ≥ 门槛"),
    "get_up_from_back": ("从仰面开始，多久站回四脚着地、最后还站着吗",
                         "中途站住过 0.5 秒以上，并且结束时直立程度 ≥ 0.5"),
    "get_up_from_side": ("从侧躺开始", "同 get_up_from_back"),
    "nose_up_recover": ("从头朝上竖着开始", "同 get_up_from_back"),
    "nose_down_recover": ("从头朝下栽着开始", "同 get_up_from_back"),
    "stay_up_when_pushed": ("站好后被 140 牛推 0.2 秒", "同 get_up_from_back"),
    "eyes_follow_ball": ("球扫过时眼球角度与球真实方位之差的平均",
                         "平均差 ≤ 0.16 弧度"),
    "eyes_follow_fast_ball": ("两倍速度扫过时的平均差", "平均差 ≤ 0.30 弧度"),
    "eyes_follow_the_ball_both_ways": ("左扫一次、右扫一次各自的平均差",
                                       "两次都 ≤ 0.16 弧度"),
    "eyes_on_a_new_thing": ("东西出现并移动后，眼球摆动多少、朝哪边",
                            "朝它那边摆动 ≥ 0.01 弧度"),
    "eyes_hold_still_on_a_still_thing": ("东西出现但一动不动时，眼球摆动多少",
                                         "摆动 ≤ 0.02 弧度"),
    "eyes_tell_near_from_far": ("球从 2 米到 0.25 米，脑里距离细胞的总活动",
                                "2 米处 ≤ 0.80，0.25 米处 ≥ 2.00"),
    "eyes_see_a_pattern_not_a_blank": ("有花纹的画面和全黑画面下感光那一层的活动差",
                                       "差值 ≥ 0.02"),
    "turn_to_sound": ("同一声放右、放左时眼球各停在哪",
                      "右边的读数比左边更偏右，差 ≥ 0.02 弧度"),
    "ears_tell_front_from_back": ("同一个声音在前面和后面时耳廓四个格子的逐格差",
                                  "逐格平均差 ≥ 0.08"),
    "ears_tell_loud_from_far": ("同一个声音在 1.1 米和 3.2 米时耳蜗细胞的活动差",
                                "差值 ≥ 门槛"),
    "ears_tell_high_from_low": ("高音和低音各自在耳蜗里上三格减下三格",
                                "高音的这个差值比低音高出 ≥ 门槛"),
    "a_bang_makes_it_startle": ("有砰一声和没有时受惊细胞的活动差", "差值 ≥ 门槛"),
    "a_sound_keeps_startling_less": ("砰声一直响着，受惊细胞头 0.2 秒减尾 0.2 秒",
                                     "往下掉 ≥ 0.20"),
    "feels_a_touch_on_the_front": ("身体前面被碰和后面被碰时后退细胞的活动差",
                                   "差值 ≥ 0.30"),
    "lifts_the_foot_that_was_caught": ("绊住某一只脚时那条腿的抬脚细胞，减去另一条腿的",
                                       "两个方向都要高出 ≥ 门槛"),
    "feels_a_slipping_foot": ("某一只脚打滑时那条腿的后退细胞，减去另一条腿的",
                              "两个方向都要高出 ≥ 0.05"),
    "feels_which_way_it_is_leaning": ("身体往一边滚时四个方向细胞各自相对平放时升了多少",
                                      "最大的一格升 ≥ 门槛"),
    "braces_when_it_is_tipped_over": ("快倒和平放时护身细胞的平均活动差", "差值 ≥ 门槛"),
    "feels_a_load_on_one_foot": ("某一只脚压上重物时那条腿的承重细胞，减去另一条腿的",
                                 "两个方向都要高出 ≥ 门槛"),
    "taught_reflex_sticks": ("教之前、教完只听声音、只听安静三种情况下后退细胞的活动",
                             "撤掉手后声音带来的后退比安静高 ≥ 0.05"),
    "nothing_is_taught_without_a_teacher": ("同样的时长同样的声音，只是没有手",
                                            "声音带来的后退比安静高 < 0.05"),
    "the_lesson_is_still_there_later": ("教完先安静 10 秒，再只听声音",
                                        "仍然比安静高 ≥ 0.05"),
    "visual_memory_holds": ("蓝块与身体被摸配对 5 轮后，只给蓝块时回避细胞比安静高多少",
                            "高出 ≥ 0.01"),
    "the_pattern_means_nothing_without_pairing": ("同样的蓝块，一次都没配对过",
                                                  "高出 < 0.01"),
    "the_memory_is_still_there_later": ("配对后再空转 4 段，然后只给蓝块", "高出 ≥ 0.01"),
    "an_idle_animal_does_not_walk_for_a_touch": ("没人叫它走、只喂进身上接触时的位移",
                                                 "直立 ≥ 0.9 并且移动 ≤ 0.10 米"),
    "an_idle_animal_does_not_walk_for_a_ball": ("没人叫它走、道具都在原地时的位移",
                                                "直立 ≥ 0.9 并且移动 ≤ 0.10 米"),
    "moving_is_not_being_pushed": ("自己走 6 秒和被人推 7 秒里，快倒细胞各自的峰值",
                                   "被推的峰值比自己走的高出 ≥ 门槛"),
    "the_body_is_not_a_foot": ("身上被碰和脚上被绊时，护身细胞各自的活动差",
                               "身上那一侧高出 ≥ 0.05"),
    "startled_but_still_hearing": ("砰声一直响着：受惊细胞的头尾落差，和耳蜗细胞的尾段",
                                   "落差 ≥ 0.20 并且耳蜗尾段 ≥ 0.20"),
    "walks_in_a_straight_line": ("受驱动走 10 秒：净位移除以走过的路程",
                                 "直度 ≥ 0.70，并且全程不倒、≥ 0.30 米"),
    "does_not_crab_sideways": ("受驱动走 10 秒后，横向偏出出发朝向多少米",
                               "横向 ≤ 0.25 米，并且全程不倒、≥ 0.30 米"),
    "holds_its_heading_while_it_walks": ("受驱动走 10 秒里朝向累计转过多少弧度",
                                       "累计 ≤ 1.20 弧度，并且全程不倒、≥ 0.30 米"),
    "walks_for_twenty_seconds": ("驱动一直给着，20 秒走了多远",
                                 "全程不倒、位移 ≥ 0.90 米"),
    "walks_at_a_steady_pace": ("受驱动走 10 秒里的平均水平速度",
                               "≥ 0.15 米/秒，并且全程不倒、≥ 0.30 米"),
    "covers_ground_in_sixteen_seconds": ("16 秒里的净位移", "全程不倒、位移 ≥ 1.00 米"),
    "turns_towards_a_sound_while_walking": ("左边响和右边响各走 8 秒，收尾朝向差多少",
                                            "右边那次要比左边那次更偏右 ≥ 0.30 弧度"),
    "turns_towards_a_sound_while_standing": ("站着不动，左边响和右边响各 8 秒的收尾朝向",
                                             "同样是右边比左边更偏右 ≥ 0.10 弧度"),
    "does_not_spin_on_the_spot": ("没人碰它时 8 秒里朝向累计转过多少",
                                  "直立 ≥ 0.9 并且累计 ≤ 1.20 弧度"),
    "walks_down_the_ramp": ("从坡顶朝坡下走 10 秒", "全程不倒、≥ 0.25 米，并且越过了坡的外沿"),
    "steps_down_the_low_step": ("从矮台阶上朝外走 10 秒", "全程不倒、≥ 0.25 米，并且越过了台阶外沿"),
    "steps_down_from_the_platform": ("从矮台上朝外走 10 秒", "全程不倒、≥ 0.25 米，并且越过了台面外沿"),
    "goes_around_the_block": ("挡板正前方出发走 12 秒",
                              "全程不倒、≥ 0.60 米，越过挡板，并且横向挪过 ≥ 0.35 米"),
    "a_touch_on_the_front_holds_it_back": ("同样驱动同样时长，身上一直有接触和没有接触各走多远",
                                           "两边都不倒，被碰那次 ≤ 自由那次的 0.70"),
    "a_bang_stops_it_walking": ("同样驱动同样时长，一直响着砰声和没有砰声各走多远",
                                "两边都不倒，有砰声那次 ≤ 自由那次的 0.70"),
    "gets_up_after_a_hard_shove": ("站好后被 220 牛推 0.2 秒", "同 stay_up_when_pushed 的判据"),
    "eyes_converge_on_a_near_thing": ("球在 0.25 米和 2 米时，眼睛内收细胞减外展细胞",
                                      "近距离比远距离高出 ≥ 0.10"),
    "eyes_look_at_a_still_thing_at_the_side": ("球停在左边和停在右边，各自稳定后眼球角度",
                                               "两次都要朝球那边偏 ≥ 0.05 弧度"),
    "its_own_eyes_are_felt": ("眼球被命令转开和不转时，眼球本体细胞的活动差",
                              "差值 ≥ 0.0002"),
    "eyes_hold_a_slow_thing": ("球用一半速度扫过时的平均跟踪误差", "平均差 ≤ 0.16 弧度"),
    "ears_hear_something_far_away": ("三米外的一声和没有声音时耳蜗细胞的活动差",
                                     "差值 ≥ 0.02"),
    "ears_hear_a_sound_behind": ("同一声在背后和在前头时耳廓细胞的逐格差",
                                 "逐格平均差 ≥ 0.02"),
    "two_sounds_at_once_are_still_two": ("两个声源同时亮和只亮一个时耳廓细胞的逐格差",
                                         "逐格平均差 ≥ 0.02（比的是花样，不是总量）"),
    "feels_a_touch_on_the_left_side": ("左边被碰和右边被碰时两条让路细胞各自的差",
                                       "两个方向都要高出 ≥ 0.30"),
    "feels_a_touch_behind_it": ("后面被碰和前面被碰时四格「贴着东西」细胞各自的差",
                                "两个方向都要高出 ≥ 0.20"),
    "feels_a_touch_on_both_sides_at_once": ("只碰前面，和前面加左边一起碰时，四格「贴着东西」细胞亮了几格",
                                            "只碰前面亮 1 格，两处一起碰要亮 ≥ 2 格"),
    "feels_the_ground_under_the_back_legs": ("后腿压上重物时那条腿的承重细胞，减去另一条",
                                             "两个方向都要高出 ≥ 0.05"),
    "learns_the_lesson_in_five_seconds": ("只教五秒，之后只听声音和只听安静两种情况下后退细胞的活动",
                                          "声音带来的后退比安静高 ≥ 0.05"),
    "the_lesson_survives_a_longer_pause": ("教完先安静 30 秒，再只听声音", "仍然比安静高 ≥ 0.05"),
    "remembers_a_second_pattern": ("绿色图案配对 5 轮后，只给图案时回避细胞比安静高多少",
                                   "高出 ≥ 0.01"),
    "remembers_after_one_pairing": ("只配对 1 轮，然后只给图案", "高出 ≥ 0.01"),
    "stands_still_for_twenty_seconds": ("20 秒里的位移、全程最低的直立程度",
                                        "直立程度 ≥ 0.9 并且移动 ≤ 1.40 米"),
    "stands_still_on_a_slope": ("坡顶站 8 秒，全程最低的直立程度", "直立程度 ≥ 0.85"),
    "stays_up_when_nudged_gently": ("站好后被 90 牛推 0.2 秒",
                                    "中途站住过 0.5 秒以上，结束时直立程度 ≥ 0.5"),
    "all_four_feet_leave_the_ground": ("走 10 秒里每只脚离地超过 2 毫米的时间占比",
                                       "全程不倒、≥ 0.25 米，四只脚每只 ≥ 0.02"),
    "the_two_sides_step_alike": ("同一批抬脚占比里，同一条带子两只脚的差",
                                 "全程不倒、≥ 0.25 米，前对和后对的差都 ≤ 0.20"),
    "keeps_walking_for_thirty_seconds": ("三十秒走了多远", "全程不倒并且移动 ≥ 1.30 米"),
    "does_not_slow_down_on_a_long_walk": ("二十秒里头五秒和末五秒各自的平均速度",
                                          "全程不倒、头五秒 ≥ 0.15 米/秒，末五秒 ≥ 头五秒的 0.50"),
    "keeps_walking_while_it_turns": ("左边响和右边响各走 16 秒，各自的前进量和收尾朝向",
                                     "两次都不倒、都要走 ≥ 0.45 米，朝向差 ≥ 0.30 弧度"),
    "backs_away_from_a_hand_on_its_chest": ("胸口一直有接触的 10 秒里最大的后退量",
                                            "不倒，并且往后退 ≥ 0.10 米"),
    "keeps_its_line_with_a_sound_behind": ("背后有声音时走 12 秒的收尾朝向和距离",
                                           "全程不倒、≥ 0.30 米，并且朝向偏 ≤ 0.35 弧度"),
    "does_not_fall_walking_into_the_wall": ("朝墙走 12 秒里最低的直立程度、最远到的 x",
                                            "直立程度 ≥ 0.9 并且没穿过 x=3.42"),
    "threads_the_passage_without_touching": ("从通道口走 10 秒，越过通道口没有、墙接触细胞的峰值",
                                             "不倒、≥ 0.25 米、越过通道口，并且墙细胞峰值 ≤ 0.10"),
    "sees_a_red_square_from_a_green_one": ("红方块和绿方块分别给眼睛，对色那层细胞逐格的差",
                                           "逐格平均差 ≥ 0.02"),
    "sees_a_big_ball_from_a_small_one": ("同一个球换大小（0.18 米和 0.05 米），感光变化层逐格的差",
                                         "逐格平均差 ≥ 0.02"),
    "eyes_follow_the_ball_up_and_down": ("球上下扫过时眼球俯仰角与球仰角之差的平均",
                                         "平均差 ≤ 0.22 弧度"),
    "eyes_follow_the_ball_far_away": ("球在两米外扫过时的平均跟踪误差", "平均差 ≤ 0.20 弧度"),
    "eyes_follow_the_ball_close_up": ("球在四十厘米处扫过时的平均跟踪误差", "平均差 ≤ 0.20 弧度"),
    "the_eyes_let_go_when_it_is_gone": ("球停在旁边时的眼球角度、球撤走后的眼球角度",
                                        "停着时朝球偏 ≥ 0.05 弧度，撤走后偏 ≤ 0.05 弧度"),
    "notices_a_thing_that_creeps_in": ("东西出现并慢慢移动后，眼球摆动多少、朝哪边",
                                       "朝它那边摆动 ≥ 0.01 弧度"),
    "a_sound_that_stops_is_no_longer_heard": ("声源挪走前后各半秒的耳蜗活动",
                                              "响着时 ≥ 0.02，停掉后 ≤ 响着时的 0.50"),
    "a_quiet_room_is_not_a_sound": ("安静的屋子里，耳蜗那一路和它旁边那格的平均活动",
                                    "耳蜗每一格 ≤ 0.02，旁边那格每一格 ≥ 0.05"),
    "nothing_touching_it_is_not_nothing": ("没被碰的站姿里，碰、绊、打滑三路和它们旁边那格",
                                           "三路的每一格 ≤ 0.02，旁边那三格的每一格 ≥ 0.05"),
    "the_weight_pair_still_carries_the_weight": ("四只脚各自的受力格和旁边那格之和",
                                                 "每一对相加都落在 1 ± 0.10 内"),
    "ears_tell_left_from_right": ("同一声在左肩和右肩时耳廓四个格子的逐格差", "逐格平均差 ≥ 0.02"),
    "two_sounds_at_one_shoulder_are_still_two": ("同一边两个声源一起亮和只亮一个时耳廓细胞的逐格差",
                                                 "逐格平均差 ≥ 0.02"),
    "a_bang_behind_it_still_startles": ("背后有砰声和没有时受惊细胞的活动差", "差值 ≥ 0.10"),
    "feels_a_touch_on_the_right_side": ("右边被碰和左边被碰时两条让路细胞各自的差",
                                        "两个方向都要高出 ≥ 0.30"),
    "feels_a_light_hand_too": ("轻轻碰（0.30）左边和右边时两条让路细胞各自的差",
                               "两个方向都要高出 ≥ 0.10"),
    "feels_the_ground_under_the_front_legs": ("前腿压上重物时那条腿的承重细胞，减去另一条",
                                              "两个方向都要高出 ≥ 0.05"),
    "lifts_each_of_the_four_feet": ("四条腿各绊一次，每次抬脚那组细胞里最亮的一格是不是那条腿",
                                    "四条腿都要是自己那条腿最亮"),
    "the_flat_ground_is_not_a_bump": ("平地上站着的抬脚细胞和绊住细胞", "两组的每一格都不超过 0.05"),
    "it_forgets_when_nobody_teaches_anymore": ("教会时的成绩，和之后没人教三十秒的成绩",
                                               "先学会（≥ 0.05），再掉到原来的 70% 以下"),
    "a_second_short_lesson_adds_to_the_first": ("第一个五秒教完的成绩，和第二个五秒教完的成绩",
                                                "第一遍学会（≥ 0.05），第二遍还要再高 ≥ 0.01"),
    "the_lesson_is_still_there_after_a_minute": ("教完先安静 60 秒，再只听声音",
                                                 "仍然比安静高 ≥ 0.05"),
    "remembers_two_patterns_at_once": ("配对的那块图案和没配过的那块，各自让回避细胞高多少",
                                       "配对那块 ≥ 0.01，没配的那块 < 0.01"),
    "the_memory_survives_a_small_change": ("配对的图案，和把它挪几行以后的同一个图案",
                                           "原图案 ≥ 0.01，挪过的那块 ≥ 原图案的 30%"),
    "walks_towards_a_bright_thing": ("同样 10 秒，前面有亮东西和没有，各走多远",
                                     "两边都不倒、空场那次 ≥ 0.30 米，有亮东西那次多走 ≥ 0.05 米"),
    "knows_when_it_is_tipped_over": ("身体被扳歪 0.45 弧度和站着时，它自己那组姿态细胞的活动差",
                                     "平均差 ≥ 0.05"),
    "the_hidden_layer_sees_the_picture": ("横条纹和竖条纹分别给眼睛，第一层特征细胞逐格的差",
                                          "逐格平均差 ≥ 0.01"),
    "the_compressed_layer_sees_the_picture": ("同一对比下，再往下一层的逐格差",
                                              "逐格平均差 ≥ 0.01"),
    "hurries_towards_a_bright_thing": ("同样十秒、同一条路，前面有亮东西和空房间两次的平均速度之比",
                                       "两次都不倒、空房间那次平均速度 ≥ 0.15 米/秒，"
                                       "并且有亮东西那次快 ≥ 1.25 倍"),
    "runs_without_being_told": ("没人给它任何指令，它自己走十秒的全程平均速度",
                                "全程不倒，平均 ≥ 0.25 米/秒"),
    "runs_for_twenty_seconds": ("没人管它，二十秒的净位移",
                                "≥ 2.00 米，且全程不倒"),
    "keeps_running_while_it_turns": ("一边跑一边左右各响一次：两次的位移、平均速度和收尾朝向差",
                                     "两次位移 ≥ 0.45 米、平均速度 ≥ 0.25 米/秒、朝向差 ≥ 0.30 弧度"),
    "the_dark_room_is_not_silence": ("全黑画面和全白画面下，输入层两半"
                                     "和它们各自的汇总层的平均活动",
                                     "黑画面要亮着暗那一半（≥ 0.05）、"
                                     "不亮亮那一半（≤ 0.05）、"
                                     "暗那一半要把自己的汇总层点亮（≥ 0.05），"
                                     "白画面反过来"),
    "the_memory_survives_a_long_life": ("教完的那一刻的成绩，和它自己走一分钟（边听边走、规则开着）之后的成绩",
                                        "先学会（≥ 0.05），之后不低于原来的 80%"),
    "a_second_tone_does_not_wipe_the_first": ("先教 262 Hz、再教 880 Hz，之后分别只听这两个声音",
                                              "两个音都要 ≥ 0.05，且第一个不低于原来的 80%"),
    "the_route_stops_growing_at_its_ceiling": ("一直握着不放六十秒，那几根可教连接头、中、后各二十秒各涨多少",
                                               "头二十秒涨 > 0.01，最后二十秒的涨幅 ≤ 头二十秒的 25%"),
}

# The two sentences published with each question live in one table above, and
# the questions themselves are written down without them.  Nothing joined the
# two: ``--explain`` and the tests read ``entry["reads"]`` and found it empty
# on 126 of the 130 questions, so the bank's own description of itself was
# dead text.  Join them here, where both exist.  A question that states its own
# sentences keeps them.
for _entry in TASKS:
    _reads, _bar_text = TASK_DOC.get(_entry["name"], ("", ""))
    if not _entry["reads"]:
        _entry["reads"] = _reads
    if not _entry["bar_text"]:
        _entry["bar_text"] = _bar_text
_missing = [entry["name"] for entry in TASKS
            if not entry["reads"] or not entry["bar_text"]]
if _missing:
    raise ValueError("questions with no published reading or bar: %s" % _missing)
del _entry, _reads, _bar_text, _missing



# ---------------------------------------------------------------------------
# the cheap filter a round draws with
# ---------------------------------------------------------------------------
# A round cannot afford 27 minutes of simulation per animal, so a candidate is
# put through this list first and only the survivors are put through the whole
# bank.  The list is not "the interesting questions": it is the questions the
# birth animal fails at its cheapest.  Every name here was measured on the
# birth genome at scaffold 0 and costs what that run says it costs.  Eight
# groups are covered - righting, the feature layers, seeing, the association
# layer, terrain, walking a line, running and turning - so a candidate with one
# unusual ability somewhere else is not thrown away for having no righting
# reflex.
#
# Two things this list is not.  It is not a score: a candidate that passes all
# eight is not "better" than one that passes six, it is different.  And it is
# not the exam: a candidate that survives is put through the whole bank over
# three scaffolds before anything is claimed about it.
#
# The bank's own cost is worth writing down here, because it is what makes the
# filter necessary.  The birth animal spends 1609 s - 26.8 min - of simulation
# on the 128 questions at one scaffold.  A thousand candidates on the whole
# bank would be 447 h of simulation, which is not a round, it is a month.  The
# eight below cost about 59 s, so a thousand of them is a night.
#
# The old list had two questions the birth animal passes (nose_down_recover,
# stay_up_when_pushed) sitting in it while claiming to hold "the failing ones",
# and spent 95 of its 158 s - 60% - on two questions (steps_over_the_curb,
# runs_without_being_told).  This one holds only questions it fails and covers
# one more group for less than the old price.
# Cheapest first, and the order of this tuple is the order the questions are
# asked.  The probe is a filter, and it can be run as a chain that stops at the
# first miss (``--stop-on-first-miss``); a candidate that cannot hold a loop
# should not first pay eleven seconds to fail a walking question.
PROBE = ("the_association_loop_holds_what_it_was_given",  # 前额叶 1.6 s
         "the_compressed_layer_sees_the_picture",   # 特征层  1.7 s
         "eyes_converge_on_a_near_thing",           # 看      3.9 s
         "get_up_from_back",                        # 自救    4.0 s
         "threads_the_passage_without_touching",    # 地形   11.3 s
         "backs_away_from_a_hand_on_its_chest",     # 转弯   11.5 s
         "walks_in_a_straight_line",                # 走路   11.7 s
         "runs_without_being_told")                 # 跑     13.6 s


def stage_tasks(stage):
    if stage == "screen":
        return [t for t in TASKS if t["stage"] == "screen"]
    if stage == "probe":
        # In PROBE order, not in the order the questions were written down:
        # this stage is a chain and its order is part of what it is.
        by_name = {entry["name"]: entry for entry in TASKS}
        return [by_name[name] for name in PROBE]
    return list(TASKS)


def run_task(entry, genome, seed):
    ctx = context(genome, seed)
    started = time.perf_counter()
    try:
        measures = entry["measure"](ctx)
    except Exception as exc:                        # a failed exam is a result
        measures = dict(status="error", error="%s: %s" % (type(exc).__name__, exc))
    try:
        passed, why = entry["bar"](measures)
    except Exception as exc:
        passed, why = False, "bar failed: %s: %s" % (type(exc).__name__, exc)
    return dict(task=entry["name"], group=entry["group"], seed=int(seed),
                passed=bool(passed), why=why, measures=measures,
                margin=margin(entry["name"], measures),
                wall_seconds=time.perf_counter() - started)


def run_bank(genome, seed, stage="screen", names=None, on_result=None,
             stop_at_first_miss=False):
    """Every question once, in order.  ``on_result`` is called per answer,
    so a long run reports as it goes instead of all at the end.

    ``stop_at_first_miss`` turns the bank into a chain: the first question the
    animal fails ends the exam.  On the probe stage that is the difference
    between paying for all eight questions and paying for the two a candidate
    usually gets through, and the stage is ordered cheapest first for exactly
    this.  The answer that stopped it is still recorded - a chain that stopped
    is a different thing from a bank that was finished, and the caller has to
    say which one it holds.
    """
    entries = stage_tasks(stage)
    if names:
        entries = [t for t in entries if t["name"] in set(names)]
    results = []
    for entry in entries:
        result = run_task(entry, genome, seed)
        results.append(result)
        if on_result is not None:
            on_result(result)
        if stop_at_first_miss and not result["passed"]:
            break
    return results


def summarise(results):
    """Counts, plus where on the ladder this animal got to.

    ``frontier`` is computed over the whole bank and then cut down to the tasks
    that were actually run, so a screening round over a handful of questions
    still reports the rung those questions open.
    """
    ran = {r["task"] for r in results}
    passed = {r["task"] for r in results if r["passed"]}
    depth, _ = deepest_reached(passed)
    return dict(passed=sum(1 for r in results if r["passed"]), total=len(results),
                failed=[r["task"] for r in results if not r["passed"]],
                deepest_rung=depth,
                frontier=[name for name in frontier(passed) if name in ran],
                wall_seconds=sum(r["wall_seconds"] for r in results))
# ---------------------------------------------------------------------------
# printing and running
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="run the exam bank once")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--tree", action="store_true",
                        help="the ladder: which questions come after which")
    parser.add_argument("--explain", action="store_true",
                        help="every question, what it reads, its bar, the birth reading")
    parser.add_argument("--unlocked", default=None,
                        help="comma-separated tasks already passed; print what is open")
    parser.add_argument("--stage", default="screen", choices=("screen", "probe", "full"))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--genome", default=None,
                        help="JSON file or inline JSON of named gains")
    parser.add_argument("--tasks", default=None, help="comma-separated task names")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--birth-text", action="store_true",
                        help="turn a bank report into the BIRTH dictionary")
    parser.add_argument("--birth-source", type=Path,
                        default=Path("artifacts/exam_birth_full.json"))
    return parser.parse_args(argv)


def load_genome(value):
    if value is None:
        return {}
    path = Path(value)
    text = path.read_text(encoding="utf-8-sig") if path.exists() else value
    return json.loads(text)


def split_names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def print_ladder():
    depth = ladder()
    by_rung = {}
    for entry in TASKS:
        by_rung.setdefault(depth[entry["name"]], []).append(entry)
    for rung in sorted(by_rung):
        print("第 %d 级" % rung)
        for entry in by_rung[rung]:
            after = "，先过 %s" % "、".join(entry["requires"]) if entry["requires"] else ""
            print("  %-38s %s%s" % (entry["name"], entry["question"], after))
    print("\n共 %d 级、%d 道题。" % (max(by_rung) + 1 if by_rung else 0, len(TASKS)))


def print_birth_text(path):
    report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    lines = {}
    for run in report["runs"]:
        for result in run["results"]:
            reading = "seed %d %s" % (result["seed"],
                                      "PASS" if result["passed"] else "fail")
            detail = result.get("why") or ""
            lines.setdefault(result["task"], []).append("%s (%s)" % (reading, detail))
    print("BIRTH = {")
    for entry in TASKS:
        name = entry["name"]
        text = "; ".join(lines.get(name, ["not run"]))
        print('    "%s": %s,' % (name, json.dumps(text, ensure_ascii=False)))
    print("}")


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for entry in TASKS:
            print("%-38s %-10s %-6s %6.1f s-big  %s"
                  % (entry["name"], entry["group"], entry["stage"], entry["sim_seconds"],
                     entry["question"]))
        print("\n%d tasks, %d genes" % (len(TASKS), len(GENES)))
        return 0
    if args.tree:
        print_ladder()
        return 0
    if args.unlocked is not None:
        passed = split_names(args.unlocked)
        open_now = frontier(passed)
        print("已经会：%d 道" % len(passed))
        print("接下来能考的：%s" % ("、".join(open_now) if open_now else "没有新的了"))
        return 0
    if args.birth_text:
        print_birth_text(args.birth_source)
        return 0
    if args.explain:
        depth = ladder()
        by_group = {}
        for entry in TASKS:
            by_group.setdefault(entry["group"], []).append(entry)
        for group, entries in by_group.items():
            print("=" * 78)
            print("【%s】 %d 道" % (group, len(entries)))
            for entry in entries:
                print("  %s  (%s, 第 %d 级, 模拟 %.1f 秒)"
                      % (entry["name"], entry["stage"], depth[entry["name"]],
                         entry["sim_seconds"]))
                if entry["requires"]:
                    print("     先过：%s" % "、".join(entry["requires"]))
                print("     问：%s" % entry["question"])
                print("     量：%s" % entry["reads"])
                print("     门槛：%s" % entry["bar_text"])
                print("     出厂动物：%s" % BIRTH.get(entry["name"], "未测"))
                print("     这题不证明：%s" % entry["note"])
        print("=" * 78)
        print("共 %d 道题，%d 个可动基因；screen 档 %d 道，full 档 %d 道。"
              % (len(TASKS), len(GENES), len(stage_tasks("screen")), len(TASKS)))
        return 0
    genome = load_genome(args.genome)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    names = split_names(args.tasks) if args.tasks else None
    report = dict(stage=args.stage, genome=genome, seeds=seeds, runs=[],
                  reference_size=REFERENCE_SIZE, bar=BAR,
                  arena=str(ARENA), genes=list(GENES))
    def show(result):
        print("%2d %-38s %s  %s" % (result["seed"], result["task"],
                                     "PASS" if result["passed"] else "fail",
                                     result["why"]))
        sys.stdout.flush()

    for seed in seeds:
        results = run_bank(genome, seed, stage=args.stage, names=names, on_result=show)
        summary = summarise(results)
        report["runs"].append(dict(seed=seed, results=results, summary=summary))
        print("%2d 会了 %d/%d，最高到第 %d 级" % (seed, summary["passed"], summary["total"],
                                                summary["deepest_rung"]))
        sys.stdout.flush()
    report["wall_seconds"] = sum(r["summary"]["wall_seconds"] for r in report["runs"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
        print("-> %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())