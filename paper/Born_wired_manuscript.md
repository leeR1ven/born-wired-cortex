# Born wired: innate cortical connectivity plus local plasticity is enough to stand, walk and look

**Author:** Zhiwen Li.

**Affiliation:** Independent Researcher, No. 67 Yuanren Street, Huangjing Town, Taicang,
Suzhou, Jiangsu, China.

**Correspondence:** rivenlee94@gmail.com . ORCID: 0009-0005-8289-6393

*Version of 2026-09-17. Every number quoted below was produced by the script named next to it;
the scripts and the log of every run are in the repository, and Section 5.9 says where to find
them. Experiments whose outcome contradicted our prediction are reported as such and are listed
in Section 4.5.*

---

**Abstract**

Brains are born with largely pre-specified cortical wiring, and the only rule locally available to a synapse is correlation between the neurons it connects; machine-learning agents instead optimise a global objective. How far does the biological combination reach on its own? We built a system of 143,796 model cortical neurons connected by 308,809 pre-specified "instinct" synapses, driving a simulated quadruped with no reward, no error signal, no gradient and no training loop. Placed prone, it stands up on 5/5 network seeds; shown red, it walks toward the stimulus on 5/5, over a median 1.73 m; with no sensory input it collapses; with the instinct table cleared the same brains can neither stand nor approach (0/5). Silencing the prefrontal population abolishes visually guided walking (0/200 ticks, 5/5) while leaving a wired reflex unchanged. The behaviour is produced by the cortex rather than by the stimulus: removing the recurrent step removes the action while sparing the reflex. Adding innate structure grows the repertoire without retraining; a 200-cell dopamine population writes one new sense-to-action link during the life of a single brain; gaze tracking emerges from static look-at rules and a motion-onset reflex rather than being written down. No rule in the system lowers a weight, and the acquired walk does not yet stay upright; both are reported rather than hidden.

## Author Summary

Most artificial intelligence today works like a machine that is fed examples and graded until it gets them right. A brain does not work that way. It is born with a great deal of wiring already in place, and the only rule its synapses can use is that cells which fire together tend to connect more strongly. I wanted to know how far that starting point alone can carry a body. To find out, I built a simulated brain of about 140,000 cells, gave it a fixed set of inborn connections, put it inside a simulated four-legged body, and let it run with no reward and no training. It got up from lying down, walked toward a red object it could see, and followed a moving object with its eyes. When I removed the inborn connections, the same brain could do nothing. When I silenced one part of it, the animal stopped using its eyes to walk but kept its reflexes. The animal still falls over after a while, and I report that rather than hide it. The result suggests that what a brain is born with matters at least as much as what it learns.

## 1. Introduction

### 1.1 Two pictures of a layered network

The dominant picture of a deep network is a feed-forward transducer: a token or a pixel enters
on the left, an answer leaves on the right (Brown et al., 2020), and the loss is back-propagated
through the stack (Lillicrap et al., 2020). In that picture the intermediate layers are the
computation and the output layer *is* the answer.

We work from a different picture, and the difference is not cosmetic. We state it as six
claims, (P1) to (P6), so that the results can be read as tests of specific ones.

* **(P1) A layered sensory network is a compressor.** It exists to take a very large number of
  peripheral channels and reduce them to a small number of cortical channels. Its output is not an
  answer; it is a bus. Reading the answer off the last layer is mistaking signal conditioning for
  thought. The anatomical version of the same idea -- a hierarchy of cortical areas in which each
  stage re-represents its input -- is classical (Hubel & Wiesel, 1962; Felleman & Van Essen, 1991;
  Mountcastle, 1997).
* **(P2) Thought is co-activation of cortical neurons** (Hebb, 1949; Hopfield, 1982; Amit, 1989;
  Wang, 2002). A thought is a sequence of populations, and the next population is fully determined
  by the current weights and the currents arriving on the somata. There is no separate inference
  stage. The testable consequence is sharp: remove the recurrent step and the behaviour should go
  with it, while anything the wiring maps directly from sense to muscle should survive untouched
  (R8).
* **(P3) The prefrontal population compresses again** (Fuster, 2001; Miller & Cohen, 2001). It takes
  the already-compressed signals of several other regions and reduces them further, and then thinks
  with exactly the same rules as everything else. Information is unavoidably lost in this second
  compression; that is a property of the design, not a defect.
* **(P4) Instinct is the initial state of the sheet** (Tinbergen, 1951). The innate part of the
  design is not a separate module and not a set of conditions in code: it is the connectivity the
  cortical sheet is born with, both within a region and between regions, and it includes the
  suppressive connections (a rule that must suppress something drives an inhibitory cell; there are
  no negative weights anywhere in the system). Everything the sheet does it does with that wiring,
  and what experience changes, it changes slowly and locally.
* **(P5) A network cannot start from silence** (Pfeifer & Bongard, 2006). A newborn cortex already
  has ordered wiring, because an organism that must discover standing, breathing and gaze
  stabilisation from an uninformative start does not survive the discovery period. The innate wiring
  provides an already-ordered starting repertoire; experience then writes content into it.
* **(P6) A capability can be acquired during the life of one brain, without an objective.** The
  innate wiring fixes the starting repertoire, but it is not the whole of what the sheet will
  ever do: the same local correlation rule that slowly adjusts the innate connections can also
  write new ones while the brain is behaving. If a reward-like population of cells is present at
  all, it is one more population -- a population whose firing licenses that writing -- and not an
  error signal computed from a target. What it buys is a link from something the brain can
  already sense to something it can already do (R9).

### 1.2 What this paper tests

If the picture above is right, then a system should be able to do real things with

* a single recurrent population of model cortical neurons,
* a fixed, pre-specified wiring diagram (which we call the *instinct table*), and
* a purely local plasticity rule that has no access to any task objective,

driving a real body. We therefore built one and asked seven questions, each of which has a
falsifiable answer:

1. Does the innate wiring alone close the sensorimotor loop? (**R1**)
2. Can we attribute a behaviour to a specific population by silencing it? (**R2**)
3. Can the repertoire be *grown* by adding innate structure, without retraining and without
   destroying what is already there? (**R3**)
4. Does local plasticity buy anything that a fixed recogniser does not have? (**R4**)
5. Does hierarchical compression cost anything measurable, and does it cost anything
   behaviourally? (**R5**, **R6**)
6. While a behaviour is running, is it the stimulus or the cortical population that is producing
   it? (**R8**)
7. Can a capability be *acquired* during the life of one brain, with no objective and with
   nothing added to the system except a reward-like population of cells? (**R9**)

We also report two supporting measurements: gaze tracking that emerges from primitive reflexes
with no tracking rule anywhere in the wiring (**R6**), and the tolerance of behaviour to
*blurred* innate wiring (**R7**).
---

## 2. System

Figure 1 is a schematic of one tick. Senses drive two sensory hierarchies; the hierarchies drive
one recurrent cortical sheet; the sheet drives the muscles. There is nothing else in the loop.

### 2.1 A single cortical population

The model cortex is one flat array of **143,796 binary neurons**. It is not divided into
modules with separate code paths; the divisions are address ranges inside the array, and every
region is updated by the same update rule on the same tick.

| region | cells | role |
|---|---|---|
| visual cortex | 23,040 | output of the visual hierarchy (compressed) |
| visual detail | 23,040 | first layer of the same hierarchy (uncompressed) |
| visual motion | 1,920 | frame-difference / relay cells used by the gaze reflex |
| auditory cortex | 30,000 | output of the auditory hierarchy |
| prefrontal | 53,200 | second-stage compressor over visual, auditory and motor cortex |
| motor cortex | 160 | 16 muscles x 10 cells; the number of lit cells is the force |
| motor memory | 3,576 | time cells; lighting the first cell of an action plays the whole action |
| proprioception | 480 | joint-angle-derived body sense |
| vestibular | 120 | trunk tilt / tilt rate |
| body state | 60 | e.g. "is the trunk low", "is it lying down" |
| dopamine | 200 | reward signal; when lit, co-active synapses are written immediately
(only used in R9, silent otherwise) |
| inhibitory pool | 8,000 | every suppressive instinct borrows a cell from here |
| **total** | **143,796** | |

All 143,796 cells are updated simultaneously, once per 20 ms tick (50 Hz), by

> a cell fires on this tick if the excitatory current it receives plus whatever else arrives on
> its soma exceeds a fixed threshold of 1.0, and it is not held below threshold by an
> inhibitory cell.

There is no separate "inference" pass and no read-out layer. The motor command is literally the
activity of 160 cells in the same array.

One region departs from "each tick replaces the state". Inside each named block of the prefrontal
population the cells are wired head-to-tail into a closed chain (6,281 synapses in five blocks), and
the main loop *adds* the newly computed prefrontal code to the code still running, so that a block
outlasts the stimulus that lit it. We report R1-R8 with that persistence **switched off** -- the
prefrontal code is recomputed from its input on every tick -- because that is the configuration
those experiments were run in, and because it makes the causal reading of R2 and R8 unambiguous.
R9, and the rows marked as persistent in R2 and R8, report both configurations side by side;
Methods 5.5 gives the switch and why persistence is there at all: a code that cannot outlast its
stimulus cannot be thought with (P2).

### 2.2 The instinct table

Behaviour-relevant wiring is written down as a table of rules of the form

```
source_block  ->  target_block   strength   frozen?
```

where a *block* is a named set of cells (a name is defined by a calibration procedure that
lights a physical condition, e.g. "a red region is straight ahead", and records which cells
responded). At load time each rule is expanded into per-cell synapses, with the stated strength
divided evenly across the synapses of the rule; a strength of 2.0 therefore means "if the whole
source block is lit, the target cell is driven to fire, and if only a small part of it is lit it
is not".

The current table contains **585 rules**: 581 in the written table and 4 in a derived reflex
table. They group as follows.

| group | rules | what it is |
|---|---|---|
| motor repertoire | 221 | one rule per action the body can perform (10 static postures, 4 slow actions, 207 locomotion gaits). Writing one rule for an action wires the whole action: its time cells chain into each other so that lighting the first cell plays the action out. |
| balance / righting | 6 + 4 | trunk low or inverted -> stand up; leaning forward -> shift weight back; and 4 reflexes that hold a specific shank force above a floor when the body is tilted. |
| visually guided approach | 6 | red ahead -> forward (two rules, see R2); red on the left -> turn left; red on the right -> turn right |
| auditory | 3 | a broadband sound -> shuffle in place; a fixed narrow-band tone -> hold the standing posture **and** suppress the walking action |
| gaze | 345 | 9 coarse direction rules, 14 per-column rules driven from the *uncompressed* visual layer, and 322 rules implementing the motion-onset reflex of R7 |

Expanding the table produces **308,809** excitatory synapses (plus 6,855 inhibitory ones). Everything else in the network is
sparse random background connectivity generated from a fixed seed; we refer to different seeds
as different *brains*.

Two of the encoding choices are old ideas in model form. A motor program that runs itself out
once triggered, rather than being recomputed every tick, is the classical central-pattern-
generator proposal (Grillner & Wallen, 1985); and "one cell per instant of the action" is a time
cell in the sense of Eichenbaum (2014).

Three properties of this design matter for the interpretation of everything below.

1. **The innate wiring is authored, not learned.** Whoever writes the table is the "genome".
   For the locomotion repertoire the rules were distilled from a separately trained
   reinforcement-learning policy (see Methods 5.4); this is supervision that happened *before*
   birth, not during life.
2. **No rule is an `if` statement.** Action selection is competition between currents on the
   somata of motor-memory cells. Two rules can each be insufficient on their own and sufficient
   together (R2 uses exactly this), and the balance between regions is not fixed.
3. **Suppression is done by cells, never by negative weights.** Where a rule must suppress
   something (the tone that stops walking, the motion-onset subtraction) the table borrows an
   inhibitory cell and drives it, so the "negative" part of the diagram is itself a population
   of neurons that other rules can act on. Expanded, every synapse in the network has a positive
   weight; the minus sign exists only in the text file (Methods 5.1).

### 2.3 Plasticity

The only learning rule in the system is local and correlation-based (Hebb, 1949; Turrigiano &
Nelson, 2004; Magee & Grienberger, 2020). Between consecutive ticks,
pairs of cells that were co-active have their synapse strengthened, pairs that were not have it
decayed, and a synapse that has been co-active often enough is *created* if it did not exist.
The rule has a small fixed learning rate, saturating bounds, a co-activation requirement before
a new synapse is created, and a per-tick cap on how many synapses can be created. It never sees
a reward, a target, or an error. Frozen synapses in the instinct table are exempt, which is how
we express "this one must not move" (for example the balance reflexes). All constants are listed
in Methods 5.5.

### 2.4 Body and sensors

The body is a Unitree Go2 quadruped simulated in MuJoCo with joint-angle servos: the 160 motor
cells decode to 16 target angles, and the physics runs ten MuJoCo sub-steps of 2 ms per cortical tick.
The visual hierarchy receives a 1920x1080 first-person image, downsamples it to a 24x16
macro-pixel grid with three colour channels and 10 cell-pairs per level, so one visual cell
subtends about 6.25 degrees horizontally. The field of view is 100 degrees horizontal by 75
degrees vertical. The auditory hierarchy receives a 4096-bin spectrum. Proprioception,
vestibular and body-state regions receive hand-defined predicates over the joint angles and
trunk pose. The eyes have two degrees of freedom with a 30 degree range, a strong viscous load
and a muscle acceleration of 500 degrees/s^2, so a single tick of maximum force moves the eye
about 0.1 degrees; sustained force over roughly ten ticks is needed to make a large saccade.
---

## 3. Results

### R1. The innate wiring alone closes the loop (Figure 2)

We built a brain from the instinct table plus random background connectivity, attached it to
the body, and gave it nothing but its own senses.

| condition | measure | 5 seeds |
|---|---|---|
| placed prone, no instruction | final trunk height | **5/5 above 0.20 m**; median 0.261 m (min 0.261) from a starting 0.099 m |
| placed upright, red region in front | forward displacement in 4 s | **5/5 above 0.30 m**; median +1.73 m (min +1.48, max +2.09) |
| placed upright, **no sensory input at all** | final trunk height / tilt | **5/5 collapsed**; median 0.184 m, tilted 41 degrees |
| placed prone, **instinct table cleared** (background connectivity only) | final trunk height, then forward displacement with red ahead | **0/5 above 0.20 m**; median 0.184 m, every seed exactly the no-sensation value; **0/5 approach** (median -0.12 m) |

(logs: `日志_闭环多种子_新.log`, `日志_白脑_新.log`; scripts `实验_闭环前提_多种子.py`, `实验_白脑对照.py`)

The third row is the control that matters. With the sensory regions blanked, the body does not
"do nothing gracefully" — it falls over, because the motor cells are receiving no current from
anywhere and the servo set-points collapse to rest. Nothing in the simulation holds the body up
except the cortex. Standing up happens because the body-state and proprioceptive regions report
*lying down*, and the instinct table contains the rule "lying down -> stand up".

Because the stand-up trajectory lives in the motor repertoire, the *height* reached is identical
across seeds (0.261 m); what the random background changes is the walking distance, which varies
between 1.48 and 2.09 m across the five seeds. We report this rather than a single run because
the single-run version of this table appeared in an earlier draft of this project and was not
reproducible evidence.


**Standing up is not, in this body, a control problem, and R1 should not be read as if it were.**
Holding all twelve muscles at the constant vector 0.5 — the stance pose — brings the trunk to 0.25 m
by tick 9 and leaves it at 0.261 m tilted 0.6 degrees, and the published RL policy of Section 4.4,
driven on the same body, also recovers from the prone posture. What row 1 measures is therefore the
*link*: the body-state and proprioceptive regions reporting "lying down" are wired, in the table, to
the cells that emit that vector. Remove the table and the same body, the same muscles and the same
seed stay on the floor (row 4).
**The control that isolates the instinct table itself.** Row 4 is a within-seed ablation of the
one thing the claims of Section 1.1 are about. Rows 1-3 leave open the possibility that the
random background connectivity is doing the work and the measured table is decoration: the body,
the sensors, the seed and the 100 sparse synapses per cell are all still present in row 4, and
only the 585 rules are removed. On the same five seeds, with nothing else changed, not one brain
stands up (every run ends at exactly 0.184 m -- the value the no-sensation control settles at,
i.e. what the body does when nothing drives it) and not one approaches the red region (median
displacement -0.12 m; the body drifts backwards while sagging). The behaviour in rows 1-2 is
carried by the written wiring and not by the random background.

### R2. The behaviour is computed by a specific population (Figure 3)

Rules in the instinct table are currents, not conditions. We exploited that to run an
intervention: keep the visual input fixed (a red region straight ahead, identical image every
tick), and silence the entire prefrontal population by forcing it to zero *after* it is computed
and before the cortical step. Nothing else changes.

| condition | ticks with the walking action lit (of 200) | first fire | path length | prefrontal cells active (mean) |
|---|---|---|---|---|
| red, prefrontal intact | **199/200** (5/5) | tick 2 | 0.93-2.31 m | 2782-2785 |
| red, prefrontal silenced | **0/200** (5/5) | never | 0.21-1.15 m | 0 |
| black screen, prefrontal intact | 0/200 (5/5) | never | 0.55 m | 368 |
| blue screen, prefrontal intact | **183-194/200** (5/5) | ticks 7-18 | 0.76-2.25 m | 2872-2880 |
| blue screen, prefrontal recomputed each tick | 0/200 (5/5) | never | 0.18-1.48 m | 1444-1445 |

(logs: `日志_前额叶多种子_新A.log`, `日志_前额叶多种子_新B.log`; script
`实验_前额叶_多种子.py`. Row 5 is the control in which the prefrontal code is *not* allowed to
persist -- it is recomputed from its input on every tick, which is what this table measured
before the population was given recurrent wiring.)

Two things are worth separating here. First, the ablation is *specific*: silencing a population
of 53,200 cells removes a specific visually guided behaviour and leaves the sensory response
itself intact — a blue screen still drives 1444 prefrontal cells when the code is recomputed
every tick, and 2872 when it persists (rows 5 and 4) — while the behaviour that needs that
population's contribution is gone on 5/5 seeds.

Second, the mechanism is arithmetic rather than logical. The rule "red ahead -> walk" is written
as **two** rules: the visual region contributes 0.80 and the prefrontal region contributes 0.85
to the same motor-memory cells. The firing threshold is 1.0, so neither half is sufficient alone
and the sum is. Removing either half abolishes the behaviour. We deliberately do **not** call
this an AND gate: the motor cell is a linear summing junction, the contributions from other
regions can be present or absent depending on the situation, and the same behaviour could in
another context be driven by one region alone. The ablation shows that *this* behaviour, in
*this* situation, requires the prefrontal contribution; it does not show that behaviour in
general is gated by the prefrontal cortex.

The path length in the two rows that read 0/200 is the body sagging and being dragged by
whatever posture reflex is still running; the walking action itself never lights. Row 5 is the
clean control for row 4: same stimulus, same weights, one difference — the prefrontal code is
recomputed on every tick instead of being allowed to persist — and the behaviour disappears.

**The same stimulus, with and without persistence.** Per tick a blue screen delivers 0.73 of the
1.0 that the walk needs, so the arithmetic of the next paragraph is unchanged. But a deficit that
is never repaired inside one tick can still be repaired across ticks, and with persistence it is:
the walk starts at tick 7 to 18 and then runs on its own dynamics (R8). The boundary is still a
threshold; what persistence changes is whether it is crossed in *space* (how red the image is, on
this tick) or in *time* (how long the image has been there).

**The two halves, measured separately.** R2 shows that the behaviour needs both halves; it does
not show what each half contributes. We measured both, tick by tick, in the same closed loop. Each
row gives the mean fraction of the named block that is lit over 200 ticks, the drive that fraction
contributes, the sum, and the walking tick count.

| scene | `视觉:中有红` lit (of 1346) | `前额叶:看着红` lit (of 1360) | drive from the visual half (0.80 x) | from the prefrontal half (0.85 x) | sum | walking |
|---|---|---|---|---|---|---|
| *prefrontal recomputed each tick (the R1-R8 configuration)* | | | | | | |
| red ahead | **1.00** | **1.00** | 0.80 | 0.85 | **1.65** | 199/200 (5/5 seeds) |
| black | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/200 (5/5) |
| blue | 0.29 | 0.58 | 0.23 | 0.50 | 0.73 | 0/200 (5/5) |
| green | 0.30 | 0.62 | 0.24 | 0.53 | 0.77 | 0/200 (5/5) |
| *with the prefrontal loops (the R9 configuration)* | | | | | | |
| red ahead | 1.00 | 1.00 | 0.80 | 0.85 | **1.65** | 199/200 (5/5) |
| black | 0.00 | 0.11 | 0.00 | 0.09 | 0.09 | 0/200 (5/5) |
| blue | 0.29 | **0.99** | 0.23 | **0.84** | **1.07** | **194, 194, 192, 194, 183 / 200 (5/5)** |
| green | 0.30 | **0.99** | 0.24 | **0.84** | **1.08** | **193, 193, 194, 195, 0 / 200 (4/5)** |

(logs: `日志_两半_无环.log` for the first block and `日志_两半_新.log` for the second, same script,
5 seeds each, both re-run on the build of this paper; `实验_两半各亮多少.py`; the firing threshold
is 1.0)

The visual half is the same in both blocks, because it is the front end: near-binary (1.00 for red,
0.29-0.30 for blue and green, 0.00 for black). The prefrontal half is what changes. Recomputed
every tick it is a partial responder (0.58 for blue, 0.62 for green); persistent, it saturates at
0.99 for any colour that lights it at all, the sum rises from 0.73 to 1.07 and from 0.77 to 1.08,
and blue and green now start the walk. Black, whose two halves still sum to 0.09, still does not
(0/200 on 5/5).

Two things follow. First, in the configuration of R1-R8 both halves are colour-selective but not
equally so: the visual half is close to binary for this colour (1.00 for red against 0.29 for
blue), while the prefrontal half is a partial responder (1.00 against 0.58). Neither half reaches
1.0 by itself for any colour in this table; only the sum does, and only for the colour the instinct
was calibrated on. That is the mechanism of the approach instinct in full, and it is arithmetic on
the somata of the action's first time cell, not a condition in code. It is also the mechanism that
persistence breaks: a population that keeps what it has lit stops being colour-selective, and the
threshold that discriminated between colours becomes a threshold in time instead (R9).

Second, this is the quantitative reason the ablation above is specific rather than catastrophic. A
blue screen still drives 1,444 prefrontal cells; but for *this* rule the two halves deliver 0.73
on that tick, which is short of 1.0. The system is not deciding "not red" -- it is falling short
of a threshold, and how long it stays short is what decides the behaviour.

### R3. The repertoire grows by adding innate structure, not by retraining (Figure 4)

If behaviour is built from innate wiring, then adding wiring should add capability — and,
critically, it should not cost the capabilities that are already there. We tested this by
building five nested instinct tables **on the same random background connectivity** (same build
seed, so the only thing that differs between brains is the table):

| stage | what is added | rules in the staged brain |
|---|---|---|
| A | motor repertoire only | 221 |
| B | + balance / righting premises | 227 |
| C | + visually guided approach | 233 |
| D | + auditory reactions | 236 |
| E | + gaze (static, per-column, motion-onset) | 581 |

The four derived reflex rules (Methods 5.3) are loaded from a separate file and are **not** part of
the staged subsets, so the staged brains are built from the written table alone; the counts above
are the counts printed by the script, not the counts in the grouping table of Section 2.2.

At every stage we re-measure **all** behaviours, not only the new one. The measures are the
protocols of 5.7: standing up from prone (final trunk height), walking toward a red region (the
walking action's time cells must actually fire; distance alone is not accepted, since a
collapsing body also travels), reacting to a broadband sound, being stopped by the narrow-band
tone *while* red is present, and holding the gaze on a small moving ball.

| stage | rules | stand up | walk to red | react to sound | stopped by the tone | gaze tracking |
|---|---|---|---|---|---|---|
| A motor repertoire only | 221 | **0/3** | 0/3 | 0/3 | 0/3 | 0/3 |
| B + balance / righting | 227 | **3/3** | 0/3 | 0/3 | 0/3 | 0/3 |
| C + visually guided approach | 233 | 3/3 | **3/3** | 0/3 | 0/3 | 0/3 |
| D + auditory reactions | 236 | 3/3 | 3/3 | **3/3** | 0/3 | 0/3 |
| E + gaze | 581 | 3/3 | 3/3 | 3/3 | **0/3** | **3/3** |

(log: `日志_分阶段_新.log`; script `实验_本能分阶段.py`; three build seeds per stage, and all five
behaviours re-measured at every stage)

The medians behind the matrix: standing up leaves the trunk at 0.184 m at stage A and 0.261 m from
stage B onwards; the walking action is lit on 0 of 200 ticks at stages A and B and on 199 of 200
from stage C; the sound response is 0 ticks at A-C and 100 ticks from D; gaze error is 15.3 degrees
(unwired) at A-D and 5.6 degrees at E. The no-sensation control collapses to 0.184 m at every
stage, as it should.

Three things in this table are worth stating explicitly.

*The repertoire is not the behaviour.* With the motor repertoire alone (stage A) the brain
contains the standing-up action but never performs it: the body is placed upright and simply
falls over, exactly as it does when all sensation is removed. A capability that exists in the
wiring is not a capability until something lights it. This is the cleanest statement in the
paper of why a brain cannot start from silence.

*Each addition buys exactly the capability it encodes and nothing else.* Stages C, D and E add
6, 3 and 345 rules respectively; each addition turns on its own behaviour without turning on
any other.

*Additions do not damage the earlier repertoire.* Every capability that is present at one stage is
still present at the next. Standing up, once it appears at B, passes 3/3 at C, D and E; walking,
once it appears at C, passes 3/3 at D and E; the sound response, once it appears at D, passes 3/3 at
E. Adding the 345 gaze rules does not disturb the two rules that drive walking, and adding the six
approach rules does not disturb the rules that stand the body up. The stages are nested in
behaviour, not only in the table.

**One cell of the matrix fails.** "Stopped by the tone" is 0/3 at every stage, including E, where
all 585 rules are present. The tone *does* stop the walking: with red alone the walking action is
lit on 199 of 200 ticks, and with red and the tone together it is lit on 15 of 200 (median), and the
body travels 0.11 m instead of 1.83 m. Fifteen ticks is one further cycle of the fifteen-tick walk,
so the action terminates within one cycle of the tone arriving. It nevertheless fails the criterion
we fixed in advance (at most 10 of 200 ticks) and we report it as a failure rather than move the
threshold afterwards..

### R4. What sets the boundary of recognition, and how far local plasticity moves it (Figure 5)

A recogniser that is fixed -- a fixed set of weights, a threshold, a match -- has a fixed
tolerance. A synapse that follows a local correlation rule has no tolerance, only a rate: it
follows whatever its two cells have been doing. We tested whether that difference has behavioural
consequences. The experiment produced a result we did not predict, and the result is more
informative than the prediction was.

**Protocol.** The hue of a full-field colour starts at 0 degrees (red, the colour the approach
instinct was calibrated on) and is then stepped by a fixed increment of 3, 6 or 12 degrees. After
each step the walking action is **stopped** (the cortical state is cleared) and the current colour
is presented afresh for 8 ticks; if the action re-lights on at least 6 of those 8 ticks the colour
counts as recognised and the sweep continues, otherwise the sweep stops and we record the hue at
which it stopped. Restarting the action at every step is what makes this a test of recognition: a
walking action that has been lit chains its time cells into each other and runs by its own
dynamics (R8), so "still walking" would measure inertia.

**The boundary, before and after.** The instinct that starts the walk is the sum of two cell
populations (R2), so which colours start it ought to follow from those two populations. We measured
both, and then measured, in the same **five** brains, whether each hue starts the walk -- once
before any drift, and once after a 3-degree sweep has run to its stopping point. The "before"
columns are the control for the "after" columns: the same brains, the same weights, the same
stimulus schedule, with nothing but that one sweep between them.

| hue | `视觉:中有红` lit (of 1346) | `前额叶:看着红` lit (of 1360) | visual half | prefrontal half | sum | seeds that start the walk, **before** the drift | **after** it |
|---|---|---|---|---|---|---|---|
| 0 deg (red) | 1346 | 1357 | 0.80 | 0.85 | 1.65 | 5/5 | 5/5 |
| 24 | 1192 | 1248 | 0.71 | 0.78 | 1.49 | 5/5 | 5/5 |
| 48 | 1149 | 1248 | 0.68 | 0.78 | 1.46 | 5/5 | 5/5 |
| 72 | 1038 | 1243 | 0.62 | 0.78 | 1.39 | 5/5 | 5/5 |
| **96** | 663 | 1051 | 0.39 | 0.66 | **1.05** | **4/5** | **5/5** |
| 120 (green) | 367 | 844 | 0.22 | 0.53 | 0.75 | 0/5 | 0/5 |
| 144 | 439 | 959 | 0.26 | 0.60 | 0.86 | 0/5 | 0/5 |
| **168** | 529 | 1106 | 0.31 | 0.69 | **1.01** | **4/5** | **5/5** |
| **180 (cyan)** | 565 | 1111 | 0.34 | 0.69 | **1.03** | **4/5** | **5/5** |
| 204 | 507 | 1070 | 0.30 | 0.67 | 0.97 | 0/5 | 0/5 |
| 240 (blue) | 337 | 794 | 0.20 | 0.50 | 0.70 | 0/5 | 0/5 |
| black screen | 0 | 3 | 0.00 | 0.00 | 0.00 | 0/5 | 0/5 |

(logs: `日志_色相红块_新.log` for the two populations (identical to the pre-loops
`旧日志_20260916_加环前/日志_色相红块.log`), `日志_渐变边界_无环.log` for the behaviour; scripts
`诊断_色相_红块.py`, `实验_渐变_边界.py`. R4, like R1-R3 and R5-R8, was measured with the prefrontal
non-persistent -- `前额叶不保留=1`, Methods 5.5.)

Two things are in this table, and they came apart.

**The boundary is where the sum crosses, and it is there in a brain that has never learned.** Every
hue whose two halves sum to 1.01 or more starts the walk on **4-5 of 5** brains; every hue whose
halves sum to 0.97 or less starts it on **0 of 5**. Nothing else predicts the column, and the pattern
is monotone in the sum and *not* monotone in hue: cyan at 180 degrees, the opposite of red, comes
closer to starting the walk (sum 1.03) than green at 120 degrees (sum 0.75), which is only half as
far around the circle. The column is also not something learning built. With the correlation rule of
2.3 switched off entirely -- no weight changes at all, before the probe or during it -- the same five
brains give the same column: 96, 168 and 180 degrees recognised on four of five, every hue at 0.97 or
below silent on all five (`日志_色相点亮_无环不学.log`, `诊断_色相点亮_前测.py` with `不学=1`).

**The drift moves one brain, and it is the brain that was sitting on the line.** The three hues that
change between the two columns are exactly the three whose sums lie inside 0.05 of the threshold: 96,
168 and 180 degrees, going from 4 of 5 brains to 5 of 5. The brain that changes is seed 20260918,
which sat at 0 of 8 ticks on all three of those hues before the drift and at 7 of 8 after it. Every
hue at 0.97 or less stays silent in both columns, and no hue moves in the other direction. The drift
does move the boundary, then -- in one brain of five, at the three hues that were within a twentieth
of the line.

**The sweep measures how far the rule reaches.** The protocol above also gives a single number per
run: how far the colour can drift before recognition stops.

| step size | plasticity | hue at which the sweep stopped (one value per seed, in seed order) | median |
|---|---|---|---|
| 3 deg | **on** | 96, 96, 96, 96, 96 | **96** |
| 3 deg | off | 96, 96, 96, 96, **72** | 96 |
| 6 deg | **on** | 102, 102, 102, 102, 102 | **102** |
| 6 deg | off | 102, 96, 102, 102, **72** | 102 |
| 12 deg | **on** | 96, 96, 96, 96, 96 | **96** |
| 12 deg | off | 96, 96, 96, 96, **72** | 96 |

(logs: `日志_渐变正式_新.log` for the 3-degree rows and `日志_渐变正式_6和12.log` for the rest, both
on the build of this paper; script `实验_渐变_正式协议.py`; the same non-persistent configuration as
the table above.)

The brain that moves at every step size is seed 20260918, and it is the one whose sums at 96, 168
and 180 degrees lie within 0.05 of the threshold. With the rule off it stops at 72 degrees; with the
rule on it reaches the top of the grid -- 96 at a step of 3 and of 12 degrees, 102 at 6 -- and the
tail probes of that run then start the walk at 180 degrees on 7 of 8 ticks, where with the rule off
they start it on 0 of 8. One further brain, seed 20260915, moves at the 6-degree step and not at the
other two, and the difference between those runs is the grid rather than the rule: the sum of the
two populations is not monotone in hue (it falls from 1.05 at 96 degrees to 0.75 at 120 and rises
again to 1.03 at 180), so the largest grid point below the boundary is 96 degrees at a step of 3 or
12 and 102 degrees at a step of 6, and whether a brain crosses at 102 depends on which side of the
line its sum at 102 degrees falls. Counting how many brains move across step sizes would be counting
grid points rather than brains.

**So the rule carries a marginal stimulus to the edge of its island, and no further.** On the build
of the main table, four brains of five already sit just above the line at those three hues, so a rule
that can push a marginal stimulus across the line has almost nothing left to push, and the effect
shows up in one brain of five at a step of 3 or 12 degrees and two at a step of 6. On the earlier build all five brains sat just *below* the line at the same three
hues, and there the same rule carried all five out to 96 degrees and across the gap to 168 and 180. In both builds the rule moves the boundary outward as far as the
edge of the visual island the colour belongs to -- 96 degrees, where the visual half of the sum
falls from 0.39 to 0.22 -- and in neither build does it cross the gap at 120-144 degrees, which
stays refused by every brain in every condition. The prediction we started from -- that a synapse
with no tolerance, only a rate, would widen recognition wherever its cells had been co-active -- is
therefore **half right, and the half that fails is the instructive one**: the rule does widen
recognition, but only over the island the sensory front end already formed, and only for stimuli
whose drive is already within a few hundredths of the line, so on the build we report it is visible
in one brain of five rather than in all five. Learning fills in a boundary; it does not draw one.

**Why we quote two builds of the system.** A hue whose two halves sum to 1.05 is 0.05 above the
threshold; a hue at 1.01 is 0.01 above it. Whether a hue that marginal ignites is decided by the
random background connectivity the brain happens to be born with, and by nothing we control. We know
this because we measured the same protocol twice, on two builds that differ by 200 cells (the
dopamine region of R9) and by the draw of that background connectivity. On the earlier build the
same five brains sat *below* the line at all three of those hues before any drift -- 0 of 5 at 96, at
168 and at 180 -- while with the correlation rule off their sweeps stopped between 48 and 72 degrees,
and with it on between 84 and 96: 96 degrees in 14 of the 15 runs, across step sizes of 3, 6 and 12
degrees. On the build of the main table,
four brains of five sit above the line at the same three hues and one does not, and only that one
moves. The measurement of the *sums* is stable across builds; which side of the line a 1.05 hue falls
on is a draw, and it is that draw which decides how much work the rule has left to do. We therefore
make the claim about the sums and about the reach of the rule rather than about how many brains will
show it: the boundary of a behaviour assembled from two populations is the value at which those
populations sum past threshold, and the local rule can carry a stimulus sitting within a few
hundredths of that value across the line and as far as the edge of its island, and no further.

**And the population it all rests on is not a concept.** The same table shows that the block the
calibration labelled "there is red in front of you" is lit on **42 percent** of its cells by a cyan
screen and on 25 percent by a blue one. What was calibrated is not the category *red*: it is
"these cells responded to this image and not to an empty one". This is the point of R5 measured on
colour instead of direction, and it is why the behaviour's boundary is nothing more than wherever
the sum of two populations crosses a threshold. Nothing in the system represents *red*.

The recorded value is the last hue the sweep still recognised, so it understates the tolerance by
at most one step.

### R5. Hierarchical compression has a measurable representational cost (Figure 6)

The visual hierarchy has four parameter layers, but the third and fourth are the same tensor:
the value returned by the forward pass *is* the last exposed layer. There are therefore three
distinguishable stages: layer 1 (uncompressed, 23,040 cells), layer 2, and layer 3 (the
compressed output, 23,040 cells, which is what the rest of the cortex receives as "vision").

We measured, for a ball subtending **0.31 of a visual cell** (radius 0.05 m at 3 m), which cells
in each layer newly respond relative to an empty scene.

| direction | ball's column | layer 1 | layer 2 | layer 3 (compressed) | input layer (thermometer code) |
|---|---|---|---|---|---|
| left 3 | 2 | 7 cells, columns **2** | 5 | 7 cells, columns 1-10 | 11 |
| left 2 | 4 | 5 cells, columns **3-4** | 5 | 4 cells, columns 4-12 | 11 |
| left 1 | 6 | 6 cells, columns **6** | 3 | 6 cells, columns 3-15 | 11 |
| centre | 8 | 11 cells, columns **8** | 3 | **0 cells** | 11 |
| right 1 | 9 | 10 cells, columns **9** | 1 | 1 cell, column 1 | 11 |
| right 2 | 12 | 7 cells, columns **12** | 0 | 4 cells, columns 3-14 | 11 |
| right 3 | 14 | 11 cells, columns **14** | 5 | 8 cells, columns 4-15 | 11 |

(log: `日志_前几层.log`; script `实验_前几层看得见小东西吗.py`)

Two things are visible. First, the uncompressed layer localises: every one of its responses lies
inside the column the ball is in. Second, the compressed layer does not: its responses are
scattered over as many as 13 columns, and for a ball at dead centre it produces **no new cells
at all** — the object is simply not there, as far as the compressed representation is concerned.
Note that the *input* layer (before any weights) sees the ball perfectly well in every case
(11 cells, always) — so the loss happens inside the hierarchy, not in the image.

We then quantified the selectivity of the direction names themselves, using the same calibration
procedure that the instinct table uses.

| layer | cells lit by an empty scene | columns with a direction name | median cross-column sharing | worst columns |
|---|---|---|---|---|
| 1 | 7293 | 14 / 16 | **0.00** | all columns 0.00 |
| 2 | 5054 | 13 / 16 | **0.00** | all columns 0.00 |
| 3 (compressed) | 4030 | 13 / 16 | **0.14** | 0.75, 0.67, 0.50, 0.33, 0.25 |

"Cross-column sharing" for column c is the fraction of the cells in column c's name that also
appear in some other column's name (0 = the name is exclusive to that column, 1 = every other
column also lights it). Layers 1 and 2 are perfectly exclusive. In the compressed layer the
names overlap by 14% at the median and by up to 75%, and one column (the centre) has no name at
all. (log: `日志_方位分辨力.log`; script `实验_方位分辨力_按层.py`)

**We also tried to show the behavioural consequence and it did not come out as predicted.**
The prediction was that a gaze reflex re-wired to the compressed layer would stop working. We
re-calibrated the same per-column names at each layer, wired the identical reflex to each in
turn, and placed the same small ball in all 16 columns:

| reflex wired to | columns where the layer has a name for that direction | columns the eye turns toward |
|---|---|---|
| nothing (control) | 0 / 16 | 0 / 16 |
| layer 1 | 14 / 16 | 10 / 16 |
| layer 2 | 13 / 16 | 9 / 16 |
| layer 3 (compressed) | 13 / 16 | 10 / 16 |

(log: `日志_分层.log`; script `实验_追踪靠的是哪一层.py`)

The compressed layer still supports saccades. The reason is that a "blurred" name is still a
*selective* name: the cells that respond to a ball in column c are scattered in space, but they
still respond preferentially to column c, and in this reflex it is the identity of the active
name that matters, not where its cells sit. So the cost of compression in this system is
representational precision, not the loss of the function. We report the prediction failing
rather than dropping the experiment, because it defines the boundary of the claim: *the
compressed layer is not the answer, but neither is it useless.* What the measurement does
establish is the thing the framing needs — that the amount of information about the outside
world carried by the last layer of a hierarchy is strictly smaller than what the first layer
carries, and that "read the answer off the last layer" would therefore be reading the wrong
place.
### R6. Gaze tracking is emergent, not written down (Figure 7)

Nothing in the instinct table refers to movement, velocity, or tracking. The gaze behaviour is
assembled from two groups of rules that only refer to *where something is right now*:

1. **14 static rules**: "column c contains something -> pull the eye muscle that turns toward
   column c" (strength 2.0). These keep pulling as long as the object is off-centre, which is
   what makes the eye *hold* an object. The two columns straddling dead centre are deliberately
   left unwired, so a centred object produces no pull and the eye settles instead of oscillating.
2. **322 motion-onset rules**: for each column,
   `column c -> trace_c` (arrives one tick late, strength 2.0),
   `column c -> change_c` (+1.2) and `trace_c -> change_c` (-1.2, implemented with an
   inhibitory cell), so that `change_c` fires only if column c is active *now but was not one
   tick ago* — i.e. only on the onset of something. `change_c` then drives a chain of ten relay
   cells, and every relay cell drives the corresponding eye muscle (1.5).

The relay chain exists because the eye is heavily damped: one tick of maximum force moves the
eye about 0.1 degrees, so a single-tick onset event would be invisible in the eye's motion. The
relay stretches the onset into a ten-tick (0.2 s) pull. Functionally, the onset path is the
"something moved — look at it" signal and the static path is the "keep looking at it" signal.

Measured behaviour:

| scene | mean angular distance between the gaze centre and the object | eye at the end |
|---|---|---|
| 0.05 m ball crossing the field from -25 deg at 0.5 m/s | **5.6 deg** (max 29.2 while the object is still far off) | +24.6 deg |
| 0.05 m ball suddenly appearing at +20 deg, 2 s in | **4.8 deg** | +21.4 deg |
| empty scene for the whole trial | **0.0 deg — the eye does not move at all** | +0.0 deg |

(logs: `日志_眼睛跟随.log`, which is re-run from `实验_眼睛跟随.py` every time Figure 7 is
generated, so the numbers above are from the build of this paper; the same run written tick by tick
is in `日志_眼睛_空场.log`, `日志_眼睛_突然出现.log` and `日志_模糊随机2.log`)

The third row is the important control: with nothing in the field the eye is perfectly still,
so rows 1 and 2 are not an artefact of a jittery or spontaneously drifting eye. Rows 1 and 2
also show the transformation that the onset path performs — a sequence of very weak,
time-extended signals (the object is in column c, then column c+1, then c+2 ...) is converted
into a single strong signal at one moment.

### R7. A shifted innate map produces ordered behaviour with a systematic bias (Figure 8)

"Blurred innate wiring" is hard to test by randomising connections because randomisation
destroys direction information as well as precision. We therefore used a cleaner manipulation:
take every direction-bearing source name in the instinct table (`细_列c`, and the
`trace_/change_/relay k_` names derived from them) and shift it by a whole number of columns,
leaving everything else untouched. This is what "the inherited map is roughly right but not
exactly right" looks like.

Ball placed straight ahead, 4 s, eye free:

| shift | rules changed | walking still works | final eye angle | where the object ends up relative to gaze centre |
|---|---|---|---|---|
| -3 columns (-18.8 deg) | 222 | 7/8 ticks | +16.4 | -16.4 |
| -2 columns (-12.5 deg) | 246 | 7/8 ticks | +10.7 | -10.7 |
| -1 column (-6.2 deg) | 291 | 7/8 ticks | +3.6 | -3.6 |
| 0 (baseline) | 0 | 7/8 ticks | +0.0 | +0.0 |
| +1 | 291 | 7/8 ticks | +0.0 | +0.0 |
| +2 | 246 | 7/8 ticks | -6.4 | +6.4 |
| +3 | 222 | 7/8 ticks | -12.9 | +12.9 |

(log: `日志_整体偏移_正前方_新.log`, re-run on the current code in the non-persistent
configuration, reproducing every entry of this table; script `实验_本能表整体偏移_正前方.py`)

Outside the plateau, the resting point moves by one cell for every cell of imposed shift —
6.4 degrees, within measurement noise of the 6.25-degree cell width — in both directions. The
plateau at shifts of 0 and +1 is a consequence of the dead zone being two cells wide: while the
object is inside the (shifted) dead zone it receives no pull, so the eye does not move.
Throughout, the behaviour remains ordered: the eye turns, stops, and leaves the object
somewhere in the field; what changes is *where*. With the same manipulation
but the object placed 25 degrees off to the side, the same linear relationship holds for
positive shifts while negative shifts drive the eye into its 30-degree mechanical stop
(`日志_整体偏移2.log`).

The same idea, done the destructive way, gives a dose-response curve for how much of the innate
wiring a behaviour can lose. We re-wired a fraction p of the 585 rules, either to a **random** other
name in the same region (which destroys the direction) or to the name **one column over** (which
keeps the direction but blurs it). Within one draw the substitutions are nested -- a larger p
replaces everything a smaller p replaced, plus more -- and each condition was drawn three times
independently, so the score is how many of three draws still work.

| re-wired | walk to red | react to sound | gaze tracking |
|---|---|---|---|
| none (baseline) | 1/1 | 1/1 | 1/1 |
| random, 10% | 3/3 | 3/3 | 3/3 |
| random, 25% | 3/3 | 3/3 | **0/3** |
| random, 50% | **0/3** | 3/3 | 1/3 |
| random, 75% | 1/3 | 3/3 | **0/3** |
| random, 100% | 0/3 | 3/3 | 0/3 |
| one column over, 25% | 3/3 | 3/3 | 1/3 |
| one column over, 50% | 0/3 | 3/3 | 1/3 |
| one column over, 75% | 1/3 | 3/3 | 1/3 |
| one column over, 100% | 0/3 | 3/3 | 0/3 |

(log: `日志_存活曲线_新.log`; script `实验_模糊化_存活曲线.py`; 3 independent draws per cell, and the
baseline was run once because all draws coincide when nothing is replaced)

The pattern is not the one we expected, and it is more informative for it. The number of rules a
behaviour is built from does not predict its tolerance. Walking is driven by exactly two rules that
each contribute half the threshold, and it survives losing a quarter of the table; the gaze reflex
is driven by 336 rules and fails on all three draws once a quarter of the table has been re-wired,
because each of its rules names a specific visual column, and a destroyed column is a direction the
eye no longer knows at all. What survives everything is the auditory pathway -- a single rule whose
source is one named block, still working on 3/3 draws after *every* name in the auditory region has
been replaced by some other name in that same region. The broadband sound is a broad stimulus;
whatever the rule now points at, the stimulus lights it. Redundancy in the *stimulus* buys more
robustness than redundancy in the *rules*.

### R8. The stimulus starts the behaviour; the cortex runs it (Figure 9)

R2 showed that a named cortical population is necessary for a behaviour. It left open *where the
behaviour is* while it is running. We therefore measured the behaviour and the two cortical codes
that produce it, tick by tick, and then removed the stimulus.

A red region is present for the first half second and is then replaced by a black screen. Walking
is counted by the same criterion as everywhere in this paper (the walking action's time cells are
lit); the two codes are counted as the number of active cells inside the named blocks.

| condition, after t = 1.0 s | walking action lit | `视觉:中有红` (1346 cells) | `前额叶:看着红` (1360 cells) |
|---|---|---|---|
| *prefrontal recomputed each tick (the R1-R8 configuration)* | | | |
| red throughout | 495 / 495 | 1346 | 1360 |
| red removed at 0.5 s | **495 / 495** | **0** | **3** |
| red removed at 0.5 s, action's time cells cleared once at 1.0 s | **0 / 495** | 0 | 1 |
| never red, same clearing at 1.0 s | 0 / 495 | 0 | 2 |
| never red (control, no clearing) | 0 / 495 | 0 | 3 |
| *with the prefrontal loops (the R9 configuration)* | | | |
| red throughout | 495 / 495 | 1346 | 1360 |
| red removed at 0.5 s | **495 / 495** | **0** | **1360** |
| red removed at 0.5 s, action's time cells cleared once at 1.0 s | **0 / 495** | 0 | 1360 |
| never red, same clearing at 1.0 s | 0 / 495 | 0 | 148 |
| never red (control, no clearing) | 0 / 495 | 0 | 148 |

(5 seeds each; logs `日志_想还在吗_无环.log` and `日志_想还在吗_有环.log`, per-tick traces
`日志_想还在吗_轨迹_无环.csv` and `日志_想还在吗_轨迹_有环.csv`; script `实验_想还在吗.py`)

Row 2 makes the point in both configurations, and the difference between them is what R9 is about.
The stimulus is gone, the visual code is at zero in both -- and the walking action is still lit on
**every one of the 495 ticks**. What differs is the prefrontal code: 3 cells of 1360 when it is
recomputed every tick, and all **1360** when it persists. In the persistent configuration there is
literally a code for *red is there* running in the cortex while there is no red; in the
non-persistent one there is not, and the walking continues anyway, because what is running the
walk is the action's own trajectory.

Row 3 is where the two part company, and it is worth being careful about it. Clearing the action's
fifteen time cells once stops the walk for good in both configurations (0 of 495 ticks) -- but for
different reasons. Without persistence there is simply no drive left: the prefrontal code is at one
cell. With persistence the thought is still fully lit, all 1360 cells, and the walk still does not
come back -- because the drive that starts the walk is the sum of two halves, 0.80 from the visual
region and 0.85 from the prefrontal (R2), and with the visual half at zero the surviving 0.85 is
below the threshold of 1.0. The persistent code holds the thought; it cannot on its own re-start
the action.

The last two rows of the second block also show what persistence costs. With a black screen and no
red anywhere in the run, the block drifts up to 148 of its 1360 cells over three seconds --
activation that no sensory input is producing -- and it does not start the walk (0 of 495 ticks).
This is the same accumulation that R9 reports as the unfinished part of the system.

The reading we take is that after the first half second the movement is not being produced by the
stimulus at all. The stimulus *selected* an action; the action is a trajectory through the
cortical population, and once selected it is the trajectory that produces the movement. This is
also why the recognition protocol of R4 stops the action and clears the cortex between probes:
"still doing it" measures the trajectory, and a trajectory runs itself.

**The converse intervention.** If behaviour is a trajectory through a recurrent population, then
removing the recurrence should remove the trajectories -- and leave alone anything the instinct
table wires directly from sense to muscle. We kept the instinct table exactly as it was and cleared
the cortex at the start of every tick, so that only the current sensory signals were injected: a
purely feed-forward arc, sense to cortex to muscle, built from the same wiring.

| behaviour | intact cortex (3 seeds) | recurrent step removed |
|---|---|---|
| stand up from prone, final trunk height | **0.261 m** (3/3) | **0.184 m** (0/3) |
| red ahead: ticks with the walking action lit, of 200 | **199 / 199 / 199** | **0 / 0 / 0** |
| small moving object: mean gaze error | 5.6 deg (3/3) | **5.6 deg (3/3)** |

(log: `日志_切断回响_有环.log`; the same numbers are obtained in either configuration -- the
control clears the whole sheet at the start of every tick, so persistence has nothing to act on;
script `实验_切断回响.py`)

The manipulation is not a blunt instrument, and the third row is how we know. The gaze reflex is
wired as a direct map from a visual column to an eye muscle; it is unchanged **to one decimal
place**, 5.6 degrees in every seed and in both conditions. Everything that consists of *doing an
action* disappears: the final height of the no-recurrence body, 0.184 m, is the same value that
R1's no-sensation control reaches, because the standing-up action never runs and the body does
what it does when nothing drives it. (Its forward displacement is likewise not a measure: the
no-recurrence bodies "travel" between -0.25 m and +1.08 m by sagging and being dragged.)

The dissociation is the point, and it is also the answer to the falsifier we wrote down in 4.5: a
same-wiring feed-forward controller does not reproduce the behaviours, but it does reproduce the
reflexes, which localises what the recurrent step is for. In this system, reflexes are wired and
actions are trajectories.

### R9. A reward-like signal acquires a new link between a sense and an innate action (Figure 10)

R1-R8 measure a repertoire that is present at birth. The claim this project is built on -- that
complex behaviour is assembled on top of innate behaviour rather than trained from nothing -- is a
claim about *development*, and the experiment that bears on it is one in which a capability appears
during the life of one brain. R9 is that experiment. It adds two things, and it measures what each
of them buys.

*(a) A 200-cell dopamine region.* It is silent in every other experiment in this paper. When it is
active, a connection is written *on the spot* between cells that were active on the previous tick
and cells that *became* active on this tick -- the same "who brought whom in" relation the slow
correlation rule of 2.3 uses, cashed immediately instead of after four co-activations. Three
constraints were each forced by a measurement. (i) Cells active on both ticks are excluded: with
them in, 60 teaching ticks wrote 1.15 million connections and the auditory block grew from 609
active cells to 2202 of 2749 -- the population collapsed onto itself. (ii) Each target receives at
most eight sources per reward and the drive is divided equally among them, exactly as the instinct
table divides a block's drive among its targets: without this, a target received 300 units where
the walk needs 0.85, and any cell that happened to light up once was written full and stayed lit.
(iii) The fast path is multiplied by the same per-region plasticity factor as the slow path
(0.02 into the motor regions, see 2.3): without it, 200 teaching ticks wrote 260,000 connections
into the motor cortex itself and the gait came apart -- three of five seeds fell after about a
metre.

*(b) Loops inside the prefrontal population.* Inside each named block of the prefrontal population
the cells are wired head-to-tail into a closed chain (6,281 synapses in five blocks) with the same
current as any other instinct, and the main loop *adds* the newly computed prefrontal code to the
code already running instead of replacing it, so that a named block stays lit after the stimulus
that lit it has stopped being present. The loops are written only in the prefrontal population: a
loop in the motor-memory chain would block the sequence. The reason for adding this is the framing
of 1.1 -- a code that cannot outlast its stimulus cannot be thought with. R9 measures separately
whether the loops are needed for *acquiring* an association (they are not) and for *acting* on one
(for standing up, they are).

**Protocol: four stages, one brain, nothing rebuilt in between.**

| stage | body and senses | dopamine | walking action lit |
|---|---|---|---|
| 1 birth | prone, black screen, broadband sound only, 30 ticks | off | **0 / 30** (5/5 seeds) |
| 2 innate | standing, red region ahead, no sound, 30 ticks | off | **30 / 30** (5/5) |
| 3 teaching | standing, red ahead *and* sound, 200 ticks | **on** | running |
| 4 test | cortical sheet cleared, prone again, black screen, **sound only**, 200 ticks | off | **199-200 / 200** (5/5) |

(logs: `日志_发育多种子_A.log`, `日志_发育多种子_B.log`; script
`实验_发育_声音叫走路_多种子.py`. The dopamine-dark control is `实验_发育_无奖励对照.py`
(`日志_无奖励对照.log`); the no-loops control is `实验_发育_无环对照.py` (`日志_无环对照.log`,
`日志_无环对照B.log`); the gait control is `诊断_天生步态稳不稳.py` (`日志_天生步态.log`).)

Stages 1 and 4 are the *same stimulus on the same brain*, before and after. In stage 1 the sound
lights its own block -- 18 to 34 of the 1097 cells of the named block `听觉:低频响` -- and nothing
else happens: the walking action is lit on none of the 30 ticks, and the body only settles onto the
floor (0.035-0.046 m). In stage 4 the walking action is lit on essentially every tick, and the
body, which has been placed on its belly, first stands up (trunk height 0.099 m to 0.327-0.391 m,
5/5 seeds) and then walks 1.00-2.43 m.

Teaching writes **490,659-552,186** new excitatory connections on top of the 308,809 innate ones,
and they land where the teaching happened: auditory 137,688-155,459, visual 74,402-90,584,
proprioceptive 47,393-61,491, prefrontal 10,827-10,911, motor 2,570-3,167, vestibular 2,133-4,897.
Nothing here is written by hand and there is no teacher signal; the only difference between stage 1
and stage 3 is that 200 cells were lit while the behaviour was running.

**The control the claim needs.** The same four stages, with the dopamine region left dark: teaching
writes **0 new connections**, and stage 4 stays at **0/200 ticks** of walking on 3/3 seeds -- the
body never stands up (final height 0.103-0.205 m, which is the prone height). It is the
reward-like signal, not the passage of time and not the sound, that does the writing.

**What each addition buys.** Repeating the same four stages with the prefrontal replaced every tick
-- no loops, no persistence -- still acquires the behaviour: the walking action is lit on 199-200 of
200 test ticks on 5/5 seeds and the body travels 1.13-2.43 m, with 640,698-716,731 connections
written. So **the association does not need persistence**. What persistence buys is the *standing
up*: with the loops, 5/5 brains stand up in stage 4; without them, 1/5 does (final heights of 0.155,
0.060, 0.061 and 0.155 m against 0.352 m). The reading we take is that the sound reaches the walking entry
either way, but acting on a thought about standing up requires the thought about standing up to
last longer than the tick that produced it.

**What persistence costs, and what we did not get right.** With the loops, the prefrontal
population is an accumulator with no decay: under a constant stimulus it grows from 1581 active
cells on the first tick to 2916 by tick 200, because each tick's code is added to the last and
nothing removes it. The same property shows up in behaviour: with the loops, a *blue* screen drives
the walk on 194, 194, 192, 194 and 183 of 200 ticks (first fire at ticks 7, 7, 9, 7 and 18), where
the same protocol without the loops gives 0/200 on five of five seeds -- the instantaneous boundary
of R4 becomes a boundary in *time* once the population integrates. And the accumulator over-drives
the innate locomotor loop: the innate walk toward red, with no learning and no sound at all,
topples after 66 and 68 ticks on seeds 20260915 and 20260916 with the loops, while the same two
brains walk 2.26 m and 2.51 m without falling when the prefrontal is replaced every tick. A
recurrent population needs a way to forget, and the loops we wrote have none: that, and not the
reward rule, is the unfinished part of this system.

**A knob for the accumulation, and what it buys.** The accumulation is not intrinsic to the loops:
what is missing is anything that subtracts. One global quantity supplies it -- let every prefrontal
cell receive an extra inhibitory current equal to a constant *k* times the number of prefrontal cells
active on this tick, delivered uniformly to that population and to nothing else. It is a knob, not a
written rule, and it adds no connection. Three seeds, 120 ticks, standing:

| *k* | prefrontal cells active, red, ticks 20 to 120 (three seeds) | blue | green | black screen | walk lit, red | walk lit, blue |
|---|---|---|---|---|---|---|
| 0 (no knob) | 2747 - 2851 | 2748 - 2849 | 2738 - 2839 | 47 - 441 | 119/120 | 119/120 |
| 0.0005 | 546, 540, 548 | 566, 509, 554 | 561, 567, 535 | 47 - 441 | 119/120 | 119/120, 0, 0 |
| 0.001 | 275, 271, 278 | 267, 258, 270 | 283, 275, 287 | 47 - 441 | 119/120 | 119/120, 0, 0 |
| 0.0015 | 30, 29, 30 | 24, 25, 24 | 24, 24, 24 | 47 - 441 | 119/120 | 119/120, 0, 0 |
| 0.004 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 47 - 441 | 119/120 | 119/120 |

(logs: `日志_前额叶自压_窗口.log`, `前额叶自压=0.001`; script `诊断_前额叶自压_窗口.py`.)

Three things follow. (i) The knob does what it is meant to do: at *k* = 0.001 the red code stops
growing -- 2747 rising to 2851 becomes 271 holding at 275 -- and the population is sparse in the
sense 1.1 asks for. (ii) It removes the cost that mattered. On the two seeds that topple under the
loops, E0 -- never taught, red *and* sound, no learning -- now travels 1.66 m and 2.65 m tilted 4
degrees, against topples at ticks 76 and 68 with the loops and no knob, and those are the same
values the no-loop configuration gives. (iii) It is not free and the window is narrow. At
*k* = 0.004 the prefrontal is dark for every stimulus, yet the walk is lit on 119 of 120 ticks: the
ignition happens before the population is suppressed, and afterwards the action runs itself (R8), so
what is left is a one-tick prefrontal event rather than a code that persists. The black screen is
untouched at every *k*, because with no stimulus in the image the population is small and a constant
times a small number is a small number. We report the knob as a bounded result, not as a solution:
it trades unbounded growth for a population that is either sparse-hundreds or dark, with little room
between.

**What fails in R9.** On 4 of 5 seeds the acquired behaviour does not stay upright: the body
topples 42-101 ticks into the test, after 1.00-1.18 m (the fifth seed walks 2.43 m with a
3-degree tilt). Two controls on the same taught brains rule out the reward-written connections and
locate the rest.

| seed | never taught: red *and* sound, no learning | taught: test, learning on | taught: test, learning off |
|---|---|---|---|
| 20260915 | **topples at tick 76** | topples at tick 61 | topples at tick 61 |
| 20260918 | walks 1.69 m, 2 degrees | **topples at tick 42** | walks 2.13 m, 2 degrees |

(log `日志_摔是谁干的2.log`; script `诊断_摔是谁干的2.py`. The instability is present one stage
earlier and is not confined to the test: with the loops, all five seeds also topple during the 200
*teaching* ticks -- at ticks 78, 154, 174, 154 and 175, log `日志_打折R9.log` -- so from the middle of
the teaching onwards the body is on the floor, while the association is still written. That the acquisition survives the body falling over is itself consistent with (a): what the
reward path writes is the co-activation of the auditory code and the action's cells, and those are
still running while the body is down.)

The two seeds fail for different reasons, and neither is the reward-written connections: on
20260915 the body topples under the *innate* repertoire, in a brain that has not been taught
anything; on 20260918 the innate repertoire is fine and what breaks the walk is the slow
correlation rule of 2.3 continuing to write while the behaviour runs -- the same brain, with the
same connections, walks 2.13 m without falling if learning is switched off at test time. What the
two have in common is measured in the next paragraph but one: in both cases the walk is perturbed
while it runs, and in neither case is the perturbation written into the weights that end on the
muscles.

**Which of the two additions owns which failure.** The same three conditions, on the same two
brains, with the prefrontal code recomputed every tick -- the configuration in which R1-R8 were
run -- separate them.

| seed | never taught: red *and* sound, no learning | taught: test, learning on | taught: test, learning off |
|---|---|---|---|
| 20260915 | 1.66 m, 4 degrees | 1.47 m, 6 degrees | 1.63 m, 3 degrees |
| 20260918 | 1.69 m, 2 degrees | **topples at tick 40** | 2.09 m, 2 degrees |

(log `日志_摔是谁干的3_无环.log`; script `诊断_摔是谁干的2.py --无环`. With the knob of the
previous paragraph set to *k* = 0.001 the same two seeds give 1.66 m and 2.65 m and do not topple,
log `日志_摔是谁干的4_自压001.log`.)

Seed 20260915 topples only when the prefrontal accumulates: it topples at tick 76 with the loops
in a brain that has never been taught, and it walks 1.66 m without them. Seed 20260918 topples in
both configurations as soon as learning is switched on during the test -- 42 ticks with the loops,
40 without -- and walks 2.09-2.13 m when it is switched off. The two failures therefore have two
different owners, the accumulating prefrontal code in one case and the slow correlation rule
continuing to write while the behaviour is expressed in the other, and neither is the
reward-written connection.

**Where the perturbation comes from.** The two controls above locate the failures but stop short of
the mechanism: what they establish is *when* the walk breaks, not what breaks it, and one of the two
owners they name turns out not to be supported. We measured the weight of everything that ends on
the motor cortex -- the pathway that literally pulls the muscles --
across a whole 200-tick test. It does not change: 265,436 excitatory synapses onto the 160 motor
cells at the start of the test and 265,436 at the end, carrying 494,436 units of weight both
times, a difference of zero, with no new synapse formed. Freezing that pathway outright -- the
same brain, the same test, with the plasticity factor onto the motor cells set to 0 instead of
0.02 -- moves the topple from tick 63 to tick 64. The motor weights are therefore not what breaks
the walk. That does not exonerate the slow correlation rule -- on 20260918, switching learning off at
test time still saves the walk, so something the rule writes is implicated -- but it is not written
here, into the pathway that ends on the muscles, and the sentence in an earlier draft of this
section which attributed the failure to the motor loop being rewritten while the behaviour ran is
not supported by these measurements.

The measurement that does explain it is smaller and much more specific. The walk is played by a
15-cell time chain in the motor-memory region, and on *every tick* of the test the excitatory
current arriving on that chain **from the prefrontal population** is **0.850** -- and 0.850 is
exactly the current that chain needs to fire, the same number that had to be divided among sources
when the reward path was calibrated in (a) above. That is not a coincidence: the instinct table was
calibrated so that the compressed sensory code for red and the prefrontal code for red *sum* past
the chain's threshold, and each of them contributes about half. The calibration is right for a code
that is recomputed from its input every tick -- the configuration of R1 to R8 -- because the
contribution is present only while the stimulus is. It is wrong for a code that persists. A rule
written to *trigger* an action becomes a *standing drive at the action's own threshold*; the chain
is no longer free-running, and the gait drifts until the body goes over. Two measurements make this
concrete. Without the loops the same brain, the same picture and the same sound produce an exactly
periodic walk: the twelve muscle forces at ticks 30, 60, 90 and 120 are identical to two decimals,
with the gait resetting on a 30-tick cycle. With the loops the forces acquire a slowly growing
offset on four of the twelve muscles -- 0.2 to 0.5 of full force by tick 60 -- and the difference in
the current reaching the motor cortex, on those ticks, comes entirely from the time chain (+138 to
+140 units) rather than from the prefrontal population, whose direct contribution to the motor
cortex is 0.00. The drive also does not come back through the return line: cutting the return line's
projection into the motor strip leaves the topple exactly where it was (tick 76 against 76), and
cutting the whole return line moves it to tick 63.

Scaling those 5,175 synapses at build time, with nothing else changed, tests that reading, and the
answer is that the effect is real and localised but the number is not a repair. At 0.75 and at 0.5
the seed that toppled under the innate repertoire is repaired: 20260915 holds the stand for 191 of
200 teaching ticks (against a topple at tick 78) and in the test it stands up and walks 187 of 200
ticks at 3 degrees (against a topple at 63, after which the body lay on the floor). Nothing else is
repaired. At both strengths three of the five seeds lose the walk altogether -- 0 of 200 test ticks
(at 0.5 one of them, 20260918, gives 0 of 30 innate ticks as well: the red region no longer starts
the action at all) -- and a fourth, 20260916, topples at tick 101 in the test in all three
configurations, so its topple was never this pathway. Five seeds, two strengths, one seed repaired:
moving the same half of the sum moves the failure around rather than removing it. We therefore do not
adopt the change; the table's strengths belong to the build on which every other number in this
paper was measured. We report the measurement because it localises the defect. What is missing is not
a weight rule that forgets; it is that a trigger written as a constant current stops being a trigger
the moment the code emitting it stops going away. Fixing that needs a rule that decays while the code
persists, which is a property of the rule and not of the learning.

(logs: `日志_冻死进运动.log`, `日志_诊断步态.log`, `日志_前额叶打走路链.log`,
`日志_返回线截断.log`, `日志_打折R9.log`; scripts `诊断_冻死进运动.py`, `诊断_前额叶打走路链.py`,
`诊断_返回线截断.py`, `试_打折R9_参数.py`.)

### R10. The reward can be delivered by a sensory afferent rather than by a switch (no figure)

R9 delivers its reward with a flag: the script sets `多巴胺亮 = True`. The population that does the
work is inside the system, but the *occasion* for reward is outside it. The framing of 1.1 says a
drive is one more innate circuit, so we wired one and asked whether the acquisition of R9 survives
it.

**A 210-cell touch region and one innate rule per site.** Seven skin sites -- head, back, belly and
the four legs -- get 30 cells each, appended after the inhibitory pool. The region sits in all
three "non-participating" lists of Section 2.1: it neither sends nor receives random connections and
has no internal recurrence, so its only route to the rest of the cortex is innate. The instinct
table gains seven fixed rules of the form `触觉:<site> -> 多巴胺:全部` at weight 2.0, 42,000
synapses in total. Touching one site at full pressure lights all 30 of its cells, and the dopamine
region is lit on the following tick -- 200 of its 200 cells -- and stays lit while the contact is
maintained; in the *resting* arm below it is lit on **200 of 200** teaching ticks. A brush along the
same site (10 of its 30 cells) lights the dopamine region on 0 of 200 cells, and no touch gives
0 of 200. Reward requires contact, and contact of a kind that engages a whole site.

**Protocol.** The four stages of R9, unchanged, on three brains, in three arms that differ only in
when -- if ever -- the hand is on the body: *resting* (contact on every teaching tick), *contingent*
(contact only on ticks on which the walking action was active, the way an animal is trained), and
*no touch*. The flag of R9 is permanently off in all three arms; there is no "give reward" statement
left in the script. Adding a region re-draws the whole background connectivity, so these are not
R9's brains, and the comparison of interest is within this run. Seeds are 20260914, 20260915,
20260916 in that order.

| arm | hand on body, teaching ticks | new excitatory connections | test: walking ticks | max trunk height at test |
|---|---|---|---|---|
| resting | 200 / 200 / 200 | 480,928 / 524,379 / 514,138 | 0/200, **200/200**, **200/200** | 0.115 / 0.155 / 0.058 m |
| contingent | 0 / 199 / 198 | 0 / 520,017 / 511,099 | 0/200, **200/200**, **200/200** | 0.115 / 0.155 / 0.061 m |
| no touch | 0 / 0 / 0 | 0 / 0 / 0 | 0/200, 0/200, 0/200 | 0.115 / 0.095 / 0.116 m |

(logs: `日志_仿生奖励_一直摸.log`, `日志_仿生奖励_声音叫走路.log`; scripts
`实验_发育_仿生奖励_三臂.py`, `实验_发育_仿生奖励_声音叫走路.py`, `触觉区_touch.py`.)

Birth is the same in every arm (0 of 90 ticks across the three brains). The no-touch arm writes
nothing and the sound still does nothing (0 of 600 test ticks), which reproduces R9's dopamine-dark
control in this build. The two arms that deliver contact acquire the link, and the connection counts
are in the same range as R9's 490,659-552,186 despite the different brains: the reward arrives as a
sensory afferent here, and the acquisition does not depend on how the reward is delivered.

**The instructive row is the first one.** Seed 20260914 in the *resting* arm was rewarded on every
one of the 200 teaching ticks and wrote 480,928 connections -- it is not that nothing happened -- and
it still fails at test (0/200). Its teaching stage differs from the other two in one respect: the
walking action was lit on **0 of 200** teaching ticks, where the red region alone lit it on 30 of 30:
in this brain adding the sound to the red suppresses the walk it is supposed to be paired with. The
reward therefore reinforced co-activity that did not include the behaviour, and it bought
nothing. A reward is not a teacher: it can only strengthen what
the brain actually did. The same brain in the *contingent* arm was never touched, for the same
reason -- a protocol that pays only for the behaviour cannot pay a brain that does not perform it.

**What fails in R10.** Both arms that acquire the link acquire it, and in both the body does not get
up at test: the walking action is lit on 200 of 200 ticks while the trunk stays between 0.058 and
0.155 m and the body topples at tick 41 to 95. Two of the three brains stand up in R9's build
(0.388 and 0.391 m for seeds 20260914 and 20260915). Three measurements, all in this build, locate
the difference, and none of them is the reward. (i) The touch pathway is silent at test: the hand is
not on the body, and nothing else enters the region. (ii) The region is not the cause: with learning
off and no teaching at all, the same three brains placed prone on a black screen stand up and hold
the posture -- maximum trunk height 0.261 m, the standing cells lit on 149 of 150 ticks -- which is
exactly what the unmodified build gives (`诊断_触觉版_天生站起来.py`). (iii) What differs is the
learned state, and the way it differs is that the acquired walk now wins against the righting chain.
That is the class of failure R9 already reports -- acquisition that does not stay upright -- and we
do not have a measurement that separates the two builds' failures from each other. We report the
endogenous reward as established, and the post-acquisition stability as open, in both builds.

### R11. What a wider cortex costs, and what widening does to behaviour (no figure)

Nothing in the architecture refers to the number of cells, but the number of cells here is set by
the eye: the visual field is divided into a mosaic, each mosaic cell carries three channels of ten
threshold pairs, and the prefrontal region is sized from the result. Widening the mosaic is
therefore the natural way to scale the system, and we measured what it costs. Two things had to be
true first.

**The eye and the mosaic have to be the same number.** The world is rendered analytically -- objects
are projected onto the mosaic grid by geometry, with no rasterisation -- so the mosaic *is* the
eye's resolution. Widening the front end while the world keeps drawing 24x16 would have produced
interpolated cells carrying no new information. Both now read the mosaic size from one place, so a
wider mosaic is finer vision: 6.25 x 3.1 degrees per cell at 24x16, and 0.52 x 1.6 degrees at
24x192.

**A software ceiling had to be removed rather than worked around.** The previous peak memory of the
entire system was 7,005 MB against a steady state under 1 GB that the same script now reports as 543 MB -- about 50 MB of that is the index the plasticity fix below adds -- and it was one line: the sensory front ends
built the "one strength per threshold pair" table densely, because the ring-wrapped pair offset spans
the whole ring. At the auditory width of 30,000 cells that is 30,000 x 29,999 doubles, or 7.2 GB --
larger than the cortex itself by an order of magnitude, and the only thing standing in the way of a
wider brain. The table is now filled in row blocks; because `rng.uniform(size=(n, m))` fills row by
row, the block version draws the same numbers in the same order, and the resulting weights are
bit-identical to the dense version for both front ends (checked by `验证_感觉网分块.py`, which
compares both front ends
against the dense version, and confirmed by R9 reproducing digit for digit on three seeds in the
unmodified configuration). The fix takes
that peak at 24x16 from 7,005 MB to 1,434 MB and changes no number in this paper.

| mosaic | degrees per cell (h x v) | cortical cells | excitatory / inhibitory synapses | build | tick*, no plasticity | tick*, with plasticity | resident | peak |
|---|---|---|---|---|---|---|---|---|
| 24x16 (this paper) | 6.25 x 3.13 | 143,796 | 11.6 M / 4.9 M | 10.7 s | 1 ms (970 Hz) | 2 ms | 543 MB | 1,434 MB |
| 24x32 | 3.13 x 3.13 | 214,836 | 16.2 M / 6.1 M | 14.0 s | 2 ms (653 Hz) | 3 ms | 700 MB | 1,890 MB |
| 24x48 | 2.08 x 3.13 | 285,876 | 20.9 M / 7.4 M | 19.3 s | 2 ms (458 Hz) | 5 ms | 855 MB | 2,599 MB |
| 24x64 | 1.56 x 3.13 | 356,916 | 25.5 M / 9.2 M | 24.1 s | 3 ms (363 Hz) | 5 ms | 1,030 MB | 3,054 MB |
| 24x96 | 1.04 x 3.13 | 498,996 | 34.7 M / 13.8 M | 37.2 s | 4 ms (268 Hz) | 9 ms | 1,411 MB | 4,474 MB |
| 24x128 | 0.78 x 3.13 | 641,076 | 43.9 M / 18.4 M | 47.9 s | 5 ms (187 Hz) | 11 ms | 1,791 MB | 5,383 MB |
| 24x192 | 0.52 x 3.13 | 925,236 | 62.4 M / 27.6 M | 68.3 s | 8 ms (126 Hz) | 16 ms | 2,555 MB | 7,200 MB |

(logs `规模试验_列16.log` ... `规模试验_列192.log`; script `规模试验_测量.py`, which builds a
network at the requested width and times it without running any behaviour, and which is run for all
seven widths by `量规模_七档.py`. One host, 16 GB, 16 logical cores. The pre-fix logs are kept
beside them as `规模试验_列*.log.bak_优化前`.)

\* These two columns are a *step benchmark*, not a live tick: the network is driven with a uniform
2 per cent of cells active and no behaviour is run, so what they time is the cost of one pass over
the synapse tables. The live cost of a tick, with the world rendering the mosaic and the body being
driven, is 58 ms with the rule disabled and 158 ms with it enabled at 24x16 -- see (ii) below and
Methods 5.5. The two benchmarks are not comparable to each other, and the live one is the one that
matters.

Three things follow, and the third of them is a warning about the first two. (i) Cells are cheap in
memory; plasticity is what costs, and what it costs depends on how much of the cortex is active
rather than on how many synapses exist. The previous implementation of the rule gathered over all
27.6 million inhibitory synapses on every tick to decide which of them to touch, when only the
synapses incident on a cell that is active can be touched at all. Indexing them by source and by
target once, at build time, makes the per-tick work proportional to the active population, and
against the benchmark above it takes the update on every tick from 554 ms to 16 ms at 925,236 cells.
The rule is unchanged: the new update is bit-identical to the old one, checked tick by tick over 100
ticks on all 4,890,945 inhibitory synapses at 24x16 (`验证_学抑制_逐拍对照.py`, maximum difference
0.0). (ii) That benchmark flatters the result, and the live number is the one that matters. It drives
the network with a uniform 2 percent of cells active, and after the first step that pattern does not
sustain itself, so what it times is the cost of the scan; a brain that is actually looking at
something holds about 6,000 of its 143,796 cells active on every tick, where the scan was not the
dominant cost in the first place. In the closed loop at 24x16, with the world rendering the mosaic
and the body being driven, a tick costs 58 ms with the rule disabled and **158 ms** with it enabled
(`诊断_整拍耗时.py`; the same live tick cost 215 ms before the fix, at identical activity, log
`日志_对照_学抑制_改前改后.log`). (iii) The ordering is therefore unchanged -- the body is faster
than the brain when the brain learns -- but the gap is a factor of eight rather than thirty, the
remaining cost is spread over the current calculation and both halves of the rule rather than
concentrated in one avoidable scan, and the honest statement of the ceiling is the live one: 158 ms
against the 20 ms the body allows, with the four-orders-of-magnitude gap to a human cortex on top of
it. The ceiling on this machine is still memory rather than time: the peak is 7,200 MB at 925,236
cells and the next doubling exceeds the host. This is a cost curve, not a scaling law, and the same
asymmetry is what R9 reports from the behavioural side, where the correlation rule changing while a
behaviour is expressed is what breaks it.

**The cheap end of that programme, run.** Nothing had to be re-derived by hand: the calibration
and the table are generated from the mosaic size, so widening to 24x32 produced 34 direction names
instead of 16 and 965 rules instead of 581, expanded to 357,274 innate excitatory synapses where
24x16 has 308,809. Calibration runs in 20 s and a brain builds in 51 s. We then re-ran the
behaviours that go through the eye, and the righting reflex, which does not.

| behaviour | 24x16 | 24x32 |
|---|---|---|
| righting: prone, black screen, no learning | 3/3 brains stand, 0.261 m, 149/150 ticks | **3/3, 0.261 m, 149/150** |
| walk to a full-field red | 5/5 seeds | 5/5 seeds |
| gaze, 0.30 m ball crossing the field | 5.5 deg mean | **4.3 deg** |
| gaze, 0.05 m ball appearing | 4.8 deg | **3.5 deg** |
| gaze, empty scene | 0.0 deg, eye does not move | 0.0 deg, eye does not move |
| hues that start the walk | 0-72: 5/5; 96, 168, 180: 4/5; 120, 144, 204, 240: 0/5 | 0: 5/5; 24: 4/5; 48, 72: 3/5; 96: 2/5; 168: 1/5; 180, 204, 240: 0/5 |

(logs `日志_天生站起来_列16.log`, `日志_天生站起来_列32.log`, `日志_放大走路边界_列16.log`,
`日志_放大走路边界_列32.log`, `日志_跟随_列16_*.log`, `日志_跟随_列32_*.log`,
`日志_放大_色相红块_列32.log`, `日志_走路电流_列16.log`, `日志_走路电流_列32.log`; scripts
`实验_放大_走路边界.py`, `实验_放大_色相红块.py`, `实验_眼睛跟随.py`,
`诊断_触觉版_天生站起来.py`, `诊断_走路点火_多种子.py`. The wider build is kept apart from the
narrow one with `AGI数据后缀`, which names its calibration file and its table; without it the wider
brain would overwrite the tables R1-R10 rest on.)

Four things follow.

**(i) What does not go through the eye is unchanged, to the digit.** Righting is driven by
proprioception, the vestibular names and body state, and none of those region widths move with the
mosaic. All three seeds stand in 0.261 m with the righting chain lit on 149 of 150 ticks -- the same
numbers R1 and R10 report -- so widening the eye did not perturb the rest of the brain. The random
background connectivity is redrawn, so this is not the same brain; it is the same *anatomy* doing
the same thing.

**(ii) A finer eye is a better eye.** Both gaze measures improve by about a quarter, and the eye
comes to rest closer to the object: 2.1 and 0.2 degrees off-axis at 24x32 against 3.2 and 2.3 at
24x16, with the empty-scene control still exactly 0.0. This is the first measurement in which a
wider cortex buys behaviour rather than only costing memory.

**(iii) The instinct strengths are calibrated to the build, and widening does not re-calibrate
them.** What a rule writes is the total current a source block delivers, spread over as many
synapses as the block has cells (Methods 2.1), and that design survives the widening intact: the
current reaching the walking head cell from its visual and prefrontal halves is 0.80 and 0.85, the
same two numbers R2 reports, and the total excitation onto that cell is 1.750 at 24x16 and 1.750 at
24x32, to three decimals, because the source blocks light fully at both widths. The inhibition onto that same cell is a different matter: half of the random
inhibitory wiring is scattered over the whole brain (Methods 2.1), so the inhibitory current a cell
receives grows with how much cortex is active, and the eye is what changed. It rises from 2.775 on
four of five seeds to a mean of 3.07, and its spread across seeds widens. Every threshold in the
table therefore loses margin, and the first place it shows is the boundary of recognition. At 24x16
that boundary is exactly where the two halves sum: everything at 1.01 or more starts the walk on 4-5
of 5 brains, everything at 0.97 or less on 0 of 5. At 24x32 the static sum stops predicting: 168,
180 and 204 degrees sum to 1.05, 1.10 and 1.05 and start the walk on 1, 0 and 0 of 5. What the finer
eye buys in precision it takes back in tolerance, and it gives back the false positive as well: cyan
at 180 degrees, the opposite of red, reached the line at 24x16 and does not at 24x32. The
consequence for scaling is that the instinct table is not width-free in effect even though it is in
form -- a wider brain needs its strengths re-tuned, or a rule that normalises against total
activity -- and that is a cost of the same kind as the plasticity bottleneck, not a defect of the
widening.

**(iv) The compressed layer still does not localise.** The hierarchy's loss is a property of the
compression, not of the mosaic. At 24x16 a ball one third of a cell wide is followed perfectly by
layer 1 and missed entirely by the compressed layer for one direction of seven (R5). At 24x32 the
same physical ball is two thirds of a cell wide, and the uncompressed layer is still perfect -- all
seven directions, every newly responding cell inside the ball's own column -- while the compressed
layer still scatters its few responses over up to 30 of 32 columns and still returns nothing at all
for two directions of seven. The front end's receptive-field sizes are written in ring positions,
not in degrees, so they shrink in angular terms as the mosaic widens. Whatever else scale buys, it
does not buy the compressed layer the ability to localise.

**What we did not measure.** No behaviour was run above 24x32, and the rows of the table above from
24x48 to 24x192 remain a cost curve. Of the experiments, the static boundary and the gaze reflex
were re-run at 24x32 and the drift of R4, the shifted map of R7, the stimulus-removal and
recurrence-removal of R8, and the acquisition of R9 and R10 were not. The re-tuning that (iii)
calls for has not been done either: we report the mis-calibration rather than a fix.

---
## 4. Discussion

### 4.1 What the results support

Taken together, the measurements support a specific and unglamorous claim: **a recurrent
cortical population with a fixed, pre-specified wiring diagram and a purely local correlation
rule is already enough to close a sensorimotor loop and to produce behaviour that can be
attributed cell population by cell population.** Nothing in the results requires a global
objective, a critic, a reward, or a backward pass.

The six claims of Section 1.1 are not equally supported, and the difference is the useful part.
Table 1 states, for each of them, the experiment that bears on it and what came out of that
experiment; the last row is not a claim but the question that R4 was built to answer.

**Table 1. What each claim rests on.**

| claim | test | outcome |
|---|---|---|
| **(P1)** a sensory hierarchy is a compressor whose output is a bus | R5: localisation and name overlap, measured layer by layer | **supported.** Layer 1 places every newly active cell inside the column a 0.31-cell ball occupies; layer 3 scatters the same ball over up to 13 columns and produces nothing at all for a ball at dead centre. Direction names are perfectly exclusive in layers 1-2 and 14 percent shared (worst column 75 percent) in layer 3, where one column is lost. |
| **(P2)** thought is co-activation of a cortical population | R8: remove the stimulus, then remove the recurrence | **supported.** Stimulus removed: both codes read zero (0 and 3 cells of 1346 and 1360) while the walking action stays lit on 495 of 495 ticks. Recurrence removed: the one wired reflex is unchanged to a decimal place and every action disappears. |
| **(P3)** the prefrontal population compresses a second time | R2: silence it; measure both halves of the drive. R9: give it loops and watch it over 200 ticks | **partly supported.** It supplies 0.85 of the 1.65 that crosses the threshold, and removing it abolishes the behaviour. Whether it accumulates its own state over time *is* now tested (R9): given loops it does, and with no mechanism for decay the accumulation is unbounded (1581 to 2916 active cells over 200 ticks), and it over-drives the motor loop. |
| **(P4)** instinct is the initial state of the sheet | R1 row 4: clear the instinct table, leave everything else untouched | **supported.** The same five brains then stand on 0/5 seeds, approach on 0/5, and settle at exactly the height of the no-sensation control. |
| **(P5)** a network cannot start from silence | R1 row 4, R3 (staged additions), R7 (blurred map) | **supported.** Random background connectivity with no written wiring produces no ordered behaviour, and each group of rules that is added buys exactly its own capability without disturbing the earlier ones. |
| **(P6)** acquisition during life needs a reward-like signal and no objective | R9: four stages, one brain, dopamine region on or off | **supported, with a boundary.** With the dopamine region lit during stage 3 the sound acquires the walk on 5/5 seeds (0/30 ticks before, 199-200/200 after, standing up included); with it dark, teaching writes 0 connections and the sound still does nothing (0/200, 3/3). The behaviour does not stay upright (4/5 seeds topple 42-101 ticks in), and controls on the same brains show that is the weight-growth rule, not the reward rule. |
| **(Q4)** is local plasticity doing anything a fixed recogniser is not? (question 4 of 1.2, not one of P1-P6) | R4: which hues start the walk, in the same brains, before and after a drift, with the rule on and off, and at three step sizes; the same protocol on two builds | **supported, with a bounded reach and a size that is not reproducible.** The boundary is where the sum of two cell populations crosses: every hue summing to 1.39 or more is accepted on 5/5 and every hue at 0.97 or less is rejected on 0/5. The three hues whose sums fall within 0.05 of the line (96, 168, 180) are decided by the random background connectivity each brain is born with, and the rule carries exactly those hues across the line: on the build of the main table that shows up in one brain of five at a 3- and a 12-degree step and two at 6 degrees, and in 5 of 5 on the earlier build, where every brain sat below the line. In neither build does any brain cross the gap at 120-144 degrees. What is reproducible is the reach of the rule -- to the edge of the island and no further -- not the number of brains in which it has anything to do. |

Four sub-claims are worth separating, because they have different evidential status -- one of
them, the last, we now report as a negative result.

*Ordered behaviour without an error signal.* This is the best-supported claim. It rests on R1
(five seeds, plus a no-sensation control that collapses), R2 (a population-specific ablation
that removes exactly one behaviour), and R3 (a nested sequence of stages in which each addition
buys a capability).

*A repertoire that is assembled rather than trained.* R3 supports the weaker version: adding
innate structure adds capability without destroying earlier capability, and without any
retraining. It does not show a repertoire that *develops*. We return to this in 4.3.

*Where a behaviour lives.* R8 is the result that most directly tests the framing the paper starts
from. It shows both directions: after the stimulus is removed the two codes that carry it read zero
while the action is still lit on every tick, and after the cortical state is removed the same
wiring still supports the one reflex but no action at all. The claim that thought is co-activation
of a cortical population is therefore not just a description of the diagram; removing the
co-activation removes exactly the behaviours that the diagram predicts it should, and leaves the
rest.

*Local plasticity as the reason that recognition can drift.* R4 supports a bounded version of this,
and the boundary on the claim is as informative as the claim. The drift moves the recognition of one
brain of five on the build of the main table, and all five on the earlier build, and the difference
between the two builds is entirely in the draw of random background connectivity -- so the part that
is reproducible is not how many brains move but how far: the rule carries a marginal hue across the
line and out to the edge of the visual island it belongs to, and never across the gap at 120-144
degrees. What it cannot do is create a boundary: with the correlation rule switched off entirely, in
a brain that has never drifted, the same five brains accept and refuse exactly the same hues, because
the boundary is the value at which two populations sum past threshold, and that value does not
depend on the rule. This is the result we would most like to see replicated, because it is the one
that speaks against reading a fixed recogniser and a correlation rule as interchangeable designs:
here they are distinguishable, and what the rule buys is the toleration of drift that is already
under way rather than a wider category to begin with.

### 4.2 What "the last layer" is for

The framing we started from says that a layered sensory network is a compressor: many
peripheral channels in, few cortical channels out, and the output is a bus rather than an
answer. R5 gives that framing a number. A ball a third of a cell across displaces the
uncompressed layer exactly where it should and the compressed layer almost nowhere — at dead
centre, nowhere at all. The compressed layer's direction names are 14% shared at the median and
up to 75% shared in the worst columns, where the first two layers are perfectly exclusive.

It is worth being explicit about the implication, because it cuts against a habit. If one
attaches a read-out to the last layer of such a hierarchy and trains it, one is training a
classifier on a *deliberately impoverished* signal. That is a reasonable engineering choice when
the read-out is the point. It is a poor model of a brain, in which the compressed signal is an
input to further recurrent processing rather than a terminal answer. R5's failure of our own
prediction — the reflex still works when wired to the compressed layer — makes the point sharper
rather than weaker: the compressed layer is not *useless*, it is *less informative*, and the
difference only becomes visible when one asks a question that needs the missing detail (where
exactly, and is it there at all).

### 4.3 What this does not show

**Development.** The headline claim of this project is that complex behaviour is built on top of
innate behaviour, in the way that an infant's reflexes precede its skills. R3 measures the
*dependence* -- stage by stage, in a fixed brain -- and R9 measures the *acquisition*: one brain,
four stages, one new link between a sound and an innate action, with a reward-like signal as the
only thing added to the system. What R9 does **not** show is a repertoire that develops. It is one
link, learned in one session, from a sense to an action the brain could already perform, with the
innate wiring and the body unchanged, and with us deciding when the reward is on. R10 takes one part of the experimenter out of that loop: the reward is now delivered by
contact with the body, through a sensory region, rather than by a flag in the script. Development in
the strong sense -- a repertoire that grows because of what the brain has already done -- remains a
design commitment, and we say so rather than dressing up R9 and R10 as more than they are.

**Scale.** Everything here runs on 143,796 neurons on one CPU core, and R11 measures what a wider
cortex costs on the same machine: up to 925,236 neurons and 62.4 million excitatory synapses, with
7,200 MB of peak memory as the ceiling. The cheap end of that programme -- 24x32 -- we have since run
with behaviour, and it is instructive in both directions: the righting reflex is unchanged to the
digit, gaze improves by about a quarter, and the colour boundary narrows, because the current a rule
delivers is fixed by construction while the inhibition a cell receives grows with the amount of
active cortex. Widening is therefore not mechanical. The table's strengths are calibrated to the
activity level of the build they were measured in, and a wider brain needs them re-tuned. Nothing
above 24x32 is anything but a cost curve, what happens at 10^8 or 10^10 neurons is not measured, and
the reader should not assume that the observed qualitative behaviour is scale-invariant.

**Authoring the innate wiring.** The instinct table is written by us. For the walking
repertoire it is distilled from a separately trained reinforcement-learning policy: we ran that
policy over a grid of velocity commands, recorded the joint trajectories, and wrote one rule per
trajectory. This is supervision, and it happens before the brain exists rather than during its
life. A model that had to *acquire* its motor repertoire from its own reflexes would be a
strictly stronger result, and it is not what we have. The repertoire is innate either way -- the
motor library is written into the instinct table before the brain runs, and hand-writing every
weight would be equivalent in kind; distillation is a convenience that saves writing 221 action
trajectories by hand, not a training procedure that the rest of the system depends on. We have since measured that source
directly (Section 4.4): its own cost is 99,483,648 physics steps, and its input contains no
exteroceptive channel at all — what it supplies is a gait, not the decision to use one.

**One body, one sensory layout.** The body is a quadruped, the eyes have one degree of
freedom each, and the visual field is 100 by 75 degrees. Nothing in the architecture is specific
to quadrupeds, but we have not demonstrated that.

### 4.4 Relation to other work

*Reinforcement learning* (Mnih et al., 2015; Hwangbo et al., 2019). Every behaviour here would be
straightforwardly learnable by RL. The point is not that RL cannot do it; the point is that in this
system the behaviour *is not learned at all* — it is a property of the initial wiring, and local
plasticity, on its own, did not change the behaviour we measured (R4). The relevant scientific
question is not "can RL match this" but "how much of animal behaviour is of this kind".

We did not train a baseline of our own; we measured a published one, because a published artefact
states its own cost. `diasAiMaster/unitree-go2-velocity-flat` is a PPO policy for this same Go2
model (flat ground, velocity-command task); its model card gives 8,192 parallel environments x 24
steps x 506 iterations = **99,483,648 physics steps**, 10x RTX A4000 for 18 minutes. We drove it on
**our** body at our muscle parameters (`对照_现成RL_驱动.py`, log `日志_对照_现成RL.log`): told
"1.2 m/s forward" it covers 3.71 m in 4 s, and 4.98 m told 1.6; told zero it covers 0.07 m; told
-1.6 it covers -2.90 m. It stays upright in every case, and it also gets up from the prone posture.
Its input layer is 45 numbers: body angular velocity (3), gravity direction (3), the three velocity
commands (3), joint offsets (12), joint velocities (12), and its own previous action (12). **Not one
of those 45 comes from an eye or an ear.** The artefact is therefore a very good translator from
three externally supplied numbers into a gait: 10^8 samples buy the lower half of the problem, and
the upper half — whether to move at all — is not in it. Supplying that half is itself a learning
problem, and not a small one: our own attempt (three seeds, PPO, R1's criterion) first crossed the
0.30 m line at 1.2-1.8 x 10^5 steps, and no seed ever reached 1.0 m, where the instinct table covers
the same thing with one row and no samples at all. That asymmetry is what "innate" means here.

*Predictive-coding and free-energy accounts* (Rao & Ballard, 1999; Friston, 2010). Those frameworks
also avoid an explicit loss, but they replace it with a generative model and an inference procedure.
Here there is no generative model and no inference: there is a fixed wiring diagram, a threshold,
and a correlation rule.

*Developmental robotics and intrinsic motivation* (Oudeyer & Kaplan, 2007). That literature builds
repertoires by staged curriculum learning. R3 is an unusually literal version of the idea — the
curriculum *is* the wiring — but we add nothing to the theory of curricula.

*Language models* (Brown et al., 2020; Kaplan et al., 2020). A separate build in this project runs
the same mechanism on a completely different substrate, and we report it here for what it says about
*this* system rather than about language. In that build the cells are single Unicode characters, the
text is 344,787,049 characters of modern Chinese, the only rule is the same local correlation rule,
and there is still no objective, no back-propagation and no language model anywhere in the loop.

  What that build shows, verified step by step, is how a population of this kind holds a state and
  moves it. Characters are active in **groups**, not one at a time; after the prompt ends the group
  does not fall silent, and it does not freeze either -- membership turns over from step to step
  while the population keeps its characteristic size, and every prompt eventually settles into an
  exact periodic orbit (periods 1 to 6 in the observed window). That is a dynamical observation and
  we do not read it as a measure of thought.

  That is the mechanism R8 and R9 need, and it is also why they need it. The prefrontal code for red
  outlives the red for the same reason the character group outlives the prompt: a recurrent
  population with a local correlation rule holds a state. But holding one is not free at any size.
  In the character build the same protocol with a small activity budget left 45 of 46 prompts
  completely dark -- the state died instead of circulating. The cortical sheet at 143,796 cells is
  in that regime: it does not spontaneously grow the loops that would let a thought persist, which
  is why R9 has to give the prefrontal population its loops explicitly. What separates the two
  builds is scale and connection density, not the rule.

  It is **not** evidence about language and we do not claim that it is: it behaves as a bounded
  associative memory, not a generative model -- lighting some characters makes other characters
  light, but it does not learn to answer a question or to follow an instruction -- and the paper
  that would be needed to make a language claim is not this one. The reports and the scripts of that
  build are in the repository under `语言规模实验_language_scaling_20260913/`.

*Attractors, reservoirs and large-scale models.* The behaviour of R8 is a trajectory through a
recurrent population, which places this system in a long tradition in which the computation is
the settling of a population rather than a feed-forward pass (Hopfield, 1982; Amit, 1989; Maass
et al., 2002). It shares the ambition of the large-scale reconstructions of cortex but none of
their method: those build a detailed model of a known circuit, whereas here the circuit is
trivially simple and the question is what the *initial condition* buys (Eliasmith et al., 2012;
Markram et al., 2015; Gewaltig & Diesmann, 2007).

### 4.5 What would falsify the framing

We list these because a framing that cannot fail is not doing work.

* If a system with the same innate wiring but *no* recurrent cortical step (i.e. the instincts
  driving the muscles directly) reproduced every behaviour, the claim that thought is cortical
  co-activation would be empty. We ran that experiment (R8): the feed-forward controller built
  from the same wiring reproduces the one wired reflex to a decimal place and reproduces no
  action at all.
* If the gradual recognition of R4 survived with plasticity switched off in every brain, the result
  would be attributable to the sensory front end and not to the cortical rule. We ran that control.
  In four of the five brains of the main build it did survive, and in the fifth it did not: that
  brain stops at 72 degrees with the rule off and at 96 with it on, and reaches the far island at
  168-180 only with the rule on. On the earlier build of the same system it did not survive in any
  of the five. The claim is therefore restricted to what both builds agree on -- the reach of the
  rule -- and not to how many brains show it.
* If a network with the same body, the same sensors and the same random background connectivity
  but *no* innate wiring produced ordered behaviour, the premise that a cortex cannot start from
  silence would be wrong. We ran that experiment (R1, row 4): on five seeds those brains stand on
  0/5, approach on 0/5, and settle at exactly the height the no-sensation control reaches.
* If the repertoire of an animal were fully described by its reflex arc -- sensor to muscle
  without a recurrent population in between -- then writing innate structure would be the same
  thing as writing a lookup table. R8 dissociates the two: with the recurrent cortical state
  removed, the wired sensor-to-muscle reflex (gaze) survives untouched while every behaviour
  that consists of *doing an action* disappears.

### 4.6 Outlook: what would be needed for a general architecture

We are asked, reasonably, whether a system of this kind could grow into something general. The
honest answer is that the results here demonstrate one ingredient and are silent about the
others. It is worth being precise about which ingredient, and about what the others would cost,
because the framing makes sharp predictions in both directions.

**What is demonstrated.** A born-wired recurrent cortical sheet, with no objective function and
with a plasticity rule that cannot see a target, already closes a sensorimotor loop, produces an
ordered repertoire, and lets a behaviour be attributed to a named population. The repertoire is
extended by *writing more innate structure*, and each extension leaves the earlier ones intact.

**The learning rule was never the bottleneck for the first problem.** The tempting reading of
these results is "learning is overrated". That is not our reading. Our reading is narrower and, we
think, more useful: the first problem an organism has is not to *learn* — it is to *do something
ordered*, and a random network with a correlation rule has nothing ordered to correlate. The
scarcity is in initial structure, not in optimisation.

**The falsifiable prediction for AI.** If that is right, then an agent given a large innate
behavioural repertoire before training should need dramatically less experience than an identical
agent that must discover its first ordered behaviour. Every current agent is of the second kind.
R3 is a one-step version of the claim inside this system; the cross-system version has not been
run.

**The falsifiable prediction for neuroscience.** If behaviour is produced by co-activation of a
cortical population rather than by a sensor-to-muscle map, then removing the recurrence should
remove behaviours whose expression is an action, while leaving wired reflexes intact. R8 is that
experiment, and it comes out as predicted.

**The bet, stated so that it can be lost.** The position this paper takes compresses into one
sentence: in an architecture of this kind, capability is bought with *structure*, not with
optimisation, and structure can be written down as fast as it can be understood. If that is
right, then the distance between the 585 rules used here and a general agent is not a missing
algorithm. It is more rules, more regions, a richer body and a much larger sheet -- expensive,
and mostly engineering. Every step of that programme has a worked example in this paper: add a
group of rules, get the capability (R3); blur the written map, get a biased but ordered behaviour
rather than a broken one (R7); clear the written map, get nothing (R1, row 4).

If the position is wrong, it is wrong in a way that is visible rather than vague. At some point,
adding structure must stop buying capability, and what is missing will be something the local
correlation rule cannot supply. The experiments that would expose that are ordinary ones: keep
adding groups of rules until a group that ought to work does not, or run a development experiment
in which a brain with an innate repertoire fails to acquire a capability that its structure
appears to allow. This paper fixes the starting point of that argument and shows that the first
three steps are real; it does not settle the outcome, and the reader should not let it.

**What is missing, concretely.** We list the ingredients in the order in which we think they
should be attacked, with the honest cost of each.

1. *A trigger that stays a trigger while the code that emits it persists.* This is now the first
   problem, and R9 is where it shows, but the first two explanations we offered for it were both
   wrong and the measured one is narrower. It is not the reward-written connections (R9 excludes
   them). It is not a weight rule that fails to weaken connections it does not need: the weight of
   everything ending on the motor cortex is unchanged across a 200-tick test (265,436 synapses,
   494,436 units, difference 0), and freezing that pathway entirely moves the topple by one tick
   (R9). It is one calibrated quantity. The innate rules from the prefrontal population to the
   15-cell walking chain deliver 0.850 on every tick, and 0.850 is exactly what the chain needs to
   fire, because the table was calibrated so that the sensory code and the prefrontal code *sum*
   past the threshold. While the prefrontal code is recomputed from its input every tick, that is a
   trigger; once the loops make the code persist it is a standing drive at the action's own
   threshold and the gait is no longer free-running. Scaling those 5,175 synapses by 0.75 and by 0.5 repairs the
   seed that toppled under the innate repertoire, costs the walk on three of five seeds and leaves a
   fourth toppling at the same tick as before, so the repair is not a coefficient: a rule that triggers an action has to be allowed to decay while the code that emits
   it does not. A homeostatic weight decay -- the standard fix for unbounded connection growth --
   was the obvious candidate and it does not help here; it hurts. A per-tick multiplicative decay on
   every plastic excitatory synapse is a knob in `皮层连接_cortex_links.py` (`全局衰减率`, off by
   default, so that no other number in this paper moves). With it on at 0.004, the seed that
   survives the test without it topples at tick 101; at 0.0005, an eighth of that dose, it topples
   at the same tick 101 and the second seed topples *earlier* than it does without the decay (tick
   45 against 75). The reason is in the design rather than in the dose: only 30% of the innate
   synapses are frozen (`固化比例`), so a flat decay eats the 70% that carry the calibrated current
   the instinct table exists to deliver -- and that calibration, not an unbounded weight, is what
   this failure is made of. A homeostatic term is therefore not a bolt-on: the table would have to
   be re-tuned to the equilibrium of growth and decay, exactly as R11 reports for a wider mosaic.
   (Logs `日志_全局衰减000.log`, `日志_全局衰减0005.log`, `日志_全局衰减004.log`; script
   `试_全局衰减_修R9.py`.)
   The other half of the same failure is R9's prefrontal accumulator, in which every
   tick's code is added to the last with nothing subtracting it; that one is answerable with a
   single global inhibitory current (R9's knob), and it is the one place in this paper where a
   structure we added is clearly not finished.
   The two ways a recurrent state can fail -- it dies, or it settles into a cycle it cannot leave --
   are both visible in the character build of 4.4, on the same rule and at a much larger scale,
   which is why we treat them as properties of the rule rather than accidents of this build.
2. *Development over lived time.* R9 acquires one link in one session, with us deciding when the
   reward is on. The header claim needs a repertoire that grows over lived time because of what the
   brain has already done, with no experimenter in the loop. Nothing in the framing says this is
   different in kind from R9, and R3 says the repertoire is additive; what is missing is a drive
   that is not satisfied by the first action it releases (item 4) together with the stable rule of
   item 1. R10 removes one part of the experimenter from the loop -- the reward arrives as an
   afferent rather than as a flag -- but the experimenter still decides when the contact happens.
3. *More sensory channels and modality-appropriate front ends.* Vision here is 24x16 macro-pixels
   with a thermometer code; audition is a spectrum. The architecture does not care what an input
   cell represents — a pixel level is an arbitrary choice of the designer — but a general system
   would need many more channels and would need the hierarchy to be deeper.
4. *A body with more than one thing to do.* The Go2 has 16 muscles and the repertoire is 221
   actions. Behaviour of the kind that is interesting — tool use, social interaction — needs a
   body with many more degrees of freedom and with hands. This is engineering, not theory, but it
   is not small.
5. *Drives.* Nothing here wants anything. In our framing a drive is not an objective function but
   one more innate circuit — a salient object lights a population that lights locomotion. The
   system already does exactly this for a red region; what it does not have is a drive that is
   *not* satisfied by the first action it releases.
6. *Scale.* R11 measures the cost of a wider cortex on one machine: 925,236 neurons and 62.4
   million excitatory synapses cost 68 s to build, 16 ms per tick with plasticity on in the step
   benchmark and 2,555 MB resident with a 7,200 MB peak, which is the ceiling of a 16 GB host. At 24x32, the one wider mosaic on which
   behaviour was run, the eye becomes measurably finer (gaze error 5.5 to 4.3 degrees) while the
   innate strengths, calibrated to the 24x16 activity level, lose margin and the boundary of
   recognition narrows. That is four orders of magnitude below a human cortex, and it is a cost curve
   rather than a scaling law: nothing above 24x32 was run. Nothing here is a scaling claim of the
   kind made for language models (Kaplan et al., 2020).

7. *A posture that is held, not a posture that is assumed.* The instinct table carries the body
   from prone to standing, and it has balance reflexes that push against a tilt once one has
   developed, but it contains no route that engages the standing pattern when the body is already
   upright and nothing else is happening. Placed directly into the standing pose, the same brain,
   the same seed and the same wiring emit no muscle current for the first ~30 ticks (0.6 s), tip
   past the point the reflexes can recover, and end on their side; having stood up by themselves,
   they hold 0.260 m for six seconds without moving a step. We tried the two obvious repairs --
   driving the standing motor pattern, and driving the standing memory entry, from the trunk-height
   population that is already lit at 0.261 m. The first leaves the behaviour unchanged; the second
   disrupts walking (median displacement over 4 s falls from 2.13 m to 0.70 m). The posture that
   survives a stand-up is therefore held through the action sequence rather than by a posture rule.
   We record the negative result because it runs against the intuition one has while writing the
   table: a posture here is not a state the body is put into, it is something the cortex has to
   keep doing, and it is not yet a single missing line.

**What we would not claim.** We do not claim that this architecture will reach general
intelligence, and we would distrust any paper that claimed it on this evidence. What we do claim
is that the *first* step — an ordered, attributable, extensible behavioural repertoire with no
objective function anywhere in the system — is not the step that is blocking, and that current
agents have been paying for it with an optimisation loop they may not need for that step.
---

## 5. Methods

### 5.1 Cortex

The cortex is a flat boolean array of 143,796 cells (`皮层连接_cortex_links.py`). One tick is 20 ms. The
unit is deliberately cruder than a conductance-based or spiking model (Izhikevich, 2003) and the
tick is a coarse stand-in for cortical dynamics (Buzsaki & Draguhn, 2004); both simplifications are
choices, and R8 is the experiment that asks how much they cost. The update is a single vectorised
operation over the whole array:

> incoming excitatory current is the sum of the weights of all synapses whose source fired on
> the previous tick, restricted to that cell's incoming synapses; the cell fires if this current
> is at least the firing threshold (1.0) and it is not being held down by an inhibitory cell.

Both the excitatory and the inhibitory part are pre-computed as edge lists
(`源, 目标, 权重, 固化`) so that a tick is a pair of `np.bincount` scatter-adds. Suppression is
implemented as a separate population: an inhibitory cell receives excitatory synapses from its
source and projects to its targets with weight 1.5; a cell that is the target of an active
inhibitory cell is held sub-threshold. There are 8,000 inhibitory cells and the instinct
expansion borrows them: a rule that must suppress something carries a negative number in the
*text file's* strength column, which is a notation in the table only and never a weight, and is
expanded into `source -> inhibitory cell -> target`. No synapse in the network is negative.

Background connectivity is sparse and random: 100 outgoing excitatory synapses per cell, 50% of
them to other regions, born with weights uniform in [0, 0.15], 30% of them frozen; 100 outgoing
inhibitory synapses per cell, half of them within a 24-cell radius. Three regions receive no
random excitation (motor, motor memory, visual motion) and one region emits none (visual
detail), so that adding the detail region could not perturb the global excitation/inhibition
balance. The background is generated from a seed; different seeds are different brains.

### 5.2 Regions and calibration

Names are not hand-written. Calibration (`本能工具_instincts.py: 标定`) lights a physical
condition — an image, a sound, a body posture — runs the relevant front end, and records which
output cells respond *relative to a reference condition*. For example a visual direction name
is defined by rendering a red ball in that direction minus rendering an empty scene. The eyes
were calibrated with a ball of radius 0.30 m; the per-column names used by the gaze reflex were
re-calibrated with a ball of radius 0.05 m, because a ball that fills a whole cell is visible to
every layer and would not distinguish them.

Region widths: visual 23,040; visual detail 23,040; auditory 30,000; prefrontal 53,200;
motor 160 (16 muscles x 10 cells); motor memory 3,576; proprioception 480; vestibular 120;
body state 60; visual motion 1,920; inhibitory 8,000.

### 5.3 Instinct table

`本能表.txt` is a text file, one rule per line:
`source_region:name -> target_region:name strength frozen`. A second, derived file
(`反射表.txt`) contributes 4 further rules computed from the same trained locomotion policy: in
a given tilt condition, hold a given shank above a force floor. The table currently holds 585
rules and expands to 308,809 excitatory synapses plus the inhibitory edges.

A rule whose source is a *motor memory* name (rather than a sensory name) is expanded
differently: the action's time cells are chained to each other, each time cell is wired to the
muscles that should be active at that instant, each time cell suppresses itself on the next tick
so that exactly one instant is active at a time, and the intermediate time cells suppress the
first one so that a still-present sensory drive cannot restart the action mid-execution. The
effect is that lighting a single cell commits the system to a whole action, and the action then
runs itself out of the cortex's own dynamics. 221 of the 585 rules are of this kind: 10 static
postures, 4 slow actions, 207 locomotion gaits.

### 5.4 Where the motor repertoire came from

The locomotion gaits were copied from a reinforcement-learning policy trained to make the same
Go2 model walk (`取动作_从训练好的模型.py`, `建动作库.py`), following the approach of Hwangbo et
al. (2019) and Rudin et al. (2022). We swept forward/lateral/turning
velocity commands on a grid, recorded the resulting joint trajectories, and stored each as one
action of 1-2 s. This is the only place in the system where a learned artefact is used, and it
is used *before* the brain is created: the trajectories become fixed synapses in the instinct
table and are never updated online. A reader who objects that this is supervision is right; our
claim is about what happens *after* birth, and we would rather state the provenance than hide it.
The source is the public checkpoint `diasAiMaster/unitree-go2-velocity-flat` (a 4.5 MB PPO actor,
`C:\go2_policy\vel\model_500.pt`), and its behaviour on our body is measured separately by
`对照_现成RL_驱动.py`.

What the repertoire is, and is not, is worth two numbers. Our forward row is 15 time cells, i.e. a
0.30 s cycle at 20 ms per tick, and the motor chain advances exactly one cell per tick, both
open-loop and under the whole brain (`诊断_步态相位.py`). Replayed from its head cell with no eye and
no prefrontal it carries the body 2.58 m in 4 s; under the whole brain 2.0 m, against R1's five-seed
median of 1.73 m with the rule running. The source policy covers 3.71 m in the same 4 s on the same
body. The row is a faithful copy of the source's *trajectory* and not of its *feedback*: what the
missing margin buys the source is a correction on every tick, which this brain applies to posture
but not to a gait it is playing out.

### 5.5 Plasticity

Plasticity is part of the tick, not a mode of the system: in every run reported above the rule runs
on every tick from birth, on the same synapse table the instinct table wrote (Section 2.3). The
"learning off" condition appears only where a result needs a control — R4's hue boundary with the
rule disabled, R9's dark-dopamine condition — and it disables the update; it is a knife for
attribution, not a second way of running the brain. Its price at present is real and is stated in
R11: in the closed loop at 24x16 a tick costs 58 ms with the rule disabled and 158 ms with it
enabled, against the 20 ms the body allows, so a brain that learns on every tick currently runs
slower than the body it drives. Most of that gap was one avoidable scan and is now gone -- the
update used to touch all 27.6 million inhibitory synapses per tick and now touches only those
incident on an active cell, a change that is bit-identical to the code it replaces
(`验证_学抑制_逐拍对照.py`) and that takes the 24x192 step from 554 ms to 16 ms. Because the update touches the same
synapses with the same numbers, it also produces the same behaviour, and we checked that at the
level of the results rather than only at the level of the rule: R1, R4 and R6 were re-run after
the change and reproduce their published numbers to the digit -- 5/5 seeds standing and
approaching (median 1.73 m), the hue sweep stopping at 96 degrees on seed 20260914 in a
1040-tick run with the same closing controls, the hue boundary table unchanged (0-96, 168 and
180 degrees lit on 4/5 seeds; 120-144 and 204-240 on 0/5; black on 0/5), and gaze error 5.6 and
4.8 degrees with the empty scene at 0.0 (logs `日志_行为回归_改后.log`, `日志_回归_R4_不保留.log`,
`日志_回归_色相前测_不保留.log`, `日志_回归_R6_改后.log`). Making the whole
live loop fit inside the tick remains an engineering target rather than an architectural question;
until it is met, what the results above support is that the rule is sufficient, not that it is
cheap.

`皮层连接_cortex_links.py`:

| parameter | value |
|---|---|
| excitatory learning rate | 0.005 |
| decay per tick | 0.995 |
| weight of a newly created synapse | 0.001 |
| co-activations required before creating a synapse | 4 |
| maximum pairs examined per tick | 4,000 |
| maximum synapses created per tick | 200 |
| trace decay / threshold | 0.9 / 0.05 |
| inhibitory learning rate / cap | 0.0005 / 0.6 |
| disinhibition learning rate | 0.02 |
| plasticity multiplier for motor, motor memory, vestibular, proprioception, body state | 0.02 |

The update on a synapse that is co-activated in the accepted order is w <- w + eta * m * (1 - w),
where m is the per-region multiplier of the last row of the table; a synapse whose presynaptic cell
was active but whose postsynaptic cell did not follow is decayed by w <- w * (1 - (1 - d) * m) with
d the per-tick decay. A new synapse is created at 0.001 once its pair has been seen in the accepted
order four times, and then follows the same rule. There is no term anywhere that lowers a weight
because the postsynaptic cell fired anyway, which is the failure discussed in 4.6. Frozen synapses
(a flag in the instinct table) are never modified. The 0.02 multiplier is how "balance reflexes
barely change in a lifetime" is expressed; it is a single number, not a special case in code.

R9 adds a second, faster path on the same principle. The dopamine region is a named block of 200
cells (`本能工具_instincts.py`); when any of it is active on a tick,
`皮层连接_cortex_links._多巴胺强化` writes, on that tick, a synapse from every cell that was active
on the previous tick to every cell that *became* active on this one. Per target cell at most eight
sources are used, and the total drive delivered by one reward is fixed at 0.50 and divided equally
among them (the same convention as the instinct table); the per-region factor of the slow path is
applied to the fast path as well (`这对学多快`). Three experimental switches in
`主循环_完整的一拍.py` are used by R2 and R9 and are not part of the architecture: `前额叶按灭`
forces the whole prefrontal population to zero; `前额叶不保留` replaces the prefrontal code every
tick instead of adding to it, i.e. removes the loops described in R9; and the prefrontal
self-inhibition of R9 is one more inhibitory current on that population, *k* times the number of
prefrontal cells active on the tick, with *k* read from the environment (`前额叶自压=0.001`) and 0
by default (`皮层连接_cortex_links.前额叶自压`).

**The two configurations of the same system.** R1-R8 were run with `前额叶不保留`, i.e. with the
prefrontal code recomputed from its input on every tick and no persistence. R9, and the two
persistence rows of the R2 table, are the only measurements in this paper that use the loops; the
configuration is stated wherever the two differ. Either configuration can be selected for any
script from the environment (`前额叶不保留=1`), so every protocol in this paper can be re-run both
ways with the code in the repository.

### 5.6 Body, sensors, eyes

The body is a Unitree Go2 in MuJoCo, driven by joint-angle servos (stiffness 80, damping 4,
maximum torque 25 Nm), ten 2 ms physics sub-steps per cortical tick. The visual hierarchy
receives 1920x1080 RGB, reduces it to 24x16 macro-pixels (three channels, 10 threshold pairs
each, i.e. a thermometer code), and passes it through four parameter layers of radius 10 with
weights uniform in a range of 0.1 around a threshold of 0.6; the exposed tensors are layers 1, 2
and the compressed output do. Visual field 100 x 75 degrees. The auditory hierarchy receives a
4096-bin spectrum through the same kind of network. The eyes have range 30 degrees, muscle
acceleration 500 degrees/s^2 and damping 14 /s, so a single tick of maximum force moves the eye
about 0.1 degrees. The touch afferent of R10 is a 210-cell region -- seven skin sites of 30 cells
each, appended after the inhibitory pool -- with no random connections in or out and no internal
recurrence, so it is entered only by contact and reaches the rest of the cortex only through its
seven fixed rules into the dopamine region. It is absent from the configuration that produced every other number in this paper, and the code that adds it is switched on by an environment
variable, so that the measurements above are made on the build they were made on.

### 5.7 Protocols

Every experiment in this paper is a frozen protocol: the brain is built from a fixed seed, the
condition is applied, and the run is recorded to a log file that is kept in the repository. Each
result section names its script and its log. Multi-seed results use seeds
20260914-20260918. Where a result is a single run we say so. Results that contradicted our
prediction (R4, R5) are reported with the prediction stated, and where a result is visible on one
build of the system and not on another, both builds are reported (R4).

The scale measurements of R11 do not perturb any frozen protocol. The mosaic size is read from the
environment (`AGI行数` / `AGI列数`) by both the visual front end and the world, and the cost numbers
are single runs on one host (16 GB, 16 logical cores) in which the network is built at the requested
width and stepped, with and without plasticity, but no body and no behaviour are run. The
touch-region configuration of R10 is likewise switched on by an environment variable
(`AGI触觉宽度`, 210 in R10) rather than being part of the default build.

The two 20-second-of-simulation baselines used throughout are: *walking* is counted as present
if the walking action's time cells fire on at least 40 of 200 ticks, and *gaze* is counted as
tracking if the mean angular distance between the gaze centre and the object is at most 12
degrees. A reflex that is not wired at all gives 15.3 degrees on the gaze measure, so 12 degrees
sits inside the gap between "wired" (5.6) and "unwired" (15.3).

### 5.8 Reproducing

```
# R1-R8 are run with the prefrontal recurrence switched off (Methods 5.5). In PowerShell:
$env:前额叶不保留 = '1'                  # bash:  export 前额叶不保留=1
python 本能工具_instincts.py 标定        # rebuild the calibrated names (needed after changing a front end)
python 生成本能表.py                     # regenerate the instinct table (585 rules)
python 实验_闭环前提_多种子.py            # R1
python 实验_前额叶_多种子.py              # R2, with and without persistence
python 实验_本能分阶段.py                # R3
python 实验_渐变_正式协议.py 3,6,12 20260914,...   # R4
python 实验_方位分辨力_按层.py            # R5, representational part
python 实验_前几层看得见小东西吗.py        # R5, localisation part
python 实验_追踪靠的是哪一层.py            # R5, behavioural part
python 实验_眼睛跟随.py 小横着飘           # R6
python 实验_本能表整体偏移_正前方.py       # R7
python 实验_模糊化_存活曲线.py             # R7, destructive version
python 实验_想还在吗.py                     # R8, stimulus removed / action cleared
python 实验_切断回响.py                     # R8, recurrent step removed
python 实验_两半各亮多少.py                 # R2, the two halves measured separately
python 实验_发育_声音叫走路_多种子.py        # R9, four stages (logs written into `logs/`)
python 实验_发育_无奖励对照.py               # R9, dopamine left dark
python 实验_发育_无环对照.py                 # R9, without the loops
python 诊断_天生步态稳不稳.py                # R9, innate gait with and without the loops
python 诊断_摔是谁干的2.py 20260915 20260918            # R9, where the topple comes from
python 诊断_摔是谁干的2.py --无环 20260915 20260918     #   ... and the same without the loops
python 诊断_前额叶打走路链.py 20260915                  # R9, where the topple comes from
python 诊断_冻死进运动.py 20260915                      # R9, the motor weights do not change
python 诊断_返回线截断.py 20260915                      # R9, the return line is not the cause
python 试_打折R9_参数.py 0.75                          # R9, the one-number repair, five seeds
python 实验_发育_仿生奖励_三臂.py 一直摸 20260914 20260915 20260916   # R10, reward by contact
python 实验_发育_仿生奖励_声音叫走路.py 20260914,20260915,20260916     # R10, contact only while walking
python 诊断_触觉版_天生站起来.py 20260914 20260915 20260916            # R10, innate righting in the touch build
python 规模试验_测量.py                                               # R11, cost (set AGI列数 for other widths)
python 验证_感觉网分块.py                                             # R11, the memory fix is bit-identical
python 实验_放大_走路边界.py                                           # R11, behaviour at 24x32
python 实验_放大_色相红块.py                                           # R11, the two halves at 24x32
python 诊断_走路点火_多种子.py                                         # R11, the current onto the walk head cell
python 对照_现成RL_驱动.py                # 4.4, the published RL checkpoint driven on our body
python 诊断_步态相位.py                   # 5.4, the gait cycle is played at the recorded rhythm
python 验证_学抑制_逐拍对照.py            # 5.5, the fast inhibition update is bit-identical
python 诊断_整拍耗时.py                   # R11, what a live tick costs (rule on / off)
python 对照_学抑制_改前改后.py            # R11, the same live tick before the fix
python 量规模_七档.py                     # R11, the cost table (seven widths, one log each)
```

R10's touch region is off unless `AGI触觉宽度` is set, and R11's widths are selected with
`AGI列数`; both are environment variables rather than edits, so that every number above stays
attached to the configuration that produced it.
The 24x32 build of R11 is kept apart from the 24x16 one with `AGI数据后缀=_列32`, which names
its calibration file and its instinct table; without it the wider brain would overwrite the
tables that R1-R10 rest on.

`对照_现成RL_驱动.py` needs the third-party checkpoint at `C:\go2_policy\vel\model_500.pt`
(4.5 MB, downloaded from Hugging Face); it trains nothing and takes seconds to run.

R1-R8 are run with `前额叶不保留=1`, which recomputes the prefrontal code from its input on
every tick and so removes the loops of R9; R9 and R10 are run without it. It is a live switch
and not a build option, and it changes the numbers of the vision experiments (R4, R5, R6): run
without it, a script reproduces the persistent configuration, which is what the third row of
R2's table and panels b and c of Figure 9 report. The two R8 scripts were run both ways, and
they take the name of their output from `轨迹名` so that the two runs do not overwrite each
other.

### 5.9 Data and code availability

The repository addressed by this manuscript is
https://github.com/leeR1ven/born-wired-cortex . It contains: the cortex simulator
(`皮层连接_cortex_links.py`), the instinct-table generator (`生成本能表.py`) and the calibration
tool it depends on (`本能工具_instincts.py`), the full tick (`主循环_完整的一拍.py`), the MuJoCo
body and world (`身体_go2.py`, `世界_world.py`), one script per result section named in that
section, the log file that each script wrote, and the plotting script
(`出一张图_论文_20260915.py`) that turns those logs into the figures. Every number in this paper
can be regenerated by running the named script; no number is quoted from a run that is not in the
repository. The locomotion trajectories were distilled from a policy trained with a public
quadruped-learning implementation; the distillation script (`取动作_从训练好的模型.py`) and the
resulting action library (`建动作库.py`) are included so that the provenance of the repertoire can be
checked. The third-party checkpoint itself is not redistributed: it is
`diasAiMaster/unitree-go2-velocity-flat` on Hugging Face, and `对照_现成RL_驱动.py` is the script
that drives it on our body. The repository also carries a four-panel playback page
(`回放_大脑_四幕.html`, built by `出一份四幕页面.py`) which renders the four behaviours of Figure 2
and R8 as looping animations, one panel per sensory condition, with each panel recorded from a
freshly built brain of the same seed, so a reader can watch the behaviour without running the
simulator.

### 5.10 Use of AI assistance

The code, the experimental protocols and this manuscript were produced in an interactive
session with the DeepSeek large language model acting as the programmer, at the direction of the
human author, who does not write code. The human author specified the architecture, set the research
questions, rejected results and interpretations he judged wrong, and required the reporting of
contradicted predictions. Every number in this paper is produced by a script that is in the
repository, with the log of the run that produced it; the model's output was not accepted
without a run. Readers should treat the *codebase* as reviewed by the human author's behaviour
tests rather than by line-by-line reading, and should reproduce independently.

### 5.11 Statements

**Author contributions.** L.Z. conceived the architecture, wrote the six claims of Section 1.1,
authored the instinct table, directed every experiment, rejected results and interpretations he
judged wrong, and required contradicted predictions to be reported rather than dropped. The code
was written in an interactive session with a large language model (Section 5.10).

**Competing interests.** The author declares no competing interests.

**Funding.** This work received no external funding.

**Acknowledgements.** The locomotion repertoire was distilled from a reinforcement-learning
policy trained with a public quadruped-learning implementation; the citation is in the reference
list. The code, the experimental protocols and the manuscript were written in an interactive
session with the DeepSeek large language model, at the direction of the human author
(Section 5.10).
---

## Figures

Each figure is generated from the logs cited in the corresponding result section; the plotting
script is `出一张图_论文_20260915.py` (Figures 1-9) and `出一张图_发育.py` (Figure 10).

* **Figure 1 — One tick of the system.** A schematic of the whole tick: senses drive the two
  sensory hierarchies, which drive one recurrent cortical sheet; the prefrontal population
  compresses the already-compressed signals a second time and returns its output to the sensory and
  motor regions through the return line; the motor-memory time cells drive the 160 motor cells,
  which are the muscles. Generated by `出一张图_系统总览.py`.
* **Figure 2 — The innate wiring closes the loop.** Four panels, five network seeds each:
  (a) final trunk height for a body placed prone; (b) forward displacement for a body shown a red
  region; (c) final trunk height for the no-sensation control, which collapses; (d) the same five
  brains with the instinct table cleared — final trunk height and approach displacement,
  both at or below the no-sensation control. Sources: `日志_闭环多种子_新.log`, `日志_白脑_新.log`.
* **Figure 3 — Causal intervention on the prefrontal population.** Bar chart of walking ticks out
  of 200 for the five conditions of R2 (five seeds each), with the mean number of active
  prefrontal cells annotated. Sources: `日志_前额叶多种子_新A.log`, `日志_前额叶多种子_新B.log`.
* **Figure 4 — The repertoire grows stage by stage.** A five-row matrix (stages A-E) x five
  behaviours, cells shaded by how many seeds passed. Source: `日志_分阶段_新.log`.
* **Figure 5 — Gradual recognition.** (a) how far the colour may drift before recognition
  stops, as a function of step size, with plasticity on and off, one point per seed; (b) the same
  five brains, hue by hue, whether a hue starts the walk before any drift and after the 3-degree
  sweep has run to its stopping point. Sources: `日志_渐变正式_新.log`, `日志_渐变边界_无环.log`.
* **Figure 6 — The cost of compression.** (a) Cross-column sharing of direction names in layers
  1, 2 and 3; (b) for a ball one third of a cell across, the columns in which each layer's newly
  active cells lie. Sources: `日志_方位分辨力.log`, `日志_前几层.log`.
* **Figure 7 — Emergent gaze tracking.** Eye azimuth and object azimuth over time for (a) an
  object crossing the field, (b) an object appearing suddenly, (c) an empty scene. Source:
  `日志_眼睛跟随.log` (regenerate with `实验_眼睛跟随.py 小横着飘`).
* **Figure 8 — Blurred innate wiring.** (a) Resting position of the object relative to the gaze
  centre as a function of the imposed map shift (R7); (b) survival of three behaviours as a
  function of the fraction of the instinct table that has been re-randomised or shifted by one
  cell. Sources: `日志_整体偏移_正前方_新.log`, `日志_存活曲线_新.log`.
* **Figure 9 — Where the behaviour lives.** (a) three behaviours with the intact cortex and with the
  recurrent step removed (dots = individual seeds); (b) walking, the visual code and the prefrontal
  code tick by tick, with the stimulus removed at 0.5 s; (c) the same run with the action's time
  cells cleared once at 1.0 s. Sources: `日志_切断回响_有环.log`, `日志_想还在吗_轨迹_有环.csv` (panels b and c are the
  persistent configuration of R8, in which the prefrontal code for red is still lit after the red
  is gone).
* **Figure 10 — A capability acquired during life.** (a) Walking-action ticks in each of the four
  stages of R9, five seeds: none before teaching, nearly all after. (b) Connections written by
  stage 3 with the dopamine region lit and with it dark. (c) Trunk height reached in stage 1 and
  stage 4, and forward displacement, for each seed. Sources: `日志_发育多种子_A.log`,
  `日志_发育多种子_B.log`, `日志_无奖励对照.log`. Generated by `出一张图_发育.py`.

---

## References

* Amit, D. J. (1989). *Modeling Brain Function: The World of Attractor Neural Networks.* Cambridge University Press.
* Brown, T. B., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems* 33.
* Buzsaki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science* 304, 1926-1929.
* Eichenbaum, H. (2014). Time cells in the hippocampus: a new dimension for mapping memories. *Nature Reviews Neuroscience* 15, 732-744.
* Eliasmith, C., et al. (2012). A large-scale model of the functioning brain. *Science* 338, 1202-1205.
* Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. *Cerebral Cortex* 1, 1-47.
* Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11, 127-138.
* Fuster, J. M. (2001). The prefrontal cortex - an update: time is of the essence. *Neuron* 30, 319-333.
* Gewaltig, M.-O., & Diesmann, M. (2007). NEST (NEural Simulation Tool). *Scholarpedia* 2, 1430.
* Grillner, S., & Wallen, P. (1985). Central pattern generators for locomotion, with special reference to vertebrates. *Annual Review of Neuroscience* 8, 233-261.
* Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory.* Wiley.
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS* 79, 2554-2558.
* Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *The Journal of Physiology* 160, 106-154.
* Hwangbo, J., et al. (2019). Learning agile and dynamic motor skills for legged robots. *Science Robotics* 4, eaau5872.
* Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks* 14, 1569-1572.
* Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*.
* Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. *Nature Reviews Neuroscience* 21, 335-346.
* Maass, W., Natschlager, T., & Markram, H. (2002). Real-time computing without stable states: a new framework for neural computation based on perturbations. *Neural Computation* 14, 2531-2560.
* Magee, J. C., & Grienberger, C. (2020). Synaptic plasticity forms and functions. *Annual Review of Neuroscience* 43, 95-117.
* Markram, H., et al. (2015). Reconstruction and simulation of neocortical microcircuitry. *Cell* 163, 456-492.
* Miller, E. K., & Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience* 24, 167-202.
* Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518, 529-533.
* Mountcastle, V. B. (1997). The columnar organization of the neocortex. *Brain* 120, 701-722.
* Oudeyer, P.-Y., & Kaplan, F. (2007). What is intrinsic motivation? A typology of computational approaches. *Frontiers in Neurorobotics* 1, 6.
* Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think: A New View of Intelligence.* MIT Press.
* Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience* 2, 79-87.
* Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to walk in minutes using massively parallel deep reinforcement learning. *Conference on Robot Learning (CoRL)*.
* Tinbergen, N. (1951). *The Study of Instinct.* Oxford University Press.
* Turrigiano, G. G., & Nelson, S. B. (2004). Homeostatic plasticity in the developing nervous system. *Nature Reviews Neuroscience* 5, 97-107.
* Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. *Neuron* 36, 955-968.

*(Bibliographic details in this list still need a check against the original sources before
submission; the citation list has not been machine-verified.)*
