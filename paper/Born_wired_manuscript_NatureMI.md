# Born wired: innate cortical connectivity plus local plasticity is enough to stand, walk and look

**Zhiwen Li**

Independent Researcher, No. 67 Yuanren Street, Huangjing Town, Taicang, Suzhou, Jiangsu, China

rivenlee94@gmail.com · ORCID: 0009-0005-8289-6393

## Abstract

Brains are born with largely pre-specified cortical wiring, and the only rule locally available to a
synapse is correlation between the neurons it connects; machine-learning agents instead optimise a
global objective. How far does that combination reach alone? A system of 143,796 model cortical
neurons and 308,809 pre-specified "instinct" synapses drives a simulated quadruped with no reward,
gradient or training loop. Placed prone it stands up on 5/5 network seeds; shown red it walks toward
it on 5/5; with no sensory input it collapses; with the table cleared it cannot stand or
approach (0/5). Silencing the prefrontal population abolishes visually guided walking
(0/200 ticks, 5/5) while leaving a wired reflex unchanged. Adding innate
structure grows the repertoire without retraining; a 200-cell dopamine population writes one new
sense-to-action link during life; gaze tracking emerges from static look-at rules.
The acquired walk does not yet stay upright.

A deep network is usually drawn as a feed-forward transducer: a token or a pixel enters on the left,
an answer leaves on the right, and a loss is back-propagated through the stack <sup>1, 2</sup>; the intermediate layers are the computation and the output layer *is* the
answer. We work from a different picture, stated as six claims (P1)-(P6) that the results below test.

**(P1) A layered sensory network is a compressor.** It reduces a very large number of peripheral channels to a small number of cortical channels. Its
output is not an answer but a bus, and reading the answer off the last layer mistakes signal conditioning
for thought: a hierarchy in which each stage re-represents its input is classical <sup>3-5</sup>.

**(P2) Thought is co-activation of cortical neurons** <sup>6-9</sup>. A thought is a sequence of populations, the next determined entirely by the current weights and
the currents arriving on the somata. The testable consequence is sharp: remove the recurrent step and
the behaviour should go with it, while anything the wiring maps directly from sense to muscle survives
untouched.

**(P3) The prefrontal population compresses again** <sup>10, 11</sup>: it reduces
the already-compressed signals of several regions further, and then thinks with the same rules as
everything else. Information is unavoidably lost in that second compression.

**(P4) Instinct is the initial state of the sheet** <sup>12</sup>. The innate part is not a module and not conditions in code: it is the connectivity the sheet is born
with, within and between regions, including the suppressive connections (a rule that must suppress
something drives an inhibitory cell; there are no negative weights anywhere). What experience changes it
changes slowly and locally.

**(P5) A network cannot start from silence** <sup>13</sup>. A newborn cortex already has
ordered wiring, because an organism that must discover standing and gaze stabilisation from an
uninformative start does not survive the discovery period.

**(P6) A capability can be acquired during the life of one brain, without an objective.** The innate
wiring fixes the starting repertoire but is not all the sheet will ever do: the same local correlation
rule can write new connections while the brain behaves. A reward-like population, if present, is one more
population whose firing licenses that writing, not an error signal computed from a target.

If this is right, a system should do real things with a single recurrent population of model cortical
neurons, a fixed pre-specified wiring diagram (the *instinct table*) and a purely local plasticity rule
that cannot see any task objective, while driving a real body. We built one and asked seven falsifiable
questions: whether the innate wiring alone closes the loop (R1); whether a behaviour can be attributed to
a specified population by silencing it (R2); whether the repertoire grows by adding innate structure
without retraining (R3); whether local plasticity buys anything a fixed recogniser does not (R4); whether
compression costs anything measurable (R5); whether the stimulus or the cortex produces a running
behaviour (R8); and whether a capability can be acquired during life (R9). A schematic of one tick is in
Supplementary Fig. 1.

## Results

### The innate wiring alone closes the loop (R1)

We built a brain from the instinct table (585 rules, expanded to 308,809 excitatory synapses;
Methods) plus sparse random background connectivity, attached it to a simulated quadruped, and gave
it nothing but its own senses (Fig. 1).

| condition | measure | 5 seeds |
|---|---|---|
| placed prone, no instruction | final trunk height | **5/5 above 0.20 m**; median 0.261 m (min 0.261) from a starting 0.099 m |
| upright, red region in front | forward displacement in 4 s | **5/5 above 0.30 m**; median +1.73 m (min +1.48, max +2.09) |
| upright, **no sensory input at all** | final trunk height / tilt | **5/5 collapsed**; median 0.184 m, tilted 41 degrees |
| prone, **instinct table cleared** | final trunk height, then displacement with red ahead | **0/5 above 0.20 m**; median 0.184 m; **0/5 approach** (median -0.12 m) |

The third row is the control that matters: with the sensory regions blanked the body does not "do
nothing gracefully", it falls over, because the motor cells receive no current and nothing in the
simulation holds the body up except the cortex. The fourth row is a within-seed ablation of the one thing
the framing is about -- the body, the sensors, the seed and the 100 sparse synapses per cell are all
present and only the 585 rules are removed. On the same five seeds not one brain stands up and not one
approaches the red region, so the behaviour in rows 1-2 is carried by the written wiring, not by the
random background.

Standing up is not, in this body, a control problem -- holding all twelve muscles at the constant
stance vector brings the trunk to 0.261 m -- so row 1 measures the *link*: the regions reporting
"lying down" are wired to the cells that emit that vector. The random background changes only the
walking distance, 1.48 to 2.09 m.

### The behaviour is computed by a specific population (R2)

Rules in the table are currents, not conditions. We kept the visual input fixed (a red region straight
ahead, identical every tick) and silenced the whole prefrontal population by forcing it to zero *after*
it was computed and before the cortical step (Fig. 2).

| condition | walking action lit (of 200 ticks) | first fire | prefrontal cells active (mean) |
|---|---|---|---|
| red, prefrontal intact | **199/200** (5/5) | tick 2 | 2782-2785 |
| red, prefrontal silenced | **0/200** (5/5) | never | 0 |
| black screen, prefrontal intact | 0/200 (5/5) | never | 368 |
| blue screen, prefrontal intact, loops on | **183-194/200** (5/5) | ticks 7-18 | 2872-2880 |
| blue screen, prefrontal recomputed each tick | 0/200 (5/5) | never | 1444-1445 |

The ablation is *specific*: silencing 53,200 cells removes one visually guided behaviour and leaves the
sensory response itself intact -- a blue screen still drives 1,444 prefrontal cells when the code is
recomputed every tick -- while the behaviour that needs that population is gone on 5/5 seeds. The
mechanism, however, is arithmetic rather than logical: "red ahead -> walk" is written as **two** rules,
the visual region contributing 0.80 and the prefrontal region 0.85 to the same motor-memory cells, and
the firing threshold is 1.0, so neither half is sufficient alone and the sum is. We do not call this an
AND gate: the motor cell is a linear summing junction, and one region alone could drive the same
behaviour in another context.

The blue rows are where the two configurations part. Per tick a blue screen delivers 0.73 of the 1.0
the walk needs, so with the code recomputed every tick the walk never starts; once that code outlasts
the tick that produced it the deficit is repaired across ticks, the walk starts at tick 7 to 18, and it
runs on its own dynamics. The boundary is still a threshold; what persistence changes is whether it is
crossed in *space* or in *time*.

### The stimulus starts the behaviour; the cortex runs it (R8)

A population-specific ablation shows that a named population is necessary; it does not show where the
behaviour *is* while it runs. We therefore measured the behaviour and the two cortical codes that
produce it, tick by tick, and then removed the stimulus: a red region is present for the first half
second and is then replaced by a black screen. Walking is counted by the walking action's time cells
being lit, and the codes as the number of active cells inside the named blocks.

| condition, after t = 1.0 s | walking action lit | `视觉:中有红` (1346 cells) | `前额叶:看着红` (1360 cells) |
|---|---|---|---|
| red throughout | 495/495 | 1346 | 1360 |
| red removed at 0.5 s | **495/495** | **0** | **3** |
| red removed at 0.5 s, action's time cells cleared once at 1.0 s | **0/495** | 0 | 1 |
| never red, same clearing at 1.0 s | 0/495 | 0 | 2 |
| never red (control, no clearing) | 0/495 | 0 | 3 |

Five seeds each. The stimulus is gone, the visual code reads zero,
and the walking action is still lit on **every one of the 495 ticks**: the stimulus *selected* an
action, and once selected it is the trajectory that produces the movement. Clearing the action's own
fifteen time cells once stops it for good in both configurations -- what survives is a drive of 0.85
from the prefrontal half, below the threshold of 1.0 that the two halves together supply.

The converse intervention is the falsifier we wrote down in advance: we kept the instinct table exactly
as it was and cleared the cortex at the start of every tick, so that only the current sensory signals
were injected (Supplementary Fig. 4) -- a purely feed-forward arc, sense to cortex to muscle, built from the same wiring.

| behaviour | intact cortex (3 seeds) | recurrent step removed |
|---|---|---|
| stand up from prone, final trunk height | **0.261 m** (3/3) | **0.184 m** (0/3) |
| red ahead: ticks with the walking action lit, of 200 | **199 / 199 / 199** | **0 / 0 / 0** |
| small moving object: mean gaze error | 5.6 deg (3/3) | **5.6 deg (3/3)** |

The manipulation is not a blunt instrument: the gaze reflex, a direct map from a visual column to an eye
muscle, is unchanged **to one decimal place**, while everything that consists of *doing an action*
disappears.

### The repertoire grows by adding innate structure, not by retraining (R3)

We built five nested instinct tables on the same random background connectivity and re-measured **all**
behaviours at every stage, not only the new one (Fig. 3). 

| stage | rules | stand up | walk to red | react to sound | stopped by the tone | gaze tracking |
|---|---|---|---|---|---|---|
| A motor repertoire only | 221 | **0/3** | 0/3 | 0/3 | 0/3 | 0/3 |
| B + balance / righting | 227 | **3/3** | 0/3 | 0/3 | 0/3 | 0/3 |
| C + visually guided approach | 233 | 3/3 | **3/3** | 0/3 | 0/3 | 0/3 |
| D + auditory reactions | 236 | 3/3 | 3/3 | **3/3** | 0/3 | 0/3 |
| E + gaze | 581 | 3/3 | 3/3 | 3/3 | **0/3** | **3/3** |

Three properties matter. *The repertoire is not the behaviour*: with the motor repertoire alone the
brain contains the standing-up action and never performs it -- the body is placed upright and simply
falls over. A capability that exists in the wiring is not a capability until something lights it. *Each
addition buys exactly the capability it encodes*: stages C, D and E add 6, 3 and 345 rules and each turns
on its own behaviour without turning on any other. *Additions do not damage the earlier repertoire*: the
stages are nested in behaviour, not only in the table.

One cell of the matrix fails. "Stopped by the tone" is 0/3 at every stage, including E. The tone *does*
stop the walking -- with red alone the action is lit on 199 of 200 ticks, with red and the tone together
on 15 of 200 -- but fifteen ticks is one further cycle of the fifteen-tick walk, so it fails the criterion
we fixed in advance (at most 10 of 200 ticks), and we report it as a failure rather than move the
threshold afterwards.
### What sets the boundary of recognition, and how far local plasticity moves it (R4)

A recogniser that is fixed has a fixed tolerance; a synapse that follows a local correlation rule has no
tolerance, only a rate, because it follows whatever its two cells have been doing. We tested whether that
difference has behavioural consequences, and the experiment produced a result we did not predict (Fig. 4). The hue
of a full-field colour starts at 0 degrees (red, the colour the approach instinct was calibrated on) and
is stepped by 3, 6 or 12 degrees; after each step the walking action is **stopped**, the cortical state
cleared, and the colour presented afresh for 8 ticks. If the action re-lights on at least 6 of those 8 the
colour counts as recognised and the sweep continues -- restarting the action at every step is what makes
this a test of recognition, because a lit action runs by its own dynamics.

**The boundary is where the sum crosses, and it is there in a brain that has never learned.** Every hue
whose two halves sum to 1.39 or more starts the walk on 5/5 brains; every hue whose halves sum to 0.97 or
less starts it on 0/5. Nothing else predicts the column, and the pattern is monotone in the sum and *not*
monotone in hue: cyan at 180 degrees, the opposite of red, comes closer to starting the walk (sum 1.03)
than green at 120 degrees (sum 0.75).
Nor is the column something learning built: with the correlation rule switched off entirely, the same five brains give the same column.

**The drift moves one brain, and it is the brain that was sitting on the line.** The three hues whose
sums lie inside 0.05 of the threshold -- 96, 168 and 180 degrees -- go from 4 of 5 brains to 5 of 5 after
a 3-degree sweep. Across step sizes the rule carries a marginal stimulus to the edge of the visual island
the colour belongs to (96 degrees, where the visual half of the sum falls from 0.39 to 0.22) and no
further: no brain, in any condition, crosses the gap at 120-144 degrees. The prediction we started from is
therefore **half right, and the half that fails is the instructive one**: the rule widens recognition, but
only over the island the sensory front end already formed, and only for stimuli already within a few
hundredths of the line. Learning fills in a boundary; it does not draw one. And the block the calibration
labelled "there is red in front of you" is lit on **42 percent** of its cells by a cyan screen, so nothing
in the system represents *red* (Supplementary Fig. 2).

### Gaze tracking is emergent, not written down (R6)

Nothing in the instinct table refers to movement, velocity or tracking (Fig. 5). The gaze behaviour is assembled
from fourteen static rules -- "column c contains something -> pull the eye muscle that turns toward column
c", with the two columns straddling dead centre left unwired so a centred object produces no pull and the
eye settles instead of oscillating -- and 322 motion-onset rules, in which each column drives a trace
arriving one tick late and a change cell that fires only if the column is active *now but was not one tick
ago*.  The onset path says "something moved -- look
at it"; the static path says "keep looking at it".

| scene | mean angular distance between gaze centre and object | eye at the end |
|---|---|---|
| 0.05 m ball crossing the field from -25 deg at 0.5 m/s | **5.6 deg** | +24.6 deg |
| 0.05 m ball appearing suddenly at +20 deg, 2 s in | **4.8 deg** | +21.4 deg |
| empty scene for the whole trial | **0.0 deg -- the eye does not move at all** | +0.0 deg |

The third row is the important control: with nothing in the field the eye is perfectly still, so the
first two rows are not an artefact of a drifting eye. They also show what the onset path does -- it
converts a sequence of very weak, time-extended signals into a single strong signal at one moment. A
separate manipulation, in which every direction-bearing name in the table is shifted by one or more whole
columns, shows that this map can be *blurred* without breaking behaviour: the resting position of the eye
moves by one cell per cell of imposed shift and the behaviour stays ordered (Supplementary Fig. 3).

### A capability can be acquired during the life of one brain (R9)

Everything above measures a repertoire present at birth. The claim this project is built on -- that
complex behaviour is assembled on top of innate behaviour rather than trained from nothing -- is a claim
about *development*, and the experiment that bears on it is one in which a capability appears during the
life of one brain (Fig. 6). We added two things. A **200-cell dopamine region**, silent in every other experiment
here: when active, it writes a connection *on the spot* between cells active on the previous tick and
cells that *became* active on this one -- the same "who brought whom in" relation the slow correlation
rule uses. And **loops inside the prefrontal population**: inside each named block the cells are wired
head-to-tail into a closed chain (6,281 synapses in five blocks) and the main loop *adds* the newly
computed code to the code already running, so a block stays lit after the stimulus that lit it has gone. 

| stage | body and senses | dopamine | walking action lit |
|---|---|---|---|
| 1 birth | prone, black screen, broadband sound only, 30 ticks | off | **0 / 30** (5/5 seeds) |
| 2 innate | standing, red region ahead, no sound, 30 ticks | off | **30 / 30** (5/5) |
| 3 teaching | standing, red ahead *and* sound, 200 ticks | **on** | running |
| 4 test | sheet cleared, prone again, black screen, **sound only**, 200 ticks | off | **199-200 / 200** (5/5) |

Stages 1 and 4 are the *same stimulus on the same brain*, before and after. In stage 1 the sound lights its own block -- 18 to 34 of the 1097 cells of the named block
`听觉:低频响` -- and the walking action is lit on none of the 30 ticks: the body only settles onto the
floor. In stage 4 the
action is lit on essentially every tick and the body, placed on its belly, first stands up (0.099 m to
0.327-0.391 m, 5/5) and then walks 1.00-2.43 m. Teaching writes **490,659-552,186** new excitatory
connections on top of the 308,809 innate ones, and nothing is written by hand.

**The control the claim needs.** With the dopamine region left dark, the same four stages write
**0 new connections** and stage 4 stays at **0/200 ticks** on 3/3 seeds: it is the reward-like
signal, not the passage of time and not the sound, that does the writing. Repeating the stages with
the prefrontal replaced every tick also acquires the behaviour (199-200 of 200 test ticks, 5/5), so
the association does not need persistence; what persistence buys is the *standing up* -- with the
loops 5/5 brains stand up in stage 4, without them 1/5 does.

**What is unfinished, and we report it as unfinished.** With the loops the prefrontal population is an
accumulator with no decay: under a constant stimulus it grows from 1581 active cells on the first tick to
2916 by tick 200, because nothing removes it. The instantaneous
boundary of Fig. 2 becomes a boundary in *time* -- a blue screen drives the walk on 183-194 of 200 ticks,
against 0/200 without the loops -- and the accumulator over-drives the innate locomotor loop: the innate
walk toward red, with no learning and no sound, topples on two of five seeds with the loops while the same
brains walk 2.26 m and 2.51 m without falling when the prefrontal is replaced every tick. In the
acquisition test itself 4 of 5 seeds topple 42-101 ticks in, after 1.00-1.18 m, and controls on the same
taught brains show it is not the reward-written connections: one seed topples under the *innate* repertoire
with no teaching at all, and another walks 2.13 m without falling if the slow correlation rule is off at
test time. A recurrent population needs a way to forget, and the loops we wrote have none.

## Discussion

The measurements support a specific and unglamorous claim: **a recurrent cortical population with a fixed,
pre-specified wiring diagram and a purely local correlation rule is already enough to close a sensorimotor
loop and to produce behaviour that can be attributed, population by population.** Nothing in the results
requires a global objective, a critic, a reward or a backward pass. Of the six claims, four are supported
(P1, P2, P4, P5), one with a boundary (P6: acquisition works, staying upright does not) and one is partly
unfinished (P3: the prefrontal population supplies 0.85 of the 1.65 that crosses the threshold, but once
given loops it accumulates without decay). Supplementary Table 1 gives the experiment behind each.

Three things the results do not show. *Development*: the acquisition experiment acquires one link, in one
session, from a sense to an action the brain could already perform, with us deciding when the reward is on;
a repertoire that grows over lived time because of what the brain has already done remains a design
commitment. *Scale*: everything runs on 143,796 neurons on one CPU core, and a wider cortex is not a
mechanical change -- at 24x32 the eye is measurably finer (gaze error 5.5 to 4.3 degrees) while the colour
boundary *narrows*, because the current a rule delivers is fixed by construction while the inhibition a
cell receives grows with the amount of active cortex. A build of 925,236 neurons costs 68 s to construct
and 2,555 MB resident on a 16 GB host: four orders of magnitude below a human cortex, and a cost curve
rather than a scaling law. *Authoring*: the table is written by us, and the locomotion repertoire is
distilled from a separately trained policy, supervision that happens before birth rather than during life.


The relation to reinforcement learning is worth stating precisely. Every behaviour here would be
learnable by RL <sup>14, 15</sup>; the point is not that RL cannot do it, but that
here the behaviour is not learned at all, it is a property of the initial wiring. We measured a published
policy rather than training our own, because a published artefact states its own cost: a PPO controller
for this same quadruped, 8,192 parallel environments x 24 steps x 506 iterations = 99,483,648 physics
steps on ten GPUs. Its input layer is 45 numbers and not one of them comes from an eye or an ear, so 10^8
samples buy the lower half of the problem -- given a velocity command, move -- and the upper half,
whether to move at all, is not in it. Our own attempt at that half first crossed the 0.30 m line at
1.2-1.8 x 10^5 steps and never reached 1.0 m, where one row of the instinct table covers the same thing
with no samples. Predictive-coding and free-energy accounts <sup>16, 17</sup> also avoid an explicit
loss, but replace it with a generative model and an inference procedure; here there is neither. The
repertoire-building literature instead trains a curriculum <sup>18</sup>; here the curriculum
*is* the wiring.

A separate build runs the same mechanism on a different substrate -- 344,787,049 characters of Chinese
text as cells, the same correlation rule, no objective -- and shows how such a population holds a state:
characters are active in groups, the group outlives the prompt that lit it while turning its membership
over, and every prompt eventually settles into an exact periodic orbit (Supplementary Note 1). That is the mechanism the last two results need, and holding a state is not free at any size: in that
build a small activity budget left 45 of 46 prompts dark, and the cortical sheet at 143,796 cells is in
that regime, which is why the prefrontal loops had to be given explicitly.

The tempting reading of all this is "learning is overrated". That is not our reading: the first problem
an organism has is not to *learn*, it is to do something ordered, and a random network with a correlation
rule has nothing ordered to correlate. The scarcity is in initial structure, not in optimisation. Three
falsifiable consequences follow: an agent given a large innate behavioural repertoire before training
should need dramatically less experience than one that must discover its first ordered behaviour, and the
cross-system version of that comparison has not been run; removing the recurrence should remove actions
while leaving wired reflexes intact, and it does; and at some point adding structure must stop buying
capability. We do not claim that this architecture will reach general intelligence, and we would distrust
any paper that claimed it on this evidence.

## Methods

### Cortex

The cortex is a flat boolean array of 143,796 cells (`皮层连接_cortex_links.py`). One tick is 20 ms. The
unit is deliberately cruder than a conductance-based or spiking model <sup>19</sup> and the
tick is a coarse stand-in for cortical dynamics <sup>20</sup>; both simplifications are
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

### Regions and calibration

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

### Instinct table

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

### Where the motor repertoire came from

The locomotion gaits were copied from a reinforcement-learning policy trained to make the same
Go2 model walk (`取动作_从训练好的模型.py`, `建动作库.py`), following the approach of Hwangbo et al.<sup>15</sup> and Rudin et al.<sup>21</sup>. We swept forward/lateral/turning
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

### Plasticity

Plasticity is part of the tick, not a mode of the system: in every run reported above the rule runs
on every tick from birth, on the same synapse table the instinct table wrote (Methods). The
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
because the postsynaptic cell fired anyway, which is the failure discussed in the Discussion. Frozen synapses
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

### Body, sensors, eyes

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

### Protocols

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

### Reproducing

```
All experiments except the development ones are run with the prefrontal recurrence switched off. In
PowerShell:
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
python 诊断_步态相位.py                   # the gait cycle is played at the recorded rhythm
python 验证_学抑制_逐拍对照.py            # the fast inhibition update is bit-identical
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
R2's table and panels b and c of Supplementary Fig. 4 report. The two R8 scripts were run both ways, and
they take the name of their output from `轨迹名` so that the two runs do not overwrite each
other.

### Data and code availability

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
(`回放_大脑_四幕.html`, built by `出一份四幕页面.py`) which renders the four behaviours of Fig. 1 and Supplementary Fig. 4 as looping animations, one panel per sensory condition, with each panel recorded from a
freshly built brain of the same seed, so a reader can watch the behaviour without running the
simulator.

### Use of AI assistance

The code, the experimental protocols and this manuscript were produced in an interactive
session with the DeepSeek large language model acting as the programmer, at the direction of the
human author, who does not write code. The human author specified the architecture, set the research
questions, rejected results and interpretations he judged wrong, and required the reporting of
contradicted predictions. Every number in this paper is produced by a script that is in the
repository, with the log of the run that produced it; the model's output was not accepted
without a run. Readers should treat the *codebase* as reviewed by the human author's behaviour
tests rather than by line-by-line reading, and should reproduce independently.

### Statements

**Author contributions.** L.Z. conceived the architecture, wrote the six claims of the Introduction,
authored the instinct table, directed every experiment, rejected results and interpretations he
judged wrong, and required contradicted predictions to be reported rather than dropped. The code
was written in an interactive session with a large language model (Use of AI assistance).

**Competing interests.** The author declares no competing interests.

**Funding.** This work received no external funding.

**Acknowledgements.** The locomotion repertoire was distilled from a reinforcement-learning
policy trained with a public quadruped-learning implementation; the citation is in the reference
list. The code, the experimental protocols and the manuscript were written in an interactive
session with the DeepSeek large language model, at the direction of the human author
(Use of AI assistance).
---

## Supplementary information

Supplementary information accompanies this paper and contains Supplementary Figs. 1-4 and
Supplementary Notes 1-3. Supplementary Fig. 1 is a schematic of one tick. Supplementary Fig. 2 is
the layer-by-layer measurement of the visual hierarchy (R5). Supplementary Fig. 3 is the
blurred-wiring manipulation (R7). Supplementary Fig. 4 is the recurrence removal (R8).
Supplementary Note 1 reports the character-substrate build, Supplementary Note 2 the reward
delivered by contact rather than by a switch (R10) and Supplementary Note 3 the cost of a wider
cortex (R11), together with the wider build's behaviour. Supplementary Table 1 states, for each of
the six claims, the experiment that bears on it and the outcome.

## Supplementary Notes

**Supplementary Note 1 | The same rule on a character substrate.** A separate build of this project
replaces the cortical cells with single Unicode characters and the sensory hierarchies with a text
stream: 344,787,049 characters of modern Chinese, one cell per character, the same local correlation
rule, no objective, no back-propagation and no language model anywhere in the loop. What that build
shows, verified step by step, is how a population of this kind holds a state and moves it.
Characters are active in groups, not one at a time; after the prompt ends the group neither falls
silent nor freezes, turning its membership over from step to step while keeping its characteristic
size; and every prompt eventually settles into an exact periodic orbit (periods of 1 to 6 steps in
the observed window). That is a dynamical observation and we do not read it as a measure of thought.
It is the mechanism R8 and R9 need, and it is also why they need it: the prefrontal code for red
outlives the red for the same reason a character group outlives its prompt. Holding a state is not
free at any size -- in that build a small activity budget left 45 of 46 prompts completely dark --
and the cortical sheet at 143,796 cells is in that regime, which is why the prefrontal loops had to
be given explicitly rather than left to emerge. The build behaves as a bounded associative memory: it
does not learn to answer a question or to follow an instruction, and it is not evidence about
language. Its reports and scripts are in the repository under `语言规模实验_language_scaling_20260913/`.

**Supplementary Note 2 | The reward delivered by the body rather than by a switch (R10).** A 210-cell
touch region -- seven skin sites of 30 cells each, wired to the dopamine region by seven fixed rules
and to nothing else -- delivers the reward of R9 through contact instead of through an experimenter's
flag. The afferent runs through the touch cortex and from there to the dopamine region, and never
from the skin to the reward cells directly. With the hand resting on the model, the dopamine region
is lit on 200 of 200 teaching ticks and the sound-triggered walk appears on 2 of 3 brains (200 of 200
test ticks); a no-touch control writes **0** connections and still does nothing at test (0 of 600
ticks). The instructive failure is a brain that was rewarded on all 200 teaching ticks, wrote 480,928
connections and learned nothing (0 of 200 ticks), because the walk was not running while the reward
was delivered. A reward reinforces what the brain did; it cannot teach what the brain did not do. Two
of the three brains that acquire the link do not stay upright, which is R9's failure rather than a new
one. The touch region is switched on by an environment variable and is absent from the default build.

**Supplementary Note 3 | What a wider cortex costs, and what widening does to behaviour (R11).** The
eye and the visual front end take their mosaic size from one place and the world renders at that
resolution, so a wider mosaic is finer vision rather than an interpolation: 24x16 cells of 6.25
degrees each up to 24x192 cells of 0.52 degrees. The previous peak memory of the whole system (7,005
MB) was a single dense random table in the auditory front end; filling it in row blocks draws the same
random numbers in the same order and is bit-identical, which we checked for both front ends. With
that removed, **925,236 neurons and 62.4 million excitatory synapses** cost 68 s to build, 8 ms per
tick without plasticity and 16 ms with it in the step benchmark, and 2,555 MB resident with a 7,200 MB
peak, which is the ceiling on the 16 GB host. The per-tick update used to scan all 27.6 million
inhibitory synapses; indexing them by source and by target once, at build time, leaves work
proportional to the active population and is bit-identical to what it replaces (checked tick by tick
on all 4,890,945 inhibitory synapses at 24x16, maximum difference 0.0). The benchmark flatters that: a
live tick at 24x16, with the world rendering the mosaic and the body driven, costs **58 ms** with the
rule disabled and **158 ms** with it enabled, so the brain still runs about eight times slower than
the 50 Hz body it drives. At the cheap end, 24x32, behaviour was run, and it is instructive in both
directions. What does not go through the eye is unchanged to the digit: righting stands in 0.261 m on
149 of 150 ticks, exactly as at 24x16. What does go through it is better: gaze error falls from 5.5 to
4.3 degrees and from 4.8 to 3.5, with the empty-scene control still exactly 0.0. And the colour
boundary **narrows**, because the excitation an instinct rule delivers is 1.750 at both widths -- what
a rule writes is the total current, not a per-synapse weight -- while the inhibition onto the same
cells is not fixed, half of the random inhibitory wiring being scattered over the whole brain, so a
wider cortex active over more cells loses margin on every threshold. Widening is therefore not
mechanical: the table's strengths are calibrated to the activity level of the build they were measured
in.

**Supplementary Note 4 | Relation to other modelling traditions.** The behaviour of R8 is a trajectory
through a recurrent population, which places this system in a long tradition in which the computation
is the settling of a population rather than a feed-forward pass <sup>7, 8, 26</sup>. It shares the ambition of the large-scale reconstructions of cortex but none of their
method: those build a detailed model of a known circuit, whereas here the circuit is trivially simple
and the question is what the *initial condition* buys <sup>27-29</sup>. Two of the encoding choices are old ideas in model form: a motor program
that runs itself out once triggered, rather than being recomputed every tick, is the classical
central-pattern-generator proposal <sup>22</sup>, and "one cell per instant of the action"
is a time cell in the sense of Eichenbaum<sup>23</sup>. The plasticity rule is a Hebbian rule of the kind
the cortex is known to use, with saturating bounds and a slow rate; homeostatic plasticity
<sup>24, 25</sup> is the obvious candidate for the forgetting
that this system lacks, and in our hands a flat weight decay did not supply it (see the Discussion).
Nothing here is a scaling claim of the kind made for language models <sup>30</sup>.

**Supplementary Table 1 | What each claim rests on.**

| claim | test | outcome |
|---|---|---|
| **(P1)** a sensory hierarchy is a compressor whose output is a bus | R5: localisation and name overlap, measured layer by layer (Supplementary Fig. 2) | **supported.** Layer 1 places every newly active cell inside the column a 0.31-cell ball occupies; layer 3 scatters the same ball over up to 13 columns and produces nothing at all for a ball at dead centre. Direction names are perfectly exclusive in layers 1-2 and 14 percent shared (worst column 75 percent) in layer 3, where one column is lost. The *input* layer sees the ball perfectly well in every case, so the loss happens inside the hierarchy. |
| **(P2)** thought is co-activation of a cortical population | R8: remove the stimulus, then remove the recurrence (Supplementary Fig. 4) | **supported.** Stimulus removed: both codes read zero (0 and 3 cells of 1346 and 1360) while the walking action stays lit on 495 of 495 ticks. Recurrence removed: the one wired reflex is unchanged to a decimal place and every action disappears. |
| **(P3)** the prefrontal population compresses a second time | R2: silence it and measure both halves of the drive; R9: give it loops and watch it for 200 ticks | **partly supported, partly unfinished.** It supplies 0.85 of the 1.65 that crosses the threshold and removing it abolishes the behaviour. Given loops it does accumulate its own state over time, and with no mechanism for decay the accumulation is unbounded (1581 to 2916 active cells over 200 ticks) and it over-drives the motor loop. |
| **(P4)** instinct is the initial state of the sheet | R1 row 4: clear the instinct table, leave everything else untouched | **supported.** The same five brains then stand on 0/5 seeds, approach on 0/5, and settle at exactly the height of the no-sensation control. |
| **(P5)** a network cannot start from silence | R1 row 4, R3 (staged additions), R7 (blurred map) | **supported.** Random background connectivity with no written wiring produces no ordered behaviour, and each group of rules that is added buys exactly its own capability without disturbing the earlier ones. |
| **(P6)** acquisition during life needs a reward-like signal and no objective | R9: four stages, one brain, dopamine region on or off; R10 for the afferent route (Supplementary Note 2) | **supported, with a boundary.** With the dopamine region lit during teaching the sound acquires the walk on 5/5 seeds (0/30 ticks before, 199-200/200 after, standing up included); with it dark, teaching writes 0 connections and the sound still does nothing (0/200, 3/3). The behaviour does not stay upright (4/5 seeds topple 42-101 ticks in), and controls on the same brains show that is the weight-growth rule, not the reward rule. |
| **(Q4)** is local plasticity doing anything a fixed recogniser is not? | R4: which hues start the walk, in the same brains, before and after a drift, with the rule on and off, and at three step sizes | **supported, with a bounded reach.** The boundary is where the sum of two cell populations crosses: every hue summing to 1.39 or more is accepted on 5/5 and every hue at 0.97 or less is rejected on 0/5. The rule carries the three hues within 0.05 of the line across it, up to the edge of the island at 96 degrees, and no brain in any condition crosses the gap at 120-144 degrees. |

## Figure legends

**Fig. 1 | The innate wiring alone closes the loop.** Four conditions, five network seeds each:
final trunk height for a body placed prone with no instruction; forward displacement for a body
shown a red region; final trunk height and tilt for the no-sensation control; and, with the
instinct table cleared, final trunk height and forward displacement with red ahead. Sources:
`日志_闭环多种子_新.log`, `日志_白脑_新.log`; scripts `实验_闭环前提_多种子.py`,
`实验_白脑对照.py`.

**Fig. 2 | The behaviour is computed by a specific population.** Walking-action ticks out of 200,
five seeds per condition, with the mean number of active prefrontal cells annotated: red ahead with
the prefrontal intact and with it silenced; a black screen; a blue screen with the prefrontal loops
on and with the prefrontal code recomputed on every tick. Sources:
`日志_前额叶多种子_新A.log`, `日志_前额叶多种子_新B.log`; script `实验_前额叶_多种子.py`.

**Fig. 3 | The repertoire grows by adding innate structure.** A five-row matrix (stages A-E, from
the motor repertoire alone to the full table) against five behaviours, cells shaded by how many of
three build seeds passed. Source: `日志_分阶段_新.log`; script `实验_本能分阶段.py`.

**Fig. 4 | How far local plasticity moves the boundary of recognition.** (a) How far the colour may
drift before recognition stops, as a function of step size, with the correlation rule on and off,
one point per seed; (b) the same five brains, hue by hue, before any drift and after the 3-degree
sweep has run to its stopping point. Sources: `日志_渐变正式_新.log`, `日志_渐变边界_无环.log`,
`日志_色相红块_新.log`; scripts `实验_渐变_正式协议.py`, `实验_渐变_边界.py`.

**Fig. 5 | Emergent gaze tracking.** Eye azimuth and object azimuth over time for an object crossing
the field, an object appearing suddenly, and an empty scene. Source: `日志_眼睛跟随.log`
(regenerated by `实验_眼睛跟随.py` every time the figure is drawn).

**Fig. 6 | A capability acquired during the life of one brain.** (a) Walking-action ticks in each of
the four stages, five seeds: none before teaching, nearly all after; (b) connections written during
teaching with the dopamine region lit and with it dark; (c) trunk height and forward displacement in
stages 1 and 4, per seed. Sources: `日志_发育多种子_A.log`, `日志_发育多种子_B.log`,
`日志_无奖励对照.log`; script `实验_发育_声音叫走路_多种子.py`.

**Supplementary Fig. 1 | One tick of the system.** Senses drive the two sensory hierarchies, which
drive one recurrent cortical sheet; the prefrontal population compresses the already-compressed
signals a second time and returns its output to the sensory and motor regions; the motor-memory time
cells drive the 160 motor cells, which are the muscles. Generated by `出一张图_系统总览.py`.

**Supplementary Fig. 2 | The cost of hierarchical compression (R5).** (a) Cross-column sharing of
direction names in layers 1, 2 and 3 of the visual hierarchy; (b) for a ball one third of a cell
across, the columns in which each layer's newly active cells lie. Sources: `日志_方位分辨力.log`,
`日志_前几层.log`.

**Supplementary Fig. 3 | Blurred innate wiring (R7).** (a) Resting position of the object relative
to the gaze centre as a function of an imposed shift of every direction-bearing name in the table;
(b) survival of three behaviours as a function of the fraction of the table that has been
re-randomised or shifted by one column. Sources: `日志_整体偏移_正前方_新.log`, `日志_存活曲线_新.log`.

**Supplementary Fig. 4 | Where the behaviour lives (R8).** (a) Three behaviours with the intact
cortex and with the recurrent step removed; (b) walking, the visual code and the prefrontal code tick
by tick, with the stimulus removed at 0.5 s; (c) the same run with the action's time cells cleared
once at 1.0 s. Sources: `日志_切断回响_有环.log`, `日志_想还在吗_轨迹_有环.csv`.

## References

1. Brown, T. B. et al. Language models are few-shot learners. *Adv. Neural Inf. Process. Syst.* **33**, 1877-1901 (2020).
2. Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J. & Hinton, G. Backpropagation and the brain. *Nat. Rev. Neurosci.* **21**, 335-346 (2020).
3. Hubel, D. H. & Wiesel, T. N. Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *J. Physiol.* **160**, 106-154 (1962).
4. Felleman, D. J. & Van Essen, D. C. Distributed hierarchical processing in the primate cerebral cortex. *Cereb. Cortex* **1**, 1-47 (1991).
5. Mountcastle, V. B. The columnar organization of the neocortex. *Brain* **120**, 701-722 (1997).
6. Hebb, D. O. *The Organization of Behavior: A Neuropsychological Theory.* (Wiley, 1949).
7. Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. *Proc. Natl Acad. Sci. USA* **79**, 2554-2558 (1982).
8. Amit, D. J. *Modeling Brain Function: The World of Attractor Neural Networks.* (Cambridge Univ. Press, 1989).
9. Wang, X.-J. Probabilistic decision making by slow reverberation in cortical circuits. *Neuron* **36**, 955-968 (2002).
10. Fuster, J. M. The prefrontal cortex - an update: time is of the essence. *Neuron* **30**, 319-333 (2001).
11. Miller, E. K. & Cohen, J. D. An integrative theory of prefrontal cortex function. *Annu. Rev. Neurosci.* **24**, 167-202 (2001).
12. Tinbergen, N. *The Study of Instinct.* (Oxford Univ. Press, 1951).
13. Pfeifer, R. & Bongard, J. *How the Body Shapes the Way We Think: A New View of Intelligence.* (MIT Press, 2006).
14. Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature* **518**, 529-533 (2015).
15. Hwangbo, J. et al. Learning agile and dynamic motor skills for legged robots. *Sci. Robot.* **4**, eaau5872 (2019).
16. Rao, R. P. N. & Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nat. Neurosci.* **2**, 79-87 (1999).
17. Friston, K. The free-energy principle: a unified brain theory? *Nat. Rev. Neurosci.* **11**, 127-138 (2010).
18. Oudeyer, P.-Y. & Kaplan, F. What is intrinsic motivation? A typology of computational approaches. *Front. Neurorobot.* **1**, 6 (2007).
19. Izhikevich, E. M. Simple model of spiking neurons. *IEEE Trans. Neural Netw.* **14**, 1569-1572 (2003).
20. Buzsaki, G. & Draguhn, A. Neuronal oscillations in cortical networks. *Science* **304**, 1926-1929 (2004).
21. Rudin, N., Hoeller, D., Reist, P. & Hutter, M. Learning to walk in minutes using massively parallel deep reinforcement learning. *Conf. Robot Learn. (CoRL)* 91-100 (2022).
22. Grillner, S. & Wallen, P. Central pattern generators for locomotion, with special reference to vertebrates. *Annu. Rev. Neurosci.* **8**, 233-261 (1985).
23. Eichenbaum, H. Time cells in the hippocampus: a new dimension for mapping memories. *Nat. Rev. Neurosci.* **15**, 732-744 (2014).
24. Turrigiano, G. G. & Nelson, S. B. Homeostatic plasticity in the developing nervous system. *Nat. Rev. Neurosci.* **5**, 97-107 (2004).
25. Magee, J. C. & Grienberger, C. Synaptic plasticity forms and functions. *Annu. Rev. Neurosci.* **43**, 95-117 (2020).
26. Maass, W., Natschlager, T. & Markram, H. Real-time computing without stable states: a new framework for neural computation based on perturbations. *Neural Comput.* **14**, 2531-2560 (2002).
27. Eliasmith, C. et al. A large-scale model of the functioning brain. *Science* **338**, 1202-1205 (2012).
28. Markram, H. et al. Reconstruction and simulation of neocortical microcircuitry. *Cell* **163**, 456-492 (2015).
29. Gewaltig, M.-O. & Diesmann, M. NEST (NEural Simulation Tool). *Scholarpedia* **2**, 1430 (2007).
30. Kaplan, J. et al. Scaling laws for neural language models. *arXiv* 2001.08361 (2020).
