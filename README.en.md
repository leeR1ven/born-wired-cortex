# Born wired - innate cortical connectivity plus local plasticity is enough

**One sentence.** 143,796 cortical cells, 308,809 excitatory synapses written in at birth, and a
single local rule (a cell that fired on the previous tick strengthens its synapse onto a cell that
fires on this tick). No reward function, no backpropagation, no objective, no token in / token out.
The body still rights itself from lying down, walks to a red patch it sees, and turns its gaze onto
a moving ball.

This repository holds the **code, logs, figures and data** of the manuscript
*Born wired: innate cortical connectivity plus local plasticity is enough to stand, walk and look*
by Li Zhiwen (Independent Researcher, ORCID 0009-0005-8289-6393, rivenlee94@gmail.com).

| Directory | Contents |
|---|---|
| `paper/` | manuscript (Markdown and PDF), the ten figures (`paper/figures/`), supplementary methods |
| `model/` | the cortex simulator (156 scripts) plus its data tables (`本能表.txt`, `动作库*.json`, ...) |
| `logs/` | every run log the manuscript cites (292 files); each number in the paper has one |
| `snapshots/` | process snapshots (standing up, walking, gaze following) |
| `playback/` | four-panel playback page; open `playback/回放_大脑_四幕.html` in any browser |
| `experiments/` | staged sub-experiments (repertoire distillation, colour-beacon navigation, closed loop, cognition, text-network) |
| `docs/` | architecture notes, rule table, wiring audit, reproduction guide, progress log |

Only material belonging to this manuscript is here; unrelated work (a 3.9 GB pet-home simulation,
56 GB of language corpora, a legacy architecture archive, outreach material) stays in the older
development repository.

## Reproducing

```powershell
cd model
python -X utf8 核对_论文数字.py        # seconds: checks the paper's 52 numbers against logs/
python -X utf8 核对_论文数字.py 重跑    # ~6 minutes: re-runs R1 and compares
python -X utf8 看_大脑开车.py          # regenerates the playback page
```

Section 5.8 of the manuscript names the script for every result (R1-R11). `docs/复现说明.md` is
the full number-to-script-to-log table. Everything runs on CPU; the core needs only Python 3 +
NumPy, the body uses MuJoCo (the Go2 model is fetched automatically if missing), figures use
matplotlib, and only the third-party baseline scripts need PyTorch plus an external checkpoint.

## Honest limitations

After learning "hearing the sound makes me stand up and walk", the model **eventually falls over**
(all five seeds fall during the teaching stage; in the exam stage some seeds fall between tick 45
and tick 101). This is a stated gap in Section 4.6, not a hidden one. It is not caused by weights
growing (265,436 synapses into the motor region, zero change over 200 exam ticks), not by the
return line, and not repaired by global weight decay (a negative result the paper reports).
The real cause is that the prefrontal current onto the 15 walking time cells equals the walking
chain's ignition threshold (0.850) exactly, so any drift in the thought knocks the chain out.

Licence: MIT (`LICENSE`). (c) 2026 Li Zhiwen.  AI assistance is declared in Section 5.10.
