# Structural Priors and Locally Learned Relations in a Transparent Associative Architecture

**Zhiwen Li (李秩文)**  
Independent researcher  
Project: [born-wired-cortex](https://github.com/leeR1ven/born-wired-cortex)

Local journal manuscript draft | 13 September 2026 | Awaiting author review; not submitted or peer reviewed

## Abstract

Associative architectures need both acquired relationships and a structure through which those relationships influence behavior. We investigate their interaction in an inspectable model with local plasticity, recurrence, inhibitory gates, and shared-time memory. Three sensorimotor studies distinguish supplied connectivity from learned content. Sensory pairing supports correct-first target arrival in 36/36 new-layout conditions across three models, compared with 2/36 after removing learned cue links or goal recurrence. Matched recurrent topologies produce different goal-maintenance and switching outcomes. Separately learned elementary relations support previously unpaired root-cue combinations: unique-intersection responses are correct in 144/144 conditions at each of zero- and four-frame separation, but 0/144 at twelve frames. Pathway interventions and graph counterexamples expose dependence on learned relations, persistence, and independent intermediate paths. A separate symbolic experiment freezes a prior learned from 344.8 million characters and writes internally generated population sequences into the original temporal memory. On 24 new episodes with three partial-cue masks each, reinstating content and recent activity-dependent inhibition reproduces two subsequent recurrent populations in 68/72 half-cue conditions; removing recent-history restoration gives 0/72. Future populations are computed without future-memory reads. These are restricted composition and state-reinstatement results, with explicit supplied mechanisms, event-aliasing failures, and reproducible records. They do not establish spontaneous acquisition of general logic, semantic language, or general intelligence.

**Keywords:** cognitive architecture; associative learning; structural priors; relational composition; persistent activity; temporal memory

## 1. Introduction

An associative connection can store that two events occurred together, but its behavioral effect depends on the surrounding circuit. Recurrence can preserve a disappearing cue; convergence can select a jointly supported outcome; inhibition can withhold an unresolved response. Which aspects of the resulting behavior were learned, and which were supplied by the architecture?

Three sensorimotor studies and a separate symbolic memory experiment address this question. The first tests learned auditory cue meanings across changed target positions. The second changes recurrent topology while holding neuron count, edge count, weights, and degrees constant. The third teaches individual directed relations and tests root-cue combinations never presented together during teaching. A separate experiment asks whether internally generated populations can be stored and later resume recurrence. The contribution is a mechanistic account of learned content interacting with explicit structural priors and state reconstruction, including failures.

The framework integrates sensation, recurrent association, shared-time multimodal memory, and neural motor decoding. Labels such as “prefrontal” and “hippocampal” describe software functions, not anatomical equivalence. All three studies execute the full pipeline, although interventions locate demonstrated functionality in particular added pathways. Execution and causal necessity are different propositions.

An earlier single-checkpoint pet diagnostic showed that retuning competing motor currents increased associative influence while increasing collisions. It motivates measuring actual output and adverse effects, but is background rather than a primary result here.

We report outputs, interventions, learning doses, development history, and counterexamples. The model encounters new root pairs but receives a designed mechanism for combining their effects. Separating novel combinations from novel computational structures permits a positive, bounded claim about composition.

## 2. Related work and conceptual scope

Content-addressable retrieval through collective recurrent dynamics has a substantial precedent in Hopfield networks [1]. The present directed, stateful architecture includes temporal references and motor interfaces, but no Hopfield-type energy function or convergence guarantee has been demonstrated. Associative completion itself is therefore background, rather than a novelty claim.

Bi and Poo [2] demonstrated dependence of plasticity on spike timing, initial strength, and cell type. Turrigiano and colleagues [3] studied activity-dependent synaptic scaling, while Tsodyks and Markram [4] described transmission through synaptic resource dynamics. These provide comparisons with regulated learning and activity. Our bounded updates, divisive inhibition, and fatigue variables do not reproduce their timing windows, scaling mechanisms, or calibrated resource equations; saturation alone does not establish biological equivalence.

The successor representation predicts future state occupancy [5], and an integrative framework relates episodic memory to reinforcement-learning decisions [6]. Retrieving a stored motor pattern does not implement those algorithms or demonstrate prospective planning. Soar [7] integrates learning, knowledge, and decisions, while Nengo [8] supports construction and inspection of functional neural models. Modular integration and accessible activity are therefore insufficient grounds for a priority claim.

Three further studies clarify the interpretation of internal activity. Mixed selectivity in monkey prefrontal populations supports examining distributed task information [9], without identifying arbitrary coactive model cells with thoughts. Evidence for hippocampal pattern completion and cortical reinstatement motivates associative reconstruction of event elements [10], but does not validate this implementation's temporal index or imply that every memory is an exact experiential copy. Content-specific auditory attenuation during covert phoneme production supports sensory prediction in inner speech [11]; it does not establish that inner speech is recorded-sound replay or that a silent model has generated language. Population activation, modality-specific reinstatement, and overt expression therefore require separate measurements.

This literature review establishes bounded comparisons. It is not an exhaustive survey establishing that the architecture or its constituent mechanisms are unprecedented. The empirical contribution lies in the supplied-versus-learned separation and interventions reported below.

<!-- page -->

## 3. Architecture and implementation boundaries

![Functional overview of the shared-time associative architecture](figures/architecture_overview.png)

*Figure 1. Functional organization of the full sensorimotor architecture. Sensory processing, reciprocal and recurrent association, shared-time modality memories, inhibitory regulation, and neural motor decoding form an integrated loop. This is an overview rather than a strict execution schedule, and it does not establish that every depicted component is necessary for the tasks below. The figure was produced locally with Matplotlib; OpenAI Codex/GPT-6 assisted with the plotting code.*

### 3.1 A common sensory, temporal, and motor loop

The actor receives egocentric proximity and RGB ray values, body signals, synthetic auditory amplitudes, and feedback from its muscle states. Paired thermometer populations encode the continuous channel values. These inputs enter a four-layer forward network with reciprocal projections and a recurrent associative population. The recurrent system combines binary activity, weighted currents, thresholds, and rate-like propagation; it is not a calibrated spiking-neuron simulation.

The two goal studies use nine rays and six encoding levels, giving recurrent width 648; the relation study uses twenty-one rays and six levels, giving width 1,224. These activity-vector widths exclude additional forward layers, populations, and temporal addresses. The configurations do not constitute a controlled scaling comparison.

Visual, auditory, and motor memories share temporal identifiers. A ring of 1,728,000 addresses has one active temporal cell at a time. Each 0.1-second physical frame advances two 0.05-second internal ticks, giving a nominal twenty-four-hour span before wraparound. The “single active temporal neuron” represents the current shared address, not a biological neuron storing all experience. Feature-to-time and time-to-feature associations return multimodal content at a selected address; successful reconstruction depends on stored links and cue matching.

Within a frame, current input and prior associative state contribute to an initial recurrent update. Memory queries return content from selected original times into subsequent recurrent processing, whose command evidence enters motor competition. Storage, retrieval, and interaction share the same sequence. Separate trial branches, described below, are not an uninterrupted lifetime trajectory.

### 3.2 Local regulation and muscle output

Excitation, inhibition, and disinhibition regulate propagation and action release. In the added circuits, a long-range excitatory projection can recruit an interneuron near its target; inhibition is then exerted through local connections. Disinhibition reduces an existing inhibitory constraint. Activity-dependent fatigue changes effective thresholds, while recurrent connections can sustain selected activity after an external cue disappears. These mechanisms have different roles from long-term changes to synaptic weights, and short-term state can continue evolving when learning is frozen.

The motor interface outputs forward, backward, left, right, eating, and vocalization forces. A forward encoder represents muscle levels; an inverse Hebbian decoder translates selected neural commands into muscle features. “Inverse” denotes decoding direction, not backpropagation. Competition, antagonist inhibition, and output gating precede physical simulation, so a correct internal goal need not produce a timely response.

The controller boundary is the call from the brain's six outputs to the world's physical step. Stimulus programs can present teaching cards, place the body at a trial's start, and record ground truth for scoring. They do not issue the actor's movement commands. Geometry-based arrival scoring and graph-based answer checking are evaluation procedures; they are not route or answer controllers. Nevertheless, sensory encoders, retinal approach projections, propagation thresholds, and response-release structures are supplied algorithms and connections. Neural command generation should not obscure that fact.

### 3.3 What local learning changes

Plasticity acts on eligible activity relationships. In the goal studies, cue-to-color weights begin at zero and follow

\[
\Delta W_{ac}=0.08\,a\,g\,(e_c-W_{ac}),
\]

where \(a\) is auditory cue activity, \(e_c\) is current color evidence, and \(g\) is an eligibility gate. The first study uses cue-conditional updating; the second introduces a color-dominance gate. Repeated supporting evidence reduces the remaining increment as a weight approaches that evidence, while contradictory evidence can weaken a previous association. This is bounded local associative updating, without a task-level error gradient or teacher action.

The relation study instead uses bounded strengthening of eligible concept connections, with a ceiling of 1.8 and smaller increments near saturation. Concept input eligibility excludes nonconcept sources from learning excitatory inputs to the concept cells. Generic decay, protection of fixed links, inhibitory rules, and memory indexing remain explicit implementation choices. Sparse dictionaries store weighted edges and temporal references; transparency means that these structures and their causal effects can be inspected, not that every long behavior is already explained.

## 4. Experimental organization

The three sensorimotor studies in Section 5 use separate model extensions and protocols. They should not be interpreted as one animal successively acquiring all reported abilities. Table 1 identifies the boundary between supplied structure and learned content.

**Table 1. Architectural commitments and learned content.**

| Study | Supplied structure | Content acquired through teaching |
| --- | --- | --- |
| Shared target selection | Position-shared color features, goal maintenance, goal–vision conjunctions, retinal approach projections, local scanning inhibition | Two auditory cue-to-color meanings |
| Online goal updating | Three-cell recurrent topology, the preceding sensory–motor structure, color-dominance learning gate | Three auditory cue-to-color meanings and subsequent re-pairing |
| Relation composition | Fixed symbol receptive fields and pooling, concept eligibility, propagation/convergence thresholds, fatigue, unique-output release and muscle projections | Thirty-six nonself relations and twenty-eight concept self-connections per graph |

Each sensorimotor study in Section 5 contains three formal seed blocks. Layouts, cue mappings, graph queries, and topology permutations are repeated conditions within those blocks. We report raw counts and selected per-seed results; hundreds of correlated trajectories are not treated as hundreds of independently sampled models. In particular, we do not attach independent-Bernoulli confidence intervals to pooled conditions or claim population-level significance.

Development trials, formal conditions, confirmation conditions, and post hoc follow-ups are retained separately. Protocol and source hashes were saved locally for the formal runs. This provides an auditable record of the implementation and its stated schedule, but is not equivalent to externally timestamped public preregistration. Test branches begin from specified checkpoints and continue on their own shared clock without an intervening state reset. The first study additionally clears transient activity before its cue-and-delay procedure. Query learning is frozen in that study's confirmation test and remains enabled in the main conditions of the later studies, except for labeled interventions.

## 5. Studies of supplied structure and learned relations

### 5.1 Learning cue meanings and transferring target location

Three models, with seeds 7, 19, and 31, received alternating presentations of two synthetic auditory cues and full-field color cards. Seed 19 used reversed meanings. Because seed and cue mapping partly covary, this is evidence against a single fixed cue interpretation rather than an independent factorial estimate of mapping effects. Cumulative teaching doses were 0, 4, 16, 64, and 256 pairings, counted across both cues. Teaching supplied no movement trajectory, route, or action reward.

The added circuit pools opponent RGB evidence across retinal positions. Cue activity can recruit a learned color goal; fixed recurrence maintains that goal, and goal–current-vision conjunctions recruit egocentric approach currents. A local inhibitory pathway reduces interference from the existing scanning mechanism. The approach projections are engineered priors. Transfer to a new location tests use of a shared color goal through these projections, not spontaneous acquisition of spatial invariance or route planning.

Correct first turns increased from 18/36 conditions after zero or four pairings to 36/36 after sixteen, sixty-four, and 256 pairings. A first turn required the correct sign of the left–right muscle difference and an absolute difference greater than 0.02. Balanced shuffled pairing produced 18/36 first turns. Sixty-four reversed pairings produced 36/36 when scored using the new meanings. These learning probes measure immediate orientation; they do not establish full arrival after each dose or after revaluation.

After parameter selection and checkpoint saving, six additional scenes changed start pose, target position, color values, and border background. They were not used for development, although the earlier two layouts were. Each confirmation trial presented one cue frame, followed by twelve neutral frames and 120 movement frames. The apparatus held the body during the cue and delay, then released it. Correct-first arrival meant entering a radius of 0.95 simulation units around the requested target before entering that radius around the alternative.

The intact models achieved 36/36 arrivals, with 12/12 for each seed. Removing learned cue links or goal recurrent connections reduced arrival to 2/36 in each case, distributed as 1/12, 1/12, and 0/12. These results connect learned cue meaning and persistent goal activity to actual movement in new layouts. They do not demonstrate autonomous stopping to deliberate: the holding interval was imposed externally. Targets remained directly visible and approachable. Removing the local scanning inhibition in an earlier movement diagnostic still allowed 11/12 arrivals, so this pathway cannot be described as necessary for reaching the target.

### 5.2 Goal updating and matched recurrent topology

The second study allowed movement from the first cue onward. Models with seeds 11, 23, and 47 learned three distinct cue-to-color mappings through 0, 12, 48, or 192 cumulative pairings; the final dose contains sixty-four presentations per cue. Each seed was instantiated under all six permutations of three recurrent edges. The structured condition supplied three self-connections. Each alternative preserved three neurons, three edges, weight 1.05 on every edge, and one incoming and one outgoing edge per neuron. Other sensory, motor, and inhibitory connections were unchanged.

Three two-target scenes varied geometry and starting orientation. One was used in development; two were not. A one-frame cue occurred at frame zero, and a second one-frame cue at frame eight, sixteen, or twenty-four, depending on the scene. The second cue either preserved or changed the requested target. The actor continued moving and updating its memories, without a teacher action, body hold, or midtrial reset. Success required correct-first arrival after the second cue within 120 frames. No trial reached the old target before that second cue.

**Table 2. Goal updating under matched recurrent wiring.**

| Measure | Three self-connections | Five other permutations combined |
| --- | ---: | ---: |
| Correct-first arrival, unchanged goal | 18/18 | 51/90 |
| Correct-first arrival, switched goal | 14/18 | 30/90 |
| Correct goal continuously active for eight silent frames after update | 36/36 | 36/180 |
| Arrival with continuous auditory support | 6/6 | 30/30 |

The self-connected condition preserved the unchanged goal in all six arrival conditions per seed. Switching success was 4/6, 6/6, and 4/6. The topology contrast therefore supports the role of maintenance organization beyond edge number or total recurrent weight. It does not establish that self-connections are optimal for other tasks. Continuous auditory support enabled every permutation to succeed in its matched positive-control conditions, but cue weights changed in all thirty-six such trials. That control demonstrates available sensory–motor functionality; it does not isolate memory demand under strictly identical fixed weights.

At teaching doses 0, 12, 48, and 192, the structured circuit maintained the updated goal for the full eight-frame interval in 0/18, 0/18, 18/18, and 18/18 probes. Supplied recurrence thus did not replace learning of particular cue meanings. A matched movement subset yielded 6/6 intact successes, 1/6 after removing goal recurrence, and 0/6 after removing the shared motor projection. Masking the original broad PFC associative current retained 6/6. The causal conclusion concerns the added goal circuit, rather than the necessity of the complete original PFC.

A local learning gate opens when one color's retinal coverage exceeds the combined competing coverage by at least 0.3. Without it, blank-plus-sound exposure reduced the established correct weight from 0.9879 to 0.1247, and subsequent equal-color interference reduced the correct-minus-largest-incorrect margin to 0.0157. The gated condition retained 0.9879 and could learn a clear replacement mapping, reaching a new-meaning margin of 0.9760. Uneven mixtures can still produce erroneous strengthening; the gate detects dominance, not universal unambiguity. These are weight-level interference results, rather than a separate behavioral generalization test after re-pairing.

Although the principal brief-cue movement trials permitted online learning, the cue weights did not actually change in them. Their switching result concerns updating activity using previously learned meanings. The four structured timeouts all retained the correct internal goal; post hoc continuation reached it at frames 121, 129, 127, and 141. Their formal score remains 14/18. Continued pushing after arrival further exposes an incomplete goal-completion and release mechanism.

### 5.3 Composing separately learned relations

#### 5.3.1 Task, representation, and teaching

The principal study moves from a three-cell goal circuit to learned connections within the main recurrent association matrix. Each graph contains twelve root symbols, twelve independent intermediate symbols, and four terminal symbols. Every root connects to its own intermediate, which connects to two terminals. The graph therefore contains thirty-six nonself teaching relations. Let \(C(r)\) denote the two terminal outcomes reachable from root \(r\). A query should produce a categorical response only when \(C(r_1)\cap C(r_2)\) contains exactly one terminal. Set intersection defines the external scoring rule; the actor does not execute this notation or read the graph.

Twenty-eight fixed nine-bit grayscale patterns are presented at either of two retinal positions. Fifty-six supplied receptive-field detectors pool into twenty-eight concept cells placed in available population slots. Root and intermediate cells have lower propagation thresholds than terminal cells, which require converging input. Local inhibition and disinhibition release one of four existing muscle channels only when a unique terminal is supported. These symbol detectors, eligibility restrictions, convergence thresholds, and output projections are manually established. Their role is part of the proposed mechanism, not hidden preprocessing.

Each relation is taught twenty-four times. A presentation shows the source, then the target, then two neutral frames; 864 presentations and three initial neutral frames produce 3,459 teaching frames per model. Neither body nor neural activity is reset during teaching. Learned concept connectivity contains thirty-six nonself edges, each approximately 1.7212 in strength, and twenty-eight self-connections. The self-connections are acquired during training, unlike the supplied goal recurrence in the earlier studies. There is no learned root-to-terminal shortcut or stored answer for a test pair.

Development used seed 7. Formal graph seeds 201, 307, and 419 were not used before parameter freezing. Each graph supplies all sixty-six unordered distinct-root pairs: forty-eight with a unique intersection and eighteen without one. Root pairs were never jointly presented during teaching. Query trials fork from the trained checkpoint, run for twenty-four frames, and allow action from their first frame. Branching prevents associations acquired during one query from contaminating later queries; it does not remove ordinary online learning within a query.

#### 5.3.2 Responses and connection interventions

A response occurs when any designated response muscle exceeds 0.2. For a unique-answer query, the model must respond at least once and every categorical response must be correct. Nonunique queries must elicit no categorical response. This output criterion is stronger than observing a correct concept at some isolated internal microstep.

**Table 3. Formal relation-composition results across three graph seeds.**

| Separation between root cues | Correct unique-answer response | No erroneous response on nonunique queries |
| --- | ---: | ---: |
| Simultaneous | 144/144 | 54/54 |
| Four frames | 144/144 | 54/54 |
| Twelve frames | 0/144 | 54/54 |

Each seed achieved 48/48 unique responses at zero and four frames and 0/48 at twelve frames. Single-root and repeated-root controls produced no erroneous selections in 144/144 trials. Silence alone cannot establish conflict understanding: the release mechanism is designed to withhold unresolved responses, and an untrained network can also remain silent. Its interpretation depends on selective success in the unique-answer conditions.

**Table 4. Matched four-frame connection diagnostics.**

| Intervention | Correct unique-answer response |
| --- | ---: |
| Intact model | 24/24 |
| Remove learned nonself concept relations at trial start | 0/24 |
| Remove intermediate-to-terminal relations at trial start | 0/24 |
| Remove concept self-connections at trial start | 0/24 |
| Permute learned terminal relations | 0/24; all twenty-four emitted incorrect responses |
| Remove plastic association outside concept groups | 24/24 |
| Disable original episodic-content return | 24/24 |
| Freeze concept learning during the query | 24/24 |

The diagnostic subset contains eight unique-answer conditions per graph. Learning-dose checks on matched conditions yielded 0/24 after zero or eight teaching rounds and 24/24 after twenty-four. Together with the terminal permutation, the interventions support a causal role for acquired relation content. Freezing query concept learning shows that the model need not acquire a new pair-to-answer association during testing. Conversely, preserving performance without nonconcept plastic links or old episodic return limits the claim to the concept pathways used by this task. Most connection removals occur at trial start; later online updates may regenerate edges, so these are not continuous connection clamps.

#### 5.3.3 Development, temporal failure, and graph boundaries

Early development exposed a mismatch between saturated current and the terminal threshold: two learned inputs produced approximately 2.41 after saturation, below the original threshold of 3.0. A uniform threshold of 2.2 permitted convergence while a single path remained insufficient. Correct motor commands also lost competition before a late inhibition stage acted. Moving local inhibition before competition and uniformly increasing terminal motor gain addressed this failure. Formal settings were fixed after development; no test-specific edge edits were made.

The twelve-frame failure remained in the formal results. A post hoc comparison used the same three trained checkpoints and forty-frame queries in both groups. Reducing fatigue gain from 0.025 to 0.01 changed unique-answer performance from 0/144 to 144/144; both groups retained 54/54 nonunique silence. The initial learned weights matched, but query learning could subsequently diverge. This follow-up identifies a plausible persistence limitation under a uniform parameter change. It supplies no new independent graph seeds and does not replace the formal failure or establish arbitrary-delay retention.

Two off-family graphs reveal a deeper representation boundary. If one root branches through two intermediates that reconverge, two active paths can be mistaken for independent root evidence. If two roots share an intermediate, binary intermediate activity can merge their provenance and erase multiplicity. Both counterexamples fail. Their graph sizes and teaching doses differ from the formal family, so they are mechanism tests rather than fair comparative benchmarks. They expose absent source binding and evidence deduplication: the circuit tracks active intermediates without adequately representing which root supports each activation.

## 6. Population cues and shared-time episodic reinstatement

### 6.1 A pretrained prior and newly stored internal events

A separate symbolic proxy tests a narrower memory question: can activity produced by a learned associative prior be stored as a new event and later support recurrent continuation? A character labels one unit, and simultaneously active characters form an unordered population. Pretraining supplies an initial prior for this experiment; it is not literal biological innateness. Internally generated population sequences supply new event-specific feature-time bindings. The experiment does not assume that these events constitute meaningful sentences or new semantic knowledge.

The frozen prior contains 19,168 units and 13,074,732 nonzero directed connections, acquired from 344,787,049 cleaned characters of LCCC dialogue and a selected Chinese Wikipedia shard. Training removed punctuation and whitespace and retained traditional and simplified forms separately. The short eligibility trace is multiplied by 0.25 per observed character, with a 0.02 floor. Eligible source activity a increases a connection of weight w by 0.25a/(1+w), reducing further growth at high weights. No test-response targets, semantic categories or external language-model outputs are supplied to population generation. The prior is frozen throughout the new experiments; only episodic bindings are newly learned.

Generation uses the previously specified binary population rule: positive top-32 incoming currents, amplitude one for selected units, no diagonal self-current, and a three-step receiver refractory period. An already active unit still transmits its outgoing current. Warm input retains the short sensory trace; after this initialization, eight internally generated groups are recorded. The input protocol reinitializes population state between episodes, while all episodes are written successively into one pair of memory connection tables and one original shared temporal ring. The iteration counter in the population routine is not another temporal-neuron population.

The unchanged original auditory-memory class is reused here as a generic symbolic feature store. This does not simulate acoustic perception. Two internal ticks are retained per frame: previous features strengthen connections to the new time, and the previous time strengthens connections to current features. Querying the first generated group G0 can therefore select a time whose outgoing connections contain G1. Memory learning is disabled during query evaluation, the ring does not wrap in these runs, and the stored tables remain below the original forgetting threshold.

### 6.2 Content addressing and state restoration

The first protocol used the first 24 eligible held-out test records as seeds; the follow-up used the next 24. Eligibility required at least three characters in the first turn, truncated to 32 characters. Selection did not filter trajectories or results. The protocols were saved locally before their respective runs, without an external preregistration timestamp. A post-run integrity manifest additionally binds the input metadata and source split. Four cue sizes, 4, 8, 16 and 32 active units from G0, were tested under three deterministic mask seeds. Only the partial population enters retrieval. Target times, episode labels and future groups are available solely to the evaluator.

The original feature-to-time currents add across active cue units, and the earliest maximum selects an address. This is learned connection-based content addressing, implemented by sparse synapse tables; the mechanism does not receive a target-address lookup. Two adjacent internal times often tie because they represent one physical frame. Cross-episode ambiguity is measured separately.

In v1, normalized recurrent currents from the cue and returned group were added with equal gain. Although this recovered stored content, it did not resume the original trajectory. The query had lost the short-term inhibition that accompanied the generated event. The v2 follow-up explicitly changes the retrieval mechanism: the returned group reinstates current activity, and the previous two frame positions on the same selected timeline reinstate refractory counters of three, two and one for current and older activity, taking the maximum for overlapping units. The initial warm population is also stored. From that recovered state, two further populations are computed through the frozen recurrent prior, with no future-memory reads.

V2 also changes the current-integration rule and seed records relative to v1. Their overall difference cannot be assigned solely to restoring inhibition. Matched v2 interventions therefore remove memory writing, block return, retain content without past history, shift only past history, shift content and history together, or cut recurrent current. The history and content shifts use the corresponding time offset in the next episode as an intervention. The intact computation does not use episode boundaries to select its answer. With return absent, an extra forward update occupies the initial slot taken by reinstatement, aligning the continuation comparison with the stored G2 and G3 groups. All query conditions begin without the original working-state pointer, and query operations leave episodic weights unchanged.

### 6.3 Results and interpretation

V1 exact content recovery was 38/72, 56/72, 68/72 and 69/72 for the four increasing cue sizes. The subsequent-state overlap did not establish successful continuation: at 32 units its mean Jaccard overlap with the stored third group was zero. These results remain in the archive rather than being replaced by the follow-up.

V2 produced the following results. Each denominator represents 24 episodes and three masks, not 72 independent trained models.

**Table 5. Separate outcomes for memory addressing and recurrent continuation.**

| Cue units | Exact event address | Exact returned G1 | Both computed G2 and G3 exact |
|---|---:|---:|---:|
| 4 | 44/72 | 49/72 | 49/72 |
| 8 | 55/72 | 61/72 | 61/72 |
| 16 | 62/72 | 68/72 | 68/72 |
| 32 | 63/72 | 69/72 | 69/72 |

At 16 units, intact reinstatement reproduced both future groups in 68/72 conditions; 22 of the 24 episodes succeeded under all three masks. No writing, blocked return, content without recent history, and cut recurrent current each gave 0/72. Shifting only history or shifting content and history gave 6/72 each. At 32 units, the no-memory and return-blocked conditions gave 9/72. Full cues are identical across the three masks, so the intact 32-unit result is equivalently 23/24 episodes.

![Memory continuation and matched interventions](figures/journal_memory_continuity.png)

*Figure 2. Partial-cue continuation and matched interventions in v2. Success requires exact recovery of both populations computed after reinstatement, rather than merely copying a returned group. The plotted counts are recomputed from frozen trial records by the supplied Matplotlib script. OpenAI Codex (GPT-6, OpenAI; September 2026 session) assisted with plotting code; no generative image model produced or altered experimental results. Repeated masks share episodes; no independent-seed confidence interval is implied.*

Only 21 distinct initial groups occurred among the 24 v2 episodes. An incorrect event address can return the same population and continuation as the intended event; thus content and continuation scores exceed event-identification scores. Cross-episode top-current ties occurred in 40, 29, 19 and 15 of the 72 conditions for increasing cue sizes. This is a measured aliasing limitation, not evidence that a larger population necessarily resolves event identity.

The internally stored groups can therefore become usable episodic state in this proxy. The within-v2 comparisons identify contributions from past-state reconstruction and recurrent propagation. Restoring sufficient state in a deterministic system can reproduce its trajectory; doing so is not the discovery of a new logical rule. These tests do not establish semantic thought, inner speech, motor expression, long-term consolidation into the pretrained prior, or full PetBrain memory necessity. They refine the population/reinstatement distinction motivated by prior work [9-11] with a specific computational result, without claiming biological equivalence.

## 7. Discussion

The three sensorimotor studies identify complementary contributions of structure and learning. Sensory pairing determines which color a sound requests; a position-shared approach circuit uses that learned request in changed layouts. Recurrent topology determines whether a transient request persists and can guide movement after an update. In the relational task, acquired edges carry specific factual content, while designed convergence and response-release circuitry determine how two pathways jointly affect the output. The successful novel root combinations were absent from teaching, even though the computational arrangement supporting them was supplied.

Association and logical computation are not mutually exclusive descriptions. “Associative” identifies how relationships are represented or acquired; “logical” can describe a rule-governed input–output operation implemented through those relationships. Here, a restricted intersection-like selection is realized through weighted propagation, convergence thresholds, and inhibition. The evidence supports that operation within the stated graph family. It does not establish that the architecture discovered the operation itself, learned arbitrary relational rules, or can bind sources under changing graph structures. Simply calling the mechanism Hebbian would overlook the demonstrated composition; calling its behavior general reasoning would overlook the supplied rules and observed failures.

The interventions locate successful computation: goal selection survives masking of broad PFC associative current, and relation composition survives removal of original episodic return. A small embedded circuit can therefore demonstrate a capacity without establishing the necessity of surrounding machinery. Section 6 supplies a separate symbolic test: learned feature-time connections address a stored population, and reconstructed recent state enables recurrent continuation without future-memory reads. The paired v2 controls identify the role of recent inhibition. This does not establish full PetBrain memory necessity.

Generalization covers changed layouts, cue re-pairing in specified probes, new graphs within one construction, and previously unpaired root combinations. Learned perception, language, hidden-goal route composition, indefinite delay, arbitrary graphs, and effects of scale remain untested. Three seed blocks do not establish robustness across a broad initialization or curriculum space.

The work is best assessed as a reproducible mechanism construction with identified failure modes. No matched-budget comparison establishes superiority over a simpler associative circuit, finite-state controller, or graph algorithm supplied with equivalent priors. Future comparisons should hold input information and representational assistance constant. Stronger architectural tests would require newly sampled graph families, source-sensitive representations, and tasks whose success changes under interventions on the proposed temporal feedback. Such extensions should preserve failed conditions and distinguish uniform mechanism changes from answer-specific repairs.

## 8. Reproducibility and evidence availability

The historical evidence is retained in three versioned reproduction archives containing frozen source, protocols, checkpoints, connection listings, raw trial records, and verification programs. A local audit on 13 September 2026 checked external archive SHA256 values, ZIP integrity, and 546 manifest entries. Current protocol-listed sources matched the frozen hashes. Verification used Windows, CPython 3.13.7, and NumPy 2.5.2 on a local CPU; a GPU was not required.

Independent scoring code recomputed 516 shared-goal conditions; 702 online-goal trajectories containing 47,952 recorded frames; and 1,062 formal relation trajectories containing 25,488 query frames. Relation teaching contains 10,377 frames across the three models. An additional 398 trajectories cover post hoc retention and boundary tests. “Independent scoring” denotes a separate scoring procedure using geometry, actual muscle output, and graph intersections, not replication by an independent laboratory.

Checkpoint replays covered nine shared-goal trajectories, eighteen online-goal trajectories, eighteen formal relation trajectories, and three retention trajectories: 3,792 frames in total. Discrete states matched; numerical checks used absolute tolerance 10^-12 and zero relative tolerance. The online-goal and formal relation replays had zero maximum force and position differences. Replay calls the frozen original implementation; it is not an independent algorithmic reimplementation or full retraining. Relation teaching logs omit some raw RGB inputs, so verification reconstructs fixed-symbol stimuli and checks their visual temporal encodings rather than claiming an entirely independent replay of all teaching.

The symbolic experiment uses CPython 3.13.7 and NumPy 2.4.6. A separate verifier, without importing either experimental runner, the original memory class or the population class, reconstructs all 384 generated groups, memory connections, addressing currents and 3,168 query-condition records across both protocols. Original runners were also rerun in unused directories. The full prior is supplied locally as 19 lossless CSR shards, totaling 41,629,785 bytes with metadata; all reconstructed rows were compared bitwise with the original dense matrix, and an independent check recomputes its complete semantic digest. This reduces storage rather than neuron count or connectivity. It permits the new experiments to run offline on CPU without the dense 1.47-GB file or raw text. Full corpus pretraining is not rerun by this verification.

Data provenance and source notices accompany the review materials. Raw LCCC dialogue and Wikipedia text remain external. Dataset restrictions and derived-weight redistribution must be checked separately from the code license; the local sparse prior is not represented as an already authorized public weight release. A final archival repository identifier has not been issued.

The public repository is the project entry point. The three audited archives and their checksum files are retained for the submission materials; this draft does not assert that their final journal-linked release has already been published. The submission must identify the exact manuscript, archive, and evidence-index versions together.

## 9. Conclusion

Supplied structural priors and locally acquired associations jointly support the measured cue-guided behaviors and restricted relation composition. Connection interventions identify the learned content and recurrent pathways that matter, while delay failures and source-binding counterexamples constrain the interpretation. The separate memory experiment shows that internally generated activity can acquire retrievable event bindings, and that restoring recent local state can enable subsequent recurrence. These results provide inspectable evidence for specific computational mechanisms. They establish neither general intelligence nor an impossibility result for the broader architecture, and they motivate tests that connect additional proposed mechanisms to behavior under explicit controls.

## Declaration of generative AI and AI-assisted technologies

OpenAI Codex (GPT-6, OpenAI; September 2026 session) assisted substantially with experimental code, diagnostic analysis, source checking, manuscript drafting, and preparation of figures and layout. AI tools are not listed as authors. This is a local draft awaiting human review: Zhiwen Li must review and take responsibility for the code, data interpretation, citations, and final text before submission. This statement does not assert that author approval or the journal's required declarations have already been completed.

<!-- page -->

## References

[1] Hopfield J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences of the United States of America, 79(8):2554–2558. [doi:10.1073/pnas.79.8.2554](https://doi.org/10.1073/pnas.79.8.2554)

[2] Bi G., Poo M. (1998). Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type. The Journal of Neuroscience, 18(24):10464–10472. [doi:10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998)

[3] Turrigiano G.G., Leslie K.R., Desai N.S., Rutherford L.C., Nelson S.B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. Nature, 391(6670):892–896. [doi:10.1038/36103](https://doi.org/10.1038/36103)

[4] Tsodyks M.V., Markram H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. Proceedings of the National Academy of Sciences of the United States of America, 94(2):719–723. [doi:10.1073/pnas.94.2.719](https://doi.org/10.1073/pnas.94.2.719)

[5] Dayan P. (1993). Improving Generalization for Temporal Difference Learning: The Successor Representation. Neural Computation, 5(4):613–624. [doi:10.1162/neco.1993.5.4.613](https://doi.org/10.1162/neco.1993.5.4.613)

[6] Gershman S.J., Daw N.D. (2017). Reinforcement Learning and Episodic Memory in Humans and Animals: An Integrative Framework. Annual Review of Psychology, 68:101–128. [doi:10.1146/annurev-psych-122414-033625](https://doi.org/10.1146/annurev-psych-122414-033625)

[7] Laird J.E., Newell A., Rosenbloom P.S. (1987). SOAR: An architecture for general intelligence. Artificial Intelligence, 33(1):1–64. [doi:10.1016/0004-3702(87)90050-6](https://doi.org/10.1016/0004-3702(87)90050-6)

[8] Bekolay T., Bergstra J., Hunsberger E., DeWolf T., Stewart T.C., Rasmussen D., Choo X., Voelker A.R., Eliasmith C. (2014). Nengo: a Python tool for building large-scale functional brain models. Frontiers in Neuroinformatics, 7:48. [doi:10.3389/fninf.2013.00048](https://doi.org/10.3389/fninf.2013.00048)

[9] Rigotti M., Barak O., Warden M.R., Wang X., Daw N.D., Miller E.K., Fusi S. (2013). The importance of mixed selectivity in complex cognitive tasks. Nature, 497(7451):585–590. [doi:10.1038/nature12160](https://doi.org/10.1038/nature12160)

[10] Horner A.J., Bisby J.A., Bush D., Lin W., Burgess N. (2015). Evidence for holistic episodic recollection via hippocampal pattern completion. Nature Communications, 6:7462. [doi:10.1038/ncomms8462](https://doi.org/10.1038/ncomms8462)

[11] Whitford T.J., Jack B.N., Pearson D., Griffiths O., Luque D., Harris A.W.F., Spencer K.M., Le Pelley M.E. (2017). Neurophysiological evidence of efference copies to inner speech. eLife, 6:e28197. [doi:10.7554/eLife.28197](https://doi.org/10.7554/eLife.28197)
