# SkillMaster: Additional Rebuttal Evidence

This document provides the detailed evidence referenced in our rebuttal. It covers:

1. quantitative skill-management dynamics during training;
2. examples of `propose_skill` and `update_skill` decisions;
3. an independently observed, non-probe reuse of an updated skill;
4. empty-bank reconstruction on ALFWorld and WebShop; and
5. representative ALFWorld failure-to-success trajectory analyses.


## 1. Skill-Management Dynamics During Training

The ALFWorld skill bank starts with 58 skills. The table reports all executed management decisions at each checkpoint. 
| Step | `keep_skill` | `propose_skill` | `update_skill` | Cumulative bank size |
|---:|---:|---:|---:|---:|
| 5 | 81.25% | 9.38% | 9.38% | 70 |
| 10 | 89.06% | 6.25% | 4.69% | 78 |
| 15 | 90.62% | 5.47% | 3.91% | 85 |
| 20 | 94.53% | 3.12% | 2.34% | 89 |
| 25 | 96.09% | 3.12% | 0.78% | 93 |
| 30 | 96.09% | 2.34% | 1.56% | 96 |
| 35 | 93.75% | 5.47% | 0.78% | 103 |
| 40 | 92.19% | 7.03% | 0.78% | 112 |
| 45 | 93.75% | 3.12% | 3.12% | 116 |
| 50 | 97.66% | 1.56% | 0.78% | 118 |

The combined edit rate falls from 18.76% at step 5 to 2.34% at step 50, while `keep_skill` rises from 81.25% to 97.66%. The bank continues to grow, but its growth slows as the policy increasingly preserves existing entries.

## 2. Skill-Management Examples

The complete analysis pool contains 214 non-`keep_skill` decisions:

| Environment | `propose_skill` | `update_skill` | Total non-keep |
|---|---:|---:|---:|
| ALFWorld | 72 | 52 | 124 |
| WebShop | 81 | 9 | 90 |
| Total | 153 | 61 | 214 |

The following compact sample illustrates the decisions discussed in the rebuttal. The evidence column describes the episode evidence that led to the management decision.

| Environment | Decision | Episode task | Proposed or revised skill | Trajectory evidence / decision rationale |
|---|---|---|---|---|
| ALFWorld | `propose_skill` | Cool a tomato and place it in a microwave | Cooling Search Efficiency | The agent spent most of the episode exhaustively searching cabinets and drawers and found the tomato too late, motivating earlier rotation toward food-heavy locations. |
| ALFWorld | `update_skill` | Heat a tomato and place it on a dining table | `Open Then Heat` -> `Heat While Holding Target` | Heating attempts failed until the agent returned to the microwave while holding the tomato and executed the explicit microwave heat action. |
| ALFWorld | `propose_skill` | Put two spray bottles in a garbage can | Inventory Limit for Same-Type Objects | Taking the second bottle while holding the first did not advance the task. The reusable correction is to deliver the held object, return for the second object, and track the remaining count. |
| WebShop | `propose_skill` | Buy a loose-fit women's top satisfying sleeve, color, size, and price constraints | Screen Results for Required Attributes | The clicked result did not satisfy the required sleeve and fit constraints, motivating attribute screening before opening a candidate. |
| WebShop | `propose_skill` | Buy men's pants satisfying drawstring, waist, color, size, and price constraints | Verify Product Type Before Buying | The agent bought shorts instead of pants because it prioritized matching color and size while overlooking the product-type mismatch. |
| WebShop | `propose_skill` | Buy a men's shirt satisfying material, fit, color, and size constraints | Include Core Attributes in Query | The search query omitted or poorly organized core constraints and failed to retrieve the intended product. |

The next section supplies stronger evidence than a management decision alone: an updated skill is retrieved and used by a different, later task that was not a utility probe for the edit.

### 2.1 Skill-bank consistency and lifecycle audit

We additionally audited the logged ALFWorld entries by title, principle, and application condition. We found no accepted pair that prescribes directly contradictory action orders, and no superseded version retained alongside its replacement. The lifecycle behavior is visible in the update record: `update_skill(hea_003)` replaces `Open Then Heat` in place with `Heat While Holding Target`, so the outdated instruction and its correction do not coexist as separate active entries.

Several search skills are related but not contradictory. `Systematic Exploration` defines complete coverage, category priors identify where to start, and search-rotation skills specify when to leave a low-yield location type after repeated misses. Their scopes and activation conditions are complementary. The audit therefore found neither a retained obsolete version after an update nor a directly conflicting accepted proposal.

## 3. Independently Observed Non-Probe Skill Reuse

### 3.1 Edit-producing episode

| Field | Value |
|---|---|
| Task | Put a hot tomato on a dining table |
| Trajectory ID | `aae37844-c66b-4fff-81ed-2a8f99222ab8` |
| Outcome | Success |
| Episode length | 48 actions |
| Retrieved heat skill | `hea_003: Open Then Heat` |
| Management decision | `update_skill(hea_003)` |

The retrieved skill stated:

> **Open Then Heat.** Upon reaching the microwave with the target in hand, open the door, place the object inside, and execute the heat action before leaving.

The episode exposed an operational problem with this instruction. After acquiring the tomato, the policy spent many actions trying incomplete or ineffective heat operations. The decisive transition occurred at the end of the trajectory:

| Action index | Observation before action | Action | Result |
|---:|---|---|---|
| 43 | At the dining table while carrying the tomato | `move tomato 1 to diningtable 1` | The tomato is placed on the dining table. |
| 44 | Tomato is on the dining table | `take tomato 1 from diningtable 1` | The tomato is held again. |
| 45 | Tomato is held | `go to microwave 1` | The agent reaches the microwave. |
| 46 | At the open microwave while holding the tomato | `heat tomato 1 with microwave 1` | The environment confirms: `You heat the tomato 1 using the microwave 1.` |
| 47 | Tomato is successfully heated and held | `go to diningtable 1` | The agent returns to the destination. |
| 48 | At the dining table with the heated tomato | `move tomato 1 to diningtable 1` | The task succeeds. |

The manager replaced the old entry with:

> **Heat While Holding Target.** For ALFWorld heat tasks, keep holding the target object and execute the heat action with the microwave. Do not place the object inside the microwave before heating; if heating returns `Nothing happens` after the object is inside, take it back into hand before retrying.

The tool result records `status=updated`, `dry_run=false`, and `bank_mutated=true`.

### 3.2 Later natural episode

A later batch in the same validation stream contains a different object and game instance:

| Field | Value |
|---|---|
| Task | Heat a plate and put it on a dining table |
| Trajectory ID | `b0426cf6-733e-4588-8255-639e0965b4cd` |
| Outcome | Success |
| Episode length | 15 actions |
| Retrieved revised skill | `hea_003: Heat While Holding Target` |

The complete action trace is:

| Action index | Action | Environment result |
|---:|---|---|
| 1 | `go to cabinet 1` | Reaches cabinet 1. |
| 2 | `open cabinet 1` | Empty. |
| 3 | `go to cabinet 2` | Reaches cabinet 2. |
| 4 | `open cabinet 2` | Empty. |
| 5 | `go to cabinet 3` | Reaches cabinet 3. |
| 6 | `open cabinet 3` | Empty. |
| 7 | `go to cabinet 4` | Reaches cabinet 4. |
| 8 | `open cabinet 4` | Empty. |
| 9 | `go to cabinet 5` | Reaches cabinet 5. |
| 10 | `open cabinet 5` | Plate 3 is visible. |
| 11 | `take plate 3 from cabinet 5` | Plate 3 is held. |
| 12 | `go to microwave 1` | Reaches the microwave while holding the plate. |
| 13 | `heat plate 3 with microwave 1` | The environment confirms that plate 3 is heated. |
| 14 | `go to diningtable 1` | Reaches the destination. |
| 15 | `move plate 3 to diningtable 1` | The task succeeds. |

The subsequent management action is `keep_skill`. Its logged rationale explicitly states that the successful behavior is already covered by `Systematic One-Pass Search`, `Heat While Holding Target`, and `Direct Post-Heat Placement`.

This episode is not a utility probe for the tomato edit. It uses a different object, a different game file, and occurs later in the natural validation rollout stream. It therefore provides independent qualitative evidence that the revised skill was retrieved and reused outside the probe that supplied the edit reward.

## 4. Empty-Bank Reconstruction

### 4.1 Protocol and WebShop results

We evaluate the same frozen WebShop checkpoint on the same fixed set of 128 test goals under three conditions:

1. **Static empty bank:** the original bank is empty, retrieval returns no skill text, and bank mutation is disabled.
2. **Mutable empty bank:** the original bank is empty, but the policy may propose skills from completed trajectories; later episodes may retrieve those generated skills.
3. **Complete bank:** the original skill bank is available.

No original skill text is exposed in either empty-bank condition.

| WebShop setting | Task score (%) | Success rate (%) |
|---|---:|---:|
| Empty bank; reconstruction disabled | 72.35 | 65.63 |
| Empty bank; reconstruction and retrieval enabled | 78.45 | 71.88 |
| Complete bank; same frozen checkpoint | 93.55 | 85.16 |

Reconstruction recovers 6.10 task-score points and 6.25 success-rate points over the static empty bank. The complete-bank gap remains substantial, so the appropriate conclusion is partial, task-dependent internalization rather than complete internalization.

### 4.2 ALFWorld reconstruction mapping

Starting from an empty bank, the frozen ALFWorld policy writes 10 skills. Nine have a clear semantic counterpart, or reconstruct a composition of counterparts, in the original bank. `Food Target Must Be Fresh` is a new state-specific extension rather than a reconstruction of a specific original entry.

| Rebuilt ID | Reconstructed skill | Core reconstructed rule | Original-bank correspondence | Match type |
|---|---|---|---|---|
| `dyn_001` | Cool Before Placement | Cool the target at the fridge before final placement. | `coo_004 Enforce Cooling Before Placement` | Direct |
| `dyn_002` | Seek Likely Locations First | Prioritize likely receptacles, then continue systematic coverage. | `cle_006 Use Location Priors`; `gen_001 Systematic Exploration` | Composition |
| `dyn_003` | Place After Acquisition | Once the target is held, go directly to its destination. | `pic_004 Place Before More Search`; `gen_003 Destination First Policy` | Direct/composition |
| `dyn_004` | Take-Clean-Place Fork | Acquire, clean at the sink, and then place. | `cle_001 Phase-Ordered Plan`; `cle_002 Pick Before You Wander`; `cle_003 Sink First for Cleaning` | Composition |
| `dyn_005` | Food Target Must Be Fresh | Reject an already heat-treated food target after a failed heat attempt. | No single original counterpart | New extension |
| `dyn_006` | Cool Search Should Not Waste Steps | Rotate among food-heavy search zones after repeated misses. | `coo_007 Food-Target Search Prior` | Direct |
| `dyn_007` | Return for Second Object | After delivering one object, return for the remaining object. | `pic_005 Track Counts for Multiples`; `gen_004 Track Counts & Progress` | Direct/composition |
| `dyn_008` | Stoveburner Search | Rotate among plausible kitchen zones for cookware-like targets. | `coo_006 Cookware Search Priority` | Direct |
| `dyn_009` | Examine with Desklamp | Turn on the lamp and acquire the target for examination. | `loo_002 Switch Lamp On`; `loo_003 Acquire Or Fetch Target` | Composition |
| `dyn_010` | Examine with Lamp | When holding the target at the lamp, use the lamp to complete the task. | `exa_003 Shortcut To Tool` | Direct |

The table below counts later episodes whose retrieved prompt contains each rebuilt title. Success counts are descriptive associations, not isolated causal effects, because an episode may retrieve multiple skills.

| Reconstructed skill | Episodes retrieving it | Successful episodes |
|---|---:|---:|
| Cool Before Placement | 21 | 19 |
| Seek Likely Locations First | 42 | 42 |
| Place After Acquisition | 42 | 42 |
| Take-Clean-Place Fork | 15 | 14 |
| Food Target Must Be Fresh | 22 | 22 |
| Cool Search Should Not Waste Steps | 12 | 11 |
| Return for Second Object | 32 | 32 |
| Stoveburner Search | 10 | 10 |
| Examine with Desklamp | 3 | 3 |
| Examine with Lamp | 3 | 3 |

The five correspondences highlighted in the rebuttal (`Cool Before Placement`, `Seek Likely Locations First`, `Place After Acquisition`, `Cool Search Should Not Waste Steps`, and `Return for Second Object`) account for 149 retrieval occurrences and 146 successful corresponding episodes. 

### 4.3 WebShop reconstruction mapping

Starting from an empty bank, the frozen WebShop policy writes eight skills:

| Rebuilt ID | Reconstructed skill | Core reconstructed rule | Original-bank correspondence | Match type |
|---|---|---|---|---|
| `dyn_001` | Avoid Overly Specific Queries | Avoid redundant variant strings that reduce search recall. | `err_005`: rare color or variant terms can hurt recall | Direct |
| `dyn_002` | Screen Result List for Required Attributes | Check visible hard constraints before opening a candidate. | `gen_001 Include Hard Constraints In Search` | Extension |
| `dyn_003` | Search for Exact Product and Select Options | Search, open the intended family, select required options, then buy. | `oth_001 Simple Purchase Flow` | Direct |
| `dyn_004` | Broad Search Query for Specific Attributes | Search with the product type and essential attributes. | `gen_001 Include Hard Constraints In Search` | Direct |
| `dyn_005` | Include Required Attributes and Select Options | Combine hard-constraint search with explicit on-page option selection. | `gen_001 Include Hard Constraints In Search`; `oth_001 Simple Purchase Flow` | Composition |
| `dyn_006` | Handle Overloaded Queries | Balance required constraints against query over-specificity and screen results. | `gen_001 Include Hard Constraints In Search`; `err_005` | Composition |
| `dyn_007` | Overloaded Query Search | Keep only necessary attributes in the query. | `gen_001 Include Hard Constraints In Search`; `err_005` | Composition |
| `dyn_008` | Verify Option Before Buying | Confirm required visible options before purchase. | `err_003`: buying before selecting mandatory options | Direct |

The first eight episodes have no retrievable skills and produce these eight writes. Of the remaining 120 episodes, all 120 retrieve at least one rebuilt skill. The first reconstructed skill, `Avoid Overly Specific Queries`, appears in all 120 later prompts; later skills enter the stream progressively as they are written.

The reconstructed entries are semantically similar to rules in the withheld original banks even though the original text is never shown. Together with later retrieval and the performance recovery over the static empty bank, this supports the narrower claim that part of the procedural knowledge is retained in the policy parameters and can be regenerated from trajectory evidence.

## 5. Sources of ALFWorld Gains

### 5.1 Aggregate control

The SFT+GRPO control uses the same SFT initialization and initial skill bank as SkillMaster. Results are mean `+/-` standard deviation over three runs.

| Family | SFT+GRPO | SkillMaster |
|---|---:|---:|
| Pick | 93.8 +/- 0.8 | 100.0 +/- 0.0 |
| Look | 92.9 +/- 1.4 | 100.0 +/- 0.0 |
| Clean | 93.6 +/- 1.1 | 100.0 +/- 0.0 |
| Heat | 91.7 +/- 0.8 | 97.1 +/- 0.3 |
| Cool | 93.3 +/- 0.5 | 95.7 +/- 1.5 |
| Pick2 | 92.8 +/- 0.7 | 100.0 +/- 0.0 |
| Overall | 92.2 +/- 0.8 | 98.7 +/- 0.4 |

### 5.2 Representative trajectory audit

The observed management patterns were produced from failed trajectories of related tasks within the same task family. To illustrate the corresponding behavioral changes, we then identified a failed and a successful validation rollout from the same ALFWorld game instance for each family.

| Family | Observed management pattern | Same-game failure-to-success comparison |
|---|---|---|
| Pick | `propose_skill: Rotate Search Zones` | Tomato -> microwave: a 50-action cabinet/drawer sweep failed. A successful rollout checked three countertops, rotated to the fridge, acquired the tomato, and placed it in the microwave in 9 actions. |
| Look | `propose_skill: Examine Target After Lamp On` | Vase under a desk lamp: activating the lamp and issuing generic `look`/desk examinations failed. A successful rollout went to the desk, activated the lamp, interacted with the vase, and completed in 3 actions. |
| Clean | `propose_skill: Phase-Ordered Plan` | Clean ladle -> dining table: a 50-action cabinet sweep never reached cleaning. A successful rollout acquired the ladle, cleaned and checked it at the sink, returned, and placed it in 30 actions. |
| Heat | `update_skill: Open Then Heat -> Heat While Holding Target` | Hot tomato -> fridge: placing the tomato inside first led to ineffective microwave interactions; after heating, a cooling action destroyed the hot state and the policy entered a recovery loop. A successful rollout heated while holding and placed the tomato in 9 actions. |
| Cool | `propose_skill: Food Search Shortcut` | Cool tomato -> microwave: a 50-action rollout found and cooled the tomato too late to place it. A successful rollout left the cabinet sweep earlier, acquired and cooled the tomato, and completed placement in 35 actions. |
| Pick2 | `propose_skill: Check Inventory Before Taking` | Two spatulas -> drawer: the failed rollout reached the two spatulas at the end of its 50-action budget and could not take the second while holding the first. A later same-game rollout delivered the first spatula before returning for the second and completed both placements in 20 actions. |


#### Pick: rotating search zones

Failures on related Pick tasks prompted the manager to propose `Rotate Search Zones`: after repeated misses in one location type, switch to another plausible zone while preserving systematic coverage. In the representative tomato-to-microwave instance, the failed rollout spends all 50 actions moving through cabinets 1-26 and drawers 1-6, never reaching a food-heavy zone or acquiring the tomato. Consistent with the learned management pattern, the successful rollout checks countertops 1-3, rotates to the fridge, opens it, takes the tomato, and immediately delivers it to the microwave.

#### Look: completing the target interaction after lamp activation


Failures on related Look tasks prompted the manager to identify `Examine Target After Lamp On`, linking satisfaction of the lighting precondition to interaction with the specified target. In the representative vase-under-lamp instance, the failed rollout reaches the correct desk and activates the lamp but repeatedly examines the desk or issues a generic `look`, without completing the target interaction. The successful rollout follows the identified correction and completes the task in exactly three actions: `go to desk 1`, `use desklamp 1`, and `take vase 4 from desk 1`.

#### Clean: preserving budget for the ordered execution phases


Failures on related Clean tasks prompted the manager to propose `Phase-Ordered Plan`, which preserves separate budgets for acquisition, cleaning, verification, and placement. In the representative clean-ladle instance, the failed rollout spends all 50 actions on cabinets and nearby surfaces and never reaches the execution phases. The successful rollout follows the phase-ordered behavior: it exits the search earlier, finds the ladle, cleans it at the sink, verifies its state, and returns it to the dining table.

#### Heat: correcting the microwave operation and preserving state


Failures on related Heat tasks prompted the manager to update `Open Then Heat` to `Heat While Holding Target`, correcting the operational sequence and emphasizing preservation of the transformed state. In the representative hot-tomato instance, the failed rollout places the tomato inside the microwave, repeatedly opens and closes the appliance after ineffective attempts, finally heats the tomato, and then executes `cool tomato 1 with fridge 1`, destroying the required hot state. The successful rollout exhibits the corrected procedure: it heats the tomato while holding it, navigates directly to the fridge, and places it without undoing the hot state.

#### Cool: rotating toward plausible food locations


Failures on related Cool tasks prompted `Food Search Shortcut`: rotate among fridges, countertops, dining tables, sink basins, microwaves, and other plausible food zones after repeated misses. In the representative cool-tomato instance, the failed rollout spends its first 45 actions in a cabinet sweep, then finds and cools the tomato but reaches the microwave at action 50 without enough budget to place it. The successful rollout exhibits the proposed search-rotation behavior, leaves the cabinet sequence earlier, takes and cools the tomato, and completes placement at action 35.

#### Pick2: inventory-aware sequential delivery


Failures on related Pick2 tasks prompted the manager to propose `Check Inventory Before Taking`, which captures the one-object inventory constraint and the need for sequential delivery. In the representative two-spatula instance, the failed rollout finds both objects near the end of its budget and attempts to take the second while still holding the first, ending without placing either object. The successful rollout follows the inventory-aware sequence: it delivers the held spatula first, returns for the remaining one, and completes both placements at action 20.

Across the six families, management behaviors identified from related-task failures align with changes in both pre-acquisition and post-acquisition execution. Pick, Clean, and Cool show better search-zone rotation and task-phase transitions, while Look, Heat, and Pick2 show improved target interaction, transformed-state preservation, and multi-object tracking. The remaining Heat and Cool failures are concentrated in compositional state tracking: one Heat trajectory cancels a successful heat operation by cooling the object, while one Cool trajectory attempts placement before completing and retaining the required cooled state.


## 6. Additional ALFWorld and WebShop Cases

### Response to Concern 2 For Review Z5L6

The following cases provide the additional trajectory-level evidence referenced in the rebuttal. 

| Env./Family | Decision | Episode task | Skill | Trajectory evidence |
|---|---|---|---|---|
| ALFWorld/Pick | `propose_skill` | Put a tomato in a microwave | Rotate Search Zones | The rollout spent all 50 actions sweeping cabinets 1-26 and drawers 1-6, never entered a food-heavy search zone, and never acquired the tomato. The proposed rule switches to another plausible receptacle type after repeated misses instead of exhausting the budget in one zone. |
| ALFWorld/Look | `propose_skill` | Look at a vase under a desk lamp | Examine Target After Lamp On | The agent reached the correct desk and activated the lamp, but then examined the desk or issued generic `look` actions without interacting with the vase. The proposed rule connects lamp activation to the remaining target-object interaction. |
| ALFWorld/Clean | `propose_skill` | Clean a ladle; place it on a countertop | Reduce Search Steps for Clean Tasks | The rollout searched cabinets 1-18 and drawers 1-7 and reached a countertop only at the end of the 50-action budget. It never acquired the ladle or entered the clean-and-place phases, motivating earlier rotation toward likely utensil locations. |
| ALFWorld/Heat | `update_skill` | Heat an apple; place it in a sink | Heat While Holding Target | Following the old place-first procedure, the agent put the apple inside the microwave and obtained ineffective heat attempts. Heating succeeded only after it took the apple back and executed the microwave heat action while holding it, motivating the procedural update. |
| ALFWorld/Cool | `update_skill` | Cool a cup; place it in a microwave | Cool Before Placement, refined | The trajectory interleaved placement-oriented actions with cooling attempts and did not preserve a clear transformation-then-delivery order. The revision explicitly enforces acquisition, cooling at the fridge while retaining the target, and final placement only after the cooled state is established. |
| ALFWorld/Pick2 | `propose_skill` | Put two credit cards in a safe | Check Inventory Before Taking | After taking one credit card, the agent immediately tried to take the second and received `Nothing happens` because it was still holding the first. It recovered by placing the first card, returning for the second, and completing sequential delivery; the proposed rule generalizes this inventory-aware procedure. |
| WebShop | `propose_skill` | Buy a rectangular blue furniture set | Verify Product Shape and Size Before Buying | The request specified a blue rectangular furniture item with exact dimensions, but the agent purchased an area rug configured as a runner in a different size. The proposed rule requires a final shape-and-dimension check after option selection and before purchase. |
| WebShop | `propose_skill` | Buy a youth-size men's shirt | Abandon Failed Product Candidates | After opening a non-matching product, the agent spent the rollout repeatedly selecting the same `youth`, `asphalt`, and `small` options without obtaining new evidence or moving to another result. The proposed rule abandons a candidate after repeated unproductive interactions. |
| WebShop | `propose_skill` | Buy an underwater-photography backdrop | Select Product Type Before Buying | A relevant backdrop appeared in the results, but the agent opened and purchased a ball-head adapter instead. The proposed rule makes agreement with the requested product family a gating check before option selection and purchase. |
| WebShop | `propose_skill` | Buy a black-or-gray furniture set | Avoid Overloaded Search Queries | The initial query packed the product type and all variant constraints into one search, surfaced an irrelevant area rug, and led the agent to open that candidate. The proposed rule keeps the initial query focused on product type and essential constraints, leaving remaining attributes for result and option screening. |
| WebShop | `update_skill` | Buy a bronze steel-frame floor lamp | Include Hard Constraints in Search, expanded | The agent proceeded toward purchase without establishing that the lamp had the required steel frame. The update extends the existing hard-constraint rule to include material and construction requirements, rather than checking only product type, color, size, and price. |
| WebShop | `update_skill` | Buy a sunflower-pattern women's top | Verify Product Type After Option Selection | The agent selected the visible sunflower pattern and size and then purchased without rechecking short sleeves, fit, closure, and material constraints. The update adds a post-option-selection verification step immediately before `Buy Now`. |

