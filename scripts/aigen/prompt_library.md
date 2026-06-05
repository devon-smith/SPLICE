<!-- AI-USE: This prompt library was AI-assisted with Claude (claude-sonnet-4-6) via Claude Code. -->
<!-- Scope: drafted prompt-pair templates for sourcing the AI-gen evaluation set. -->

# AI-Gen Prompt Library

Fifty ready-to-use prompt-pair templates for sourcing the SPLICE AI-gen evaluation
set. Each template is one **pair** of clips — generate clip A from *Prompt A* and
clip B from *Prompt B*, then drop them as `pair_<id>_left.mp4` (A) and
`pair_<id>_right.mp4` (B). See [sourcing_protocol.md](sourcing_protocol.md) for
the end-to-end workflow.

**Three pair types:**

| Type | Intended label | What A and B are |
|---|---|---|
| continuous-action | `0` (consistent) | Same subject, same location — only the camera framing changes. |
| reverse-shot | `0` (consistent) | Two framings *within one scene* — a shot/reverse-shot dialogue, or a wide-to-close on one character. |
| cross-scene | `1` (inconsistent) | Two clearly different scenes — different place, time, or style. |

**Labelling is intent-based.** The label is what the *prompt pair was designed to
produce*, not what the AI actually rendered. If a continuous-action prompt pair
comes back with the character's face changed, the label is still `0` — detecting
that failure is the model's job. See the protocol for the full rule.

**Adapt freely.** These are templates, not a fixed list. Swap the subject, the
location, the time of day — keep the *structure* of the pair (what A vs B holds
constant and what it changes). Aim for the per-person mix in the protocol
(≈30 continuous-action / 10 reverse-shot / 10 cross-scene).

---

## Continuous-action pairs (intended label 0)

Same subject, same place; A and B differ only in camera framing. These probe
whether a framing change *within* a scene is correctly read as continuous, and
whether the AI holds character identity and setting across the cut.

### continuous_action_001
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide tracking shot of a man in his thirties wearing a navy peacoat and grey scarf walking briskly down a busy Manhattan sidewalk, midday, overcast soft light, shallow depth of field, cinematic, 24fps
- **Prompt B:** Medium shot of the same man in a navy peacoat and grey scarf walking down the same Manhattan sidewalk, midday, overcast light, slight low angle, cinematic, 24fps
- **Notes:** Urban exterior, day, single character, in motion. Tests whether a wide-to-medium framing change reads as continuous; probes coat/scarf identity hold.

### continuous_action_002
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman with short blonde hair in a red trench coat crossing a rain-slicked city street at night, neon signs reflecting on the pavement, cinematic, 24fps
- **Prompt B:** Close-up from the front of the same blonde woman in a red trench coat as she crosses the rain-slicked city street at night, neon reflections, cinematic, 24fps
- **Notes:** Urban exterior, night. Wet reflections stress lighting consistency; tests identity hold across a wide-to-close cut.

### continuous_action_003
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a man in a brown leather jacket walking through an autumn forest at golden hour, fallen orange leaves, dappled sunlight through bare branches, cinematic, 24fps
- **Prompt B:** Medium close-up of the same man in a brown leather jacket walking through the autumn forest at golden hour, dappled light, cinematic, 24fps
- **Notes:** Rural exterior, golden hour. Classic continuity case — same subject and forest, framing change only.

### continuous_action_004
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of an elderly farmer in denim overalls and a straw hat standing still in a green wheat field under a clear blue sky, gentle breeze, cinematic, 24fps
- **Prompt B:** Medium shot of the same elderly farmer in denim overalls and straw hat standing in the green wheat field, clear blue sky, cinematic, 24fps
- **Notes:** Rural exterior, day, static subject. Isolates the framing change with no motion — probes whether stillness reads as consistent.

### continuous_action_005
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a young woman with curly dark hair reading a book in a sunlit living room armchair, warm afternoon light through tall windows, cinematic, 24fps
- **Prompt B:** Close-up of the same young woman with curly dark hair reading in the sunlit living room armchair, warm afternoon light, cinematic, 24fps
- **Notes:** Interior, day, static. Tests identity and consistent window lighting across a wide-to-close cut.

### continuous_action_006
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a bearded man in a grey t-shirt cooking pasta at a stove in a dim apartment kitchen at night, warm overhead light, steam rising, cinematic, 24fps
- **Prompt B:** Medium close-up of the same bearded man in a grey t-shirt stirring the pasta pot in the dim apartment kitchen, warm light, steam, cinematic, 24fps
- **Notes:** Interior, night, single character. Practical warm light source — tests lighting consistency and identity.

### continuous_action_007
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of two young women in summer dresses walking and laughing together along a sunny European cobblestone street, cafe awnings, cinematic, 24fps
- **Prompt B:** Medium shot of the same two young women in summer dresses walking and laughing along the cobblestone street, cafe awnings behind them, cinematic, 24fps
- **Notes:** Urban exterior, day, multi-character. Both identities must hold across the framing change.

### continuous_action_008
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a family of four seated around a wooden dining table for lunch, bright kitchen, natural light, cinematic, 24fps
- **Prompt B:** Medium shot of the same family of four seated around the wooden dining table, bright kitchen, natural light, cinematic, 24fps
- **Notes:** Interior, day, multi-character, static. Tests whether a wider/closer view of one group reads as a single scene.

### continuous_action_009
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman in a flowing white dress riding a brown horse across an open grassy meadow, mountains in the distance, bright daylight, cinematic, 24fps
- **Prompt B:** Medium shot of the same woman in a white dress riding the brown horse across the grassy meadow, mountains behind, bright daylight, cinematic, 24fps
- **Notes:** Rural exterior, day, fast subject motion. Tests identity hold (dress, horse) under movement.

### continuous_action_010
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a man in a charcoal business suit standing and checking his phone at a city bus stop, glass shelter, overcast daylight, cinematic, 24fps
- **Prompt B:** Close-up of the same man in a charcoal suit checking his phone at the bus stop, glass shelter behind, overcast light, cinematic, 24fps
- **Notes:** Urban exterior, day, static. Isolates framing on a still subject in a recognisable location.

### continuous_action_011
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman in a beige blazer typing at a desk in a modern open-plan office, large windows, daylight, cinematic, 24fps
- **Prompt B:** Medium close-up of the same woman in a beige blazer typing at her desk in the open-plan office, daylight from the windows, cinematic, 24fps
- **Notes:** Interior, day, single character. Tests identity and consistent daylight across framing.

### continuous_action_012
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a lively crowd at an outdoor night street festival, string lights overhead, food stalls, people milling, cinematic, 24fps
- **Prompt B:** Medium shot within the same outdoor night street festival, string lights and food stalls, people milling, cinematic, 24fps
- **Notes:** Exterior, night, multi-character crowd, no hero subject. Tests whether a busy scene reads as one continuous location.

### continuous_action_013
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a man in a yellow raincoat repairing a wooden fence on a misty overcast farm, muted green fields, cinematic, 24fps
- **Prompt B:** Medium shot of the same man in a yellow raincoat working on the wooden fence, misty overcast farm, muted greens, cinematic, 24fps
- **Notes:** Rural exterior, overcast. Flat light and low saturation — a hard lighting case for continuity.

### continuous_action_014
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of six friends seated around a candlelit dinner table at a dinner party, warm low light, wine glasses, cinematic, 24fps
- **Prompt B:** Medium shot of the same six friends around the candlelit dinner table, warm low light, wine glasses, cinematic, 24fps
- **Notes:** Interior, night, multi-character, static. Low-key candlelight — tests dark-scene continuity.

### continuous_action_015
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a man in a denim jacket leaning on a rooftop railing looking at a city skyline at sunset, orange sky, cinematic, 24fps
- **Prompt B:** Close-up profile of the same man in a denim jacket on the rooftop at sunset, city skyline soft behind him, orange sky, cinematic, 24fps
- **Notes:** Urban exterior, sunset, static. Strong warm grade — tests colour/lighting consistency across a wide-to-close cut.

### continuous_action_016
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman with a long braid painting a large canvas on an easel in a bright artist's studio, paint-splattered floor, daylight, cinematic, 24fps
- **Prompt B:** Medium close-up of the same woman with a long braid painting the canvas in the bright studio, daylight, cinematic, 24fps
- **Notes:** Interior, day, single character, in motion. Tests identity (braid) and consistent setting across framing.

### continuous_action_017
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a man in red swim shorts jogging along a sandy tropical beach, turquoise ocean, bright midday sun, cinematic, 24fps
- **Prompt B:** Medium shot of the same man in red swim shorts jogging along the tropical beach, turquoise ocean behind, bright sun, cinematic, 24fps
- **Notes:** Exterior, day, single character. High-key beach lighting — tests continuity under bright, high-contrast conditions.

### continuous_action_018
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman holding a black umbrella walking down a quiet city street in heavy rain at night, streetlight glow, cinematic, 24fps
- **Prompt B:** Medium close-up of the same woman with a black umbrella walking the rainy night street, streetlight glow, cinematic, 24fps
- **Notes:** Urban exterior, rainy night. Low visibility — a hard case for both identity and lighting continuity.

### continuous_action_019
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of two warehouse workers in orange hi-vis vests carrying a wooden crate across a large storage warehouse, daylight from high windows, cinematic, 24fps
- **Prompt B:** Medium shot of the same two workers in orange hi-vis vests carrying the wooden crate through the warehouse, daylight from high windows, cinematic, 24fps
- **Notes:** Interior, day, multi-character with a shared action. Tests joint identity hold under motion.

### continuous_action_020
- **Type:** continuous-action
- **Intended label:** 0
- **Prompt A:** Wide shot of a person in a thick red parka trudging through deep snow in a quiet pine forest, soft grey winter light, cinematic, 24fps
- **Prompt B:** Medium shot of the same person in a red parka trudging through the snowy pine forest, soft grey winter light, cinematic, 24fps
- **Notes:** Rural exterior, snow. Low-saturation winter palette — tests continuity in a near-monochrome scene.

---

## Reverse-shot pairs (intended label 0)

Two framings *within one scene* — a shot/reverse-shot dialogue between two
characters, or a wide-to-close on a single character. The camera position jumps
hard (often ~180°), but the scene does not change. These probe whether a large
camera-position change is mistaken for a scene cut.

### reverse_shot_001
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Over-the-shoulder shot of a woman with red hair sitting at a wooden kitchen table talking, warm morning light through a window behind her, cinematic, 24fps
- **Prompt B:** Reverse over-the-shoulder shot of a man with glasses sitting across the same wooden kitchen table listening, warm morning light, cinematic, 24fps
- **Notes:** Classic shot/reverse-shot. The camera flips 180° — tests that opposite framings of one conversation read as the same scene.

### reverse_shot_002
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of a man driving a car, hands on the wheel, talking, daylight through the windscreen, dashboard visible, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a woman in the passenger seat of the same car listening and replying, daylight through the side window, cinematic, 24fps
- **Notes:** Confined two-shot scene. Tests reverse framing inside a single car interior.

### reverse_shot_003
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Over-the-shoulder shot of a job candidate in a blue shirt seated answering questions in a modern office, daylight, cinematic, 24fps
- **Prompt B:** Reverse over-the-shoulder shot of the interviewer in a grey blazer seated behind a desk in the same office, daylight, cinematic, 24fps
- **Notes:** Desk-facing reverse shots. Tests one-room continuity across a 180° flip.

### reverse_shot_004
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium close-up of a woman in a black dress talking across a candlelit restaurant booth, warm low light, cinematic, 24fps
- **Prompt B:** Reverse medium close-up of a man in a dark suit listening across the same candlelit restaurant booth, warm low light, cinematic, 24fps
- **Notes:** Low-key restaurant scene. Tests reverse-shot continuity in dim lighting.

### reverse_shot_005
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of an older man on a park bench talking, green trees behind, soft daylight, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a young woman on the same park bench listening, green trees behind, soft daylight, cinematic, 24fps
- **Notes:** Outdoor two-hander. Tests reverse framing in an exterior scene.

### reverse_shot_006
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of a woman seated on a living room couch talking, table lamp glow, evening, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a man seated on the same living room couch replying, table lamp glow, evening, cinematic, 24fps
- **Notes:** Domestic interior reverse. Tests lamp-lit evening continuity.

### reverse_shot_007
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Wide shot of a woman standing alone in an empty art gallery looking at a painting, polished floor, soft daylight, cinematic, 24fps
- **Prompt B:** Close-up of the same woman's face in the empty art gallery looking at the painting, soft daylight, cinematic, 24fps
- **Notes:** Wide-to-close on one character, no second person. Tests that a large scale jump within a scene is not read as a cut to a new scene.

### reverse_shot_008
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Over-the-shoulder shot of a patron on a barstool talking to the bartender, warm bar lighting, bottles behind, cinematic, 24fps
- **Prompt B:** Reverse over-the-shoulder shot of the bartender behind the same bar counter replying, warm lighting, bottles behind, cinematic, 24fps
- **Notes:** Bar two-hander. Tests reverse framing with a busy practical background.

### reverse_shot_009
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of a teacher standing at the front of a classroom speaking, whiteboard behind, daylight, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a student seated at a desk in the same classroom raising a hand, daylight, cinematic, 24fps
- **Notes:** Front-to-back reverse across one room. Tests continuity over a large spatial framing change.

### reverse_shot_010
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium close-up of a woman talking on a therapist's couch, soft window light, a plant in the corner, cinematic, 24fps
- **Prompt B:** Reverse medium close-up of the therapist in an armchair listening in the same room, soft window light, cinematic, 24fps
- **Notes:** Quiet two-person interior. Tests gentle reverse continuity.

### reverse_shot_011
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Over-the-shoulder shot of a detective leaning over a metal table questioning someone, harsh overhead light, bare walls, cinematic, 24fps
- **Prompt B:** Reverse over-the-shoulder shot of the suspect seated at the same metal table, harsh overhead light, bare walls, cinematic, 24fps
- **Notes:** Stark single-source lighting. Tests reverse-shot continuity in a high-contrast room.

### reverse_shot_012
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Tracking medium shot of a man talking as he walks down a tree-lined path, dappled daylight, cinematic, 24fps
- **Prompt B:** Reverse tracking medium shot of a woman walking beside him on the same tree-lined path replying, dappled daylight, cinematic, 24fps
- **Notes:** Moving reverse shots. Tests continuity when both framing and camera are in motion within one scene.

### reverse_shot_013
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of a patient lying in a hospital bed talking, pale clinical light, monitors beside the bed, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a visitor seated at the same hospital bedside listening, pale clinical light, cinematic, 24fps
- **Notes:** Clinical interior. Tests reverse continuity with cool, even lighting.

### reverse_shot_014
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Wide establishing shot of a cosy bookshop interior, shelves of books, warm lamplight, one customer browsing, cinematic, 24fps
- **Prompt B:** Close-up of the same customer's face browsing a shelf inside the cosy bookshop, warm lamplight, cinematic, 24fps
- **Notes:** Establishing-to-detail pair. Tests that an establishing wide and a close-up of someone inside it are the same scene.

### reverse_shot_015
- **Type:** reverse-shot
- **Intended label:** 0
- **Prompt A:** Medium shot of a woman standing inside a doorway talking, warm interior light behind her, cinematic, 24fps
- **Prompt B:** Reverse medium shot of a man standing just outside the same doorway on a porch replying, soft daylight, cinematic, 24fps
- **Notes:** Interior/exterior straddle within one scene. Tests continuity when lighting shifts across a threshold but the scene is still one.

---

## Cross-scene pairs (intended label 1)

A and B are clearly *different scenes* — different location, time of day, or
visual style. The model should score these as inconsistent. These are the
positive class.

### cross_scene_001
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a snowy mountain peak at dawn, blue hour, cold pale light, untouched snow, cinematic, 24fps
- **Prompt B:** Interior shot of a crowded nightclub with pulsing neon lights, dancers, haze, cinematic, 24fps
- **Notes:** Maximal contrast — cold empty nature vs warm crowded interior. An easy positive.

### cross_scene_002
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of an empty desert at noon, cracked earth, harsh sun, heat haze, cinematic, 24fps
- **Prompt B:** Wide shot of a rainy city street at night, neon reflections, traffic, cinematic, 24fps
- **Notes:** Opposite climates, times of day, and densities. A clear scene change.

### cross_scene_003
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Interior shot of a quiet wood-panelled library, rows of books, warm reading lamps, cinematic, 24fps
- **Prompt B:** Wide shot of a stormy ocean with crashing grey waves under a dark sky, cinematic, 24fps
- **Notes:** Calm interior vs violent exterior. A clear positive.

### cross_scene_004
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a stone medieval castle courtyard at midday, hanging banners, cobblestones, cinematic, 24fps
- **Prompt B:** Interior shot of a modern underground subway platform, fluorescent light, tiled walls, cinematic, 24fps
- **Notes:** Different historical era and architecture. Tests style/era discontinuity.

### cross_scene_005
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a sunny tropical beach with palm trees and turquoise water, cinematic, 24fps
- **Prompt B:** Wide shot of a flat arctic tundra under a grey sky, snow and ice to the horizon, cinematic, 24fps
- **Notes:** Polar-opposite biomes. An easy positive.

### cross_scene_006
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Interior shot of a busy professional kitchen, chefs working, steam, stainless steel, cinematic, 24fps
- **Prompt B:** Interior shot of a vast empty cathedral, stone columns, light through stained glass, cinematic, 24fps
- **Notes:** Both interiors — tests cross-scene detection when only the content changes, not indoors/outdoors.

### cross_scene_007
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a misty forest at dawn, soft light through tall pines, cinematic, 24fps
- **Prompt B:** Interior shot of an industrial factory floor, machinery, sparks, harsh light, cinematic, 24fps
- **Notes:** Natural calm vs industrial. A clear positive.

### cross_scene_008
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a crowded outdoor market in daylight, stalls, produce, people, cinematic, 24fps
- **Prompt B:** Interior shot of a quiet bedroom at night, one bedside lamp, still, cinematic, 24fps
- **Notes:** Crowded/loud vs empty/quiet. Tests density and time-of-day discontinuity.

### cross_scene_009
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Interior shot of a futuristic spaceship corridor, glowing panels, metallic surfaces, cinematic, 24fps
- **Prompt B:** Wide shot of a wooden farmhouse porch at sunset, fields beyond, warm light, cinematic, 24fps
- **Notes:** Sci-fi vs rural — a strong style and palette change.

### cross_scene_010
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Underwater shot of a vibrant coral reef with tropical fish, sunlight rays through blue water, cinematic, 24fps
- **Prompt B:** Wide shot of a congested city highway in a daytime traffic jam, cinematic, 24fps
- **Notes:** Underwater vs urban. Tests an extreme environment change.

### cross_scene_011
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a ballet dancer on a theatre stage under a single spotlight, dark auditorium, cinematic, 24fps
- **Prompt B:** Wide shot of a construction site at midday, cranes, scaffolding, workers, cinematic, 24fps
- **Notes:** Performance vs labour — different lighting logic and palette.

### cross_scene_012
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a peaceful autumn park, golden leaves, a winding path, soft daylight, cinematic, 24fps
- **Prompt B:** Wide shot of a volcanic landscape with black rock and glowing lava, smoke, cinematic, 24fps
- **Notes:** Gentle vs hostile terrain. An easy positive with a strong colour shift.

### cross_scene_013
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Interior shot of an elegant wedding reception, flowers, fairy lights, guests in formal wear, cinematic, 24fps
- **Prompt B:** Interior shot of a derelict abandoned warehouse, broken windows, dust, cinematic, 24fps
- **Notes:** Both large interiors — tests cross-scene detection on mood and condition rather than indoors/outdoors.

### cross_scene_014
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Interior shot of a sunny morning cafe, customers with coffee, plants, soft daylight, cinematic, 24fps
- **Prompt B:** Wide shot of a violent thunderstorm at sea, towering waves, lightning, cinematic, 24fps
- **Notes:** Calm and cosy vs dramatic and dangerous. A clear positive.

### cross_scene_015
- **Type:** cross-scene
- **Intended label:** 1
- **Prompt A:** Wide shot of a children's playground on a bright sunny afternoon, swings, slides, cinematic, 24fps
- **Prompt B:** Wide shot of a misty graveyard at night, weathered headstones, fog, pale moonlight, cinematic, 24fps
- **Notes:** Day/innocent vs night/eerie. A strong time-of-day and tone discontinuity.
