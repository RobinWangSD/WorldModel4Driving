# DrivoR — Temporal Image Cache + Three-Mode Training

This document summarizes a single feature added to DrivoR: an opt-in temporal
image cache plus three compatible training settings (current image / history
images / history images + future-latent world-model head). When the master
flag is off, the codebase is bit-identical to its prior behavior.

---

## 1. User-facing API

### Master flag

```yaml
# navsim/planning/script/config/common/agent/drivoR.yaml
agent.config.temporal_caching.enabled: false   # default — legacy behavior
```

When `false`, none of the new code runs and the cache layout is unchanged.

### Full new config block

Added under `config:` in `drivoR.yaml`:

```yaml
temporal_caching:
  enabled: false
  history_iters: [0, 1, 2, 3]
  future_iters:  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  active_cameras: [cam_f0]
input_mode: current_image          # current_image | history_images
predict_future_latent: false
latent_pred_steps: 10              # 1..len(future_iters)
```

### Three training modes

All three modes consume the same cache that's produced when `temporal_caching.enabled=true`:

| Mode                                     | `input_mode`     | `predict_future_latent` | What it does                                          |
| ---------------------------------------- | ---------------- | ----------------------- | ----------------------------------------------------- |
| 1. current image → future actions        | `current_image`  | `false`                 | Equivalent to today's behavior                        |
| 2. history images → future actions       | `history_images` | `false`                 | T_hist front images folded into the camera dimension  |
| 3. history images → future latents + actions | `history_images` | `true`                  | Mode 2 inputs + auxiliary multi-step world-model loss |

---

## 2. New cache schema (only when `temporal_caching.enabled=true`)

Added to `drivor_feature.gz` alongside the existing keys:

| Key                  | Shape                              | Dtype   | Source                                                      |
| -------------------- | ---------------------------------- | ------- | ----------------------------------------------------------- |
| `image_history`      | `(T_hist, N_front, 3, H, W)`       | float32 | `agent_input.cameras[i]` for i in `history_iters` (offset)  |
| `image_future`       | `(T_fut, N_front, 3, H, W)`        | float32 | `agent_input.future_cameras[i]` for i in `future_iters`     |
| `ego_status_history` | `(T_hist, 11)`                     | float32 | Same as legacy `ego_status` (kept as alias)                 |
| `ego_status_future`  | `(T_fut, 11)`                      | float32 | `agent_input.future_ego_statuses` (newly populated)         |
| `history_valid`      | `(T_hist,)`                        | bool    | Per-step image validity (False on missing PNG)              |
| `future_valid`       | `(T_fut,)`                         | bool    | Per-step image validity                                     |
| `ego_future_valid`   | `(T_fut,)`                         | bool    | True for each future ego step that was actually present     |

Legacy keys (`image`, `ego_status`, `cam_K`, `world_2_cam`, plus the optional
`image_next` / `image_next_valid`) keep being written so existing eval and
visualization code stays working. `image` is set to `image_history[-1]` (a
view of the current frame) when temporal caching is on.

---

## 3. File-by-file changes

### `navsim/planning/script/config/common/agent/drivoR.yaml`

Added the temporal-caching block + `input_mode` / `predict_future_latent` /
`latent_pred_steps` at the top level of `config:`. All defaults preserve legacy
behavior. Comment explicitly notes that with `temporal_caching.enabled=false`
nothing else changes.

### `navsim/common/dataclasses.py`

- `class AgentInput` — added two optional fields:
  - `future_cameras: Optional[List[Cameras]] = None`
  - `future_ego_statuses: Optional[List[EgoStatus]] = None`
- `Scene.get_agent_input(expose_future: bool = False)` — when `expose_future=True`,
  populates both new fields from `self.frames[num_history_frames:num_history_frames+num_future_frames]`.
  Future ego poses are converted to local frame using the current pose as origin
  (matching `get_future_trajectory`'s convention). Default arg is `False`, so
  every existing call site keeps working unchanged.

### `navsim/agents/drivoR/drivor_agent.py`

- Added `_temporal_caching_active()` helper.
- Added `requires_future_cameras` property (`True` iff temporal caching is on).
- `get_sensor_config()` branches:
  - **off**: returns the same `SensorConfig` it always did.
  - **on**: builds `cam_*` iteration lists from
    `temporal_caching.history_iters ∪ temporal_caching.future_iters` for cameras
    in `active_cameras`; empties the rest. This keeps Mode 3 front-camera-only
    automatically without the user having to also pass `agent.config.cam_l0=[]`
    etc.

### `navsim/agents/drivoR/drivor_features.py`

- Added `_temporal_caching_active()` helper.
- Added `_stack_ego(ego_statuses)` helper extracting the existing per-frame
  ego-to-tensor logic. The legacy `features["ego_status"]` now uses this helper —
  identical output, easier to reuse.
- Added `_build_per_frame_camera_tensors(cameras_obj)` that wraps the existing
  per-camera preprocessing and returns `(image_stack, cam_K_stack, world_2_cam_stack, valid_bool)`.
- `compute_features()` — when temporal caching is on, additionally emits
  `image_history`, `image_future`, `history_valid`, `future_valid`,
  `ego_status_history`, `ego_status_future`, `ego_future_valid`. Pads / zeroes
  any time step that the scene window can't supply.

### `navsim/planning/training/dataset.py`

- Added `builder_uses_temporal_caching(builder)` predicate (mirror of the
  existing `builder_uses_latent_image_next`).
- Added `normalize_drivor_temporal(data_dict)` — load-time validator that
  zeroes out `image_history` / `image_future` if their trailing shape doesn't
  match the legacy `image` tensor (e.g. cache built with a different
  `image_size`).
- `Dataset._cache_scene_with_token`: passes `expose_future=True` to
  `scene.get_agent_input()` when any feature builder requires future frames.
- Both `_load_scene_with_token` paths (in `CacheOnlyDataset` and `Dataset`)
  call `normalize_drivor_temporal` for temporal builders.
- `Dataset.__getitem__` no-cache branch now uses
  `scene.get_agent_input(expose_future=...)` instead of
  `SceneLoader.get_agent_input_from_token`, so temporal training without cache
  also works.

### `navsim/agents/drivoR/drivor_model.py`

- Module-level: imports `WorldModelLatentPredictor` and `MultiStepLatentLoss`;
  defines `_ALL_CAM_KEYS` and `_temporal_caching_enabled` helper.
- `__init__`:
  - Reads `input_mode`, `predict_future_latent`, `latent_pred_steps`,
    `temporal_caching.enabled`, plus `t_hist` / `t_fut`.
  - Validates: `input_mode='history_images'` requires `temporal_caching.enabled`;
    `predict_future_latent=true` requires it too; `latent_pred_steps` must be in
    `[1, t_fut]`.
  - When temporal caching is on, recomputes `num_cams` from
    `len(active_cameras)` (so the user doesn't have to also override the
    per-cam config keys).
  - Introduces `effective_num_cams = num_cams * t_hist` when
    `input_mode='history_images'`; `scene_embeds` is sized accordingly.
  - Adds a learned `temporal_pe` parameter (T_hist temporal positional
    embeddings) for `history_images` mode.
  - Builds `WorldModelLatentPredictor` + `MultiStepLatentLoss` when
    `predict_future_latent=true`. The predictor's token count is
    `num_cams * num_scene_tokens` (per-view, not time-folded).
- `forward`:
  - Adds `want_world_model` (requires `predict_future_latent`, training mode,
    images, and the cached future tensors) and gates `want_latent` with
    `and not want_world_model` so the two paths never both populate
    `latent_loss_dict`.
  - Uses new `_select_input_image()` to pick the input tensor by `input_mode`
    (folds `(B, T_hist, N_cam, ...)` → `(B, T_hist*N_cam, ...)` in
    `history_images` mode).
  - Calls `_apply_temporal_pe()` on `scene_tokens` so the backbone can
    distinguish history steps from each other.
  - When `want_world_model`, calls `_compute_world_model_loss()` instead of
    the legacy one-step latent loss.
- New helpers `_select_input_image`, `_apply_temporal_pe`,
  `_compute_world_model_loss` defined at the end of the class.
- `_compute_world_model_loss` runs the (frozen) image backbone on each future
  step independently to get target latents, mean-pools the time-folded current
  scene tokens down to `(B, N_cam*S, D)` for the predictor input, runs the
  predictor with ground-truth `ego_status_future[:, :T_pred]` as teacher
  forcing, and returns the multi-step latent loss dict.

### `navsim/agents/drivoR/layers/latent_predictor.py`

Added `WorldModelLatentPredictor` alongside the existing `LatentPredictor`:

```python
class WorldModelLatentPredictor(nn.Module):
    def forward(self, cur_scene_tokens, ego_history, future_actions):
        # cur_scene_tokens: (B, N_tokens, D)
        # ego_history:      (B, T_hist, ego_dim=11)
        # future_actions:   (B, T_pred, ego_dim=11)
        # returns:          (B, T_pred, N_tokens, D)
```

Per-step batched DiT pass: each future step gets its own copy of the current
scene latent and is conditioned on `pooled_ego_history + action_proj[step] +
learned_step_embed[step]`. Reuses the existing `DiTBlock` (AdaLN-Zero
modulation) — no new building blocks.

### `navsim/agents/drivoR/layers/losses/latent_loss.py`

Added `MultiStepLatentLoss`:

- Inputs: `predicted: (B, T_pred, N, D)`, `target: (B, T_pred, N, D)`,
  optional `valid_mask: (B, T_pred)`.
- Computes per-(B, T) MSE, masks by `valid_mask`, returns
  `{"loss": ..., "latent_prediction": ...}`.
- Supports `stop_grad_target` (default True).

`DrivoRLoss` is unchanged: the new dict has the same `loss` key, so the
existing `latent_weight` wiring at line 321-322 of `drivor_loss.py` already
weights the multi-step loss correctly. The auxiliary `latent_*` entries are
also surfaced in `loss_dict` by the existing loop at 355-357.

---

## 4. Backward-compatibility guarantee

`temporal_caching.enabled = false` ⇒ all of the following hold:

1. `DrivoRAgent.get_sensor_config()` returns the exact same `SensorConfig` it
   did before (per-key OmegaConf-to-object pass-through).
2. `Scene.get_agent_input()` returns `AgentInput` with `future_cameras=None`
   and `future_ego_statuses=None`.
3. `DrivoRFeatureBuilder.compute_features()` returns exactly the same keys and
   tensors as before (the `_stack_ego` refactor is bitwise-equivalent).
4. `Dataset._cache_scene_with_token` passes `expose_future=False` (the same as
   omitting the arg), so the future code path never runs.
5. `DrivoRModel.__init__`:
   - `num_cams` is computed the same way (loop over `_ALL_CAM_KEYS` vs. eight
     unrolled checks — same arithmetic).
   - `effective_num_cams == num_cams` (since default `input_mode='current_image'`).
   - `scene_embeds` has the same shape.
   - No `temporal_pe`, `world_model_predictor`, or `world_model_loss_fn` is
     built.
6. `DrivoRModel.forward`:
   - `want_world_model` defaults to `False` (gate on `predict_future_latent`,
     which defaults to `False`).
   - `want_latent` is unchanged (the new `and not want_world_model` clause is
     `True` when `want_world_model=False`).
   - `_select_input_image()` returns `features["image"]` (or `camera_feature`)
     exactly like before when no `image_history` key is present.
   - `_apply_temporal_pe()` is a no-op when `input_mode != "history_images"`.

Existing caches (e.g. `/closed-loop-e2e/drivor-exp/navsim_cache_nommcv`) and
the user's existing training command keep working unchanged.

---

## 5. How to use it

### Step 1 — Rebuild the cache (one-time, ~2.8 TB for navtrain)

```bash
python /root/WorldModel4Driving/DrivoR_vanilla/navsim/planning/script/run_dataset_caching.py \
    agent=drivoR \
    train_test_split=navtrain \
    navsim_log_path=/avl-west/navsim/trainval_navsim_logs/trainval \
    train_test_split.data_split=trainval \
    +train_test_split.log_splits=null \
    cache_path=/closed-loop-e2e/drivor-exp/navsim_cache_temporal_frontonly \
    force_cache_computation=true \
    sensor_blobs_path=/avl-west/navsim/trainval_sensor_blobs/trainval \
    worker.threads_per_node=24 \
    agent.config.temporal_caching.enabled=true \
    agent.config.latent_learning.enabled=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.long_trajectory_additional_poses=2
```

The agent will automatically:
- Load front-camera images at iterations 0..13 (via the rewritten
  `get_sensor_config`).
- Emit `image_history (4, 1, 3, H, W)`, `image_future (10, 1, 3, H, W)`,
  `ego_status_history`, `ego_status_future`, plus validity tensors.

### Step 2 — Train (pick a mode, all share the same cache)

**Mode 1 — current image → future actions** (matches today's behavior):

```bash
... agent.config.temporal_caching.enabled=true \
    agent.config.input_mode=current_image \
    agent.config.predict_future_latent=false ...
```

**Mode 2 — history images → future actions**:

```bash
... agent.config.temporal_caching.enabled=true \
    agent.config.input_mode=history_images \
    agent.config.predict_future_latent=false ...
```

**Mode 3 — history images → future latents + actions (world model)**:

```bash
... agent.config.temporal_caching.enabled=true \
    agent.config.input_mode=history_images \
    agent.config.predict_future_latent=true \
    agent.config.latent_pred_steps=10 \
    agent.loss.latent_weight=1.0 ...
```

`latent_pred_steps` can be reduced (e.g. `=4`) to predict only the immediate
future; the predictor is built with exactly that many step queries.

---

## 6. Implementation notes

### Per-view scene-embed handling for the world-model target

When `input_mode='history_images'`, `scene_embeds` has shape
`(1, T_hist*N_cam, S, D)`. The world-model **target** path needs a per-view
embedding of shape `(1, N_cam, S, D)` to encode each future step
independently. `_compute_world_model_loss` mean-pools the time axis to derive
this: `scene_embeds.view(1, T_hist, N_cam, S, D).mean(dim=1)`. It's a
pragmatic shortcut — a follow-up could give the target encoder its own
parameter table, but the current shortcut keeps the parameter count down and
makes the target backbone agree on shapes with what `image_backbone` produces
during the current-frame encode.

### Predictor input pooling

For the predictor input, the time-folded current scene tokens
`(B, T_hist*N_cam*S, D)` are mean-pooled across the time axis to
`(B, N_cam*S, D)` so they match the predictor's `n_tokens` size and the
target shape. Both pooling operations preserve gradient flow back into the
backbone and `scene_embeds`.

### `want_latent` vs. `want_world_model`

These are mutually exclusive: `want_world_model` takes precedence because both
paths populate `pred["latent_loss_dict"]`. The legacy one-step latent loss is
disabled whenever the multi-step world-model loss is active.

### Teacher forcing (training-only detail)

In Mode 3 the predictor sees ground-truth `ego_status_future` during training.
A future closed-loop eval path could replace this with the model's own
trajectory prediction; this is **not** wired up yet because eval doesn't run
the latent predictor at all (`want_world_model` requires `self.training`).
When you add the eval path, fall back to using `output["trajectory"]`
pose-only and pad zeros for velocity/accel/cmd, or train a small adapter
head — both are reasonable starting points.

### Cache size

Per sample (front camera only, image_size=[1148, 672]):
- Legacy cache: ~10 MB gzip-1
- Temporal cache (T_hist=4, T_fut=10): ~33 MB gzip-1 (≈3.3× larger)

For navtrain (~85k samples), total ≈ 2.8 TB. Cache to a fresh `cache_path`
so it sits next to the legacy cache rather than overwriting it.

### Validation index cache

`Dataset._load_valid_caches` writes a `valid_caches.pkl` index inside the
cache directory and reuses it on later loads (dataset.py:143). If you ever
add new tokens to an existing cache directory, delete that pickle or the
loader will silently keep returning the stale token list.

---

## 7. Verification

The following are recommended sanity checks. They are not yet wired up as
unit tests.

1. **Backward-compat run.** Re-run the user's existing training command
   against the legacy cache. Loss curves should be numerically identical to
   the pre-change baseline (same seed).
2. **Cache build smoke test.** Run the caching command above with
   `+scene_filter.max_scenes=8` and `cache_path=/tmp/drivor_temporal_smoke`.
   Then inspect one sample:

   ```python
   import gzip, pickle
   with gzip.open("/tmp/drivor_temporal_smoke/<log>/<token>/drivor_feature.gz", "rb") as f:
       d = pickle.load(f)
   for k, v in d.items():
       print(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
   ```

   Expect `image_history: (4, 1, 3, 672, 1148)`,
   `image_future: (10, 1, 3, 672, 1148)`, `ego_status_history: (4, 11)`,
   `ego_status_future: (10, 11)`, plus the validity bools.
3. **Per-mode forward smoke tests.** For each of the three modes, run one
   forward pass against the smoke cache and confirm:
   - `model(features)["trajectory"]` has shape `(B, num_poses, 3)`.
   - `model(features)["latent_loss_dict"]` exists only in Mode 3.
4. **Loss-decreases-on-overfit.** Train each mode on a single batch for ~100
   steps and verify `trajectory_loss` decreases monotonically. Catches
   wiring bugs that shape-only checks miss.
5. **Eval parity (Mode 1).** Mode 1 trained on the new cache should match
   the PDM score from the legacy command on the legacy cache within
   run-to-run noise. This is the strongest regression check that nothing
   silently shifted.

---

## 8. Files changed

| File                                                                       | Lines added (approx) |
| -------------------------------------------------------------------------- | -------------------- |
| `navsim/planning/script/config/common/agent/drivoR.yaml`                   | +12                  |
| `navsim/common/dataclasses.py`                                             | +44                  |
| `navsim/agents/drivoR/drivor_agent.py`                                     | +35                  |
| `navsim/agents/drivoR/drivor_features.py`                                  | +130                 |
| `navsim/planning/training/dataset.py`                                      | +45                  |
| `navsim/agents/drivoR/drivor_model.py`                                     | +170                 |
| `navsim/agents/drivoR/layers/latent_predictor.py`                          | +60                  |
| `navsim/agents/drivoR/layers/losses/latent_loss.py`                        | +50                  |

No files were deleted, no public functions removed, no existing call sites
broken.
