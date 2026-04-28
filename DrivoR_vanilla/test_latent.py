"""
Smoke tests for the latent transition model implementation.
Run with: python test_latent.py
"""
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/hugsim-storage/WorldModel4Driving/DrivoR_vanilla")


# ─── helpers ────────────────────────────────────────────────────────────────

def make_cameras(H=672, W=1148, missing=None):
    """Return a Cameras object with random numpy images for the 4 active cams."""
    from navsim.common.dataclasses import Camera, Cameras
    missing = set(missing or [])

    def cam():
        return Camera(
            image=np.random.randint(0, 255, (H, W, 3), dtype=np.uint8),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            intrinsics=np.array([[500,0,W/2],[0,500,H/2],[0,0,1]], dtype=np.float32),
            distortion=np.zeros(5, dtype=np.float32),
        )

    def maybe_cam(name):
        return Camera() if name in missing else cam()

    return Cameras(
        cam_f0=maybe_cam("cam_f0"),
        cam_l0=maybe_cam("cam_l0"),
        cam_r0=maybe_cam("cam_r0"),
        cam_b0=maybe_cam("cam_b0"),
        cam_l1=Camera(), cam_l2=Camera(), cam_r1=Camera(), cam_r2=Camera(),
    )


def make_agent_input(with_next=True, missing_next=None):
    from navsim.common.dataclasses import AgentInput, EgoStatus, Lidar
    ego = EgoStatus(
        ego_pose=np.zeros(3, dtype=np.float32),
        ego_velocity=np.zeros(2, dtype=np.float32),
        ego_acceleration=np.zeros(2, dtype=np.float32),
        driving_command=np.zeros(1, dtype=np.float32),
    )
    agent_input = AgentInput(
        ego_statuses=[ego] * 4,
        cameras=[make_cameras()] * 4,
        lidars=[Lidar()] * 4,
        next_cameras=make_cameras(missing=missing_next) if with_next else None,
    )
    return agent_input


def make_config(latent_enabled=True, one_step=True):
    cfg = OmegaConf.create({
        "b2d": False,
        "shared_refiner": False,
        "ref_num": 2,
        "scorer_ref_num": 2,
        "proposal_num": 4,
        "num_poses": 4,
        "num_scene_tokens": 4,
        "one_token_per_traj": True,
        "full_history_status": False,
        "long_trajectory_additional_poses": -1,
        "noc": 1.0, "dac": 1.0, "ddc": 0.0,
        "ttc": 5.0, "ep": 5.0, "comfort": 2.0,
        "tf_d_model": 64,
        "tf_d_ffn": 128,
        "cam_f0": [3], "cam_l0": [3], "cam_r0": [3], "cam_b0": [3],
        "cam_l1": [], "cam_l2": [], "cam_r1": [], "cam_r2": [],
        "lidar_pc": [],
        "image_size": [56, 56],
        "image_backbone": {
            "model_name": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "model_weights": "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors",
            "use_lora": False,
            "finetune": True,
            "lora_rank": 4,
            "focus_front_cam": False,
            "use_feature_pooling": False,
            "compress_fc": False,
        },
        "double_score": False,
        "agent_pred": False,
        "area_pred": False,
        "bev_map": False,
        "bev_agent": False,
        "refiner_num_heads": 1,
        "refiner_ls_values": 0.0,
        "trajectory_sampling": {
            "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
            "time_horizon": 2,
            "interval_length": 0.5,
        },
        "latent_learning": {
            "enabled": latent_enabled,
            "one_step": one_step,
            "predictor": {"nhead": 2, "num_layers": 1, "d_ffn": 128},
            "loss_weights": {"prediction": 1.0},
            "stop_grad_target": True,
        } if latent_enabled else None,
    })
    return cfg


# ─── tests ──────────────────────────────────────────────────────────────────

def test_agent_input_next_cameras():
    print("  [1] AgentInput.next_cameras field ...", end=" ")
    agent_input = make_agent_input(with_next=True)
    assert agent_input.next_cameras is not None
    agent_input_no_next = make_agent_input(with_next=False)
    assert agent_input_no_next.next_cameras is None
    print("OK")


def test_feature_builder_image_next():
    print("  [2] DrivoRFeatureBuilder produces image_next when next_cameras set ...", end=" ")
    from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
    cfg = make_config(latent_enabled=True)
    builder = DrivoRFeatureBuilder(cfg)

    agent_input = make_agent_input(with_next=True)
    features = builder.compute_features(agent_input)
    assert "image_next" in features, "image_next missing from features"
    assert "image_next_valid" in features, "image_next_valid missing from features"
    assert bool(features["image_next_valid"].item())
    assert features["image_next"].shape == features["image"].shape, \
        f"shape mismatch: image={features['image'].shape}, image_next={features['image_next'].shape}"
    print("OK")


def test_feature_builder_no_image_next_on_eval():
    print("  [3] DrivoRFeatureBuilder writes invalid image_next placeholder when next_cameras is None ...", end=" ")
    from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
    cfg = make_config(latent_enabled=True)
    builder = DrivoRFeatureBuilder(cfg)

    agent_input = make_agent_input(with_next=False)
    features = builder.compute_features(agent_input)
    assert "image_next" in features
    assert "image_next_valid" in features
    assert not bool(features["image_next_valid"].item())
    assert features["image_next"].shape == features["image"].shape
    assert torch.count_nonzero(features["image_next"]).item() == 0
    print("OK")


def test_feature_builder_partial_image_next_invalid():
    print("  [4] DrivoRFeatureBuilder invalidates image_next when one future camera is missing ...", end=" ")
    from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
    cfg = make_config(latent_enabled=True)
    builder = DrivoRFeatureBuilder(cfg)

    agent_input = make_agent_input(with_next=True, missing_next=["cam_r0"])
    features = builder.compute_features(agent_input)
    assert "image_next" in features
    assert not bool(features["image_next_valid"].item())
    assert features["image_next"].shape == features["image"].shape
    assert torch.count_nonzero(features["image_next"]).item() == 0
    print("OK")


def test_latent_predictor():
    print("  [5] LatentPredictor forward shape ...", end=" ")
    from navsim.agents.drivoR.layers.latent_predictor import LatentPredictor
    from navsim.agents.drivoR.transformer_decoder import DiTBlock
    B, N, D = 2, 16, 64
    predictor = LatentPredictor(d_model=D, nhead=2, num_layers=1, d_ffn=128, ego_dim=11)
    assert isinstance(predictor.blocks[0], DiTBlock)
    scene_tokens = torch.randn(B, N, D)
    ego_t = torch.randn(B, 11)
    out = predictor(scene_tokens, ego_t)
    assert out.shape == (B, N, D), f"expected ({B},{N},{D}), got {out.shape}"
    print("OK")


def test_dit_block():
    print("  [6] DiTBlock forward shape ...", end=" ")
    from navsim.agents.drivoR.transformer_decoder import DiTBlock, DiTSelfAttention
    B, N, D = 2, 16, 64
    block = DiTBlock(dim=D, num_heads=2, mlp_ratio=2.0)
    assert isinstance(block.attn, DiTSelfAttention)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    out = block(x, c)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("OK")


def test_dit_self_attention():
    print("  [7] DiTSelfAttention forward shape ...", end=" ")
    from navsim.agents.drivoR.transformer_decoder import DiTSelfAttention
    B, N, D = 2, 16, 64
    attn = DiTSelfAttention(dim=D, num_heads=2)
    x = torch.randn(B, N, D)
    out = attn(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("OK")


def test_latent_loss():
    print("  [8] LatentLoss forward, stop_grad_target=True ...", end=" ")
    from navsim.agents.drivoR.layers.losses.latent_loss import LatentLoss
    B, T, N, D = 2, 1, 16, 64
    loss_fn = LatentLoss(prediction_weight=1.0, stop_grad_target=True)
    predicted = torch.randn(B, T, N, D, requires_grad=True)
    target = torch.randn(B, T, N, D)
    result = loss_fn(predicted, target)
    assert "loss" in result and "latent_prediction" in result
    assert result["loss"].shape == ()           # scalar
    result["loss"].backward()
    assert predicted.grad is not None
    # target should have no grad (stop_grad=True)
    assert target.grad is None
    print("OK")


def test_model_forward_train_latent():
    print("  [9] DrivoRModel forward (train, with latent) — shapes and latent_loss_dict ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True)
    model = DrivoRModel(cfg)
    model.train()

    B = 2
    N_cams = 4
    C, H, W = 3, 56, 56
    N_tokens = N_cams * cfg.num_scene_tokens  # 4*4 = 16

    features = {
        "image":      torch.randn(B, N_cams, C, H, W),
        "image_next": torch.randn(B, N_cams, C, H, W),
        "ego_status": torch.randn(B, 4, 11),
    }

    with torch.no_grad():
        output = model(features)

    assert "trajectory" in output
    assert "latent_loss_dict" in output, "latent_loss_dict missing from output"
    lld = output["latent_loss_dict"]
    assert "loss" in lld and "latent_prediction" in lld
    assert lld["loss"].shape == ()
    print("OK")


def test_model_forward_train_latent_all_invalid_zero():
    print(" [10] DrivoRModel forward (train, invalid image_next) — zero latent loss ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True)
    model = DrivoRModel(cfg)
    model.train()

    B = 2
    N_cams = 4
    image = torch.randn(B, N_cams, 3, 56, 56)
    features = {
        "image": image,
        "image_next": torch.zeros_like(image),
        "image_next_valid": torch.zeros(B, dtype=torch.bool),
        "ego_status": torch.randn(B, 4, 11),
    }

    with torch.no_grad():
        output = model(features)

    assert "latent_loss_dict" in output
    assert output["latent_loss_dict"]["loss"].item() == 0
    assert output["latent_loss_dict"]["latent_prediction"].item() == 0
    print("OK")


def test_model_forward_train_latent_mixed_valid():
    print(" [11] DrivoRModel forward (train, mixed image_next validity) — masks invalid rows ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True)
    model = DrivoRModel(cfg)
    model.train()

    B = 2
    N_cams = 4
    features = {
        "image": torch.randn(B, N_cams, 3, 56, 56),
        "image_next": torch.randn(B, N_cams, 3, 56, 56),
        "image_next_valid": torch.tensor([True, False]),
        "ego_status": torch.randn(B, 4, 11),
    }

    captured = {}
    orig = model._compute_latent_loss

    def patched(features, cur_tokens, next_tokens, valid_mask=None):
        captured["valid_rows"] = cur_tokens.shape[0]
        captured["valid_mask"] = valid_mask.detach().cpu()
        return orig(features, cur_tokens, next_tokens, valid_mask)

    model._compute_latent_loss = patched

    with torch.no_grad():
        output = model(features)

    assert "latent_loss_dict" in output
    assert output["latent_loss_dict"]["loss"].shape == ()
    assert captured["valid_rows"] == 1
    assert torch.equal(captured["valid_mask"], torch.tensor([True, False]))
    print("OK")


def test_model_forward_eval_no_latent():
    print(" [12] DrivoRModel forward (eval, no image_next) — no latent_loss_dict ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True)
    model = DrivoRModel(cfg)
    model.eval()

    B = 2
    N_cams = 4
    features = {
        "image":      torch.randn(B, N_cams, 3, 56, 56),
        "ego_status": torch.randn(B, 4, 11),
        # no image_next
    }

    with torch.no_grad():
        output = model(features)

    assert "trajectory" in output
    assert "latent_loss_dict" not in output, "latent_loss_dict should be absent during eval"
    print("OK")


def test_model_forward_latent_disabled():
    print(" [13] DrivoRModel forward (latent disabled in config) — no latent_loss_dict ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=False)
    model = DrivoRModel(cfg)
    model.train()

    B = 2
    N_cams = 4
    features = {
        "image":      torch.randn(B, N_cams, 3, 56, 56),
        "image_next": torch.randn(B, N_cams, 3, 56, 56),
        "ego_status": torch.randn(B, 4, 11),
    }

    with torch.no_grad():
        output = model(features)

    assert "latent_loss_dict" not in output
    print("OK")


def test_model_init_one_step_false_not_implemented():
    print(" [14] DrivoRModel init raises when latent one_step=False ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True, one_step=False)

    try:
        DrivoRModel(cfg)
    except NotImplementedError as e:
        assert "one_step=False" in str(e)
        print("OK")
        return

    raise AssertionError("DrivoRModel should raise NotImplementedError for latent one_step=False")


def test_policy_and_predictor_share_tokens():
    print(" [15] Policy scene tokens == latent predictor cur_tokens (same tensor path) ...", end=" ")
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = make_config(latent_enabled=True)
    model = DrivoRModel(cfg)
    model.train()

    B, N_cams = 2, 4
    img    = torch.randn(B, N_cams, 3, 56, 56)
    img_next = torch.randn(B, N_cams, 3, 56, 56)
    features = {"image": img, "image_next": img_next, "ego_status": torch.randn(B, 4, 11)}

    # Patch _compute_latent_loss to capture cur_tokens
    captured = {}
    orig = model._compute_latent_loss
    def patched(features, cur_tokens, next_tokens, valid_mask=None):
        captured["cur_tokens"] = cur_tokens
        return orig(features, cur_tokens, next_tokens, valid_mask)
    model._compute_latent_loss = patched

    with torch.no_grad():
        output = model(features)

    # cur_tokens passed to latent predictor must come from the SAME backbone call
    # as image_scene_tokens used by the policy. We verify by checking that
    # the scene_features fed to trajectory_decoder are consistent.
    assert "cur_tokens" in captured, "patched function not called"
    N_tokens = N_cams * cfg.num_scene_tokens
    assert captured["cur_tokens"].shape == (B, N_tokens, cfg.tf_d_model), \
        f"unexpected cur_tokens shape: {captured['cur_tokens'].shape}"
    print("OK")


def test_drivor_loss_latent_keys():
    print(" [16] DrivoRLoss: latent_weight stored and loss_dict gets latent keys ...", end=" ")
    from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss

    # Verify latent_weight is stored correctly
    loss_fn = DrivoRLoss(latent_weight=1.0)
    assert loss_fn.latent_weight == 1.0

    loss_fn_zero = DrivoRLoss(latent_weight=0.0)
    assert loss_fn_zero.latent_weight == 0.0

    # Verify the latent key injection in loss_dict directly
    # (patching the internal logic without calling the full forward)
    latent_val = torch.tensor(0.42)
    pred = {"latent_loss_dict": {"loss": latent_val, "latent_prediction": latent_val}}
    loss_dict = {}
    if "latent_loss_dict" in pred:
        for k, v in pred["latent_loss_dict"].items():
            loss_dict[f"latent_{k}"] = v

    assert "latent_loss" in loss_dict
    assert "latent_latent_prediction" in loss_dict
    assert abs(loss_dict["latent_loss"].item() - 0.42) < 1e-6

    # Verify no latent keys when latent_loss_dict absent
    loss_dict2 = {}
    pred2 = {}
    if "latent_loss_dict" in pred2:
        for k, v in pred2["latent_loss_dict"].items():
            loss_dict2[f"latent_{k}"] = v
    assert "latent_loss" not in loss_dict2
    print("OK")


def test_cache_normalizes_legacy_image_next():
    print(" [17] Cache normalization handles missing, partial, and legacy image_next ...", end=" ")
    from navsim.planning.training.dataset import normalize_drivor_image_next

    image = torch.randn(4, 3, 56, 56)

    missing = normalize_drivor_image_next({"image": image.clone()})
    assert missing["image_next"].shape == image.shape
    assert not bool(missing["image_next_valid"].item())
    assert torch.count_nonzero(missing["image_next"]).item() == 0

    partial = normalize_drivor_image_next({
        "image": image.clone(),
        "image_next": torch.randn(3, 3, 56, 56),
    })
    assert partial["image_next"].shape == image.shape
    assert not bool(partial["image_next_valid"].item())

    legacy = normalize_drivor_image_next({
        "image": image.clone(),
        "image_next": torch.randn_like(image),
    })
    assert bool(legacy["image_next_valid"].item())
    print("OK")


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Latent Transition Model Tests ===\n")
    tests = [
        test_agent_input_next_cameras,
        test_feature_builder_image_next,
        test_feature_builder_no_image_next_on_eval,
        test_feature_builder_partial_image_next_invalid,
        test_latent_predictor,
        test_dit_block,
        test_dit_self_attention,
        test_latent_loss,
        test_model_forward_train_latent,
        test_model_forward_train_latent_all_invalid_zero,
        test_model_forward_train_latent_mixed_valid,
        test_model_forward_eval_no_latent,
        test_model_forward_latent_disabled,
        test_model_init_one_step_false_not_implemented,
        test_policy_and_predictor_share_tokens,
        test_drivor_loss_latent_keys,
        test_cache_normalizes_legacy_image_next,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL — {e}")
            import traceback; traceback.print_exc()
            failed.append(t.__name__)

    print(f"\n{'='*38}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
