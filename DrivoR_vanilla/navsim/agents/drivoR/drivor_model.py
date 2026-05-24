from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .layers.latent_predictor import LatentPredictor, WorldModelLatentPredictor
from .layers.losses.latent_loss import LatentLoss, MultiStepLatentLoss
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

_ALL_CAM_KEYS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")


def _temporal_caching_enabled(config) -> bool:
    tc = config.get("temporal_caching", None) if hasattr(config, "get") else getattr(config, "temporal_caching", None)
    if tc is None:
        return False
    return bool(tc.get("enabled", False) if hasattr(tc, "get") else getattr(tc, "enabled", False))


class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model
        latent_cfg = config.get("latent_learning", None)
        self.use_latent = latent_cfg is not None and latent_cfg.get("enabled", False)
        self.one_step = True
        if self.use_latent:
            self.one_step = latent_cfg.get("one_step", True)
            if not self.one_step:
                raise NotImplementedError(
                    "latent_learning.one_step=False is reserved for future history latent learning "
                    "and is not implemented yet."
                )

        # Temporal caching / multi-mode training
        self.temporal_caching_enabled = _temporal_caching_enabled(config)
        self.input_mode = config.get("input_mode", "current_image") if hasattr(config, "get") else getattr(config, "input_mode", "current_image")
        self.predict_future_latent = bool(config.get("predict_future_latent", False)) if hasattr(config, "get") else bool(getattr(config, "predict_future_latent", False))
        self.latent_pred_steps = int(config.get("latent_pred_steps", 10)) if hasattr(config, "get") else int(getattr(config, "latent_pred_steps", 10))

        if self.temporal_caching_enabled:
            tc = config["temporal_caching"]
            self.t_hist = len(tc["history_iters"])
            self.t_fut = len(tc["future_iters"])
            assert 1 <= self.latent_pred_steps <= self.t_fut, (
                f"latent_pred_steps={self.latent_pred_steps} must be in [1, {self.t_fut}]"
            )
        else:
            self.t_hist = 1
            self.t_fut = 0

        assert self.input_mode in ("current_image", "history_images"), (
            f"Unknown input_mode={self.input_mode!r}; must be 'current_image' or 'history_images'"
        )
        if self.input_mode == "history_images" and not self.temporal_caching_enabled:
            raise ValueError("input_mode='history_images' requires temporal_caching.enabled=true")
        if self.predict_future_latent and not self.temporal_caching_enabled:
            raise ValueError("predict_future_latent=true requires temporal_caching.enabled=true")

        ###########################################
        # camera embedding
        if self.temporal_caching_enabled:
            active = set(config["temporal_caching"]["active_cameras"])
            self.num_cams = sum(1 for k in _ALL_CAM_KEYS if k in active)
        else:
            self.num_cams = 0
            for k in _ALL_CAM_KEYS:
                if len(self._config[k]) > 0:
                    self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # Effective number of "views" the image backbone sees per sample.
        # In history_images mode we fold T_hist into the camera dimension so the
        # backbone treats each (t, cam) as an independent view.
        if self.input_mode == "history_images":
            self.effective_num_cams = self.num_cams * self.t_hist
        else:
            self.effective_num_cams = self.num_cams

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(
                torch.randn(1, self.effective_num_cams, self._config.num_scene_tokens, self.image_backbone.num_features) * 1e-6,
                requires_grad=True,
            )

            # Temporal positional embedding for history_images: one vector per
            # history step, broadcast over cameras and scene tokens.
            if self.input_mode == "history_images":
                self.temporal_pe = nn.Parameter(
                    torch.zeros(1, self.t_hist, 1, 1, self.image_backbone.num_features),
                    requires_grad=True,
                )

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        self.b2d=config.b2d

        # Latent space learning (optional)
        if self.use_latent:
            pred_cfg = latent_cfg.get("predictor", {})
            self.latent_predictor = LatentPredictor(
                d_model=config.tf_d_model,
                nhead=pred_cfg.get("nhead", 4),
                num_layers=pred_cfg.get("num_layers", 2),
                d_ffn=pred_cfg.get("d_ffn", 512),
                ego_dim=11,
            )
            loss_w = latent_cfg.get("loss_weights", {})
            self.latent_loss_fn = LatentLoss(
                prediction_weight=loss_w.get("prediction", 1.0),
                stop_grad_target=latent_cfg.get("stop_grad_target", True),
                sigreg_weight=loss_w.get("sigreg", 0.0),
                sigreg_knots=latent_cfg.get("sigreg_knots", 17),
                sigreg_num_proj=latent_cfg.get("sigreg_num_proj", 1024),
            )

        # World-model multi-step latent prediction (Mode 3).
        # Independent of the legacy one-step `latent_learning` path above —
        # this one runs when `predict_future_latent=true` and uses cached
        # `image_future` as targets and `ego_status_future` as teacher-forced
        # action conditioning.
        if self.predict_future_latent:
            if self.num_cams == 0:
                raise ValueError("predict_future_latent=true requires at least one active camera")
            # Reuse latent_learning predictor sub-config if available so the user
            # can tune nhead/num_layers/d_ffn through the existing config block.
            pred_cfg = latent_cfg.get("predictor", {}) if latent_cfg is not None else {}
            stop_grad = latent_cfg.get("stop_grad_target", True) if latent_cfg is not None else True
            n_tokens_per_view = config.num_scene_tokens
            # Predictor output is matched to a single-step target latent shape:
            # (num_cams * num_scene_tokens) tokens per future step. When input_mode
            # folds T_hist into the camera dimension, the predictor input is
            # collapsed back over time by mean-pooling in _compute_world_model_loss.
            self.world_model_predictor = WorldModelLatentPredictor(
                d_model=config.tf_d_model,
                nhead=pred_cfg.get("nhead", 4),
                num_layers=pred_cfg.get("num_layers", 2),
                d_ffn=pred_cfg.get("d_ffn", 512),
                ego_dim=11,
                n_pred_steps=self.latent_pred_steps,
                n_tokens=self.num_cams * n_tokens_per_view,
            )
            self.world_model_loss_fn = MultiStepLatentLoss(stop_grad_target=stop_grad)


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        next_tokens = None
        latent_valid_mask = None
        want_world_model = (
            self.predict_future_latent
            and self.training
            and self.num_cams > 0
            and "image_future" in features
            and "ego_status_future" in features
        )
        # The world-model head supersedes the legacy one-step latent path: both
        # consume cached "next image" data and emit a `latent_loss_dict`, so
        # enabling both at once would double-count the latent loss.
        want_latent = (
            self.use_latent
            and self.training
            and self.one_step
            and self.num_cams > 0
            and not want_world_model
        )
        # image features
        if self.num_cams > 0:
            img = self._select_input_image(features)  # (B, effective_num_cams, C, H, W)

            if want_latent:
                # legacy one-step latent path: needs same-shape img_next as img
                img_next, latent_valid_mask = self._get_valid_image_next(features, img)

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            scene_tokens = self._apply_temporal_pe(scene_tokens)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            if want_latent:
                next_scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
                if self.latent_loss_fn.stop_grad_target:
                    with torch.no_grad():
                        next_tokens = self.image_backbone(img_next, next_scene_tokens)
                else:
                    next_tokens = self.image_backbone(img_next, next_scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # Latent transition model: p(o_{t+1} | o_t, a_t)
        latent_loss_dict = None
        if want_latent:
            latent_loss_dict = self._compute_latent_loss(
                features,
                image_scene_tokens,
                next_tokens,
                latent_valid_mask,
            )

        # World-model multi-step latent prediction (Mode 3).
        if want_world_model:
            latent_loss_dict = self._compute_world_model_loss(features, image_scene_tokens)

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        if latent_loss_dict is not None:
            output["latent_loss_dict"] = latent_loss_dict

        return output

    def _get_valid_image_next(self, features, img):
        batch_size = img.shape[0]
        invalid_mask = torch.zeros(batch_size, dtype=torch.bool, device=img.device)
        img_next = features.get("image_next", None)
        if not isinstance(img_next, torch.Tensor) or img_next.shape != img.shape:
            return torch.zeros_like(img), invalid_mask
        img_next = img_next.to(device=img.device, dtype=img.dtype)

        valid_mask = features.get("image_next_valid", None)
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=img.device)
        elif isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.to(device=img.device, dtype=torch.bool).reshape(-1)
            if valid_mask.numel() == 1 and batch_size != 1:
                valid_mask = valid_mask.expand(batch_size)
            elif valid_mask.numel() != batch_size:
                valid_mask = invalid_mask
        else:
            valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=img.device).reshape(-1)
            if valid_mask.numel() == 1 and batch_size != 1:
                valid_mask = valid_mask.expand(batch_size)
            elif valid_mask.numel() != batch_size:
                valid_mask = invalid_mask

        return img_next, valid_mask

    def _compute_latent_loss(self, features, cur_tokens, next_tokens, valid_mask=None):
        """p(o_{t+1} | o_t, a_t): predict next-frame tokens from current scene tokens."""
        ego_status = features["ego_status"]
        ego_t = ego_status[:, -1] if ego_status.dim() == 3 else ego_status
        predicted = self.latent_predictor(cur_tokens, ego_t)
        return self.latent_loss_fn(predicted.unsqueeze(1), next_tokens.unsqueeze(1), valid_mask=valid_mask)

    def _select_input_image(self, features):
        """Return (B, effective_num_cams, C, H, W) image tensor based on input_mode.

        - current_image: last history step or legacy `image` key
        - history_images: full image_history tensor, T_hist folded into the camera dim
        """
        if self.input_mode == "history_images":
            assert "image_history" in features, "input_mode=history_images requires features['image_history']"
            img_hist = features["image_history"]  # (B, T_hist, N_cam, C, H, W)
            T, N = img_hist.shape[1], img_hist.shape[2]
            assert T == self.t_hist, f"image_history T={T} but model expects {self.t_hist}"
            assert N == self.num_cams, f"image_history N={N} but model expects {self.num_cams}"
            return rearrange(img_hist, "b t n c h w -> b (t n) c h w")

        # current_image
        if "image_history" in features:
            img = features["image_history"][:, -1]  # (B, N_cam, C, H, W)
        elif "image" in features:
            img = features["image"]
        elif "camera_feature" in features:
            img = features["camera_feature"]
        else:
            raise ValueError("No image tensor found in features")
        return img

    def _apply_temporal_pe(self, scene_tokens):
        """Add a per-history-step bias to scene_tokens shaped (B, T_hist*N_cam, num_scene_tokens, D)."""
        if self.input_mode != "history_images":
            return scene_tokens
        B, TN, S, D = scene_tokens.shape
        pe = self.temporal_pe.expand(B, self.t_hist, self.num_cams, S, D).reshape(B, TN, S, D)
        return scene_tokens + pe

    def _compute_world_model_loss(self, features, image_scene_tokens):
        """Multi-step latent prediction loss using cached image_future and ego_status_future.

        Predictor I/O: (B, num_cams*S, D) -> (B, T_pred, num_cams*S, D).
        Target: backbone applied to each future step independently (single time step
        worth of cameras), shape (B, T_pred, num_cams*S, D).
        """
        image_future = features["image_future"]  # (B, T_fut, N_cam, C, H, W)
        ego_future = features["ego_status_future"]  # (B, T_fut, 11)
        future_valid = features.get("future_valid", None)  # (B, T_fut) bool

        if image_scene_tokens.dim() != 3:
            raise RuntimeError(
                f"image_scene_tokens expected (B, T, D); got shape {tuple(image_scene_tokens.shape)}"
            )
        D = image_scene_tokens.shape[-1]
        N_cam = self.num_cams
        S = self._config.num_scene_tokens

        T_pred = self.latent_pred_steps
        B = image_future.shape[0]
        img_f = image_future[:, :T_pred]  # (B, T_pred, N_cam, C, H, W)
        ego_f = ego_future[:, :T_pred]
        T_pred_, C, H, W = img_f.shape[1], img_f.shape[3], img_f.shape[4], img_f.shape[5]
        img_f_flat = img_f.reshape(B * T_pred_, N_cam, C, H, W)

        # Build a per-view scene-embed (collapse the time dim if input_mode folded it in).
        if self.input_mode == "history_images":
            per_view_embeds = self.scene_embeds.view(1, self.t_hist, N_cam, S, D).mean(dim=1)
        else:
            per_view_embeds = self.scene_embeds  # (1, N_cam, S, D)
        target_embeds = per_view_embeds.expand(B * T_pred_, -1, -1, -1).contiguous()

        if self.world_model_loss_fn.stop_grad_target:
            with torch.no_grad():
                target_tokens = self.image_backbone(img_f_flat, target_embeds)
        else:
            target_tokens = self.image_backbone(img_f_flat, target_embeds)
        # target_tokens: (B*T_pred, N_cam*S, D) -> (B, T_pred, N_cam*S, D)
        target_tokens = target_tokens.view(B, T_pred_, N_cam * S, D)

        # Collapse the time-folded input down to per-view token count for the predictor.
        if self.input_mode == "history_images":
            cur_tokens = image_scene_tokens.view(B, self.t_hist, N_cam * S, D).mean(dim=1)
        else:
            cur_tokens = image_scene_tokens

        # Predict
        ego_hist = features.get("ego_status_history", features["ego_status"])  # (B, T_hist, 11)
        predicted = self.world_model_predictor(cur_tokens, ego_hist, ego_f)  # (B, T_pred, N_cam*S, D)

        return self.world_model_loss_fn(
            predicted,
            target_tokens,
            valid_mask=(future_valid[:, :T_pred] if future_valid is not None else None),
        )
