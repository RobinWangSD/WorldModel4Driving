from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .layers.latent_predictor import LatentPredictor
from .layers.losses.latent_loss import LatentLoss
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

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

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

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
            )


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
        has_valid_latent = False
        want_latent = self.use_latent and self.training and self.one_step and self.num_cams > 0
        # image features
        if self.num_cams > 0:
            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            if want_latent:
                img_next, latent_valid_mask = self._get_valid_image_next(features, img)
                has_valid_latent = img_next is not None and bool(latent_valid_mask.any().item())

            if has_valid_latent:
                # Encode all current frames and only valid next frames in one backbone call.
                img_pair = torch.cat([img, img_next[latent_valid_mask]], dim=0)
                scene_tokens = self.scene_embeds.repeat(img_pair.shape[0], 1, 1, 1)
                pair_tokens = self.image_backbone(img_pair, scene_tokens)
                image_scene_tokens = pair_tokens[:batch_size]   # o_t → policy + predictor input
                next_tokens = pair_tokens[batch_size:]           # o_{t+1} → latent target
            else:
                scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
                image_scene_tokens = self.image_backbone(img, scene_tokens)

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
            if has_valid_latent:
                latent_loss_dict = self._compute_latent_loss(
                    features,
                    image_scene_tokens[latent_valid_mask],
                    next_tokens,
                    latent_valid_mask,
                )
            else:
                latent_loss_dict = self._zero_latent_loss(image_scene_tokens)

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
            return None, invalid_mask
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

    def _zero_latent_loss(self, reference):
        zero = reference.new_zeros(())
        return {
            "loss": zero,
            "latent_prediction": zero,
        }

    def _compute_latent_loss(self, features, cur_tokens, next_tokens, valid_mask=None):
        """p(o_{t+1} | o_t, a_t): predict next-frame tokens from current scene tokens."""
        ego_status = features["ego_status"]
        if valid_mask is not None:
            ego_status = ego_status[valid_mask]
        ego_t = ego_status[:, -1] if ego_status.dim() == 3 else ego_status
        predicted = self.latent_predictor(cur_tokens, ego_t)
        return self.latent_loss_fn(predicted.unsqueeze(1), next_tokens.unsqueeze(1))
