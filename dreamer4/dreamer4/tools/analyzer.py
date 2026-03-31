import os
import sys
import gym
import time
import json
import faiss
import torch
from torch import nn
import argparse
# import open_clip
import numpy as np
from PIL import Image
# from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from safety_rl.gym_reachability import gym_reachability  # Custom Gym env.

sys.path.append('model_based_irl_torch')
sys.path.append('safety_rl')
sys.path.append('scripts')
import ruamel.yaml as yaml
# from train_dubins_wm import *
import model_based_irl_torch.dreamer.tools as tools
import model_based_irl_torch.dreamer.models as models
import model_based_irl_torch.dreamer.exploration as expl
from model_based_irl_torch.common.constants import HORIZONS

import io
import copy
import random
import collections
import matplotlib.patches as patches
import imageio.v2 as imageio  # use v2 for stable API

from model_based_irl_torch.dreamer.dreamer import Dreamer
from model_based_irl_torch.dreamer.FlowMatching.conditional_unet1d import *

from generate_data_traj_failure_expert import *

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def _paper_rcparams():
    # Conservative defaults that look good in most conference templates.
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 30,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,   # TrueType fonts in PDF (better compatibility)
        "ps.fonttype": 42,
        "axes.titleweight": "bold",
    })

def get_img_from_idx(episodes, eps_key, idx):

    counter = 0
    for k_ep in eps_key:
        
        episode = episodes[k_ep]
        N = len(episode['image'])
        if counter + N > idx:
            local_idx = idx - counter
            img = episode['image'][local_idx]
            # state = episode[''][local_idx]
            return img
        counter += N 
    assert False

class Analyzer:

    def __init__(self, path, config,
                 dynamics_sample = False,
                 loaded_agent = None,
                 dataset_type = "large"):

        self._config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamics_sample = dynamics_sample
        self.dataset_type = dataset_type
        self.path = path
        self.feat_list = ['stoch', 'deter']
        if path is not None:
            self.parent_folder = os.path.dirname(path)

        if loaded_agent is None:

            env_name = "dubins_car_img-v1"
            sample_inside_obs = config.doneType not in ['TF', 'fail']
            train_env = gym.make(env_name, config=config, device=device, mode=config.mode, doneType=config.doneType, sample_inside_obs=sample_inside_obs)
            self.agent = Dreamer(train_env.observation_space, train_env.action_space, config, None, None).to(config.device)
            self.agent.requires_grad_(requires_grad=False)
    
            checkpoint = torch.load(path, 
                                    map_location=config.device)
    
            self.agent._wm.dynamics.sample = dynamics_sample
    
            # agent_state_dict = checkpoint['agent_state_dict']
            # self.agent.load_state_dict(agent_state_dict)
        
            state_dict = {k[4:]: v for k, v in checkpoint['agent_state_dict'].items() if '_wm' in k}
            self.agent._wm.load_state_dict(state_dict)
            self.agent._wm.eval()
    
            state_dict = {k[4:]: v for k, v in checkpoint['agent_state_dict'].items() if '_fm' in k}
            self.agent._fm.load_state_dict(state_dict)
            self.agent._fm.eval()
    
            state_dict = {k[16:]: v for k, v in checkpoint['agent_state_dict'].items() if '_disag_ensemble' in k}
            self.agent._disag_ensemble.load_state_dict(state_dict)
            self.agent._disag_ensemble.eval()

        else:

            self.agent = loaded_agent

        u_max = self._config.u_max
        self.u_max = u_max
        self.v = self._config.speed
        self.dt = self._config.dt
        self.ac_pool = np.array([-u_max, 0, u_max])
        self.x_min = self._config.x_min + 0.1
        self.y_min = self._config.y_min + 0.1
        self.x_max = self._config.x_max - 0.1
        self.y_max = self._config.y_max - 0.1
        self.state_pos_limits = np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_max],
        ])

        BRT_slice, grid_x, grid_y, grid_theta = load_gt('logs/v_1_w_1.25_brt.mat')
        self.interpolator = create_interpolator(BRT_slice, grid_x, grid_y, grid_theta)

        self.calibrated_values = None
        self.index = None # nearest neighbor
        self.k_list = None

    def reset(self,):

        self.h = self.agent._wm.dynamics.initial(1)
        ac = torch.zeros((1, self.agent._wm.dynamics._num_actions)).to(self.agent._wm.dynamics._device)
        self.unroll_latent_deter(ac)
        # self.unroll_latent_stoch(obs)
        

    def unroll_latent_deter(self, ac,
                            override = True,
                            external_h = None):

        if external_h is not None:
            h = external_h
        else:
            h = self.h
        if len(ac.shape) == 1:
            ac = ac.unsqueeze(dim = 0)

        h_prior = self.agent._wm.dynamics.img_step(h, ac, 
                                                   sample = self.dynamics_sample)
        if override:
            self.h = h_prior
        return h_prior

    def unroll_latent_stoch(self, obs,
                            override = True,
                            external_h = None):

        if external_h is not None:
            h = external_h 
        else:
            h = self.h

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs).to(self.agent._wm.dynamics._device).float() 
        embed = self.agent._wm.encoder({
            'image': obs.unsqueeze(dim = 0) / 255.
        })
        x = torch.cat([h['deter'], embed], dim = -1)
        x = self.agent._wm.dynamics._obs_out_layers(x)
        stats = self.agent._wm.dynamics._suff_stats_layer("obs", x)
        if self.dynamics_sample:
            stoch = self.agent._wm.dynamics.get_dist(stats).sample() # post z_1
        else:
            stoch = self.agent._wm.dynamics.get_dist(stats).mode() # post z_1
        h = {"stoch": stoch, "deter": h["deter"], **stats}
        if override:
            self.h = h
        return h

    def get_obs(self, state, radius = 0.4, dpi = 128):

        dt, v = self.dt, self.v

        # Render the environment
        fig, ax = plt.subplots()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis('off')
        fig.set_size_inches(1, 1)
    
        # Draw the circle
        circle = patches.Circle((0., 0.), 
                                radius, edgecolor="#b3b3b3", facecolor="#b3b3b3", linewidth=2)
        ax.add_patch(circle)
    
        # Draw the trajectory
        plt.quiver(
          state[0], state[1],
          dt * v * np.cos(state[2]),
          dt * v * np.sin(state[2]),
          angles='xy', scale_units='xy', minlength=0,
          width=0.1, scale=0.18, color='black', zorder=3
        )
    
        plt.scatter(
            state[0], state[1],
            s=20, color='black', zorder=3
        )
    
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Save the frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close()
        
        return np.array(img) 

    def unroll_dubins(self, state, ac):

        next_state = np.zeros(3)
        next_state[0] = state[0] + self.v * self.dt * np.cos(state[2])
        next_state[1] = state[1] + self.v * self.dt * np.sin(state[2])
        next_state[2] = state[2] + self.dt * ac
        if next_state[2] > 2 * np.pi: 
            next_state[2] -= 2*np.pi
        if next_state[2] < 0: 
            next_state[2] += 2*np.pi
        
        return next_state

    def get_safety_value(self, state, ac):

        # unroll dubins 
        next_state = self.unroll_dubins(state, ac)
        # infer the value
        value = self.interpolator(next_state)
        return value[0]

    def get_latent_stats(self, state, ac, latent,
                         fm_num_step = 1,):

        # unroll dubins 
        next_state = self.unroll_dubins(state, ac)
        # get visual obs 
        gt_next_obs = self.get_obs(next_state)

        # unroll the latent dynamics
        ac_tensor = torch.tensor(ac).float().cuda().view([1, -1])
        prior = self.unroll_latent_deter(ac = ac_tensor,
                                         external_h = latent,
                                         override = False) 
        post = self.unroll_latent_stoch(obs = gt_next_obs,
                                        external_h = prior,
                                        override = False)

        # latent error
        error = ((prior['stoch'] - post['stoch'])**2).mean()
        # grad norm 
        feat = self.agent._wm.dynamics.get_feat(latent)
        norm = self.agent.compute_grad_norm(latent, ac_tensor,
                                            use_gt_jacobian = True)
        # uncertainty
        uncertainty_fm = self.agent._fm._compute_logpZO(X0 = prior['deter'] / 10.,
                                                        context = torch.concat([
                                                            feat, ac_tensor,
                                                        ], dim = -1),
                                                        num_step = 1) #fm_num_step)
        uncertainty_ensemble = self.agent._disag_ensemble._intrinsic_reward_penn(feat, ac_tensor)

        embeds = torch.cat([prior[feat_name] for feat_name in self.feat_list],
                           dim = -1).detach().cpu().numpy()
        distances, _ = self.index.search(embeds, k = self._config.nn_k)
        uncertainty_nn = np.mean(distances, axis = -1)

        return error.item(), norm.item(), uncertainty_fm.item(), uncertainty_ensemble.item(), uncertainty_nn.item()
        

    """
        Per step, do 1) observation visual, 2) safety values, 
        3) errors between (z_{t+1}|z_{t}, u_{t}) vs (z_{t+1}|z_{t}, u_{t}, o_{t+1}), 
        4) grad norm, 5) uncertainty fm and 6) uncertainty ensemble
    """
    def eval_prediction(self, info, step,
                        N_ac = 100,
                        fm_num_step = 1):

        state = info['state']
        state_np = state.detach().cpu().numpy()
        ac_raw = info['ac'].item()

        print ("At step {} with control: {:.3f}, state: {}".format(step+1, 
																   ac_raw, 
																   np.round(state_np, decimals = 3)))
        if self.index is not None and self.k_list is not None:
            # embed = torch.cat([
            #     self.h['stoch'], self.h['deter']
            # ], dim = -1).detach().cpu().numpy()
            # embed = self.h['stoch'].detach().cpu().numpy()
            embed = torch.cat([self.h[feat_name] for feat_name in self.feat_list], dim = -1)
            embed = embed.detach().cpu().numpy()
            for k, nn_scores_sorted in zip(self.k_list,
                                           self.nn_scores_sorted_all):
                distances, _ = self.index.search(embed, k)
                nn_score = np.mean(distances, axis = -1).item()
                percentile = np.searchsorted(nn_scores_sorted, 
                                             nn_score, 
                                             side="right") / len(nn_scores_sorted)
                print ("Nearest Neighbor (K = {}) score: {:.3f} [p @ {:.3f}]".format(
                    k, nn_score, percentile
                ))
            
        fig, axes = plt.subplots(1, 7, figsize=(21, 3))
        axes[0].imshow(info['obs'].detach().cpu().numpy() / 255.)
        axes[0].axis('off')
        axes[0].set_title("observation")

        ac_pool = np.linspace(-self.u_max, self.u_max, N_ac)
        safety_value_list = []
        latent_error_list = []
        grad_norm_list = []
        uncertainty_fm_list = []
        uncertainty_ensemble_list = []
        uncertainty_nn_list = []
        for ac_temp in ac_pool:
            # getting GT safety score over (z_t, a) in which
            # we unroll the dubins vehicle dynamics
            safety_value = self.get_safety_value(state_np, ac_temp)
            safety_value_list.append(safety_value)
            # getting 1) latent prediction errors using
            # GT o_{t+1} obtained via unrolling, and 
            # 2) grad norm, 3) and 4) uncertainties.
            latent_error, grad_norm, uncertainty_fm, uncertainty_ensemble, uncertainty_nn = self.get_latent_stats(state_np, ac_temp, info['h'])
            latent_error_list.append(latent_error)
            grad_norm_list.append(grad_norm)
            uncertainty_fm_list.append(uncertainty_fm)
            uncertainty_ensemble_list.append(uncertainty_ensemble)
            uncertainty_nn_list.append(uncertainty_nn)

        safety_value_list = np.array(safety_value_list)
        axes[1].plot(ac_pool, safety_value_list)
        axes[1].fill_between(
            ac_pool,
            safety_value_list,
            -0.1,
            where=(safety_value_list >= 0),
            interpolate=True,
            alpha=0.3
        )
        axes[1].axvline(x=ac_raw, color="red")
        axes[1].set_ylim([-0.1, 0.2])
        axes[1].set_xlabel("u")
        axes[1].set_title("GT safety")

        latent_error_list = np.array(latent_error_list)
        axes[2].plot(ac_pool, latent_error_list)
        axes[2].axvline(x=ac_raw, color="red")
        axes[2].set_ylim([0., 0.3])
        axes[2].set_xlabel("u")
        axes[2].set_title("latent error")

        grad_norm_list = np.array(grad_norm_list)
        axes[3].plot(ac_pool, grad_norm_list)
        axes[3].axvline(x=ac_raw, color="red")
        axes[3].set_ylim([0., 8.])
        axes[3].set_xlabel("u")
        axes[3].set_title("grad norm")

        uncertainty_fm_list = np.array(uncertainty_fm_list)
        axes[4].plot(ac_pool, uncertainty_fm_list)
        axes[4].axvline(x=ac_raw, color="red")
        if self.calibrated_values is not None:
            for k, v in self.calibrated_values['fm'].items():
                axes[4].axhline(y=v, linestyle='--',
                                color='black', alpha=0.3,
                                label=k)
        axes[4].legend()
        axes[4].axvline(x=ac_raw, color="red")
                
        axes[4].set_ylim([0, 35])
        axes[4].set_xlabel("u")
        axes[4].set_title("uncertainty (FM)")

        uncertainty_ensemble_list = np.array(uncertainty_ensemble_list)
        axes[5].plot(ac_pool, uncertainty_ensemble_list)
        axes[5].axvline(x=ac_raw, color="red")
        if self.calibrated_values is not None:
            for k, v in self.calibrated_values['ensemble'].items():
                axes[5].axhline(y=v, linestyle='--',
                                color='black', alpha=0.3,
                                label=k)
        axes[5].legend()
        axes[5].set_ylim([1., 5.])
        axes[5].set_xlabel("u")
        axes[5].set_title("uncertainty (ensemble)")

        uncertainty_nn_list = np.array(uncertainty_nn_list)
        axes[6].plot(ac_pool, uncertainty_nn_list)
        axes[6].axvline(x=ac_raw, color="red")
        if self.calibrated_values is not None:
            for k, v in self.calibrated_values['nn'].items():
                axes[6].axhline(y=v, linestyle='--',
                                color='black', alpha=0.3,
                                label=k)
        axes[6].legend()
        axes[6].set_ylim([0., 100.])
        axes[6].set_xlabel("u")
        axes[6].set_title("uncertainty (NN)")

        plt.subplots_adjust(wspace=0.4)   # default ~0.2
        plt.show()
        print ("--------------------------------------")

    def draw_trajectory_from_episodes(self, episodes,
                                      traj_id = None):

        if traj_id is None:
            traj_id = np.random.randint(len(episodes))
        # print (traj_id)
        episode = episodes[list(episodes.keys())[traj_id]]
        trajectory = {}
        for k, v in episode.items():
            trajectory[k] = torch.tensor(np.array(v)).float().cuda()
        return trajectory, traj_id
        
    def calibrate_thresholds(self, episodes, path,
                             quantiles = [0.75, 0.9, 0.95],
                             override = False,
                             # ema = 0.01, num_batches = 250,
                            ):

        quantiles_tensor = torch.tensor(quantiles)

        N = 0 
        for k_ep in episodes:
            episode = episodes[k_ep]
            N += len(episode['image'])

        if (not os.path.exists(path)) or override:

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # 3 for [nn, fm, ensemble]
            E = np.memmap(path, mode="w+", dtype=np.float32, shape=(N, 3))
    
            with torch.no_grad():
                
                idx = 0
                from tqdm import tqdm
                for k_ep in tqdm(episodes):
                    
                    episode = episodes[k_ep]
                    trajectory = {}
                    for k, v in episode.items():
                        trajectory[k] = torch.tensor(np.array(v)).float().cuda()
    
                    self.reset()
                    observations = trajectory['image']
                    actions = trajectory['action']
                    N = len(observations)
                    all_embeds = []
                    for counter in range(N):
                        obs = observations[counter]
                        ac = actions[counter].view(1, -1)
                        self.unroll_latent_stoch(observations[counter]) # h_{t}, s_{t}
                        u_fm_candidate = self.current_uncertainty(ac, 'fm')
                        u_nn_candidate = self.current_uncertainty(ac, 'nn') 
                        u_ensemble_candidate = self.current_uncertainty(ac, 'ensemble') 
                        E[idx] = np.array([u_fm_candidate, u_nn_candidate, u_ensemble_candidate])
                        idx += 1
                        self.unroll_latent_deter(ac) # h_{t+1}, s_{t} 
    
            E.flush()    
        
        scores = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(N, 3),
        )

        quantiled_scores = np.quantile(scores, quantiles, axis = 0)

        self.calibrated_values = {
            'fm': {
                str(quantiles[i]): quantiled_scores[i, 0] for i in range(len(quantiles))
            },
            'ensemble': {
                str(quantiles[i]): quantiled_scores[i, 2] for i in range(len(quantiles))
            },
            'nn': {
                str(quantiles[i]): quantiled_scores[i, 1] for i in range(len(quantiles))
            }
        }
        print (self.calibrated_values)
        return scores

    def current_uncertainty(self, ac, 
                            uncertainty_type = 'fm'):

        if not isinstance(ac, torch.Tensor):
            ac = torch.tensor(ac).float().cuda().view([1, -1])
        feat = self.agent._wm.dynamics.get_feat(self.h)

        if uncertainty_type == 'fm':
            prior = self.unroll_latent_deter(ac = ac,
                                             override = False)
            uncertainty = self.agent._fm._compute_logpZO(X0 = prior['deter'] / 10.,
                                                         context = torch.concat([
                                                             feat, ac,
                                                         ], dim = -1),
                                                         num_step = 1)
        elif uncertainty_type == 'ensemble':
            uncertainty = self.agent._disag_ensemble._intrinsic_reward_penn(feat, ac)
        elif uncertainty_type == 'nn':
            uncertainty = self.compute_nn_score(self.h, ac)
        return uncertainty.item()

    def current_uncertainty_from_feat(self, feat, ac, 
                            uncertainty_type = 'fm'):

        if not isinstance(ac, torch.Tensor):
            ac = torch.tensor(ac).float().cuda().view([1, -1])
        next_feat = self.unroll_from_feat(feat[:, :32],
                                          feat[:, 32:], 
                                          ac) 
        
        if uncertainty_type == 'fm':
            uncertainty = self.agent._fm._compute_logpZO(X0 = next_feat[:, 32:] / 10.,
                                                         context = torch.concat([
                                                             feat, ac,
                                                         ], dim = -1),
                                                         num_step = 1)
        elif uncertainty_type == 'ensemble':
            uncertainty = self.agent._disag_ensemble._intrinsic_reward_penn(feat, ac)
        elif uncertainty_type == 'nn':
            uncertainty = self.compute_nn_score(None, ac, feat = feat)

        return uncertainty

    def unroll_from_feat(self, stoch, deter, a):
        wm = self.agent._wm
        x = torch.cat([stoch, a], dim = -1)
        x = wm.dynamics._img_in_layers(x)
        x, deter = wm.dynamics._cell(x, [deter])
        deter = deter[0]
        x = wm.dynamics._img_out_layers(x)
        stats = wm.dynamics._suff_stats_layer("ims", x)
        stoch = wm.dynamics.get_dist(stats).mode()
        return torch.cat([stoch, deter], dim = -1) 

    def compute_nn_score(self, latent, ac,
                         feat = None):

        # first, unroll z_{t+1} from z_{t} and u_{t} 
        if feat is None:
            next_latent = self.agent._wm.dynamics.img_step(latent, ac, sample = self.dynamics_sample)
            next_feat = torch.cat([next_latent['stoch'],
                                   next_latent['deter']], dim = -1)
        else:
            next_feat = self.unroll_from_feat(feat[:, :32],
                                         feat[:, 32:],
                                         ac)
            
        if len(next_feat.shape) == 1:
            next_feat = next_feat.unsqueeze(dim = 0)

        assert self.index is not None
        distances, _ = self.index.search(next_feat.detach().cpu().numpy(),
                                         self._config.nn_k)
        distances = np.mean(distances, axis = -1)
        return distances

    def eval_batch(self, 
                   uncertainty_type = 'fm', 
                   quantile = 0.9,
                   N_sample = 100,
                   N_eval = 100,
                   terminate_on_ood = False):

        from tqdm.auto import trange

        threshold = None
        num_failure_ep = 0
        num_filtered_step = 0
        num_total_step = 0
        num_violated_step = 0
        all_violated_values = []
        successful_min_distances = []
        pbar = trange(1, N_eval + 1, desc="Eval", leave=True, dynamic_ncols=True)
        for counter in pbar:
            trajectory = self.eval_single(uncertainty_type = uncertainty_type,
                                          N_sample = N_sample, 
                                          quantile = quantile,
                                          terminate_on_ood = terminate_on_ood)
            if threshold is None:
                threshold = trajectory['u_threshold']
            assert threshold == trajectory['u_threshold']

            if trajectory['if_failure']:
                num_failure_ep += 1 
            else:
                states = torch.stack(trajectory['privileged_state'], 
                                     dim = 0)[:, :2]
                distances = states.norm(dim = -1)
                successful_min_distances.append(distances.min().item())

            uncertainty_traj = np.array(trajectory['uncertainty'])
            violated_mask = uncertainty_traj > threshold
            violated_values = uncertainty_traj[violated_mask]
            all_violated_values = np.concatenate([all_violated_values,
                                                  violated_values])
            num_violated_step += np.sum(violated_mask)
            num_filtered_step += np.sum(trajectory['filtered'])
            num_total_step += len(trajectory['filtered'])
            
            fail_rate = num_failure_ep / counter
            viol_rate = (num_violated_step / num_total_step) if num_total_step else 0.0
            viol_values = (np.mean(all_violated_values) / threshold - 1.) if len(all_violated_values) != 0 else 0.0
            filt_rate = (num_filtered_step / num_total_step) if num_total_step else 0.0

            md = np.mean(successful_min_distances) if len(successful_min_distances) != 0. else 0.
            pbar.set_postfix(
                fail=f"{fail_rate:.1%}",
                viol=f"{viol_rate:.1%}",
                # filt=f"{filt_rate:.1%}",
                # viol_v=f"{viol_values:.2f}"
                mdist=f"{md:.2f}"
            )
            
        fail_rate = num_failure_ep / max(N_eval, 1)
        viol_rate = (num_violated_step / num_total_step) if num_total_step else 0.0
        viol_values = (np.mean(all_violated_values) / threshold - 1.) if len(all_violated_values) != 0 else 0.0
        filt_rate = (num_filtered_step / num_total_step) if num_total_step else 0.0
    
        summary = (
            "\n"
            "==================== Eval Summary ====================\n"
            f"uncertainty_type : {uncertainty_type}\n"
            f"quantile         : {quantile}\n"
            f"N_sample         : {N_sample}\n"
            f"N_eval           : {N_eval}\n"
            f"threshold        : {threshold}\n"
            "------------------------------------------------------\n"
            f"failure eps      : {num_failure_ep:>6d} / {N_eval:<6d}  ({fail_rate:>7.2%})\n"
            f"violated steps   : {num_violated_step:>6d} / {num_total_step:<6d}  ({viol_rate:>7.2%})[{viol_values:.3f}]\n"
            f"filtered steps   : {num_filtered_step:>6d} / {num_total_step:<6d}  ({filt_rate:>7.2%})\n"
            f"min distance     : {np.mean(successful_min_distances):>6.3f}\n"
            
            "======================================================\n"
        )
        print(summary)

        return {
            "threshold": threshold,
            "failure": fail_rate,
            "violation": viol_rate,
            "filter": filt_rate,
            "distance": np.mean(successful_min_distances)
        }
            
    def eval_single(self, 
                    uncertainty_type = 'fm', 
                    quantile = 0.9,
                    initial_state = None,
                    T = 100,
                    N_sample = 100,
                    seed = None,
                    terminate_on_ood = False):

        random.seed(seed)
        np.random.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if initial_state is None:
            while True:
                state = np.random.uniform(low = 0., high = 1., size = [2])
                state = self.state_pos_limits[0] + (self.state_pos_limits[1] - self.state_pos_limits[0]) * state
                ang = np.arctan2(np.array([-state[1]]), 
                                 np.array([-state[0]])).reshape([-1]) + np.random.normal() * 0.2
                if ang < 0.:
                    ang += 2. * np.pi
                elif ang > 2. * np.pi:
                    ang -= 2. * np.pi 
                state = np.concatenate([state, ang])
                if is_safe_state(state, self.interpolator):
                    break 
        else:
            state = initial_state.clone().detach().cpu()

        threshold = self.calibrated_values[uncertainty_type][str(quantile)]
        trajectory = {
            'image': [],
            'action': [],
            'privileged_state': [],
            'u_threshold': threshold,
            'uncertainty': [],
            'filtered': [],
        }
        if_failure = False
        self.reset() 
        for t in range(T):

            if np.linalg.norm(state[:2]) <= 0.4:
                if_failure = True
                break 
            elif np.abs(state[0]) >= 1.0 or np.abs(state[1]) >= 1.0:
                break
            
            obs = self.get_obs(state)
            self.unroll_latent_stoch(obs)

            ac = np.random.uniform(-self.u_max, self.u_max)
            u = self.current_uncertainty(ac, uncertainty_type)
            if_filtered = False
            if u > threshold:
                # ac_candidates = (np.random.uniform(low = 0.,
                #                                    high = 1.,
                #                                    size = [N_sample]) - 0.5) * 2 * self.u_max 
                ac_candidates = np.linspace(-self.u_max, self.u_max, N_sample)
                scores = []
                for ac_candidate in ac_candidates:
                    u_candidate = self.current_uncertainty(ac_candidate, uncertainty_type)
                    scores.append(u_candidate)
                ac = ac_candidates[np.argmin(scores)]
                u = np.min(scores)
                if_filtered = True


            ac_tensor = torch.tensor(ac).view([1, -1]).cuda().float()
            obs_tensor = torch.tensor(obs).cuda()
            state_tensor = torch.tensor(state).cuda().float()
            trajectory['image'].append(obs_tensor)
            trajectory['action'].append(ac_tensor)
            trajectory['privileged_state'].append(state_tensor)
            trajectory['uncertainty'].append(u)
            trajectory['filtered'].append(if_filtered)

            if terminate_on_ood and u > threshold and t <= 30:
                if_failure = True
                break 

            self.unroll_latent_deter(ac_tensor)
            state = self.unroll_dubins(state, ac)

        trajectory['if_failure'] = if_failure
        return trajectory

    def get_projection(self, embeds_ID, embeds_rdn, scores,
                  high_support_ratio = 0.3,
                  path = None, override = False,):

        ID_idx = embeds_ID.shape[0] 
        num_hs = int(ID_idx * high_support_ratio)
        hs_indices = np.argsort(scores[:ID_idx])[:num_hs]

        embeds_ID_hs = embeds_ID[hs_indices]
        if embeds_rdn is not None:
            embeds_rdn_hs = embeds_rdn[hs_indices]
            all_embeds = np.vstack([embeds_ID, embeds_rdn],)
            all_embeds_hs = np.vstack([embeds_ID_hs, embeds_rdn_hs],)
            target = all_embeds_hs
        else:
            all_embeds = embeds_ID
            all_embeds_hs = embeds_ID_hs
            target = all_embeds   

        if path is not None and (not os.path.exists(path)) or override:
                 
            from sklearn.decomposition import IncrementalPCA
            from sklearn.manifold import TSNE
            ipca = IncrementalPCA(n_components=50, batch_size=4096)
            for i in range(0, target.shape[0], 4096):
                ipca.partial_fit(target[i:i+4096])
            PCA50 = np.vstack([ipca.transform(target[i:i+4096]) for i in range(0, target.shape[0], 4096)])
            tsne = TSNE(
                n_components=2,
                perplexity=30,        # try 10, 30, 50
                learning_rate="auto", # good default
                init="pca",           # stable init
                max_iter=2000,
                metric="euclidean",
                random_state=0,
                verbose=1,
            )
            Y = tsne.fit_transform(PCA50)   # (81698, 2)
            
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            LD_features = np.memmap(path, mode="w+", dtype=np.float32, shape=tuple(Y.shape))
            LD_features[:] = Y.astype(np.float32, copy=False)
            LD_features.flush()

        elif path is not None:
            
            LD_features = np.memmap(
                path, dtype=np.float32, mode="r", shape=(target.shape[0], 2),
            )

        return LD_features, hs_indices


    def construct_comparison_embeddings(self, episodes, 
                                        limits = None,
                                        feat_list = ['stoch', 'deter'],
                                        B = 4096, k_nn = 20,
                                        path_dict = None,
                                        override = False,):

        import random
        episodes_keys = list(episodes.keys())
        # random.shuffle(episodes_keys)
        N = 0
        for k_ep in episodes:
            episode = episodes[k_ep]
            N += len(episode['image'])
            if limits is not None and N > limits:
                break
        D = 0 
        if 'stoch' in feat_list: D += 32 
        if 'deter' in feat_list: D += 512 
        print (N, D)

        if path_dict is not None and (not os.path.exists(path_dict['embed_path']) or override):
             
            os.makedirs(os.path.dirname(path_dict['embed_rdn_path']) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(path_dict['embed_path']) or ".", exist_ok=True)
            all_embeds_rdn = np.memmap(path_dict['embed_rdn_path'], mode="w+", dtype=np.float32, shape=(N, D))
            all_embeds = np.memmap(path_dict['embed_path'], mode="w+", dtype=np.float32, shape=(N, D))

            with torch.no_grad():
                from tqdm import tqdm
                size = 0
                for k_ep in tqdm(episodes_keys):
                    episode = episodes[k_ep]
                    trajectory = {}
                    for k, v in episode.items():
                        trajectory[k] = torch.tensor(np.array(v)).float().cuda()
                           
                    self.reset()
                    observations = trajectory['image']
                    actions = trajectory['action']
                    N_traj = len(observations)
                    all_embeds_ = []
                    all_embeds_rdn_ = []
                    for counter in range(N_traj):
                        obs = observations[counter]
                        ac = actions[counter]
                        self.unroll_latent_stoch(observations[counter]) # h_{t}, s_{t}
    
                        random_ac = torch.sign(actions[counter].view(1, -1))
                        random_ac *= (-self.u_max)
                        h_rdn = self.unroll_latent_deter(random_ac, override = False) # h_{t+1}, s_{t} 
                        embeds_rdn = torch.cat([h_rdn[feat_name] for feat_name in feat_list],
                                           dim = -1)
                        all_embeds_rdn_.append(embeds_rdn)
                          
                        self.unroll_latent_deter(actions[counter].view(1, -1)) # h_{t+1}, s_{t} 
                        embeds = torch.cat([self.h[feat_name] for feat_name in feat_list],
                                           dim = -1)
                        all_embeds_.append(embeds)
    
                    all_embeds_ = torch.cat(all_embeds_, dim = 0)
                    all_embeds_rdn_ = torch.cat(all_embeds_rdn_, dim = 0)
                    # all_embeds.append(all_embeds_)
                    # all_embeds_rdn.append(all_embeds_rdn_)

                    all_embeds[size:(size + N_traj)] = all_embeds_.detach().cpu().numpy()
                    all_embeds_rdn[size:(size + N_traj)] = all_embeds_rdn_.detach().cpu().numpy()
                    size += N_traj
                    if limits is not None and size > limits:
                          break

            all_embeds.flush()
            all_embeds_rdn.flush()

        all_embeds = np.memmap(
            path_dict['embed_path'], dtype=np.float32, mode="r", shape=(N, D),
        )
        all_embeds_rdn = np.memmap(
            path_dict['embed_rdn_path'], dtype=np.float32, mode="r", shape=(N, D),
        )

        index = faiss.IndexFlatL2(all_embeds.shape[1]) # l2
        for i in range(0, all_embeds.shape[0], B):
            embeds_ = all_embeds[i:(i+B)]
            index.add(embeds_)

        if path_dict is not None and (not os.path.exists(path_dict['score_path']) or override):

            os.makedirs(os.path.dirname(path_dict['score_path']) or ".", exist_ok=True)
            distances = np.memmap(path_dict['score_path'], mode="w+", dtype=np.float32, shape=(N))

            for i in range(0, all_embeds.shape[0], B):
                embeds_ = all_embeds[i:(i+B)]
                distances_, _ = index.search(embeds_, k_nn)
                distances_ = np.mean(distances_, axis = -1)
                distances[i:(i + distances_.shape[0])] = distances_
            distances.flush()
                    
                          
        distances = np.memmap(
            path_dict['score_path'], dtype=np.float32, mode="r", shape=(N),
        )
          
        return all_embeds, all_embeds_rdn, distances, episodes_keys
    
    def visualize(
        self,
        episodes,
        override: bool = False,
        sample_img: bool = False,
        include_rdn: bool = True,
        color_by_score: bool = False,   # optional: encode scores as a colormap
        show_hs: bool = True,           # highlight high-support indices (if provided)
        seed: int | None = 0,
        save_path: str | None = None,   # e.g. ".../projection.pdf"
        title: str | None = None,
        high_support_ratio = 0.3,
        emphasize_indices: np.ndarray | list[int] | None = None,   # global indices into LD_features to emphasize
        emphasize_kwargs: dict | None = None,                      # override style for emphasized points
        emphasize_labels: np.ndarray | list[str] | None = None,
        show_img = False,
        show_all_emphasize = False,
    ):
        # """
        # Paper-ready visualization of latent projections.
    
        # - If include_rdn=True: plots ID and RDN embeddings side-by-side on the same axes.
        # - If include_rdn=False: plots ID only and optionally highlights hs_indices.
        # - If sample_img=True: shows a randomly selected high-support sample as an inset.
        # - If save_path is provided: saves as PDF/PNG with publication-friendly defaults.
        # """
    
        _paper_rcparams()
    
        model_path = self.path
        parent_folder = os.path.dirname(model_path)
        path_dict = {
            "embed_path": os.path.join(parent_folder, "embeds_final.dat"),
            "embed_rdn_path": os.path.join(parent_folder, "embeds_rdn_final.dat"),
            "score_path": os.path.join(parent_folder, "scores_final.dat"),
        }

        print (path_dict)
        embeds_id, embeds_rdn, scores, eps_keys = self.construct_comparison_embeddings(
            episodes,
            limits=None,
            path_dict=path_dict,
            override=override,
        )
    
        # --- Projection
        if include_rdn:

            if high_support_ratio == 0.3:
                LD_path = os.path.join(parent_folder, "LD_final.dat")
            else:
                LD_path = os.path.join(parent_folder, "LD_final_{:.1e}.dat".format(high_support_ratio))
            print (LD_path)
            LD_features, hs_indices = self.get_projection(
                embeds_id, embeds_rdn, scores,
                path=LD_path,
                override=override,
                high_support_ratio=high_support_ratio,
            )
            n = LD_features.shape[0]
            half = n // 2
            xy_id = LD_features[:half]
            xy_rdn = LD_features[half:]
            scores_id = scores[:half] if scores is not None and len(scores) == n else None
            scores_rdn = scores[half:] if scores is not None and len(scores) == n else None
        else:
            LD_path = os.path.join(parent_folder, "LD_ID_only_final.dat")
            LD_features, hs_indices = self.get_projection(
                embeds_id, None, scores,
                path=LD_path,
                override=override,
                high_support_ratio=high_support_ratio,
            )
            xy_id = LD_features
            xy_rdn = None
            scores_id = scores if scores is not None and len(scores) == LD_features.shape[0] else None
            scores_rdn = None
    
        # --- Figure
        fig, ax = plt.subplots(figsize=(6., 6.))  # ~single-column width; adjust if needed
    
        # For clean vector output: rasterize scatter points but keep text/legend as vector.
        # Everything with zorder <= this will be rasterized in vector backends.
        ax.set_rasterization_zorder(0)
    
        s = 25.0           # point size
        a = 0.1          # alpha
        lw = 1.          # no edges (faster + cleaner)
        z_scatter = -1    # rasterized (<=0)
    
        # --- Plot ID (and optionally RDN)
        if color_by_score and scores_id is not None:
            sc_id = ax.scatter(
                xy_id[:, 0], xy_id[:, 1],
                c=scores_id,
                s=s, alpha=a, linewidths=lw,
                cmap="viridis",
                rasterized=True,
                zorder=z_scatter,
                label="ID",
            )
        else:
            sc_id = ax.scatter(
                xy_id[:, 0], xy_id[:, 1],
                s=s, alpha=a, linewidths=lw,
                rasterized=True,
                zorder=z_scatter,
                facecolors='none',
                edgecolors='#1f77b4',
                label="ID",
            )
    
        if include_rdn and xy_rdn is not None:
            if color_by_score and scores_rdn is not None:
                ax.scatter(
                    xy_rdn[:, 0], xy_rdn[:, 1],
                    c=scores_rdn,
                    s=s, alpha=a, linewidths=lw,
                    cmap="plasma",
                    rasterized=True,
                    zorder=z_scatter,
                    label="RDN",
                )
            else:
                ax.scatter(
                    xy_rdn[:, 0], xy_rdn[:, 1],
                    s=s, alpha=a, linewidths=lw,
                    rasterized=True,
                    zorder=z_scatter,
                    facecolors='none',
                    edgecolors='#ff7f0e',
                    label="RDN",
                )
    
        # --- Highlight high-support points (if available)
        if show_hs and hs_indices is not None and len(hs_indices) > 0:
            hs_indices = np.asarray(hs_indices, dtype=int)
            hs_indices = hs_indices[(hs_indices >= 0) & (hs_indices < LD_features.shape[0])]
            ax.scatter(
                LD_features[hs_indices, 0], LD_features[hs_indices, 1],
                s=6.0, alpha=0.9,
                linewidths=0.0,
                rasterized=True,
                zorder=z_scatter,
                label="High-support",
            )

        # print (hs_indices.shape, LD_features.shape, xy_id.shape)
    
        # --- Sample image inset (uses a *high-support* idx if possible)
        sampled_idx = None
        if show_img and (sample_img or len(emphasize_indices) != 0):
            # print (len(hs_indices), LD_features.shape)
            if not sample_img:
                local_indices = np.where(hs_indices == emphasize_indices[0])[0]
                if len(local_indices) <= 0:
                    print ("Not IN")
                    assert False
                sampled_idx = local_indices[0]

            else:
                rng = np.random.default_rng(None)
                sampled_idx = int(rng.integers(0, hs_indices.shape[0]))
    
            # Correct: scatter the actual sampled point
            ax.scatter(
                LD_features[sampled_idx, 0], LD_features[sampled_idx, 1],
                s=60.0, alpha=0.,
                linewidths=0.0,
                rasterized=True,
                zorder=z_scatter,
                label="Sampled",
            )

            print ("local: {}, global: {}".format(sampled_idx, hs_indices[sampled_idx]))
            img = get_img_from_idx(episodes, eps_keys, hs_indices[sampled_idx])                           
            axins = inset_axes(ax, width="35%", height="35%", loc="upper right", borderpad=0.5)
            axins.imshow(img)
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_linewidth(0.6)

            if sample_img:
                if emphasize_labels is None:
                    emphasize_labels = []
                    emphasize_indices = []
                emphasize_indices.append(hs_indices[sampled_idx])
                emphasize_labels.append("S")
                

        if emphasize_indices is not None and len(emphasize_indices) > 0:
            assert emphasize_labels is not None and len(emphasize_labels) == len(emphasize_indices)

            emphasize_kwargs = {} if emphasize_kwargs is None else dict(emphasize_kwargs)
            default_emph = dict(s=180.0, linewidths=2.5, facecolors="none",
                            edgecolors="black", zorder=6, rasterized=False)
            default_emph.update(emphasize_kwargs)
                        
            emphasize_indices = np.asarray(emphasize_indices, dtype=int)
            # emphasize_indices = emphasize_indices[(emphasize_indices >= 0) & (emphasize_indices < LD_features.shape[0])]

            local_emphasize_indices = []
            for gidx in emphasize_indices:
                local_indices = np.where(hs_indices == gidx)[0]
                if len(local_indices) <= 0:
                    print ("Not IN")
                    assert False
                local_emphasize_indices.append(local_indices[0])      

            local_emphasize_indices = np.array(local_emphasize_indices)
            ax.scatter(LD_features[local_emphasize_indices, 0], LD_features[local_emphasize_indices, 1], **default_emph)

            # spread annotations radially to avoid overlap for nearby points
            base_offset = 8
            angles = np.linspace(0, 2*np.pi, len(emphasize_indices[:20]), endpoint=False)
            
            for j, gi in enumerate(local_emphasize_indices[:20]):
                dx = base_offset * np.cos(angles[j])
                dy = base_offset * np.sin(angles[j])
            
                ax.annotate(
                    emphasize_labels[j],
                    (LD_features[gi, 0], LD_features[gi, 1]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=30,
                    fontweight="bold",
                    color="black",
                    zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=0.9,
                        color="black",
                        shrinkA=6,   # keep line from touching the label box
                        shrinkB=8,   # keep line off the emphasized marker
                    ),
                )

            # also emphasize the paired (shifted) points: emphasize_indices + xy_id.shape[0]
            pair_offset = xy_id.shape[0]
            emphasize_pairs = local_emphasize_indices + pair_offset
            emphasize_pairs = emphasize_pairs[emphasize_pairs < LD_features.shape[0]]

            pair_color = "#d62728"   # colorblind-safe, distinct from black

            ax.scatter(
                LD_features[emphasize_pairs, 0],
                LD_features[emphasize_pairs, 1],
                s=180.0, linewidths=2.5, facecolors="none",
                edgecolors=pair_color, zorder=5, rasterized=False,
            )

            # draw straight correspondence lines between each emphasized pair
            for gi, gj in zip(local_emphasize_indices, emphasize_pairs):
                ax.plot(
                    [LD_features[gi, 0], LD_features[gj, 0]],
                    [LD_features[gi, 1], LD_features[gj, 1]],
                    color=pair_color,
                    lw=4.,
                    alpha=0.9,
                    zorder=4,
                )

            # for j, gi in enumerate(emphasize_pairs[:20]):
            #     dx = base_offset * np.cos(angles[j])
            #     dy = base_offset * np.sin(angles[j])
            
            #     ax.annotate(
            #         emphasize_labels[j],
            #         (LD_features[gi, 0], LD_features[gi, 1]),
            #         xytext=(dx, dy),
            #         textcoords="offset points",
            #         ha="left" if dx >= 0 else "right",
            #         va="bottom" if dy >= 0 else "top",
            #         fontsize=15,
            #         fontweight="bold",
            #         color=pair_color,
            #         zorder=6,
            #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
            #         arrowprops=dict(
            #             arrowstyle="-",
            #             lw=0.9,
            #             color="black",
            #             shrinkA=6,   # keep line from touching the label box
            #             shrinkB=8,   # keep line off the emphasized marker
            #         ),
            #     )

    
        # --- Aesthetics
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
        if title is not None:
            ax.set_title(title,
                         pad=6               # vertical spacing from axes
            )
    
        # Legend (keep minimal + unobtrusive)
        handles, labels = ax.get_legend_handles_labels()
        # if len(labels) > 0:
        #     ax.legend(
        #         loc="lower left",
        #         frameon=False,
        #         handletextpad=0.3,
        #         borderaxespad=0.2,
        #         markerscale=2.0,
        #     )
    
        # Optional colorbar if we color by score
        if color_by_score and scores_id is not None:
            cbar = fig.colorbar(sc_id, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Score")

        ax.axis("on")          # re-enable axes
        for spine in ax.spines.values():
            spine.set_visible(True)
    
        fig.tight_layout(pad=0.02)
    
        if save_path is not None:
            # PDF for papers; use PNG for quick viewing if desired.
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
        plt.show()

        if show_all_emphasize and emphasize_indices is not None and len(emphasize_indices) > 0:
            n_show = len(emphasize_indices)
            ncols = min(5, n_show)
            nrows = int(np.ceil(n_show / ncols))

            fig2, axs = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.2*nrows))
            axs = np.atleast_1d(axs).ravel()

            for k, gi in enumerate(emphasize_indices):
                img = get_img_from_idx(episodes, eps_keys, gi)
                axs[k].imshow(img)
                # axs[k].set_title(emphasize_labels[k], fontsize=18, fontweight="bold", pad=8)
                # place label INSIDE the subfigure (top-left corner)
                axs[k].text(
                    0.02, 0.98, emphasize_labels[k],
                    transform=axs[k].transAxes,
                    ha="left", va="top",
                    fontsize=40, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", edgecolor="black",
                              linewidth=1.5, pad=0.3, alpha=0.95),
                    zorder=10,
                )
                axs[k].set_xticks([]); axs[k].set_yticks([])
                for sp in axs[k].spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(2.0)
                 

            for ax_unused in axs[n_show:]:
                ax_unused.axis("off")

            fig2.tight_layout(pad=0.8)
            plt.show()

        # if show_all_emphasize and emphasize_indices is not None and len(emphasize_indices) > 0:
        #     n_show = len(emphasize_indices)
        #     ncols = min(5, n_show)
        #     nrows = int(np.ceil(n_show / ncols))

        #     # --- vertical, paper-ready layout (single column)
        #     ncols = 1
        #     nrows = n_show
        #     fig2, axs = plt.subplots(nrows, 1, figsize=(4.5, 3.8 * nrows))
        #     axs = np.atleast_1d(axs).ravel()
            
        #     fig2.patch.set_facecolor("white")
            
        #     for k, gi in enumerate(emphasize_indices):
        #         img = get_img_from_idx(episodes, eps_keys, gi)
        #         axs[k].imshow(img)
            
        #         # label INSIDE each subfigure (top-left)
        #         axs[k].text(
        #             0.02, 0.98, emphasize_labels[k],
        #             transform=axs[k].transAxes,
        #             ha="left", va="top",
        #             fontsize=28, fontweight="bold", color="black",
        #             bbox=dict(facecolor="white", edgecolor="black",
        #                       linewidth=1.5, pad=0.3, alpha=0.95),
        #             zorder=10,
        #         )
            
        #         axs[k].set_xticks([])
        #         axs[k].set_yticks([])
        #         for sp in axs[k].spines.values():
        #             sp.set_visible(True)
        #             sp.set_linewidth(2.0)
            
        #     fig2.subplots_adjust(hspace=0., top=0.98, bottom=0.02)
        #     plt.show()
             
        return sampled_idx
                
    def visualize_draft(self, episodes,
                  override = False,
                  sample_img = False,
                  high_support_ratio = 0.3,
                  include_rdn = True):

        model_path = self.path
        parent_folder = os.path.dirname(model_path)
        path_dict = {
            'embed_path': os.path.join(parent_folder, "embeds_final.dat"),
            'embed_rdn_path': os.path.join(parent_folder, "embeds_rdn_final.dat"),
            'score_path': os.path.join(parent_folder, "scores_final.dat"),
        }
        embeds_id, embeds_rdn, scores, eps_keys = self.construct_comparison_embeddings(episodes,
            limits = None, path_dict = path_dict, override = override
        )
        if include_rdn:
            if high_support_ratio == 0.3:
                LD_path = os.path.join(parent_folder, "LD_final.dat")
            else:
                LD_path = os.path.join(parent_folder, "LD_final_{:.1e}.dat".format(high_support_ratio))
            print (LD_path)
            LD_features, hs_indices = self.get_projection(embeds_id, embeds_rdn, scores,
               path = LD_path, override = override, high_support_ratio = high_support_ratio,
            )
            print (LD_features.shape, embeds_id.shape)

            # dim = hs_indices.shape[0] 
            # num_hs = int(dim * 0.5)
            # hs_indices2 = np.argsort(scores[hs_indices])[:num_hs]
                     
            half_idx = int(LD_features.shape[0]/2)
            plt.figure(figsize=(8, 8))
            plt.scatter(LD_features[:half_idx, 0], 
                        LD_features[:half_idx, 1], s=1, alpha=0.35, rasterized=True)
            plt.scatter(LD_features[half_idx:, 0], 
                        LD_features[half_idx:, 1], s=1, alpha=0.35, rasterized=True)
            # plt.scatter(LD_features[:half_idx, 0][hs_indices2], 
            #             LD_features[:half_idx, 1][hs_indices2], s=1, alpha=0.35, rasterized=True)
            # plt.scatter(LD_features[half_idx:, 0][hs_indices2], 
            #             LD_features[half_idx:, 1][hs_indices2], s=1, alpha=0.35, rasterized=True)
            
        else:
            LD_path = os.path.join(parent_folder, "LD_ID_only_final.dat")
            LD_features, hs_indices = self.get_projection(embeds_id, None, scores,
               path = LD_path, override = override, high_support_ratio = high_support_ratio,
            )     
            print (LD_features.shape, embeds_id.shape, scores.shape)
            plt.scatter(LD_features[:, 0], 
                        LD_features[:, 1], s=1, alpha=0.35, rasterized=True)
            plt.scatter(LD_features[hs_indices][:, 0], 
                        LD_features[hs_indices][:, 1], 
                        s=1, alpha=0.35, rasterized=True, color = 'orange')
  
        
        if sample_img:
            hs_i = np.random.randint(hs_indices.shape[0])
            idx = hs_indices[hs_i]
            plt.scatter(LD_features[hs_i, 0], 
                        LD_features[hs_i, 1], s=100, 
                        alpha=0.75, rasterized=True, color = 'red')
        
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        if sample_img:
            img = get_img_from_idx(episodes, eps_keys, idx)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

    def get_OOD_actions(self, actions):

        actions_max = torch.ones_like(actions).float().to(actions.device) * self._config.u_max
        actions_min = torch.ones_like(actions).float().to(actions.device) * (-self._config.u_max)
        diff_max = (actions_max - actions).norm(dim = -1)
        diff_min = (actions_min - actions).norm(dim = -1)
        actions_max[diff_max > diff_min] = actions_min[diff_max > diff_min]        
        return actions_max

    def extract_uncertainties_to_memmap(self, episodes,
                                        uncertainty_path,
                                        embed_path,
                                        OOD_interpolation_list,
                                        bin_width, mask):

        N = 136166
        embeds = np.memmap(
            embed_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 512 + 32),
        )
        os.makedirs(os.path.dirname(uncertainty_path) or ".", exist_ok=True)
        uncertainties = np.memmap(uncertainty_path, mode="w+", dtype=np.float32, 
                                shape=(N, len(OOD_interpolation_list), 3))

        xy_min, xy_max = -1., 1.    
        nxy = int(np.ceil((xy_max - xy_min) / bin_width))

        from tqdm import tqdm
        i = 0
        for k_ep in tqdm(episodes):
            episode = episodes[k_ep]
            states = np.array(episode["privileged_state"])
            ix = np.floor((states[:, 0] - xy_min) / bin_width).astype(int)
            iy = np.floor((states[:, 1] - xy_min) / bin_width).astype(int)
            valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy) & (mask[iy, ix])
            # print (mask[ix, iy].shape, valid.sum(), valid.shape)
            if valid.sum() == 0:
                continue
              
            actions = torch.tensor(np.array(episode['action'])).float().cuda()
            actions = actions
            actions_OOD = self.get_OOD_actions(actions)
            L = actions.shape[0]
            embeds_traj = torch.tensor(embeds[i:(i+L)]).float().cuda()
            stoch, deter = embeds_traj[:, :32], embeds_traj[:, 32:]

            for t_counter, threshold in enumerate(OOD_interpolation_list):
                actions_ = actions + (actions_OOD - actions) * threshold

                uncertainty_nn = self.current_uncertainty_from_feat(embeds_traj[valid],
                                                                    actions_[valid],
                                                                    'nn')
                uncertainty_fm = self.current_uncertainty_from_feat(embeds_traj[valid],
                                                                    actions_[valid],
                                                                    'fm')
                uncertainty_ensemble = self.current_uncertainty_from_feat(embeds_traj[valid],
                                                                    actions_[valid],
                                                                    'ensemble').view([-1])
                # if t_counter == 0:
                #     print (uncertainty_nn)
                uncertainties[i:(i+L), t_counter, 0][valid] = uncertainty_nn
                uncertainties[i:(i+L), t_counter, 1][valid] = uncertainty_fm.detach().cpu().numpy()
                uncertainties[i:(i+L), t_counter, 2][valid] = uncertainty_ensemble.detach().cpu().numpy()
                                                                                                 
            i += L
        uncertainties.flush()

    def extract_transition_to_memmap(self, episodes, 
                                     prediction_path, 
                                     # uncertainty_path,
                                     embed_path,
                                     OOD_interpolation_list): 

        N = 136166
        embeds = np.memmap(
            embed_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 512 + 32),
        )

        os.makedirs(os.path.dirname(prediction_path) or ".", exist_ok=True)
        predictions = np.memmap(prediction_path, mode="w+", dtype=np.float32, 
                                shape=(N, len(OOD_interpolation_list), 512 + 32))
        # os.makedirs(os.path.dirname(uncertainty_path) or ".", exist_ok=True)
        # uncertainties = np.memmap(uncertainty_path, mode="w+", dtype=np.float32, 
        #                         shape=(N, len(OOD_interpolation_list), 3))
        from tqdm import tqdm
        i = 0
        for k_ep in tqdm(episodes):
            episode = episodes[k_ep]
            actions = torch.tensor(np.array(episode['action'])).float().cuda()
            actions = actions
            actions_OOD = self.get_OOD_actions(actions)
            L = actions.shape[0]
            embeds_traj = torch.tensor(embeds[i:(i+L)]).float().cuda()
            stoch, deter = embeds_traj[:, :32], embeds_traj[:, 32:]

            for t_counter, threshold in enumerate(OOD_interpolation_list):
                actions_ = actions + (actions_OOD - actions) * threshold
                # actions_ = actions_.detach().clone().requires_grad_(True)
                with torch.inference_mode():
                    pred_feat = self.unroll_from_feat(stoch, deter, actions_)
                predictions[i:(i+L), t_counter] = pred_feat.detach().cpu().numpy()

                # uncertainty_nn = self.current_uncertainty_from_feat(embeds_traj,
                #                                                     actions_,
                #                                                     'nn')
                # uncertainty_fm = self.current_uncertainty_from_feat(embeds_traj,
                #                                                     actions_,
                #                                                     'fm')
                # uncertainty_ensemble = self.current_uncertainty_from_feat(embeds_traj,
                #                                                     actions_,
                #                                                     'ensemble').view([-1])
                # uncertainties[i:(i+L), t_counter, 0] = uncertainty_nn
                # uncertainties[i:(i+L), t_counter, 1] = uncertainty_fm.detach().cpu().numpy()
                # uncertainties[i:(i+L), t_counter, 2] = uncertainty_ensemble.detach().cpu().numpy()
                                                                                                 
            i += L
        predictions.flush()
        # uncertainties.flush()

    def analyze_discretized_system_uncertainty(self, episodes,
                                   uncertainty_path,
                                   bin_width, 
                                   OOD_interpolation_list, OOD_threshold_idx,
                                   uncertainty_type, mask):

        if uncertainty_type == 'nn':
            uncertainty_idx = 0
        elif uncertainty_type == 'fm':
            uncertainty_idx = 1
        elif uncertainty_type == 'ensemble':
            uncertainty_idx = 2

        # assert OOD_threshold_idx > 0

        N = 136166
        uncertainties = np.memmap(uncertainty_path, mode="r", 
                                dtype=np.float32, 
                                shape=(N, len(OOD_interpolation_list), 3),
                               )
                             
        xy_min, xy_max = -1., 1.    
        nxy = int(np.ceil((xy_max - xy_min) / bin_width))
        counts = np.zeros((nxy, nxy), dtype=int)
        sums = np.zeros((nxy, nxy), dtype=float)
                             
        # from tqdm import tqdm
        i = 0
        # for k_ep in tqdm(episodes):
        for k_ep in episodes:
            episode = episodes[k_ep]
            states = np.array(episode["privileged_state"])
                  
            L = states.shape[0] 
            if OOD_threshold_idx > 0:
                ID_uncertainties = uncertainties[i:(i+L), 0, uncertainty_idx]
                OOD_uncertainties = uncertainties[i:(i+L), OOD_threshold_idx, uncertainty_idx]
                scores = OOD_uncertainties - ID_uncertainties
            else:
                scores = uncertainties[i:(i+L), 0, uncertainty_idx]

            ix = np.floor((states[:, 0] - xy_min) / bin_width).astype(int)
            iy = np.floor((states[:, 1] - xy_min) / bin_width).astype(int)
            valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy) & (mask[iy, ix]) & (scores != 0)
            ix, iy, scores = ix[valid], iy[valid], scores[valid]
            # print (scores)

            np.add.at(counts, (iy, ix), 1)
            np.add.at(sums, (iy, ix), scores)
            i += L 

        means = sums / np.maximum(counts, 1)
        return means

    def analyze_discretized_system_prediction(self, episodes,
                                   prediction_path,
                                   bin_width, 
                                   OOD_interpolation_list, OOD_threshold_idx):

        assert OOD_threshold_idx > 0

        N = 136166
        predictions = np.memmap(prediction_path, mode="r", 
                                dtype=np.float32, 
                                shape=(N, len(OOD_interpolation_list), 512 + 32),
                               )
                             
        xy_min, xy_max = -1., 1.    
        nxy = int(np.ceil((xy_max - xy_min) / bin_width))
        counts = np.zeros((nxy, nxy), dtype=int)
        sums = np.zeros((nxy, nxy), dtype=float)
                             
        from tqdm import tqdm
        i = 0
        for k_ep in tqdm(episodes):
            episode = episodes[k_ep]
            states = np.array(episode["privileged_state"])
                  
            L = states.shape[0] 
            ID_predictions = predictions[i:(i+L), 0]
            OOD_predictions = predictions[i:(i+L), OOD_threshold_idx]
            diff = np.linalg.norm(ID_predictions - OOD_predictions, axis = -1)
                  
            ix = np.floor((states[:, 0] - xy_min) / bin_width).astype(int)
            iy = np.floor((states[:, 1] - xy_min) / bin_width).astype(int)
            valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy)
            ix, iy, scores = ix[valid], iy[valid], diff[valid]

            np.add.at(counts, (iy, ix), 1)
            np.add.at(sums, (iy, ix), scores)
            i += L 

        means = sums / np.maximum(counts, 1)
        return means
                 

    def analyze_discretized_system(self, episodes,
                                   jacobian_path,
                                   bin_width,
                                   theta = None):

        N = 136166
        jacobian = np.memmap(jacobian_path, mode="r", dtype=np.float32, shape=(N))

        xy_min, xy_max = -1., 1.    
        nxy = int(np.ceil((xy_max - xy_min) / bin_width))
        counts = np.zeros((nxy, nxy), dtype=int)
        sums = np.zeros((nxy, nxy), dtype=float)

        sums_ac   = np.zeros((nxy, nxy), dtype=np.float64)
        sumsq_ac  = np.zeros((nxy, nxy), dtype=np.float64)

        from tqdm import tqdm
        i = 0
        for k_ep in tqdm(episodes):
            episode = episodes[k_ep]
            states = np.array(episode["privileged_state"])
            actions = np.array(episode["action"]).reshape([-1])

            L = states.shape[0] 
            J = jacobian[i:(i+L)]

            ix = np.floor((states[:, 0] - xy_min) / bin_width).astype(int)
            iy = np.floor((states[:, 1] - xy_min) / bin_width).astype(int)
            valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy)
            if theta is not None:
                valid = valid & (np.abs(states[:, 2] - theta) < 2e-1)
            ix, iy, scores, acs = ix[valid], iy[valid], J[valid], actions[valid]

            np.add.at(counts, (iy, ix), 1)
            np.add.at(sums, (iy, ix), scores)

            np.add.at(sums_ac,   (iy, ix), acs)
            np.add.at(sumsq_ac,  (iy, ix), acs ** 2)

            i += L 

        means = sums / np.maximum(counts, 1)

        means_ac = sums_ac / np.maximum(counts, 1)
        vars_ac = sumsq_ac / np.maximum(counts, 1) - means_ac ** 2
        vars_ac = np.maximum(vars_ac, 0.0)
              
        return means, counts, vars_ac

    def extract_jacobian_to_memmap(self, episodes, path, embed_path,):

        def unroll(stoch, deter, a):
            x = torch.cat([stoch, a], dim = -1)
            x = wm.dynamics._img_in_layers(x)
            x, deter = wm.dynamics._cell(x, [deter])
            deter = deter[0]
            x = wm.dynamics._img_out_layers(x)
            stats = wm.dynamics._suff_stats_layer("ims", x)
            stoch = wm.dynamics.get_dist(stats).mode()
            return torch.cat([stoch, deter], dim = -1)      

        N = 136166
        embeds = np.memmap(
            embed_path,
            dtype=np.float32,
            mode="r",
            shape=(N, 512 + 32),
        )

        wm = self.agent._wm

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        jacobian = np.memmap(path, mode="w+", dtype=np.float32, shape=(N))
        from tqdm import tqdm
        i = 0
        for k_ep in tqdm(episodes):
            episode = episodes[k_ep]
            actions = torch.tensor(np.array(episode['action'])).float().cuda()
            actions = actions.detach().clone().requires_grad_(True)
            L = actions.shape[0]
            embeds_traj = torch.tensor(embeds[i:(i+L)]).float().cuda()
            stoch, deter = embeds_traj[:, :32], embeds_traj[:, 32:]

            pred_feat = unroll(stoch, deter, actions)
            grads = [] 
            for j in range(pred_feat.shape[1]):
                g = torch.autograd.grad(
                    outputs=pred_feat[:, j].sum(),   # scalar
                    inputs=actions,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )[0]                              # [B]
                grads.append(g)
            grads = torch.cat(grads, dim = -1)
            # print (grads.shape, pred_feat.shape)
            grad_norms = grads.view(grads.shape[0], -1).norm(dim = -1)
            jacobian[i:(i+L)] = grad_norms.detach().cpu().numpy()
            i += L
        jacobian.flush()

    def extract_embeddings_to_memmap(self, episodes, path,
                                     feat_list = ['stoch', 'deter'],):

        N = 0 
        for k_ep in episodes:
            episode = episodes[k_ep]
            N += len(episode['image'])
        D = 0 
        if 'stoch' in feat_list: D += 32 
        if 'deter' in feat_list: D += 512 
        print (N, D)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        E = np.memmap(path, mode="w+", dtype=np.float32, shape=(N, D))

        with torch.no_grad():
            
            idx = 0
            from tqdm import tqdm
            for k_ep in tqdm(episodes):
                
                episode = episodes[k_ep]
                trajectory = {}
                for k, v in episode.items():
                    trajectory[k] = torch.tensor(np.array(v)).float().cuda()

                self.reset()
                observations = trajectory['image']
                actions = trajectory['action']
                N = len(observations)
                all_embeds = []
                for counter in range(N):
                    obs = observations[counter]
                    ac = actions[counter]
                    self.unroll_latent_stoch(observations[counter]) # h_{t}, s_{t}
                    embeds = torch.cat([self.h[feat_name] for feat_name in feat_list],
                                       dim = -1)
                    all_embeds.append(embeds)
                    self.unroll_latent_deter(actions[counter].view(1, -1)) # h_{t+1}, s_{t} 

                all_embeds = torch.cat(all_embeds, dim = 0)
                E[idx:(idx+N)] = all_embeds.detach().cpu().numpy()
                idx += N

            E.flush()    
        return D

    def build_nearest_neighbor_from_memmap(self, path,
                                           B = 4096,
                                           D = 32,):

        if self.dataset_type == "large":
            N = 136166
        elif self.dataset_type == "small":
            N = 44597 # for small dataset
        embeds = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(N, D),
        )

        # index = faiss.IndexFlatIP(D) # cosine
        index = faiss.IndexFlatL2(D) # l2
        for i in range(0, N, B):
            embeds_ = embeds[i:(i+B)]
            index.add(embeds_)
        self.index = index 

    def derive_nearest_neighbor_scores(self, path, k,
                                       save_path,
                                       B = 2048, D = 32,):

        if self.dataset_type == "large":
            N = 136166
        elif self.dataset_type == "small":
            N = 44597 # for small dataset
        embeds = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(N, D),
        )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        distances = np.memmap(save_path, mode="w+", dtype=np.float32, 
                              shape=(N))

        for i in tqdm(range(0, N, B)):
            embeds_ = embeds[i:(i+B)] 
            distances_, _ = self.index.search(embeds_, k)
            distances_ = np.mean(distances_, axis = -1)
            distances[i:(i+B)] = distances_.reshape([-1])
        distances.flush()
        return distances

    def perform_nearest_neighbor(self, episodes,
                                 k_list = [],
                                 feat_list = ['stoch', 'deter'],
                                 override = False,
                                 index_only = False):

        if len(k_list) == 0:
            k_list.append(self._config.nn_k)
        # print (k)

        embed_path = os.path.join(self.parent_folder,
                                  "embeds_[{}]_analysis.dat".format('_'.join(feat_list)))        
        if (not os.path.exists(embed_path)) or override:
            D = self.extract_embeddings_to_memmap(episodes, 
                                              embed_path,
                                              feat_list = feat_list,
                                             )
        else:
            D = 0 
            if 'stoch' in feat_list: D += 32 
            if 'deter' in feat_list: D += 512 
        self.build_nearest_neighbor_from_memmap(embed_path, D = D)
        self.feat_list = feat_list
        
        if not index_only:
            self.nn_scores_all = []
            self.nn_scores_sorted_all = []
            for k in k_list:
                score_path = os.path.join(self.parent_folder,
                              "scores_[{}]_k{}_analysis.dat".format('_'.join(feat_list),
                                                                    k))
                if (not os.path.exists(score_path)) or override:
                    self.derive_nearest_neighbor_scores(embed_path,
                                                        k = k, 
                                                        save_path = score_path,
                                                        D = D)
                if self.dataset_type == "large":
                    N = 136166
                elif self.dataset_type == "small":
                    N = 44597 # for small dataset
                nn_scores = np.memmap(
                    score_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(N,),
                )
                nn_scores_sorted = np.sort(nn_scores)
                self.nn_scores_all.append(nn_scores)
                self.nn_scores_sorted_all.append(nn_scores_sorted)
            self.k_list = k_list

    def eval_prediction_performance(self, episodes, path,
                                    N_eval = 500,
                                    ):

        from tqdm.auto import trange
        import random 

        if N_eval > len(episodes):
            N_eval = len(episodes)
        pbar = trange(0, N_eval, desc="Eval", leave=True, dynamic_ncols=True)
        episodes_keys = list(episodes.keys())
        random.shuffle(episodes_keys)
        stats = []
        for k_ep_counter in pbar: 

            k_ep = episodes_keys[k_ep_counter]
            episode = episodes[k_ep]
            trajectory = {}
            for k, v in episode.items():
                trajectory[k] = torch.tensor(np.array(v)).float().cuda()

            self.reset()
            observations = trajectory['image']
            actions = trajectory['action']
            N = len(observations)
            self.h_pred = None
            for counter in range(N):
                obs = observations[counter]
                ac = actions[counter].view(1, -1)
                self.unroll_latent_stoch(observations[counter]) # h_{t}, s_{t}
                if self.h_pred is not None:
                    z_diff, recon_diff, norm = self.eval_prediction_eps(observations[counter],
                                             ac)
                    if len(stats) == 0:
                        stats = np.array([[z_diff, recon_diff, norm]])
                    else:
                        stats = np.vstack([stats,
                                           np.array([[z_diff, recon_diff, norm]])])
                self.unroll_latent_deter(ac) # h_{t+1}, s_{t} 
                self.h_pred = {k: v.clone() for k, v in self.h.items()}

            stats_mean = np.mean(stats, axis = 0)
            assert len(stats_mean) == 3
            pbar.set_postfix(
                z=f"{stats_mean[0]:.3f}",
                r=f"{stats_mean[1]:.6f}",
                g=f"{stats_mean[2]:.1f}"
            )
        stats_mean = np.mean(stats, axis = 0)
        return {
            "latent_error": stats_mean[0].item(),
            "recon_error": stats_mean[1].item(),
            "grad_norm": stats_mean[2].item(),
        }

    def eval_prediction_eps(self, obs, ac):

        feat_h_pred = torch.cat([
            self.h_pred['stoch'], self.h_pred['deter']
        ], dim = -1)
        z_diff = ((self.h['stoch'].view([-1]) - self.h_pred['stoch'].view([-1]))).abs().mean().item()

        recon_obs = self.agent._wm.heads["decoder"](feat_h_pred.view(1, 1, -1))['image'].mean()
        recon_obs = recon_obs.view(obs.shape)
        recon_diff = (recon_obs - obs/255.).abs().mean().item()

        norm = self.agent.compute_grad_norm(self.h, ac,
                                            use_gt_jacobian = True).item()
        return z_diff, recon_diff, norm
            