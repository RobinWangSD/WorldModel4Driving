#!/usr/bin/env python3
"""Generate a navtrain-token latent filter requiring t and t+1 all-4 DrivoR cameras."""

from latent_scene_filter_generation_utils import POOL_SCENE_FILTER_TOKENS, RULE_TPLUS1, main


if __name__ == "__main__":
    main(
        description=__doc__,
        default_output_name="navtrain_tokens_latent_t_tplus1_all4.yaml",
        candidate_pool=POOL_SCENE_FILTER_TOKENS,
        future_rule=RULE_TPLUS1,
    )
