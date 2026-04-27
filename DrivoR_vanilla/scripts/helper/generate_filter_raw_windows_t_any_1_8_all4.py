#!/usr/bin/env python3
"""Generate a raw-window latent filter requiring t and any t+1..t+8 all-4 DrivoR cameras."""

from latent_scene_filter_generation_utils import POOL_RAW_WINDOWS, RULE_ANY_1_8, main


if __name__ == "__main__":
    main(
        description=__doc__,
        default_output_name="navtrain_raw_latent_t_any_1_8_all4.yaml",
        candidate_pool=POOL_RAW_WINDOWS,
        future_rule=RULE_ANY_1_8,
    )
