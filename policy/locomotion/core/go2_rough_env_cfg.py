"""Go2 rough-terrain locomotion (terrain curriculum, per handoff step 3) with the
Hwangbo-curriculum push disturbance added on top of Isaac Lab's stock
UnitreeGo2RoughEnvCfg. Coexists with the rough task's own built-in adaptive
terrain-level curriculum (self.curriculum.terrain_levels) -- two independent
curriculum terms, not a conflict. Rewards, PPO hyperparameters, and network
architecture are untouched -- see REFERENCES.md.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from .push_disturbance_cfg import add_push_disturbance


@configclass
class UnitreeGo2RoughCerberusEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(self)

        # Stock UnitreeGo2RoughEnvCfg (super().__post_init__() above) already rescales
        # "boxes"/"random_rough" for Go2's smaller body size, but leaves "pyramid_stairs"/
        # "pyramid_stairs_inv" at their stock (ANYmal-scale) step_height_range -- an
        # absolute-meters parameter, unlike hf_pyramid_slope's scale-invariant slope_range
        # ratio. Found via a real per-sub-terrain-type push-recovery eval (2026-09-04,
        # see RUN_001.md): pyramid_stairs_inv scored just 12% success (vs 91-97% for both
        # slope types and random_rough), pyramid_stairs 49% -- both far below boxes' own
        # already-rescaled 63.5% -- and pyramid_stairs_inv was already bad at the easiest
        # level tested, not a difficulty-driven decline like the other types, the
        # signature of an unscaled absolute-length parameter rather than a genuine skill
        # ceiling. Rescaled by the same 0.5x factor stock's own boxes rescale used
        # ((0.05, 0.2) -> (0.025, 0.1)) -- a reasoned estimate following that established
        # pattern, not independently verified. See REFERENCES.md.
        stairs_scale = 0.5
        stock_step_height_range = (0.05, 0.23)
        scaled_step_height_range = tuple(v * stairs_scale for v in stock_step_height_range)
        for stairs_type in ("pyramid_stairs", "pyramid_stairs_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[stairs_type].step_height_range = (
                scaled_step_height_range
            )


@configclass
class UnitreeGo2RoughCerberusEnvCfg_PLAY(UnitreeGo2RoughCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        # Deliberately NOT shrinking terrain_generator to a smaller grid here --
        # upstream's own Play config does (5x5, curriculum=False) purely to save
        # memory at training's num_envs=4096, but that tradeoff doesn't apply to
        # play/eval's much smaller env counts. Keeping the full ROUGH_TERRAINS_CFG
        # (10 rows x 20 cols, curriculum=True) means: (a) play.py's video and
        # push_recovery_eval.py's --terrain_levels axis see the exact same
        # structured, difficulty-scaled-by-row terrain the policy actually
        # trained against, not a separately-scaled-down grid, and (b) all 6
        # sub-terrain types get proportional column coverage -- a reduced 5-col
        # grid was found to structurally drop one type entirely (see
        # core/terrain_pinning.py's column-assignment math). See REFERENCES.md.

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing events -- push recovery is evaluated separately
        # (policy/locomotion/eval/), not exercised during a play/visualization run
        self.events.base_external_force_torque = None
        self.events.push_disturbance = None
        self.curriculum.push_disturbance_curriculum = None

        # Follow-cam: the default viewer is a fixed world-space camera that
        # never tracks the robot, so footage drifts out of frame once a push
        # (or just plain walking) moves the robot away from its spawn point --
        # and, on Rough, this fixed point sits over one terrain cell regardless
        # of which env/condition is nominally being recorded, which is why
        # showcase clips for different terrain levels/sub-terrain types all
        # looked identical (see REFERENCES.md's camera-framing writeup).
        # asset_root origin makes Isaac Lab recompute eye/lookat from the
        # robot's current root pose every step instead of once. Needs a
        # pod-side smoke test to confirm (not verifiable from source alone --
        # Isaac Lab isn't installed here).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.5, -2.5, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.3)
