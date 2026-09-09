"""Go2 flat and rough locomotion env cfgs, with the Hwangbo-curriculum push
disturbance added on top of Isaac Lab's stock UnitreeGo2FlatEnvCfg /
UnitreeGo2RoughEnvCfg. Rewards, PPO hyperparameters, and network architecture
are untouched -- see policy/locomotion/REFERENCES.md. Values (push
force/torque, root body name, terrain rescale) come from this platform's GO2
PlatformCfg instance (platform.py), not hardcoded here.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from policy.locomotion.core.curriculum import terrain_level_frac_at_max, terrain_level_max
from policy.locomotion.core.platforms.go2.platform import GO2
from policy.locomotion.core.push_disturbance_cfg import add_push_disturbance


@configclass
class UnitreeGo2FlatCerberusEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(
            self,
            peak_force_n=GO2.push_peak_force_n,
            peak_torque_nm=GO2.push_peak_torque_nm,
            root_body_name=GO2.root_body_name,
        )


@configclass
class UnitreeGo2FlatCerberusEnvCfg_PLAY(UnitreeGo2FlatCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing events -- push recovery is evaluated separately
        # (policy/locomotion/eval/), not exercised during a play/visualization run
        self.events.base_external_force_torque = None
        self.events.push_disturbance = None
        # the curriculum term references push_disturbance above; must go too
        self.curriculum.push_disturbance_curriculum = None

        # Follow-cam: the default viewer is a fixed world-space camera that
        # never tracks the robot, so footage drifts out of frame once a push
        # (or just plain walking) moves the robot away from its spawn point --
        # see REFERENCES.md's camera-framing writeup. asset_root origin makes
        # Isaac Lab recompute eye/lookat from the robot's current root pose
        # every step instead of once. Needs a pod-side smoke test to confirm
        # (not verifiable from source alone -- Isaac Lab isn't installed here).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.5, -2.5, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.3)


@configclass
class UnitreeGo2RoughCerberusEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(
            self,
            peak_force_n=GO2.push_peak_force_n,
            peak_torque_nm=GO2.push_peak_torque_nm,
            root_body_name=GO2.root_body_name,
        )

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
        # ceiling. See policy/locomotion/REFERENCES.md.
        stock_step_height_range = (0.05, 0.23)
        scaled_step_height_range = tuple(v * GO2.stairs_step_height_scale for v in stock_step_height_range)
        for stairs_type in ("pyramid_stairs", "pyramid_stairs_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[stairs_type].step_height_range = (
                scaled_step_height_range
            )

        # Diagnostic-only curriculum terms (no effect on training): the stock
        # terrain_levels term above only logs the MEAN across envs, which can't
        # confirm plans/handoff_phase0.md's actual "Done when" criterion --
        # envs reaching the curriculum's top row and cycling back down, not
        # just a high average. See curriculum.py's terrain_level_max /
        # terrain_level_frac_at_max docstrings.
        max_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.curriculum.terrain_level_max = CurrTerm(func=terrain_level_max)
        self.curriculum.terrain_level_frac_at_max = CurrTerm(
            func=terrain_level_frac_at_max, params={"max_level": max_level}
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
