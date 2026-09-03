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
