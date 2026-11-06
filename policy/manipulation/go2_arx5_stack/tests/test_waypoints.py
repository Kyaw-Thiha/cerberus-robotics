import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy.manipulation.go2_arx5_stack import layout
from policy.manipulation.go2_arx5_stack import waypoints as wp

PICK = layout.CUBE_2_DEFAULT_POS
PLACE = layout.CUBE_1_DEFAULT_POS
STOW_TCP = (0.25, 0.0, 0.56)  # roughly where the stowed TCP sits in the env frame
CFG = wp.PickPlaceConfig(
    cube_size=layout.CUBE_SIZE,
    hover_height=layout.HOVER_HEIGHT,
    ik_scale=layout.IK_ACTION_SCALE,
)


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def simulate(
    machine, eef0=STOW_TCP, rot0=(0.0, 1.5, 0.0), max_steps=3000, tracking=1.0
):
    """First-order plant: the commanded pose moves by ik_scale * action and the TCP follows it (times `tracking`)."""
    eef, rot = list(eef0), list(rot0)
    log = []
    for _ in range(max_steps):
        act = machine.step(tuple(eef), tuple(rot))
        log.append((act, tuple(eef), tuple(rot)))
        if machine.finished:
            break
        cmd = wp.commanded_delta(act.pos, CFG.ik_scale)
        cmd_rot = wp.commanded_delta(act.rot, CFG.ik_scale)
        for i in range(3):
            eef[i] += tracking * cmd[i]
            rot[i] -= tracking * cmd_rot[i]
    return log


def make_machine(cfg=CFG, pick=PICK, place=PLACE):
    machine = wp.ScriptedPickPlace(cfg)
    machine.reset(pick, place)
    return machine


def test_clip_norm_preserves_direction_and_leaves_short_vectors():
    assert wp.clip_norm((0.0, 0.0, 0.0), 0.1) == (0.0, 0.0, 0.0)
    assert wp.clip_norm((0.03, 0.0, 0.04), 0.1) == (0.03, 0.0, 0.04)
    clipped = wp.clip_norm((0.3, 0.0, 0.4), 0.1)
    assert _norm(clipped) == pytest.approx(0.1)
    assert clipped[0] / clipped[2] == pytest.approx(0.3 / 0.4)


def test_relative_ik_action_is_clipped_to_the_ik_scale():
    pos, rot = wp.relative_ik_action((1.0, -2.0, 0.5), (0.0, 3.0, 0.0), CFG)
    assert _norm(pos) == pytest.approx(CFG.max_pos_action)
    assert _norm(rot) == pytest.approx(CFG.max_rot_action)
    # The commanded step (scale * action) stays within the configured per-step limits.
    assert (
        _norm(wp.commanded_delta(pos, CFG.ik_scale)) <= CFG.max_command_step_m + 1e-12
    )
    assert (
        _norm(wp.commanded_delta(rot, CFG.ik_scale)) <= CFG.max_command_rot_rad + 1e-12
    )
    # Small errors pass through unscaled (stock Mimic convention: action = target - current).
    pos, rot = wp.relative_ik_action((0.004, 0.0, -0.003), (0.0, 0.01, 0.0), CFG)
    assert pos == (0.004, 0.0, -0.003)
    assert rot == (0.0, 0.01, 0.0)


def test_config_validation():
    with pytest.raises(ValueError):
        wp.PickPlaceConfig(ik_scale=0.0)
    with pytest.raises(ValueError):
        wp.PickPlaceConfig(close_pause_steps=0)
    with pytest.raises(ValueError):
        wp.PickPlaceConfig(max_command_step_m=-0.01)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        wp.ScriptedPickPlace(CFG).step(STOW_TCP)


def test_plan_waypoints():
    machine = make_machine()
    plan = {w.phase: w for w in machine.plan}
    assert tuple(w.phase for w in machine.plan) == wp.PHASE_ORDER
    hover = CFG.hover_height
    # above/descend the pick cube
    assert plan[wp.Phase.ABOVE_PICK].target == pytest.approx(
        (PICK[0], PICK[1], PICK[2] + hover)
    )
    assert plan[wp.Phase.DESCEND_PICK].target == pytest.approx(PICK)
    # the release pose is one cube side (plus margin) above the place cube's centre
    release_z = PLACE[2] + CFG.cube_size + CFG.grasp_z_offset + CFG.place_margin
    assert plan[wp.Phase.DESCEND_PLACE].target == pytest.approx(
        (PLACE[0], PLACE[1], release_z)
    )
    assert plan[wp.Phase.RETREAT].target == pytest.approx(
        (PLACE[0], PLACE[1], release_z + hover)
    )
    # gripper commands: open before the grasp, closed from the grasp until the release
    open_phases = {
        wp.Phase.ABOVE_PICK,
        wp.Phase.DESCEND_PICK,
        wp.Phase.OPEN,
        wp.Phase.RETREAT,
    }
    for w in machine.plan:
        assert w.gripper == (
            wp.GRIPPER_ACTION_OPEN
            if w.phase in open_phases
            else wp.GRIPPER_ACTION_CLOSE
        )
    assert machine.phase is wp.Phase.ABOVE_PICK
    assert not machine.finished


def test_grasp_z_offset_shifts_grasp_and_release():
    cfg = wp.PickPlaceConfig(cube_size=0.04, grasp_z_offset=0.01)
    machine = wp.ScriptedPickPlace(cfg)
    machine.reset(PICK, PLACE)
    plan = {w.phase: w for w in machine.plan}
    assert plan[wp.Phase.DESCEND_PICK].target[2] == pytest.approx(PICK[2] + 0.01)
    assert plan[wp.Phase.DESCEND_PLACE].target[2] == pytest.approx(
        PLACE[2] + 0.04 + 0.01 + cfg.place_margin
    )


def test_full_episode_visits_phases_in_order_and_finishes():
    machine = make_machine()
    log = simulate(machine)
    assert machine.done and not machine.failed
    seen = []
    for act, _, _ in log[:-1]:
        if not seen or seen[-1] is not act.phase:
            seen.append(act.phase)
    assert tuple(seen) == wp.PHASE_ORDER
    assert machine.failure_reason is None
    # the fake TCP ends over the place cube, at retreat height
    _, eef_end, _ = log[-1]
    retreat = machine.plan[-1].target
    assert _norm(tuple(a - b for a, b in zip(eef_end, retreat))) <= CFG.hover_pos_tol


def test_actions_stay_within_the_clip_and_have_seven_dims():
    machine = make_machine()
    log = simulate(machine)
    assert len(log) < 600
    for act, _, _ in log:
        vec = act.as_list()
        assert len(vec) == wp.ACTION_DIM == layout.ACTION_DIM
        assert _norm(act.pos) <= CFG.max_pos_action + 1e-9
        assert _norm(act.rot) <= CFG.max_rot_action + 1e-9
        assert act.gripper in (wp.GRIPPER_ACTION_OPEN, wp.GRIPPER_ACTION_CLOSE)


def test_close_and_open_hold_for_their_pause_steps():
    machine = make_machine()
    log = simulate(machine)
    counts = {}
    for act, _, _ in log[:-1]:
        counts[act.phase] = counts.get(act.phase, 0) + 1
    assert counts[wp.Phase.CLOSE] == CFG.close_pause_steps
    assert counts[wp.Phase.OPEN] == CFG.open_pause_steps


def test_gripper_only_closes_after_the_descent_and_opens_over_the_place_cube():
    machine = make_machine()
    log = simulate(machine)
    closing = [(act, eef) for act, eef, _ in log[:-1] if act.phase is wp.Phase.CLOSE]
    assert closing
    for _, eef in closing:
        # TCP is at the grasp pose (within the fine tolerance) while the gripper closes
        assert _norm(tuple(a - b for a, b in zip(eef, PICK))) <= CFG.fine_pos_tol + 1e-9
    opening = [eef for act, eef, _ in log[:-1] if act.phase is wp.Phase.OPEN]
    release = machine.plan[5].target
    for eef in opening:
        assert (
            _norm(tuple(a - b for a, b in zip(eef, release))) <= CFG.fine_pos_tol + 1e-9
        )


def test_rotation_error_blocks_advancing():
    machine = make_machine()
    target = machine.plan[0].target
    # position is right on the first waypoint but the hand is still 1 rad off: no transition
    act = machine.step(target, (0.0, 1.0, 0.0))
    assert act.phase is wp.Phase.ABOVE_PICK
    assert _norm(act.rot) > 0.0
    # rotation done as well: the next waypoint is active on the same call
    act = machine.step(target, (0.0, 0.0, 0.0))
    assert act.phase is wp.Phase.DESCEND_PICK


def test_sluggish_tracking_still_converges():
    machine = make_machine()
    log = simulate(machine, tracking=0.6, max_steps=4000)
    assert machine.done, machine.failure_reason


def test_timeout_fails_with_reason_and_then_holds_still():
    cfg = wp.PickPlaceConfig(cube_size=0.04, max_phase_steps=20)
    machine = wp.ScriptedPickPlace(cfg)
    machine.reset(PICK, PLACE)
    act = None
    for _ in range(25):
        act = machine.step(STOW_TCP, (0.0, 0.0, 0.0))  # the plant never moves
        if machine.finished:
            break
    assert machine.failed and not machine.done
    assert "timeout in above_pick" in machine.failure_reason
    assert act.phase is wp.Phase.FAILED
    assert act.pos == (0.0, 0.0, 0.0) and act.rot == (0.0, 0.0, 0.0)
    after = machine.step(STOW_TCP)
    assert after.pos == (0.0, 0.0, 0.0) and after.gripper == wp.GRIPPER_ACTION_OPEN


def test_done_state_is_inert():
    machine = make_machine()
    simulate(machine)
    assert machine.done and machine.target is None
    act = machine.step(STOW_TCP)
    assert act.phase is wp.Phase.DONE
    assert act.as_list() == [0.0] * 6 + [wp.GRIPPER_ACTION_OPEN]


def test_reset_clears_previous_episode():
    machine = make_machine()
    simulate(machine)
    machine.reset(PICK, PLACE)
    assert machine.phase is wp.Phase.ABOVE_PICK
    assert machine.total_steps == 0 and not machine.finished
    assert machine.failure_reason is None
