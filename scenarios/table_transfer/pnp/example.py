"""Call from an application that owns a Context; no second simulator is started."""
from pathlib import Path
from .contracts import Goal
from .skill import PickPlaceSkill


def execute_transfer(context, request_id, source_id, destination_id, placement_xy=(0., 0.)):
    logs = Path(__file__).resolve().parents[1]/'tests/artifacts/pnp/runs'
    skill = PickPlaceSkill(context, logs)
    goal = Goal(request_id=request_id, object_id='blue_cube', dimensions=(.04, .04, .04),
                source_id=source_id, destination_id=destination_id, placement_xy=placement_xy,
                arm='auto', timeout_s=1800, max_retries=1)
    result = skill.execute(goal, feedback=lambda event: print(event, flush=True))
    # For cancellation from another thread: skill.cancel(). The context is not
    # released until hold is confirmed; this never issues an automatic open.
    return result
