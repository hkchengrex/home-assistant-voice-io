# Connect an automation

Recognition and automation are deliberately separate. The library accepts a spoken command name; your adapter decides whether that means a Home Assistant service call, a local script, or something else.

## The handler contract

A handler receives the accepted command name and returns an `ActionResult`:

```python
from ha_voice import ActionResult
from ha_voice.cli import main


def handle_command(command_name: str) -> ActionResult:
    if command_name == "lights_on":
        # Call your automation here.
        return ActionResult(status="sent")
    return ActionResult(status="unassigned")


main(command_handler=handle_command)
```

Possible statuses are:

| Status | Meaning |
| --- | --- |
| `sent` | The adapter accepted and dispatched the command. |
| `unassigned` | Recognition succeeded, but the adapter has no action for it. |
| `failed` | The adapter tried and could not complete the action. |

Set `suppress_feedback=True` when the adapter should prevent local response playback.

## Keep platform details outside the library

Store URLs, tokens, entity IDs, and household mappings in the consuming application—not in this repository or its public command examples. A clean adapter usually has three parts:

1. A private map from recognized command names to automation actions.
2. A small client that sends the action and handles timeouts.
3. A handler that converts the outcome to `ActionResult`.

## Home Assistant

Use Home Assistant's authenticated API, a webhook owned by your private deployment, or another local integration. Keep the exact endpoint and entity mappings in the private Home Assistant repository. The voice library should know only stable intent names such as `lights_on` or `good_night`.

!!! tip "Group alternate wording"
    If “lights on” and another phrase should perform the same action, give both commands the same `intent_group`. Your adapter can then map the logical intent without weakening recognition between aliases.
