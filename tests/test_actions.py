from ha_voice.actions import ActionResult, no_action


def test_no_action_is_an_unassigned_handler() -> None:
    assert no_action("lights_on") == ActionResult(status="unassigned")


def test_action_result_can_suppress_feedback() -> None:
    result = ActionResult(status="failed", error="offline", suppress_feedback=True)
    assert result.status == "failed"
    assert result.error == "offline"
    assert result.suppress_feedback is True
