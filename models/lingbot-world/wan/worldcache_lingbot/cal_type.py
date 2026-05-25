def evaluate_step_decision(cache_dic, current, step):
    mode = cache_dic["mode"]
    num_steps = current["num_steps"]
    last_full_step = current["activated_steps"][-1] if current["activated_steps"] else None
    segment_start = current.get("segment_start_step", 0)
    segment_step = step - segment_start
    k = 0 if last_full_step is None else step - last_full_step

    if mode == "original":
        return {
            "type": "full",
            "reason": "ORIGINAL_MODE",
            "k": k,
            "segment_step": segment_step,
            "error": 0.0,
        }

    config = cache_dic["worldcache_config"]
    accumulated_error = config.get("chaotic_error_accumulated", 0.0)

    if segment_step < config["warmup_steps"]:
        decision_type = "full"
        reason = "WARMUP"
    elif step == num_steps - 1:
        decision_type = "full"
        reason = "LAST_STEP"
    elif cache_dic.get("cached_curvature") is None:
        decision_type = "full"
        reason = "NO_CURVATURE"
    elif accumulated_error >= config["error_threshold"]:
        decision_type = "full"
        reason = "ERROR_EXCEEDED"
    elif last_full_step is None:
        decision_type = "full"
        reason = "NO_FULL_BASELINE"
    elif k > config["n_max"]:
        decision_type = "full"
        reason = "K_EXCEEDED"
    else:
        decision_type = "worldcache"
        reason = "NORMAL_PREDICT"

    return {
        "type": decision_type,
        "reason": reason,
        "k": k,
        "segment_step": segment_step,
        "error": accumulated_error,
    }


def combine_branch_decisions(branch_decisions):
    full_branches = {
        branch: decision
        for branch, decision in branch_decisions.items()
        if decision["type"] == "full"
    }
    if full_branches:
        return {
            "type": "full",
            "reasons": full_branches,
            "k": max(decision["k"] for decision in branch_decisions.values()),
        }
    return {
        "type": "worldcache",
        "reasons": branch_decisions,
        "k": max(decision["k"] for decision in branch_decisions.values()),
    }


def apply_shared_step_decision(cache_dic, current, step, decision_type):
    current["step"] = step
    current["type"] = decision_type

    if decision_type == "full":
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(step)
    else:
        cache_dic["cache_counter"] += 1
