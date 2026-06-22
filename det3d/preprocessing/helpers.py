def dusting_threshold(plan):
    dusting = plan.get("dusting_mm")
    if dusting is None:
        dusting = 2.0
    return float(dusting)
