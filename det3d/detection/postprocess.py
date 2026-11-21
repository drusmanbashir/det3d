POSTPROCESS_KEYS = ("score_thresh", "nms_thresh", "detections_per_img")


def postprocess_values(configs):
    """Box-selector thresholds from the postprocess sheet.

    Checkpoints trained before these keys moved out of the plans sheet carry them
    in their stored `plan_train["postprocess"]` and a postprocess sheet without them.
    """
    sheet = configs["postprocess"]
    if all(key in sheet for key in POSTPROCESS_KEYS):
        return {key: sheet[key] for key in POSTPROCESS_KEYS}
    return {key: configs["plan_train"]["postprocess"][key] for key in POSTPROCESS_KEYS}
