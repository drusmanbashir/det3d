from fran.utils.folder_names import (
    expand_by_conv,
    load_registry,
    maybe_join,
    short_code,
    spacing_to_str,
)


def obd_folder_from_plan(project, plan: dict):  #AI
    """Output folder for object-bounded detection patches under rapid_access/obd."""
    reg = load_registry()
    spc = spacing_to_str("spc", plan.get("spacing"))
    expand_by = expand_by_conv(reg, "expand_by", plan.get("expand_by"))
    remapping_src_code = short_code(plan.get("remapping_source"))
    if remapping_src_code:
        remapping_src_code = "rsc" + remapping_src_code
    source_folder_suff = maybe_join([spc, remapping_src_code, expand_by])
    return project.obd_folder / source_folder_suff


def lbd_det_folder_from_plan(project, plan: dict):  #AI
    """Output folder for label-bounded detection volumes under rapid_access/lbd."""
    reg = load_registry()
    spc = spacing_to_str("spc", plan.get("spacing"))
    expand_by = expand_by_conv(reg, "expand_by", plan.get("expand_by"))
    remapping_lbd_code = short_code(plan.get("remapping_lbd_rbd"))
    if remapping_lbd_code:
        remapping_lbd_code = "rlb" + remapping_lbd_code
    crop_label = int(plan["lbd_crop_label"])
    crop_tag = f"lbl{crop_label}"
    source_folder_suff = maybe_join([spc, crop_tag, remapping_lbd_code, expand_by])
    return project.lbd_folder / source_folder_suff
