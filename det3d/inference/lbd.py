from pathlib import Path

from utilz.helpers import chunks
from utilz.listify import listify


class DetLBDRunner:
    """Run RetinaUNet on pre-cropped LBD .pt volumes (no localiser / cascade)."""

    def __init__(
        self,
        run_p,
        project_title=None,
        devices=(0,),
        patch_overlap=0.25,
        safe_mode=False,
        params=None,
        debug=False,
        pred_run_p=None,
        save=True,
    ):
        from det3d.inference.lbd_pt import load_lbd_pt_patch_data
        from det3d.inference.patch import DetPatchLBD
        from fran.inference.helpers import load_params

        self.run_p = run_p
        self.pred_run_p = pred_run_p if pred_run_p is not None else f"{run_p}-lbd"
        if params is None:
            params = load_params(run_p)
        self.P = DetPatchLBD(
            run_name=run_p,
            project_title=project_title,
            devices=devices,
            patch_overlap=patch_overlap,
            safe_mode=safe_mode,
            params=params,
            debug=debug,
            save=save,
        )
        self.predictions_folder = self.P.project.predictions_folder

    @property
    def output_folder(self):
        return self.predictions_folder / self.pred_run_p

    def process_sublist(self, pt_paths):
        from det3d.inference.lbd_pt import load_lbd_pt_patch_data
        from det3d.inference.post import SaveDetOutputd

        self.P.create_postprocess_transforms(None)
        self.P.postprocess_transforms_dict["S"] = SaveDetOutputd(
            output_dir=self.output_folder,
            run_w="",
            run_p=self.run_p,
            write_seg=True,
        )
        self.P.set_postprocess_tfms_keys()
        self.P.set_postprocess_transforms()
        data = load_lbd_pt_patch_data(pt_paths)
        sources = [str(p) for p in pt_paths]
        from fran.inference.cascade import img_bbox_collated

        self.P.setup()
        self.P.prepare_data(data=data, collate_fn=img_bbox_collated)
        outputs = []
        for i, batch in enumerate(self.P.predict()):
            batch["source_image"] = sources[i]
            bbox = batch["bounding_box"]
            while len(bbox) == 1 and not isinstance(bbox[0], slice):
                bbox = bbox[0]
            batch["bounding_box"] = bbox
            batch = self.P.postprocess(batch)
            outputs.append(batch)
        if hasattr(self.P, "model"):
            del self.P.model
        import torch

        torch.cuda.empty_cache()
        return outputs

    def run(self, pt_paths, chunksize=12, overwrite=False):
        pt_paths = [Path(p) for p in listify(pt_paths)]
        if overwrite is False:
            out = Path(self.output_folder)
            pt_paths = [p for p in pt_paths if not (out / f"{p.stem}.json").exists()]
        if len(pt_paths) == 0:
            raise SystemExit("Stopping execution - no LBD cases remain after filtering")
        output = None
        for sublist in chunks(pt_paths, n_sized_chunks=chunksize):
            output = self.process_sublist(sublist)
        return output

# %%
# SECTION:-------------------- setup --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    import torch
    from fran.inference.common_vars import *
    from utilz.fileio import load_json
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

    devices = [0]
    safe_mode = True
    patch_overlap = 0.25
    debug_ = False
    overwrite = True
    chunksize = 4


# %%
# SECTION:-------------------- LIDC LBD — 0 config ---------------------------------------------------------------------
    run_p = "LIDCA-QUARK"
    project_title = "lidca"
    case_id = "lidc_0001"

    lbd_pt_fldr = Path(
        "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000/images"
    )
    pt_paths = sorted(lbd_pt_fldr.glob("*.pt"))
    pt_paths = [p for p in pt_paths if case_id in p.name]

# %%
# SECTION:-------------------- LIDC LBD — 1 runner -----------------------------------------------------------------------
    R = DetLBDRunner(
        run_p=run_p,
        project_title=project_title,
        devices=devices,
        patch_overlap=patch_overlap,
        safe_mode=safe_mode,
        debug=debug_,
        save=False,
    )
    print("output_folder", R.output_folder)
    print("patch postproc", R.P.keys_postproc)

# %%
# SECTION:-------------------- LIDC LBD — 2 process_sublist stepwise ----------------------------------------------------
    from det3d.inference.lbd_pt import load_lbd_pt_patch_data
    from fran.inference.cascade import img_bbox_collated

    R.P.create_postprocess_transforms(None)
    R.P.set_postprocess_tfms_keys()
    R.P.set_postprocess_transforms()
    data = load_lbd_pt_patch_data(pt_paths)
    sources = [str(p) for p in pt_paths]
    R.P.setup()
    R.P.prepare_data(data=data, collate_fn=img_bbox_collated)
    batch = next(iter(R.P.predict()))
    batch["source_image"] = sources[0]
    bbox = batch["bounding_box"]
    while len(bbox) == 1 and not isinstance(bbox[0], slice):
        bbox = bbox[0]
    batch["bounding_box"] = bbox
    print("pre-postprocess", sorted(batch.keys()))

# %%
    batch = R.P.predict_inner(batch)
    print(
        "predict_inner",
        "swi" if "merged_boxes" in batch else "tile",
        batch.get("merged_boxes", batch.get("raw_pred", "")).__class__.__name__,
    )

# %%
    tfms = R.P.postprocess_transforms_dict
    dici = dict(batch)
    for key in R.P.keys_postproc.split(","):
        if key == "S":
            continue
        dici = tfms[key](dici)
        print(
            key,
            "pred fg",
            int(dici["pred"].sum()),
            "n_boxes",
            dici["pred_box"].shape[0],
        )

# %%
    img = dici["image"][0, 0].detach().cpu()
    pred = dici["pred"][0].detach().cpu()
    boxes = dici["pred_box"].detach().cpu()
    scores = dici["pred_score"].detach().cpu()
    ImageMaskViewer([img, pred], "lbd")
    ImageBBoxViewer(img, boxes)

# %%
# SECTION:-------------------- LIDC LBD — 3 full run (optional save) ---------------------------------------------------
    R_save = DetLBDRunner(
        run_p=run_p,
        project_title=project_title,
        devices=devices,
        patch_overlap=patch_overlap,
        safe_mode=safe_mode,
        debug=debug_,
        save=True,
    )
    out = R_save.run(pt_paths, chunksize=chunksize, overwrite=overwrite)
    batch = out[0]
    sidecar = Path(R_save.output_folder) / f"{pt_paths[0].stem}.json"
    print("sidecar", sidecar.exists(), sidecar)
    if sidecar.exists():
        sc = load_json(sidecar)
        print("n_predictions", len(sc["predictions"]))
