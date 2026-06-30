# %%
if __name__ == "__main__":
# SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from fran.managers import Project

    from det3d.managers.retinaunet import RetinaUNetManager
    from fran.managers.project import Project
    from utilz.helpers import pp
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet
    from det3d.detection.nndet_train import ensure_nndet_importable
    from det3d.extra.trainer_nndet import (
        apply_det3d_plan_to_nndet_model_cfg,
        load_nndet_train_cfgs,
        plan_from_det3d,
    )
    from det3d.trainers.trainerdet import TrainerDet

    project_title = "lidca"
    plan_id = 1
    arch = "retinaunet"  # "retinanet" | "retinaunet"
    is_retinaunet = arch == "retinaunet"

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    pp(conf["plan_train"])

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    device_id = 1
    batch_tfms = True
    wandb = False
    run_name = None
    description = "det3d extra trainer scratch"
    tags = []
    conf["dataset_params"]["fold"] = 0
    conf["model_params"]["arch"] = arch
    nndet_forward_patch_size = [128, 128, 64] if is_retinaunet else None
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    val_every_n_epochs = 2
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 5
    batch_size = 1
    wandb_grid_epoch_freq = 999 if is_retinaunet else 5
    precision = "bf16-mixed"
# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    Tm = TrainerDet(P.project_title, conf, run_name)
    if run_name is not None:
        Tm.run_name = run_name
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=val_every_n_epochs,
        cbs=cbs,
        debug=debug_,
        batch_size=batch_size,
        batch_tfms=batch_tfms,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
        lr=lr,
        nndet_forward_patch_size=nndet_forward_patch_size,
    )
# %%
    # Tm.fit()
# %%
# SECTION:-------------------- TS--------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    tmt = D.train_manager
    tmv = D.valid_manager
    print("arch", arch)
    print("train", type(tmt).__name__, tmt.data_folder, tmt.plan["patch_size"])
    print("valid", type(tmv).__name__, tmv.data_folder)
# %%
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
# %%
    N = Tm.setup_model_for_cuda(device=device_id, precision=precision)
    if is_retinaunet:
        N.nndet_module.trainer_cfg["num_train_batches_per_epoch"] = len(train_dl)
    else:
        N.on_fit_start()

# %%
# SECTION:-------------------- TRAIN STEP-BY-STEP ---------------------------------------------------------------------------
# %%
    train_batch = next(train_iter)
    if tmt.transforms_batch is not None:
        train_batch = tmt.transforms_batch(train_batch)
    train_batch = Tm.fabric_infer.to_device(train_batch)
# %%
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
    num_train_batches = 2500
    plan_train = conf["plan_train"]
    plan_path = conf["model_params"].get("nndet_plan_path")
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    trainer_cfg["max_num_epochs"] = int(conf["model_params"].get("max_epochs", 600))
    plan = plan_from_det3d(plan_train, plan_path=plan_path)
# %%
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )

    #
    train_batch.keys()
    img = train_batch["image"]

# %%
# %%
# SECTION:-------------------- RETINAUNET--------------------------------------------------------------------------------------
    assert is_retinaunet, "arch must be 'retinaunet' for this test"
    N.net.train()
    losses, _, nb = N._step_losses(train_batch, batch_idx=0, evaluation=False)
# %%

    batch = train_batch
    batch_idx = 0
    evaluation = False
# %%  # T:block_start|RetinaUNetManager._step_losses
#SECTION:-------------------- _step_losses--------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetManager._step_losses
    # requires R = RetinaUNetManager(...) in __main__  # T:requires_alias|R = RetinaUNetManager(...)
    R= N
    nb = R._det3d_batch_to_nndet(batch)
    device_type = nb["data"].device.type
    with torch.autocast(device_type, enabled=device_type == "cuda"):
        losses, prediction = R.net.train_step(  # T:self_ref|    losses, prediction = self.net.train_step(
            images=nb["data"],
            targets={
                "target_boxes": nb["target_boxes"],
                "target_classes": nb["target_classes"],
                "target_seg": nb["target_seg"],
            },
            evaluation=evaluation,
            batch_num=batch_idx,
        )
    _step_losses_result = losses, prediction, nb  # T:return|return losses, prediction, nb
    # end PythonMethodScratch  # T:block_end|RetinaUNetManager._step_losses

# %%
    print({k: float(v.detach()) for k, v in losses.items()})
    img = nb["data"][0, 0].detach().cpu()
    # img = img.float()
    box = train_batch["bbox"][0].detach().cpu()
    if box.numel():
        try:
            ImageBBoxViewer(img, box)
        except Exception as e:
            print("ImageBBoxViewer skipped:", e)

    # del losses, nb, train_batch  # and any other CUDA tensors you created here
    # torch.cuda.empty_cache()
# %%
# SECTION:-------------------- RETINANET--------------------------------------------------------------------------------------
    from det3d.detection.retinanet_train import build_train_anchors, forward_network_head
    from monai.apps.detection.utils.detector_utils import check_training_targets

    N.detector.train()
    images = train_batch["image"]
    targets = N._targets_from_batch(train_batch)
    print(images.mean(), images.min(), images.max())
    targets = check_training_targets(
        images,
        targets,
        N.detector.spatial_dims,
        N.detector.target_label_key,
        N.detector.target_box_key,
    )
    N.detector._check_detector_training_components()
    n = 0
    img = images[n, 0]
    bbox = targets[n]["bbox"]
    print(bbox.numel())
    if bbox.numel():
        try:
            ImageBBoxViewer(img, bbox)
        except Exception as e:
            print("ImageBBoxViewer skipped:", e)
    head_outputs = forward_network_head(N.detector, images)
    head_outputs, num_anchor_locs = build_train_anchors(
        N.detector, images, head_outputs
    )
    outputs = N.detector.compute_loss(
        head_outputs, targets, N.detector.anchors, num_anchor_locs
    )
    cls_loss = outputs[N.detector.cls_key]
    box_loss = outputs[N.detector.box_reg_key]
    train_loss = float(N.w_cls * cls_loss + N.w_reg * box_loss)
    print(
        "train_loss",
        float(train_loss),
        "cls",
        float(cls_loss),
        "box",
        float(box_loss),
    )
# SECTION:-------------------- VAL STEP-BY-STEP -----------------------------------------------------------------------------
# %%
    val_batch = next(val_iter)
    if tmv.transforms_batch is not None:
        val_batch = tmv.transforms_batch(val_batch)
    val_batch = Tm.fabric_infer.to_device(val_batch)
# %%
    if is_retinaunet:
        N.on_validation_epoch_start()
        N.validation_step(val_batch, batch_idx=0)
        N.on_validation_epoch_end()
    else:
        N.on_validation_epoch_start()
        N.detector.eval()
        val_inputs = N._val_inputs_from_batch(val_batch)
        val_targets = N._targets_from_batch(val_batch)
        use_inferer = N._use_sliding_window_inferer(val_inputs)
        if use_inferer:
            print("retinanet scratch: skip tiled val_forward on full-volume batch")
        else:
            val_outputs = N.detector(val_inputs, use_inferer=False)
            N.val_outputs_all.extend(val_outputs)
            N.val_targets_all.extend(val_targets)
        N.on_validation_epoch_end()
# %%
    cb = Tm.get_callback("CaseIDRecorder")
    getattr(cb, "dfs", None)
    print("scratch OK", arch)
# %%

