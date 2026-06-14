# %%
from det3d.detection.retinanet_train import forward_train_batched
from det3d.trainers.trainerdet import TrainerDet
from det3d.managers.retinanet import RetinaNetManager
from utilz.imageviewers import ImageBBoxViewer


if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project
    from utilz.helpers import pp

    P = Project("lidc")
    C = ConfigMakerDet(P)
    C.setup(1)
    conf = C.configs
    pp(conf["plan_train"])

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
# %%
    device_id = 0
    wandb = True
    run_name = None
    description = "new changes after LIDC-TRAINT which failed"
    tags = []
    conf["dataset_params"]["fold"] = 0
    lr = None
    debug_ = False
    profiler = False
    cbs = []
    val_every_n_epochs = 1
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 5
    batch_size = 8
    case_id_recorder_freq = 1
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    Tm = TrainerDet(P.project_title, conf, None)
    if run_name is not None:
        Tm.run_name = run_name
# %%
    Tm.setup(
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=val_every_n_epochs,
        cbs=cbs,
        debug=debug_,
        batch_size=batch_size,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        tags=tags,
        description=description,
        lr=lr,
        case_id_recorder_freq=case_id_recorder_freq,
    )
# %%
    # Tm.fit()
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    tmt = D.train_manager
    tmv = D.valid_manager
# %%
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    train_iter = iter(train_dl)
# %%
# %%
    # images = N._image_batch_tensor(batch)
    # targets = N._targets_from_batch(batch)
    # outputs = forward_train_batched(N.detector, images, targets)
    # loss = N.w_cls * outputs[N.detector.cls_key] + outputs[N.detector.box_reg_key]
    # N.log("train0_loss", loss, prog_bar=True, sync_dist=N.sync_dist)

    N = Tm.setup_model_for_cuda(device=device_id, precision="16-mixed")
    N.on_fit_start()
# %%
# SECTION:-------------------- TRAIN STEP-BY-STEP ---------------------------------------------------------------------------
# %%
    from det3d.detection.retinanet_train import (
        build_train_anchors,
        compute_train_loss,
        forward_network_head,
        validate_train_targets,
    )

# %%
    N.detector.train()
# %%

    train_batch = next(train_iter)
    train_batch = tmt.transforms_batch(train_batch)
    batch = train_batch
    batch['image'].shape
    train_batch = Tm.fabric_infer.to_device(train_batch)
# %%
    

# %%
    batch = train_batch
# %%  # T:block_start|RetinaNetManager.train_images
#SECTION:-------------------- train_images--------------------------------------------------------------------------------------  # T:block_meta|RetinaNetManager.train_images
    # requires R = RetinaNetManager(...) in __main__  # T:requires_alias|R = RetinaNetManager(...)
    train_images_result = R._image_batch_tensor(batch)  # T:return|return self._image_batch_tensor(batch)
    image = batch["image"].to(self.device)
    if image.dim() == 4:
        image = image.unsqueeze(0)

    # end PythonMethodScratch  # T:block_end|RetinaNetManager.train_images


# %%
    images = N.train_images(train_batch)
    bbox = batch['bbox']
    label = batch[]
    targets = N.train_targets(train_batch)
    print(images.mean(),images.min(),images.max())
# %%
    targets = validate_train_targets(N.detector, images, targets)
    N.detector._check_detector_training_components()
# %%
    n = 6
    img = images[n, 0]
    tg = targets[n]
    bbox = tg['bbox']
    print(bbox.numel())
    ImageBBoxViewer(img, bbox)

# %%
    head_outputs = forward_network_head(N.detector, images)
# %%
    head_outputs, num_anchor_locs = build_train_anchors(N.detector, images, head_outputs)
# %%
    outputs = compute_train_loss(N.detector, head_outputs, targets, num_anchor_locs)
# %%
    train_loss, cls_loss, box_loss = N.train_total_loss(outputs)
# SECTION:-------------------- VAL STEP-BY-STEP -----------------------------------------------------------------------------
# %%
    N.on_validation_epoch_start()
# %%
    N.detector.eval()
    val_inputs = N.val_inputs(val_batch)
    val_targets = N.val_targets(val_batch)
    use_inferer = N.val_use_inferer(val_inputs)
# %%
    val_outputs = N.val_forward(val_inputs, use_inferer=use_inferer)
# %%
    N.val_outputs_all.extend(val_outputs)
    N.val_targets_all.extend(val_targets)
    N.on_validation_epoch_end()
# %%
    cb = Tm.get_callback("CaseIDRecorder")
    cb.dfA
# %%
