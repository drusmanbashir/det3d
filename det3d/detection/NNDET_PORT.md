# nnDetection selective port — provenance

Vendored architecture only under `det3d/detection/arch/nndet/`. No runtime `import nndet`.

| det3d | nnDetection source |
|-------|-------------------|
| `arch/nndet/conv.py` | `nndet/arch/conv.py` |
| `arch/nndet/encoder/` | `nndet/arch/encoder/` |
| `arch/nndet/decoder/base.py` | `nndet/arch/decoder/base.py` (UFPNModular only) |
| `arch/nndet/blocks/basic.py` | `nndet/arch/blocks/basic.py` (StackedConvBlock2 only) |
| `arch/nndet/layers/norm.py` | `nndet/arch/layers/norm.py` |

## MONAI / Lightning first

| Need | Use |
|------|-----|
| RetinaNet detector shell | MONAI `RetinaNetDetector` + `RetinaNetDetector2` |
| ResNet-FPN | MONAI (`detector=retinanet`) |
| GIoU / BCE | `monai.losses` + plan columns |
| Training loop | Lightning `TrainerDet` |
| Epoch LR | `GradualWarmupScheduler` + StepLR (default) |
| Iter poly LR | Lightning step `LambdaLR` (`lr_schedule=poly_iter`) |
| SWA | `StochasticWeightAveraging` callback (`use_swa=True`) |
| Val metrics | `det3d/evaluation/coco.py` |
| RetinaUNet body | vendored encoder/UFPN only |

## Config

- Common: `~/code/fran/configurations/experiment_configs.xlsx`
- Detection: `~/code/fran/configurations/experiment_configs_det.xlsx`
- Advisor: `PlanAdvisorDet` in `det3d/configs/parser.py` (offline compare only)

License: Apache-2.0 (nnDetection / DKFZ headers retained in vendored files).
