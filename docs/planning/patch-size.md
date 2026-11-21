# How det3d chooses the LBD training patch

The spreadsheet supplies `patch_dim0` and `patch_dim1`. Two numbers give a requested patch of `[dim0, dim0, dim1]`. If either cell is `auto`, BoxC002 chooses the whole patch.

| Spreadsheet dimensions | Model strides | Training patch |
|---|---|---|
| `auto`, `auto` | Planned | BoxC002 starts from the spacing and median LBD crop shape, then reduces the patch if its memory estimate exceeds the budget. |
| `128`, `64` | Preset to `[2, 2, 1]`, then `[2, 2, 2]` | `[128, 128, 64]` survives unchanged. All axes are divisible by the total stride `[4, 4, 2]`. |
| `130`, `65` | Same preset strides | Planning stops. `[130, 130, 65] is not an elementwise multiple of [4, 4, 2] in dim 0, 1, 2`. |

Numeric dimensions fix the training patch. The compatibility check multiplies the strides along each axis and reports the zero-based indices where division leaves a remainder. With an automatic model, det3d checks the fixed patch once nnDetection chooses the strides.

The ordinary LBD folder name uses the effective minimum crop shape. With a minimum of `[128, 128, 96]` and a prezoom scale of 1, both `auto` and a fixed `[64, 64, 64]` use that minimum and can share the cropped cases. A fixed `[192, 192, 96]` raises the minimum and selects another folder.

## Code path

1. `ConfigMakerDet._parse_det_plan` calls `parse_plan_row`. Its `derive_patch_size` handles `auto`, and `make_patch_size` builds the numeric list. [det3d/configs/parser.py](/home/ub/code/det3d/det3d/configs/parser.py) · [fran/configs/plan_parse.py](/home/ub/code/fran/fran/configs/plan_parse.py) · [fran/configs/helpers.py](/home/ub/code/fran/fran/configs/helpers.py)
2. `LBDBoxC002._plan_architecture` chooses the automatic or fixed path. Automatic patch search runs in upstream `BoxC002._plan_architecture`. For numeric dimensions, det3d calls `plan_pool_and_conv_pool_late`, and checks the requested patch against the preset or planned strides. [det3d/planning/c002.py](/home/ub/code/det3d/det3d/planning/c002.py) · [nnDetection c002.py](/home/ub/code/nnDetection/nndet/planning/architecture/boxes/c002.py) · [nnDetection base.py](/home/ub/code/nnDetection/nndet/planning/architecture/boxes/base.py)
3. `LBDBoxC002.fit_lbd` returns the training patch. `PreprocessingManager.plan_detection` saves it. At training setup, `ConfigMakerDet._load_computed_plan` applies the saved value to the active plan. [Planner](/home/ub/code/det3d/det3d/planning/c002.py) · [Preprocessing](/home/ub/code/fran/fran/run/preproc/analyze_resample.py) · [Training configuration](/home/ub/code/det3d/det3d/configs/parser.py)
