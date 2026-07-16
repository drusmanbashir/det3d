# create bonespublic dataset

Plan: [bonespub_dataset_combine_1889a746.plan.md](/home/ub/.cursor/plans/bonespub_dataset_combine_1889a746.plan.md)

Build script: `/t/datasets/bonespublic/build_bonespublic.py`

Sources (NIfTI only):
- `/t/datasets/spinemm/routine` — lesion → class 1
- `/t/datasets/spine-mets-lytic-sclerotic` — lytic → 1, sclerotic → 2 (case-level CSV)
- `/t/datasets/spinemets` — excluded (vertebra instances only)

Output: `/t/datasets/bonespublic/` (`images/`, `lms/`, `MANIFEST.json`)
