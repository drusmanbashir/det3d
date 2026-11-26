# det3d plan-file delivery

SHA: `46bd372c7e7be1e3c45031d07608b919879f72de`

FAIL-before: `FAIL: det3d has no shared plan-file input`
PASS-after: `PASS: det3d has shared plan-file input`

Rows run / passed / failed:
- `plan-file-launch`: 1 / 1 / 0.
- `det-config-overrides`: 1 / 1 / 0; verified workbook default, canonical and `data_params` alias precedence, and explicit `patch_stream` conflict.
- `det-fit-spacing-key`: 1 / 1 / 0; named fit key includes resolved spacing and legacy fit key stays a string.
- `det-wrapper-argv`: 1 / 1 / 0; fake E2E, local, submit and HPC calls preserve selectors and file path, plus legacy positional calls.

Callers ran: `ConfigMakerDet.setup` on test plan 7; `det3d/run/e2e.sh`; `local_train_det.sh`; `train_det.sh`; `submit_train_det.sh`; `fit_plan` with fake GPU and fit-store boundaries. No Slurm submission or GPU job ran.

Targeted test: `tests/unit/test_nndet_intensity.py` — 11 passed against composed FRAN `run-plan-integration`.
Checks: `bash -n`, `py_compile`, `git diff --check`, named row script, all passed. Rebase onto `origin/main` was up to date.

Spec deviations: none.

Unverified boundaries: no real preprocessing, GPU planning, model training, Slurm submission, or remote staging/replay was exercised.

Structural honesty: I added a boundary check that rejects explicit `dataset_params.valid_impl='patch_stream'` when validation batches are active; I kept the existing default backend selection because workbook defaults must still apply when the plan file omits that field.

Process note: the red assertion was the external work-order script and there was no repository test row in the worktree to commit before implementation.

usage: gpt-6.2-sol; tokens ~15k; wall ~45m
