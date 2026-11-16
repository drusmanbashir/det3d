# Lung lesions on CT: what to do, what carries to other organs

2026-11-13. Figures read in linked sources. F = `/home/ub/code/fran/fran`, D = `/home/ub/code/det3d/det3d`.

## Findings

1. Lung, 2024-2026: nothing beats nnU-Net ResEnc-L (segmentation) or nnDetection (detection). FRAN and det3d hold both; gap = recipe.
2. Detector score unusable: `/r/nnunet/eval/Dataset803_lungs/step2_froc/report.txt` CPM 0.060 vs published 0.930 ([nnDetection](https://arxiv.org/abs/2106.00817)). FROC is a straight line (precision ~0.08 at 1, 4, 8, 16.9 FP/scan): score does not separate hits from misses. Fits displaced boxes or scores not comparable across patches. Fix first.
3. Largest measured lung lever = training data: sensitivity at 2 FP/scan 0.72 vs 0.64 on external NLST, same RetinaNet trained on DLCS vs LUNA16 ([Duke](https://arxiv.org/pdf/2405.04605)).
4. Small lesions are the hard part: Dice 52.5 (3-10 mm) vs 77.1 (> 1 cm) ([Mathai](https://arxiv.org/pdf/2504.06924)). Report every result by size.
5. MSD Lung (63 scans) settles nothing: two baselines in one paper differ by 3.9 Dice ([MultiTalent](https://arxiv.org/html/2303.14444v2)). Lung results on pretraining, multi-dataset training, automated search: undecided, not negative.
6. Lung answer does not carry to other organs (toolkit table).

## Do, in order

| # | Step | Decided by |
|---|------|-----------|
| 1 | Audit detector score: overlay boxes on GT in world coordinates; compare scores of matched vs unmatched boxes; then re-score with LUNA16 radius rule | CPM, 100 LIDC validation cases |
| 2 | Run upstream `/home/ub/code/nnDetection/projects/Task016_Luna` unmodified; score with upstream and local scorers | CPM near 0.930 |
| 3 | det3d full recipe: 5 folds, Weighted Box Clustering, warm-up then poly (now: one fold, plateau, no TTA) | CPM; sensitivity at 1 FP/scan by size |
| 4 | FRAN ResEnc-L, nnU-Net schedule: SGD, poly, 1,000 epochs (now: plain U-Net, Adam, early stopping) | Lesion F1, strata < 10 mm and >= 10 mm, paired bootstrap |
| 5 | Add data: DLCS (detection); LNDb, MSD Lung (segmentation); hold one set out as external test | Held-out set: sensitivity at 2 FP/scan, lesion F1 |
| 6 | Pretrained vs scratch on `Dataset803_lungs`, same plan and epochs | Lesion F1, 5 folds |
| 7 | Sampling arms (below) | Sensitivity at fixed FP/scan by size |
| 8 | Fold ensemble + mirror TTA for final numbers (ULS23 TTA: 0.703 to 0.715, [paper](https://arxiv.org/html/2406.05231v2)) | As steps 3-4 |

## Toolkit for new organs

| Tool | Evidence | Use |
|------|----------|-----|
| nnU-Net as detector (score connected components) | Pancreas: only three PANORAMA teams beat the nnU-Net baseline ([paper](https://arxiv.org/abs/2503.10068)). Mediastinal nodes: recall at 2 FP 75.38 vs 68.01 Swin detector ([paper](https://arxiv.org/abs/2409.06324)). Liver, colon: ahead of nnDetection ([nnDetection](https://arxiv.org/abs/2106.00817)) | Score beside det3d for every new organ |
| Query-based detector, lymph nodes | nnDetection lowest of nine: recall 38.03 vs 56.27 LN-DETR, 1,067 patients, baseline run by the authors ([LN-DETR](https://arxiv.org/abs/2404.03819)) | Lymph nodes only, after a local nnDetection baseline |
| Multi-dataset training on partial labels | Vs same architecture per dataset: colon +11.29, pancreas +3.53, lung +2.27, liver tumour -1.03 ([MultiTalent](https://arxiv.org/html/2303.14444v2)) | When a second organ's lesion labels join; expect colon, pancreas gains |
| Pretrained encoder | Vs scratch: colon +7.1, pancreas +3.3, lung -1.8, liver -0.2, intervals overlap; ~+3 Dice at 10 cases ([nnFoundation](https://arxiv.org/html/2609.26924v1)) | Organs with few labelled scans |
| Larger patch and batch | autoPET II: patch 128³ to 192³, Dice 33.28 to 44.19; batch 2 to 12, 28.07 to 48.51; compute not matched ([paper](https://arxiv.org/pdf/2309.13747)) | Cheap arm, any organ |
| Dice loss without smoothing term | autoPET III winner, matched pairs: 66.08 to 67.01, 66.37 to 67.45 ([LesionTracer](https://arxiv.org/pdf/2409.09478)) | One-line arm, any organ |
| Instance-weighted loss | Node sensitivity 0.818 to 0.849, tumour precision 0.813 to 0.636 ([paper](https://arxiv.org/pdf/2604.10130)). LiTS recall 0.62 to 0.72, object Dice 0.63 to 0.49 ([paper](https://arxiv.org/pdf/2007.10033)) | Only where recall outranks precision |
| Second-stage false-positive classifier | Lung, lesions >= 10 mm: F1 0.680 to 0.712, intervals overlap ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361585/)) | After the detector is sound |

## Sampling

Present: det3d `RetinaUNetV001` mines hard negative anchors (`HardNegativeSamplerBatched`). FRAN `RandCropByFlatIndicesd` picks centres uniformly over foreground voxels: a lesion with 8x the volume is chosen 8x as often.

No matched sampler test on CT. Matched results elsewhere:

| Change | Result |
|--------|--------|
| Lesion first, then a voxel in it | Brain MRI: sensitivity 0.898 to 0.932, FP/patient 8.3 to 6.6, Dice flat ([Krivov](https://arxiv.org/pdf/1808.00244)) |
| Replay mined hard negatives | DeepLesion: AP 43.0 to 51.9 with unannotated lesions separated out; 34.2 to 17.3 without ([Lesion Harvester](https://arxiv.org/pdf/2001.07776)) |
| Less foreground in crops | PET/CT: Dice flat, false-positive volume 11.9 to 23.8 at 10% foreground ([paper](https://arxiv.org/pdf/2309.13675)) |

Lung precedent, no matched comparison: DSB2017 winners ([grt123](https://github.com/lfz/DSB2017), [de Wit](https://juliandewit.github.io/kaggle-ndsb2017/)), [CASED](https://arxiv.org/html/1807.10819).

Trial arms, one change each, after steps 1-4:

1. FRAN lesion-first centres.
2. FRAN false-positive replay: 75% foreground, 12.5% ordinary background, 12.5% mined background.

Risks: a mined false positive may be an unannotated lesion (review the pool, cap per scan); lesion-first sampling favours many tiny nodules over large-lesion boundaries.

## Theory under the evidence

Sources above are empirical. Each row: a proved result (or a short derivation, marked derived), its assumptions, what it changes in the plan.

| Evidence | Result | Source | Use |
|----------|--------|--------|-----|
| Finding 2: straight FROC, flat precision | If P(hit given score > t) is the same for every t, the joint distribution factorises: score independent of hit, sensitivity linear in FP/scan (derived). FROC area's core term: A = P(hit score > false-positive score), ties ½, a U-statistic | [Bandos 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2776072), eq. 7 | Step 1's matched-vs-unmatched comparison estimates A; A near 0.5 confirms the score ranks nothing |
| Finding 4: Dice 52.5 (3-10 mm) vs 77.1 (> 1 cm) | Ground-truth sphere radius r, prediction one voxel δ larger: Dice = 2r³/(r³ + (r+δ)³) ≈ 1 − 3δ/(2r). δ = 0.7 mm: 0.65 at 5 mm, 0.81 at 10 mm, 0.90 at 20 mm diameter (derived) | [Reinke 2024](https://arxiv.org/abs/2302.01790), Extended Data Fig. 1a | Part of the size gap is the metric. Compare arms within a size stratum, never across strata |
| Steps 3-4: poly schedule | SGD, convex G-Lipschitz loss: step size ∝ (1 − t/T) gives last-iterate error ≤ DG/√T, the optimal rate; poly (1 − t/T)^0.9 is near it. Convex only; deep-net support empirical | [Defazio 2023](https://arxiv.org/abs/2310.07831), Cor. 2 (preprint) | Theory for a fixed horizon T, decaying to zero |
| Step 4: SGD vs Adam | Over-parameterised separable construction: GD/SGD reach the minimum-norm solution, zero test error; AdaGrad, RMSprop, Adam reach w ∝ sign(Xᵀy), test error near ½. Synthetic | [Wilson 2017](https://arxiv.org/abs/1705.08292), Lemma 3.1, Sec. 3.3 | The optimiser selects which interpolating solution is found, not only how fast |
| Toolkit: batch 2 to 12 | SGD noise scale ≈ ηN/B (lr η, N samples, B ≪ N). Adam: batch ×κ keeps the same SDE with η ×√κ, β' = 1 − κ(1 − β), ε' = ε/√κ | [Smith & Le 2018](https://arxiv.org/abs/1710.06451); [Malladi 2022](https://arxiv.org/abs/2205.10287), Def. 5.2, Thm 5.3 | A matched batch arm rescales lr (linear for SGD, square-root rule for FRAN's Adam) and holds samples seen equal |
| Sampling: lesion-first | Crops drawn from q train E_q[loss]. Uniform foreground voxels weight each lesion by its volume; lesion-first weights lesions equally. Reweighting by p/q restores any target p; the variance-minimising q is ∝ per-sample gradient norm | [Katharopoulos & Fleuret 2018](https://arxiv.org/abs/1803.00942), Sec. 2.3; [Needell 2014](https://arxiv.org/abs/1310.5715) (convex) | Lesion-first changes the objective to the lesion-averaged loss, the one lesion sensitivity and F1 score |
| Sampling: false-positive replay | Sampling high-loss examples optimises a tail risk: the max loss, or CVaR (mean of the worst α-fraction; convex guarantees). One outlier can make the max-loss minimiser meaningless; average top-k loss is less sensitive | [Shalev-Shwartz & Wexler 2016](https://arxiv.org/abs/1602.01690), Sec. 1, 4; [Curi 2020](https://arxiv.org/abs/1910.12511), eq. 4; [Fan 2017](https://arxiv.org/abs/1705.08826) | The arm trains about 0.875 × ordinary risk + 0.125 × CVaR over background. An unannotated lesion in the pool is the outlier; the per-scan cap bounds its weight |
| Toolkit: instance-weighted loss | BCE with positives weighted w has minimiser f = wp/(wp + 1 − p): positive when p > 1/(1 + w); size-dependent w gives a size-dependent threshold (derived). Costs: threshold p* = c_FP/(c_FP + c_FN); re-weighting classes equals moving the threshold (Thm 1) | [Elkan 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf) | The paired recall-up, precision-down results are what a threshold move predicts. Compare weighted and unweighted at matched FP/scan |
| Lesion F1 (steps 4-6); false-positive classifier | Calibrated scores: F1 is maximal predicting positive iff score ≥ F*/2, F* the optimal F1 | [Lipton 2014](https://arxiv.org/abs/1402.1892), Cor. 1 | At F1 near 0.7 the optimal threshold is near 0.35. Pick each arm's threshold on validation |
| Toolkit: Dice without smoothing | Per-image Dice with negligible ε gives near-zero gradient on empty-label patches: they are ignored. Batch reduction (B > 1), or per-image with ε near the expected foreground size, gives them a gradient | [Tilborghs 2022](https://arxiv.org/abs/2207.09521), Sec. 2.2-2.3 | The arm changes how empty and near-empty patches train. Record reduction (image or batch) and ε for both arms |
| "Intervals overlap" (toolkit); step 4 paired bootstrap | Rejection by CI overlap implies rejection by the standard test, not the converse. Paired: Var(A − B) = Var A + Var B − 2 Cov(A, B), Cov > 0 on shared test cases | [Schenker & Gentleman 2001](https://doi.org/10.1198/000313001317097960) | Overlap means undecided, not no effect. Decide arms by the paired-difference interval |
| Step 6: 5 folds | No unbiased estimator of K-fold CV variance exists (Thm 6): fold scores share training data. Corrected resampled t-test: variance (1/J + n₂/n₁)σ̂², heuristic | [Bengio & Grandvalet 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html); [Nadeau & Bengio](https://papers.nips.cc/paper/1661-inference-for-the-generalization-error) | A plain t-test over 5 fold scores understates variance; use the corrected test |
| Finding 3, step 5: training data, external test | Target error ≤ source error + ½ d_HΔH(source, target) + λ, λ the best joint hypothesis's error (Thm 2, plus sample terms). α-weighted source + small target sample: optimal α set by sample sizes and divergence (Thm 3) | [Ben-David 2010](https://doi.org/10.1007/s10994-009-5152-4) | An added dataset helps the held-out domain only by lowering source error or divergence to it |
| Toolkit: pretrained encoder, multi-dataset | Shared representation, linear head: target excess risk ≲ C(Φ)/(n₁T) + k/n₂; frozen representation, diverse source tasks. Multi-task transfer is positive when task parameters and covariances align, negative otherwise | [Du 2021](https://arxiv.org/abs/2002.09434), Thm 5.1; [Wu 2020](https://arxiv.org/abs/2005.00944), Thm 2 | Gain shrinks as target labels grow, consistent with +3 Dice at 10 cases and near zero at full data. Run step 6 at two or three label budgets |
| Step 8: fold ensemble + mirror TTA | Squared loss, arithmetic mean: ensemble error = mean member error − mean disagreement. Cross-entropy: ≤ mean member error (Jensen). Mirror TTA is an ensemble of flipped passes | [Krogh & Vedelsby 1995](https://proceedings.neurips.cc/paper_files/paper/1994/file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf), eq. 10 | Gain over the average member equals disagreement; measure it on validation before paying 8x inference |

## Skip

Other architectures (SwinUNETR, Mamba, MedNeXt at 2x compute); classical hyperparameter search; synthetic nodules (LeFusion 78.26 to 78.77 like-for-like); promptable models as standalone segmenters.

## Not known

- det3d boxes fed to a promptable segmenter vs one nnU-Net: no published result.
- Organ cropping (FRAN `lbd`): no controlled comparison in liver, pancreas, kidney, colon.
- Reader ceiling for small nodules: LIDC readers agree at median Dice 0.82, not given by size ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464594)); distance of 52.5 Dice from it unknown.

## Licences (commercial translation)

- Non-commercial weights: LesionLocator, nnInteractive (CC BY-NC-SA 4.0); MedSAM2 (research and education only). Code open.
- Non-commercial data: NSCLC-Radiomics (CC BY-NC 3.0).
- Commercial use permitted: LIDC-IDRI, NSCLC-Radiogenomics (CC BY 3.0); NLST public subset (CC BY 4.0); TotalSegmentator open subtasks (Apache-2.0).
- Licence not found on an official page: DLCS, LNDb, MSD Lung, LUNA16, ULS23, NLSTseg, DeepLesion. Check before training a model meant for release.
