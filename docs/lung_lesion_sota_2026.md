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
