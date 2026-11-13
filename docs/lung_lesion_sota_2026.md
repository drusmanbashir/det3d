# Lung lesions on CT: what to do, and what carries to other organs

Date: 2026-11-13. Every figure was read in the linked source. F = `/home/ub/code/fran/fran`, D = `/home/ub/code/det3d/det3d`.

## Findings

1. On lung, nothing from 2024-2026 beats nnU-Net ResEnc-L for segmentation or nnDetection for detection. FRAN and det3d hold both models; the gap is the recipe around them.
2. The recorded detector score is not usable. `/r/nnunet/eval/Dataset803_lungs/step2_froc/report.txt` gives CPM 0.060 against a published 0.930 ([nnDetection](https://arxiv.org/abs/2106.00817)). Its FROC is a straight line (precision about 0.08 at 1, 4, 8 and 16.9 FP/scan), so the score does not separate hits from misses. That fits displaced boxes or scores not comparable across patches. Fix this first.
3. Training data is the largest measured lung lever: sensitivity at 2 FP/scan 0.72 against 0.64 on external NLST when the same RetinaNet trains on DLCS instead of LUNA16 ([Duke](https://arxiv.org/pdf/2405.04605)).
4. Small lesions are the hard part: Dice 52.5 for 3-10 mm against 77.1 above 1 cm ([Mathai](https://arxiv.org/pdf/2504.06924)). Report every result by size.
5. MSD Lung (63 scans) cannot settle anything: two baselines in one paper differ by 3.9 Dice on it ([MultiTalent](https://arxiv.org/html/2303.14444v2)). Lung results on pretraining, multi-dataset training and automated search are undecided, not negative.
6. The lung answer does not carry to other organs (see toolkit table).

## Do, in order

| # | Step | Decided by |
|---|------|-----------|
| 1 | Audit the detector score: overlay boxes on GT in world coordinates; compare scores of matched and unmatched boxes; then re-score with the LUNA16 radius rule | CPM on the 100 LIDC validation cases |
| 2 | Run upstream `/home/ub/code/nnDetection/projects/Task016_Luna` unmodified; score with upstream and local scorers | CPM near 0.930 |
| 3 | det3d full recipe: 5 folds, Weighted Box Clustering, warm-up then poly schedule (now: one fold, plateau schedule, no TTA) | CPM and sensitivity at 1 FP/scan by size |
| 4 | FRAN ResEnc-L with the nnU-Net schedule: SGD, poly decay, 1,000 epochs (now: plain U-Net, Adam, early stopping) | Lesion F1, strata under and over 10 mm, paired bootstrap |
| 5 | Add data: DLCS for detection; LNDb and MSD Lung for segmentation; hold one set out as external test | Sensitivity at 2 FP/scan and lesion F1 on the held-out set |
| 6 | Pretrained against scratch on `Dataset803_lungs`, same plan and epochs | Lesion F1, 5 folds |
| 7 | Sampling arms (next section) | Sensitivity at fixed FP/scan by size |
| 8 | Fold ensemble and mirror TTA for final numbers (ULS23: 0.703 to 0.715 from TTA, [paper](https://arxiv.org/html/2406.05231v2)) | Same metrics as steps 3-4 |

## Toolkit for new organs

| Tool | Evidence | Use |
|------|----------|-----|
| nnU-Net as the detector (score connected components) | Pancreas: only three PANORAMA teams beat the nnU-Net baseline ([paper](https://arxiv.org/abs/2503.10068)). Mediastinal nodes: recall at 2 FP 75.38 against 68.01 for a Swin detector ([paper](https://arxiv.org/abs/2409.06324)). Liver and colon: ahead of nnDetection ([nnDetection](https://arxiv.org/abs/2106.00817)) | Score it beside det3d for every new organ |
| Query-based detector for lymph nodes | nnDetection lowest of nine: recall 38.03 against 56.27 for LN-DETR, 1,067 patients, baseline run by the authors ([LN-DETR](https://arxiv.org/abs/2404.03819)) | Lymph nodes only, after a local nnDetection baseline |
| Multi-dataset training on partial labels | Against the same architecture per dataset: colon +11.29, pancreas +3.53, lung +2.27, liver tumour -1.03 ([MultiTalent](https://arxiv.org/html/2303.14444v2)) | When a second organ's lesion labels join; expect gains on colon and pancreas |
| Pretrained encoder | Against scratch: colon +7.1, pancreas +3.3, lung -1.8, liver -0.2, intervals overlap; about +3 Dice at 10 cases ([nnFoundation](https://arxiv.org/html/2609.26924v1)) | Organs with few labelled scans |
| Larger patch and batch | autoPET II: patch 128³ to 192³, Dice 33.28 to 44.19; batch 2 to 12, 28.07 to 48.51; compute not matched ([paper](https://arxiv.org/pdf/2309.13747)) | Cheap arm on any organ |
| Dice loss without the smoothing term | autoPET III winner, matched pairs: 66.08 to 67.01 and 66.37 to 67.45 ([LesionTracer](https://arxiv.org/pdf/2409.09478)) | One-line arm on any organ |
| Instance-weighted loss | Node sensitivity 0.818 to 0.849, tumour precision 0.813 to 0.636 ([paper](https://arxiv.org/pdf/2604.10130)). LiTS recall 0.62 to 0.72, object Dice 0.63 to 0.49 ([paper](https://arxiv.org/pdf/2007.10033)) | Only where recall matters more than precision |
| Second-stage false-positive classifier | Lung, lesions of 10 mm or more: F1 0.680 to 0.712, intervals overlap ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361585/)) | After the detector is sound |

## Sampling

Already present: det3d's `RetinaUNetV001` mines hard negative anchors (`HardNegativeSamplerBatched`). FRAN's `RandCropByFlatIndicesd` picks crop centres uniformly over foreground voxels, so a lesion with eight times the volume is chosen eight times as often.

No matched test of any sampler exists on CT. The matched results elsewhere:

| Change | Result |
|--------|--------|
| Pick a lesion first, then a voxel in it | Brain MRI: sensitivity 0.898 to 0.932, FP/patient 8.3 to 6.6, Dice flat ([Krivov](https://arxiv.org/pdf/1808.00244)) |
| Replay mined hard negatives | DeepLesion: AP 43.0 to 51.9 once unannotated lesions were separated out; 34.2 to 17.3 when they were not ([Lesion Harvester](https://arxiv.org/pdf/2001.07776)) |
| Less foreground in crops | PET/CT: Dice flat, false-positive volume 11.9 to 23.8 at 10% foreground ([paper](https://arxiv.org/pdf/2309.13675)) |

Lung precedent without a matched comparison: DSB2017 winners ([grt123](https://github.com/lfz/DSB2017), [de Wit](https://juliandewit.github.io/kaggle-ndsb2017/)) and [CASED](https://arxiv.org/html/1807.10819).

Trial arms, one change each, after steps 1-4:

1. FRAN lesion-first centres.
2. FRAN false-positive replay: 75% foreground, 12.5% ordinary background, 12.5% mined background.

Risks: a mined false positive may be an unannotated lesion, so review the pool and cap it per scan; lesion-first sampling favours many tiny nodules over large-lesion boundaries.

## Skip

Other architectures (SwinUNETR, Mamba, MedNeXt at twice the compute), classical hyperparameter search, synthetic nodules (LeFusion 78.26 to 78.77 like-for-like), promptable models as standalone segmenters.

## Not known

- Whether det3d boxes fed to a promptable segmenter beat one nnU-Net: no published result.
- Whether organ cropping (FRAN `lbd`) helps: no controlled comparison in liver, pancreas, kidney or colon.
- How close 52.5 Dice for small nodules is to the reader ceiling: LIDC readers agree at median Dice 0.82, not given by size ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464594)).

## Licences

Usable with these caveats for commercial translation:

- Non-commercial weights: LesionLocator and nnInteractive (CC BY-NC-SA 4.0), MedSAM2 (research and education only). Their code is open.
- Non-commercial data: NSCLC-Radiomics (CC BY-NC 3.0).
- Commercial use permitted: LIDC-IDRI and NSCLC-Radiogenomics (CC BY 3.0), NLST public subset (CC BY 4.0), TotalSegmentator open subtasks (Apache-2.0).
- Licence not found on an official page: DLCS, LNDb, MSD Lung, LUNA16, ULS23, NLSTseg, DeepLesion. Check before training a model meant for release.
