# 3D lung nodule / lesion detection on CT: SOTA notes (as of Nov 2026)

Coverage note: this pass used about 13 tool calls. Only claims opened by a page fetch are under "Cited Findings". Claims seen only in search snippets are marked [snippet-only]. Large parts of the brief (DeepLesion, ULS23, PN9, pretraining, Sybil, DeepMind, YOLO/CenterNet variants, WBF/TTA ablations) were NOT covered and are listed under Gaps.

## Frameworks and who leads on LUNA16 / LUNA25 / LIDC / NLST

### Takeaway
nnDetection (Retina U-Net, self-configuring) reports CPM 0.930 on LUNA16 detection with no FP-reduction stage; only a method with a dedicated FP-reduction stage scored higher in its table (0.952). The MONAI 3D RetinaNet bundle is the Apache-2.0, 16 GB-GPU alternative. No 2025-2026 LUNA16 detector I could open beats these on a comparable protocol.

### Cited Findings
- nnDetection LUNA16 sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan = 0.812, 0.885, 0.927, 0.950, 0.969, 0.979, 0.985; CPM 0.930. The paper says it outperforms all competitors on candidate detection; only Liu et al. 2019 + FPR is higher (CPM 0.952), attributed to its FP-reduction stage. (2021) — [ar5iv nnDetection](https://ar5iv.labs.arxiv.org/html/2106.00817)
- nnDetection design: adaptive training sample selection (ATSS); dropped the "anchor centre inside GT box" rule (it removed all positives for small objects); 27 anchors per position; anchor sizes optimised by IoU on the training split (3 sizes per axis); Weighted Box Clustering to merge ensemble/TTA predictions; 5-fold CV with the 5 models ensembled at test time. — [ar5iv nnDetection](https://ar5iv.labs.arxiv.org/html/2106.00817)
- nnDetection compute and ops: needs >=11 GB VRAM; about 2 days training with AMP on an RTX 2080 Ti (about 4 days without AMP, about 1 day on high-end GPUs); Linux only; no multi-GPU; Python 3.8+, PyTorch 1.x (2.0 discouraged in the README I saw); needs segmentation masks (boxes can be converted); has a Luna and an LIDC guide; evaluation uses AP at IoU 0.1 and FROC. — [nnDetection GitHub](https://github.com/MIC-DKFZ/nnDetection)
- MONAI 3D RetinaNet lung-nodule bundle: LUNA16 resampled to 0.703125 x 0.703125 x 1.25 mm; 192x192x80 patches; BCE (classification) + L1 (box) loss; Adam lr 1e-2; AMP; >=16 GB GPU for training; fold 0 COCO metrics mAP 0.852, AP@IoU0.1 0.858, AR@IoU0.1 1.0; Apache-2.0; non-deterministic training (max pooling). — [MONAI bundle card (HF mirror)](https://huggingface.co/monai-test/lung_nodule_ct_detection)
- Search snippet claims the MONAI card reports FROC sensitivity 0.957 at 1 FP/scan versus 0.950 for nnDetection on the bundle authors' own split. I did NOT find these figures on the fetched card, so [unverified]. Even if present, the split differs from official 10-fold CV and the numbers are not comparable. — [NGC catalog listing](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection)
- A 2026 TASK Quarterly paper reportedly finds 3D RetinaNet beats 2D RetinaNet and YOLO variants on LUNA16. [snippet-only; the PDF would not parse, no numbers verified] — [TASK Quarterly](https://journal.mostwiedzy.pl/TASKQuarterly/article/download/3934/2689/15750)
- Lung-DETR (Deformable DETR) on LUNA16 reports F1 94.2% (recall 95.2%, precision 93.3%) but works on 2D maximum-intensity projections, not 3D, and reports no CPM. [snippet-only] — [arXiv 2409.05200](https://arxiv.org/abs/2409.05200v1)
- A Jan 2026 segmentation-guided hybrid (segmentation + 2.5D refinement head + DenseNet-Swin classifier) reports LUNA16 CPM 0.944. [snippet-only; protocol (with/without FP-reduction, CV folds) not verified] — [PMC12840476](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12840476/)
- LUNA25: the challenge is malignancy-risk estimation of indeterminate nodules, not detection. A Radiology: AI paper reports the selected AI system beat the mean of 65 radiologists (AUC 0.78 vs 0.70, P=.001); the record is a preproduction copy so figures may change. Team MEVIS is listed as best overall in a newsletter, but the place mapping is unclear. [snippet-only] — [Radiology: AI](https://pubs.rsna.org/doi/10.1148/ryai.260179); [grand-challenge newsletter](https://grand-challenge.org/blogs/autumn-newsletter-2025/)
- Large ensemble model (CADe/CADx), trained on 25,709 low-dose CTs with 69,449 annotated nodules; AUC 0.98 internal, 0.945 external; no FP-rate sensitivities stated on the abstract page; no code link. — [arXiv 2512.00281](https://arxiv.org/abs/2512.00281)

### Inferences
- For a team with a simple in-house detector, the cheapest proven upgrades are the nnDetection ingredients: ATSS-style matching that does not require the anchor centre inside the box, per-dataset anchor sizes, and WBC over ensemble/TTA. Compute cost is days on one 11 GB GPU.
- CPM 0.930 vs 0.944/0.952 differences mostly reflect whether a second-stage FP-reduction classifier is used; compare like with like.

### Gaps
- No LUNA16 leaderboard fetched (grand-challenge.org pages not opened); no verified 2025-2026 detection-track SOTA. LNDb, PN9, LIDC-protocol numbers not found.
- MedYOLO / 3D YOLO, CenterNet-3D, CPM-Net, SANet, FCOS-3D, Focused Decoder, nnDETR, TransDETR numbers: not opened.

## Detection networks vs segmentation-based detection; head-to-head benchmarks

### Takeaway
The only head-to-head evidence opened: nnDetection beat a strengthened nnU-Net ("nnU-Net Plus") on 9/12 datasets in CV and 5/7 on hold-out; a 2025 breast MRI study shows plain nnU-Net-as-detector has high sensitivity but many FPs, fixable by adaptation. No 2025-2026 lung-specific comparison found.

### Cited Findings
- nnDetection vs nnU-Net Plus: better on 9/12 datasets in cross-validation; on hold-out better on 5/7 (substantial margins on 4, substantially worse on colon); 4 of the 7 hold-out datasets were in nnU-Net's development pool, so the baseline was not truly blind. — [arXiv 2106.00817 via search summary of abstract/paper](https://arxiv.org/pdf/2106.00817) (numbers taken from the search tool's reading of the paper; I did not re-read them in the fetched ar5iv extract)
- MIDL 2025 breast MRI (not lung): nnU-Net sensitivity 83.8% at 3.334 FP/case; coarse-resolution adapted nnU-Net 79.9% at 0.666 FP/case; nnDetection better in the high-sensitivity region, adapted nnU-Net better at low FP. — [MLR proceedings](https://proceedings.mlr.press/v301/geissler26a.html) [snippet-only]
- A blog (pycad.co) claims RadYOLO beats nnDetection on 4/5 datasets, with LUNA16 mAP@0.1 0.71 (nnDetection) vs 0.69 (RadYOLO small). [unverified secondary] — [pycad.co](https://pycad.co/?p=8685)

### Inferences
- Segmentation-as-detection is viable but needs FP control (coarser resolution, component-level filtering or a 2nd-stage classifier); transfer to lung is untested here.

### Gaps
- No lung-specific detection vs segmentation comparison, no 2025-2026 benchmark paper, no DeepLesion/ULS23 result found.

## Training techniques (matching, mining, losses, sampling, fusion, weak labels)

### Takeaway
Only nnDetection's matching/anchor/WBC choices were verified. Everything else is a gap.

### Cited Findings
- See nnDetection design bullets above (ATSS, 27 anchors/position, optimised anchor sizes, WBC for ensemble/TTA). Hard-negative mining is mentioned only generically; TTA settings and loss specifics for LUNA16 not given in the extract. — [ar5iv nnDetection](https://ar5iv.labs.arxiv.org/html/2106.00817)
- MONAI bundle uses BCE + L1 with 192x192x80 patches at 0.70x0.70x1.25 mm. — [MONAI bundle card](https://huggingface.co/monai-test/lung_nodule_ct_detection)

### Gaps
- Focal-loss settings, hard-negative mining, balanced lesion-centred sampling ablations, multi-window input, FP-reduction second stage gains, box-from-mask, RECIST/point/report weak supervision: not researched.

## Pretraining and vision-language grounding

### Takeaway
Not covered with verified evidence. One related item: a reproducible multi-dataset benchmark shows pretraining on detection-derived candidate patches helps malignancy classification.

### Cited Findings
- Benchmark (v5, Feb 2026) across DLCS, LUNA16/LIDC, NLST-3D, LUNA25: ResNet50 with "Strategic Warm-Start" (pretrained on detection-derived candidate patches) reached AUC 0.71 (DLCS), 0.90 (LUNA16), 0.81 (NLST-3D), 0.80 (LUNA25); compared with random init, Models Genesis, Med3D and a cancer-biomarker foundation model. This is classification, not detection. — [arXiv 2405.04605](https://arxiv.org/abs/2405.04605)

### Gaps
- nnSSL/OpenMind, SuPreM, TotalSegmentator, VoCo, CT-FM, CT-CLIP/Merlin detection gains, and report-supervised grounding detectors: not researched.

## Large-scale screening models

### Takeaway
Only data-scale facts verified; Google/DeepMind and Sybil not opened.

### Cited Findings
- Training-data dependence is large: detectors trained on DLCS reached 0.72 sensitivity at 2 FP/scan on external NLST-3D vs 0.64 for LUNA16-trained detectors (p<0.001). The page does not name the detector or give code. — [arXiv 2405.04605](https://arxiv.org/abs/2405.04605)
- Large Ensemble Model: 25,709 LDCT scans, 69,449 nodules, ensemble of shallow DL and feature-based models. — [arXiv 2512.00281](https://arxiv.org/abs/2512.00281)

### Inferences
- A cheaper win than architecture changes may be training on a larger screening-style set (e.g. DLCS) rather than LUNA16 alone.

### Gaps
- Google/DeepMind lung model, Sybil, LUNA25 winner write-ups and tricks: not opened.

## Evaluation practice

### Takeaway
LUNA16 CPM is the mean sensitivity at 7 FP rates; nnDetection also reports AP at IoU 0.1; cross-protocol comparison is the main pitfall.

### Cited Findings
- FROC/CPM: average sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan; LUNA16 = 888 CTs, 1,186 nodules (>=3 mm, agreed by >=3 of 4 radiologists), official 10-fold split. — [MONAI bundle card/NGC snippet](https://huggingface.co/monai-test/lung_nodule_ct_detection) [counts from search snippet]
- Detection-track vs FP-reduction-track results are not comparable (full-scan detection vs provided candidates). — [LUNA16 paper](https://arxiv.org/abs/1612.08012) [snippet-only]
- nnDetection reports AP at IoU 0.1; the LUNA16 hit criterion (centre-in-nodule-radius) was not confirmed in the fetched text. — [nnDetection GitHub](https://github.com/MIC-DKFZ/nnDetection)
- Benchmarking across datasets shows strong dataset dependence of results (see 2405.04605 above).

### Gaps
- Per-size sensitivity practice, centre-in-box vs IoU 0.1 discussion, DeepLesion/ULS23 evaluation protocols: not researched.

---
# SECOND PASS (appended 2026-11-13). Supersedes first-pass items where they conflict.

Rule: every claim below comes from a page or PDF I opened in full text (or an abstract page where stated). "[snippet-only]" = seen only in search results; "[unverified]" = could not open the source.

## Verification of first-pass snippet-only items

### Takeaway
nnDetection-vs-nnU-Net is confirmed from the paper text. The MONAI "0.957 vs 0.950" claim is DROPPED (not found on any card; a Duke paper gives the real MONAI-RetinaNet LUNA16 numbers: CPM 0.94 vs nnDetection 0.93). The 0.944 hybrid is real but not protocol-comparable. LUNA25 detection-style winners are not public in what I could open; only the overview-paper headline numbers are.

### Cited Findings
- nnDetection vs nnU-Net Plus: better on 9/12 datasets in CV (margin >5% on 7; lower only on liver); nnU-Net Basic worse on 11/12; on hold-out better on 5/7 (substantial in 4; lower on colon); 4 of 7 hold-out sets were in nnU-Net's development pool. Object labels for 5 datasets were made by connected components of segmentation labels, objects <3 mm dropped; metric mAP at IoU 0.1; LUNA16 labels reduced to centre point + radius, with circle masks generated for training. — [nnDetection PDF text](https://arxiv.org/pdf/2106.00817)
- nnDetection training recipe (Table in paper): hard negative mining choosing 1/3 positive and 2/3 negative anchors; BCE classification + Generalized IoU regression; segmentation branch with Dice+CE; 60 epochs x 2500 mini-batches, batch size 4, half of each batch forced to contain an object; SGD Nesterov 0.9, lr warm-up 1e-6 to 1e-2 over 4000 iterations, poly decay; anchor sizes by differential evolution (3 per axis). — [nnDetection PDF text](https://arxiv.org/pdf/2106.00817)
- MONAI 0.957 vs 0.950 claim: not on the MONAI HF model page ("No model card") nor on the monai-test card I fetched earlier. DROPPED. — [HF MONAI/lung_nodule_ct_detection](https://huggingface.co/MONAI/lung_nodule_ct_detection)
- Real MONAI-RetinaNet vs nnDetection LUNA16 numbers (Duke benchmark Table 2): LUNA16-trained MONAI RetinaNet ("LUNA16-De", median-performing fold of 10-fold CV) sensitivity 0.84/0.89/0.93/0.96/0.97/0.98/0.99, CPM 0.94; nnDetection 0.81/0.89/0.93/0.95/0.97/0.98/0.99, CPM 0.93 (copied from nnDetection paper); Liu et al. 2019 CPM 0.92. The MONAI figure is a single median fold, not the pooled 10-fold result. — [Duke benchmark PDF text, arXiv 2405.04605](https://arxiv.org/pdf/2405.04605)
- Hybrid (Seg-CADe-CADx, Diseases Jan 2026): CPM 0.944 +/- 0.012 across five folds (the paper elsewhere says 10-fold: internally inconsistent); NLST external CPM 0.903; FP reduction described as lung-mask filtering, hard-negative mining, calibrated thresholds and a 2.5D refinement head; per-size sensitivity at 1 FP/scan: <=6 mm 91.0%, 6-10 mm 95.5%, >10 mm 98.9%; compared with SCPM-Net, Faster R-CNN+OHEM and others, NOT with nnDetection; no code ("on request"); the paper's FP counts and hardware statements are inconsistent. Treat as unreliable. — [PMC12840476](https://pmc.ncbi.nlm.nih.gov/articles/PMC12840476/)
- Dual-stage cross-layer-attention detector (PMC12868601, 2025): CPM 0.878 (sens 0.676/0.793/0.891/0.939/0.948/0.949/0.949), i.e. below nnDetection; Stage 1 focal loss alpha 0.6 gamma 2 plus Smooth L1, anchors 5/10/20 voxels, IoU>0.5 positive, <0.02 negative; stage 2 FP-reduction with 1:3 positive:negative sampling; resample 1 mm iso, window -1200..600 HU; no nnDetection row, hit criterion unstated, no code. — [PMC12868601](https://pmc.ncbi.nlm.nih.gov/articles/PMC12868601/)
- LUNA25 outcome (only what I could open): overview paper in Radiology: AI (24 Jun 2026): training 555 malignant + 5,608 benign nodules from 4,069 NLST baseline scans; external test 156 malignant + 312 benign size-matched nodules from 463 baseline scans of three European screening trials; best AI beat 65 radiologists' mean (AUC 0.78 vs 0.70, P=.001), found 12% more malignant nodules at matched specificity, 20% fewer FPs at matched sensitivity; winner is a German start-up (unnamed); beat 62 of 65 readers; nodules 5-15 mm. The task is malignancy risk, not detection. — [Radboudumc news](https://www.radboudumc.nl/en/research/news/news-items-by-our-research-institute/2026/ai-estimates-cancer-risk-in-difficult-lung-nodules-better-than-radiologists) (headline numbers via search result of the RSNA record, [Radiology: AI](https://pubs.rsna.org/doi/10.1148/ryai.260179) returned 403, so AUC figures are [snippet-only])
- LUNA25 grand-challenge pages list leaderboards (Sanity Check, Closed Testing, Open Development, Post Challenge) and a top-5 presentation on 22 Sep 2025, but contain no winner names or methods in the text I could open. — [LUNA25 announcements](https://luna25.grand-challenge.org/announcements/), [workshop page](https://luna25.grand-challenge.org/miccai2025-workshop/)
- TASK Quarterly 2026 (opened, full PDF text): on its own LUNA16 test split (119 scans, 223 nodules, NOT the official 10-fold), MONAI 3D RetinaNet mFROC 0.805 (sens 0.556 at 1/8, 0.839 at 1, 0.987 at 8 FP/scan) vs hybrid 3D CNN-Transformer mFROC 0.517; inference 12.7 s/scan and 3.9 GB peak (RetinaNet) vs 0.34 s and 1.9 GB (hybrid); 2D RetinaNet/YOLO lower than 3D RetinaNet. — [TASK Quarterly PDF](https://journal.mostwiedzy.pl/TASKQuarterly/article/download/3934/2689/15750)

### Inferences
- The 0.93/0.94/0.944 numbers are all within seed/fold noise of each other; none is shown to beat nnDetection under one protocol.

### Gaps
- LUNA25 winners' architectures and the official leaderboard text: not accessible (Radiology: AI page 403; leaderboards not opened). Not detection anyway.
- Springer s10044-026-01672-5 (hard-negative-mining detector) was behind a login redirect; [unverified].

## Other detector families with numbers (LUNA16 / LIDC / PN9 / LNDb / NLST)

### Takeaway
Published LUNA16 10-fold CPM: nnDetection 0.930; SCPM-Net 0.892; CPM-Net 0.881; SANet 0.874; NoduleNet 0.873. No anchor-free, YOLO or DETR detector I could open beats nnDetection under the same protocol. MedYOLO fails on LIDC nodules; 3D DETR for nodules: none found.

### Cited Findings
- SCPM-Net (anchor-free, sphere representation, MedIA 2022) Table 5, reported 10-fold CV sensitivity at 1/8..8 FP: 74.3, 82.9, 88.9, 92.2, 93.9, 95.8, 96.4, average 89.2 (single stage, no FP reduction). CPM-Net 72.3/83.8/88.7/91.1/92.8/93.4/94.8 avg 88.1; SANet 71.17/80.18/86.49/90.09/93.69/94.59/95.50 avg 87.39 (with FP reduction, 3 stages); NoduleNet avg 87.27; DeepLung 83.4; Dou 83.9. nnDetection is not mentioned in that paper. Code: HiLab-git/SCPM-Net. Training 8 x 1080Ti, 170 epochs, 96^3 crops, 1 mm; inference 8.37 s/scan. — [SCPM-Net ar5iv](https://ar5iv.labs.arxiv.org/html/2104.05215)
- Same paper, single-split re-implementations (not comparable to above): sens at 1/2/8 FP: 3D CenterNet 71.4/75.8/77.5; 3D RetinaNet 83.6/86.8/88.7; 3D RetinaNet++ 88.8/91.3/91.4; CPM-Net 91.2/92.4/92.5; SCPM-Net 92.3/92.7/94.8. So an anchor-free CenterNet-style 3D baseline was clearly worse than RetinaNet there. — [SCPM-Net ar5iv](https://ar5iv.labs.arxiv.org/html/2104.05215)
- Cross-paper reading: nnDetection CPM 0.930 (its paper, same official 10-fold protocol) exceeds SCPM-Net 0.892, CPM-Net 0.881, SANet 0.874 as reported by SCPM-Net's authors. Only protocol-consistent in the sense that all are official-fold LUNA16 candidate-detection numbers; different hit post-processing is not stated. — [nnDetection PDF](https://arxiv.org/pdf/2106.00817); [SCPM-Net ar5iv](https://ar5iv.labs.arxiv.org/html/2104.05215)
- SANet / PN9: SANet (TPAMI 2021) introduced PN9 (8,798 CT scans, 40,439 nodules, 9 classes); code licence non-commercial. [snippet-only; PN9 numbers for nnDetection or newer methods not found] — [SANet project page](https://mmcheng.net/sanet)
- MedYOLO (3D YOLOv5-based): on LIDC (689 train / 173 val scans) it "failed to make progress in any tested configuration", attributed to coarse final feature maps (11^3 or 16^3) limiting small-object capacity; it matched or beat nnDetection (trained on the same splits) on medium/large structures (reported mAP@0.5 vs mAP@0.5:0.9 are not like-for-like). — [MedYOLO PDF text](https://arxiv.org/pdf/2312.07729)
- Lung-DETR (Deformable DETR, 2024): operates on 7.5 mm MIP slices (2D), F1 94.2% (recall 95.2%, precision 93.3%) on LUNA16; no CPM; not a 3D detector. — [arXiv 2409.05200](https://arxiv.org/pdf/2409.05200)
- Hybrid CNN-Transformer 3D detector in the TASK Quarterly study reached only 0.704 sensitivity at 8 FP/scan (see above). — [TASK Quarterly PDF](https://journal.mostwiedzy.pl/TASKQuarterly/article/download/3934/2689/15750)
- Breast-MRI (not lung) evidence on nnDetection vs nnU-Net-as-detector (MIDL 2025): [snippet-only] — [MLR](https://proceedings.mlr.press/v301/geissler26a.html)

### Inferences
- For a simple in-house detector, 3D RetinaNet-type (MONAI) or nnDetection is the evidence-backed baseline; anchor-free CenterNet-style loses to RetinaNet in the only head-to-head I could open.

### Gaps
- No Focused Decoder / nnDETR / TransDETR / FCOS-3D / DeFoD numbers found; no LNDb or PN9 numbers for any modern detector; no 2025-2026 LUNA16 paper opened that includes nnDetection as a row under the same protocol.

## Universal lesion detection (DeepLesion) and ULS23

### Takeaway
DeepLesion detection is a 2D-per-slice benchmark with saturated sensitivity (about 93-95% at 4 FP/image). ULS23 is segmentation, not detection.

### Cited Findings
- DeepLesion sensitivity at 0.5/1/2/4 FP per image (LN-DETR's Table 5; 2D key-slice test): A3D 79.24/85.04/89.15/92.71; LENS 78.60/85.50/89.60/92.50; DKMA 78.48/85.95/90.48/93.48; A3D+SATr 81.03/86.64/90.70/93.30; DiffULD 80.43/87.16/91.20/93.21; P3D* 82.22/87.42/90.91/93.65 (avg 88.55, MS-COCO pretrained); LN-DETR 79.89/87.05/92.00/94.89 (avg 88.46, ImageNet pretrained). — [LN-DETR, ECCV 2024](https://arxiv.org/html/2404.03819v2)
- LN-DETR lymph node 3D detection recall at 0.5/1/2/4 FP/patient: internal 42.48/51.02/60.60/70.96 (avg 56.27 vs Mask DINO 51.32); external 35.26/48.18/57.92/66.78. — [LN-DETR](https://arxiv.org/html/2404.03819v2)
- Self-training with VFNet on an 11.5% DeepLesion subset: sensitivity at 4 FP (IoU 0.5) 78.5% best vs 72.0% baseline; Deformable DETR "stable performance was unattainable"; YOLOX "far inferior"; mmDetection; no code link. Not comparable to full-benchmark numbers. — [arXiv 2504.05207](https://arxiv.org/html/2504.05207v1)
- ULS23: 38,693 training lesions; test 725 lesions/268 patients (+50 validation); baseline nnU-Net v2 ResEnc with pseudo-mask pretraining, test Dice 0.703 overall and lung 0.761; the task segments one lesion given a click at its centre, so no detection metric; code DIAGNijmegen/oncology-ULS23, baseline weights on Zenodo; ChallengeScore 0.729. — [ULS23 paper](https://arxiv.org/html/2406.05231v2)
- 3D ULD + tagging with self-training (arXiv 2504.05201): VFNet 2D on 30% of DeepLesion with 3D proposal mining; mean sensitivity about 46.9% over 0.125-8 FP/image vs 46.8% prior [snippet-only/machine-translated; unverified]. — [arXiv 2504.05201](https://arxiv.org/abs/2504.05201v1)

### Inferences
- The ULD frontier is near ceiling on the 4-FP metric; differences are 1-2 points and depend on pretraining (COCO vs ImageNet). Not informative for 3D nodule decisions.

### Gaps
- No 2025-2026 full-DeepLesion SOTA table opened; DetectDiffuse numbers not found; ULS23 leaderboard winners not opened.

## Pretrained backbones and vision-language detection

### Takeaway
I found NO quantified detection gain from nnSSL/OpenMind, SuPreM, VoCo, CT-FM, TotalSegmentator or Merlin/CT-CLIP for lung-nodule detection. What exists is classification/segmentation, and one warning that generic pretraining can fail.

### Cited Findings
- TAP-CT (DINOv2-style CT ViTs, 104,405 CT volumes): evaluates frozen features, no detection. LUNA16 nodule classification AUC TAP-B-3D 0.876, Curia 0.860; LUNA25 0.855 vs 0.856 (no gain); compared backbones: Curia, CT-FM, SuPreM, VISTA3D, VoCo, FMCIB. Segmentation mean DSC 0.582 vs 0.489 (Curia). — [TAP-CT arXiv 2512.00872](https://arxiv.org/html/2512.00872v1)
- Duke benchmark: for nodule classification on NLST-3D, Models Genesis reached only 0.51 AUC, which the authors read as a risk of generic pretrained weights without adaptation; detection used from-scratch MONAI RetinaNet. — [Duke benchmark PDF](https://arxiv.org/pdf/2405.04605); (that pretrained-vs-scratch classification table is in the PDF; I read its text only partially)
- CT-FM: 148,000 CT scans (IDC), contrastive pretraining, SegResNet; tasks are segmentation, head-CT triage, retrieval, semantic understanding; no detection evaluation on the abstract. — [CT-FM arXiv 2501.09001](https://arxiv.org/abs/2501.09001)
- VoCo: PreCT-160K pretraining set and a 48-task benchmark; the abstract gives no numbers or detection task. — [VoCo arXiv 2410.09890](https://arxiv.org/abs/2410.09890)
- OpenMind/nnSSL: 114k brain MRI volumes, so not CT. [snippet-only] — [OpenMind note](https://en.papernotes.org/ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning/)
- SuPreM: pretrained on AbdomenAtlas 1.1 (9,262 CTs, 25 organs, 7 tumour types); downstream examples are segmentation and pancreatic tumour detection; no lung nodule results found. [snippet-only] — [SuPreM GitHub](https://github.com/MrGiovanni/SuPreM)
- CT-CLIP / Merlin are volume-level report-aligned models; no CT lesion-detection benchmark found; a 2026 paper adds a localization objective for grounding [snippet-only]. — [search results](https://arxiv.org/pdf/2602.14879)

### Inferences
- Any gain from these backbones for a nodule detector is an untested hypothesis; the team would have to run its own A/B (scratch vs pretrained encoder inside the same RetinaNet).

### Gaps
- No paper opened that plugs these backbones into a detector and reports FROC on LUNA16/DLCS. Report-supervised CT detectors: none verified.

## Screening-scale models

### Takeaway
Screening systems train on NLST-scale data (tens of thousands of scans) and use bounding-box supervision for attention or detection; data scale and annotation quality matter more than architecture in the evidence I could open.

### Cited Findings
- Google (Ardila 2019): 94.4% AUC on 6,716 NLST cases, similar on an independent 1,139-case set; end-to-end 3D model on whole volumes with an ROI detection model feeding a risk model; without priors it beat all six radiologists (11% fewer FPs, 5% fewer FNs). Training-set size and detector type not on the page I opened. — [Google Research page](https://research.google/pubs/end-to-end-lung-cancer-screening-with-three-dimensional-deep-learning-on-low-dose-chest-computed-tomography/)
- Sybil: 3D ResNet-18 (pretrained), attention-guided pooling, ensemble of 5; NLST train 28,162 / dev 6,839 / test 6,282 scans; radiologists drew bounding boxes on scans of participants diagnosed within 1 year, used to guide attention in training only; 1-year AUC 0.92 (NLST), 0.86 (MGH), 0.94 (CGMH); 6-year C-index 0.75/0.81/0.80; code, splits, annotations, weights public (reginabarzilaygroup/Sybil). — [Sybil PMC10419602](https://pmc.ncbi.nlm.nih.gov/articles/PMC10419602)
- Large Ensemble Model: >23,000 LIDC+NLST training scans; with nnDetection retrained on NLST as the comparator: 82.0% sens at 0.5 FP and 84.9% at 1 FP on NLST Test1 vs 99.3% for the authors' model (evaluation there is on a malignant-nodule CADe task, a different, easier target than all-nodule FROC); the paper says nnDetection trained on LUNA16 had a slight advantage over its CADe on LUNA16. Also compares AUC with Ardila 0.962, Sybil 0.94 (0.92 reported), Liao et al. 0.946. No code link. — [arXiv 2512.00281v2](https://arxiv.org/html/2512.00281v2)
- DLCS (Duke): 1,613 patients, 2,487 nodules with 3D boxes and pathology outcomes; annotation was bootstrapped by MONAI RetinaNet and verified by a medical student under radiologists; nodules >=4 mm per Lung-RADS 2022; 88% dev / 12% test; code and models public (GitHub fitushar/AI-in-Lung-Health-Benchmarking...; Duke GitLab); data on Zenodo. — [Duke benchmark PDF](https://arxiv.org/pdf/2405.04605)
- DLCS vs LUNA16 training (same MONAI RetinaNet, 192x192x80 patches, 0.7x0.7x1.25 mm, HU -1000..500): on DLCS internal test sens at 2 FP 0.82 vs 0.72, average sens 0.64 vs 0.57; on external NLST-3D sens at 2 FP 0.72 vs 0.64, average 0.58 vs 0.49 (paired gains 0.083 and 0.093, P<.001). Note: an earlier arXiv version quoted by a search snippet shows different values (0.82/0.62 at 2 FP, CPM 0.63/0.45), so numbers changed between versions; I cite the latest PDF text. — [Duke benchmark PDF](https://arxiv.org/pdf/2405.04605)

### Inferences
- Training on clinically curated, all-nodule annotations from a screening cohort beats LUNA16-only training by 8-10 points sensitivity at 2 FP on external screening data. This is the single largest measured lever in the evidence I opened.

### Gaps
- DeepMind successor models, Sybil detection component details beyond the above, LUNA25 winners' methods: not accessible.

## Training tricks with evidence

### Takeaway
Only recipe-level evidence (no ablations) was opened: nnDetection's design, one focal loss setting, and weakly supervised / synthetic-data items that are classification or segmentation, not 3D detection.

### Cited Findings
- Hard-negative mining + forced-foreground batches (nnDetection recipe above): 1/3 positive, 2/3 negative anchor sampling; half of each batch forced to contain an object. No ablation in the paper text I read. — [nnDetection PDF](https://arxiv.org/pdf/2106.00817)
- Focal loss alpha 0.6, gamma 2 (+Smooth L1) with 1:3 pos:neg FP-reduction sampling in a 2025 two-stage detector whose CPM was 0.878; no ablation extracted. — [PMC12868601](https://pmc.ncbi.nlm.nih.gov/articles/PMC12868601/)
- Two-stage FP reduction: in the SCPM-Net comparison, single-stage anchor-free methods (CPM-Net, SCPM-Net) reach average sens 88.1 and 89.2 without FP reduction vs multi-stage 83-87 for earlier pipelines; LUNA16's own challenge paper separates candidate detection and FP-reduction tracks. — [SCPM-Net ar5iv](https://ar5iv.labs.arxiv.org/html/2104.05215); [LUNA16 paper](https://arxiv.org/pdf/1612.08012)
- Weighted Box Clustering (nnDetection) and weighted boxes fusion for ensembles: an ensemble of one-stage detectors with weighted boxes fusion reported 91.69% sens at 4 FP/image on DeepLesion [snippet-only] — [arXiv 2111.04886](https://arxiv.org/pdf/2111.04886)
- Weak labels: RECIST-derived dense masks with RetinaNet gave 90.77% sens at 4 FP/image on DeepLesion (2019) [snippet-only]; DeepEM trains on image-level NLST labels with a fully supervised LUNA16 start (2018, no sensitivity table seen); a 2024 TCSVT paper states point-label methods still trail fully supervised. — [arXiv 1906.02283](https://ar5iv.labs.arxiv.org/html/1906.02283); [arXiv 1805.05373](https://arxiv.org/pdf/1805.05373v1); [UCL TCSVT 2024](https://discovery.ucl.ac.uk/10194835/1/GuangyuHuang-TCSVT-2024.pdf)
- Synthetic nodules: LeFusion (ICLR 2025, 3D, LIDC lung nodules) improves nnU-Net and SwinUNETR segmentation (nodule Dice 78.77 to 80.62 in one setting, per third-party summary); no detection result; code M3DV/LeFusion. A 2023 semantic-diffusion 2D study reports +3.96% patch detection accuracy and +8.5% AP50 on LUNA16-derived 2D images. No 3D detection sensitivity gain from synthetic nodules found. — [LeFusion](https://arxiv.org/html/2403.14066v2); [Zhao & Hou 2023](https://arxiv.org/abs/2305.01138v1) (both [snippet-only])
- Annotation-source effect (DLCS vs LUNA16 above) is a measured data-side trick: +0.083 to +0.099 sens at 2 FP.

### Inferences
- Defaults worth adopting without ablation evidence: lesion-forced batches, 1:2 pos:neg anchor mining, GIoU regression, ensembling with WBC.

### Gaps
- No controlled ablation of hard-negative mining, focal-loss parameters, lesion-centred sampling, multi-window input, or TTA on a 3D lung detector was opened.

## Evaluation practice

### Takeaway
LUNA16 hit = candidate within the reference nodule's radius of its centre; CPM = mean sensitivity at 1/8..8 FP/scan; many papers do not state their hit rule.

### Cited Findings
- LUNA16 hit rule: a CAD mark is a true positive if located within distance r of the centre of any reference nodule, with r = that nodule's radius; multiple marks on one nodule keep the highest score; marks on irrelevant findings are neither TP nor FP; FROC analysis. — [LUNA16 paper](https://arxiv.org/pdf/1612.08012)
- CPM = average sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan; official LUNA16 applies an exclusion list of irrelevant findings, which DLCS/NLST-3D evaluations do not (so numbers are lower and harder). — [Duke benchmark PDF](https://arxiv.org/pdf/2405.04605)
- Box-based papers use mAP/AR at IoU 0.1 (nnDetection; TASK Quarterly), a much looser criterion than IoU 0.5 and not equal to the LUNA16 radius rule; 2D slice vs 3D evaluation changes numbers materially. — [nnDetection PDF](https://arxiv.org/pdf/2106.00817); [TASK Quarterly PDF](https://journal.mostwiedzy.pl/TASKQuarterly/article/download/3934/2689/15750)
- Per-size sensitivity reporting is rare: only the unreliable hybrid paper gave it (91.0/95.5/98.9% for <=6 / 6-10 / >10 mm at 1 FP). SCPM-Net reports none. — [PMC12840476](https://pmc.ncbi.nlm.nih.gov/articles/PMC12840476/); [SCPM-Net ar5iv](https://ar5iv.labs.arxiv.org/html/2104.05215)
- Pitfalls observed: reporting a single median fold as "LUNA16 result" (Duke, MONAI); own-split vs official folds (TASK Quarterly); published tables copy numbers across papers; version drift between arXiv versions (Duke); internal inconsistencies (hybrid paper).

### Gaps
- Exact IoU-based hit rules for DLCS/NLST-3D evaluation not extracted from the Duke text.

---
## Summary status (second pass)
Verified: nnDetection numbers, recipe and nnU-Net comparison; MONAI RetinaNet vs nnDetection (CPM 0.94 vs 0.93, single median fold); SCPM/CPM/SANet CPM 0.892/0.881/0.874; MedYOLO LIDC failure; Lung-DETR is 2D; DeepLesion table; ULS23 is segmentation; DLCS-vs-LUNA16 training effect; Sybil/Ardila/LEM facts; LUNA16 hit rule and CPM.
Dropped: MONAI 0.957 claim. Downgraded: 0.944 hybrid.
Open: 3D DETR nodule detectors, PN9/LNDb numbers, pretrained-backbone detection gains, controlled ablations of training tricks, synthetic-nodule detection gains, LUNA25 winners' methods, DeepLesion 2025-2026 SOTA.
