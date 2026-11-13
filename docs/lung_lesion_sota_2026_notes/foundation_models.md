# Foundation models and large-scale pretraining for 3D CT (lung tumour / lesion) segmentation, as of Nov 2026

Scope note: about 15 tool calls. Claims below come from pages I opened (WebFetch) or, where marked "search snippet", from search-result text only. Several key-question items were not reached (see Gaps in each section).

## Self-supervised / large-scale 3D pretraining (nnSSL, OpenMind, nnFoundation, MAE, VoCo, SuPreM, others)

### Takeaway
MAE-style reconstruction pretraining with a ResEnc-L CNN is the best-supported scheme (OpenMind benchmark and nnFoundation); gains over a well-trained from-scratch ResEnc-L are modest (about 0.4 to 3 Dice points averaged) but larger with few labels. VoCo and SwinUNETR-style pretraining gave little or no gain. No source I opened reports a lung-tumour-specific gain for nnFoundation or OpenMind.

### Cited Findings
- nnFoundation exists: arXiv 2609.26924, "nnFoundation: 3D Foundation Models for Radiology", v1 22 Sep 2026, CC BY 4.0 paper, 83 authors incl. Maier-Hein and Isensee. Two models (CNN and transformer), pretrained on 2.1 M CT/MRI/PET volumes from 125 institutional and public datasets, evaluated on 108 tasks; abstract says models are released integrated into nnU-Net and nnDetection. — [arXiv abs](https://arxiv.org/abs/2609.26924)
- nnFoundationCNN: over 34 segmentation datasets, best average, +3.0 Dice vs next-best baseline (CT-FM) and +1.9 vs nnU-Net; the only foundation model to exceed nnU-Net. Dynamic weight adaptation (nnU-Net-planned topology) added +1.2 Dice on average, better or equal on 27/34 datasets. Example MSD Pancreas: scratch static 56.5, pretrained fixed topology 61.8, ResEnc-L scratch 63.7, pretrained dynamic 67.0. — [arXiv HTML v1](https://arxiv.org/html/2609.26924v1)
- nnFoundation OOD: CNN 74.9 OOD Dice (84.4 ID, 88.8% retention) vs scratch 70.4 OOD (87.4%); Merlin 70.5; CURIA 62.7. — [arXiv HTML v1](https://arxiv.org/html/2609.26924v1)
- nnFoundation low-data: on six segmentation datasets, +3 Dice average over scratch with only 10 training cases, largest gain at smallest sizes. With 150 instead of 1,000 epochs (15% compute) CNN kept 99.16% of full performance and beat all baselines including full-schedule scratch. — [arXiv HTML v1](https://arxiv.org/html/2609.26924v1)
- nnFoundation fine-tune recipe: nnU-Net framework, 1,000 epochs x 250 steps, batch 2, 192^3 patches for fixed topology (nnU-Net-planned patch for dynamic), nnU-Net target spacing per dataset, z-score; encoder weights only, decoder random; AdamW for pretrained (SGD for the 1,000-epoch CNN runs), peak LR 1e-3, linear warm-up over first 15% of epochs then poly decay; LR reduced 10x if unstable; downstream budget a single A100 40 GB. — [arXiv HTML v1](https://arxiv.org/html/2609.26924v1)
- nnFoundation pretraining details (secondary, search snippet and a downstream MIST PR, not the paper text I read): MAE mask ratio 0.75, 16^3 masked patches; checkpoint records native resolution (median spacing 1 mm), z-score, 192^3 patches. — [MIST PR 122](https://github.com/mist-medical/MIST/pull/122)
- nnFoundation failure case: MAE reduced the CNN's classification performance vs scratch (reconstruction/global-semantics mismatch). ViT variant best for retrieval/report generation and frozen-feature settings. — [arXiv HTML v1](https://arxiv.org/html/2609.26924v1)
- OpenMind / nnSSL benchmark (arXiv 2412.17041, ICCV 2025): 114k 3D brain MRI volumes (CC-BY-4.0 dataset), 7 SSL methods, ResEnc-L and Primus-M, 15 downstream tasks; code MIC-DKFZ/nnssl, checkpoints on Hugging Face (AnonRes ones CC-BY-4.0 per search snippet). — [arXiv HTML v2](https://arxiv.org/html/2412.17041v2); [search result, HF card](https://huggingface.co/AnonRes/ResEncL-OpenMind-VF/blob/main/README.md)
- OpenMind Table 2 average Dice (ID / OOD / all): ResEnc-L scratch 150 ep 62.23 / 87.08 / 68.44; scratch 1k ep 64.15 / 89.43 / 70.47; + MAE 65.11 / 88.30 / 70.91 (best); + S3D 70.36; + MG 70.30; + SimCLR 69.44; + VF 69.29; + VoCo 68.50; + SwinUNETR 67.92; Primus-M + MAE 70.42. VoCo and SwinUNETR gave little/no gain; for Primus-M they fell well below scratch. — [arXiv HTML v2](https://arxiv.org/html/2412.17041v2)
- OpenMind fine-tune recipe: 150 epochs (1,000 for some CNN runs) x 250 steps, batch 2, poly decay from LR 1e-2 (ResEnc-L) or 1e-3 (Primus-M), 15-epoch warm-up per stage (50 in 1k-epoch runs); pretrain 1,000 epochs on 4 x A100 40 GB. Patch/spacing/VRAM not in the text I read. Downstream CT tasks there are AMOS22 and KiTS19 only; lesion tasks are MRI (ISLES, MS, HNTS-MRG, brain mets); no lung CT. — [arXiv HTML v2](https://arxiv.org/html/2412.17041v2)
- COLIPRI (chest CT vision-language, Primus-M encoder; CT-RATE + NLST): segmentation five-fold Dice on a "Lung" dataset 68.98 (COLIPRI-CRM, 37.5k steps) vs 67.12 (MAE) vs 67.56 / 67.46 at 250k steps; paper states MAE drives most dense-task gains and CNN baselines (nnU-Net ResEnc-L) remain stronger on most datasets. Which "Lung" dataset is not stated in the text I read. — [arXiv HTML v2](https://arxiv.org/html/2510.15042v2)
- PET/CT MAE foundation model (arXiv 2605.21835; 4,997 scans from ViMedPET, autoPET IV, DeepPSMA, SPADE; MAE 50% masking): autoPET lesion Dice, nnU-Net scratch vs +MAE: 100% labels 0.6149 vs 0.7160; 30% 0.5525 vs 0.6642; 10% 0.4555 vs 0.6055; best SwinUNETR-v2-base+MAE 0.7205 / 0.6801 / 0.6132. Fine-tune: DiceCE, 100 epochs, LR 1e-4 cosine, batch 4, about 22 GB (SwinUNETR) / 10 GB (nnU-Net). Weights at github.com/liu-xiaofeng/Foundation-Model-for-PET-CT, licence not stated. No lung-specific result. The paper text is inconsistent on 4,997 vs 4,797 scans. — [arXiv HTML v1](https://arxiv.org/html/2605.21835v1)
- SuPreM (supervised pretraining on AbdomenAtlas 1.1, 9,262 CTs, 25 organs + pseudo tumour labels) claimed better than SSL (Tang 2022 on 5,050 CTs; Valanarasu 2023 on 50k CT/MR), especially few-shot; authors' own claim, no tumour Dice numbers retrieved. — [SuPreM GitHub (search result)](https://github.com/MrGiovanni/SuPreM); [arXiv 2501.11253 (search result)](https://arxiv.org/pdf/2501.11253)
- ULS23 baseline: semi-supervised nnU-Net ResEnc ("nnUNet-ResEnc+SS") average Dice 0.703 +/- 0.240 on the challenge test set (all lesion types, not lung-only); a Vanderbilt study reported no ULS23 submission clearly beat the baseline. Search snippets; nnU-Net version stated inconsistently across sources. — [arXiv 2406.05231 (search result)](https://arxiv.org/pdf/2406.05231); [arXiv 2507.19230 (search result)](https://arxiv.org/pdf/2507.19230)

### Inferences
- Averaged gain of MAE-pretrained ResEnc-L over a long-trained scratch ResEnc-L is small on OpenMind (+0.44 Dice overall); larger vs the 150-epoch scratch (+2.5) and in low-label regimes (nnFoundation +3 Dice at 10 cases; PET/CT MAE +10 Dice at 10% labels in autoPET). Pretraining mostly buys compute and label efficiency, not a higher ceiling.
- The autoPET gain is large partly because the scratch baseline there is weak (0.61); do not extrapolate to a strong nnU-Net on lung CT.

### Gaps
- Not verified: per-dataset nnFoundation numbers for any lung tumour set; nnFoundation weight licence and download URL (paper page gives none); whether nnFoundation training/eval data overlap lung datasets.
- Not retrieved or opened: VoCo, MultiTalent, TotalSegmentator-pretrained weights, STU-Net, CT-FM, Merlin, Models Genesis successors (only named as baselines in nnFoundation, no numbers), SuPreM tumour Dice tables, MAE-ResEncL on lung-specific sets.
- Which scheme gives the largest gain for small lesions: not answered by any opened source.

## Promptable / interactive models (SAM-Med3D, MedSAM2, SegVol, VISTA3D, nnInteractive, BiomedParse, SAT)

### Takeaway
Prompted models can match or beat automatic nnU-Net on lung tumour Dice only when given strong prompts (MedSAM2 with boxes on a small 2D-slice-based study); in a 3D tumour-model benchmark with 20 point prompts, nnU-Net still had the highest lung Dice. They are better regarded as annotation tools than as automatic lung tumour segmenters.

### Cited Findings
- Lung CT benchmark (arXiv 2505.01239; Lung1/NSCLC-Radiomics 304 patients, MSD Task06 63 scans) tumour Dice, Lung1 / Task06: nnUNet 2D 0.9039 / 0.8736; nnUNet 3D low-res 0.7386 / 0.8650; nnUNet 3D full-res 0.7023 / 0.8487; MedSAM 0.6441 / 0.7230; MedSAM2 point 0.7974 / 0.7974; MedSAM2 box 0.9091 / 0.8770. Boxes beat points. Small test sets (58 and 12 cases); nnU-Net 3D underperforming 2D is atypical and not explained; no ResEnc. — [arXiv HTML v1](https://arxiv.org/html/2505.01239v1)
- SAT3D tumour model (arXiv 2511.09592; 11 public datasets, 3,884 train / 694 in-distribution test; Swin-T encoder, AdamW LR 8e-4, 500 epochs, 128^3, 2 x A6000 48 GB; code MIT, weights on Figshare, weights licence unstated). Lung DSC / HD95 at 20 prompt points: nnUNet 0.7607 / 40.19; SAM-Med3D 0.6871 / 5.09; SAM-Med3D Turbo 0.6077; FastSAM3D 0.6457; SAT3D 0.7282 / 4.96. nnUNet variant not stated in the part I read; VISTA3D not in the comparison. — [arXiv HTML v1](https://arxiv.org/html/2511.09592v1)
- nnInteractive (arXiv 2503.08373): trained on over 120 volumetric datasets (CT, MRI, PET, microscopy); points, scribbles, boxes, lasso; integrated in Napari and MITK; reported 1st in CVPR 2025 interactive 3D segmentation challenge (per PyPI page); boxes limited to 2D; checkpoint CC-BY-NC-SA-4.0, backend code Apache-2.0. Search snippets only; paper tables not opened. — [HF model card (search result)](https://huggingface.co/MIC-DKFZ/nnInteractive); [PyPI](https://pypi.org/project/nnInteractive/)
- LesionLocator (CVPR 2025): trained on 23,262 annotated scans plus synthetic longitudinal data; authors claim about +10 Dice over existing promptable models and above nnUNet zero-shot; per the search snippet (secondary review), unverified against the paper. — [arXiv 2502.20985 (search result)](https://arxiv.org/pdf/2502.20985)
- An ENSAM paper states no unified independent comparison of these promptable models on a common benchmark exists. — [arXiv 2509.15874 (search result)](https://arxiv.org/pdf/2509.15874)

### Inferences
- With only a single quoted head-to-head (SAT3D paper), the evidence that point-prompted 3D foundation models beat specialist nnU-Net for lung tumours is negative; box-prompted MedSAM2 evidence rests on two tiny test sets and a likely under-tuned nnU-Net.

### Gaps
- Not opened: SegVol, VISTA3D lung numbers, BiomedParse v2, SAT (text-prompted) paper, nnInteractive paper tables, LesionLocator tables. No fully verified nnInteractive lung-tumour Dice.

## Architecture verdicts 2024-2026 (nnU-Net Revisited, Primus, MedNeXt, SwinUNETR-v2, Mamba)

### Takeaway
nnU-Net Revisited (MICCAI 2024) found CNN U-Nets (ResEnc, scaled) beat Transformer and Mamba variants; 2025-2026 results with pretraining keep CNNs ahead on localised segmentation, with Primus-M close behind under MAE.

### Cited Findings
- nnU-Net Revisited (arXiv 2404.09556, MICCAI 2024, Isensee et al.): recipe for state of the art is CNN U-Net, the nnU-Net framework, and scaling to modern hardware; CNN models outperformed Transformer and Mamba variants; median of three datasets in recent claims; releases ResEnc baselines. Abstract-level; search snippets. — [arXiv abs](https://arxiv.org/abs/2404.09556v2)
- With MAE pretraining, Primus-M reaches 70.42 average Dice vs ResEnc-L 70.91 on OpenMind. — [arXiv HTML v2](https://arxiv.org/html/2412.17041v2)
- nnFoundation: CNN best on spatially localised tasks, transformer best on global-semantic tasks and frozen-feature settings. — [arXiv abs](https://arxiv.org/abs/2609.26924)
- COLIPRI: transformers still trail CNNs on segmentation. — [arXiv HTML v2](https://arxiv.org/html/2510.15042v2)

### Inferences
- For lung tumour segmentation a pretrained ResEnc-L is the evidence-backed default; no opened source shows Mamba or SwinUNETR-v2 winning against it.

### Gaps
- Not opened: MedNeXt, U-Mamba, SegMamba, SwinUNETR-v2 head-to-heads; the full nnU-Net Revisited tables; any 2026 large-scale architecture benchmark beyond nnFoundation.

## Lesion-specific and pan-cancer models (ULS23, LesionLocator, CVPR 2026 pan-cancer, FLARE)

### Takeaway
A vanilla nnU-Net ResEnc-M is the reference baseline for the CVPR 2026 FLARE pan-cancer foundation-model task, and it scores low (lesion DSC 0.33), so pan-cancer CT lesion segmentation remains far from solved; ULS23 baseline is ResEnc-based at 0.703 Dice.

### Cited Findings
- CVPR 2026 Task 1 (FLARE-MedFM pan-cancer, binary lesion segmentation): about 10,000 partially labelled cases (primary lesion only) plus 762 MedSAM2 pseudo-labelled DeepLesion5K cases; baseline nnU-Net v2.6.4 ResEnc-M 3d_fullres fold 0, 2,000 epochs: lesion DSC 0.3278 (median 0.2318), NSD 2 mm 0.1797 on 50 public validation cases; no pretrained init; failure mode scattered false positives. — [baseline page](https://cloudfare-proxy.pages.dev/KS987/CVPR2026-PanCancerSeg-baseline)
- ULS23 baseline: see first section. TSFM (arXiv 2403.06396, 2024 preprint) self-reports about +2% over nnU-Net and about 5% of the epochs for similar performance in transfer (search snippet, unverified). — [arXiv 2403.06396 (search result)](https://arxiv.org/pdf/2403.06396)

### Inferences
- Pan-cancer pretraining on partially labelled data is only useful as initialisation; the 0.33 baseline shows label sparsity, not architecture, dominates.

### Gaps
- No challenge leaderboard or winner report (what pretrained weights winners used) retrieved; PASTA, CancerUniT, autoPET winners not searched; per-organ lung numbers for ULS23 not found.

## Fine-tuning practice (full vs frozen vs LoRA, warm-up, data needed, failures)

### Takeaway
Published recipes use full fine-tuning with decoder re-initialised, LR warm-up (about 15% of epochs or 15 epochs) then poly or cosine decay; benefits are largest with 10 to 70 labelled cases.

### Cited Findings
- Recipes: see nnFoundation and OpenMind bullets above (AdamW, 1e-3 peak, 15% linear warm-up, poly decay; OpenMind: 15-epoch warm-up per stage, SGD-style LR 1e-2 for ResEnc-L). — [nnFoundation](https://arxiv.org/html/2609.26924v1); [OpenMind](https://arxiv.org/html/2412.17041v2)
- Label fraction: PET/CT MAE model gives +10 to +15 Dice at 10% labels; 10%-label MAE models comparable to full-label scratch in that paper. — [arXiv HTML v1](https://arxiv.org/html/2605.21835v1)
- Failure cases: VoCo and SwinUNETR pretraining no gain and below scratch for Primus-M (OpenMind); MAE hurts CNN classification (nnFoundation); fixed-topology transfer on anisotropic MSD Pancreas 61.8 vs 63.7 scratch ResEnc-L. — sources above.
- Frozen-encoder/linear-probe: PET/CT paper does 5-shot probing (LR 1e-3 cosine) but no number retrieved; nnFoundation reports no frozen segmentation results. — [arXiv HTML v1](https://arxiv.org/html/2605.21835v1)

### Inferences
- Because pretrained gains vanish against long-trained scratch ResEnc-L on average, a lung project with a few hundred labelled cases should compare against scratch at the same epoch budget, and consider dynamic topology adaptation to avoid the fixed-topology penalty.

### Gaps
- No LoRA/adapter results for 3D CT segmentation found; no study of labelled-data thresholds specifically for lung CT.

---

# SECOND PASS (appended 2026-11-13)

Convention: "opened" = fetched page text read; "snippet" = web-search result text only (unverified). Lung-specific evidence is collected first.

## 6. Lung tumour / lung nodule evidence across all models (opened sources)

### Takeaway
Direct lung-tumour evidence for pretrained models is thin and mostly neutral-to-negative: nnFoundationCNN and MultiTalent both scored slightly below scratch nnU-Net on MSD Lung (63 CTs); the clearer gains are for dense, label-poor targets (colon, pancreas, PET lesions). Detection pretraining on lung nodules is mixed. VISTA3D and LesionLocator report strong lung-tumour numbers, but on partly overlapping training data.

### Cited Findings
- nnFoundationCNN vs scratch, MSD Lung (63 CTs): 0.662 (95% CI 0.543-0.757) vs scratch 0.680 (0.567-0.767), scratch slightly higher, CIs overlap widely; other lung downstream sets are Rider Lung (65 CT) and Covid-19-20 (199 CT), per-dataset values not in the text I read; no NSCLC/LIDC downstream segmentation; NSCLC sets are in pretraining only. The paper itself states gains were not uniform. — [nnFoundation HTML v1, offset 100000](https://arxiv.org/html/2609.26924v1)
- MultiTalent (13 abdominal CT datasets, 47 classes), Dice vs default nnU-Net, 5-fold: colon cancer 45.53 -> 58.69 (+13.16); pancreas cancer +4.97; liver tumour +2.78; liver cancer +1.71; kidney tumour +0.12; lung nodule (MSD Lung, 63 images) 72.11 -> 70.49 (-1.62). Overall class average 82.62 -> 83.84. — [MultiTalent HTML](https://arxiv.org/html/2303.14444)
- ULS23 baseline (nnU-Net v2 ResEnc, semi-supervised) test Dice by lesion type: lung 0.761 +/- 0.211, liver 0.730, pancreas 0.707, lymph node 0.685, kidney 0.633, colon 0.541, bone 0.419; overall 0.703 +/- 0.240; test set 725 lesions / 268 patients from two Dutch centres. — [ULS23 HTML](https://arxiv.org/html/2406.05231)
- VISTA3D on MSD Task 06 lung tumour, average test Dice: nnUNet 0.554, Auto3DSeg 0.562, VISTA3D auto 0.613, single-click 0.682, auto + click 0.719 (Table 3 lists auto as 0.614). Authors' own comparison on their data split (64/16/20 per source). No SAM-Med3D numbers. — [VISTA3D HTML v3](https://arxiv.org/html/2406.05285v3)
- LesionLocator zero-shot lung tumour Dice 82.51 with box prompts (human inter-rater range 81-85); average 79.71 (point) / 83.26 (box) on held-out sets; SAM-Med3D average 48.76 (points); SegVol 67.02 (point+text) / 71.26 (box+text); table alignment hard to read in extraction, verify per-dataset values in the PDF. — [LesionLocator HTML](https://arxiv.org/html/2502.20985v1)
- Detection of lung nodules (nnDetection, mAP delta vs scratch, test split): LIDC ResEnc-RetUNet + S3D +2.12, ResEnc-DefDETR + S3D +0.63, VoCo -1.46 / -3.05; LUNA16 ResEnc-DefDETR + MG +1.57, + MAE +1.52, VoCo -3.61, RetUNet+MultiTalent -1.18; PN9 ResEnc-DefDETR + MAE +2.14. Many differences below 1 point, high variance across tasks. — [Missing Piece HTML v1](https://arxiv.org/html/2509.15947v1)
- COLIPRI "Lung" segmentation Dice 68.98 vs MAE 67.12 (see first pass); dataset identity not stated. — [COLIPRI](https://arxiv.org/html/2510.15042v2)
- VoCo downstream set contains no lung tumour task (BTCV, LiTS, MSD Spleen, BraTS21, MM-WHS, CC-CCII); pretraining data about 1.6k CTs from BTCV, TCIA Covid-19 and LUNA. — [VoCo HTML](https://arxiv.org/html/2402.17300)
- Lung CT benchmark of MedSAM2 (first pass): Lung1 box 0.9091 vs nnUNet 3D full-res 0.7023 (see section above). — [arXiv 2505.01239](https://arxiv.org/html/2505.01239v1)

### Inferences
- With scratch ResEnc-L as comparator, there is no opened evidence of a statistically reliable pretraining gain on MSD Lung; the 63-case set makes any +/-2 Dice difference noise-level.
- ULS23 lung (0.761) is a ceiling-type reference for nodule-sized lung lesions from an nnU-Net ResEnc trained on LIDC/LNDb/MDSC-Lung plus pseudo-labels; nnFoundation fine-tuning on a lung-tumour set should be judged against that kind of figure.

### Gaps
- Rider Lung and Covid-19-20 per-dataset nnFoundation Dice not read (page text beyond offset 200000 not opened).
- No NSCLC-Radiomics / LIDC / LUNA segmentation comparison for pretrained vs scratch found. Search snippet only (not opened): SMIT achieved best F1 among SSL models on lung tumour sets (LRAD 0.60, 5Rater 0.64) — [arXiv 2403.13113 search result](https://arxiv.org/pdf/2403.13113v2.pdf).

## 1. Pretrained weights: availability, licence, tumour numbers

### Takeaway
nnFoundation weights are released (HF, CC-BY-SA-4.0 model card). STU-Net, SuPreM, SegVol, CT-FM, MultiTalent-based (autoPET) and ULS23 weights are downloadable; licences vary and several are not stated on the pages I opened. Tumour-level numbers vs scratch exist for MultiTalent, STU-Net(TotalSegmentator) and the autoPET checkpoint; SuPreM and CT-FM numbers were not obtainable.

### Cited Findings
- nnFoundation weights released: nnFoundationCNN (ResEnc, 102M params, features 32-64-128-256-320-320; about 410 MB checkpoint) and nnFoundationViT (Primus, 674M) at huggingface.co/MIC-DKFZ/nnFoundationCNN and /nnFoundationViT; model card licence CC-BY-SA-4.0, copyright DKFZ; single-channel input, z-score, not resampled, recommended downstream patch 192^3; nnU-Net can load directly from the Hub. nnssl repo also CC-BY-SA-4.0. HF page lists them updated about 29 days earlier, i.e. released after the paper. — [model card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN); [nnssl repo](https://github.com/MIC-DKFZ/nnssl); [MIC-DKFZ HF org](https://huggingface.co/MIC-DKFZ)
- DKFZ HF org also hosts OpenMind ResEncL checkpoints (e.g. ResEncL-OpenMind-VoCo), nnDetection "MissingPiece" backbones (MAE, MG, S3D, VoCo, MultiTalent) and nnInteractive. — [HF org](https://huggingface.co/MIC-DKFZ)
- STU-Net (uni-medical): checkpoints S (14.6M), B (58.3M), L (440M), H (1,457M params) pretrained on TotalSegmentator (1,204 images, 104 structures, 4,000 epochs), 128^3 crop, links on Google Drive/Baidu; repo Apache-2.0, no separate weights licence stated; fine-tuning script for nnUNet v1/v2 (`run_finetuning_stunet.py ... -tr STUNetTrainer_base_ft`), README advises tuning LR per task; downstream results only as images (no lung task). — [STU-Net GitHub](https://github.com/uni-medical/STU-Net)
- nnU-Net Revisited, 5-fold Dice: STU-Net L scratch LiTS 80.31, KiTS23 85.84; STU-Net L TotalSegmentator-pretrained LiTS 81.57 (+1.26), KiTS23 88.32 (+2.48), BraTS did not converge at LR 1e-3 (0); pretraining cost about 4x the 51 GPU-h fine-tune. For reference nnU-Net ResEnc L scratch LiTS 81.60, KiTS23 88.17 (same level as pretrained STU-Net L). — [nnU-Net Revisited HTML](https://arxiv.org/html/2404.09556)
- SuPreM (supervised, AbdomenAtlas 1.1: 9,262 CTs, 25 organs + pseudo-labels for 7 tumour types): weights on HF (MrGiovanni/SuPreM) for Swin UNETR, U-Net, SegResNet (2,100-CT versions) plus comparators (Tang 5,050 CT, Valanarasu 50,000 CT/MRI, Models Genesis, Med3D, DoDNet, UniMiSS); licence not stated on README (LICENSE file linked; "patents pending" note); no tumour Dice numbers in README; re-pretraining needs about 8 A100 for 7+ days. — [SuPreM GitHub](https://github.com/MrGiovanni/SuPreM)
- autoPET III (LesionTracer, DKFZ, nnU-Net ResEncL, FDG PET/CT): MultiTalent-style pretraining (CT first, then PET and MR added; separate head per dataset; 4,000 epochs, 192^3, batch 24, 1 mm) released at zenodo.org/records/13753413; fine-tune LR 1e-3, dev 1,000 epochs batch 2, final 1,500 epochs batch 3. CT pretraining: Dice 65.31 -> 66.08, FPvol 10.47 -> 7.82, FNvol 13.63 -> 15.34 (no lesion F1 reported); TotalSegmentator pretraining gave no notable advantage over scratch. The search snippet reports 68.40 final Dice vs 57.61 default nnU-Net (snippet only). — [From FDG to PSMA HTML v2](https://arxiv.org/html/2409.09478v2); [search result](https://www.arxiv.org/abs/2409.09478v2)
- MultiTalent pretraining recipe: batch 4, 2,000 epochs, patch 96x192x192, spacing 1 mm in-plane / 1.5 mm slice; fine-tune batch 2, 10-epoch head warm-up then 50-epoch whole-network warm-up, then standard nnU-Net schedule. Weights "will be published" at MIC-DKFZ/MultiTalent (not confirmed released). — [MultiTalent HTML](https://arxiv.org/html/2303.14444)
- CT-FM (arXiv 2501.09001, Pai et al.): 148,000 CTs from Imaging Data Commons, label-agnostic contrastive learning; HF feature-extractor card lists Apache-2.0, SegResNet backbone; abstract gives no tumour numbers; GitHub README (snippet) says MSD lung/hepatic/pancreas tumour benchmark, improved Dice/ASD for lung and hepatic tumours. — [arXiv abs](https://arxiv.org/abs/2501.09001); [HF card](https://huggingface.co/project-lighter/ct_fm_feature_extractor)
- Merlin (Stanford; abdominal CT vision-language; segmentation of 20 organs; weights, code, dataset released at StanfordMIMI/Merlin; licence not stated on arXiv page). — [arXiv 2406.06512](https://arxiv.org/abs/2406.06512)
- VoCo: AdamW LR 3e-4, warmup cosine (100 steps), sliding window ROI 96^3, Dice vs scratch Swin-UNETR LiTS 93.42 -> 96.52 (+3.10); code/weights availability and licence not stated on arXiv page (nnFoundation/OpenMind results show VoCo weakest, see first pass). — [VoCo HTML](https://arxiv.org/html/2402.17300)
- ULS23 baseline weights: Zenodo doi 10.5281/zenodo.10777532, code DIAGNijmegen/oncology-ULS23; weights licence not stated; dataset CC BY-NC-SA 4.0 (LNDb excluded from archive). — [ULS23 HTML](https://arxiv.org/html/2406.05231)

### Inferences
- A pretrained STU-Net L only reaches nnU-Net ResEnc L scratch level on LiTS/KiTS23 (81.57 vs 81.60; 88.32 vs 88.17), so TotalSegmentator pretraining of that model does not beat a strong from-scratch ResEnc baseline there.
- For commercial reuse, nnFoundation (CC-BY-SA) and CT-FM (Apache-2.0 per card) are the clearest; nnInteractive and NVIDIA checkpoints are non-commercial.

### Gaps
- Not obtained: SuPreM tumour Dice tables; CT-FM per-task Dice (PDF over 10 MB, unfetchable; HTML 404); Merlin/VoCo/SuPreM weight licences; MultiTalent weights actual release; nnFoundation fine-tune LR/epoch preset values (they sit in trainer classes, not docs).

## 2. Lesion-specific models

### Takeaway
ULS23 (ResEnc baseline, 0.703 Dice, lung 0.761) and the autoPET III LesionTracer checkpoint are the openly available lesion-specific pretrained assets; LesionLocator reports the best zero-shot lung-tumour figure but weights were only promised in the paper text I read.

### Cited Findings
- ULS23 training set: fully annotated KiTS21, LiTS, NIH-LN ABD/MED, LIDC-IDRI, LNDb, MDSC-Lung/Colon/Pancreas plus new Radboudumc sets and DeepLesion3D (743); partially annotated DeepLesion and CCC18 as 2D pseudo-masks; 38,693 training lesions; baseline recipe: nnU-Net v2 ResEnc with extra layer, no resampling, patch 256x256x128, stage-1 1,000 epochs on 2D GrabCut pseudo-masks then 500 epochs at 25% LR on annotated data, stage-2 pseudo-3D masks (18,264 lesions). — [ULS23 HTML](https://arxiv.org/html/2406.05231)
- ULS23 independent arXiv 2502.10608 (Shi et al.) only a student write-up choosing SwinUnet; abstract page has no results (opened). — [arXiv abs](https://arxiv.org/abs/2502.10608)
- LesionLocator: pretraining 47 public datasets (18,035 images; CT, MRI, PET); lesion fine-tuning 16 datasets, 5,227 images, 22,580 lesions; backbone ResEncL U-Net; point prompt as 5-pixel-radius ball (Dice 79.71 vs 66.56 single pixel); authors say they will release model weights and a synthetic longitudinal dataset; licence not stated in text read. — [LesionLocator HTML](https://arxiv.org/html/2502.20985v1)
- PASTA (arXiv 2502.06171; search snippet only, page 404 on HTML): claims best on 45 of 46 oncology tasks using PASTA-Gen synthetic tumours (30,000 CTs), open release claimed. CancerUniT: no source found. — [snippet source](https://fugumt.com/fugumt/paper_check/2502.06171v1_enmode)
- CVPR 2026 FLARE pan-cancer baseline (first pass): ResEnc-M scratch lesion DSC 0.328. Challenge winners / pretrained checkpoints used: not found.

### Inferences
- LesionLocator numbers use point/box prompts from the lesion location; they are not automatic detection+segmentation results and should not be compared directly with ULS23 or nnU-Net automatic Dice.

### Gaps
- ULS23 per-team challenge results; PASTA and CancerUniT numbers; FLARE 2026 / autoPET IV winner reports; LesionLocator weights release status.

## 3. Promptable models

### Takeaway
Licences: nnInteractive checkpoint CC-BY-NC-SA-4.0 (verified), VISTA3D/NV-Segment weights under NVIDIA OneWay Non-Commercial, SegVol MIT. Best lung-tumour numbers come from authors' own comparisons; no independent benchmark covers lung with all models.

### Cited Findings
- nnInteractive licence verified: checkpoint CC-BY-NC-SA-4.0, Python backend Apache-2.0; no benchmark numbers on the card. — [HF card](https://huggingface.co/MIC-DKFZ/nnInteractive)
- nnInteractive paper: training 120+ datasets, 64,518 volumes, 717,148 objects; ResEnc-L backbone; points, scribbles, 2D boxes, lasso; under 10 GB VRAM, 120-200 ms small structures on RTX 4090, up to 1.16 s large liver CT; point prompts +10.1 Dice over ScribblePrompt, boxes +14.9 over SAM2, scribbles +23.8, +14.9 over SegVol with point refinement; VISTA3D explicitly not evaluated (closed-set); no lung numbers. — [nnInteractive HTML (first 100k chars)](https://arxiv.org/html/2503.08373)
- VISTA3D: about 11,454 CTs, 127 automatic classes (Table 2 caption says 118); code Apache-2.0 at Project-MONAI/VISTA; NV-Segment-CTMR card: weights "NVIDIA OneWay Non-Commercial License", lung tumour = class 23, recommends NV-Segment-CT for tumours. — [VISTA3D HTML](https://arxiv.org/html/2406.05285v3); [VISTA repo](https://github.com/Project-MONAI/VISTA); [HF card](https://huggingface.co/nvidia/NV-Segment-CTMR)
- SegVol: MIT licence; 90k unlabeled + 6k labeled CTs; ViT pretraining 96k CTs, 2,000 epochs; weights on HF (BAAI/SegVol); results only as images in README, no lung-tumour numbers extractable. — [SegVol GitHub](https://github.com/BAAI-DCAI/SegVol)
- SAM-Med3D: 22K 3D images, 131K training masks; code/models at uni-medical/SAM-Med3D; no lung numbers in paper text; SAT3D paper gives SAM-Med3D lung DSC 0.687 vs nnUNet 0.761 (first pass). — [SAM-Med3D HTML](https://arxiv.org/html/2310.15161v3)
- BiomedParse: "v2" not found. Search snippet: Bio2Vol (MICCAI 2025) adapts 2D BiomedParse to 3D, +4.72% DSC on AMOS22 — unverified. — [snippet](https://papers.miccai.org/miccai-2025/paper/1852_paper.pdf)

### Inferences
- Only the authors' own VISTA3D (0.719 auto+click vs nnUNet 0.554) and LesionLocator (82.5 box) lung-tumour figures favour promptable models, both against weakly tuned or non-ResEnc nnU-Net and with train/test overlap risk; SAT3D paper (nnUNet 0.761 > SAM-Med3D 0.687) and MSD Lung results above point the other way. Note nnUNet on MSD Lung is 0.554 in VISTA3D's table but 0.761 in SAT3D's and 0.680 (scratch ResEnc-L) in nnFoundation: baseline strength differs by 0.1-0.2 Dice across papers.

### Gaps
- BiomedParse v2, SegVol lung numbers, nnInteractive lung numbers, SAM-Med3D weights licence: not obtained.

## 4. Architectures

### Takeaway
Full tables from nnU-Net Revisited show ResEnc L / MedNeXt L at the top on tumour datasets; Mamba variants do not beat the CNNs; SwinUNETR is last.

### Cited Findings
- 5-fold Dice, LiTS / KiTS23 / BraTS21, trained from scratch on one A100 40 GB: nnU-Net orig 80.09 / 86.04 / 91.24; ResEnc M 80.75 / 86.79 / 91.26 (9.1 GB, 12 GPU-h); ResEnc L 81.60 / 88.17 / 91.13 (22.7 GB, 35 h); ResEnc XL 81.19 / 88.67 / 91.18 (36.6 GB, 66 h); MedNeXt L k3 82.14 / 88.25 / 91.35 (17.3 GB, 68 h); MedNeXt L k5 82.34 / 87.74 / 91.50 (233 h); STU-Net L 80.31 / 85.84 / 91.26; SwinUNETR 76.50 / 81.27 / 90.68; U-Mamba Bot 80.40 / 86.22 / 91.26; U-Mamba Enc 80.27 / 86.34 / 90.91; No-Mamba ablation 80.57 / 85.98 / 91.26. SegMamba not benchmarked in the page. KiTS fold-0 setup: ResEnc L patch 160x224x192, spacing 0.78 mm, batch 2. — [nnU-Net Revisited HTML](https://arxiv.org/html/2404.09556)
- Primus-M + MAE 70.42 vs ResEnc-L + MAE 70.91 on OpenMind (first pass); nnFoundation: CNN best for localised, ViT best for global tasks (first pass).

### Inferences
- MedNeXt L is within about 0.5-0.7 Dice of ResEnc L (higher on LiTS, equal on KiTS) at 2x the GPU-hours (k3) or 6.7x (k5); no meaningful architecture gain over ResEnc L at tumour level.

### Gaps
- No 2025-2026 large benchmark including lung lesions beyond nnFoundation (34 datasets, lung per-dataset only partly read); SegMamba numbers not retrieved.

## 5. Fine-tuning practice

### Takeaway
nnU-Net now provides encoder-only transfer from nnssl checkpoints with a dedicated trainer; published PEFT (LoRA/adapter) evidence for 3D CT is organ-focused and few-shot, not tumour. Pretraining benefit shrinks as labelled data and training length grow.

### Cited Findings
- nnU-Net nnssl doc: default trainer `PretrainedTrainer` (CNN/ResEnc) or `PretrainedTrainer_Primusx`; `DynamicPretrainedTrainer` adapts kernel sizes/depth to a ResEnc plan; variants `_150ep`, `_300ep`, `_smallerlr`, Adam versions; only encoder, stem, input projection and positional embeddings are loaded, segmentation head skipped, stem repeated if channel count differs, positional embeddings interpolated; recommended preprocessing `-am like_pretrained` (often 1 mm); `--from_scratch` for a matched baseline; warm-up length and LR values are in trainer classes, not the doc. — [nnssl fine-tuning doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md)
- nnU-Net supervised pretraining doc: transfer plans between datasets (normalisation and CT clip values transfer), `-pretrained_weights`, all layers except segmentation layers loaded, no fine-tuning-specific trainers. — [doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md)
- Encoder-only transfer outperformed full-model transfer in nnDetection pretraining study; 128^3 patch best, 192^3 caused OOM for DefDETR; pretraining gains absent when architecture is non-ResEnc or pretraining is VoCo; no low-data study run. — [Missing Piece HTML v1](https://arxiv.org/html/2509.15947v1)
- Label-count regimes: nnFoundation about +3 Dice at 10 cases (first pass; protocol 10/20/40/100 cases, 3 seeds, 6 datasets, per-size table not in text read); PET/CT MAE gain on autoPET grows as labels shrink: +10.1 Dice at 100% labels (0.6149 -> 0.7160), +15.0 at 10% (0.4555 -> 0.6055); ResEnc-L scratch 1k epochs narrows OpenMind gap to +0.44. — [nnFoundation](https://arxiv.org/html/2609.26924v1); [PET/CT MAE](https://arxiv.org/html/2605.21835v1); [OpenMind](https://arxiv.org/html/2412.17041v2)
- PEFT: Few-shot volumetric organ model (Medical Image Analysis 2025, pretraining 9 public datasets, 2,042 CTs, 29 structures): black-box adapters work for seen classes in low-data regimes; LoRA plus decoder fine-tuning transfers best to novel structures, better than popular SSL objectives; no Dice numbers or tumours on the page. — [project page](https://jusiro.github.io/projects/fseft.html)
- Dynamic-rank PEFT (MICCAI 2025, arXiv 2507.15793) is few-shot base/novel organs, no full fine-tune comparison on page; ConvAdapter for SegResNet (NeurIPS 2024 workshop), MedSAM3-LoRA, SAM federated LoRA are snippet-only (unverified). — [arXiv abs](https://arxiv.org/abs/2507.15793); [snippet](https://neurips.cc/virtual/2024/101699)
- Failure cases collected: VoCo/SwinUNETR no gain (OpenMind); VoCo -3.6 mAP on LUNA16; MultiTalent -1.6 on MSD Lung; nnFoundation -0.018 on MSD Lung; TotalSegmentator pretraining no advantage for autoPET lesions; STU-Net L TotalSegmentator fine-tune did not converge on BraTS at LR 1e-3.

### Inferences
- A defensible lung recipe: nnFoundationCNN with `DynamicPretrainedTrainer`, like_pretrained spacing, 1,000 epochs (or 150 for screening), run against `--from_scratch` with identical plan, 5-fold, because MSD Lung is small enough that single-fold comparisons are noise.

### Gaps
- No controlled 3D CT tumour-segmentation study of frozen vs LoRA vs full fine-tuning found; labelled-count crossover for tumour segmentation on CT not found (nnFoundation low-data table not read).

## Corrections to first-pass flags
- nnFoundation weights and licence: now verified (HF model cards, CC-BY-SA-4.0); the first-pass "not stated" referred to the paper page only.
- nnInteractive licence: verified from the HF card. VISTA3D is not evaluated in the nnInteractive paper.
- ULS23 lung Dice 0.761 and Zenodo weights: verified from the challenge paper; version conflict resolved (nnU-Net v2).
- Still unverified (snippet only): LesionTracer final Dice 68.40, PASTA claims, Bio2Vol, ConvAdapter, MedSAM3-LoRA, SMIT lung F1, TSFM, SuPreM "supervised beats SSL".
- Typo fix for reader: in the Label-count bullet the PET/CT absolute gain is +10.1 Dice at 100% labels and +15.0 at 10% labels (0.4555 -> 0.6055); it does not shrink with more labels.
