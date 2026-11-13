# Training, sampling, data and inference techniques for small-lesion / lung tumour segmentation in 3D CT (as of Nov 2026)

Verification key: [OPENED] = page fetched and numbers read from it. [SNIPPET] = only seen in a search-result summary, not opened; treat as unverified. Research stopped at the tool-call budget; many sub-questions have gaps (listed per section).

## Synthetic lesions and generative augmentation

### Takeaway
Synthetic-tumour training gives large, verified gains for liver/pancreas/kidney (DiffTumor) and a +5 Dice gain on LIDC lung nodules for LeFusion with its best settings. LeFusion's own baselines (hand-crafted, conditional diffusion, RePaint) lowered Dice relative to real-only. No lung-tumour (as opposed to nodule) evidence was found. Cost is high (a trained generative model plus healthy CTs), and gains depend on realism choices.

### Cited Findings
- DiffTumor (CVPR 2024), early-stage tumour-wise sensitivity with U-Net, real-only to +DiffTumor: liver 75.6 to 82.2, pancreas 64.3 to 71.4, kidney 50.0 to 78.6 (+28.6). Cross-organ transfer (liver-trained generator, pancreas target) 0 to 56.3. [OPENED] [arXiv 2402.19470](https://arxiv.org/html/2402.19470)
- DiffTumor all-stage DSC, real-only to +DiffTumor: nnU-Net liver 64.3 to 73.6, pancreas 59.9 to 63.6, kidney 73.8 to 84.5; Swin UNETR liver 65.1 to 71.4, pancreas 52.2 to 62.2, kidney 80.6 to 85.1. Real training set was small (~95 tumour CTs per organ) plus ~116-120 healthy CTs, so gains are in a low-data regime. [OPENED] [arXiv 2402.19470](https://arxiv.org/html/2402.19470)
- DiffTumor: lung was not evaluated; the diffusion model needs only one annotated tumour; code at https://github.com/MrGiovanni/DiffTumor. Radiologist reader study was near chance for real-vs-synthetic, but ~50% of synthetic tumours were still judged real by the more experienced reader, with failure modes of unrealistic shape, attenuation, noise and missing mass effect. [OPENED] [arXiv 2402.19470](https://arxiv.org/html/2402.19470)
- DiffTumor paper states a prior synthesis method (Hu et al.) slightly decreased pancreatic DSC/NSD, i.e. a negative result for a competing synthesis approach. [OPENED] [arXiv 2402.19470](https://arxiv.org/html/2402.19470)
- LeFusion (ICLR 2025), LIDC lung nodule, Dice/NSD, nnU-Net: baseline real-only 78.26/88.90; LeFusion 78.77/89.25; LeFusion-H (histogram texture control) 80.62/90.90; LeFusion-H + DiffMask with synthetic nodules inserted in normal subjects 83.44/93.35. Swin UNETR: 78.38 to 83.13 Dice. [OPENED] [arXiv 2403.14066](https://arxiv.org/html/2403.14066v2)
- LeFusion negative/neutral comparisons on the same task (nnU-Net Dice, P+P' setting): hand-crafted 76.80, conditional diffusion 77.05, RePaint 77.57, all below the 78.26 real-only baseline. [OPENED] [arXiv 2403.14066](https://arxiv.org/html/2403.14066v2)
- LeFusion synthetic volume: 2,104 synthetic ROIs from 808 real pathological cases, 3,076 x 2 synthetic samples from 135 normal subjects. Code: https://github.com/M3DV/LeFusion. [OPENED] [arXiv 2403.14066](https://arxiv.org/html/2403.14066v2)
- FreeTumor: GAN-based, adversarial filtering of low-quality synthetic tumours, uses large unlabeled CT; 2025 version reports +6.7% Dice and +16.4% early-detection sensitivity (averaged over tumour types, not lung-specific); 2024 preprint reported +8.9% DSC vs real-only. Whether the figures are absolute or relative was not checked. [SNIPPET] [arXiv 2502.18519](https://arxiv.org/pdf/2502.18519), [arXiv 2406.01264](https://arxiv.org/pdf/2406.01264)
- Pixel2Cancer (cellular-automata growth, code public): liver DSC 58.9 / NSD 63.7 per secondary review; no lung result seen. [SNIPPET] [arXiv 2403.06459](https://arxiv.org/pdf/2403.06459)
- TextoMorph (text-driven): reported +8.5 sensitivity and +6.3 DSC, unit (absolute/relative) unclear. [SNIPPET] [arXiv 2412.18589](https://arxiv.org/pdf/2412.18589)

### Inferences
- For a lung pipeline, LeFusion is the only method with verified lung-nodule numbers and open code; its gain depends on the texture-control (H) and normal-subject insertion variants, so a naive inpainting generator may hurt (LeFusion's own baselines did).
- Gains were measured in small-real-data regimes (about 100 tumour CTs); expect smaller gains with large labelled sets (inference, not tested in sources).

### Gaps
- Not covered: SynTumor, CarveMix, TumorCP, SelfMix, MAISI, GenerateCT, LUNA/LIDC nodule-synthesis papers. Search returned nothing usable for SynTumor.
- FreeTumor, Pixel2Cancer and TextoMorph full papers were not opened; lung-tumour (not nodule) synthesis results are absent.

## Patch sampling

### Takeaway
Only a snippet-level source was reached. nnU-Net's default forces at least one third of each batch's patches to contain a foreground class; nnU-Net code comments say this matters for lung because most patches are empty. Lesion-balanced and size-stratified sampling were not verified.

### Cited Findings
- Original nnU-Net: more than a third of batch samples are forced to contain a randomly chosen foreground class. [SNIPPET] [arXiv 1809.10486](https://arxiv.org/pdf/1809.10486)
- Other papers report moving to 50% lesion/background patch split to raise small-lesion frequency, and choosing a lesion first then a voxel inside it because uniform foreground-voxel sampling favours large lesions (brain, not lung). [SNIPPET] via [search results](https://arxiv.org/pdf/2412.06610)
- autoPET III winner (PET/CT) used patch 192^3 with batch 2 (dev) or batch 3 (final) in nnU-Net ResEnc L. [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)

### Inferences
- A lesion-first sampler is a cheap change for multi-lesion lung nodules; no CT lung evidence of gain was found.

### Gaps
- No verified study on hard-patch mining, size-stratified sampling, lung-ROI cropping, cascades or whole-volume training for lung lesions.

## Losses for small and multiple lesions

### Takeaway
Instance-aware losses (CC-DiceCE, blob loss) raise lesion recall on small-lesion brain MRI tasks but are dataset dependent and can add false positives. No CT lung evidence was found. Loss-odyssey conclusion is only snippet-verified.

### Cited Findings
- CC-DiceCE vs DiceCE vs Blob-DiceCE inside nnU-Net, 5-fold, lesion-wise F1: lacunes 0.333 / 0.299 / 0.366; microbleeds 0.510 / 0.525 / 0.562; brain mets 0.801 / 0.788 / 0.800; WMH 0.696 / 0.662 / 0.685; BraTS 0.493 / 0.479 / 0.084 (CC-DiceCE recall 0.529 vs 0.451 but precision 0.052 vs 0.771). All brain MRI. [OPENED] [arXiv 2511.17146](https://arxiv.org/html/2511.17146v1), code https://github.com/TIO-IKIM/Learning-to-Look-Closer
- Same paper: instance losses combined 1:1 with DiceCE; nnU-Net Dice made non-smooth (epsilon 0) after training instability; authors flag extra false positives when a few large components dominate. [OPENED] [arXiv 2511.17146](https://arxiv.org/html/2511.17146v1)
- autoPET III ablation: removing the Dice smoothing term helped (67.01 vs 66.08 Dice with CT pretraining). [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)
- Loss odyssey (Ma et al., MedIA 2021): no single loss best across 20 losses, 4 tasks, 6 datasets; compound losses strongest; Dice+TopK, focal, HD and boundary-type compounds named as robust. Not opened. [SNIPPET] [arXiv 2005.13449](https://arxiv.org/abs/2005.13449)
- BiCC (bidirectional connected-component loss) claims better lesion-wise F1 than CC-DiceCE on 4 of 5 datasets; same group, preprint, not opened. [SNIPPET] [arXiv 2609.30223](https://arxiv.org/pdf/2609.30223)

### Inferences
- If false positives are the problem in lung CT, CC-DiceCE alone could worsen them (BraTS case); pair with component filtering.

### Gaps
- Not covered: Tversky/focal-Tversky evidence, ICI loss, centre-heatmap auxiliary heads, boundary/HD verdict beyond snippet, clDice relevance, deep supervision.

## Data-centric (multi-dataset, pseudo-labels, organ co-labels)

### Takeaway
Organ co-labels as a second head plus CT/MR/PET pretraining gave the largest verified steps in the autoPET III ablation. ULS23 offers a large public lesion set (38,693 lesions) with a released semi-supervised baseline.

### Cited Findings
- autoPET III (PET/CT) ablation, Dice: default nnU-Net 57.61; ResEnc L 65.31; +CT pretraining 66.08; + misalignment DA + no smoothing 67.45; + CT/MR/PET pretraining 68.03; + organ head 68.33 (FNvol 13.52 to 10.15); + batch 3, 1500 epochs 68.40. [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)
- Organ head: second output head with TotalSegmentator labels (spleen, kidneys, liver, bladder, lungs, brain, heart, stomach, prostate, head glands), equal loss weights; lower organ-loss weight raised FP and FN volumes. Pretraining: 4000 epochs, 192^3, batch 24, 1 mm isotropic. Code: https://github.com/MIC-DKFZ/autopet-3-submission. [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)
- ULS23: training set 38,693 lesions (chest, abdomen, pelvis CT); released baseline is a semi-supervised 3D model, test Dice 0.703 +/- 0.240 (test set 775 lesions, 284 patients). Lung-lesion validation Dice 0.754 +/- 0.149 (267 lesions) per grand-challenge page, not opened. [OPENED abstract] [arXiv 2406.05231](https://arxiv.org/abs/2406.05231); [SNIPPET] [grand-challenge baseline](https://grand-challenge.org/algorithms/universal-lesion-segmentation-uls23-baseline/)

### Inferences
- The organ-head idea transfers cheaply to lung CT (TotalSegmentator lung lobes/airway as co-labels), but gain size in lung is untested.

### Gaps
- MultiTalent, DoDNet, CLIP-driven Universal Model, FLARE pseudo-labelling, active learning, weak/RECIST labels: not retrieved (the MultiTalent arXiv id I tried was wrong).

## Augmentation

### Takeaway
Misalignment augmentation helped modestly in PET/CT only; no CT-only evidence for kernel/reconstruction augmentation was retrieved.

### Cited Findings
- Misalignment DA (rotation up to 5 degrees, translation up to [2,2,0] voxels): Dice 65.31 to 65.76, FNvol 13.63 to 12.50 on ResEnc L (+0.4 to +0.5 Dice; PET/CT specific). [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)

### Gaps
- DA5/insane DA, intensity/kernel augmentation, 3D MixUp/CutMix, TTA evidence for lung CT not found. autoPET III used 5-fold ensemble with mirroring TTA but gave no isolated TTA gain.

## Optimisation and schedule

### Takeaway
Residual-encoder presets give consistent verified gains over original nnU-Net, largest on tumour-rich KiTS, smallest on BTCV; training time scales 12 h / 35 h / 66 h.

### Cited Findings
- nnU-Net Revisited, 5-fold DSC, original / ResEnc M / L / XL: KiTS2023 86.04 / 86.79 / 88.17 / 88.67; LiTS 80.09 / 80.75 / 81.60 / 81.19; AMOS 88.64 / 88.77 / 89.41 / 89.68; BTCV 83.08 / 83.31 / 83.35 / 83.28. GPU hours (A100) 9 / 12 / 35 / 66; VRAM 7.7 / 9.1 / 22.7 / 36.6 GB. LiTS gain does not grow from L to XL. [OPENED] [arXiv 2404.09556](https://arxiv.org/html/2404.09556v1)
- The same paper compares methods at equal compute and finds Auto3DSeg SegResNet needed ~247 GPU h (8 GPUs) for 87.77 KiTS DSC vs 11 GPU h for ResEnc M at 87.91. It gives no epoch-count study for nnU-Net. [OPENED] [arXiv 2404.09556](https://arxiv.org/html/2404.09556v1)
- nnU-Net docs recommend ResEnc L as the new default (24 GB GPU, ~35 h on A100); M for 9-11 GB. [OPENED] [resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- autoPET III final model: batch 3, 1500 epochs vs 1000 epochs gave +0.07 Dice but FP volume 8.93 to 7.82 (small). [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)

### Gaps
- No verified evidence for 4000-8000 epoch trainers, EMA, SWA, AdamW vs SGD, checkpoint ensembling or distillation.

## Post-processing and calibration

### Takeaway
Only snippet-level lung evidence: a thoracic-box + nnU-Net + false-positive-reduction classifier pipeline reports lesion-level Dice 81% internal, 78% external.

### Cited Findings
- LLSB_CFPR (Eur Radiol Exp 2025): nnU-Net multi-instance lesion segmenter followed by a multiscale cascade FP classifier; lesion-level Dice 81% internal, 78% external (ChAImeleon). Not opened; effect of the FP stage not isolated in the snippet. [SNIPPET] [Springer](https://link.springer.com/article/10.1186/s41747-025-00617-7)
- Uncertainty-guided anatomy-aware post-processing removed spurious components and improved Dice (Swin UNETR, Orlando set). [SNIPPET] [arXiv 2504.12215](https://arxiv.org/pdf/2504.12215v1)
- autoPET III used 5-fold ensemble, tile step 0.6, mirroring TTA time permitting; no lesion-level post-processing described. [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)

### Gaps
- No isolated ablation of connected-component or size filtering on CT lung; MSD Task06 nnU-Net post-processing not found.

## Recommendations from large empirical studies

### Takeaway
nnU-Net Revisited concludes gains come from CNN U-Nets plus scaling, and recommends equal-compute comparisons; so ResEnc L is the verified first step for a plain 3D U-Net.

### Cited Findings
- CNN-based U-Net/nnU-Net variants are the state of the art in its benchmark; many claimed architecture gains fail under rigorous baselines; median 3 datasets in claiming studies. [SNIPPET] [arXiv 2404.09556](https://arxiv.org/abs/2404.09556), numbers above [OPENED].
- autoPET III team won model-centric category with nnU-Net ResEncL + organ head + misalignment DA + pretraining (paper claim; grand-challenge preliminary listing ranked it 12th, different phase). [OPENED paper] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2); [SNIPPET] [grand-challenge](https://grand-challenge.org/algorithms/lesiontracer/)

### Gaps
- Metrics Reloaded, "why is the winner the best", KiTS/LiTS reports and 2025-2026 lung-specific studies were not retrieved.

## First-pass ranking (superseded by the final ranking at the end of the second pass; tags above updated there)
1. ResEnc L preset: +0.7 to +2.1 DSC on tumour sets, ~4x GPU time.
2. Multitask anatomy/organ head plus large-scale CT pretraining: about +1 to +2.7 Dice combined in autoPET III (PET/CT).
3. Synthetic nodules (LeFusion-H with normal-subject insertion): +5 Dice on LIDC, costly, only lung evidence.
4. Instance losses (CC-DiceCE): recall gain, FP risk, brain evidence only.
5. Misalignment DA: small, PET/CT-specific.

# SECOND PASS (all claims below come from pages I opened unless tagged otherwise; tag changes for first-pass items are stated here)

## 1. First-pass SNIPPET items, now checked

### Takeaway
FreeTumor is the only synthesis method here with verified lung-tumour numbers (MSD06 Lung: +6.1 Dice average vs SwinUNETR, +17.8 small-tumour sensitivity). Pixel2Cancer and TextoMorph were not evaluated on lung. Loss Odyssey conclusion is confirmed only from the authors' README. The lung FP-reduction pipeline gains are small (+0.03 F1 at constant lesion Dice).

### Cited Findings
- FreeTumor (Nat Commun 2025 version) evaluates liver, pancreas, kidney, lung tumours (MSD06-Lung, RIDER) and COVID-19 lesions. Average Dice gain vs SwinUNETR trained on real data: liver +10.6, pancreas +5.5, kidney +3.8, lung +6.1, COVID-19 +7.9. Small-tumour sensitivity gain: liver +22.9, pancreas +10.3, kidney +16.7, lung +17.8, COVID-19 +14.1. Overall small-tumour sensitivity 49.7 to 66.1 (+16.4, absolute points); the +6.7 Dice headline is not labelled absolute or relative (probably absolute, by the sensitivity convention). Pretraining of the generator used 33 public sources, 161k CTs, 141,784 chest volumes for lung, some organ labels were pseudo-labels. Code: https://github.com/Luffy03/FreeTumor. [OPENED, first 100k of 158k chars] [arXiv 2502.18519](https://arxiv.org/html/2502.18519v1)
- FreeTumor limitations: CT only; some radiologists still spot synthetic tumours; FID judged unreliable for tumour quality. [OPENED] same source.
- Pixel2Cancer evaluates liver, pancreas, kidney only (no lung). U-Net DSC/NSD, Pixel2Cancer vs real-tumour training: liver 58.9/63.7 vs 56.7/58.0; pancreas 60.9/57.1 vs 57.8/56.5; kidney 73.2/65.0 vs 71.3/62.8. Small tumours: liver DSC 47.2 vs 46.7, pancreas 36.5 vs 34.3, kidney 18.1 vs 17.9 (tiny gains over real). Code: https://github.com/MrGiovanni/Pixel2Cancer. [OPENED] [arXiv 2403.06459](https://arxiv.org/html/2403.06459v1)
- TextoMorph evaluates liver, pancreas, kidney (no lung), baseline is DiffTumor, U-Net via MONAI. Table differences are absolute points, e.g. fold 0 liver DSC 64.2 to 71.6, pancreas 58.9 to 67.3, kidney 78.9 to 85.2; the paper does not trace the +6.3 DSC / +8.5 sensitivity headline to a table. Code: https://github.com/MrGiovanni/TextoMorph. [OPENED] [arXiv 2412.18589](https://arxiv.org/html/2412.18589)
- Loss Odyssey (Ma et al., MedIA 2021): authors' repo states "compound loss functions are the most robust losses, especially for the highly imbalanced segmentation tasks"; HECKTOR 2020 winner used DiceFocal, ADAM 2020 winner and runner-up DiceTopK. I could not open the journal paper (sciencedirect returned 403), so the 20-loss/4-task/6-dataset details stay unverified. [OPENED README only] [SegLossOdyssey](https://github.com/JunMa11/SegLossOdyssey); taxonomy preprint abstract [arXiv 2005.13449](https://arxiv.org/abs/2005.13449)
- nnU-Net foreground oversampling, implementation (v2 data_loader.py): by default the trailing share of each batch is forced to contain foreground (`not sample_idx < round(batch_size*(1-oversample_foreground_percent))`); a probabilistic mode forces foreground per sample with that probability; the class is chosen uniformly among eligible foreground classes, then a random voxel of that class is the patch centre; cases with no foreground fall back to random crops; a code comment says oversampling stabilises training "especially if many patches are empty (Lung for example)". The default value 0.33 is not in the file I read, and I could not read the nnU-Net paper PDF text (binary), so the one-third default remains SNIPPET-level. [OPENED code] [data_loader.py](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/dataloading/data_loader.py)
- Lung FP-reduction pipeline LLSB_CFPR (Eur Radiol Exp 2025): thoracic-box-cropped 3D nnU-Net (5-fold, default post-processing replaced by 26-connectivity components), then a cascade of two classifiers (2D ResNet-50 on central coronal slice removes extrapulmonary candidates; 3D ResNet-50 on 96^3 patches removes non-lesions; thresholds tuned to maximise global Dice at preserved sensitivity). Test set lesions >=10 mm: no FP reduction F1 0.680, precision 0.567, recall 0.849, image DSC 0.746; full cascade F1 0.712, precision 0.615, recall 0.847, image DSC 0.756; lesion-level DSC unchanged at 0.810. External 188 scans: lesion DSC 78.3%, F1 73.3%, precision 64.5%, recall 84.9%. Training data 1,081 CTs / 5,322 lesions; classifier data 9,706 candidates. About 2 FP per patient at 80% sensitivity. No code released. [OPENED] [PMC12361585](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361585/)

### Inferences
- FreeTumor is the one candidate with lung-tumour evidence, but its comparison model is SwinUNETR and the generator needs a large unlabeled-CT pretraining effort; gains against a ResEnc-L baseline are untested.
- The FP-reduction stage gives +0.03 F1 and about +0.01 image DSC: useful for detection precision, not segmentation overlap.

### Gaps
- FreeTumor absolute baseline Dice for lung (Extended Data Table 5) was beyond the characters read.
- A search summary claimed LeFusion Dice 80.19 vs copy-paste 77.29 on LIDC; this does not match the LeFusion table I opened (hand-crafted 76.80, LeFusion 78.77, LeFusion-H 80.62), so I do not use it.

## 2. Copy-paste / lesion-mixing augmentation

### Takeaway
Gains are measured mostly on brain MRI and kidney/liver CT, with no lung CT copy-paste study with nnU-Net found. Reported gains are 1-7 Dice points, larger in low-label regimes.

### Cited Findings
- CarveMix (MICCAI 2021): lesion-aware carve-and-paste for brain lesions; abstract says better than simple augmentations on two brain lesion datasets, no numbers on the abstract page. [OPENED abstract] [arXiv 2108.06883](https://arxiv.org/abs/2108.06883)
- TumorCP: object-level copy-paste, kidney tumour; +7.12% tumour Dice over a strong baseline, +2.32% over state of the art with image-level augmentation, +21.87% with 10% labels (relative or absolute not stated; backbone not named on the page). [OPENED abstract] [arXiv 2107.09843](https://arxiv.org/abs/2107.09843)
- LesionMix (nnU-Net 3D fullres, 1000 epochs, LiTS liver tumours, training on 28 subjects, test on 103): Dice 63.51 at 100% data (+3.31 vs CarveMix, +5.12 vs CutMix, +1.58 vs traditional augmentation), 30.12 at 10% data (+9.59 vs CarveMix, +5.89 vs traditional). SelfMix could not be compared (code not public). Code link announced: https://github.com/dogabasaran/lesionmix. [OPENED] [arXiv 2308.09026](https://arxiv.org/html/2308.09026)
- SelfMix (MICCAI 2022): lesion-aware mixing that weights tumour/non-tumour fusion by tumour geometry; abstract claims gains on two public datasets; I could not open numbers. [SNIPPET] [aminer](https://www.aminer.org/pub/63730cc990e50fcafd31f50e/selfmix-a-self-adaptive-data-augmentation-method-for-lesion-segmentation)
- Meningioma MRI study 2026: lesion paste into healthy backgrounds with nnU-Net: baseline 72.69 Dice, semantic copy-paste only 74.06, spatial only 75.29, both 79.93; synthetic-only training 68.34. 73 training cases (low data); table labels inconsistent in the paper. Brain MRI, not lung. [OPENED] [arXiv 2601.17031](https://arxiv.org/html/2601.17031v1)
- Scaling tumour segmentation (ICCV 2025, AbdomenAtlas 2.0, pancreas): real + 3x synthetic improved DSC at small real sizes (60 real: 40.2 to 48.2; 278 real: 52.7 to 58.1) and was equal at 1,500+ real (59.3 vs 59.2); 500 real + synthetic matched 1,500 real. Out-of-distribution: gains kept growing with more real data. This is a negative-for-large-data result. [OPENED secondary summary; original https://arxiv.org/abs/2510.14831 not opened] [papernotes](https://en.papernotes.org/ICCV2025/medical_imaging/scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data/)

### Inferences
- Synthetic or pasted lesions help most when labelled lesions are few (<~300 cases); with large labelled sets the pancreas scaling data show no gain.

### Gaps
- No lung tumour copy-paste/synthesis study with nnU-Net, 2024-2026, other than FreeTumor (above) and LeFusion (nodules); a 2025 thesis (2D U-Net) and SYN-LUNGS (physics simulation) appeared in search but were not opened.

## 3. Partial-label / multi-dataset training

### Takeaway
MultiTalent helps colon and pancreas cancer classes but was slightly worse for lung (MSD06) than default nnU-Net; multi-dataset training is not a guaranteed win for a single lung target.

### Cited Findings
- MultiTalent (MICCAI 2023): 13 abdominal CT datasets, 47 classes, 2000 epochs, patch 96x192x192. Tumour Dice, default nnU-Net vs MultiTalent ResEnc: MSD03 liver tumour 63.72 to 66.50; MSD06 lung nodule 72.11 to 70.49 (-1.62); MSD07 pancreas cancer 52.74 to 57.71; MSD08 hepatic tumour 72.17 to 73.88; MSD10 colon 45.53 to 58.69; KiTS19 83.67 to 83.79. Mean over all classes +1.56 Dice vs default nnU-Net, +1.26 vs per-dataset ResEnc. Pretraining for KiTS19 83.22 to 84.01 (p = 0.12). Code: https://github.com/MIC-DKFZ/MultiTalent. [OPENED] [arXiv 2303.14444](https://arxiv.org/html/2303.14444v2)
- CLIP-driven Universal Model (ICCV 2023): 25 organs + 6 tumour types from 3,410 CTs of 14 datasets, 6,162 external CTs, ~6x faster inference, first on MSD leaderboard; no per-tumour numbers or nnU-Net comparison on the abstract page. [OPENED abstract] [arXiv 2301.00785](https://arxiv.org/abs/2301.00785)
- DoDNet: one shared network with task-encoding and dynamic head for partially labelled organ/tumour datasets, better than competitors on 7 organ/tumour tasks; no numbers on the abstract page. [OPENED abstract] [arXiv 2011.10217](https://arxiv.org/abs/2011.10217)
- Organ co-labels: autoPET III organ head gave Dice 68.03 to 68.33 and FNvol 13.52 to 10.15 (first-pass finding; PET/CT). Lower organ loss weight was worse. [OPENED] [arXiv 2409.09478](https://arxiv.org/html/2409.09478v2)

### Inferences
- Multi-dataset gains concentrate on classes with few training cases; for a lung-only project with enough lung data, expect little (MultiTalent lung was negative).

### Gaps
- Universal Model and DoDNet tumour numbers (incl. MSD06) were not read; no study isolating TotalSegmentator organ co-labels on lung CT.

## 4. Pseudo-labelling / semi-supervised

### Takeaway
Pseudo-labelling helps when unlabeled data is abundant and labels are scarce (FLARE22), but in ULS23 the semi-supervised step added only +0.002 to +0.008 Dice on lung, much less than switching to the ResEnc architecture (+0.034).

### Cited Findings
- ULS23 baseline: GrabCut 2D pseudo-masks from RECIST measurements (DeepLesion, CCC18), ResEnc nnU-Net trained on them, then 3D pseudo-masks kept only if long/short-axis error <=5 pixels (18,264 lesions), second nnU-Net pretrained then fine-tuned on fully annotated data. Lung lesion Dice (267 test lesions, no TTA): nnU-Net 0.719, ResEnc 0.753, ResEnc+SS 0.761; held-out 0.753 / 0.751 / 0.754. Overall test 0.651 / 0.700 / 0.703. SS helped colon (0.430 to 0.541) and liver (0.706 to 0.730) but hurt pancreas (0.756 to 0.707; fewer than 20 lesions). VOI 256x256x128, lesion voxel placed at centre, batch 3, 1000 epochs, no lesion oversampling. [OPENED] [arXiv 2406.05231](https://arxiv.org/html/2406.05231v2)
- FLARE22 winner (Huang et al., "Revisiting nnU-Net for Iterative Pseudo Labeling and Efficient Sliding Window Inference"): Big nnU-Net trained on 50 labelled cases, pseudo-labels 2,000 unlabeled cases, iterate, drop high-variance pseudo-labels (1,924 kept), train Small nnU-Net on 50 + 1,924, inference with TTA disabled. The README gives no with/without-pseudo-label Dice. Code: https://github.com/Ziyan-Huang/FLARE22; paper https://openreview.net/forum?id=FNMbe2vLvev. [OPENED README] [GitHub](https://github.com/Ziyan-Huang/FLARE22)
- Test-set Dice 0.8864, NSD 0.9465, 14.59 s inference for the FLARE22 winner. [SNIPPET, search summary] [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-23911-3_16)

### Inferences
- Cheap, evidenced variant: filter pseudo-labels by cross-round variance. No evidence reached that semi-supervised consistency methods beat a tuned nnU-Net; a targeted search found none.

### Gaps
- No benchmark directly testing semi-supervised methods against a strong nnU-Net baseline on lesions. FLARE23/24/25 winner recipes not retrieved.

## 5. Sampling

### Takeaway
Fixed foreground oversampling is the defensible default; evidence for hard-patch mining is negative in the one study found; progressive patch growing speeds training roughly 2x with parity on lung; lung-ROI cropping through lung segmentation + thresholding hurt.

### Cited Findings
- nnU-Net foreground oversampling mechanism and class-uniform centre choice: see section 1 above (code). [OPENED]
- autoPET 2023 (nnU-Net, PET/CT) ablation: reducing oversampling from 50% to 33% (test Dice 0.682 to 0.660 at 1000 epochs, config e vs f) and to 10% did not improve results; 128 vs 192 patch at 1000 epochs: test Dice 0.664 vs 0.682, FP volume 11.9 vs 3.2; removing mirroring raised test Dice 0.680 to 0.685. Note config f differs from e in patch size as well, so the comparison is confounded. [OPENED] [arXiv 2309.13675](https://arxiv.org/html/2309.13675v2)
- Online hard example mining vs fixed oversampling (MS lesions, FLAIR MRI, 32^3 patches, 3-hour budget): "OHEM did not perform better than a predefined fixed oversampling"; uniform sampling with nonzero oversampling performed best. MRI, not CT, but a direct head-to-head. [OPENED] [PMC9672803](https://pmc.ncbi.nlm.nih.gov/articles/PMC9672803/)
- Progressive Growing of Patch Size (Fischer et al., MICCAI 2024) on nnU-Net 3d_fullres, MSD: Lung (Task06) Dice CPS 0.7211 / PGPS 0.7263 / PGPS+ 0.7333 (not significant), runtime 13.55 h vs 5.70 h vs 6.13 h; colon 0.4553 to 0.4925 (significant); hepatic vessel significantly worse for PGPS (0.6837 to 0.6782); about half the runtime on average. Code: https://github.com/compai-lab/2024-miccai-fischer. [OPENED] [arXiv 2407.07853](https://arxiv.org/html/2407.07853v2)
- UniToChest nnU-Net (2D, 144 runs, 22,713 scans with nodules 1-136 mm): small nodules (<10 mm) mean Dice 0.627 vs 0.703 for big; windowing 0.688 and raw HU 0.686 best, lung segmentation + thresholding 0.614 worst, CLAHE 0.672; ResEnc 0.666 vs 0.663 (twice the training time); epochs 100/200/325: 0.656/0.665/0.674; polynomial LR exponent not significant. [OPENED] [PMC11508652](https://pmc.ncbi.nlm.nih.gov/articles/PMC11508652)
- ULS23 baseline places the lesion at the VOI centre (256x256x128) and tried no lesion oversampling. [OPENED] [arXiv 2406.05231](https://arxiv.org/html/2406.05231v2)

### Inferences
- The UniToChest "lung ROI" arm masks the CT with a lung segmentation; that is not the same as cropping to a lung bounding box (LLSB_CFPR crops to the thoracic box, with no ablation shown), so cropping effect is unmeasured.

### Gaps
- No verified lesion-instance-balanced or size-stratified sampling study on CT; no whole-volume vs patch comparison for lung.

## 6. Schedule / optimisation, augmentation, TTA, ensembling

### Takeaway
Longer training gives small gains in available evidence (+0.016 Dice 1000 to 1500 epochs in autoPET 2023; +0.018 from 100 to 325 epochs in UniToChest). nnU-Net ships 4000- and 8000-epoch trainers but I found no study quantifying them.

### Cited Findings
- nnU-Net v2 trainer variants exist for 1, 2, 5, 10, 20, 50, 100, 250, 500, 750, 2000, 4000, 8000 epochs (class `nnUNetTrainer_Nepochs`, subclassing nnUNetTrainer, no performance claims). [OPENED] [nnUNetTrainer_Xepochs.py](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py)
- autoPET 2023: 1000 to 1500 epochs, CV Dice 0.716 to 0.734, test 0.664 to 0.680; LR 0.01 vs 0.001 at 1500 epochs gave much worse (test 0.611 vs 0.680); ensembling and TTA not tested. [OPENED] [arXiv 2309.13675](https://arxiv.org/html/2309.13675v2)
- DA5 trainer (nnU-Net v2): rotation p=0.4, scaling p=0.2, median filter/Gaussian blur p=0.2, noise 0.1, brightness 0.1, contrast 0.2, low-resolution simulation 0.15, gamma 0.1, blank rectangle 0.4, brightness gradient 0.3, local gamma 0.3, sharpening 0.2; code comments flag unresolved rotation-range choices; no recommendation or measured gain in the file. [OPENED code] [nnUNetTrainerDA5.py](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py)
- Winners of 80 ISBI/MICCAI 2021 challenges: augmentation used by 100%, preprocessing 97%, data curation 79%, postprocessing 66%, multi-task learning 63%, multi-stage pipelines 61%; top teams designed methods around the evaluation metric and analysed failure cases. Descriptive, not causal. [OPENED abstract page] [arXiv 2303.17719](https://arxiv.org/abs/2303.17719)
- autoPET III: 5-fold ensemble, mirroring TTA time permitting, no isolated gain. FLARE22 winner disabled TTA for speed. ULS23: TTA raised overall Dice 0.707 to 0.715 for ResEnc+SS (PSUP types 0.649 to 0.661). [OPENED] [ULS23](https://arxiv.org/html/2406.05231v2), [autoPET III](https://arxiv.org/html/2409.09478v2)
- nnU-Net Revisited: scaling gains are dataset-dependent (see first-pass section); original SGD-Nesterov rests on Isensee 2021 benchmarking (secondary, SNIPPET). AdamW on small MS lesions was hypothesised to help; numbers not seen (SNIPPET) [arXiv 2605.12753](https://arxiv.org/pdf/2605.12753).

### Gaps
- No verified numbers for EMA, SWA, 4000-8000-epoch trainers, DA5 vs default, isolated fold-ensemble or TTA gain on lung CT, or distillation.

## 7. Post-processing

### Takeaway
Component-size filtering at a small threshold lowers FP volume at no Dice cost; thresholds above ~20 voxels-equivalent start to cost Dice. Lesion-level classifiers improve precision/F1 with unchanged recall.

### Cited Findings
- autoPET 2023 minimum connected-component size (internal test Dice / FP / FN): 0: 0.685/5.778/6.270; 5: 0.685/5.763/6.510; 10: 0.686/5.744/6.723; 20: 0.686/5.679/7.909; 40: 0.653/5.576/9.195; 80: 0.621/5.312/11.623 (units of the paper's size threshold and mL volumes). Authors pick 10. [OPENED] [arXiv 2309.13675](https://arxiv.org/html/2309.13675v2)
- LLSB_CFPR: cascade of two classifiers, thresholds tuned to maximise Dice at fixed sensitivity; F1 0.680 to 0.712 (see section 1). [OPENED] [PMC12361585](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361585/)
- CC-DiceCE/blob loss as training-time alternatives: see first-pass Losses section. [OPENED]

### Gaps
- No verified study of uncertainty-based filtering on lung CT; no systematic threshold-tuning-for-lesion-F1 study beyond LLSB_CFPR.

## 8. Meta-evidence

### Takeaway
Challenge-winner meta-analysis and Metrics Reloaded argue that voxel Dice alone is inadequate for small/multiple lesions; pair it with lesion-level detection metrics (F-beta, FROC) and boundary NSD.

### Cited Findings
- Metrics Reloaded (Nature Methods 2024): for detection/instance segmentation the counting metric is F-beta (no scores) complemented by AP or FROC with scores; Panoptic Quality offered for instance segmentation; overlap Dice/IoU per instance plus NSD when annotation imprecision matters; exception to Dice when targets are consistently small relative to the grid and references noisy (F-beta). Dice equals pixel-level F1 and shares its small-structure limits. Supplementary Note 2.4 on aggregation was not visible. [OPENED, first 100k of 210k chars] [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Winners study: see section 6. [OPENED] [arXiv 2303.17719](https://arxiv.org/abs/2303.17719)

## FINAL RANKING for a plain patch-based 3D U-Net on lung lesions (inference from verified items)
1. ResEnc L preset, 1500+ epochs, foreground oversampling kept at nnU-Net default or higher (verified: +0.7 to +2.1 DSC on tumour sets; +0.016 Dice 1000 to 1500 epochs; lower oversampling did not help). Lowest risk.
2. Lesion-level post-processing: small component filter (autoPET: no Dice cost at threshold 10-20) and, if precision matters, a lesion FP classifier (+0.03 F1, lesion DSC unchanged).
3. Organ/anatomy auxiliary head and large-scale CT pretraining (autoPET III: +0.3 Dice for the head, ~+1 to +2 for pretraining; PET/CT, lung untested).
4. Synthetic lesions only if labelled lung cases are few: FreeTumor (lung +6.1 Dice, +17.8 small-tumour sensitivity vs SwinUNETR) or LeFusion (nodules, +2 to +5 Dice); expect smaller gains with large real sets (pancreas scaling shows parity at 1,500 real cases).
5. Progressive patch growth: roughly half the training time with lung parity (0.7263 vs 0.7211, n.s.); a compute-saver, not an accuracy gain.
6. Instance losses (CC-DiceCE/blob): brain MRI evidence only, FP risk.
7. Multi-dataset training (MultiTalent): helps colon/pancreas, lung -1.6 Dice; low priority for a lung-only target.
8. Misalignment augmentation: PET/CT only. Skip for CT-only.
9. Semi-supervised/pseudo-labelling: ULS23 lung +0.008 Dice; worthwhile only with large unlabeled sets and filtering.
Not recommended on current evidence: hard-patch mining (OHEM lost to fixed oversampling), lung-mask thresholding preprocessing (lowest Dice in UniToChest).
