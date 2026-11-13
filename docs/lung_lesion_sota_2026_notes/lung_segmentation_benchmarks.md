# Lung lesion segmentation SOTA on CT and PET/CT (as of Nov 2026)

Coverage note: this pass was limited to about 15 tool calls. Only items I opened (WebFetch of the page) carry a firm citation. Items seen only in search snippets are marked UNVERIFIED. Large parts of the brief (dataset catalogue, evaluation pitfalls, pipeline design, external validation) are mostly in Gaps and need a follow-up pass.

## Datasets and sizes with licence

### Takeaway
Lung-relevant public sets I verified are ULS23 (CC BY-NC-SA 4.0, lung built from LIDC-IDRI, LNDb, MSD Lung), LNQ2023 (513 CT, TCIA release promised), and MSD Task06 (63 CT). The remaining listed datasets are unchecked.

### Cited Findings
- ULS23 training set has 38,693 lesions; the evaluation set has 775 lesions from 284 patients. — [ULS23 abstract](https://arxiv.org/abs/2406.05231)
- ULS23 lung sources: LIDC-IDRI 750 series / 2,236 lesions; LNDb 208 series / 692 lesions; MDSC-Lung (MSD Task06) 63 series / 70 lesions, all fully annotated. — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2)
- ULS23 dataset licence is CC BY-NC-SA 4.0. LNDb cannot be repackaged, so it is excluded from the archive and participants prepare its VOIs with released code. The test set is hidden (containerised submission on Grand Challenge). — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2)
- LNQ2023 (mediastinal lymph nodes): 513 chest CTs from 513 patients, 383 train (partially annotated), 20 val, 100 test (fully annotated). The authors say it "will be released soon on TCIA as an unrestricted dataset"; no licence stated in the paper. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- MSD Task06 Lung has 63 CT scans (51 train / 12 test split used in one benchmark). NSCLC-Radiomics (Lung1): 304 of 422 patients used (246 train / 58 test). — [Foundation-model lung CT benchmark](https://arxiv.org/html/2505.01239v1)
- FLARE 2025 pan-cancer Task 1 Subtask 2 converts RECIST annotations to volumetric tumour masks; validation and test sets cover abdominal cancers (liver, kidney, pancreas, colon, gastric). Lung data (LIDC, LNDb) appear only in training lists. — [Lite ENSAM](https://arxiv.org/html/2511.01600v1)

### Inferences
- For a lung-only study, ULS23 lung (about 3,000 lesions, mostly nodules from LIDC/LNDb) is the largest openly documented lesion-level set verified here, but it is nodule-dominated, not mass-dominated.

### Gaps
- Not opened: NSCLC-Radiogenomics, LIDC licence, LUNA16/LUNA25, MELA, DeepLesion, Lung-PET-CT-Dx, RIDER, NLST sets, autoPET I-IV sizes/licences, CancerVerse/AbdomenAtlas lung classes, TotalSegmentator lung nodule class, any 2025-2026 new lung tumour sets.

## Best results per dataset and nnU-Net baselines

### Takeaway
ULS23 shows that a residual-encoder nnU-Net (ResEnc) beats plain nnU-Net on lung lesions (0.753 vs 0.719 test Dice), and on small MSD/NSCLC sets nnU-Net 2D/3D is a very strong baseline (tumour Dice 0.85 to 0.90) that a fine-tuned MedSAM 2 only matches. I found no verified leaderboard-level SOTA for MSD Task06 or for lung-only splits of the large challenges.

### Cited Findings
- ULS23 lung, held-out training data (N=267, from MDSC-Lung, LIDC-IDRI, LNDb), Dice: nnU-Net 0.753 ± 0.137; ResEnc 0.751 ± 0.147; ResEnc+SS (semi-supervised) 0.754 ± 0.149. — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2)
- ULS23 lung, challenge test set Dice: nnU-Net 0.719 ± 0.243; ResEnc 0.753 ± 0.222; ResEnc+SS 0.761 ± 0.211. With TTA: 0.719 / 0.760 / 0.764. — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2)
- ULS23 all categories, test Dice: nnU-Net 0.651; ResEnc 0.700; ResEnc+SS 0.703 (the abstract's overall baseline figure is 0.703 ± 0.240). Lymph-node category: 0.646 / 0.685 / 0.685. — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2), [abstract](https://arxiv.org/abs/2406.05231)
- Baseline settings: nnU-Net default resampling off, fixed 256x256x128 VOIs (baseline described in search summary only; see UNVERIFIED below).
- MSD Task06 tumour Dice (scratch / fine-tuned, 12 test cases): nnU-Net 2D 0.8736; 3D lowres 0.8650; 3D fullres 0.8487; MedSAM 0.7230; MedSAM 2 box prompt 0.8770; MedSAM 2 point prompt 0.7974. U-Net and DeepLabV3 from scratch fail (tumour Dice 0.009 to 0.018). — [Foundation-model lung CT benchmark](https://arxiv.org/html/2505.01239v1)
- NSCLC-Radiomics (Lung1) tumour Dice (58 test): nnU-Net 2D 0.9039; 3D lowres 0.7386; 3D fullres 0.7023; MedSAM 2 box 0.9091; point 0.7974. MedSAM 2 box with 25% of training data: 0.9006 (Lung1), 0.8780 (Task06). Note: box prompts require a user-supplied lesion box, and the 3D nnU-Net's lower score than 2D on Lung1 is unexplained in the source. — [Foundation-model lung CT benchmark](https://arxiv.org/html/2505.01239v1)
- LNQ2023 test, median Dice / ASSD: Skeleton Suns 71.2 / 2.95 mm (rank 1); IMR 70.0 / 4.15; CompAI 69.0 / 5.05; Hilab 61.0; IMI 61.2; SKJP 43.3. The text gives Skeleton Suns rank score 2.7, the table 2.1 (internal inconsistency in the paper). — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- autoPET III (whole-body FDG and PSMA, not lung-specific): 5-fold CV Dice: default nnU-Net 57.61 (FP vol 19.32, FN vol 15.69); ResEncL 65.31; final LesionTracer model 68.40 (FP vol 7.82, FN vol 10.35). By tracer, final: FDG 77.28, PSMA 60.01. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- A different autoPET III entry (placed third, per the author) reports default nnU-Net CV Dice 0.68, preliminary-test 0.77; best variant (z-score PET normalisation) 0.80 preliminary test. — [AlphaXiv summary of arXiv 2605.08161](https://www.alphaxiv.org/abs/2605.08161.md) (secondary summary page)
- FLARE 2025 Task 1 Subtask 2 (RECIST-to-volume, abdominal): Lite ENSAM hidden-test DSC 60.7, NSD 63.6; public validation 76.1 ± 16.3 DSC. Not lung. — [Lite ENSAM](https://arxiv.org/html/2511.01600v1)

### Inferences
- The default-to-ResEnc gap is about +3 to +5 Dice points on ULS23 lung test and +7.7 points on autoPET III CV. Semi-supervised/pseudo-label add-on gives about +0.8 to +1 point on ULS23 lung.
- The 25-point swing between nnU-Net 2D and 3D on Lung1 (0.904 vs 0.702) suggests single-split small-sample variance or an evaluation artefact; do not treat it as an architecture ranking.

### Gaps
- RESOLVED in pass 2 (verified via RSNA page, see Pass 2 section 6): a Stanford Radiology 2025 3D U-Net ensemble trained on 1,504 CTs from 1,295 patients reached median model-physician Dice 0.77 vs 0.80 inter-physician on 100 single-tumour scans (private data). Source seen: [PubMed 39835976](https://pubmed.ncbi.nlm.nih.gov/39835976/) (page blocked by cookie wall; not read).
- PARTLY RESOLVED in pass 2: the 2026 PET/CT preprint is a Vietnamese NSCLC benchmark; the 83.4 figure is from a search-result abstract only (preprints.org returned 403), so it stays UNVERIFIED at page level.
- No verified numbers for LUNA25, LNDb alone, LIDC segmentation, NSCLC-Radiogenomics, Lung-PET-CT-Dx, MELA, autoPET lung-cancer subset, autoPET IV final leaderboard, ULS23 final leaderboard and winners, FLARE 2024/2025 lung class.

## Winning challenge entries 2024-2026

### Takeaway
Verified winners (autoPET III model-centric, LNQ2023 top three) all use nnU-Net with a residual encoder and extra supervision (multi-dataset pretraining, organ heads, TotalSegmentator pseudo-background, lung/mediastinum cropping). ULS23, autoPET IV, FLARE and LUNA25 winners were not verified.

### Cited Findings
- autoPET III winner (Team LesionTracer, DKFZ): nnU-Net 3d_fullres, ResEncL preset, two heads (lesion + organ). Pretraining: MultiTalent-style, first CT only then CT+MR+PET, 4000 epochs, patch 192^3, batch 24, 1 mm resampling, z-score, dataset sampling inversely proportional to sqrt(image count). Fine-tuning on autoPET III with lr 1e-3. Final: batch 3, 1500 epochs, resampling [3, 2.04, 2.04] mm (better than 1 mm iso). Dice loss without smoothing term, softmax heads with equal weights. Misalignment augmentation shifting PET vs CT (up to 5 deg, up to [2,2,0] voxels). 5-fold ensemble, step size 0.6, partial mirroring TTA to fit 5 min/case. No lung-specific results; no numeric test scores in the paper. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- Ablations of that paper (CV): plain nnU-Net 57.61 -> ResEncL 65.31 -> final 68.40 Dice. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- LNQ2023 1st (Skeleton Suns): nnU-Net 3d_fullres with residual encoder, trained on LNQ plus refined TCIA CT Lymph Node annotations; TotalSegmentator structures used as background, unlabelled regions masked from loss; BodyPartRegression restricted training to the mediastinum; 5-fold ensemble, no TTA. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- LNQ2023 2nd (IMR): lung-mask and airway-map guided cropping to mediastinal VOI; stage 1 nnU-Net from scratch on refined external data; stage 2 pseudo-labels on LNQ train data, fine-tune 300 epochs; remove components smaller than 5 mm-radius sphere. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- LNQ2023 3rd (CompAI): single nnU-Net trained on LNQ, refined TCIA lymph nodes and three public NSCLC sets (Radiomics, Radiogenomics, Interobserver); TotalSegmentator pseudo-background; crop to lung bounding box; the team's later report says their 10 mm size post-processing hurt. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- LNQ2023 organisers: the top three combined full supervision from external fully annotated data with weak supervision from LNQ; weak-label-only methods did worse (search-summary statement, consistent with the table above). — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- autoPET IV Task 1 (interactive) DKFZ entry extends the autoPET III nnU-Net pipeline with user clicks as input channels; distance-transform click encoding beat Gaussian kernels in their CV. Final standings not found. Code: github.com/MIC-DKFZ/autoPET-interactive. — [arXiv 2508.21680](https://arxiv.org/abs/2508.21680v1) (abstract seen via search; not opened in full)

### Inferences
- Common recipe among verified winners: nnU-Net ResEnc + large external/pretraining data + masking or pseudo-labelling of unlabelled structures + 5-fold ensemble. No transformer or foundation-model entry won any verified track.

### Gaps
- ULS23 final results and winners (forum post "Final Results" seen in search only, not opened); autoPET III final test numbers; autoPET IV winner; FLARE 2024/2025 winners (only one participant, Lite ENSAM, seen); LUNA25 challenge; MICCAI 2025/2026 lung challenges.

## Lung-specific pipeline design

### Takeaway
Verified design ingredients come from LNQ2023 (lung-mask/bounding-box crop, mediastinum restriction, TotalSegmentator pseudo-background) and autoPET III (PET/CT misalignment augmentation, organ auxiliary head, anisotropic resampling). Nothing verified on juxtapleural lesions, atelectasis, part-solid lesions or multi-window input.

### Cited Findings
- Lung ROI cropping used by IMR (lung mask + airway map to mediastinal VOI) and CompAI (lung bounding box); body-part regression to restrict to mediastinum used by the winner. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- Auxiliary organ head helps lesion Dice: organ supervision before final retraining scored 68.33 CV Dice; down-weighting organ loss slightly raised lesion Dice but increased FP/FN volume. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- PET/CT: random PET-to-CT spatial misalignment augmentation was part of the autoPET III winner. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- Size-threshold post-processing can hurt (CompAI 10 mm filter), vs helpful for IMR (5 mm). — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)

### Gaps
- Cascade detect-then-segment, small nodule vs mass handling, juxtapleural/juxtavascular, atelectasis confusion, ground-glass/part-solid, multi-window CT: no sources opened.

## Evaluation pitfalls and recommended metrics

### Takeaway
Only indirect evidence verified: ULS23 and autoPET report SD-heavy per-case Dice, and autoPET adds FP/FN volume. Metrics Reloaded and CC-Dice/panoptic quality were not opened.

### Cited Findings
- autoPET III reports false-positive and false-negative volume alongside Dice (e.g. FP vol 19.32 to 7.82 across the nnU-Net to final-model steps), which Dice alone hides. — [LesionTracer paper](https://arxiv.org/html/2409.09478v2)
- ULS23 lung test Dice SD is 0.21 to 0.24 versus 0.14 on held-out training data, indicating heavy per-case variance on the hidden set. — [ULS23 paper HTML](https://arxiv.org/html/2406.05231v2)

### Gaps
- Metrics Reloaded, lesion-wise F1, CC-Dice, panoptic quality, large-tumour bias: not retrieved.

## External validation and radiotherapy GTV studies

### Takeaway
Not verified in this pass.

### Cited Findings
- RESOLVED in second pass (see Pass 2, section 6): Stanford study verified from the RSNA page.

### Gaps
- Multi-centre 2024-2026 studies and GTV auto-contouring papers not searched.

---

# PASS 2 (appended 2026-11-13): gap-filling, only pages opened

Method note: pages were read via WebFetch or by downloading the PDF and reading its text. Items from search snippets only are tagged SNIPPET-ONLY and should not be relied on. Zenodo record pages and some publisher pages returned 403/429; those are listed as unread.

## 1. ULS23 final standings and top-team methods

### Takeaway
The organisers' announcement names team U-Mamba as ULS23 winner (12 April 2024). The numeric ranking and the winner's method write-up could not be opened. Per-lung numbers exist only for the organisers' nnU-Net variants (see first-pass notes).

### Cited Findings
- Announcement: "Congratulations to team U-Mamba for winning the ULS23 challenge with their novel architecture!" (12 April 2024); results were announced 10 April 2024 after a grace period. — [ULS23 announcements](https://uls23.grand-challenge.org/announcements/)
- ULS23 site notes (11 Oct 2026): two additional leaderboards with extended inference time (up to 2 h) were opened, with LesionLocator and nnInteractive results added; the challenge and baseline paper was published in Medical Image Analysis (28 Mar 2025). The leaderboard tables are rendered by JavaScript and were empty in my fetch, so scores were not read. — [ULS23 home](https://uls23.grand-challenge.org/)
- ULS23 scoring: ChallengeScore combines segmentation Dice, long/short-axis SMAPE and consistency Dice (leaderboard column headers). — [ULS23 open test leaderboard](https://uls23.grand-challenge.org/evaluation/test-phase-leaderboard/leaderboard/)
- ULS23 training data on Zenodo: part 1 novel annotations (DeepLesion3D, Radboudumc Bone, Radboudumc Pancreas), part 2 kits21/LIDC-IDRI/LiTS, part 3 MDSC tasks 6/7/10, NIH-LN, CCC18, parts 4-6 DeepLesion. — [ULS23 GitHub](https://www.github.com/MJJdG/ULS23)
- ULS23 baseline is applied by the organisers in a longitudinal study: it assumes a lesion-centred fixed-size VOI, disables nnU-Net resampling, and collapses when the lesion is displaced from the VOI centre. — [Rocholl et al. 2025](https://arxiv.org/pdf/2507.19230) (read from PDF text)
- A Vanderbilt student-team report compared nnU-Net v2, DeepLabV3+, Medical Transformer, SwinUNet and TransUNet on the ULS23 bone-lesion subset and settled on TransUNet; it is a small-scale study, not a top entry. — [Shi et al. 2025](https://arxiv.org/pdf/2502.10608)

### Gaps
- SNIPPET-ONLY (search summary of the forum post, which returned 403 to me): scores U-Mamba 0.735, Fungus Among Us 0.731, Prof. Ching-Wei Wang's team 0.730; a third-party aggregator lists U-Mamba average Dice 0.708 and baseline challenge score 0.729. Not opened, so treat as unverified.
- U-Mamba team write-up, architecture details and lung-specific Dice for top entries not found. Note the paradox implied by the snippet (baseline 0.729 vs winner 0.735 is a gap of about 0.6 points); verify before using.

## 2. autoPET IV (2025) outcome and autoPET lung-cancer numbers

### Takeaway
autoPET IV Task 1 (interactive PET/CT, FDG+PSMA) best mean Dice at the last interaction step was 0.74 (team BIRTH); the default nnU-Net baseline reached 0.63 with 10 ml false-positive volume. No lung-subgroup numbers are in the overview paper. Independent lung-cancer evaluations of autoPET models exist and show over-prediction.

### Cited Findings
- autoPET IV overview: 191 registered participants, 14 teams; final phase 5 teams for Task 1 and 2 for Task 2; six of seven algorithms are nnU-Net 3D U-Nets, one is a masked-autoencoder + BiXLSTM-ResNet. Task 1 has about 30,000 training and 3,000 test lesions; Task 2 (longitudinal CT) about 7,500 training and 4,500 test lesions. — [autoPET IV overview preprint](https://www.researchsquare.com/article/rs-10893740/v1.pdf) (read from PDF text; the HTML page shows only a title)
- Task 1 winner: team BIRTH, mean DSC@last 0.74, FNV 2.17 ml, FPV 1.30 ml; top four DSC@last 0.71 to 0.74. The nnU-Net baseline ranked fifth with DSC@last 0.63, FNV 3.00 ml, FPV 10.0 ml; the SW-FastEdit baseline was beaten by all entries. The best entry beat the autoPET III top algorithm by 5.9% (FDG-LMU) to 21.9% (PSMA-UKT) Dice and cut FPV by up to 77.1%. — [autoPET IV overview preprint](https://www.researchsquare.com/article/rs-10893740/v1.pdf)
- Task 1 ranking weights: DSC@last 0.25, FPV@last 0.125, FNV@last 0.125, AUC-DSC 0.25, AUC-FPV 0.125, AUC-FNV 0.125; bootstrap 1000x for ranking robustness; lesion-wise sensitivity analysed by lesion size and SUVmax deciles. — [autoPET IV overview preprint](https://www.researchsquare.com/article/rs-10893740/v1.pdf)
- Task 2 (longitudinal CT): LesionLocator best, DSC@last 0.38, FNV 1.09 ml, FPV 0.63 ml; nRocholl (out of competition, organiser) 0.30; baseline 0.25. — [autoPET IV overview preprint](https://www.researchsquare.com/article/rs-10893740/v1.pdf)
- DKFZ autoPET IV entry (LesionLocator-based, Task 1 and 2 PDFs): extends the autoPET III winning nnU-Net with click channels; distance-transform click encoding beats Gaussian kernels, extra training data helps Dice and FNV. — [LesionLocator autoPET IV Task 1 PDF](https://www.autopet.org/docu/autopetiv/Task1/2_LesionLocator_autoPET4_Task1_LesionLocator.pdf)
- Tsinghua/Subtle entry: nnU-Net ResEnc with tracer classification, organ supervision (organ labels for lung, brain, heart etc.), simulated click sampling; Dice:CE 2:1 weighting with no smoothing term; the paper notes a high-Dice version with higher FPV. — [Huang et al., arXiv 2509.02402](https://arxiv.org/pdf/2509.02402)
- External lung-cancer test of the top autoPET III model: 306 treatment-naive NSCLC patients, single centre, mean Dice 0.64 vs manual segmentation, lesion sensitivity 95.8%, precision 87.5%; most frequent error was false-positive M-category lesions (n=196; 35.7% benign, 34.7% non-oncologic); overprediction, hilar undersegmentation drove upstaging; UICC stage matched in 207/306 (67.7%). — [DZL/EJNMMI abstract](https://dzl.de/publication/10-1007-s00259-025-07677-2/)
- GLOW-FDG (2026 preprint, nnU-Net ResEncL, trained on 1,563 FDG PET/CT scans, evaluated on 185 external scans): lesion-wise F1 on nonmetastatic lung cancer 0.97 (table) vs autoPET DKFZ 0.87, autoPET IKIM 0.85, onlyPET 0.83 (the prose says 0.98 vs 0.88; table and text differ); oligometastatic lung cancer F1 0.85 (table) vs 0.79 DKFZ; oligometastatic lung median lesion Dice 0.69. Lung training data in this model: 67 NSCLC-Radiogenomics scans, 61 SAKK scans, plus lung cases from autoPET. — [GLOW-FDG arXiv 2607.03931](https://arxiv.org/pdf/2607.03931) (read from PDF text)
- Independent cross-cancer test: nnU-Net trained on an oesophageal cohort scored Dice 3.4 or lower on an external lung PET/CT cohort; autoPET-only training gave 51.6 on the 54-patient lung set; combined 52.9; the paper concludes that dataset diversity outweighs architecture. — [Ghosh et al., arXiv 2508.18612](https://arxiv.org/pdf/2508.18612)

### Inferences
- PET-driven models detect lung lesions well (sensitivity 95.8%) but overall Dice sits around 0.64 externally, driven by FP and by hilar/mediastinal confusion. Lung-specific top-line numbers are 0.5 to 0.7 Dice on external PET/CT, well below the 0.8+ seen on single-centre internal sets.

### Gaps
- No autoPET III/IV per-lung-cancer subgroup table found. BIRTH team method not found. autoPET III test-phase final numbers still not read. EmergentMind claim that a Kirchhoff et al. entry "secured first place" in autoPET IV: SNIPPET-ONLY, not verified.

## 3. FLARE 2024/2025 pan-cancer task

### Takeaway
No organiser summary for FLARE 2024/2025 was found. The only verified entry is Lite ENSAM (no lung results). The FLARE 2025 pan-cancer subtask is RECIST-prompted and abdominal-focused, so it is of limited use as a lung benchmark.

### Cited Findings
- FLARE25 Task 1 Subtask 2 entry Lite ENSAM: hidden test DSC 60.7, NSD 63.6; public validation DSC 76.1 ± 16.3; no lung numbers; lung sets only in training. — [Lite ENSAM](https://arxiv.org/html/2511.01600v1) (opened in first pass)
- FLARE 2023 summary (prior year) is the nearest organiser report: cascaded winner reached 92.3% Dice organs and 64.9% lesions on hidden multi-national test. — SNIPPET-ONLY via search summary of [arXiv 2408.12534](https://arxiv.org/abs/2408.12534v1); not opened.

### Gaps
- FLARE 2024 and 2025 winners, recipes and lung lesion class results not located; LETT-NeXt (arXiv 2606.30108) uses the same data per a snippet but was not opened.

## 4. Datasets not yet covered (size, licence)

### Takeaway
Licences vary: LIDC-IDRI CC BY 3.0; NSCLC-Radiomics CC BY-NC 3.0; NSCLC-Radiogenomics CC BY 3.0; Lung-PET-CT-Dx CC BY 4.0 (boxes only); LUNA25 CC BY 4.0 (malignancy classification, not masks). The largest public voxel-level lung tumour mask releases verified are NLSTseg (605 patients) and the NLST/IDC annotation sets.

### Cited Findings
- LIDC-IDRI: 1,010 subjects (page also says 1,018 cases), 1,308 studies, 244,527 images; radiologist XML annotations for nodules at least 3 mm, nodules under 3 mm, and non-nodules at least 3 mm, with nodule contours accessible via pylidc; CC BY 3.0; nodule total not stated on the page. — [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- NSCLC-Radiomics (Lung1): 422 NSCLC patients; manual 3D delineations in RTSTRUCT and SEG for primary GTV ("GTV-1") plus lung, heart, esophagus, made by a radiation oncologist; CC BY-NC 3.0 (non-commercial). — [TCIA NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)
- NSCLC-Radiogenomics: 211 subjects with CT, PET/CT and SEG tumour segmentations (BAMF under the NCI AIMI Annotations initiative, hosted outside TCIA), CC BY 3.0. — [TCIA NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/)
- Lung-PET-CT-Dx: 355 subjects (436 studies, 1,295 series), CT and PET, annotations are PASCAL VOC bounding boxes from five thoracic radiologists (not masks), CC BY 4.0; classes adenocarcinoma, small cell, large cell, squamous. — [TCIA Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
- LUNA25 public training set: 4,069 low-dose CTs, 2,120 patients (NLST), 555 malignant and 5,608 benign nodules; CC BY 4.0; 221 GB; the page does not describe segmentation masks, so LUNA25 is a malignancy-risk task, not a segmentation task. — [Zenodo LUNA25](https://zenodo.org/records/14223624)
- LUNA25 results (Radiology: AI 2026): best AI AUC 0.78 (0.73 to 0.84) vs 65 radiologists 0.70 (0.65 to 0.74), external test 156 malignant and 312 benign nodules. — SNIPPET-ONLY via [RSNA record](https://pubs.rsna.org/doi/10.1148/ryai.260179); page not opened.
- NLSTseg (Scientific Data 2025): 605 patients, 715 voxel-level lesions (662 tumours, 53 nodules, volumes 0.03 to 372.21 cm3) on NLST low-dose CT; a 2D U-Net IoU 0.95 on its training data is reported in the abstract; licence not shown on the catalogue page I opened (DOI 10.1038/s41597-025-05742-x unread). The IDC harmonised version covers 581 patients; segmentations were by a researcher plus a radiologist and verified by a radiation oncologist and a radiologist. — [NLSTseg record](https://library.kab.ac.ug/Record/doaj-art-2e1d2e8800164be79ffcbf3885b83164/Description), [Krishnaswamy et al., NLST annotations](https://arxiv.org/pdf/2609.10858)
- Additional NLST annotation sets (Bodard et al.): 2,004 DICOM SEG objects for 501 patients, 20 radiologists with at least 4 reads per case; Sybil bounding boxes; IDC also hosts TotalSegmentator segmentations (not lung tumours) for 26,194 NLST patients. — [Krishnaswamy et al.](https://arxiv.org/pdf/2609.10858)
- Cross-check on location: about 26% of NLSTseg tumour lobe locations disagree with the NLST clinical lobe label in that paper's analysis; use caution. — [Krishnaswamy et al.](https://arxiv.org/pdf/2609.10858)
- ULS23 licence CC BY-NC-SA 4.0 and lung sources already in first-pass notes. UniToChest (lung lesions, used for nnU-Net training in Mathai et al.) is mentioned but its size and licence were not read.
- CancerVerse/AbdomenAtlas cover abdominal and pelvic tumour types (lung not among 13 types); expert voxel masks still being released with manual annotation expected by Sept 2026. — [CancerVerse (HF)](https://huggingface.co/datasets/BodyMaps/CancerVerse) (SNIPPET-ONLY, page not opened)

### Inferences
- For commercial use, NSCLC-Radiomics (NC) and ULS23 (NC-SA) are restrictive; LUNA25, Lung-PET-CT-Dx and LIDC are permissive.

### Gaps
- MELA: searches returned nothing; unconfirmed whether it exists. DeepLesion licence and size not read. TotalSegmentator lung nodule class and its accuracy not found: the only related result is that TotalSegmentator anatomical priors did not help lung lesion segmentation (below). LNDb licence not read beyond the ULS23 note. Zenodo records for NLSTseg unreadable (403).

## 5. Lung-specific pipeline design with evidence

### Takeaway
Evidence verified is thin. Size stratification is large: nnU-Net on UniToChest gets Dice 77.1 for lesions over 1 cm but 52.5 for 3-10 mm; anatomical priors from TotalSegmentator did not help; PET adds modest gain in a small SBRT study.

### Cited Findings
- UniToChest nnU-Net (full-res, 3D): Dice all lesions 74.5 (19.1), 3-10 mm 52.5 (26.9), over 1 cm 77.1 (20.3); HD 21.3 / 40.5 / 11.7 mm. Detection: precision 65.6, sensitivity 49.9, F1 56.7 overall; for lesions over 1 cm precision 71.3, sensitivity 68.4, F1 69.8; for 3-10 mm sensitivity 42.5. — [Mathai et al. 2025](https://arxiv.org/pdf/2504.06924) (read from PDF text)
- Adding 28 TotalSegmentator anatomy priors as extra labels lowered Dice (72.8 vs 74.5 all; 49.7 vs 52.5 for 3-10 mm), significant at p below .001, though HD was lower (19.2 vs 21.3 mm) because predictions encroached less on adjacent anatomy. — [Mathai et al. 2025](https://arxiv.org/pdf/2504.06924)
- PET plus CT co-segmentation on 60 SBRT NSCLC PET/CT pairs (22 test): 3D U-Net CT-only Dice 0.811 vs co-segmentation 0.861 on CT contours; PET-only 0.794 vs 0.828 with co-segmentation. Reference was a STAPLE consensus of three oncologists. — [Zhong et al., PMC6527327](https://pmc.ncbi.nlm.nih.gov/articles/PMC6527327)
- nnU-Net PET vs CT lung tumour models from the PET-Plan/STRIPE trials: PET model Dice 0.83 ± 0.13 (4D PET, PET-Plan), 0.83 ± 0.11 on 19 external 3D PET; CT model 0.59 ± 0.24 on 19 external 3D CT with PPV 0.77 (undersegmentation), 0.61 ± 0.28 on 80 4D CT. Test sets differ, so the PET-over-CT gap (about 0.2) is indirect. Experts scored 0.55 to 0.64 against the reference in the retrospective cohort. — [Development and evaluation of two open-source nnU-Net models, PMC11399280](https://pmc.ncbi.nlm.nih.gov/articles/PMC11399280/)
- LNQ2023 entries used lung-mask/airway guided cropping and TotalSegmentator pseudo-background; see section on winning entries. — [LNQ2023 paper](https://arxiv.org/html/2408.10069v1)
- RAU-Net lung GTV (Sci Rep 2025): Dice about 77% and 74% on two datasets. — SNIPPET-ONLY, not opened ([PubMed 40301479](https://pubmed.ncbi.nlm.nih.gov/40301479/)).

### Inferences
- Detect-then-segment is implicit in ULS23 (lesion-centred VOIs from a click/box) and in LNQ2023 (crop to mediastinum); size-dependent difficulty is large, so nodule-scale and mass-scale performance should be reported separately.
- A PET-to-CT gap of a few Dice points in matched settings (0.81 to 0.86) is far smaller than cross-study gaps; the larger win is PET's lower false-positive rate on atelectasis and consolidation, but I found no study quantifying that.

### Gaps
- Juxtapleural/juxtavascular handling, atelectasis/consolidation confusion rates, part-solid/ground-glass segmentation accuracy, multi-window input ablations: no verified source found.

## 6. Radiotherapy GTV and multi-centre validation; resolving flagged items

### Takeaway
Multi-site lung GTV models land at Dice 0.70 to 0.77 externally, against inter-physician Dice around 0.80. The Stanford study is verified; the 2026 PET/CT preprint remains abstract-level only.

### Cited Findings
- Stanford (Kashyap et al., Radiology Jan 2025; RSNA release): 3D U-Net ensemble, 1,504 CTs with 1,828 tumours for training, 150-scan test (internal plus external); detection sensitivity 92% and specificity 82%; on 100 single-tumour scans median model-physician Dice 0.77 vs physician-physician 0.80, three physician contours per case; underestimates volume, worse on very large tumours; retrospective; external test size not given. — [RSNA news](https://www.rsna.org/news/2025/january/ai-aids-lung-tumor-detection-on-ct)
- iSeg multicentre lung GTV: 3D U-Net, 739 pre-treatment CTs from 9 clinics in 2 health systems; median Dice 0.73 (IQR 0.62 to 0.80) internal, 0.70 and 0.71 on two external cohorts (n=161 and n=102). — SNIPPET-ONLY (abstract text on a library catalogue page found via search; not opened).
- 2026 Vietnamese NSCLC PET/CT benchmark preprint (Preprints.org, 202607.0245, not peer reviewed): fine-tuning an AutoPET-II-initialised nnU-Net gives Dice 83.4 ± 6.5, HD95 5.1 mm, precision 89.6%; +6.8 points over nnU-Net v2 from scratch with 37.5% less training time; MTV correlation 0.96. — SNIPPET-ONLY: the page returned 403; the figures come from the search tool's abstract summary.
- Domain shift in lung GTV: an open-data Dutch-trained lung GTV model performed poorly initially in Indian clinical data and improved after transfer learning. — SNIPPET-ONLY ([PMC10860512](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10860512/)).
- Cross-cancer PET/CT: see section 2 (Ghosh et al., lung external Dice about 52).

### Gaps
- The 2026 preprint needs a direct read; most GTV studies with inter-observer comparison (2024-2026) were not opened.

## 7. Evaluation and metrics

### Takeaway
Metrics Reloaded (Nature Methods 2024) explicitly warns against Dice on very small structures, recommends instance-aware detection metrics (F-beta or Panoptic Quality) plus separate segmentation metrics on matched instances, and boundary-based metrics for small structures with noisy reference. autoPET reports false-positive/negative volume and lesion sensitivity per size decile; ULS23 reports Dice with SMAPE size errors.

### Cited Findings
- Metrics Reloaded is a Nature Methods 2024 framework (arXiv 2206.01653 v8, Feb 2024) with a problem fingerprint; categories include object detection, semantic and instance segmentation. — [arXiv 2206.01653](https://arxiv.org/abs/2206.01653)
- From the paper text: Dice is a poor choice for particularly small structures; for instance segmentation select detection metrics (recommended alternative to F-beta is Panoptic Quality, PQ, which merges detection and segmentation quality of matched instances), then segmentation metrics on TP instances; for small structures (FP3.1) consider treating the problem as object detection, add boundary-based metrics where the reference is noisy, and use a lower localisation threshold; FPPI and sensitivity are listed for detection. Hierarchical data (several scans per patient or centre) must be respected when aggregating. — [Metrics Reloaded full paper](https://arxiv.org/pdf/2206.01653) (read from PDF text)
- autoPET IV analyses lesion-wise sensitivity by lesion size and SUVmax interdecile bins and reports FP and FN volumes in ml per patient. — [autoPET IV overview preprint](https://www.researchsquare.com/article/rs-10893740/v1.pdf)
- Per-size effect in lung: Dice 52.5 for 3-10 mm vs 77.1 over 1 cm; detection sensitivity 42.5% vs 68.4% (UniToChest). — [Mathai et al. 2025](https://arxiv.org/pdf/2504.06924)
- Large SD of Dice on the hidden ULS23 test lung set (0.21 to 0.24). — [ULS23 HTML](https://arxiv.org/html/2406.05231v2)
- Lesion-wise Dice/F1 gating (SNIPPET-ONLY): a 2026 MS preprint argues voxel metrics are biased to large lesions, and a stroke-challenge forum thread noted that matching gates (IoU threshold) can force lesion Dice to 0. — [arXiv 2605.09666](https://arxiv.org/pdf/2605.09666) and [grand-challenge forum thread](https://grand-challenge.org/forums/forum/ischemic-stroke-lesion-segmentation-challenge-2024-722/topic/sanity-check-with-all-1s-image-passes-and-yields-dice-of-0-2625); neither opened.

### Gaps
- CC-Dice definition and a lung-specific validation were not read; no verified lung benchmark uses PQ.

## Overall inference for the size of the gap (pass 2)
- Verified nnU-Net-family gaps: plain nnU-Net to ResEnc is +3.4 Dice on ULS23 lung test (0.719 to 0.753), +7.7 Dice on autoPET III CV (57.6 to 65.3); autoPET IV best interactive Dice (0.74 with clicks) vs nnU-Net baseline 0.63 with 10 ml FPV.
- The larger gaps are domain and size effects: external lung PET/CT Dice about 0.52 to 0.64, CT-only external lung 0.59, size 3-10 mm Dice about 0.5, versus 0.77 to 0.83 on within-site or large lesions. Ingredients that closed gaps in verified sources: residual-encoder preset, multi-dataset pretraining, extra training data diversity, organ auxiliary supervision, masking unlabeled structures. Foundation-model and transformer entries won no verified track (U-Mamba, an SSM architecture, won ULS23 by the organisers' announcement but its margin and recipe are unverified).
