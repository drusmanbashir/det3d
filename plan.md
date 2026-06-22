plan a validation of e2e inference pipeline for postprocessing transforms:

create a random image same nifti metadatas a: /media/UB/datasets/lidc/images/lidc_0008.nii.gz. create a corresponding mask fg with a cuboid centered over exactly wherre the largest lesion in this labelmap is centered (use labelmapgeometry in ~/code/label_analysis/) to  get this measurement. Then create a labelbounded image and mask of the saame both centered over the fg lesion but smallr than the source image, somewhere sized between 1/3rd of source  image but larger than patch_size [128,128,64]. store for all above slicer compatible json markup ROI in woorld coordinates which enveloped the fg bbox. tell me when done i wil check both .


once the groundtruth is ready. you will run e2e pipeline, insteda of models and inference, for patch inferer you will use adapter method which will read the input image AND mask (for convenience - in the actual cascade.py mask is inferred) and will compute bounding box of the cuboid accurately and return a dictionary in retinaunet format populating rermaining keys with sensible dummy values.  for the extract_fg method your adapter will simply return the pre-computed labelbounded mask. 

after patch inferer adapter returns its value you will apply the REAL postprocessing transforms (no fake here onward) and confirm that the output is correctly aligned with the larger real world coordinate bounding box you created at the very beginning.
