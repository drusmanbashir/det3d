# %%
import pickle, numpy as np, SimpleITK as sitk
import pandas as pd
import torch
from label_analysis.geometry import LabelMapGeometry
from label_analysis.geometry_itk import LabelMapGeometryITK
from label_analysis.geometry_pt import LabelMapGeometryPT
from utilz.helpers import pp
from utilz.imageviewers import ImageMaskViewer
# %%

if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    
    case = "lidc_0583"
    img = sitk.ReadImage(f"/media/UB/datasets/lidc_all/images/{case}.nii.gz")
    lm  = sitk.ReadImage(f"/media/UB/datasets/lidc_all/lms/{case}.nii.gz")
    inst = sitk.ReadImage(f"/r/datasets/nndet_data/Task012_LIDC/raw_splitted/labelsTr/{case}.nii.gz")
# %%
    ImageMaskViewer([img, inst],'im')
    z = np.load(f"/r/datasets/nndet_data/Task012_LIDC/preprocessed/D3V001_3d/imagesTr/{case}.npz")
    car = pickle.load(open(f"/r/datasets/nndet_data/Task012_LIDC/preprocessed/D3V001_3d/imagesTr/{case}.pkl","rb"))
    car
# %%
# %%
    L1 = LabelMapGeometryITK(ignore_labels=[1], li=lm)
    L2 = LabelMapGeometryITK(ignore_labels=[1], li=inst)
    L1.nbrhoods
    L2.nbrhoods
    
# %%
    fn = "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000/bboxes/lidc_0076.csv"
    fldr = Path("/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000/bboxes/")
    fns = list(fldr.glob("*.csv"))
# %%
    pat = "bbox_xyzxyz_"
    for fn in fns:
        df = pd.read_csv(fn)
        cols_out = []
        for col in df.columns:
            if pat in col:
                col_new = col.replace(pat,"bbox_extended_")
            else:
                col_new = col
            cols_out.append(col_new)
        df.columns = cols_out
        df.to_csv(fn, index=False)

# %%
