import cv2
import argparse
import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path
import pyheif
from numpy.typing import NDArray
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from utils.prep import prep_case
from utils.parallel import mod_parallel
from utils.macenko import NumpyMacenkoNormalizer
import polars as pl

def read_image(path: str, scale: float = 1/4) -> NDArray:
    """
    path: path to image
    scale: resized scale (default: 0.25)
    """
    # separated flow for heic
    if path.split(".")[-1].lower() == "heic":
        # Read HEIC file
        heif_file = pyheif.read(path)
        # Convert to PIL Image
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        img = np.asanyarray(img)
    else:
        img = Image.open(path)
        img = np.asanyarray(img)
    
    H, W = img.shape[:-1]
    new_H = int(H * scale)
    new_W = int(W * scale)
    img = cv2.resize(img, (new_W, new_H))
    return img

def get_dab(img: NDArray) -> NDArray:
    """
    img = IHC image (rgb)
    """
    # color deconvolution (CD) step to get grayscale image of DAB
    # Macenko CD is based on matrix factorizaiton.
    # Some images without the stain, such as 0 and 1+ can't be factorized.
    # Thus, a separated zero array is initialized instead to represent no stain.
    try:
        normalizer = NumpyMacenkoNormalizer()
        normalizer.fit(img)
        _, _, dab = normalizer.normalize(img)
        gray = 255 - cv2.cvtColor(dab, cv2.COLOR_RGB2GRAY)
    except:
        gray = np.zeros(img.shape[:-1].astype(np.uint8))
    _, otsu_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = otsu_th > 0
    return gray, mask

def extract_feat(path: str, quantized_level: int = 16, scale: float = 1/4) -> list:
    """
    This function extracts a total of 19 features:
        - Avg. gray intensity of DAB, ratio of stained area (2)
        - Local binary patterns (10)
        - Haralick Features (7)

    path: path to image
    scale: resized scale (default: 0.25)
    quantized_level: for quantizing glcm (default: 16)
    """
    # 1) read image and extract the stain region
    img = read_image(path, scale)
    # gray = gray intensity image of DAB stain
    # mask = binary mask of the DAB stain from Otsu's thresholding
    gray, mask = get_dab(img)

    # 2) extract the avg. gray intensity and ratio of stained area
    color_feat = (gray * mask).mean()
    area_ratio = mask.mean()

    # 3) extract lbp (neighbors=8, raidus=1, uniform lbp)
    # see: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    # only the lbps within the mask (stained region) are aggreated in the histogram
    selected_lbp = lbp[mask] 
    # aggregate patterns to uniform and non-uniform lbps
    hists = np.histogram(selected_lbp, bins=10)[0] 
    # normalize to prob
    hists = (hists / hists.sum() + 1e-8).tolist()

    # 4) extract glcm
    # see: 
    # quantize the gray level (8bit=256) to the specified quantized_level
    # using the transformation g' = (g - min_g) / (max_g - min_g) * (q_level - 1).
    g_values = gray[mask]
    if len(g_values) == 0:
        # incase there is not segmented pixel.
        # which could happen in the case of 0 and 1+
        min_g = 0
        max_g = 0
    else:
        min_g = g_values.min()
        max_g = g_values.max()
    # As the lowest level is 0, the quantized_level is minus by 1
    norm_gray = (
        np.round(gray - min_g) / (max_g - min_g + 1e-8) * (quantized_level - 1) * mask
    )
    # In the implementation, I should the segmented region with 1 to 
    # isolate them from the background.
    # Thus, the background will occupy level = 0 while the stain will have level >= 1
    norm_gray += 1 * mask
    norm_gray = norm_gray.astype(np.uint8)
    glcm = graycomatrix(
        norm_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=quantized_level+1, # + 1 is from the shift from background 
        symmetric=True, normed=False
    ).astype(float)
    # The shape of the glcm is (levels x levels x number of distances x number of angles)
    # We don't count the co-occurrence between background and any pixel.
    # Thus, we set the values of the first row (co-occurrence between background and other pixels) = 0
    for i in range(4):
        glcm[0,:,0,i] = 0
        # normalize for each angle
        glcm[:,:,0,i] = glcm[:,:,0,i] / (glcm[:,:,0,i].sum() + 1e08)
    contrast = graycoprops(glcm, "contrast").ravel().mean()
    dissim = graycoprops(glcm, "dissimilarity").ravel().mean()
    homo = graycoprops(glcm, "homogeneity").ravel().mean()
    asm = graycoprops(glcm, "ASM").ravel().mean()
    energy = graycoprops(glcm, "energy").ravel().mean()
    corr = graycoprops(glcm, "correlation").ravel().mean()
    entropy = np.array([shannon_entropy(glcm[:,:,0,i]) for i in range(4)]).mean()

    hara_feats = [
        contrast, dissim, homo,
        asm, energy, corr, entropy
    ]

    # concat feature
    total_feat = [color_feat, area_ratio] + hists + hara_feats

    case_name = Path(path).parent.name
    return [path, case_name] + total_feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=12, type=int, help="number of workers for parallel feature extraction.")
    parser.add_argument("--scale", default=0.25, type=float, help="resizing scale")
    parser.add_argument("--level", default=16, type=int, help="quantized level")
    parser.add_argument("--src", default="./Data_Chula", type=str, help="sorce to the dataset")
    parser.add_argument("--dst", default="./extracted_features", type=str, help="dst to store features")
    args = parser.parse_args()

    level = args.level
    scale = args.scale
    paths = glob(str(Path(args.src) / "*" / "*")) # The structure of the dataset must be src/cases/images

    # parallel feature extraction
    data = mod_parallel(
        extract_feat,
        workers=args.workers,
        inputs=paths,
        quantized_level=level,
        scale=scale,
    )

    # create polars dataframe from the feataures
    cols = (
        ["path", "case"] + 
        ["color_feat", "area_ratio"] + 
        [f"lbp{i}" for i in range(10)] + 
        [
            "contrast", "dissim", "homo", "asm",
            "energy", "corr", "entropy"
        ]
    )
    df = pl.DataFrame(data, schema=cols, orient="row")

    # IMPORTANT
    # As all the cases don't follow the pattern name, we have to manually assign the label for each case.
    # If you use this code to test on your dataset, you need modify the 'prep_case' function.
    df = prep_case(df)
    
    dst_file = (
        Path(args.dst) /
        f"feat_level_{level}_scale_{scale}.csv"
    )

    if not dst_file.parent.is_dir():
        dst_file.parent.mkdir(parents=True)
    df.write_csv(dst_file)

        



