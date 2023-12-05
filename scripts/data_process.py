
from glob import glob
import pandas as pd

SOURCE_DIR = "/MULTIX/DATA/HOME/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data" # /split/class/ [images/lung masks]

img_files = glob(SOURCE_DIR+"/*/*/images/*")
print(img_files[-20:])
mask_files = glob(SOURCE_DIR+"/*/*/lung masks/*")
outcomes = [x.split('/')[8] for x in img_files]
print(outcomes[-10:])
split = [x.split('/')[7] for x in img_files]
print(split[-10:])
df = pd.DataFrame()
df['mask'] = mask_files
df['img_files'] = img_files
df['img_class'] = outcomes
df['split'] = split

df.to_csv('/MULTIX/DATA/HOME/COVID_QU_Ex/covid_qu_ex.csv')