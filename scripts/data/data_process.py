from glob import glob
import pandas as pd
import os

def dir_to_df(SOURCE_DIR):
    img_files = glob(SOURCE_DIR+"/*/*/images/*")
    mask_files = glob(SOURCE_DIR+"/*/*/lung masks/*")
    outcomes = [x.split('/')[8] for x in img_files]
    split = [x.split('/')[7] for x in img_files]

    df = pd.DataFrame()
    df['mask'] = mask_files
    df['img_files'] = img_files
    df['img_class'] = outcomes
    df['split'] = split

    df.to_csv(f'{SOURCE_DIR}/covid_qu_ex.csv')
    return df

def process_path(path, test, root_dict={'nccid':'/MULTIX/DATA/nccid_dcm_seg', 
                                        'ltht_binary14_21':'/MULTIX/DATA/INPUT/ltht_dcm_seg', 
                                        'covidgr':'/MULTIX/DATA/HOME/covid-19-benchmarking/data/covidgr_seg'}):
    root = root_dict[test]

    if test == 'nccid':
        patientid = path.split('/')[5:]
    else:
        patientid = path.split('/')[4:]
    
    patientid = '/'.join(patientid)
    new_path = os.path.join(root, patientid)       
    
    return new_path