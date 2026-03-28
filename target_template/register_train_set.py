import ants
import numpy as np
from medpy import metric
import pandas as pd
import os
from pathlib import Path


def crop_mask_CT(mask_array, ct_array):
    """
    仅裁剪 Mask: 
    Z轴: CTV中心上下32; Y轴: CTV中心上100,下156; X轴: [128:384]
    """
    z_dim, y_dim, x_dim = mask_array.shape
    coords = np.argwhere(mask_array > 0)
    
    if coords.size == 0:
        z_center, y_center = z_dim // 2, y_dim // 2
    else:
        z_center = (coords[:, 0].min() + coords[:, 0].max()) // 2
        y_center = (coords[:, 1].min() + coords[:, 1].max()) // 2

    z_start = np.clip(z_center - 32, 0, z_dim - 64)
    y_start = np.clip(y_center - 100, 0, y_dim - 256)
    x_start = np.clip(128, 0, x_dim - 256) # 确保不越界

    mask_array = mask_array[int(z_start):int(z_start+64), 
                      int(y_start):int(y_start+256), 
                      int(x_start):int(x_start+256)]
    ct_array = ct_array[int(z_start):int(z_start+64), 
                      int(y_start):int(y_start+256), 
                      int(x_start):int(x_start+256)]
    return mask_array,ct_array

def register_ct_and_mask(ct1_array, mask1_array, ct2_array, mode='rigid'):
    """
    将 CT1 及其 Mask 配准到 CT2 的空间
    mode: 'rigid' (刚性) 或 'syN' (形变/非线性)
    """
    
    # 1. 将 Numpy Array 转换为 ANTsImage
    # 注意：在实际工程中，建议指定 spacing (如 1.0, 1.0, 1.0)，否则默认为 1
    fixed = ants.from_numpy(ct2_array.astype('float32'))
    moving = ants.from_numpy(ct1_array.astype('float32'))
    moving_mask = ants.from_numpy(mask1_array.astype('float32'))

    # 2. 执行配准
    # 'Rigid': 仅包含平移和旋转
    # 'SyN': 典型的非线性/形变配准
    reg_type = 'Rigid' if mode == 'rigid' else 'SyN'
    
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=reg_type,
        verbose=False
    )

    # 3. 应用变换到 Mask
    # 注意：Mask 必须使用 'nearestNeighbor' 插值，以保持二值特性（0 或 1）
    # 如果使用线性插值，边缘会产生 0.5 之类的小数
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=registration['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    # 4. 获取配准后的 CT 用于检查对齐效果
    warped_ct = registration['warpedmovout']

    return warped_ct.numpy(), warped_mask.numpy()

def test_train_set(test_id, test_target, 
                   root_dir=r'/data/maia/hzhao/data/H&NCTV/adjuvant/numpy_CTVp_OARs',
                   save_dir=r'/data/maia/hzhao/H&NCTV/target_template/registered_targets'):
    all_ids = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    val_ids = ["72354212", "91899161", "92577870", "93078367", "94007319", "94313318", "94807664"]
    
    test_ids = ["70041725", "71545860", "72849928", "73809568", 
                "90033680", "90045746", "90589733", "91484475", "91698750", 
                "91919502","92003371", "92063686", "92520472", 
                "93592484", "93616032", "93791421", "94248525","94515053"]
    
    exclude = set(val_ids + test_ids)
    train_ids = [id for id in all_ids if id not in exclude]
    csv_path: str=r'/data/maia/hzhao/H&NCTV/output/data_stat/ctv_analysis_summary.csv'
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    df = pd.read_csv(csv_path)
    df['MRN'] = df['MRN'].astype(str)
    df.set_index('MRN', inplace=True)
    
    test_im=np.load(root_dir/test_id/'ct_image.npz')['volume']
    test_ctv=np.load(root_dir/test_id/test_target)['volume']
    test_ctv,test_im=crop_mask_CT(test_ctv,test_im)
    test_desc=np.load(root_dir/test_id/test_target)['description'].item()
    if 'left' in test_desc:
        test_laterality='left'
    elif 'right' in test_desc:
        test_laterality='right'
    else:
        test_laterality='bilateral'
    
    dice_list=[]
    for patient_id in train_ids:
        patient_dir=root_dir/patient_id
        ctv_files = sorted(patient_dir.glob('ctvp*.npz'))
        for ctv_file in ctv_files:
            row = df.loc[[patient_id]]
            row = row[row['File'] == os.path.basename(ctv_file)]
            row=row.iloc[0]
            train_im=np.load(patient_dir/'ct_image.npz')['volume']
            train_ctv=np.load(ctv_file)['volume']
            train_ctv,train_im=crop_mask_CT(train_ctv,train_im)
            train_desc=np.load(ctv_file)['description'].item()
            if 'left' in train_desc:
                train_laterality='left'
            elif 'right' in train_desc:
                train_laterality='right'
            else:
                train_laterality='bilateral'
            if test_laterality=='bilateral':
                if train_laterality!='bilateral':
                    continue
            elif test_laterality!=train_laterality:
                train_im=np.flip(train_im,axis=2)
                train_ctv=np.flip(train_ctv,axis=2)
            train_ct_deformed, train_mask_deformed = register_ct_and_mask(train_im, train_ctv, test_im, mode='syN')
            dice_deformed=metric.dc(train_mask_deformed,test_ctv)
            dice_list.append({"train_id":patient_id,"target":ctv_file.name,"mode":"syN","Dice":dice_deformed})
            print(dice_list[-1])
            np.savez_compressed(save_dir/(patient_id+ctv_file.name),train_mask_deformed)
    results_sorted = sorted(dice_list, key=lambda x: x['Dice'], reverse=True)
    # 3. 打印前 5 个最具有代表性（即最接近模板）的样本
    print(results_sorted[:5])
    return

test_train_set("93592484","ctvp0.npz")
pass
