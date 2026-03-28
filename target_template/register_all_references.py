import ants
import numpy as np
from medpy import metric
import pandas as pd
import os
from pathlib import Path
import re


import numpy as np
import ants

def register_ct_and_mask(ct1_array, mask1_array, ct2_array, mask2_array, mode='syN'):
    """
    功能：
    1. 分别根据 mask1 和 mask2 的中心，将图像裁剪至 [64, 256, 256] 尺寸。
    2. 在裁剪后的尺寸上进行配准 (CT1/Mask1 -> CT2)。
    3. 将配准结果还原回 CT2 的原始尺寸和空间位置。
    """
    target_shape = (64, 256, 256)
    
    # --- 辅助函数：获取裁剪的起始坐标 ---
    def get_crop_starts(mask, target_shape):
        z_dim, y_dim, x_dim = mask.shape
        tz, ty, tx = target_shape
        
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            zc, yc, xc = z_dim // 2, y_dim // 2, x_dim // 2
        else:
            zc = (coords[:, 0].min() + coords[:, 0].max()) // 2
            yc = (coords[:, 1].min() + coords[:, 1].max()) // 2

        # 计算起点，并确保加上 target_shape 后不会越界
        z_start = max(0, min(zc - tz // 2, z_dim - tz))
        y_start = yc-100
        x_start = x_dim//2-tx//2
        
        return z_start, y_start, x_start

    # ================== 1. 裁剪阶段 ==================
    tz, ty, tx = target_shape
    
    # 针对 CT1 和 Mask1 计算并裁剪
    z1, y1, x1 = get_crop_starts(mask1_array, target_shape)
    ct1_crop = ct1_array[z1:z1+tz, y1:y1+ty, x1:x1+tx]
    mask1_crop = mask1_array[z1:z1+tz, y1:y1+ty, x1:x1+tx]
    
    # 针对 CT2 计算并裁剪
    z2, y2, x2 = get_crop_starts(mask2_array, target_shape)
    ct2_crop = ct2_array[z2:z2+tz, y2:y2+ty, x2:x2+tx]

    # ================== 2. 配准阶段 ==================
    # 将 NumPy 转换为 ANTsImage
    fixed = ants.from_numpy(ct2_crop.astype('float32'))
    moving = ants.from_numpy(ct1_crop.astype('float32'))
    moving_mask = ants.from_numpy(mask1_crop.astype('float32'))

    reg_type = 'Rigid' if mode == 'rigid' else 'SyN'
    
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=reg_type,
        verbose=False
    )

    # 对 Mask 应用形变场 (最近邻插值)
    warped_mask_crop = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=registration['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    warped_ct_crop = registration['warpedmovout']

    # ================== 3. 还原阶段 ==================
    # 创建与原 CT2 同尺寸的全零背景数组
    final_warped_mask = np.zeros_like(ct2_array, dtype=mask1_array.dtype)
    
    # 建议 CT 的背景值填入原本图像的最小值(通常是 -1000 左右的空气值)，而不是 0
    bg_value = np.min(ct2_array) 
    final_warped_ct = np.full_like(ct2_array, fill_value=bg_value, dtype=ct1_array.dtype)

    # 将配准好的局部块，贴回原 CT2 被裁剪时的坐标位置
    final_warped_mask[z2:z2+tz, y2:y2+ty, x2:x2+tx] = warped_mask_crop.numpy()
    final_warped_ct[z2:z2+tz, y2:y2+ty, x2:x2+tx] = warped_ct_crop.numpy()

    return final_warped_ct, final_warped_mask

def generate_all_ref(data_dir, ref_dir, save_dir):
    """
    批量执行配准任务，基于 CSV 映射表寻找参考图像进行配准。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 3. 读取 ref_dir 的 csv 文件

    df = pd.read_csv(ref_dir) # 默认读取找到的第一个 CSV
    
    # 获取 center_sample_label 列的所有不同值
    all_references = df['center_sample_label'].dropna().unique().tolist()
    print(f"成功加载 CSV，共找到 {len(all_references)} 个不同的中心参考样本。")

    # 4. 遍历 data_dir 内每一个文件夹
    for mrn_folder in os.listdir(data_dir):
        mrn_path = os.path.join(data_dir, mrn_folder)
        if not os.path.isdir(mrn_path):
            continue
        
        MRN = mrn_folder
        
        # 查找所有含有 "ctv" 的文件 (忽略大小写)
        ctv_files = [f for f in os.listdir(mrn_path) if 'ctv' in f.lower() and f.endswith('.npz')]
        if not ctv_files:
            continue
            
        # 读取同文件夹下 ct_image.npz
        ct_path = os.path.join(mrn_path, 'ct_image.npz')
        if not os.path.exists(ct_path):
            print(f"跳过: 未找到 {ct_path}")
            continue
        test_ct = np.load(ct_path)['volume']

        # 5. 对于每个 CTV 文件
        for ctv_name in ctv_files:
            ctv_path = os.path.join(mrn_path, ctv_name)
            test_ctv = np.load(ctv_path)['volume']
            test_desc=np.load(ctv_path)['description'].item()
            if 'left' in test_desc:
                test_laterality='left'
            elif 'right' in test_desc:
                test_laterality='right'
            else:
                test_laterality='bilateral'
            
            # 生成查找字符串
            search_str = f"{MRN}{ctv_name}"
            
            # 在 csv 文件中 sample_label 列查找
            match_row = df[df['sample_label'] == search_str]
            if not match_row.empty:
                # 查找到，获取 center_sample_label 的值，存为 reference 列表
                reference = [match_row.iloc[0]['center_sample_label']]
            else:
                # 查找不到，使用全局 all_references
                reference = all_references
            
            # 6. 对于 reference 列表中每个值，分出 MRN 和 ctv_name
            for ref in reference:
                # 使用正则拆分：前面匹配任意字符(MRN)，后面匹配 ctv 开头的字符串(ctv_name)
                # 示例匹配：70763599ctvp0.npz -> group(1)="70763599", group(2)="ctvp0.npz"
                match = re.match(r'^(.*?)(ctv.*\.npz)$', ref, re.IGNORECASE)
                if not match:
                    print(f"警告: 无法解析参考样本名称格式 {ref}")
                    continue
                
                ref_MRN = match.group(1)
                ref_ctv_name = match.group(2)
                
                # 7. 读取 data_dir 内 MRN 目录的参考数据
                ref_ctv_path = os.path.join(data_dir, ref_MRN, ref_ctv_name)
                ref_ct_path = os.path.join(data_dir, ref_MRN, 'ct_image.npz')
                
                if not os.path.exists(ref_ctv_path) or not os.path.exists(ref_ct_path):
                    print(f"跳过配准: 参考文件缺失 -> {ref_MRN}/{ref_ctv_name}")
                    continue
                
                ref_ct = np.load(ref_ct_path)['volume']
                ref_ctv = np.load(ref_ctv_path)['volume']
                ref_desc=np.load(ref_ctv_path)['description'].item()
                if 'left' in ref_desc:
                    ref_laterality='left'
                elif 'right' in ref_desc:
                    ref_laterality='right'
                else:
                    ref_laterality='bilateral'
                if test_laterality!='bilateral':
                    if test_laterality!=ref_laterality:
                        ref_ct=np.flip(ref_ct,axis=2)
                        ref_ctv=np.flip(ref_ctv,axis=2)
                # 8. 调用配准函数
                print(f"正在配准: {search_str} -> 到参考 -> {ref} ...")
                warped_ct, warped_mask = register_ct_and_mask(ref_ct,ref_ctv,test_ct, test_ctv, mode='syN')
                # 9. 保存结果
                # 清理后缀以保持文件名整洁 (去除中间的 .npz)
                clean_curr_ctv = ctv_name.replace('.npz', '')
                clean_ref_ctv = ref_ctv_name.replace('.npz', '')
                
                save_filename = f"{MRN}{clean_curr_ctv}_from_{ref_MRN}{clean_ref_ctv}.npz"
                save_filepath = os.path.join(save_dir, save_filename)
                
                # 保存为字典形式的 npz，以防后续还需要 'volume' 键名读取
                np.savez_compressed(save_filepath, volume=warped_mask)
                print(f"保存成功: {save_filepath}")

generate_all_ref(r'/data/maia/hzhao/data/H&NCTV/adjuvant/numpy_CTVp_OARs',
                   r'/data/maia/hzhao/H&NCTV/target_template/data_stat/reference_tagrets.csv',
                   r'/data/maia/hzhao/H&NCTV/target_template/all_refereces')
pass
