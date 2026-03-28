from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from scipy.spatial.distance import cdist
import joblib


n_neighbors = 30

embeddings = {
    "Random projection embedding": SparseRandomProjection(
        n_components=2, random_state=42
    ),
    "Truncated SVD embedding": TruncatedSVD(n_components=2),
    "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=2),
    "Standard LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="standard"
    ),
    "Modified LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="modified"
    ),
    "LTSA LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="ltsa"
    ),
    "MDS embedding": MDS(
        n_components=2, n_init=1, max_iter=120, n_jobs=2, normalized_stress="auto"
    ),
    "Random Trees embedding": make_pipeline(
        RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
        TruncatedSVD(n_components=2),
    ),
    "Spectral embedding": SpectralEmbedding(
        n_components=2, random_state=0, eigen_solver="arpack"
    ),
    "t-SNE embedding": TSNE(
        n_components=2
    ),
    "umap":umap.UMAP(random_state=42, n_jobs=1)
}

def get_value(df,fn,key="Physician"):
    PID=fn.split('ctv')[0]
    ctv=fn.split(PID)[1]
    row = df.loc[[PID]]
    row = row[row['File'] == ctv]
    row=row.iloc[0]
    value = row[key]
    return value

def crop_mask(mask_array):
    """
    实现功能：
    1. 寻找 Mask 在 XYZ 三个方向的中心。
    2. 裁剪尺寸：Z轴上下共40（即中心±20），Y轴上下共80（即中心±40），X轴上下共80（即中心±40）。
    """
    z_dim, y_dim, x_dim = mask_array.shape
    coords = np.argwhere(mask_array > 0)
    
    if coords.size == 0:
        # 如果 mask 为空，默认取图像物理中心
        z_center, y_center, x_center = z_dim // 2, y_dim // 2, x_dim // 2
    else:
        # 1. 找出 XYZ 三个方向的中心
        z_center = (coords[:, 0].min() + coords[:, 0].max()) // 2
        y_center = (coords[:, 1].min() + coords[:, 1].max()) // 2
        x_center = (coords[:, 2].min() + coords[:, 2].max()) // 2

    # 2. 计算裁剪边界 (使用 np.clip 防止索引越界)
    # Z方向：中心上下共40像素 (radius = 20)
    z_start = max(0, z_center - 20)
    z_end = min(z_dim, z_center + 20)
    
    # Y方向：中心上下共80像素 (radius = 40)
    y_start = max(0, y_center - 40)
    y_end = min(y_dim, y_center + 40)
    
    # X方向：中心上下共80像素 (radius = 40)
    x_start = max(0, x_center - 40)
    x_end = min(x_dim, x_center + 40)
    
    # 输出裁剪后的坐标范围（可选，方便调试）
    print(f"Crop Range -> Z: [{z_start}:{z_end}], Y: [{y_start}:{y_end}], X: [{x_start}:{x_end}]")
    
    return z_start, z_end, y_start, y_end, x_start, x_end

def get_data_by_label(data_path, key="Physician"):
    df = pd.read_csv(r'/data/maia/hzhao/H&NCTV/target_template/data_stat/ctv_analysis_summary.csv')
    df['MRN'] = df['MRN'].astype(str)
    df.set_index('MRN', inplace=True)
    X=[]
    y=[]
    label=[]
    for im in data_path.glob('*.npz'):
        X.append(im)
        y.append(get_value(df, im.name, key))
        label.append(im.name)
    indexes=crop_mask(np.load(str(X[0]))['arr_0'])
    X_input=[]
    for x in X:
        x=np.load(str(x))['arr_0']
        x=x[indexes[0]:indexes[1],indexes[2]:indexes[3],indexes[4]:indexes[5]]
        x=x.flatten()
        X_input.append(x)
    X_input=np.stack(X_input,axis=0).astype(np.float32)
    y=np.array(y)#.astype(np.float32)
    return X_input,y, label

def mse_var(X):
    X_mean=np.mean(X,axis=0)
    var=0
    for x in X:
        mse=((x - X_mean)**2).mean()
        var+=mse
    var/=X.shape[0]
    return var

def select_by_type(X,y,type):
    X_select=[]
    y_select=[]
    for i,x in enumerate(X):
        if y[i]==type:
            X_select.append(x)
            y_select.append(y[i])
    X_select=np.stack(X_select,axis=0).astype(np.float32)
    y_select=np.stack(y_select,axis=0).astype(np.float32)
    return X_select

def weighted_var(X,y):
    cents=np.unique(y)
    var_w=0
    for c in cents:
        X_c=select_by_type(X,y,c)
        print(f'number of subtype {c}: {len(X_c)}')
        var_X_c=mse_var(X_c)
        var_w+=var_X_c*len(X_c)
    return var_w

def plot_embedding(X,y, title, ax):
    X = MinMaxScaler().fit_transform(X)

    for digit, key in enumerate(np.unique(y)):
        ax.scatter(
            *X[y == key].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
            label=f"{key}"
        )

    ax.set_title(title)
    ax.set_xlabel('c1')
    ax.set_ylabel('c2')
    ax.legend(title="Value: Key", bbox_to_anchor=(1.05, 1), loc='upper left')
    return

def plot_embedding_continuous(X,y, title, ax):
    X = MinMaxScaler().fit_transform(X)

    sc = ax.scatter(
        X[:, 0], X[:, 1], # 所有的 X 和 Y 坐标
        c=y,             # 核心：根据 y 的值自动映射颜色
        cmap='viridis',  # 颜色谱
        s=30,            # 散点大小，用普通圆点可以稍微调小一点
        alpha=0.6,       # 透明度
        zorder=2
    )
    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('c1')
    ax.set_ylabel('c2')
    return

def save_references(projection, y_kmeans, labels, output_path):
    """
    功能：
    1. 计算每个簇的几何中心。
    2. 找到每个簇中距离几何中心最近的真实样本（中心点样本）。
    3. 记录：当前样本label、所属簇中心样本label、y_kmeans值。
    4. 保存为 CSV。
    """
    # 获取唯一的聚类标签 (例如 0, 1, 2...)
    unique_clusters = np.unique(y_kmeans)
    
    # 存储每个簇对应的“中心点样本”的 label
    # 格式：{cluster_id: center_sample_label}
    cluster_to_center_label = {}

    for cluster_id in unique_clusters:
        # 1. 提取当前簇的所有样本坐标
        mask = (y_kmeans == cluster_id)
        cluster_points = projection[mask]
        cluster_labels = np.array(labels)[mask]
        
        # 2. 计算当前簇的几何中心 (Centroid)
        centroid = cluster_points.mean(axis=0).reshape(1, -1)
        
        # 3. 计算该簇内所有点到几何中心的欧氏距离
        distances = cdist(cluster_points, centroid, metric='euclidean').flatten()
        
        # 4. 找到距离最近的样本索引
        closest_idx_in_cluster = np.argmin(distances)
        center_label = cluster_labels[closest_idx_in_cluster]
        
        # 5. 存入映射表
        cluster_to_center_label[cluster_id] = center_label

    # --- 构造输出数据 ---
    results = []
    for i in range(len(labels)):
        current_label = labels[i]
        current_cluster = y_kmeans[i]
        # 根据当前样本所属的 y_kmeans 找到对应的中心点 label
        center_label = cluster_to_center_label[current_cluster]
        
        results.append({
            "sample_label": current_label,
            "center_sample_label": center_label,
            "y_kmeans": current_cluster
        })

    # 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"成功保存参考表至: {output_path}")
    return df

#X, y, labels=get_data_by_label(data_path=Path(r'/data/maia/hzhao/H&NCTV/target_template/initial_registered_targets_93592484_ctv0'))
X, y, labels=get_data_by_label(data_path=Path(r'/data/maia/hzhao/H&NCTV/target_template/initial_registered_targets_93592484_ctv0'),key='LateralDeviation_px')

#X,y=remove_by_type(X,y,2)

if (0):
    projections = {}
    for name, transformer in embeddings.items():
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X

        print(f"Computing {name}...")
        projections[name] = transformer.fit_transform(data)
    clustered=KMeans(n_clusters=3, n_init="auto",random_state=0).fit(X)
    y_clustered=clustered.labels_

    fig, ax = plt.subplots(ncols=1,nrows=len(projections),figsize=(5,4*len(projections)))

    for i,name in enumerate(projections):
        plot_embedding(projections[name],y, name,ax[i])


    plt.tight_layout()
    plt.savefig(r'/data/maia/hzhao/H&NCTV/target_template/clustered.png')

else:    
    #loaded_transformer = joblib.load(r'/data/maia/hzhao/H&NCTV/target_template/HN_CTVP_umap_model.joblib')
    transformer = umap.UMAP(random_state=42, n_jobs=1)
    print(f"Computing projection ...")
    projection = transformer.fit_transform(X)
    joblib.dump(transformer, r'/data/maia/hzhao/H&NCTV/target_template/HN_CTVP_umap_model.joblib')
    np.savez_compressed(r'/data/maia/hzhao/H&NCTV/target_template/umap_projection', projection=projection, labels=labels)

    #pred=transformer.transform(X)

    print(f'weighted_var:{weighted_var(X,[0]*len(X))}')

    n_clusters=10
    clustered=KMeans(n_clusters, n_init="auto",random_state=0).fit(projection)
    y_clustered=clustered.labels_
    X_size=np.sum(X, axis=1)
    
    print(f'weighted_var:{weighted_var(X,y_clustered)}')
    #exit(0)

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))
    title = r"UMAP embedding"
    #X_avg,y_avg, p_list=avg_patient(imgs,projections[name],y)
    X_avg=projection

    save_references(X_avg, y_clustered, labels, output_path=r'/data/maia/hzhao/H&NCTV/target_template/data_stat/reference_tagrets.csv')

    #plot_embedding(X_avg, y_clustered, title,ax)
    #plot_embedding(X_avg, y, title,ax)
    #plot_embedding_continuous(X_avg, X_size, title,ax)
    y=np.abs(y)
    plot_embedding_continuous(X_avg, y, title,ax)
    plt.tight_layout()
    #plt.savefig(f'/data/maia/hzhao/H&NCTV/target_template/umap_physician_{n_clusters}clusters.png')
    plt.savefig(f'/data/maia/hzhao/H&NCTV/target_template/abs_laterality.png')

