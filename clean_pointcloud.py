import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def extract_and_render_ply(ply_path, save_prefix="filtered"):
    # -----------------------------
    # 1. Load point cloud
    # -----------------------------
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points)

    print("Total points:", len(xyz))

    # -----------------------------
    # 2. Density pruning
    # -----------------------------
    nbrs = NearestNeighbors(n_neighbors=10).fit(xyz)
    distances, _ = nbrs.kneighbors(xyz)

    density = np.mean(distances, axis=1)

    thresh = np.percentile(density, 50)
    mask = density < thresh

    xyz_clean = xyz[mask]

    print("After density pruning:", len(xyz_clean))

    # -----------------------------
    # 3. Save cleaned PLY
    # -----------------------------
    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(xyz_clean)

    o3d.io.write_point_cloud(f"{save_prefix}.ply", pcd_clean)

    # -----------------------------
    # 4. Normalize for rendering
    # -----------------------------
    center = np.mean(xyz_clean, axis=0)
    xyz_norm = xyz_clean - center

    scale = np.max(np.linalg.norm(xyz_norm, axis=1))
    xyz_norm /= (scale + 1e-8)

    # -----------------------------
    # 5. Render function
    # -----------------------------
    def render_view(points, elev=30, azim=45, name="view"):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=1, alpha=0.8)

        ax.view_init(elev=elev, azim=azim)

        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{name}.png", dpi=200)
        plt.close()

    # -----------------------------
    # 6. Save multiple views
    # -----------------------------
    render_view(xyz_norm, elev=20, azim=45, name="front")
    render_view(xyz_norm, elev=90, azim=0, name="top")
    render_view(xyz_norm, elev=0, azim=0, name="side")

    print("Saved images:")
    print(f" - {save_prefix}_front.png")
    print(f" - {save_prefix}_top.png")
    print(f" - {save_prefix}_side.png")

    return xyz_clean

clean_points = extract_and_render_ply(
    "third_party/semantic-gaussians/output/scannet_format/point_cloud/iteration_30000/point_cloud.ply",
    save_prefix="gaussian_filtered"
)