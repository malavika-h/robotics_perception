import streamlit as st
import subprocess
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Semantic 3D Lifting Demo", layout="centered")
st.title("Semantic 3D Lifting Pipeline")

st.markdown("""
This app runs the semantic lifting pipeline for a given object query and displays the resulting Oriented Bounding Box (OBB).
""")

query = st.text_input("Object query", value="power_socket")
run_button = st.button("Run Semantic Lifting")


output_json_path = Path("outputs") / f"{query}_result.json"
mesh_path = Path("outputs/reconstruction/point_cloud.ply")
images_dir = Path("Data/images")

def run_semantic_lift(query):
    cmd = [
        "python3", "run_semantic_lift.py",
        "--text-query", query,
        "--image-dir", "Data/images",
        "--poses-path", "Data/poses.json",
        "--intrinsics-path", "Data/intrinsic.json",
        "--ply-path", str(mesh_path),
        "--output-json", str(output_json_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


# --- Mesh Visualization ---
def read_ply_vertices(path, limit=100_000):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii")
            header.append(line.strip())
            if line.strip() == "end_header":
                break

        is_binary = any("binary_little_endian" in h for h in header)

        vertex_count = 0
        for h in header:
            if h.startswith("element vertex"):
                vertex_count = int(h.split()[-1])
                break

        property_names = []
        for h in header:
            if h.startswith("property") and not h.startswith("property list"):
                property_names.append(h.split()[-1])

        xyz_indices = [property_names.index(a) for a in ("x", "y", "z")]

        if not is_binary:
            # ASCII fallback
            rows = []
            for _ in range(min(vertex_count, limit)):
                parts = f.readline().decode("ascii").split()
                if len(parts) >= len(property_names):
                    rows.append([float(parts[i]) for i in xyz_indices])
            return np.array(rows)

        # -------- BINARY CASE --------
        dtype = []
        for name in property_names:
            dtype.append((name, np.float32))

        data = np.fromfile(f, dtype=dtype, count=min(vertex_count, limit))

        vertices = np.vstack([data[name] for name in ("x", "y", "z")]).T
        return vertices
    
def show_mesh(path):
    if not path.exists():
        st.info("No mesh file found at outputs/reconstruction/point_cloud.ply")
        return
    try:
        vertices = read_ply_vertices(path)
    except Exception as e:
        st.warning(f"Could not read mesh: {e}")
        return
    if vertices.size == 0:
        st.warning("Mesh file exists, but no vertices were readable.")
        return
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker={"size": 1.5, "opacity": 0.65},
            )
        ]
    )
    fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

if run_button:
    with st.spinner(f"Running semantic lifting for '{query}'..."):
        result = run_semantic_lift(query)
        if result.returncode == 0:
            st.success("Semantic lifting completed.")
        else:
            st.error(f"Error running semantic lifting: {result.stderr}")


# --- Tabs for Results, Mesh, Images ---
tab1, tab2, tab3 = st.tabs(["OBB Result", "Mesh Viewer", "Images"])

with tab1:
    if output_json_path.exists():
        st.subheader(f"OBB Result for '{query}'")
        with open(output_json_path) as f:
            obb_json = json.load(f)
        st.json(obb_json)
    else:
        st.info("No OBB result found. Run the pipeline to generate output.")

with tab2:
    st.subheader("Mesh Viewer")
    show_mesh(mesh_path)

with tab3:
    st.subheader("Input Images")
    if images_dir.exists():
        image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
        if image_files:
            cols = st.columns(4)
            for i, img_path in enumerate(image_files):
                with cols[i % 4]:
                    img = Image.open(img_path)
                    st.image(img, caption=img_path.name, use_container_width=True)
        else:
            st.info("No images found in Data/images.")
    else:
        st.info("Data/images directory does not exist.")
