def read_points3d_txt(file_path):
    points = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments
            if line.startswith('#') or len(line) == 0:
                continue
            
            parts = line.split()
            
            # Parse values
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            r = int(parts[4])
            g = int(parts[5])
            b = int(parts[6])
            
            points.append((x, y, z, r, g, b))
    
    return points


def write_ply(points, output_path):
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")


if __name__ == "__main__":
    input_file = "tri_txt/points3D.txt"
    output_file = "tri_txt/points3D.ply"
    
    pts = read_points3d_txt(input_file)
    write_ply(pts, output_file)
    
    print(f"Converted {len(pts)} points to {output_file}")