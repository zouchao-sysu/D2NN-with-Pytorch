import os
import glob
import importlib.util
import time
import sys

import numpy as np


def load_lumapi(lumapi_path: str):
    spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 lumapi，请检查路径")
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)
    return lumapi


def get_project_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_existing_file(project_dir: str, filename: str) -> str:
    candidates = [
        os.path.join(project_dir, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    patterns = [
        os.path.join(project_dir, "**", filename),
        os.path.join(os.getcwd(), "**", filename),
    ]
    for pat in patterns:
        hits = [h for h in glob.glob(pat, recursive=True) if os.path.isfile(h)]
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"找不到文件 `{filename}`。\n"
        f"cwd={os.getcwd()}\n"
        f"project_dir={project_dir}\n"
        "请确认该文件在项目内，或把路径改成其真实绝对路径。"
    )


def _row_merge_segments(mask_row: np.ndarray):
    """
    生成 mask_row 中连续 True 段的 (i0, i1)（i1 为 exclusive）。
    """
    n = int(mask_row.size)
    i = 0
    while i < n:
        if not bool(mask_row[i]):
            i += 1
            continue
        i0 = i
        i += 1
        while i < n and bool(mask_row[i]):
            i += 1
        i1 = i
        yield i0, i1


def build_etched_sio2_layer_rowmerge(
    fdtd,
    *,
    groupname: str,
    etch_depth_map_m: np.ndarray,   # [size,size]，刻蚀深度（m）
    size: int,
    pixel_size: float,
    layer_z_top: float,             # 该层 SiO2 顶面 z 坐标
    t_oxide: float,                 # SiO2 薄膜厚度（m）
    use_3d: bool,
    n_sio2: float,
    etch_fill_index: float,         # 槽内填充（空气=1.0）
    h_min: float = 0.0,             # 小于该值不刻蚀（m）
    log_every: int = 200,
):
    """
    工艺一致建模：
    \- 先建整片 SiO2 薄膜（厚度 t_oxide）
    \- 再对刻蚀区域建立“槽”（填充为空气/背景介质），槽深=etch_depth
    \- 槽使用按行合并连续段，减少 addrect 次数
    """
    x_span = float(pixel_size)
    y_span = float(pixel_size)

    hm2d = np.asarray(etch_depth_map_m, dtype=float)
    if hm2d.shape != (size, size):
        raise ValueError(f"`etch_depth_map_m` 期望 shape=({size},{size})，实际={hm2d.shape}")

    hm2d = np.clip(hm2d, 0.0, float(t_oxide))
    mask = hm2d > float(h_min)

    fdtd.addstructuregroup()
    fdtd.set("name", groupname)
    fdtd.groupscope(groupname)

    # \- (1) blanket SiO2 film
    film_x_center = (size * x_span) / 2.0
    film_y_center = -(size * y_span) / 2.0

    fdtd.addrect()
    fdtd.set("name", "sio2_film")
    fdtd.set("x", film_x_center)
    fdtd.set("x span", size * x_span)
    if use_3d:
        fdtd.set("y", film_y_center)
        fdtd.set("y span", size * y_span)
    fdtd.set("z max", float(layer_z_top))
    fdtd.set("z min", float(layer_z_top - t_oxide))
    fdtd.set("material", "<Object defined dielectric>")
    fdtd.set("index", float(n_sio2))

    # \- (2) etch trenches (filled by air/background)
    t0 = time.perf_counter()
    created_slots = 0

    for j in range(size):
        row_mask = mask[j]
        if not np.any(row_mask):
            continue

        for i0, i1 in _row_merge_segments(row_mask):
            h = float(np.max(hm2d[j, i0:i1]))
            if h <= h_min:
                continue

            seg_len = int(i1 - i0)
            x_center = x_span / 2.0 + (i0 + (seg_len - 1) / 2.0) * x_span
            x_seg_span = seg_len * x_span

            fdtd.addrect()
            fdtd.set("name", f"etch_{j}_{i0}_{i1}")
            fdtd.set("x", float(x_center))
            fdtd.set("x span", float(x_seg_span))
            if use_3d:
                fdtd.set("y", float(-y_span / 2.0 - j * y_span))
                fdtd.set("y span", float(y_span))
            fdtd.set("z max", float(layer_z_top))
            fdtd.set("z min", float(layer_z_top - h))
            fdtd.set("material", "<Object defined dielectric>")
            fdtd.set("index", float(etch_fill_index))

            created_slots += 1
            if log_every > 0 and created_slots % int(log_every) == 0:
                dt = time.perf_counter() - t0
                print(f"[build] {groupname}: slots={created_slots}, elapsed={dt:.1f}s")
                sys.stdout.flush()

    fdtd.groupscope("::model")
    dt = time.perf_counter() - t0
    print(f"[build] {groupname} done: slots={created_slots}, elapsed={dt:.1f}s")
    sys.stdout.flush()

    return created_slots


def main():
    # ---------- 路径 ----------
    project_dir = get_project_dir()
    lumapi_path = r"D:\Program Files\Lumerical\v241\api\python\lumapi.py"

    height_map_path = resolve_existing_file(project_dir, "height_map.npy")
    out_fsp_path = os.path.join(project_dir, "D2NN_visible_532nm_etched_sio2_rowmerge.fsp")

    # ---------- 基本参数 ----------
    lumapi = load_lumapi(lumapi_path)

    wl0 = 532e-9
    n_sio2 = 1.46
    background_index = 1.0  # 同时也用于刻蚀槽的填充折射率

    # 网络尺寸与像素
    size = 128
    pixel_size = 2e-6
    x_span = pixel_size
    y_span = pixel_size

    # 层数与传播距离
    num_layer = 2
    z_between_layers = 4e-3
    z_last_to_detector = 5e-3

    # ---------- 工艺参数（对应截图） ----------
    # SiO2 总膜厚（最大刻蚀深度）
    t_oxide = 519e-9
    # 刻蚀后填充（空气/背景）
    etch_fill_index = background_index
    # 小于该刻蚀深度不建槽（可设为 0 或 10nm 以滤掉噪声）
    h_min = 0.0

    # ---------- 载入刻蚀深度图（m） ----------
    height_map = np.load(height_map_path)
    height_map = np.asarray(height_map, dtype=float)

    if height_map.ndim != 3:
        raise ValueError(f"`height_map.npy` 需要是 3D 数组 [num_layer,size,size]，实际 ndim={height_map.ndim}")
    if height_map.shape[0] != num_layer or height_map.shape[1] != size or height_map.shape[2] != size:
        raise ValueError(
            f"`height_map` 期望 shape=[{num_layer},{size},{size}]，实际={height_map.shape}。\n"
            "请让 num_layer/size 与 height_map.npy 匹配，或改这里的参数。"
        )

    # 认为 height_map 存的是“刻蚀深度”，直接 clip 到 [0,t_oxide]
    height_map = np.clip(height_map, 0.0, float(t_oxide))
    max_etch = float(np.max(height_map))

    # ---------- 建模维度 ----------
    use_3d = True
    hide = True

    fdtd = lumapi.FDTD(hide=hide)
    fdtd.groupscope("::model")

    # 全局源/监视器设置：单频减少存储
    fdtd.setglobalsource("wavelength start", wl0)
    fdtd.setglobalsource("wavelength stop", wl0)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("use linear wavelength spacing", 1)

    # ---------- FDTD 仿真区域 ----------
    fdtd.addfdtd()
    fdtd.set("dimension", "3D" if use_3d else "2D")

    x0 = size * x_span / 2.0
    y0 = -size * y_span / 2.0

    pad_xy = 6 * pixel_size
    sim_x_span = size * x_span + 2 * pad_xy
    sim_y_span = size * y_span + 2 * pad_xy

    z_front = 2e-6
    z_back = 2e-6
    # 结构位于 z<=0（顶面 layer_z_top=0），向下刻蚀到 -t_oxide
    z_min = -z_front - max(t_oxide, max_etch)
    z_max = (num_layer - 1) * z_between_layers + z_last_to_detector + z_back

    fdtd.set("x", x0)
    fdtd.set("x span", sim_x_span)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", sim_y_span)
    fdtd.set("z min", z_min)
    fdtd.set("z max", z_max)

    # PML 边界
    fdtd.set("x min bc", "PML")
    fdtd.set("x max bc", "PML")
    if use_3d:
        fdtd.set("y min bc", "PML")
        fdtd.set("y max bc", "PML")
    fdtd.set("z min bc", "PML")
    fdtd.set("z max bc", "PML")

    fdtd.set("background index", background_index)

    # 全局网格：传播区用粗网格，结构区用 override 加细
    fdtd.set("mesh accuracy", 2)

    # ---------- 源：平面波 ----------
    fdtd.addplane()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "forward")
    fdtd.set("x", x0)
    fdtd.set("x span", size * x_span)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", size * y_span)
    fdtd.set("z", z_min + 1e-6)
    fdtd.set("wavelength start", wl0)
    fdtd.set("wavelength stop", wl0)

    # ---------- 探测器：功率监视器 ----------
    det_z = (num_layer - 1) * z_between_layers + z_last_to_detector
    fdtd.addpower()
    fdtd.set("name", "detector_power")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", x0)
    fdtd.set("x span", size * x_span)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", size * y_span)
    fdtd.set("z", det_z)

    # ---------- 可选调试 monitor（范围不要覆盖整段毫米传播） ----------
    fdtd.addprofile()
    fdtd.set("name", "xz_profile_near_layers")
    fdtd.set("monitor type", "2D Y-normal")
    fdtd.set("x", x0)
    fdtd.set("x span", size * x_span)
    if use_3d:
        fdtd.set("y", y0)
    fdtd.set("z min", -t_oxide - 2e-6)
    fdtd.set("z max", (num_layer - 1) * z_between_layers + 2e-6)

    # ---------- 局部 mesh override：只覆盖结构附近 ----------
    fdtd.addmesh()
    fdtd.set("name", "mesh_near_structures")
    fdtd.set("x", x0)
    fdtd.set("x span", size * x_span)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", size * y_span)

    z_struct_min = -t_oxide - 1e-6
    z_struct_max = (num_layer - 1) * z_between_layers + 1e-6
    fdtd.set("z min", z_struct_min)
    fdtd.set("z max", z_struct_max)

    fdtd.set("override x mesh", 1)
    if use_3d:
        fdtd.set("override y mesh", 1)
    fdtd.set("override z mesh", 1)
    fdtd.set("set maximum mesh step", 1)

    dx = 200e-9
    dy = 200e-9
    dz = 200e-9
    fdtd.set("dx", dx)
    if use_3d:
        fdtd.set("dy", dy)
    fdtd.set("dz", dz)

    # ---------- 构建 D2NN 层（SiO2 薄膜 + 刻蚀槽，行合并） ----------
    t_build0 = time.perf_counter()
    total_slots = 0

    print(f"[build] start etched-SiO2: num_layer={num_layer}, size={size}")
    sys.stdout.flush()

    for l in range(num_layer):
        layer_z_top = l * z_between_layers
        etch2d = height_map[l]

        slots = build_etched_sio2_layer_rowmerge(
            fdtd,
            groupname=f"diffraction_layer_{l}",
            etch_depth_map_m=etch2d,
            size=size,
            pixel_size=pixel_size,
            layer_z_top=layer_z_top,
            t_oxide=t_oxide,
            use_3d=use_3d,
            n_sio2=n_sio2,
            etch_fill_index=etch_fill_index,
            h_min=h_min,
            log_every=200,
        )
        total_slots += int(slots)

    dt_build = time.perf_counter() - t_build0
    print(f"[build] all done: total_slots={total_slots}, elapsed={dt_build:.1f}s")
    sys.stdout.flush()

    # ---------- 保存并关闭 ----------
    fdtd.save(out_fsp_path)
    fdtd.close()

    print("Loaded height_map from:", height_map_path, "shape=", height_map.shape)
    print("Saved:", out_fsp_path)


if __name__ == "__main__":
    main()
