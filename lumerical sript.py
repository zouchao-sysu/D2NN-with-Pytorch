import os
import glob
import importlib.util
import time
import sys

import numpy as np


# ============================================================
# 工艺常量（与 Notebook 和 Fig. S1 一致）
# ============================================================
WL0 = 632.8e-9
N_SIO2 = 1.46
N_AIR = 1.0
T_OXIDE = 519e-9


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
    """生成 mask_row 中连续 True 段的 (i0, i1)（i1 为 exclusive）。"""
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
    etch_depth_map_m: np.ndarray,
    size: int,
    pixel_size: float,
    layer_z_top: float,
    t_oxide: float,
    use_3d: bool,
    n_sio2: float,
    etch_fill_index: float,
    h_min: float = 0.0,
    log_every: int = 200,
):
    x_span = float(pixel_size)
    y_span = float(pixel_size)

    hm2d = np.asarray(etch_depth_map_m, dtype=float)
    if hm2d.shape != (size, size):
        raise ValueError(f"etch_depth_map_m 期望 shape=({size},{size})，实际={hm2d.shape}")

    hm2d = np.clip(hm2d, 0.0, float(t_oxide))
    mask = hm2d > float(h_min)

    fdtd.addstructuregroup()
    fdtd.set("name", groupname)
    fdtd.groupscope(groupname)

    # 整片 SiO2 薄膜
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

    # 刻蚀槽
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
                print(f"  [build] {groupname}: slots={created_slots}, elapsed={dt:.1f}s")
                sys.stdout.flush()

    fdtd.groupscope("::model")
    dt = time.perf_counter() - t0
    print(f"  [build] {groupname} done: slots={created_slots}, elapsed={dt:.1f}s")
    sys.stdout.flush()

    return created_slots


def build_single_layer_fdtd(
    fdtd,
    *,
    layer_name: str,
    etch_depth_map_m: np.ndarray,
    size: int,
    pixel_size: float,
    wl0: float,
    t_oxide: float,
    n_sio2: float,
    background_index: float,
    use_3d: bool,
    h_min: float = 0.0,
):
    fdtd.groupscope("::model")

    x_span_total = size * pixel_size
    y_span_total = size * pixel_size
    x0 = x_span_total / 2.0
    y0 = -y_span_total / 2.0

    layer_z_top = 0.0

    pad_xy = 4 * pixel_size
    pad_z = 1.5e-6

    z_min_sim = layer_z_top - t_oxide - pad_z
    z_max_sim = layer_z_top + pad_z

    # ----------------------------------------------------------
    # 1. FDTD 仿真区域
    # ----------------------------------------------------------
    fdtd.addfdtd()

    if use_3d:
        fdtd.set("dimension", "3D")
    else:
        fdtd.set("dimension", "2D")

    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total + 2 * pad_xy)

    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", y_span_total + 2 * pad_xy)

    fdtd.set("z min", z_min_sim)
    fdtd.set("z max", z_max_sim)

    fdtd.set("background index", background_index)
    fdtd.set("mesh accuracy", 2)

    # 【修正】边界条件设置——按 Lumerical 要求的顺序
    # 2D 模式下 y 方向的边界条件是 inactive 的，不能设置
    fdtd.set("x min bc", "PML")
    # x max bc 会自动跟随 x min bc，如果需要单独设可以设
    if use_3d:
        fdtd.set("y min bc", "PML")
    # z 方向：先在 set z min/z max 之后，才能设 bc
    # Lumerical 的 z bc 默认跟 FDTD region 一起设，无需单独调用
    # 如果默认不是 PML，再显式设置：
    try:
        fdtd.set("z min bc", "PML")
    except Exception:
        pass  # 某些版本/维度下该属性可能 inactive
    try:
        fdtd.set("z max bc", "PML")
    except Exception:
        pass

    # 全局波长
    fdtd.setglobalsource("wavelength start", wl0)
    fdtd.setglobalsource("wavelength stop", wl0)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("use linear wavelength spacing", 1)

    # ----------------------------------------------------------
    # 2. 光源
    # ----------------------------------------------------------
    source_z = layer_z_top - t_oxide - 0.5e-6

    fdtd.addplane()
    fdtd.set("name", "plane_source")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "forward")
    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", y_span_total)
    fdtd.set("z", source_z)
    fdtd.set("wavelength start", wl0)
    fdtd.set("wavelength stop", wl0)

    # ----------------------------------------------------------
    # 3. 网格加密
    # ----------------------------------------------------------
    fdtd.addmesh()
    fdtd.set("name", "mesh_structure")
    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", y_span_total)
    fdtd.set("z min", layer_z_top - t_oxide - 0.2e-6)
    fdtd.set("z max", layer_z_top + 0.2e-6)

    fdtd.set("override x mesh", 1)
    if use_3d:
        fdtd.set("override y mesh", 1)
    fdtd.set("override z mesh", 1)
    fdtd.set("set maximum mesh step", 1)

    dx = 200e-9
    dz = 50e-9
    fdtd.set("dx", dx)
    if use_3d:
        fdtd.set("dy", dx)
    fdtd.set("dz", dz)

    # ----------------------------------------------------------
    # 4. SiO2 刻蚀结构
    # ----------------------------------------------------------
    slots = build_etched_sio2_layer_rowmerge(
        fdtd,
        groupname=layer_name,
        etch_depth_map_m=etch_depth_map_m,
        size=size,
        pixel_size=pixel_size,
        layer_z_top=layer_z_top,
        t_oxide=t_oxide,
        use_3d=use_3d,
        n_sio2=n_sio2,
        etch_fill_index=background_index,
        h_min=h_min,
        log_every=200,
    )

    # ----------------------------------------------------------
    # 5. 透射场监视器
    # ----------------------------------------------------------
    mon_z = layer_z_top + 0.3e-6

    fdtd.addpower()
    fdtd.set("name", "transmitted_field")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", y_span_total)
    fdtd.set("z", mon_z)

    # ----------------------------------------------------------
    # 6. 参考监视器
    # ----------------------------------------------------------
    ref_z = source_z + 0.2e-6

    fdtd.addpower()
    fdtd.set("name", "reference_field")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total)
    if use_3d:
        fdtd.set("y", y0)
        fdtd.set("y span", y_span_total)
    fdtd.set("z", ref_z)

    # ----------------------------------------------------------
    # 7. xz 截面 profile
    # ----------------------------------------------------------
    fdtd.addprofile()
    fdtd.set("name", "xz_profile")
    fdtd.set("monitor type", "2D Y-normal")
    fdtd.set("x", x0)
    fdtd.set("x span", x_span_total)
    if use_3d:
        fdtd.set("y", y0)
    fdtd.set("z min", z_min_sim + 0.2e-6)
    fdtd.set("z max", z_max_sim - 0.2e-6)

    return slots


def main():
    project_dir = get_project_dir()
    lumapi_path = r"D:\Program Files\Lumerical\v241\api\python\lumapi.py"
    height_map_path = resolve_existing_file(project_dir, "height_map.npy")

    lumapi = load_lumapi(lumapi_path)

    wl0 = WL0
    n_sio2 = N_SIO2
    background_index = N_AIR
    t_oxide = T_OXIDE

    size = 128
    pixel_size = 2e-6
    h_min = 0.0

    # 【修改】使用 2D 仿真以大幅减少内存（12GB → ~100MB）
    use_3d = False
    hide = True

    # 载入刻蚀深度图
    height_map = np.load(height_map_path)
    height_map = np.asarray(height_map, dtype=float)

    if height_map.ndim != 3:
        raise ValueError(f"height_map.npy 需要是 3D [num_layer,size,size]，实际 ndim={height_map.ndim}")

    num_layer = height_map.shape[0]
    if height_map.shape[1] != size or height_map.shape[2] != size:
        raise ValueError(
            f"height_map 期望 shape=[{num_layer},{size},{size}]，实际={height_map.shape}"
        )

    height_map = np.clip(height_map, 0.0, float(t_oxide))

    unique_depths_nm = np.unique(np.round(height_map * 1e9)).astype(int)
    print(f"[verify] 刻蚀深度唯一值(nm): {unique_depths_nm}")
    print(f"[verify] 期望值(nm): [0, 173, 346, 519]")

    # 检查是否有缺失的级别
    expected_levels = {0, 173, 346, 519}
    actual_levels = set(unique_depths_nm)
    missing = expected_levels - actual_levels
    if missing:
        print(f"[verify] ⚠ 缺少的刻蚀级别: {missing} nm")
        print(f"         这说明训练后的相位参数经 sigmoid 后没有覆盖到某些量化级别，属于正常现象。")
    print()

    # 逐层建模
    for l in range(num_layer):
        print(f"{'='*60}")
        print(f"[main] 构建第 {l} 层衍射层 FDTD 工程...")
        print(f"{'='*60}")

        out_fsp = os.path.join(project_dir, f"D2NN_layer{l}_verify.fsp")
        etch2d = height_map[l]

        vals_nm, counts = np.unique(np.round(etch2d * 1e9).astype(int), return_counts=True)
        print(f"  Layer {l} 刻蚀深度分布:")
        for v, c in zip(vals_nm, counts):
            print(f"    {v} nm: {c} pixels")

        fdtd = lumapi.FDTD(hide=hide)
        fdtd.groupscope("::model")

        t0 = time.perf_counter()

        slots = build_single_layer_fdtd(
            fdtd,
            layer_name=f"diffraction_layer_{l}",
            etch_depth_map_m=etch2d,
            size=size,
            pixel_size=pixel_size,
            wl0=wl0,
            t_oxide=t_oxide,
            n_sio2=n_sio2,
            background_index=background_index,
            use_3d=use_3d,
            h_min=h_min,
        )

        dt = time.perf_counter() - t0

        fdtd.save(out_fsp)
        fdtd.close()

        print(f"  [main] Layer {l}: slots={slots}, build time={dt:.1f}s")
        print(f"  [main] Saved: {out_fsp}")
        print()

    # 汇总
    print(f"{'='*60}")
    print(f"建模完成！共 {num_layer} 层。")
    print()
    print(f"参数汇总:")
    print(f"  波长:       {wl0*1e9:.0f} nm")
    print(f"  像素:       {size}x{size}, pitch={pixel_size*1e6:.1f} um")
    print(f"  t_oxide:    {t_oxide*1e9:.0f} nm (Fig. S1)")
    print(f"  n_SiO2:     {n_sio2}")
    print(f"  背景折射率: {background_index}")
    print(f"  仿真模式:   {'3D' if use_3d else '2D'}（{'完整' if use_3d else '截面，低内存'}）")
    print()
    print(f"下一步:")
    print(f"  1. 在 Lumerical 中打开 .fsp 文件并运行仿真")
    print(f"  2. 运行 export_transmittance.py 导出透过率")
    print(f"  3. 运行 verify_phase.py 验证相位一致性")
    print(f"  4. 运行 d2nn_full_inference.py 做完整推理")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()