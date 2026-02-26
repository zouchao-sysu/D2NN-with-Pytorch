"""
最小化验证脚本 v8.1
- 在每次 FDTD 实例之间加入等待时间，避免引擎未释放导致提取失败
"""
import os
import importlib.util
import numpy as np
import time


def load_lumapi(lumapi_path: str):
    spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 lumapi，请检查路径")
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)
    return lumapi


# ============================================================
# 论文公式
# ============================================================
WL = 632.8e-9
N_SIO2 = 1.46
N_AIR = 1.0
DN = N_SIO2 - N_AIR

PHI_MAX = 3 * np.pi / 4

def phase_to_height(phi):
    return WL * phi / (2 * np.pi * DN)

T_FILM = phase_to_height(PHI_MAX)

levels = []
for phi in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    h = phase_to_height(phi)
    levels.append({"phi": phi, "remaining_h": h})

print("="*60)
print(f"论文公式: Δh = λΦ/(2πΔn)")
print(f"  λ = {WL*1e9:.1f} nm, Δn = {DN:.2f}")
print(f"  膜厚 = {T_FILM*1e9:.1f} nm")
for lv in levels:
    print(f"  Φ={lv['phi']/np.pi:.2f}π → 残留={lv['remaining_h']*1e9:.1f}nm")
print("="*60)


def run_single(lumapi, project_dir, name, remaining_h, pixel_size):
    """运行单个 case，只放一个透射监视器"""

    out_fsp = os.path.join(project_dir, f"verify_{name}.fsp")

    # 多次重试启动 FDTD
    fdtd = None
    for attempt in range(3):
        try:
            fdtd = lumapi.FDTD(hide=True)
            break
        except Exception as e:
            print(f"  ⚠ 启动 FDTD 失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(5)
    if fdtd is None:
        print(f"  ❌ 无法启动 FDTD，跳过此 case")
        return None

    layer_z_top = 0.0
    xc = pixel_size / 2
    yc = -pixel_size / 2

    # FDTD 区域
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("x", xc)
    fdtd.set("x span", pixel_size)
    fdtd.set("y", yc)
    fdtd.set("y span", pixel_size)
    fdtd.set("z min", -T_FILM - 1.5e-6)
    fdtd.set("z max", 1.5e-6)
    fdtd.set("background index", N_AIR)
    fdtd.set("mesh accuracy", 3)
    fdtd.set("simulation time", 300e-15)
    fdtd.set("x min bc", "Periodic")
    fdtd.set("y min bc", "Periodic")

    fdtd.setglobalsource("wavelength start", WL)
    fdtd.setglobalsource("wavelength stop", WL)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("use linear wavelength spacing", 1)
    fdtd.setglobalmonitor("frequency points", 1)

    # z 方向加密
    fdtd.addmesh()
    fdtd.set("name", "mesh_z")
    fdtd.set("x", xc)
    fdtd.set("x span", pixel_size)
    fdtd.set("y", yc)
    fdtd.set("y span", pixel_size)
    fdtd.set("z min", -T_FILM - 0.2e-6)
    fdtd.set("z max", 0.2e-6)
    fdtd.set("override x mesh", 0)
    fdtd.set("override y mesh", 0)
    fdtd.set("override z mesh", 1)
    fdtd.set("set maximum mesh step", 1)
    fdtd.set("dz", 10e-9)

    # 光源
    src_z = -T_FILM - 1.0e-6
    fdtd.addplane()
    fdtd.set("name", "src")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "forward")
    fdtd.set("x", xc)
    fdtd.set("x span", pixel_size)
    fdtd.set("y", yc)
    fdtd.set("y span", pixel_size)
    fdtd.set("z", src_z)

    # SiO2 残留
    if remaining_h > 1e-12:
        fdtd.addrect()
        fdtd.set("name", "sio2")
        fdtd.set("x", xc)
        fdtd.set("x span", pixel_size * 2)
        fdtd.set("y", yc)
        fdtd.set("y span", pixel_size * 2)
        fdtd.set("z max", layer_z_top)
        fdtd.set("z min", layer_z_top - remaining_h)
        fdtd.set("material", "<Object defined dielectric>")
        fdtd.set("index", N_SIO2)

    # 透射监视器
    mon_z = layer_z_top + 1.0e-6
    fdtd.addpower()
    fdtd.set("name", "T_mon")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", xc)
    fdtd.set("x span", pixel_size)
    fdtd.set("y", yc)
    fdtd.set("y span", pixel_size)
    fdtd.set("z", mon_z)

    fdtd.save(out_fsp)
    print(f"  已保存: {out_fsp}")

    t0 = time.perf_counter()
    fdtd.run()
    dt = time.perf_counter() - t0
    print(f"  仿真完成, 用时 {dt:.1f}s")

    # 提取数据（带重试）
    Ex = None
    for attempt in range(3):
        try:
            Ex = np.squeeze(fdtd.getdata("T_mon", "Ex"))
            print(f"  Ex: shape={np.shape(Ex)}, max|Ex|={np.max(np.abs(Ex)):.4e}")
            break
        except Exception as e:
            print(f"  提取失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(2)

    if Ex is None:
        fdtd.close()
        time.sleep(5)
        return None

    E_mean = np.mean(Ex)
    abs_phase = float(np.angle(E_mean))
    amplitude = float(np.abs(E_mean))
    print(f"  |E|={amplitude:.4f}, arg(E)={abs_phase/np.pi:.4f}π")

    fdtd.close()
    # 【关键】等待引擎完全释放后再启动下一个
    time.sleep(5)

    return {"name": name, "remaining_nm": remaining_h*1e9,
            "abs_phase": abs_phase, "amplitude": amplitude}


def main():
    project_dir = os.getcwd()
    lumapi_path = r"D:\Program Files\Lumerical\v241\api\python\lumapi.py"
    lumapi_mod = load_lumapi(lumapi_path)
    pixel_size = 2e-6

    results = []
    for lv in levels:
        phi = lv["phi"]
        h = lv["remaining_h"]
        name = f"phi_{phi/np.pi:.2f}pi"

        print(f"\n{'='*60}")
        print(f"  {name}: Φ={phi/np.pi:.2f}π, 残留={h*1e9:.1f}nm")
        print(f"{'='*60}")

        r = run_single(lumapi_mod, project_dir, name, h, pixel_size)
        if r is not None:
            r["expected_phi"] = phi
            results.append(r)

    if len(results) < 2:
        print("\n❌ 结果不足")
        return

    # 找 Φ=0 作为参考
    ref = None
    for r in results:
        if r["expected_phi"] == 0:
            ref = r
            break
    if ref is None:
        ref = results[0]

    ref_abs_phase = ref["abs_phase"]

    print(f"\n\n{'='*70}")
    print(f"  参考: {ref['name']}, 绝对相位={ref_abs_phase/np.pi:.4f}π")
    print(f"{'='*70}")
    print(f"{'Φ期望':>8s}  {'残留nm':>8s}  {'相对相位':>10s}  {'误差':>8s}  {'振幅':>8s}  {'判定':>4s}")
    print(f"{'-'*70}")

    all_pass = True
    for r in results:
        rel_phase = r["abs_phase"] - ref_abs_phase
        while rel_phase < -0.01:
            rel_phase += 2 * np.pi
        while rel_phase > 2 * np.pi:
            rel_phase -= 2 * np.pi

        expected = r["expected_phi"]
        err = rel_phase - expected
        if err > np.pi:
            err -= 2 * np.pi
        if err < -np.pi:
            err += 2 * np.pi

        ok = abs(err) < 0.05 * np.pi
        if not ok:
            all_pass = False

        r["rel_phase"] = rel_phase
        r["error"] = err

        print(f"{expected/np.pi:>6.2f}π  "
              f"{r['remaining_nm']:>6.1f}  "
              f"{rel_phase/np.pi:>8.4f}π  "
              f"{err/np.pi:>6.4f}π  "
              f"{r['amplitude']:>8.4f}  "
              f"{'✓' if ok else '✗':>4s}")

    print(f"{'-'*70}")

    # 逐级相位差
    print(f"\n逐级相位差 (期望 = 0.2500π = 45°):")
    for i in range(1, len(results)):
        step = results[i]["rel_phase"] - results[i-1]["rel_phase"]
        if step < 0:
            step += 2 * np.pi
        expected_step = results[i]["expected_phi"] - results[i-1]["expected_phi"]
        err_step = step - expected_step
        print(f"  {results[i-1]['name']} → {results[i]['name']}: "
              f"Δφ = {step/np.pi:.4f}π (期望 {expected_step/np.pi:.2f}π, "
              f"误差 {err_step/np.pi:.4f}π)")

    print()
    if all_pass:
        print(f"  ✅ 全部通过! 论文公式验证正确。")
        print(f"")
        print(f"  正确参数汇总:")
        print(f"    波长:     {WL*1e9:.1f} nm")
        print(f"    膜厚:     {T_FILM*1e9:.1f} nm")
        print(f"    Φ 范围:   0 ~ {PHI_MAX/np.pi:.2f}π")
        print(f"    量化级别: 4 级, 间距 π/4")
    else:
        print(f"  ❌ 存在不一致")

    out = os.path.join(project_dir, "phase_verification_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("Name, Phi_expected(rad), Remaining(nm), AbsPhase(rad), "
                "RelPhase(rad), Error(rad), Amplitude\n")
        for r in results:
            f.write(f"{r['name']}, {r['expected_phi']:.6f}, {r['remaining_nm']:.1f}, "
                    f"{r['abs_phase']:.6f}, {r['rel_phase']:.6f}, "
                    f"{r['error']:.6f}, {r['amplitude']:.6f}\n")
    print(f"\n  已保存: {out}")


if __name__ == "__main__":
    main()