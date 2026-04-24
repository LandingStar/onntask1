from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    sys.path.append(r"E:\Program Files\ANSYS Inc\v252\Lumerical\api\python")
    import lumapi  # type: ignore

    case_path = Path(
        r"e:\workspace\onn training\task1\lumerical-val\batch_runs_v252_show\floating_5det_one_phase2_jitter005_inside_only_20260423_1723\case\case_data.npz"
    )
    manifest_path = case_path.parent / "case_manifest.json"

    case = np.load(case_path, allow_pickle=False)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = manifest["config"]

    amplitude_hw = case["amplitude_hw"]
    x = case["x"]
    y = case["y"]
    z = case["z"]
    ex = case["Ex"]
    ey = case["Ey"]
    ez = case["Ez"]

    width = int(amplitude_hw.shape[1])
    height = int(amplitude_hw.shape[0])
    pixel_size = float(cfg["pixel_size"])
    detector_z = int(cfg["num_layers"]) * float(cfg["distance_between_layers"]) + float(
        cfg["distance_to_detectors"]
    )

    print("import ok", flush=True)
    fdtd = lumapi.FDTD(hide=False)
    print("fdtd ok", flush=True)
    fdtd.eval("switchtolayout; deleteall;")
    print("layout ok", flush=True)

    fdtd.addfdtd()
    print("addfdtd ok", flush=True)
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", 0)
    fdtd.set("x span", width * pixel_size)
    fdtd.set("y span", height * pixel_size)
    fdtd.set("z span", detector_z + 20e-6)
    print("region ok", flush=True)

    fdtd.addimportedsource()
    print("source ok", flush=True)
    fdtd.set("name", "src")
    fdtd.set("injection axis", "z-axis")
    fdtd.set("direction", "forward")
    print("source set ok", flush=True)

    fdtd.putv("x_arr", x)
    fdtd.putv("y_arr", y)
    fdtd.putv("z_arr", z)
    fdtd.putv("Ex_arr", ex)
    fdtd.putv("Ey_arr", ey)
    fdtd.putv("Ez_arr", ez)
    print("putv ok", flush=True)

    fdtd.eval(
        """
        EM = rectilineardataset("EM fields", x_arr, y_arr, z_arr);
        EM.addattribute("E", Ex_arr, Ey_arr, Ez_arr);
        select("src");
        importdataset(EM);
        """
    )
    print("importdataset ok", flush=True)

    fdtd.addprofile()
    print("profile ok", flush=True)
    fdtd.set("name", "detector_monitor")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", detector_z)
    fdtd.set("x span", width * pixel_size)
    fdtd.set("y span", height * pixel_size)
    print("monitor set ok", flush=True)

    fdtd.save(
        r"e:\workspace\onn training\task1\lumerical-val\batch_runs_v252_show\probe.fsp"
    )
    print("save ok", flush=True)
    fdtd.close()
    print("closed", flush=True)


if __name__ == "__main__":
    main()
