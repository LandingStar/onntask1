from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.append(r"e:\workspace\onn training\task1\lumerical-val")
import common  # noqa: E402


def main() -> None:
    run_dir = (
        r"e:\workspace\onn training\task1\lumerical-val\models"
        r"\floating_5det_one_phase2_jitter005_inside_only_20260423_1723"
    )
    artifacts = common.load_run_artifacts(run_dir)
    sample = common.load_sample(artifacts.config, dataset_split="val", sample_index=0)
    ref = common.run_python_reference(artifacts, sample.amplitude)
    exit_res = common.run_python_exit_field(artifacts, sample.amplitude)

    train = common.load_train_module()
    cfg = dict(artifacts.config)
    train.config = cfg
    train.device = torch.device("cpu")
    train.IMG_SIZE = cfg.get("img_size", [1000, 1000])
    train.PhaseMask = cfg.get("phase_mask_size", [1200, 1200])
    train.PIXEL_SIZE = float(cfg.get("pixel_size", 8e-6))
    train.wl = float(cfg.get("wavelength", 532e-9))
    train.PADDINGx = 0
    train.PADDINGy = 0
    train.detector_pos_xy = [tuple(map(float, pos)) for pos in artifacts.detector_pos.tolist()]

    model = train.DNN(
        num_layers=int(cfg.get("num_layers", artifacts.phase_masks.shape[0])),
        wl_param=float(cfg.get("wavelength", 532e-9)),
        PhaseMask_param=cfg.get("phase_mask_size", [1200, 1200]),
        pixel_size_param=float(cfg.get("pixel_size", 8e-6)),
        distance_between_layers=float(cfg.get("distance_between_layers", 0.15)),
        distance_to_detectors=float(cfg.get("distance_to_detectors", 0.2)),
        num_classes=int(cfg.get("num_classes", len(artifacts.detector_pos))),
        train_detector_pos=bool(cfg.get("train_detector_pos", False)),
    )
    model.load_state_dict(artifacts.state_dict, strict=True)
    model.eval()

    exit_t = torch.from_numpy(exit_res["exit_field"]).unsqueeze(0).to(torch.cfloat)
    with torch.no_grad():
        det_field = model.last_diffractive_layer(exit_t)
    det_int = (torch.abs(det_field) ** 2).squeeze(0).cpu().numpy()

    intensity_nrmse = np.linalg.norm(det_int - ref["intensity"]) / (np.linalg.norm(ref["intensity"]) + 1e-12)
    intensity_cosine = float(
        np.dot(det_int.reshape(-1), ref["intensity"].reshape(-1))
        / (np.linalg.norm(det_int.reshape(-1)) * np.linalg.norm(ref["intensity"].reshape(-1)) + 1e-12)
    )

    print(
        {
            "intensity_nrmse": float(intensity_nrmse),
            "intensity_cosine": intensity_cosine,
            "reference_predicted_detector": int(np.argmax(ref["output_vec"])),
        }
    )


if __name__ == "__main__":
    main()
