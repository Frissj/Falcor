"""Prepare a capture-derived silhouette probe and compare Base/Alto placement volumes.

Requires NumPy. The probe recovers occupancy from the finished SDF; it does not
recover the original density, component textures, or authored placement.
"""

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = ("Base", "Alto")


def read_volume(folder, mip):
    shape = (64 >> mip, 512 >> mip, 512 >> mip, 2)
    path = folder / f"CloudTex_BASE_SDF_Tex_mip{mip}.bin"
    size = int(np.prod(shape))
    if path.stat().st_size != size:
        raise ValueError(f"{path}: expected exactly {size} bytes")
    return np.memmap(path, mode="r", dtype="u1", shape=shape)


def decode(values):
    t = np.clip(values.astype(np.float32) / 255, 1e-6, 1 - 1e-6)
    return (2 * np.clip(0.5 + np.log(t / (1 - t)) / 15, 0, 1) - (1 + 1 / 255)) * 32


def prepare(capture, density):
    volume = read_volume(capture, 0)
    density.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(LAYERS):
        # The encoded zero lies between codes 129 and 130 for this curve.
        mask = decode(volume[..., i]) < 0
        mask.astype("<f4").tofile(density / f"{layer}_density.f32")
        print(f"{layer}: {np.count_nonzero(mask):,}/{mask.size:,} inside voxels (derived from the captured SDF).")


def compare(capture, generated, band):
    result = {"distanceUnits": "mip-0 texels", "referenceBand": band, "captureMipFilter": {}, "mips": {}}
    for mip in range(3):
        ref = read_volume(capture, mip)
        out = read_volume(generated, mip)
        if mip:
            z, y, x, channels = ref.shape
            parent = read_volume(capture, mip - 1).reshape(z, 2, y, 2, x, 2, channels)
            mean = np.rint(parent.mean(axis=(1, 3, 5))).astype(np.int16)
            error = np.abs(mean - ref)
            result["captureMipFilter"][str(mip)] = {
                "rule": "2x2x2 encoded-byte mean, ties to even",
                "identicalBytesFraction": float(np.mean(error == 0)),
                "maxByteDifference": int(error.max()),
            }
        layers = result["mips"][str(mip)] = {}
        for i, layer in enumerate(LAYERS):
            a, b = decode(ref[..., i]), decode(out[..., i])
            near = np.abs(a) <= band
            inside_a, inside_b = a < 0, b < 0
            union = np.count_nonzero(inside_a | inside_b)
            error = np.abs(a[near] - b[near])
            layers[layer] = {
                "referenceInsideVoxels": int(np.count_nonzero(inside_a)),
                "generatedInsideVoxels": int(np.count_nonzero(inside_b)),
                "insideIoU": float(np.count_nonzero(inside_a & inside_b) / union) if union else None,
                "identicalBytesFraction": float(np.mean(ref[..., i] == out[..., i])),
                "nearSurfaceVoxels": int(np.count_nonzero(near)),
                "nearSurfaceMAE": float(error.mean()) if error.size else None,
                "nearSurfaceP95": float(np.percentile(error, 95)) if error.size else None,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    probe = sub.add_parser("prepare", help="Build diagnostic binary density fields from capture silhouettes")
    probe.add_argument("capture", type=Path)
    probe.add_argument("density", type=Path)
    diff = sub.add_parser("compare", help="Compare captured and generated Base/Alto mip files")
    diff.add_argument("capture", type=Path)
    diff.add_argument("generated", type=Path)
    diff.add_argument("--band", type=float, default=4.0)
    diff.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.capture, args.density)
    else:
        if not np.isfinite(args.band) or args.band <= 0:
            parser.error("--band must be finite and positive")
        output = json.dumps(compare(args.capture, args.generated, args.band), indent=2)
        if args.json:
            args.json.write_text(output + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
