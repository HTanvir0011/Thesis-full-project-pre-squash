"""Load a CT series to a head-first HU volume [z, y, x] (z index 0 = most superior)."""
import os

import numpy as np
import pydicom


class SkipCT(Exception):
    """A CT that cannot be used; the caller logs it and continues (corrupt files are skipped, never fatal)."""


def load_series(files, min_slices=80):
    heads = []
    for f in files:
        try:
            heads.append((f, pydicom.dcmread(f)))
        except Exception as e:
            raise SkipCT(f"unreadable {os.path.basename(f)}: {type(e).__name__}: {e}")
    heads = [(f, d) for f, d in heads if getattr(d, "Modality", "") == "CT" and hasattr(d, "ImagePositionPatient")]
    if len(heads) < min_slices:
        raise SkipCT(f"only {len(heads)} CT slices")
    heads.sort(key=lambda fd: float(fd[1].ImagePositionPatient[2]))  # feet -> head
    d0 = heads[0][1]
    iop = np.array(d0.ImageOrientationPatient, float)
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=0.02):
        raise SkipCT(f"non-axial orientation {iop.tolist()}")
    z = np.array([float(d.ImagePositionPatient[2]) for _, d in heads])
    dz_all = np.diff(z)
    dz = float(np.median(dz_all))
    if dz <= 0:
        raise SkipCT("duplicate slice positions")
    nonuni = float(np.max(np.abs(dz_all - dz)) / dz)
    if nonuni > 0.25:
        raise SkipCT(f"slice spacing varies by {nonuni:.2f} of median")
    rows, cols = int(d0.Rows), int(d0.Columns)
    vol = np.empty((len(heads), rows, cols), np.int16)
    for i, (_, d) in enumerate(heads):
        if (d.Rows, d.Columns) != (rows, cols):
            raise SkipCT("inconsistent slice size")
        hu = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        vol[i] = np.clip(hu, -1024, 3071)
    vol = vol[::-1]  # head first
    sop_to_idx = {str(d.SOPInstanceUID): len(heads) - 1 - i for i, (_, d) in enumerate(heads)}
    dy, dx = (float(v) for v in d0.PixelSpacing)
    locate = lambda sop, zmm: sop_to_idx.get(sop)  # noqa: E731  DICOM: exact SOP UID match
    meta = dict(n_slices=len(heads), dz=dz, dy=dy, dx=dx, z_nonuniform=nonuni, rows=rows, cols=cols,
                manufacturer=str(getattr(d0, "Manufacturer", "")), kvp=str(getattr(d0, "KVP", "")),
                kernel=str(getattr(d0, "ConvolutionKernel", "")), series_uid=str(getattr(d0, "SeriesInstanceUID", "")))
    return dict(hu=vol, spacing=(dz, dy, dx), sop_to_idx=sop_to_idx, locate=locate, meta=meta)


def load_mhd(path, min_slices=80):
    """LUNA16 MetaImage (.mhd + .raw) -> same dict as load_series. Slices are located by z position (mm)."""
    hdr = dict(line.split("=", 1) for line in open(path).read().splitlines() if "=" in line)
    hdr = {k.strip(): v.strip() for k, v in hdr.items()}
    nx, ny, nz = (int(v) for v in hdr["DimSize"].split())
    sx, sy, sz = (float(v) for v in hdr["ElementSpacing"].split())
    ox, oy, oz = (float(v) for v in hdr["Offset"].split())
    if hdr.get("ElementType") != "MET_SHORT" or hdr.get("ElementDataFile", "").endswith("zraw"):
        raise SkipCT(f"unsupported mhd: {hdr.get('ElementType')} {hdr.get('ElementDataFile')}")
    if nz < min_slices:
        raise SkipCT(f"only {nz} slices")
    raw = os.path.join(os.path.dirname(path), hdr["ElementDataFile"])
    vol = np.fromfile(raw, np.int16).reshape(nz, ny, nx)
    vol = np.clip(vol, -1024, 3071)[::-1]  # ascending z -> head first
    locate = lambda sop, zmm: (nz - 1) - int(round((zmm - oz) / sz)) if zmm is not None else None  # noqa: E731
    meta = dict(n_slices=nz, dz=sz, dy=sy, dx=sx, z_nonuniform=0.0, rows=ny, cols=nx, manufacturer="", kvp="", kernel="",
                series_uid=os.path.basename(path)[:-4], origin=[ox, oy, oz], transform=hdr.get("TransformMatrix", ""))
    return dict(hu=np.ascontiguousarray(vol), spacing=(sz, sy, sx), sop_to_idx={}, locate=locate, meta=meta)
