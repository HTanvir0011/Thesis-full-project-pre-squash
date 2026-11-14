"""LIDC-IDRI XML -> consensus nodule masks (>=2 readers, 50% voxel agreement) on the CT voxel grid."""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from skimage.draw import polygon


def _name(tag):
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    return [c for c in el if _name(c.tag) == name]


def xml_series_uid(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        m = re.search(r"<SeriesInstanceUid>([^<]+)</SeriesInstanceUid>", f.read(8000))
    return m.group(1).strip() if m else None


def parse_xml(path):
    """-> list over readers of list of nodules {rois:[(sop, xs, ys)], malignancy}. Nodules need >=3-vertex contours."""
    readers = []
    for sess in (e for e in ET.parse(path).getroot() if _name(e.tag) == "readingSession"):
        nods = []
        for nod in _child(sess, "unblindedReadNodule"):
            rois = []
            for roi in _child(nod, "roi"):
                if (_child(roi, "inclusion") or [None])[0] is None or _child(roi, "inclusion")[0].text.strip().upper() != "TRUE":
                    continue
                pts = [(int(_child(e, "xCoord")[0].text), int(_child(e, "yCoord")[0].text)) for e in _child(roi, "edgeMap")]
                if len(pts) >= 3:
                    zt = _child(roi, "imageZposition")
                    rois.append((_child(roi, "imageSOP_UID")[0].text.strip(), [p[0] for p in pts], [p[1] for p in pts],
                                 float(zt[0].text) if zt else None))
            mal = None
            for ch in _child(nod, "characteristics"):
                v = _child(ch, "malignancy")
                mal = float(v[0].text) if v else None
            if rois:
                nods.append(dict(rois=rois, malignancy=mal))
        readers.append(nods)
    return readers


def _reader_masks(nod, locate, shape_yx, nz):
    m = {}
    for sop, xs, ys, zmm in nod["rois"]:
        z = locate(sop, zmm)
        if z is None or not 0 <= z < nz:
            continue
        rr, cc = polygon(ys, xs, shape_yx)
        m.setdefault(z, np.zeros(shape_yx, bool))[rr, cc] = True
    return m


def consensus_nodules(xml_paths, locate, shape_yx, nz, spacing, min_readers=2, clevel=0.5, link_mm=6.0):
    """locate(sop_uid, z_mm) -> slice index or None. -> list of dict(mask={z: 2D bool}, n_readers, malignancy, vol_mm3, diam_mm,
    centroid_zyx_vox)."""
    dz, dy, dx = spacing
    items = []  # (reader_id, masks, centroid_mm, malignancy)
    rid = 0
    for p in xml_paths:
        for nods in parse_xml(p):
            for nod in nods:
                masks = _reader_masks(nod, locate, shape_yx, nz)
                if not masks:
                    continue
                zs = np.concatenate([[z] * int(m.sum()) for z, m in masks.items()])
                ys_, xs_ = np.concatenate([np.nonzero(m) for m in masks.values()], axis=1)
                items.append((rid, masks, np.array([zs.mean() * dz, ys_.mean() * dy, xs_.mean() * dx]), nod["malignancy"]))
            rid += 1
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][0] == items[j][0]:
                continue
            near = np.linalg.norm(items[i][2] - items[j][2]) <= link_mm
            overlap = any(z in items[j][1] and (m & items[j][1][z]).any() for z, m in items[i][1].items())
            if near or overlap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idxs in groups.values():
        readers = {items[i][0] for i in idxs}
        if len(readers) < min_readers:
            continue
        thr = math.ceil(clevel * len(readers))
        votes = {}
        for i in idxs:
            for z, m in items[i][1].items():
                votes[z] = votes.get(z, 0) + m.astype(np.uint8)
        mask = {z: v >= thr for z, v in votes.items() if (v >= thr).any()}
        if not mask:
            continue
        nvox = sum(int(m.sum()) for m in mask.values())
        vol = nvox * dz * dy * dx
        zz = np.concatenate([[z] * int(m.sum()) for z, m in mask.items()])
        yy, xx = np.concatenate([np.nonzero(m) for m in mask.values()], axis=1)
        mals = [items[i][3] for i in idxs if items[i][3] is not None]
        out.append(dict(mask=mask, n_readers=len(readers), malignancy=float(np.mean(mals)) if mals else None,
                        vol_mm3=vol, diam_mm=(6 * vol / math.pi) ** (1 / 3),
                        centroid_zyx_vox=[float(zz.mean()), float(yy.mean()), float(xx.mean())]))
    return out
