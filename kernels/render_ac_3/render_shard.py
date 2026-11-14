"""Kaggle CPU kernel (template): render one shard of the 1,018 LIDC CTs into areal-density maps.
Placeholders SHARD / NSHARD / LIMIT / TEST are substituted by ops/launch_render.py. Corrupt CTs are logged and skipped."""
import csv
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

SHARD, NSHARD, LIMIT, TEST = 2, 3, None, False
OUT = Path("/kaggle/working/render")
TMP = Path("/kaggle/working/tmp")
code = glob.glob("/kaggle/input/*/s4/__init__.py") + glob.glob("/kaggle/input/datasets/*/*/s4/__init__.py")
assert code, "s4-code dataset not mounted"
CODE = Path(code[0]).parent.parent
sys.path.insert(0, str(CODE))
from s4.ct import SkipCT, load_mhd, load_series  # noqa: E402
from s4.nodule import xml_series_uid  # noqa: E402
from s4.render import render_ct, save  # noqa: E402

DATA = CODE / "s4" / "data"
LIDC_ROOT = next(iter(glob.glob("/kaggle/input/datasets/washingtongold/lidcidri30") + glob.glob("/kaggle/input/lidcidri30")), None)
LUNA_ROOT = next(iter(glob.glob("/kaggle/input/datasets/vafaeii/luna16") + glob.glob("/kaggle/input/luna16")), None)
UA = {"User-Agent": "Mozilla/5.0 (swintrust4 research; contact via thesis author)"}
MAX_SECONDS = 9 * 3600  # stop cleanly before the 12 h session cap
TCIA_SECONDS = 900  # one CT's download budget; a stuck download is skipped, never allowed to hang the shard


class _Budget(Exception):
    pass


def tcia_download(uid, dest):
    import signal

    def _stop(*_):
        raise _Budget()

    signal.signal(signal.SIGALRM, _stop)
    signal.alarm(TCIA_SECONDS)
    try:
        url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"
        zp = dest / "s.zip"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(zp, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zp) as z:
            assert z.testzip() is None
            z.extractall(dest)
        zp.unlink()
        return sorted(glob.glob(str(dest / "*.dcm")))
    except _Budget:
        raise SkipCT(f"tcia download exceeded {TCIA_SECONDS}s")
    except Exception as e:
        raise SkipCT(f"tcia download failed: {type(e).__name__}: {e}")
    finally:
        signal.alarm(0)


def work(job):
    uid, patient, mount_rel, luna_path, xmls = job
    name = f"{patient}_{uid[-8:]}"
    if (OUT / f"{name}.npz").exists():
        return dict(name=name, status="exists")
    t0 = time.time()
    tmp = TMP / name
    try:
        if mount_rel and LIDC_ROOT:
            ct = load_series(sorted(glob.glob(os.path.join(LIDC_ROOT, mount_rel, "*.dcm"))))
            src = "mount"
        elif luna_path:
            ct = load_mhd(luna_path)
            src = "luna"
        else:
            tmp.mkdir(parents=True, exist_ok=True)
            ct = load_series(tcia_download(uid, tmp))
            src = "tcia"
        arrays, meta = render_ct(ct, xmls)
        meta.update(name=name, patient=patient, source=src, xml_files=len(xmls), total_seconds=round(time.time() - t0, 1))
        save(OUT, name, arrays, meta)
        return dict(name=name, status="ok", seconds=meta["total_seconds"], source=src, trunc=meta["lung_truncated"],
                    nodules=meta["n_nodules"])
    except SkipCT as e:
        return dict(name=name, status="skipped", reason=str(e))
    except Exception as e:
        return dict(name=name, status="error", reason=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    xml_by_uid = {}
    for p in glob.glob(str(DATA / "lidc_xml" / "**" / "*.xml"), recursive=True):
        u = xml_series_uid(p)
        if u:
            xml_by_uid.setdefault(u, []).append(p)
    mount = {}
    for r in csv.DictReader(open(DATA / "lidc_mount_series.csv")):
        if r["Modality"] == "CT" and "lidcidri30/" in r["path"]:
            mount[r["SeriesInstanceUID"]] = r["path"].split("lidcidri30/", 1)[1]
    luna = {Path(p).stem: p for p in glob.glob(os.path.join(LUNA_ROOT, "subset*", "subset*", "*.mhd"))} if LUNA_ROOT else {}
    series = sorted(csv.DictReader(open(DATA / "tcia_lidc_ct_series.csv")), key=lambda r: r["PatientID"] + r["SeriesInstanceUID"])
    jobs = [(r["SeriesInstanceUID"], r["PatientID"], mount.get(r["SeriesInstanceUID"]), luna.get(r["SeriesInstanceUID"]),
             xml_by_uid.get(r["SeriesInstanceUID"], [])) for r in series]
    if TEST:  # 3 from the mount, 3 from LUNA16 only, 2 that must come from TCIA
        jobs = [j for j in jobs if j[2]][:3] + [j for j in jobs if not j[2] and j[3]][:3] + [j for j in jobs if not j[2] and not j[3]][:2]
    else:
        jobs = jobs[SHARD::NSHARD]
    if LIMIT:
        jobs = jobs[:LIMIT]
    print(f"shard {SHARD}/{NSHARD}: {len(jobs)} CTs: mount {sum(bool(j[2]) for j in jobs)}, luna "
          f"{sum(not j[2] and bool(j[3]) for j in jobs)}, tcia {sum(not j[2] and not j[3] for j in jobs)}; "
          f"{sum(bool(j[4]) for j in jobs)} with xml; roots {LIDC_ROOT} {LUNA_ROOT}", flush=True)
    t0, log = time.time(), []
    pool = mp.Pool(4, maxtasksperchild=1)
    it = pool.imap_unordered(work, jobs)
    with open("/kaggle/working/render_log.jsonl", "w") as lf:
        for i in range(1, len(jobs) + 1):
            try:
                res = it.next(timeout=max(1, MAX_SECONDS - (time.time() - t0)))
            except mp.TimeoutError:
                print("deadline reached, stopping with", len(log), "done", flush=True)
                break
            log.append(res)
            lf.write(json.dumps(res) + "\n")
            lf.flush()
            if i % 10 == 0 or res["status"] not in ("ok", "exists"):
                print(i, "/", len(jobs), f"{time.time() - t0:.0f}s", res, flush=True)
    pool.terminate()
    st = {k: sum(r["status"] == k for r in log) for k in ("ok", "exists", "skipped", "error")}
    print("DONE", st, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(dict(counts=st, seconds=round(time.time() - t0)), open("/kaggle/working/render_summary.json", "w"))
    shutil.rmtree(TMP, ignore_errors=True)
