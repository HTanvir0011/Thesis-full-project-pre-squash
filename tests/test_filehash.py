"""R6: sha256 of prep files (s4/filehash.py), the hash kernel, tools/compare_hashes.py and the ops/launch_hash.py arguments."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from s4.filehash import hash_mounted, hash_prep, sha256
from tools.compare_hashes import compare

ROOT = Path(__file__).resolve().parents[1]


def _prep(d, files):
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


def _hash_json(path, preps):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(preps, timings={p: 1.0 for p in preps})))
    return path


def test_sha256_known_values(tmp_path):
    (tmp_path / "abc").write_bytes(b"abc")
    abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # FIPS 180-2 test vector
    assert sha256(tmp_path / "abc") == abc and sha256(tmp_path / "abc", block=1) == abc  # streamed block by block: same digest
    (tmp_path / "empty").write_bytes(b"")
    assert sha256(tmp_path / "empty") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_prep_hashes_the_listed_files_present(tmp_path):
    d = _prep(tmp_path / "nih", {"split.npy": b"s", "orig.npy": b"o" * 1000, "ids.json": b"[]", "bone.npy": b"b", "split_f0.npy": b"f"})
    assert hash_prep(d) == {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("orig.npy", "split.npy", "ids.json")}


def test_hash_mounted_finds_preps_at_any_depth(tmp_path, capsys):
    nih = _prep(tmp_path / "s4-prep-nih" / "nih", {"split.npy": b"1", "soft.npy": b"2"})
    _prep(tmp_path / "notebooks" / "u" / "s4-prep-vin" / "vin", {"split.npy": b"3", "ann.json": b"{}"})
    _prep(tmp_path / "s4-code" / "s4", {"__init__.py": b""})
    _prep(tmp_path / "zz" / "nih", {"split.npy": b"other"})  # same name, later in sorted order: not the dir run_down's find_data reads
    hashes, secs = hash_mounted(str(tmp_path))
    assert set(hashes) == set(secs) == {"nih", "vin"}
    assert hashes["nih"] == hash_prep(nih) and set(hashes["vin"]) == {"split.npy", "ann.json"}
    assert f"not hashed: {tmp_path / 'zz' / 'nih'} (same name as {nih})" in capsys.readouterr().out  # in the log, not dropped silently
    with pytest.raises(AssertionError):
        hash_mounted(str(tmp_path / "s4-code"))  # no prep mounted: the kernel fails instead of writing an empty hash.json


def test_kernel_writes_hash_json(tmp_path):
    """kernels/hash/run_hash.py end to end, with /kaggle/ redirected to tmp_path; its output is what compare_hashes reads."""
    code = tmp_path / "input" / "s4-code" / "s4"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("")
    shutil.copy(ROOT / "s4" / "filehash.py", code / "filehash.py")
    nih = _prep(tmp_path / "input" / "s4-prep-nih" / "nih", {"split.npy": b"1", "orig.npy": b"2"})
    (tmp_path / "working").mkdir()
    k = tmp_path / "run_hash.py"
    k.write_text((ROOT / "kernels" / "hash" / "run_hash.py").read_text().replace("/kaggle/", tmp_path.as_posix() + "/"))
    r = subprocess.run([sys.executable, str(k)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "working" / "hash.json").read_text())
    assert out["nih"] == {"orig.npy": sha256(nih / "orig.npy"), "split.npy": sha256(nih / "split.npy")} and set(out["timings"]) == {"nih"}


def test_compare_identical_and_different(tmp_path):
    nih = {"orig.npy": "a" * 64, "soft.npy": "b" * 64}
    p6 = _hash_json(tmp_path / "ac_6" / "hash.json", {"nih": nih})
    p7 = _hash_json(tmp_path / "ac_7" / "hash.json", {"nih": nih, "vin": {"split.npy": "c" * 64}})
    lines, ok = compare([p6, p7])
    assert ok and lines[0] == "nih: 2 copies (ac_6, ac_7)" and "vin: 1 copy, nothing to compare (ac_7)" in lines
    assert [l.split() for l in lines[1:3]] == [["orig.npy", "identical"], ["soft.npy", "identical"]]
    p2 = _hash_json(tmp_path / "ac_2" / "hash.json", {"nih": {"orig.npy": "a" * 64, "soft.npy": "d" * 64}})
    lines, ok = compare([p6, p7, p2])
    assert not ok and lines[2] == "  soft.npy     DIFFERENT  dddddddddddd: ac_2 | bbbbbbbbbbbb: ac_6, ac_7"
    p3 = _hash_json(tmp_path / "ac_3" / "hash.json", {"nih": {"orig.npy": "a" * 64}})
    lines, ok = compare([p6, p3])
    assert not ok and "missing: ac_3" in lines[2]  # a file missing from one copy counts as different
    with pytest.raises(SystemExit):
        compare([p6, p6])  # one copy given twice would hide a missing account


def test_compare_cli_exit_code(tmp_path):
    for case, soft in (("same", "b"), ("diff", "d")):
        _hash_json(tmp_path / case / "ac_6" / "hash.json", {"nih": {"soft.npy": "b" * 64}})
        _hash_json(tmp_path / case / "ac_7" / "hash.json", {"nih": {"soft.npy": soft * 64}})
    cmd = [sys.executable, str(ROOT / "tools" / "compare_hashes.py")]
    r = subprocess.run(cmd + [str(tmp_path / "same" / "*" / "hash.json")], capture_output=True, text=True)  # unexpanded, as from PowerShell
    assert r.returncode == 0 and "identical" in r.stdout
    r = subprocess.run(cmd + [str(tmp_path / "diff" / "*" / "hash.json")], capture_output=True, text=True)
    assert r.returncode == 1 and "DIFFERENT" in r.stdout


def test_launch_hash_arguments():
    sys.path.insert(0, str(ROOT / "ops"))
    from launch_hash import parse
    assert parse(["ac_6", "s4-hash", "--prep", "s4-prep-nih"]) == ("ac_6", "s4-hash", ["s4-prep-nih"])
    assert parse(["ac_1", "s4-hash", "--prep", "s4-prep-nih,s4-prep-vin"])[2] == ["s4-prep-nih", "s4-prep-vin"]
    for bad in (["ac_6", "k"], ["ac_6", "k", "--prp", "a"], ["ac_6", "k", "--prep", "a", "--prev", "b"], ["ac_6", "k", "--prep", "a,"],
                ["ac_6", "--prep", "a"]):
        with pytest.raises(SystemExit):
            parse(bad)  # a typo or an extra option is an error, not silently ignored
