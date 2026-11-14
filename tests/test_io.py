import numpy as np
import pytest

from s4.ct import SkipCT, load_mhd
from s4.nodule import consensus_nodules, xml_series_uid

SQ = "<edgeMap><xCoord>{a}</xCoord><yCoord>{a}</yCoord></edgeMap><edgeMap><xCoord>{b}</xCoord><yCoord>{a}</yCoord></edgeMap>" \
     "<edgeMap><xCoord>{b}</xCoord><yCoord>{b}</yCoord></edgeMap><edgeMap><xCoord>{a}</xCoord><yCoord>{b}</yCoord></edgeMap>"


def _session(a, b, sop="s1", z=-10.0, mal=3):
    return (f"<readingSession><unblindedReadNodule><noduleID>1</noduleID><characteristics><malignancy>{mal}</malignancy></characteristics>"
            f"<roi><imageZposition>{z}</imageZposition><imageSOP_UID>{sop}</imageSOP_UID><inclusion>TRUE</inclusion>"
            f"{SQ.format(a=a, b=b)}</roi></unblindedReadNodule></readingSession>")


def _xml(tmp_path, sessions):
    p = tmp_path / "x.xml"
    p.write_text('<LidcReadMessage xmlns="http://www.nih.gov"><ResponseHeader><SeriesInstanceUid>1.2.3</SeriesInstanceUid>'
                 "</ResponseHeader>" + "".join(sessions) + "</LidcReadMessage>")
    return str(p)


def test_xml_uid_and_two_reader_consensus(tmp_path):
    p = _xml(tmp_path, [_session(10, 20), _session(12, 22)])
    assert xml_series_uid(p) == "1.2.3"
    nods = consensus_nodules([p], lambda sop, z: 5 if sop == "s1" else None, (64, 64), 10, (2.5, 1.0, 1.0))
    assert len(nods) == 1 and nods[0]["n_readers"] == 2 and nods[0]["malignancy"] == 3
    assert set(nods[0]["mask"]) == {5} and nods[0]["mask"][5].sum() > 100


def test_single_reader_nodule_is_not_consensus(tmp_path):
    p = _xml(tmp_path, [_session(10, 20)])
    assert consensus_nodules([p], lambda sop, z: 5, (64, 64), 10, (2.5, 1.0, 1.0)) == []


def test_three_readers_need_two_votes(tmp_path):
    p = _xml(tmp_path, [_session(10, 20), _session(10, 20), _session(40, 50)])
    nods = consensus_nodules([p], lambda sop, z: 5, (64, 64), 10, (2.5, 1.0, 1.0), link_mm=1.0)
    assert len(nods) == 1 and nods[0]["n_readers"] == 2  # the far reader forms its own single-reader cluster


def test_mhd_is_head_first_and_locates_by_z(tmp_path):
    nz = 100
    vol = np.arange(nz, dtype=np.int16)[:, None, None] * np.ones((nz, 8, 8), np.int16)
    vol.tofile(tmp_path / "a.raw")
    (tmp_path / "a.mhd").write_text(f"NDims = 3\nDimSize = 8 8 {nz}\nElementSpacing = 0.7 0.7 2.5\nOffset = 0 0 -300.0\n"
                                    "TransformMatrix = 1 0 0 0 1 0 0 0 1\nElementType = MET_SHORT\nElementDataFile = a.raw\n")
    ct = load_mhd(str(tmp_path / "a.mhd"))
    assert ct["hu"][0, 0, 0] == nz - 1 and ct["hu"][-1, 0, 0] == 0  # last ascending slice (head) comes first
    z_mm = -300.0 + 40 * 2.5  # ascending slice 40
    assert ct["hu"][ct["locate"](None, z_mm), 0, 0] == 40


def test_mhd_rejects_short_scans(tmp_path):
    np.zeros((10, 4, 4), np.int16).tofile(tmp_path / "b.raw")
    (tmp_path / "b.mhd").write_text("DimSize = 4 4 10\nElementSpacing = 1 1 1\nOffset = 0 0 0\nElementType = MET_SHORT\nElementDataFile = b.raw\n")
    with pytest.raises(SkipCT):
        load_mhd(str(tmp_path / "b.mhd"))
