"""Tests for _ctdata.txt parsing, angle statistics and identifier matching."""
from pathlib import Path

import numpy as np
import pytest

from industrial_xct.io_xtek import (AngFormatUnknownError, MetadataError,
                                    angles_for_matches, check_consistency,
                                    load_angles, match_projections,
                                    parse_ang, parse_ctdata, parse_xtekct)

DATA = Path(__file__).parent / "data"


def make_ctdata(tmp_path, rows, declared=None, crlf=True):
    head = ["Projections\tFrames\tExposure(ms)",
            f"{declared if declared is not None else len(rows)}\t1\t4000.00",
            "Projection\tAngle(deg)\tTime(s)"]
    body = [f"{i}\t{a:07.3f}\t{t:07.3f}" for i, a, t in rows]
    nl = "\r\n" if crlf else "\n"
    p = tmp_path / "x_ctdata.txt"
    p.write_bytes((nl.join(head + body) + nl).encode())
    return p


def test_parse_synthetic_with_known_statistics(tmp_path):
    rows = [(i + 1, 0.5 + 2.0 * i, 4.0 * i) for i in range(180)]  # 2 deg step
    t = parse_ctdata(make_ctdata(tmp_path, rows))
    assert len(t) == 180 and t.declared_projections == 180
    assert t.exposure_ms == 4000.0 and t.frames == 1
    s = t.summary()
    assert s["first_deg"] == 0.5 and s["last_deg"] == 358.5
    assert s["fit_step_deg"] == pytest.approx(2.0, abs=1e-9)
    assert s["coverage_first_to_last_deg"] == pytest.approx(358.0)
    assert s["coverage_incl_one_step_deg"] == pytest.approx(360.0)
    assert s["fit_residual_std_deg"] < 1e-9


def test_lf_and_crlf_both_parse(tmp_path):
    rows = [(1, 1.0, 1.0), (2, 2.0, 2.0)]
    a = parse_ctdata(make_ctdata(tmp_path, rows, crlf=True))
    b = parse_ctdata(make_ctdata(tmp_path, rows, crlf=False))
    assert np.array_equal(a.angles_deg, b.angles_deg)


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.replace("Projections", "Foo", 1), "header"),
    (lambda s: s + "7\tabc\t1.0\n", "non-numeric"),
    (lambda s: s + "1\t5.0\t1.0\n", "duplicate"),
    (lambda s: "\n".join(s.splitlines()[:3]), "too short"),
])
def test_malformed_ctdata_raises(tmp_path, mutate, msg):
    p = make_ctdata(tmp_path, [(1, 0.1, 1.0), (2, 0.2, 2.0)])
    p.write_text(mutate(p.read_text()))
    with pytest.raises(MetadataError, match=msg):
        parse_ctdata(p)


def test_real_ctdata_facts():
    """Facts established by inspecting the supplied file (see report)."""
    t = parse_ctdata(DATA / "real_ctdata.txt")
    assert t.declared_projections == 3000          # header says 3000 ...
    assert len(t) == 3001                          # ... but 3001 records
    assert t.ids[0] == 1 and t.ids[-1] == 3001
    assert t.angles_deg[0] == 0.116 and t.angles_deg[-1] == 359.760
    assert t.exposure_ms == 4000.0
    s = t.summary()
    assert s["monotonic_increasing"] == 1.0
    # Fitted step equals the .xtekct AngularStep to 7 digits
    assert s["fit_step_deg"] == pytest.approx(0.119902814565578, abs=1e-7)
    assert 0.05 < s["fit_residual_std_deg"] < 0.06  # angle jitter ~0.056 deg


def test_consistency_warnings_on_real_pair():
    ct = parse_xtekct(DATA / "real.xtekct")
    t = parse_ctdata(DATA / "real_ctdata.txt")
    w = " | ".join(check_consistency(ct, t))
    assert "3001" in w                       # record-count mismatch flagged
    assert "InitialAngle" in w               # 12.33 vs 0.116 flagged
    assert "AngularStep" not in w            # step agrees -> not flagged


def test_ang_is_not_guessed(tmp_path):
    with pytest.raises(AngFormatUnknownError):
        parse_ang(tmp_path / "a.ang")
    with pytest.raises(AngFormatUnknownError):
        load_angles(tmp_path / "scan.ANG")
    with pytest.raises(MetadataError):
        load_angles(tmp_path / "other.txt")


# ---- matching ---------------------------------------------------------
def table(tmp_path, n=5):
    rows = [(i, 10.0 * i, float(i)) for i in range(1, n + 1)]
    return parse_ctdata(make_ctdata(tmp_path, rows))


def test_matching_ignores_list_order_and_zero_padding(tmp_path):
    t = table(tmp_path)
    files = ["s_0003.tif", "s_0001.tif", "s_0005.tif", "s_0002.tif",
             "s_0004.tif"]                          # shuffled on purpose
    m = match_projections(t, files)
    assert m.complete
    assert [i for i, _ in m.pairs] == [1, 2, 3, 4, 5]
    assert [f for _, f in m.pairs][0] == "s_0001.tif"
    assert list(angles_for_matches(t, m)) == [10, 20, 30, 40, 50]


def test_matching_gaps_extras_and_unnumbered(tmp_path):
    t = table(tmp_path)
    m = match_projections(t, ["a_1.tif", "a_2.tif", "a_4.tif", "a_9.tif",
                              "notes.tif"])
    assert not m.complete
    assert m.angle_ids_without_file == (3, 5)
    assert m.files_without_angle == ("a_9.tif",)
    assert m.files_without_id == ("notes.tif",)
    # angle for id 4 is 40 even though it is the 3rd matched file
    assert list(angles_for_matches(t, m)) == [10, 20, 40]


def test_matching_zero_based_files_need_offset(tmp_path):
    t = table(tmp_path)                              # ids 1..5
    files = [f"p_{i:04d}.tif" for i in range(0, 5)]  # 0..4
    assert match_projections(t, files).angle_ids_without_file == (5,)
    m = match_projections(t, files, id_offset=1)
    assert m.complete and m.pairs[0] == (1, "p_0000.tif")


def test_duplicate_ids_raise(tmp_path):
    t = table(tmp_path)
    with pytest.raises(MetadataError, match="both map"):
        match_projections(t, ["a_0001.tif", "b_0001.tif"])


def test_regex_uses_last_number_before_extension(tmp_path):
    t = table(tmp_path)
    m = match_projections(t, ["Scan2_run7_0004.tif"])
    assert m.pairs == ((4, "Scan2_run7_0004.tif"),)
