"""Tests for .xtekct parsing and config.  Expected values are either given
in the test text itself or computed independently of the parser."""
import math
from pathlib import Path

import pytest

from industrial_xct import config as cfg
from industrial_xct.io_xtek import MetadataError, parse_xtekct

DATA = Path(__file__).parent / "data"

MINIMAL = """[XTekCT]
Name=t
SrcToObject=100.0
SrcToDetector=400.0
DetectorPixelsX=64
DetectorPixelsY=32
DetectorPixelSizeX=0.5
DetectorPixelSizeY=0.25
Projections=10
"""


def write(tmp_path, text, name="a.xtekct"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_minimal_file_and_derived_values(tmp_path):
    ct = parse_xtekct(write(tmp_path, MINIMAL))
    assert ct.magnification == 4.0                    # 400/100
    assert ct.effective_pixel_size == (0.125, 0.0625)  # pitch / M
    assert ct.detector_pixels == (64, 32)
    assert ct.voxels is None and ct.cor_top is None   # absent -> None


@pytest.mark.parametrize("missing", ["SrcToObject", "SrcToDetector",
                                     "DetectorPixelsX", "Projections",
                                     "DetectorPixelSizeY"])
def test_missing_required_field_raises(tmp_path, missing):
    text = "\n".join(l for l in MINIMAL.splitlines()
                     if not l.startswith(missing + "="))
    with pytest.raises(MetadataError, match=missing):
        parse_xtekct(write(tmp_path, text))


def test_bad_values_raise(tmp_path):
    with pytest.raises(MetadataError, match="non-numeric"):
        parse_xtekct(write(tmp_path, MINIMAL.replace("100.0", "abc")))
    with pytest.raises(MetadataError, match="must exceed"):
        parse_xtekct(write(tmp_path, MINIMAL.replace("400.0", "50.0")))
    with pytest.raises(MetadataError, match="no \\[XTekCT\\]"):
        parse_xtekct(write(tmp_path, "[Other]\nx=1\n"))
    with pytest.raises(MetadataError, match="Cannot read"):
        parse_xtekct(tmp_path / "nope.xtekct")


def test_real_xtekct_values():
    """Literal values copied from the supplied scan file."""
    ct = parse_xtekct(DATA / "real.xtekct")
    assert ct.name == "Lunga_OlivChloriFluoriApatite130kV _469"  # has a space
    assert ct.src_to_object == 235.161392211914
    assert ct.src_to_detector == 1147.123
    assert ct.detector_pixels == (2048, 2048)
    assert ct.detector_pixel_size == (0.2, 0.2)
    assert ct.projections == 3000
    assert ct.voxel_size[0] == 0.041000204181534
    assert ct.white_level == 60000
    assert ct.xray_kv == 130 and ct.xray_ua == 154   # from [Xrays]
    assert ct.cor_top == 0.0 and ct.detector_offset == (0.0, 0.0)
    # Independent check: voxel size equals pitch/magnification
    assert math.isclose(ct.effective_pixel_size[0], ct.voxel_size[0],
                        rel_tol=1e-6)


def test_config_defaults_and_validation(tmp_path):
    c = cfg.load_config()
    assert c.memory.limit_gib == 4.0
    p = tmp_path / "c.yaml"
    p.write_text("memory:\n  limit_gib: 2.5\ninput:\n  id_offset: 1\n")
    c = cfg.load_config(p)
    assert c.memory.limit_gib == 2.5 and c.input.id_offset == 1
    p.write_text("memory:\n  limit_gb: 2\n")          # typo
    with pytest.raises(cfg.ConfigError, match="Unknown"):
        cfg.load_config(p)
    p.write_text("memory:\n  limit_gib: 0.1\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(p)
    p.write_text("input:\n  id_regex: '\\d+'\n")      # no capture group
    with pytest.raises(cfg.ConfigError, match="capture"):
        cfg.load_config(p)


def test_default_yaml_matches_code_defaults():
    yaml_cfg = cfg.load_config(Path(cfg.__file__).parent / "config" /
                               "default.yaml")
    assert yaml_cfg == cfg.load_config()
