"""Readers for Nikon/X-Tek CT metadata.

Responsibility: turn ``.xtekct`` and ``_ctdata.txt`` files into typed,
immutable records, and match projection image files to angle records by
*identifier*.  No geometry or physics happens here.

Inputs : file paths (str / pathlib.Path).
Outputs: ``XtekCT``, ``AngleTable``, ``MatchResult`` dataclasses.

Nothing is guessed: a missing *required* field raises ``MetadataError``;
an optional field that is absent is ``None``.  The ``.ang`` format has not
been inspected (no sample file has been supplied) so it is not parsed.
"""
from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class MetadataError(ValueError):
    """Raised when a metadata file is missing, malformed or incomplete."""


class AngFormatUnknownError(NotImplementedError):
    """Raised for ``.ang`` input: the format has not been established."""


# --------------------------------------------------------------------------
# .xtekct
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class XtekCT:
    """Selected, typed fields of a ``.xtekct`` file (units as in the file).

    Lengths are millimetres *if* ``units == 'mm'`` (the file's ``Units``
    field).  ``raw`` keeps every original ``key -> str`` for traceability.
    """

    name: str
    src_to_object: float
    src_to_detector: float
    detector_pixels: Tuple[int, int]          # (X, Y)
    detector_pixel_size: Tuple[float, float]  # (X, Y), detector plane
    projections: int
    detector_offset: Optional[Tuple[float, float]] = None
    voxels: Optional[Tuple[int, int, int]] = None
    voxel_size: Optional[Tuple[float, float, float]] = None
    object_offset: Optional[Tuple[float, float, float]] = None
    cor_top: Optional[float] = None
    cor_bottom: Optional[float] = None
    white_level: Optional[float] = None
    initial_angle: Optional[float] = None
    angular_step: Optional[float] = None
    xray_kv: Optional[float] = None
    xray_ua: Optional[float] = None
    units: Optional[str] = None
    mask_radius: Optional[float] = None
    raw: Dict[str, str] = field(default_factory=dict, compare=False)

    # -- derived quantities (calculated, not stored in the file) ----------
    @property
    def magnification(self) -> float:
        return self.src_to_detector / self.src_to_object

    @property
    def effective_pixel_size(self) -> Tuple[float, float]:
        """Detector pitch projected to the rotation axis (object plane)."""
        m = self.magnification
        return (self.detector_pixel_size[0] / m,
                self.detector_pixel_size[1] / m)


def _flat_items(path: Path) -> Dict[str, str]:
    cp = configparser.ConfigParser(interpolation=None, strict=True)
    cp.optionxform = str  # keep key case
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        cp.read_string(text)
    except (OSError, configparser.Error) as exc:
        raise MetadataError(f"Cannot read {path}: {exc}") from exc
    if "XTekCT" not in cp:
        raise MetadataError(f"{path}: no [XTekCT] section")
    items = {k: v.strip() for k, v in cp["XTekCT"].items()}
    # extra sections are kept with a section prefix
    for sec in cp.sections():
        if sec != "XTekCT":
            for k, v in cp[sec].items():
                items[f"{sec}.{k}"] = v.strip()
    return items


def _num(items: Dict[str, str], key: str, cast, required: bool):
    if key not in items or items[key] == "":
        if required:
            raise MetadataError(f"Required field {key!r} missing in .xtekct")
        return None
    try:
        return cast(items[key])
    except ValueError as exc:
        raise MetadataError(
            f"Field {key!r} has non-numeric value {items[key]!r}") from exc


def _pair(items, kx, ky, cast, required=False):
    x, y = _num(items, kx, cast, required), _num(items, ky, cast, required)
    if x is None or y is None:
        return None
    return (x, y)


def parse_xtekct(path) -> XtekCT:
    """Parse a ``.xtekct`` file.  Raises ``MetadataError`` on problems."""
    path = Path(path)
    it = _flat_items(path)
    f, i = float, int
    dpx = _pair(it, "DetectorPixelsX", "DetectorPixelsY", i, True)
    dps = _pair(it, "DetectorPixelSizeX", "DetectorPixelSizeY", f, True)
    sod = _num(it, "SrcToObject", f, True)
    sdd = _num(it, "SrcToDetector", f, True)
    nproj = _num(it, "Projections", i, True)
    for label, val in (("SrcToObject", sod), ("SrcToDetector", sdd),
                       ("DetectorPixelSizeX", dps[0]),
                       ("DetectorPixelSizeY", dps[1]),
                       ("DetectorPixelsX", dpx[0]),
                       ("DetectorPixelsY", dpx[1]), ("Projections", nproj)):
        if not val > 0:
            raise MetadataError(f"{label} must be > 0, got {val}")
    if sdd <= sod:
        raise MetadataError(
            f"SrcToDetector ({sdd}) must exceed SrcToObject ({sod})")

    def trip(prefix, cast):
        v = [_num(it, prefix + a, cast, False) for a in "XYZ"]
        return None if any(c is None for c in v) else tuple(v)

    return XtekCT(
        name=it.get("Name", ""),
        src_to_object=sod, src_to_detector=sdd,
        detector_pixels=dpx, detector_pixel_size=dps, projections=nproj,
        detector_offset=_pair(it, "DetectorOffsetX", "DetectorOffsetY", f),
        voxels=trip("Voxels", i), voxel_size=trip("VoxelSize", f),
        object_offset=trip("Offset", f),
        cor_top=_num(it, "CentreOfRotationTop", f, False),
        cor_bottom=_num(it, "CentreOfRotationBottom", f, False),
        white_level=_num(it, "WhiteLevel", f, False),
        initial_angle=_num(it, "InitialAngle", f, False),
        angular_step=_num(it, "AngularStep", f, False),
        xray_kv=_num(it, "Xrays.XraykV", f, False),
        xray_ua=_num(it, "Xrays.XrayuA", f, False),
        units=it.get("Units") or None,
        mask_radius=_num(it, "MaskRadius", f, False),
        raw=it,
    )


# --------------------------------------------------------------------------
# angles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AngleTable:
    """Per-projection angle records, keyed by the identifier in the file."""

    ids: np.ndarray                 # int64, as written in the file
    angles_deg: np.ndarray          # float64, as written in the file
    times_s: Optional[np.ndarray]   # float64 or None
    declared_projections: Optional[int]
    frames: Optional[int]
    exposure_ms: Optional[float]
    source: str                     # file path, for traceability

    def __len__(self) -> int:
        return int(self.ids.size)

    def summary(self) -> Dict[str, float]:
        """Numerical description of the angle sequence (all calculated)."""
        a = self.angles_deg
        n = a.size
        out: Dict[str, float] = {"count": n}
        if n == 0:
            return out
        out["first_deg"], out["last_deg"] = float(a[0]), float(a[-1])
        if n < 2:
            return out
        d = np.diff(a)
        idx = np.arange(n, dtype=float)
        slope, icpt = np.polyfit(idx, a, 1)
        resid = a - (slope * idx + icpt)
        out.update(
            step_mean_deg=float(d.mean()), step_median_deg=float(np.median(d)),
            step_min_deg=float(d.min()), step_max_deg=float(d.max()),
            fit_step_deg=float(slope),
            fit_residual_std_deg=float(resid.std()),
            fit_residual_max_deg=float(np.abs(resid).max()),
            monotonic_increasing=float(bool((d > 0).all())),
            coverage_first_to_last_deg=float(a[-1] - a[0]),
            coverage_incl_one_step_deg=float(a[-1] - a[0] + slope),
        )
        return out


def parse_ctdata(path) -> AngleTable:
    """Parse a Nikon ``_ctdata.txt`` file.

    Layout established from a real file (tab separated, CRLF)::

        Projections<TAB>Frames<TAB>Exposure(ms)
        3000<TAB>1<TAB>4000.00
        Projection<TAB>Angle(deg)<TAB>Time(s)
        1<TAB>000.116<TAB>024.260
        ...
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise MetadataError(f"Cannot read {path}: {exc}") from exc
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 4:
        raise MetadataError(f"{path}: too short to be a _ctdata.txt file")
    h1, h2, h3 = (lines[0].split("\t"), lines[1].split("\t"),
                  lines[2].split("\t"))
    if not (h1[0].strip() == "Projections" and h3[0].strip() == "Projection"
            and len(h3) >= 2 and h3[1].strip().startswith("Angle")):
        raise MetadataError(f"{path}: unexpected _ctdata.txt header")
    try:
        declared, frames = int(h2[0]), int(h2[1])
        exposure = float(h2[2])
    except (ValueError, IndexError) as exc:
        raise MetadataError(f"{path}: bad summary line {lines[1]!r}") from exc

    ids, ang, tim = [], [], []
    for n, ln in enumerate(lines[3:], start=4):
        parts = ln.split("\t")
        if len(parts) < 2:
            raise MetadataError(f"{path}:{n}: expected >=2 columns: {ln!r}")
        try:
            ids.append(int(parts[0]))
            ang.append(float(parts[1]))
            tim.append(float(parts[2]) if len(parts) > 2 else np.nan)
        except ValueError as exc:
            raise MetadataError(f"{path}:{n}: non-numeric row {ln!r}") from exc
    ids_a = np.asarray(ids, dtype=np.int64)
    if np.unique(ids_a).size != ids_a.size:
        raise MetadataError(f"{path}: duplicate projection identifiers")
    t = np.asarray(tim, dtype=np.float64)
    return AngleTable(ids_a, np.asarray(ang, dtype=np.float64),
                      None if np.isnan(t).all() else t,
                      declared, frames, exposure, str(path))


def parse_ang(path) -> AngleTable:  # noqa: ARG001
    """``.ang`` input is NOT implemented: no sample file has been inspected.

    The brief forbids guessing formats.  Supply an ``.ang`` file and this
    function will be implemented from its real contents.
    """
    raise AngFormatUnknownError(
        "The .ang format has not been established from a real file; "
        "use _ctdata.txt or provide an .ang sample.")


def load_angles(path) -> AngleTable:
    """Dispatch on file name: ``*_ctdata.txt`` or ``*.ang``."""
    p = Path(path)
    if p.name.lower().endswith("_ctdata.txt"):
        return parse_ctdata(p)
    if p.suffix.lower() == ".ang":
        return parse_ang(p)
    raise MetadataError(f"Unrecognised angle file type: {p.name}")


def check_consistency(ct: XtekCT, angles: AngleTable) -> List[str]:
    """Return human-readable warnings for conflicts between the two files.

    Nothing is silently resolved; the caller decides what to do.
    """
    w: List[str] = []
    if angles.declared_projections is not None \
            and angles.declared_projections != ct.projections:
        w.append(f"Projections: .xtekct={ct.projections} vs "
                 f"angle-file header={angles.declared_projections}")
    if len(angles) != ct.projections:
        w.append(f"Angle records ({len(angles)}) != Projections "
                 f"({ct.projections}); identifiers "
                 f"{angles.ids.min()}..{angles.ids.max()}")
    if ct.initial_angle is not None and angles.angles_deg.size \
            and not np.isclose(ct.initial_angle, angles.angles_deg[0],
                               atol=1.0):
        w.append(f"InitialAngle={ct.initial_angle:.4f} deg (.xtekct) vs first "
                 f"angle record {angles.angles_deg[0]:.4f} deg: convention "
                 "unresolved")
    if ct.angular_step is not None and len(angles) > 1:
        fit = angles.summary()["fit_step_deg"]
        if not np.isclose(fit, ct.angular_step, rtol=1e-3):
            w.append(f"AngularStep={ct.angular_step:.6f} vs fitted "
                     f"{fit:.6f} deg")
    return w


# --------------------------------------------------------------------------
# projection file <-> angle record matching
# --------------------------------------------------------------------------
DEFAULT_ID_REGEX = r"(\d+)(?=\.[A-Za-z0-9]+$)"  # last digit run before ext


@dataclass(frozen=True)
class MatchResult:
    """Outcome of identifier matching.  ``pairs`` is sorted by identifier."""

    pairs: Tuple[Tuple[int, str], ...]   # (angle id, file name)
    angle_ids_without_file: Tuple[int, ...]
    files_without_angle: Tuple[str, ...]
    files_without_id: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (self.angle_ids_without_file or self.files_without_angle
                    or self.files_without_id)


def match_projections(table: AngleTable, filenames: Sequence[str],
                      id_regex: str = DEFAULT_ID_REGEX,
                      id_offset: int = 0) -> MatchResult:
    """Match files to angle records by the number embedded in the name.

    ``file_id + id_offset`` must equal the angle-record identifier.  Set
    ``id_offset=1`` if files are numbered from 0 but records from 1.  The
    position of a file in the list is never used.  Two files that map to
    the same identifier raise ``MetadataError``.
    """
    rx = re.compile(id_regex)
    by_id: Dict[int, str] = {}
    no_id: List[str] = []
    for fn in filenames:
        m = rx.search(Path(fn).name)
        if not m:
            no_id.append(fn)
            continue
        fid = int(m.group(1)) + id_offset
        if fid in by_id:
            raise MetadataError(
                f"Files {by_id[fid]!r} and {fn!r} both map to id {fid}")
        by_id[fid] = fn
    known = set(int(x) for x in table.ids)
    pairs = tuple((i, by_id[i]) for i in sorted(known & by_id.keys()))
    return MatchResult(
        pairs=pairs,
        angle_ids_without_file=tuple(sorted(known - by_id.keys())),
        files_without_angle=tuple(by_id[i] for i in sorted(by_id.keys() - known)),
        files_without_id=tuple(no_id),
    )


def angles_for_matches(table: AngleTable, match: MatchResult) -> np.ndarray:
    """Angles (deg) for ``match.pairs`` in the same order, by identifier."""
    lookup = {int(i): float(a) for i, a in zip(table.ids, table.angles_deg)}
    return np.array([lookup[i] for i, _ in match.pairs], dtype=np.float64)
