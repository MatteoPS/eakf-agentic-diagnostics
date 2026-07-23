"""
extract.py

Reads EAKF pipeline output files (.mat, MATLAB v7.3 / HDF5 format) and
exposes their contents as plain numpy arrays.

Scope note: this module targets REAL-DATA runs (RunID 601-604), which have
no `truth_*` fields (no ground truth). Synthetic runs (RunID 1-140) have
additional fields (truth_para_post, truth_S_post, etc.) not handled here.

Why h5py and not scipy.io.loadmat: these files are MATLAB v7.3, which is
just HDF5 under the hood. scipy.io.loadmat only supports pre-v7.3 formats
and will raise on these files.

Shapes below are from actually opening a real file (0722_601_real_nf_n_pois.mat,
via inspect_file_schema()) -- I'd guessed at some of this from the README
and the MATLAB plotting script first and got the axis order wrong, so don't
trust anything here that isn't backed by an actual inspected file.

CONFIRMED SHAPES (run 601, 437 days x 150 ensemble x 96 locations x 196 params):
    para_post          (437, 150, 196)  -- (day, ensemble, param). I originally
                                            had this as (param, ensemble, day),
                                            which was wrong.
    alphamaps          (1, 96)          -- 2D row vector, not flat; values are
                                            float64 holding 1-indexed integer
                                            positions into para_post axis 2
    betamap            (1, 96)          -- same shape/dtype convention
    dailyIr_post_rec   (437, 150, 96)   -- (day, ensemble, location)
    S_post             (150, 437, 96)   -- (ensemble, day, location) -- different
                                            axis order than dailyIr_post_rec even
                                            though both have day + location axes,
                                            so don't assume they line up.
    paramin, paramax   (1, 196)         -- matches para_post axis 2, confirms
                                            196 is the param axis
    all_file_name      (1, 6) uint32    -- not a flat char array like I expected;
                                            it's an HDF5 object-reference array
                                            (MATLAB nests char arrays >1 char this
                                            way in v7.3). Needs h5py ref
                                            dereferencing -- haven't done that
                                            yet, see _read_matlab_string.

196 params = 96 alpha + 96 beta + 4 remaining (Z, D, mu, theta per README).
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path


# checked against a real 601 file. if a future run has a different
# top-level field set, load_model_run() should raise KeyError instead of
# silently misreading it
EXPECTED_FIELDS = [
    "para_post",            # (n_days, n_ensemble, n_params) parameter ensemble trajectories
    "alphamaps",             # (1, n_alpha_locations) 1-indexed positions into para_post axis 2
    "betamap",                # (1, n_beta_locations) 1-indexed positions into para_post axis 2
    "dailyIr_post_rec",     # (n_days, n_ensemble, n_locations) reported daily incidence, posterior
    "S_post",                # (n_ensemble, n_days, n_locations) susceptible compartment, posterior
    "paramin",                # (1, n_params) lower bound per parameter
    "paramax",                # (1, n_params) upper bound per parameter
    "prior_var_rec",        # (n_days, n_locations) ensemble variance BEFORE each EAKF update
    "post_var_rec",          # (n_days, n_locations) ensemble variance AFTER each EAKF update
    "all_file_name",        # run identifier / filename metadata (HDF5 object ref)
]

# fields that, if present, mean this is actually a synthetic run (has
# ground truth) rather than real-data -- fail loudly instead of silently
# mis-scoping it. haven't actually opened a synthetic file yet to check
# this list, only real-data 601 so far
SYNTHETIC_MARKER_FIELDS = ["truth_para_post", "truth_S_post", "truth_dailyIr_post_rec"]


@dataclass
class ModelRun:
    """Container for one Model_Runs/*.mat file's contents, as numpy arrays."""

    run_path: Path
    para_post: np.ndarray           # (n_days, n_ensemble, n_params)
    alphamaps: np.ndarray
    betamap: np.ndarray
    dailyIr_post_rec: np.ndarray    # (n_days, n_ensemble, n_locations)
    S_post: np.ndarray               # (n_ensemble, n_days, n_locations)
    paramin: np.ndarray
    paramax: np.ndarray
    prior_var_rec: np.ndarray       # (n_days, n_locations) pre-update ensemble variance
    post_var_rec: np.ndarray        # (n_days, n_locations) post-update ensemble variance
    all_file_name: str | None = None
    statecodes: pd.DataFrame | None = None  # columns: ID (1-indexed), State, Country
    raw_field_names: list[str] = field(default_factory=list)  # for debugging/audit

    @property
    def n_days(self) -> int:
        return self.para_post.shape[0]

    @property
    def n_ensemble(self) -> int:
        return self.para_post.shape[1]

    @property
    def n_params(self) -> int:
        return self.para_post.shape[2]

    def location_name(self, idx_0: int) -> str:
        """Human-readable label for a 0-indexed location. Falls back to
        'location {idx_0}' if statecodes was not loaded."""
        if self.statecodes is None:
            return f"location {idx_0}"
        row = self.statecodes.iloc[idx_0]
        return f"{row['State']} ({row['Country']})"

    def _param_bounds(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """paramin/paramax sliced to the given param indices."""
        pmin = self.paramin.flatten()[idx]
        pmax = self.paramax.flatten()[idx]
        return pmin, pmax

    @property
    def alpha_indices(self) -> np.ndarray:
        """0-indexed positions into para_post axis 2 for alpha params."""
        return self.alphamaps.astype(int).flatten() - 1  # MATLAB is 1-indexed

    @property
    def beta_indices(self) -> np.ndarray:
        return self.betamap.astype(int).flatten() - 1

    @property
    def alpha_trajectories(self) -> np.ndarray:
        """(n_days, n_ensemble, n_alpha_locations)"""
        return self.para_post[:, :, self.alpha_indices]

    @property
    def beta_trajectories(self) -> np.ndarray:
        """(n_days, n_ensemble, n_beta_locations)"""
        return self.para_post[:, :, self.beta_indices]

    @property
    def alpha_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(paramin, paramax) arrays, one value per alpha location."""
        return self._param_bounds(self.alpha_indices)

    @property
    def beta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._param_bounds(self.beta_indices)


def _read_dataset(h5_obj, key: str) -> np.ndarray:
    """Pull a numeric dataset out of an h5py File/Group and return as numpy array."""
    if key not in h5_obj:
        raise KeyError(f"Expected field '{key}' not found in .mat file")
    return np.array(h5_obj[key])


def _read_matlab_string(h5_obj, key: str) -> str | None:
    """
    Best-effort read of a MATLAB v7.3 char-array field, which is often
    stored as an HDF5 object reference (dtype uint32/uint16 holding
    either raw char codes OR a reference into the '#refs#' group,
    depending on how MATLAB wrote it).

    On the 601 file, all_file_name comes back as shape (1, 6) dtype
    uint32 -- too short to be a literal filename ("0722_601_real_nf_n_pois.mat"
    is 28+ chars), so these are probably HDF5 object refs (maybe one per
    filename-component field in a struct) rather than char codes. Haven't
    bothered dereferencing against h5_obj['#refs#'] yet since it's not
    critical -- run identity can just come from run_path.stem instead.
    Returns None for now; don't rely on this field.
    """
    try:
        raw = h5_obj[key]
        raw_arr = np.array(raw)
        # Heuristic: if every value looks like a valid unicode codepoint
        # AND the array is long enough to plausibly be a filename, try it.
        if raw_arr.size > 10 and np.all(raw_arr < 0x110000):
            return "".join(chr(int(c)) for c in raw_arr.flatten())
        return None  # likely an object-reference array; not yet dereferenced
    except Exception:
        return None


def load_model_run(path: str | Path, statecodes_path: str | Path | None = None) -> ModelRun:
    """
    Load a single Model_Runs/*.mat file (real-data run) into a ModelRun.

    Args:
        path: path to the .mat file
        statecodes_path: optional path to statecodes.csv (columns: ID, State,
            Country). If provided, ModelRun.location_name(idx) returns
            human-readable labels. If omitted, falls back to 'location N'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with h5py.File(path, "r") as f:
        field_names = list(f.keys())

        synthetic_fields_present = [k for k in SYNTHETIC_MARKER_FIELDS if k in field_names]
        if synthetic_fields_present:
            raise ValueError(
                f"{path.name} looks like a SYNTHETIC run (found fields: "
                f"{synthetic_fields_present}). This loader targets real-data "
                f"runs only. Synthetic-run loading is not yet implemented."
            )

        missing = [k for k in EXPECTED_FIELDS if k not in field_names]
        if missing:
            raise KeyError(
                f"{path.name} is missing expected field(s): {missing}. "
                f"Fields present: {field_names}. Schema may have drifted from "
                f"EXPECTED_FIELDS in extract.py (last confirmed against 601)."
            )

        para_post = _read_dataset(f, "para_post")
        alphamaps = _read_dataset(f, "alphamaps")
        betamap = _read_dataset(f, "betamap")
        dailyIr_post_rec = _read_dataset(f, "dailyIr_post_rec")
        S_post = _read_dataset(f, "S_post")
        paramin = _read_dataset(f, "paramin")
        paramax = _read_dataset(f, "paramax")
        prior_var_rec = _read_dataset(f, "prior_var_rec")
        post_var_rec = _read_dataset(f, "post_var_rec")
        all_file_name = _read_matlab_string(f, "all_file_name")

        statecodes = None
        if statecodes_path is not None:
            statecodes = pd.read_csv(statecodes_path)

        run = ModelRun(
            run_path=path,
            para_post=para_post,
            alphamaps=alphamaps,
            betamap=betamap,
            dailyIr_post_rec=dailyIr_post_rec,
            S_post=S_post,
            paramin=paramin,
            paramax=paramax,
            prior_var_rec=prior_var_rec,
            post_var_rec=post_var_rec,
            all_file_name=all_file_name,
            statecodes=statecodes,
            raw_field_names=field_names,
        )

        # catch a wrong axis assumption here instead of letting it produce
        # garbage downstream
        n_params = run.n_params
        if run.alpha_indices.max() >= n_params or run.alpha_indices.min() < 0:
            raise ValueError(
                f"{path.name}: alphamaps values out of range for para_post's "
                f"{n_params}-param axis. Schema assumption may be wrong."
            )
        if run.beta_indices.max() >= n_params or run.beta_indices.min() < 0:
            raise ValueError(
                f"{path.name}: betamap values out of range for para_post's "
                f"{n_params}-param axis. Schema assumption may be wrong."
            )

        return run


def inspect_file_schema(path: str | Path) -> dict:
    """
    Point this at a NEW file first: dumps top-level field names, shapes,
    and dtypes without assuming anything about EXPECTED_FIELDS. Run before
    trusting load_model_run() on a file you haven't seen -- schema's only
    confirmed against 601 so far, 602/603/604 or future runs could differ.
    """
    path = Path(path)
    schema = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                schema[key] = {"shape": obj.shape, "dtype": str(obj.dtype)}
            else:
                schema[key] = {"type": "group (struct/cell?)", "keys": list(obj.keys())}
    return schema

