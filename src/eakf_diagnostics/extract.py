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

SCHEMA STATUS: verified against a real file (0722_601_real_nf_n_pois.mat,
inspected via inspect_file_schema() on 2026-07-22). Original assumptions
(inferred from README + MATLAB plotting script, before any real file was
available) were WRONG on axis order -- see notes below. Do not re-guess;
this has been checked against ground truth.

CONFIRMED SHAPES (run 601, 437 days x 150 ensemble x 96 locations x 196 params):
    para_post          (437, 150, 196)  -- axis order is (day, ensemble, param)
                                            NOT (param, ensemble, day) as first assumed
    alphamaps          (1, 96)          -- 2D row vector, not flat; values are
                                            float64 holding 1-indexed integer
                                            positions into para_post axis 2
    betamap            (1, 96)          -- same shape/dtype convention
    dailyIr_post_rec   (437, 150, 96)   -- (day, ensemble, location)
    S_post             (150, 437, 96)   -- (ensemble, day, location) -- note this
                                            is NOT the same axis order as
                                            dailyIr_post_rec, despite both having
                                            day and location axes. Verified from
                                            actual file, not assumed.
    paramin, paramax   (1, 196)         -- matches para_post axis 2, confirms
                                            196 is the param axis
    all_file_name      (1, 6) uint32    -- NOT a flat char array as first assumed;
                                            this is an HDF5 object-reference array
                                            (MATLAB char arrays >1 char nest this
                                            way in v7.3). Needs h5py ref
                                            dereferencing, not a plain array read.
                                            Still unresolved -- see _read_matlab_string.

196 params = 96 alpha + 96 beta + 4 remaining (Z, D, mu, theta per README).
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path


# Confirmed against a real 601 file. If a future run's file has a
# different top-level field set, load_model_run() will raise KeyError
# rather than silently misreading it.
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

# Fields that, if present, indicate this is actually a synthetic run
# (has ground truth) rather than a real-data run. Used as a guard so we
# fail loudly instead of silently mis-scoping a run. NOT yet verified
# against an actual synthetic-run file (only real-data 601 has been
# inspected so far) -- confirm this list once a synthetic Model_Runs
# file is inspected too.
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

    STATUS: unresolved. Confirmed on the 601 file that all_file_name has
    shape (1, 6) dtype uint32 -- 6 elements is too short to be a literal
    run filename (e.g. "0722_601_real_nf_n_pois.mat" is 28+ chars), which
    strongly suggests these are HDF5 object references (6 refs, possibly
    one per filename-component field in a struct) rather than char codes.
    Needs actual dereferencing against h5_obj['#refs#'] to resolve -- not
    done here yet since it's non-critical metadata (run identity can come
    from the file path instead, via run_path.stem). Returns None until
    fixed; callers should not rely on this field yet.
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

        # Sanity check: confirm alphamaps/betamap values are in-range for
        # para_post's param axis, so a silently wrong axis assumption
        # doesn't produce garbage instead of an error.
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
    Utility for pointing this at a NEW file for the first time: dumps the
    top-level field names, shapes, and dtypes without assuming anything
    about EXPECTED_FIELDS. Run this on any new run before trusting
    load_model_run() -- schema has only been confirmed against 601 so far;
    602/603/604 or future runs could still differ.
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

