"""
extract.py

Reads EAKF pipeline output files (.mat, MATLAB v7.3 / HDF5 format) and
exposes their contents as plain numpy arrays.

Scope note: this module targets REAL-DATA runs (RunID 601-604), which have
no `truth_*` fields (no ground truth). Synthetic runs (RunID 1-140) have
additional fields (truth_para_post, truth_S_post, etc.) not handled here —
see docs/synthetic_mode.md if that path gets built later.

Why h5py and not scipy.io.loadmat: these files are MATLAB v7.3, which is
just HDF5 under the hood. scipy.io.loadmat only supports pre-v7.3 formats
and will raise on these files.
"""

from __future__ import annotations

import h5py
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# Fields we expect in a real-data Model_Runs .mat file.
# MATLAB stores arrays transposed relative to numpy's natural read order,
# and v7.3 files store each variable as an HDF5 dataset at the root group
# (or nested, for structs/cells -- verify against an actual file before
# trusting this list blindly).
EXPECTED_FIELDS = [
    "para_post",            # (n_params, n_ensemble, n_days) parameter ensemble trajectories
    "alphamaps",             # index map: which para_post rows are alpha (reporting rate)
    "betamap",                # index map: which para_post rows are beta (transmission rate)
    "dailyIr_post_rec",     # (n_locations, n_ensemble, n_days) reported daily incidence, posterior
    "S_post",                # (n_ensemble, n_locations, n_days) susceptible compartment, posterior
    "all_file_name",        # run identifier / filename metadata
]

# Fields that, if present, indicate this is actually a synthetic run
# (has ground truth) rather than a real-data run. Used as a guard so we
# fail loudly instead of silently mis-scoping a run.
SYNTHETIC_MARKER_FIELDS = ["truth_para_post", "truth_S_post", "truth_dailyIr_post_rec"]


@dataclass
class ModelRun:
    """Container for one Model_Runs/*.mat file's contents, as numpy arrays."""

    run_path: Path
    para_post: np.ndarray
    alphamaps: np.ndarray
    betamap: np.ndarray
    dailyIr_post_rec: np.ndarray
    S_post: np.ndarray
    all_file_name: str | None = None
    raw_field_names: list[str] = field(default_factory=list)  # for debugging/audit

    @property
    def n_ensemble(self) -> int:
        return self.para_post.shape[1]

    @property
    def n_days(self) -> int:
        return self.para_post.shape[2]

    @property
    def alpha_trajectories(self) -> np.ndarray:
        """(n_alpha_locations, n_ensemble, n_days)"""
        idx = self.alphamaps.astype(int).flatten() - 1  # MATLAB is 1-indexed
        return self.para_post[idx, :, :]

    @property
    def beta_trajectories(self) -> np.ndarray:
        """(n_beta_locations, n_ensemble, n_days)"""
        idx = self.betamap.astype(int).flatten() - 1
        return self.para_post[idx, :, :]


def _read_dataset(h5_obj, key: str) -> np.ndarray:
    """
    Pull a dataset out of an h5py File/Group and return as numpy array.
    MATLAB v7.3 stores numeric arrays with dims reversed relative to how
    MATLAB indexes them (Fortran vs C order) -- h5py reads them back in
    the reversed (numpy-natural, C order) shape. This has NOT yet been
    verified against a real file from this pipeline; TODO once real
    Model_Runs/*.mat files are available, confirm shapes here match what
    model_forecast_run.m actually writes (e.g. para_post should come back
    as (n_params, n_ensemble, n_days) -- verify, don't assume).
    """
    if key not in h5_obj:
        raise KeyError(f"Expected field '{key}' not found in .mat file")
    return np.array(h5_obj[key])


def load_model_run(path: str | Path) -> ModelRun:
    """
    Load a single Model_Runs/*.mat file (real-data run) into a ModelRun.

    Raises:
        FileNotFoundError: if path doesn't exist
        ValueError: if the file looks like a synthetic run (has truth_*
            fields) -- this loader is real-data-only by design; see module
            docstring.
        KeyError: if an expected field is missing (schema drift from what
            model_forecast_run.m currently writes)
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
                f"EXPECTED_FIELDS in extract.py -- update against a real file."
            )

        para_post = _read_dataset(f, "para_post")
        alphamaps = _read_dataset(f, "alphamaps")
        betamap = _read_dataset(f, "betamap")
        dailyIr_post_rec = _read_dataset(f, "dailyIr_post_rec")
        S_post = _read_dataset(f, "S_post")

        # all_file_name is likely a MATLAB char array / string ref; reading
        # it robustly with h5py needs a real file to get right. Stubbed.
        all_file_name = None
        try:
            raw = f["all_file_name"]
            all_file_name = "".join(chr(c) for c in np.array(raw).flatten())
        except Exception:
            pass  # non-critical metadata; don't fail the whole load over it

        return ModelRun(
            run_path=path,
            para_post=para_post,
            alphamaps=alphamaps,
            betamap=betamap,
            dailyIr_post_rec=dailyIr_post_rec,
            S_post=S_post,
            all_file_name=all_file_name,
            raw_field_names=field_names,
        )


def inspect_file_schema(path: str | Path) -> dict:
    """
    Utility for the FIRST time you point this at a real file: dumps the
    top-level field names, shapes, and dtypes without assuming anything
    about EXPECTED_FIELDS. Run this before trusting load_model_run() on
    a new file, since the schema here is inferred from the README/plotting
    script, not yet confirmed against an actual .mat file.
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
