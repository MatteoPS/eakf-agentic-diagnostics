"""
test_extract.py

Exercises the .mat loader against small HDF5 files written on the fly with
h5py -- same layout as a real MATLAB v7.3 Model_Runs file, but tiny and
synthetic. Covers the happy path, the alpha/beta index mapping (MATLAB is
1-indexed), and every schema guard that should fail loudly.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from eakf_diagnostics.checks import Severity, run_all_checks
from eakf_diagnostics.extract import inspect_file_schema, load_model_run

N_DAYS, N_ENS, N_LOC = 20, 8, 3
N_PARAMS = 2 * N_LOC + 1  # alpha block, beta block, one extra param


def default_fields():
    rng = np.random.default_rng(0)
    para_post = rng.normal(0.0, 0.1, size=(N_DAYS, N_ENS, N_PARAMS))
    para_post[:, :, :N_LOC] += 10.0            # alpha columns centred on 10
    para_post[:, :, N_LOC:2 * N_LOC] += 20.0   # beta columns centred on 20
    prior = np.abs(rng.normal(1.0, 0.1, size=(N_DAYS, N_LOC))) + 0.1
    return {
        "para_post": para_post,
        "alphamaps": np.arange(1, N_LOC + 1, dtype=float)[None, :],            # 1-indexed
        "betamap": np.arange(N_LOC + 1, 2 * N_LOC + 1, dtype=float)[None, :],
        "dailyIr_post_rec": np.zeros((N_DAYS, N_ENS, N_LOC)),
        "S_post": np.zeros((N_ENS, N_DAYS, N_LOC)),
        "paramin": np.arange(N_PARAMS, dtype=float)[None, :],
        "paramax": np.arange(N_PARAMS, dtype=float)[None, :] + 100,
        "prior_var_rec": prior,
        "post_var_rec": prior * 0.5,
        "all_file_name": np.zeros((1, 6), dtype=np.uint32),  # object-ref-like, as in the real 601 file
    }


def write_mat(path, drop=(), **overrides):
    fields = {**default_fields(), **overrides}
    with h5py.File(path, "w") as f:
        for key, value in fields.items():
            if key not in drop:
                f[key] = value
    return path


# ── Happy path ─────────────────────────────────────────────────────────────

def test_load_shapes_and_index_mapping(tmp_path):
    run = load_model_run(write_mat(tmp_path / "run.mat"))

    assert (run.n_days, run.n_ensemble, run.n_params) == (N_DAYS, N_ENS, N_PARAMS)
    assert run.alpha_trajectories.shape == (N_DAYS, N_ENS, N_LOC)
    # 1-indexed MATLAB maps must land on the right columns
    assert run.alpha_trajectories.mean() == pytest.approx(10.0, abs=0.1)
    assert run.beta_trajectories.mean() == pytest.approx(20.0, abs=0.1)

    alpha_min, alpha_max = run.alpha_bounds
    beta_min, _ = run.beta_bounds
    np.testing.assert_array_equal(alpha_min, [0, 1, 2])
    np.testing.assert_array_equal(alpha_max, [100, 101, 102])
    np.testing.assert_array_equal(beta_min, [3, 4, 5])

    assert run.all_file_name is None  # short uint32 array = object refs, not decoded
    assert run.location_name(1) == "location 1"
    assert set(run.raw_field_names) >= {"para_post", "prior_var_rec"}


def test_statecodes_give_readable_names(tmp_path):
    csv = tmp_path / "statecodes.csv"
    pd.DataFrame({"ID": [1, 2, 3], "State": ["Yukon", "Ohio", "Jalisco"],
                  "Country": ["Canada", "USA", "Mexico"]}).to_csv(csv, index=False)
    run = load_model_run(write_mat(tmp_path / "run.mat"), statecodes_path=csv)
    assert run.location_name(0) == "Yukon (Canada)"
    assert run.location_name(2) == "Jalisco (Mexico)"


def test_char_code_file_name_is_decoded(tmp_path):
    name = "0722_601_real_nf_n_pois.mat"
    codes = np.array([[ord(c) for c in name]], dtype=np.uint16)
    run = load_model_run(write_mat(tmp_path / "run.mat", all_file_name=codes))
    assert run.all_file_name == name


def test_run_all_checks_on_loaded_run(tmp_path):
    run = load_model_run(write_mat(tmp_path / "run.mat"))
    names = [r.check_name for r in run_all_checks(run)]
    assert names == ["ensemble_spread_collapse_alpha", "ensemble_spread_collapse_beta",
                     "kalman_update_activity"]

    with_cov = run_all_checks(run, coverage_observed=0.94)
    assert with_cov[-1].check_name == "coverage_miscalibration"
    assert with_cov[-1].severity == Severity.OK


# ── Schema guards ──────────────────────────────────────────────────────────

def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_run(tmp_path / "nope.mat")


def test_missing_field(tmp_path):
    with pytest.raises(KeyError, match="post_var_rec"):
        load_model_run(write_mat(tmp_path / "run.mat", drop=("post_var_rec",)))


def test_synthetic_run_rejected(tmp_path):
    path = write_mat(tmp_path / "run.mat", truth_para_post=np.zeros(3))
    with pytest.raises(ValueError, match="SYNTHETIC"):
        load_model_run(path)


@pytest.mark.parametrize("field", ["alphamaps", "betamap"])
def test_out_of_range_param_map(tmp_path, field):
    bad = np.array([[1.0, 2.0, N_PARAMS + 5]])
    with pytest.raises(ValueError, match=field):
        load_model_run(write_mat(tmp_path / "run.mat", **{field: bad}))


# ── Schema inspection ──────────────────────────────────────────────────────

def test_inspect_file_schema(tmp_path):
    path = write_mat(tmp_path / "run.mat")
    with h5py.File(path, "a") as f:
        f.create_group("#refs#").create_dataset("a", data=[1])

    schema = inspect_file_schema(path)
    assert schema["para_post"] == {"shape": (N_DAYS, N_ENS, N_PARAMS), "dtype": "float64"}
    assert schema["#refs#"]["keys"] == ["a"]
