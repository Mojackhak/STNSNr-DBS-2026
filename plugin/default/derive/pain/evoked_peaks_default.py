import pandas as pd
from plugin.default.tabular_default import standardize_df
from src.derive.pain.evoked_peaks import detect_evoked_peaks
from typing import Sequence

evoked_params_paths = [  
    'tfr/raw/raw-mean-trace.pkl',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    'tfr/period/period-mean-trace.pkl',
    'params/params/params-mean-trace.pkl',
    'conn/coh/coh-mean-trace.pkl',   
    'conn/ciplv/ciplv-mean-trace.pkl',
    'conn/wpli/wpli-mean-trace.pkl',
]

slice_cfg = {
    'raw-mean-trace': {'Band': ["Delta", "Theta", "Beta_low", "Gamma"]},
    'period-mean-trace': {'Band': ["Delta", "Theta", "Beta_low", "Alpha", "Gamma"]},
    'coh-mean-trace': {'Band':["Delta", ]},
    'ciplv-mean-trace': {'Band':["Delta", ]},
    'wpli-mean-trace': {'Band':["Delta", ]},
    'params-mean-trace': {'Band':["Offset", "Exponent"]},
}

peak_or_trough = {  # 'True' for peak, 'False' for trough
    'raw-mean-trace': {"Beta_low": False,}, 
    'period-mean-trace': {"Beta_low": False, "Alpha": False},          
    } 
    
detect_peaks_cfg = {
    'value_col': "Value",
    'detect_func': detect_evoked_peaks,  # to be set later
    'sep': "-",
    'exclude_keys': ("smoothed",),    
    'include_smoothed': True,         
    'boundary_mode': "min",         
}

peak_amp_sd_multiplier = 1.0  # Peak amplitude QC threshold in baseline SD units.
eps = 1e-6

phase = ['start', 'end']
qc = ['start_thr_ok', 'offset_thr_ok', 'ok']

def phase_qc_cols(phase: str) -> list[str]:
    """Return QC columns for the requested event phase only."""
    return [f"{phase}-{q}" for q in qc]

def peak_threshold_baseline(df: pd.DataFrame, phase: str) -> tuple[pd.Series, pd.Series]:
    """Return the baseline mean and SD used for peak amplitude QC."""
    if phase == "start":
        return df[f"{phase}-baseline_left_mean"], df[f"{phase}-baseline_left_sd"]
    if phase == "end":
        return df[f"{phase}-baseline_right_mean"], df[f"{phase}-baseline_right_sd"]
    raise ValueError(f"Unknown pain-evoked phase: {phase}")

def filter_df(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Filter dataframe using phase-specific boundary and amplitude QC."""
    mask_qc = df[phase_qc_cols(phase)].all(axis=1)
    mask_error = df["error"] == False
    baseline_mean, baseline_sd = peak_threshold_baseline(df, phase)
    thr_peak = peak_amp_sd_multiplier * baseline_sd + baseline_mean
    mask_peak = df[f"{phase}-peak_value"] > thr_peak
    mask_time = (df[f"{phase}-onset_time"] < df[f"{phase}-peak_time"] + eps)  & \
                (df[f"{phase}-offset_time"] > df[f"{phase}-peak_time"] + eps)
                
    mask = mask_qc & mask_error & mask_peak & mask_time
    df_filtered = df[mask].copy()
    
    return df_filtered


out_params = ['onset_time', 'peak_time', 'offset_time',
              'peak_value', 'auc', 'auc_rel_left', 'duration',
              'mean_slope_rise', 'mean_slope_fall']

def rename_cols(phase: str) -> str:
    if 'start' in phase.lower():
        return 'ONS'
    elif 'end' in phase.lower():
        return 'OFS'
    else:
        return phase

def melt_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Melt dataframe to long format for a given parameter."""
    df = df.copy()
    df.rename(columns={col: rename_cols(col) for col in cols}, inplace=True)
    cols = [rename_cols(col) for col in cols]
    df_melted = df.melt(
        id_vars=[c for c in df.columns if c not in cols],
        value_vars=cols,
        var_name="Phase", 
        value_name="Value",
    )
    df_melted = standardize_df(df_melted)
    
    return df_melted
