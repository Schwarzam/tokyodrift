import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from math import ceil
from .log_helper import open_log


# ============== Helpers básicos ==============

def _pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _linspace_palette(n):
    cmap = plt.cm.plasma
    return [cmap(v) for v in np.linspace(0.2, 0.9, max(1, n))]

def _col(df: pd.DataFrame, name: str):
    """Retorna a série existente; se não existir, devolve Série NaN do mesmo índice."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


# ============== Pré-processamento e segmentação ==============

def prepare_log(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna t0 (tempo normalizado) e estado heurístico."""
    d = df.copy()

    if "Time" not in d.columns:
        raise ValueError("Coluna 'Time' não encontrada.")
    d["t0"] = d["Time"] - d["Time"].min()

    # Estado heurístico (simples, ajustável)
    rpm = _col(d, "RPM")
    map_kpa = _col(d, "MAP")
    tps = _col(d, "TPS")
    tpsdot = _col(d, "TPS DOT")
    accel = _col(d, "Accel Enrich")
    dfco = _col(d, "DFCO")
    idle_ctl = _col(d, "Idle Control")
    boost = _col(d, "Boost PSI")
    iac_val = _col(d, "IAC value")

    cranking = (rpm > 0) & (rpm < 300)
    wot = ((tps > 80) | (map_kpa > 95) | (boost > 0.5)).fillna(False)
    accel_ev = ((accel > 0) | (tpsdot > 20)).fillna(False)
    dfco_ev = ((dfco == 1) | (((tps < 1) | tps.isna()) & (map_kpa < 25))).fillna(False)
    idle = ((tps < 2) & (rpm.between(600, 1300)) & ((idle_ctl == 1) | (iac_val > 0))).fillna(False)
    moving = (rpm >= 300).fillna(False)

    state = pd.Series("Cruise/Load", index=d.index, dtype="object")
    state[dfco_ev] = "Decel/DFCO"
    state[cranking] = "Cranking"
    state[wot] = "WOT/Boost"
    state[accel_ev & (~wot)] = "Accel"
    state[idle & (~cranking)] = "Idle"
    state[~moving] = "EngineOff"
    d["state"] = state
    return d


def add_fuel_estimates(
    df: pd.DataFrame,
    inj_flow_cc_min: float,     # vazão nominal do bico (cc/min @ inj_rated_dp_kpa)
    num_inj: int,               # número total de injetores ativos
    n_cyl: int,                 # número de cilindros
    fuel_density_g_cc: float = 0.74,   # gasolina ~0.72–0.76; E100 ~0.789; E85 ~0.76
    inj_rated_dp_kpa: float = 300.0,   # ΔP de referência do bico (geralmente 300 kPa = 3 bar)
    rail_pressure_kpa: float | None = None,  # se você souber a pressão absoluta da linha
    reg_ref: str = "manifold",        # "manifold" (regulador referenciado) ou "fixed"
    dc_col: str = "Duty Cycle",
    pw_col: str = "PW",
    rpm_col: str = "RPM",
    map_col: str = "MAP",
    baro_col: str = "Baro Pressure"
) -> pd.DataFrame:
    """
    Adiciona estimativas:
      - fuel_dc_used: duty usado (prioriza coluna 'Duty Cycle'; fallback via PW e RPM)
      - fuel_flow_per_inj_cc_s: vazão por injetor (corrigida por ΔP se possível)
      - fuel_flow_total_cc_s: vazão total (todos injetores) [cc/s]
      - fuel_flow_total_g_s: massa total [g/s]
      - fuel_mg_per_cyl: massa por cilindro por ciclo [mg/ciclo/cil]
      - fuel_mg_per_inj: massa por evento de injeção por cilindro [mg/injeção] (≈ igual ao anterior em sequencial 1x/720°)
    """
    d = df.copy()

    # Duty a usar: prioriza o log de Duty; senão estima por PW e RPM (4T, 1 injeção/720°)
    if dc_col in d.columns and d[dc_col].notna().any():
        dc = d[dc_col] / 100.0
    else:
        if pw_col in d.columns and rpm_col in d.columns:
            # DC ≈ PW(ms) * RPM / 1200 (sequencial, 1 injeção por 720°)
            dc = (d[pw_col] * d[rpm_col] / 1200.0).clip(lower=0, upper=1)
        else:
            raise ValueError("Nem 'Duty Cycle' nem (PW+RPM) estão disponíveis para estimar duty.")

    # ΔP através do injetor para correção de vazão
    # - Regulador referenciado ao coletor: ΔP efetivo ≈ constante ~ inj_rated_dp_kpa
    # - Regulador fixo (à atmosfera): ΔP = P_rail_abs - MAP_abs
    if reg_ref.lower().startswith("manifold") or rail_pressure_kpa is None:
        dp_kpa = np.full(len(d), inj_rated_dp_kpa, dtype=float)
    else:
        if map_col not in d.columns:
            map_guess = d[baro_col] if baro_col in d.columns else pd.Series(100.0, index=d.index)
        else:
            map_guess = d[map_col]
        dp_kpa = np.maximum(5.0, rail_pressure_kpa - map_guess)  # evita zero/negativo

    # Fator de correção da vazão do bico
    corr = np.sqrt(np.maximum(0.01, dp_kpa / float(inj_rated_dp_kpa)))

    # Vazão nominal por injetor [cc/s] @ ΔP_ref
    inj_cc_s_ref = float(inj_flow_cc_min) / 60.0
    # Vazão por injetor com correção de ΔP (independe do duty)
    d["fuel_flow_per_inj_cc_s"] = inj_cc_s_ref * corr

    # Duty e vazões totais
    d["fuel_dc_used"] = dc
    d["fuel_flow_total_cc_s"] = d["fuel_flow_per_inj_cc_s"] * float(num_inj) * d["fuel_dc_used"]

    # Massa total [g/s]
    d["fuel_flow_total_g_s"] = d["fuel_flow_total_cc_s"] * float(fuel_density_g_cc)

    # Massa por cilindro por ciclo (4 tempos): ciclos/s/cil = RPM/120
    rpm = d[rpm_col] if rpm_col in d.columns else pd.Series(np.nan, index=d.index)
    cycles_per_sec_total = (rpm / 120.0) * float(n_cyl)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["fuel_mg_per_cyl"] = (d["fuel_flow_total_g_s"] * 1000.0) / np.maximum(1e-6, cycles_per_sec_total)

    # Massa por injeção por cilindro (aprox. = mg/ciclo em sequencial 1x/720°)
    d["fuel_mg_per_inj"] = d["fuel_mg_per_cyl"]

    return d


def segment_by_state(df: pd.DataFrame, min_duration_s: float = 3.0) -> list:
    """Quebra o log em segmentos contínuos por 'state' com duração mínima."""
    if "state" not in df or "t0" not in df:
        raise ValueError("Chame prepare_log(df) antes.")
    segs = []
    cur_state = None
    start_i = None
    prev_t = None
    for i, (t, s) in enumerate(zip(df["t0"], df["state"])):
        if cur_state is None:
            cur_state, start_i, prev_t = s, i, t
            continue
        if (s != cur_state) or (t - prev_t > 2.0):
            t0 = df["t0"].iloc[start_i]
            t1 = prev_t
            if (t1 - t0) >= min_duration_s:
                segs.append({"state": cur_state, "t0": float(t0), "t1": float(t1)})
            cur_state, start_i = s, i
        prev_t = t
    if start_i is not None:
        t0 = df["t0"].iloc[start_i]
        t1 = df["t0"].iloc[-1]
        if (t1 - t0) >= min_duration_s:
            segs.append({"state": cur_state, "t0": float(t0), "t1": float(t1)})
    return segs


def thin_by_seconds(df: pd.DataFrame, every_s: float = 0.2) -> pd.DataFrame:
    """Reduz pontos mantendo medianas a cada 'every_s' segundos baseado em 't0'."""
    if "t0" not in df:
        raise ValueError("Chame prepare_log(df) antes.")
    g = (df["t0"] // every_s).astype(int)
    return df.groupby(g, as_index=False).median(numeric_only=True)


# ============== Render genérico multi-eixos ==============

def _render_multi(df, xcol, axes_defs, title, df_name="Log"):
    if not axes_defs:
        return None
    n = len(axes_defs)
    if n <= 5:
        ncols, nrows = 1, n
    else:
        ncols, nrows = 2, ceil(n/2)

    h_per_row = 2.6
    fig_w = 14 if ncols == 1 else 16
    fig_h = max(7, h_per_row * nrows)
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(fig_w, fig_h))
    axs = np.array(axs).reshape(-1)

    x = df[xcol]

    for i, ad in enumerate(axes_defs):
        ax = axs[i]
        series = ad["series"]
        colors = _linspace_palette(len(series))
        for (s, c) in zip(series, colors):
            ycol = s["col"]
            lbl = s.get("label", ycol)
            ax.plot(x, df[ycol], label=lbl, color=c, linewidth=1.4)
        ax.set_ylabel(ad.get("ylabel", ""))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncols=1)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    for j in range(len(axs)):
        if j >= n:
            fig.delaxes(axs[j])

    axs[min(n-1, len(axs)-1)].set_xlabel(xcol)
    fig.suptitle(f"{title} – {df_name}", y=0.995, fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig


# ============== Plots específicos ==============

def plot_partida(df: pd.DataFrame,
                 time_col='Time',
                 rpm_col='RPM',
                 map_col='MAP',
                 tps_col='TPS',
                 afr_col='AFR',
                 lambda_col='Lambda',
                 iat_col='IAT',
                 clt_col='CLT',
                 pw_col='PW',
                 duty_col='Duty Cycle',
                 gammae_col='Gammae',
                 afr_target_col='AFR Target',
                 lambda_target_col='Lambda Target',
                 iac_col='IAC value',
                 batt_col='Battery V',
                 advance_col='Advance (Current',
                 crank_rpm_thr=300,
                 figsize=(14, 9),
                 title='Partida – sinais relevantes (cranking sombreado)'):
    """Gráfico focado em cranking e pós-partida vs Time."""
    if time_col not in df.columns:
        raise ValueError(f"Coluna de tempo '{time_col}' não encontrada.")

    t = df[time_col] - df[time_col].min()
    cranking = (rpm_col in df.columns) and (df[rpm_col] > 0) & (df[rpm_col] < crank_rpm_thr)

    plt.rcParams['figure.figsize'] = figsize
    fig, axes = plt.subplots(5, 1, sharex=True)

    # 1) RPM + MAP
    ax = axes[0]
    if rpm_col in df.columns:
        ax.plot(t, df[rpm_col], label='RPM', color=plt.cm.plasma(0.3))
        ax.set_ylabel('RPM')
        ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if map_col in df.columns:
        ax2.plot(t, df[map_col], label='MAP (kPa)', color=plt.cm.plasma(0.8), alpha=0.9)
        ax2.set_ylabel('MAP (kPa)')
        ax2.legend(loc='upper right')
    if rpm_col in df.columns:
        ax.fill_between(t, 0, 1, where=cranking,
                        transform=ax.get_xaxis_transform(),
                        color='gray', alpha=0.15, label='Cranking')

    # 2) PW + Duty/Gammae
    ax = axes[1]
    if pw_col in df.columns:
        ax.plot(t, df[pw_col], label='PW (ms)', color=plt.cm.plasma(0.5))
        ax.set_ylabel('PW (ms)')
        ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    lines2, labels2 = [], []
    if duty_col in df.columns:
        l1, = ax2.plot(t, df[duty_col], label='Duty Cycle (%)', color=plt.cm.plasma(0.7), alpha=0.9)
        lines2.append(l1); labels2.append(l1.get_label())
    if gammae_col in df.columns:
        l2, = ax2.plot(t, df[gammae_col], label='Gammae (%)', color=plt.cm.plasma(0.9), alpha=0.9)
        lines2.append(l2); labels2.append(l2.get_label())
    if lines2:
        ax2.set_ylabel('%')
        ax2.legend(lines2, labels2, loc='upper right')

    # 3) Mistura
    ax = axes[2]
    if lambda_col in df.columns:
        ax.plot(t, df[lambda_col], label='Lambda', color=plt.cm.plasma(0.4))
        if lambda_target_col in df.columns:
            ax.plot(t, df[lambda_target_col], label='Lambda Target', color=plt.cm.plasma(0.85), linestyle='--')
        ax.set_ylabel('Lambda')
        ax.legend(loc='upper right')
    elif afr_col in df.columns:
        ax.plot(t, df[afr_col], label='AFR', color=plt.cm.plasma(0.4))
        if afr_target_col in df.columns:
            ax.plot(t, df[afr_target_col], label='AFR Target', color=plt.cm.plasma(0.85), linestyle='--')
        ax.set_ylabel('AFR')
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4) TPS/IAC + Advance
    ax = axes[3]
    lines, labels = [], []
    if tps_col in df.columns:
        l, = ax.plot(t, df[tps_col], label='TPS (%)', color=plt.cm.plasma(0.6))
        lines.append(l); labels.append(l.get_label())
    if iac_col in df.columns:
        l, = ax.plot(t, df[iac_col], label='IAC value', color=plt.cm.plasma(0.2))
        lines.append(l); labels.append(l.get_label())
    ax.set_ylabel('TPS / IAC')
    if lines:
        ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if advance_col in df.columns:
        ax2.plot(t, df[advance_col], label='Advance (°)', color=plt.cm.plasma(0.8))
        ax2.set_ylabel('Advance (°)')
        ax2.legend(loc='upper right')

    # 5) Temperaturas + Bateria
    ax = axes[4]
    lines, labels = [], []
    if clt_col in df.columns:
        l, = ax.plot(t, df[clt_col], label='CLT (°C)', color=plt.cm.plasma(0.25))
        lines.append(l); labels.append(l.get_label())
    if iat_col in df.columns:
        l, = ax.plot(t, df[iat_col], label='IAT (°C)', color=plt.cm.plasma(0.75))
        lines.append(l); labels.append(l.get_label())
    ax.set_ylabel('Temp (°C)')
    if lines:
        ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if batt_col in df.columns:
        ax2.plot(t, df[batt_col], label='Battery V', color=plt.cm.plasma(0.9))
        ax2.set_ylabel('Battery (V)')
        ax2.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_auto_engine_views(df: pd.DataFrame, df_name: str = "Log"):
    """Seleciona gráficos relevantes e gera figuras agrupadas por eixo X."""
    def has(col): return col in df.columns
    def pick(*cands): return _pick(df, *cands)

    used_cols = set()

    time_col = pick("Time")
    rpm_col = pick("RPM")
    map_col = pick("MAP")
    load_col = pick("FuelLoad", "Fuel Load")
    spd_col = pick("Vehicle Speed", "VSS")

    lambda_col = pick("Lambda")
    lambda_tgt = pick("Lambda Target")
    afr_col = pick("AFR")
    afr_tgt = pick("AFR Target")

    pw_cols = [c for c in ["PW", "PW2", "PW3", "PW4"] if has(c)]
    duty_col = pick("Duty Cycle")
    gammae_col = pick("Gammae")
    gwarm = pick("Gwarm")
    gego = pick("Gego")
    gbatt = pick("Gbattery")

    ve_col = pick("VE (Current", "VE Current", "VE")
    tps_col = pick("TPS")
    iac_col = pick("IAC value")
    adv_col = pick("Advance (Current", "Advance")
    dwell_col = pick("Dwell")
    batt_v = pick("Battery V")
    clt_col = pick("CLT")
    iat_col = pick("IAT")
    boost_col = pick("Boost PSI")
    mapxrpm = pick("MAPxRPM")
    ign_load = pick("IgnitionLoad")
    fuel_load = load_col
    power_col = pick("Power")
    torque_col = pick("Torque")

    # X = Time
    if time_col:
        axes_time = []
        if rpm_col:
            axes_time.append({"ylabel": "RPM", "series": [{"col": rpm_col, "label": "RPM"}]})
            used_cols.add(rpm_col)
        if map_col:
            axes_time.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
            used_cols.add(map_col)

        if lambda_col:
            ser = [{"col": lambda_col, "label": "Lambda"}]
            if lambda_tgt:
                ser.append({"col": lambda_tgt, "label": "Lambda Target"}); used_cols.add(lambda_tgt)
            axes_time.append({"ylabel": "Lambda", "series": ser}); used_cols.add(lambda_col)
        elif afr_col:
            ser = [{"col": afr_col, "label": "AFR"}]
            if afr_tgt:
                ser.append({"col": afr_tgt, "label": "AFR Target"}); used_cols.add(afr_tgt)
            axes_time.append({"ylabel": "AFR", "series": ser}); used_cols.add(afr_col)

        if pw_cols:
            axes_time.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
            used_cols.update(pw_cols)

        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if gwarm: duty_series.append({"col": gwarm, "label": "Gwarm (%)"})
        if gego: duty_series.append({"col": gego, "label": "Gego (%)"})
        if gbatt: duty_series.append({"col": gbatt, "label": "Gbattery (%)"})
        if duty_series:
            axes_time.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
            used_cols.update([s["col"] for s in duty_series])

        if ve_col:
            axes_time.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
            used_cols.add(ve_col)

        tps_iac_series = []
        if tps_col: tps_iac_series.append({"col": tps_col, "label": "TPS (%)"})
        if iac_col: tps_iac_series.append({"col": iac_col, "label": "IAC"})
        if tps_iac_series:
            axes_time.append({"ylabel": "TPS / IAC", "series": tps_iac_series})
            used_cols.update([s["col"] for s in tps_iac_series])

        if adv_col or dwell_col:
            ign_series = []
            if adv_col: ign_series.append({"col": adv_col, "label": "Advance (°)"})
            if dwell_col: ign_series.append({"col": dwell_col, "label": "Dwell (ms)"})
            axes_time.append({"ylabel": "Ignição", "series": ign_series})
            used_cols.update([s["col"] for s in ign_series])

        temp_series = []
        if clt_col: temp_series.append({"col": clt_col, "label": "CLT (°C)"})
        if iat_col: temp_series.append({"col": iat_col, "label": "IAT (°C)"})
        if temp_series:
            axes_time.append({"ylabel": "Temperatura (°C)", "series": temp_series})
            used_cols.update([s["col"] for s in temp_series])
        if batt_v:
            axes_time.append({"ylabel": "Battery (V)", "series": [{"col": batt_v, "label": "Battery V"}]})
            used_cols.add(batt_v)

        fl_series = []
        if fuel_load: fl_series.append({"col": fuel_load, "label": "FuelLoad"})
        if ign_load: fl_series.append({"col": ign_load, "label": "IgnitionLoad"})
        if mapxrpm: fl_series.append({"col": mapxrpm, "label": "MAP×RPM"})
        if fl_series:
            axes_time.append({"ylabel": "Carga/Produtos", "series": fl_series})
            used_cols.update([s["col"] for s in fl_series])

        if boost_col:
            axes_time.append({"ylabel": "Boost (PSI)", "series": [{"col": boost_col, "label": "Boost PSI"}]})
            used_cols.add(boost_col)

        if spd_col:
            axes_time.append({"ylabel": "Velocidade", "series": [{"col": spd_col, "label": "Vehicle Speed"}]})
            used_cols.add(spd_col)
        pt_series = []
        if power_col: pt_series.append({"col": power_col, "label": "Power"})
        if torque_col: pt_series.append({"col": torque_col, "label": "Torque"})
        if pt_series:
            axes_time.append({"ylabel": "Potência/Torque", "series": pt_series})
        used_cols.update([s["col"] for s in pt_series] if pt_series else [])

        _render_multi(df, time_col, axes_time, "X = Time", df_name=df_name)

    # X = RPM
    if rpm_col:
        axes_rpm = []
        if map_col:
            axes_rpm.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
        if lambda_col or afr_col:
            series = []
            if lambda_col:
                series.append({"col": lambda_col, "label": "Lambda"})
                if lambda_tgt: series.append({"col": lambda_tgt, "label": "Lambda Target"})
            else:
                series.append({"col": afr_col, "label": "AFR"})
                if afr_tgt: series.append({"col": afr_tgt, "label": "AFR Target"})
            axes_rpm.append({"ylabel": "Mistura", "series": series})
        if pw_cols:
            axes_rpm.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if duty_series:
            axes_rpm.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
        if adv_col:
            axes_rpm.append({"ylabel": "Advance (°)", "series": [{"col": adv_col, "label": "Advance"}]})
        if ve_col:
            axes_rpm.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
        pt_series = []
        if power_col: pt_series.append({"col": power_col, "label": "Power"})
        if torque_col: pt_series.append({"col": torque_col, "label": "Torque"})
        if pt_series:
            axes_rpm.append({"ylabel": "Potência/Torque", "series": pt_series})
        _render_multi(df, rpm_col, axes_rpm, "X = RPM", df_name=df_name)

    # X = Load (MAP ou FuelLoad)
    load_x = map_col or load_col
    if load_x:
        axes_load = []
        if ve_col:
            axes_load.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
        if lambda_col or afr_col:
            series = []
            if lambda_col:
                series.append({"col": lambda_col, "label": "Lambda"})
                if lambda_tgt: series.append({"col": lambda_tgt, "label": "Lambda Target"})
            else:
                series.append({"col": afr_col, "label": "AFR"})
                if afr_tgt: series.append({"col": afr_tgt, "label": "AFR Target"})
            axes_load.append({"ylabel": "Mistura", "series": series})
        if pw_cols:
            axes_load.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
        if adv_col:
            axes_load.append({"ylabel": "Advance (°)", "series": [{"col": adv_col, "label": "Advance"}]})
        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if duty_series:
            axes_load.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
        _render_multi(df, load_x, axes_load, "X = Load", df_name=df_name)

    # X = Vehicle Speed
    if spd_col:
        axes_spd = []
        if rpm_col:
            axes_spd.append({"ylabel": "RPM", "series": [{"col": rpm_col, "label": "RPM"}]})
        if map_col:
            axes_spd.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
        if lambda_col:
            axes_spd.append({"ylabel": "Lambda", "series": [{"col": lambda_col, "label": "Lambda"}]})
        elif afr_col:
            axes_spd.append({"ylabel": "AFR", "series": [{"col": afr_col, "label": "AFR"}]})
        ser = []
        if power_col: ser.append({"col": power_col, "label": "Power"})
        if torque_col: ser.append({"col": torque_col, "label": "Torque"})
        if ser:
            axes_spd.append({"ylabel": "Potência/Torque", "series": ser})
        _render_multi(df, spd_col, axes_spd, "X = Vehicle Speed", df_name=df_name)

    # Remanescentes vs Time
    if time_col:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        exclude = used_cols.union({time_col, rpm_col, map_col, load_col, spd_col})
        leftovers = [c for c in numeric_cols if c and c not in exclude and df[c].nunique(dropna=True) > 1]
        axes_left = [{"ylabel": col, "series": [{"col": col, "label": col}]} for col in leftovers]
        if axes_left:
            _render_multi(df, time_col, axes_left, "Variáveis remanescentes (X = Time)", df_name=df_name)


def plot_views_for_range(df: pd.DataFrame, t0: float, t1: float, df_name="Log"):
    """Plota gráficos relevantes no intervalo [t0, t1] (segundos da coluna t0)."""
    if "t0" not in df.columns:
        raise ValueError("Chame prepare_log(df) antes.")
    d = df[(df["t0"] >= t0) & (df["t0"] <= t1)].copy()
    if len(d) < 5:
        return
    rpm = _pick(d, "RPM"); mapc = _pick(d, "MAP")
    afr = _pick(d, "AFR"); afrt = _pick(d, "AFR Target")
    lam = _pick(d, "Lambda"); lamt = _pick(d, "Lambda Target")
    pw_cols = [c for c in ["PW","PW2","PW3","PW4"] if c in d.columns]
    duty = _pick(d, "Duty Cycle"); gammae = _pick(d, "Gammae")
    gwarm = _pick(d, "Gwarm"); gego = _pick(d, "Gego"); gbatt = _pick(d, "Gbattery")
    ve = _pick(d, "VE (Current","VE"); tps = _pick(d, "TPS"); iac = _pick(d, "IAC value")
    adv = _pick(d, "Advance (Current","Advance"); dwell = _pick(d, "Dwell")
    batt = _pick(d, "Battery V"); clt = _pick(d, "CLT"); iat = _pick(d, "IAT")
    boost = _pick(d, "Boost PSI"); spd = _pick(d, "Vehicle Speed","VSS")

    axes = []
    if rpm: axes.append({"ylabel":"RPM","series":[{"col":rpm}]})
    if mapc: axes.append({"ylabel":"MAP (kPa)","series":[{"col":mapc}]})
    if lam:
        ser=[{"col":lam,"label":"Lambda"}]
        if lamt: ser.append({"col":lamt,"label":"Lambda Target"})
        axes.append({"ylabel":"Lambda","series":ser})
    elif afr:
        ser=[{"col":afr,"label":"AFR"}]
        if afrt: ser.append({"col":afrt,"label":"AFR Target"})
        axes.append({"ylabel":"AFR","series":ser})
    if pw_cols: axes.append({"ylabel":"PW (ms)","series":[{"col":c} for c in pw_cols]})
    corr=[]
    if duty: corr.append({"col":duty,"label":"Duty (%)"})
    if gammae: corr.append({"col":gammae,"label":"GammaE (%)"})
    if gwarm: corr.append({"col":gwarm,"label":"Gwarm (%)"})
    if gego:  corr.append({"col":gego,"label":"Gego (%)"})
    if gbatt: corr.append({"col":gbatt,"label":"Gbattery (%)"})
    if corr: axes.append({"ylabel":"Correções/Duty (%)","series":corr})
    if ve: axes.append({"ylabel":"VE (Current)","series":[{"col":ve}]})
    ti=[]
    if tps: ti.append({"col":tps,"label":"TPS (%)"})
    if iac: ti.append({"col":iac,"label":"IAC"})
    if ti: axes.append({"ylabel":"TPS / IAC","series":ti})
    ign=[]
    if adv: ign.append({"col":adv,"label":"Advance (°)"})
    if dwell: ign.append({"col":dwell,"label":"Dwell (ms)"})
    if ign: axes.append({"ylabel":"Ignição","series":ign})
    temp=[]
    if clt: temp.append({"col":clt,"label":"CLT (°C)"})
    if iat: temp.append({"col":iat,"label":"IAT (°C)"})
    if temp: axes.append({"ylabel":"Temperaturas (°C)","series":temp})
    if batt: axes.append({"ylabel":"Battery (V)","series":[{"col":batt}]})
    misc=[]
    if _pick(d,"FuelLoad","Fuel Load"): misc.append({"col":_pick(d,"FuelLoad","Fuel Load"),"label":"FuelLoad"})
    if boost: misc.append({"col":boost,"label":"Boost (PSI)"})
    if spd:   misc.append({"col":spd,"label":"Vehicle Speed"})
    if misc: axes.append({"ylabel":"Carga/Velocidade","series":misc})

    _render_multi(d, "t0", axes, f"Time {t0:.1f}–{t1:.1f}s", df_name=df_name)


def plot_lambda_error_heatmap(df: pd.DataFrame, rpm_bins=None, map_bins=None, title="Lambda error (RPM × MAP)"):
    """Heatmap do erro mediano de Lambda/AFR por célula RPM×MAP."""
    d = df.dropna(subset=["RPM","MAP"]).copy()
    if "Lambda" in d.columns and "Lambda Target" in d.columns:
        d["err"] = d["Lambda"]/d["Lambda Target"] - 1.0
    elif "AFR" in d.columns and "AFR Target" in d.columns:
        d["err"] = d["AFR"]/d["AFR Target"] - 1.0
    else:
        raise ValueError("Precisa de (Lambda & Lambda Target) ou (AFR & AFR Target).")
    d = d[(d["err"] > -0.5) & (d["err"] < 0.5)]
    if rpm_bins is None:
        rpm_bins = np.arange(600, max(700, d["RPM"].max()+200), 200)
    if map_bins is None:
        map_bins = np.arange(20, max(30, d["MAP"].max()+10), 10)
    H = d.pivot_table(index=pd.cut(d["MAP"], map_bins),
                      columns=pd.cut(d["RPM"], rpm_bins),
                      values="err", aggfunc="median")
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    im = ax.imshow(H.values, origin="lower", aspect="auto",
                   cmap=plt.cm.plasma, vmin=-0.15, vmax=0.15)
    ax.set_xticks(np.arange(len(H.columns)))
    ax.set_xticklabels([f"{c.left:.0f}-{c.right:.0f}" for c in H.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(H.index)))
    ax.set_yticklabels([f"{r.left:.0f}-{r.right:.0f}" for r in H.index])
    ax.set_xlabel("RPM bins")
    ax.set_ylabel("MAP bins (kPa)")
    ax.set_title(title + " (mediana; + = magro, - = rico)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Erro (Lambda/Target - 1)")
    plt.tight_layout()
    plt.show()
    return fig, ax


# ============== Classe de alto nível e accessor do DataFrame ==============

class EngineLog:
    """
    Carrega um log a partir de caminho de arquivo (.msl/.csv/.tsv).
    Já prepara o DataFrame (prepare_log) e permite salvar.
    """
    def __init__(self, source: str, name: str = "Log", save_path: str | None = None, sep: str | None = None):
        self.name = name
        path = Path(source)
        # Usa a função open_log para abrir arquivos .msl/.csv e converter para numérico
        self.df = open_log(path)
        if name == "Log":
            self.name = path.stem
        self.df = prepare_log(self.df)
        if save_path:
            self.save(save_path)

    def save(self, path: str):
        self.df.to_csv(path, index=False)

    # Atalhos úteis
    def thin(self, every_s: float = 0.2):
        self.df = thin_by_seconds(self.df, every_s)
        return self.df

    def segments(self, min_duration_s: float = 3.0):
        return segment_by_state(self.df, min_duration_s=min_duration_s)
    
    def add_fuel_estimates(self, **kwargs):
        self.df = add_fuel_estimates(self.df, **kwargs)
        return self.df

    # Plots
    def plot_partida(self, **kwargs):
        return plot_partida(self.df, **kwargs)

    def plot_auto(self):
        return plot_auto_engine_views(self.df, df_name=self.name)

    def plot_range(self, t0: float, t1: float, df_name=None):
        return plot_views_for_range(self.df, t0, t1, df_name=(df_name or self.name))

    def plot_lambda_heatmap(self, **kwargs):
        return plot_lambda_error_heatmap(self.df, **kwargs)


# Pandas accessor para chamar como df.ecu.<método>
@pd.api.extensions.register_dataframe_accessor("ecu")
class ECUAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    # Preparação e utilidades
    def prepare(self):
        new_df = prepare_log(self._obj)
        # Atualiza/insere colunas (permite criar novas, como 't0' e 'state')
        for c in new_df.columns:
            self._obj[c] = new_df[c]
        return self._obj

    def thin(self, every_s: float = 0.2):
        return thin_by_seconds(self._obj, every_s)

    def segments(self, min_duration_s: float = 3.0):
        return segment_by_state(self._obj, min_duration_s)
    
    def add_fuel_estimates(self, **kwargs):
        new_df = add_fuel_estimates(self._obj, **kwargs)
        # Atualiza/insere colunas (permite criar novas colunas de combustível)
        for c in new_df.columns:
            self._obj[c] = new_df[c]
        return self._obj

    # Plots
    def plot_partida(self, **kwargs):
        return plot_partida(self._obj, **kwargs)

    def plot_auto(self, df_name: str = "Log"):
        return plot_auto_engine_views(self._obj, df_name=df_name)

    def plot_range(self, t0: float, t1: float, df_name="Log"):
        return plot_views_for_range(self._obj, t0, t1, df_name=df_name)

    def plot_lambda_heatmap(self, **kwargs):
        return plot_lambda_error_heatmap(self._obj, **kwargs)


# ===================== Uso esperado (no notebook) =====================
# from seu_modulo import EngineLog
# log = EngineLog("caminho_para_arquivo_log.msl", name="MeuLog")
# df = log.df
# df.ecu.add_fuel_estimates(inj_flow_cc_min=200, num_inj=4, n_cyl=4, fuel_density_g_cc=0.789, inj_rated_dp_kpa=300, reg_ref="manifold")
# df.ecu.plot_auto(df_name=log.name)
# df.ecu.plot_partida()
# segs = df.ecu.segments()
# if segs: df.ecu.plot_range(segs[0]["t0"], segs[0]["t1"], df_name=segs[0]["state"])
# df.ecu.plot_lambda_heatmap()