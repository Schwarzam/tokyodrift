import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from math import ceil

def plot_partida(
    df: pd.DataFrame,
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
    title='Partida – sinais relevantes (cranking sombreado)'
):
    """
    Plota sinais relevantes para a partida (cranking e pós-partida) com Time no eixo X.
    Retorna (fig, axes).
    """
    def has(col):
        return col in df.columns

    # Converte para numérico onde possível (sem quebrar strings)
    num_cols = [
        time_col, rpm_col, map_col, tps_col, afr_col, lambda_col, iat_col, clt_col,
        pw_col, duty_col, gammae_col, afr_target_col, lambda_target_col,
        iac_col, batt_col, advance_col
    ]
    for c in num_cols:
        if has(c):
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if not has(time_col):
        raise ValueError(f"Coluna de tempo '{time_col}' não encontrada no DataFrame.")

    # Eixo X começando em zero
    t = df[time_col] - df[time_col].min()

    # Máscara de cranking (ajuste o limiar conforme sua ECU)
    cranking = has(rpm_col) and (df[rpm_col] > 0) & (df[rpm_col] < crank_rpm_thr)

    plt.rcParams['figure.figsize'] = figsize
    fig, axes = plt.subplots(5, 1, sharex=True)

    # 1) RPM (esq) e MAP (dir)
    ax = axes[0]
    if has(rpm_col):
        ax.plot(t, df[rpm_col], label='RPM', color='tab:blue')
        ax.set_ylabel('RPM')
        ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if has(map_col):
        ax2.plot(t, df[map_col], label='MAP (kPa)', color='tab:orange', alpha=0.9)
        ax2.set_ylabel('MAP (kPa)')
        ax2.legend(loc='upper right')
    if has(rpm_col):
        ax.fill_between(t, 0, 1, where=cranking,
                        transform=ax.get_xaxis_transform(),
                        color='gray', alpha=0.15, label='Cranking')

    # 2) PW (ms) (esq) e Duty + Gammae (dir)
    ax = axes[1]
    if has(pw_col):
        ax.plot(t, df[pw_col], label='PW (ms)', color='tab:green')
        ax.set_ylabel('PW (ms)')
        ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    lines2, labels2 = [], []
    if has(duty_col):
        l1, = ax2.plot(t, df[duty_col], label='Duty Cycle (%)', color='tab:red', alpha=0.9)
        lines2.append(l1); labels2.append(l1.get_label())
    if has(gammae_col):
        l2, = ax2.plot(t, df[gammae_col], label='Gammae (%)', color='tab:purple', alpha=0.9)
        lines2.append(l2); labels2.append(l2.get_label())
    if lines2:
        ax2.set_ylabel('%')
        ax2.legend(lines2, labels2, loc='upper right')

    # 3) Mistura: Lambda + alvo (ou AFR + alvo)
    ax = axes[2]
    if has(lambda_col):
        ax.plot(t, df[lambda_col], label='Lambda', color='tab:blue')
        if has(lambda_target_col):
            ax.plot(t, df[lambda_target_col], label='Lambda Target', color='tab:orange', linestyle='--')
        ax.set_ylabel('Lambda')
        ax.legend(loc='upper right')
    elif has(afr_col):
        ax.plot(t, df[afr_col], label='AFR', color='tab:blue')
        if has(afr_target_col):
            ax.plot(t, df[afr_target_col], label='AFR Target', color='tab:orange', linestyle='--')
        ax.set_ylabel('AFR')
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4) Ar/controle e ignição: TPS + IAC (esq), Advance (dir)
    ax = axes[3]
    lines, labels = [], []
    if has(tps_col):
        l, = ax.plot(t, df[tps_col], label='TPS (%)', color='tab:gray')
        lines.append(l); labels.append(l.get_label())
    if has(iac_col):
        l, = ax.plot(t, df[iac_col], label='IAC value', color='tab:green')
        lines.append(l); labels.append(l.get_label())
    ax.set_ylabel('TPS / IAC')
    if lines:
        ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if has(advance_col):
        ax2.plot(t, df[advance_col], label='Advance (°)', color='tab:red')
        ax2.set_ylabel('Advance (°)')
        ax2.legend(loc='upper right')

    # 5) Temperaturas (esq) e Bateria (dir)
    ax = axes[4]
    lines, labels = [], []
    if has(clt_col):
        l, = ax.plot(t, df[clt_col], label='CLT (°C)', color='tab:blue')
        lines.append(l); labels.append(l.get_label())
    if has(iat_col):
        l, = ax.plot(t, df[iat_col], label='IAT (°C)', color='tab:orange')
        lines.append(l); labels.append(l.get_label())
    ax.set_ylabel('Temp (°C)')
    if lines:
        ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    if has(batt_col):
        ax2.plot(t, df[batt_col], label='Battery V', color='tab:purple')
        ax2.set_ylabel('Battery (V)')
        ax2.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.show()

    return fig, axes

def plot_auto_engine_views(df: pd.DataFrame, df_name: str = "Log"):
    """
    Seleciona automaticamente gráficos relevantes e gera figuras por eixo X:
      - Tempo (Time)
      - RPM
      - Carga do motor (MAP ou FuelLoad)
      - Velocidade do veículo (Vehicle Speed)
    Gráficos remanescentes (que não se encaixam bem) vão para uma figura extra vs Time.
    Paleta: Plasma entre 0.2 e 0.9. Se >5 gráficos, divide em 2 colunas.
    """

    # Helpers
    def has(col): return col in df.columns
    def pick(*cands):
        for c in cands:
            if has(c):
                return c
        return None

    # Converte colunas que serão usadas para numéricas
    def to_numeric(cols):
        for c in cols:
            if c and has(c):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Renderiza uma figura para um eixo X com vários "gráficos" (cada gráfico pode ter 1+ séries)
    def render_figure(xcol, axes_defs, title):
        if not axes_defs:
            return None
        n = len(axes_defs)
        if n <= 5:
            nrows, ncols = n, 1
        else:
            ncols = 2
            nrows = ceil(n / 2)

        # tamanho dinâmico
        h_per_row = 2.6
        fig_w = 14 if ncols == 1 else 16
        fig_h = max(7, h_per_row * nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(fig_w, fig_h))
        axs = np.array(axs).reshape(-1)  # flatten

        cmap = plt.cm.plasma
        xs = pd.to_numeric(df[xcol], errors="coerce")

        for i, ad in enumerate(axes_defs):
            ax = axs[i]
            series = ad["series"]
            # cores para as séries deste gráfico
            cols = [cmap(v) for v in np.linspace(0.2, 0.9, max(1, len(series)))]
            for (s, color) in zip(series, cols):
                ycol = s["col"]
                lbl = s.get("label", ycol)
                ax.plot(xs, df[ycol], label=lbl, color=color, linewidth=1.5)
            ax.set_ylabel(ad.get("ylabel", ""))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9, ncols=1)
            # Menos ticks sem quebrar ordem
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        # Remove eixos não usados
        for j in range(len(axs)):
            if j >= n:
                fig.delaxes(axs[j])

        axs[min(n-1, len(axs)-1)].set_xlabel(xcol)
        fig.suptitle(f"{title} – {df_name}", y=0.995, fontsize=12)
        fig.tight_layout()
        plt.show()
        return fig

    used_cols = set()

    # Colunas candidatas
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
    staging_dc = pick("Duty Cycle (Staging")
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

    # Garante tipos numéricos para o que vamos plotar
    num_candidates = [time_col, rpm_col, map_col, load_col, lambda_col, lambda_tgt, afr_col, afr_tgt,
                      duty_col, staging_dc, gammae_col, gwarm, gego, gbatt, ve_col, tps_col, iac_col,
                      adv_col, dwell_col, batt_v, clt_col, iat_col, boost_col, mapxrpm, ign_load,
                      fuel_load, power_col, torque_col] + pw_cols
    to_numeric([c for c in num_candidates if c])

    # 1) X = Time
    if time_col:
        axes_time = []

        # RPM
        if rpm_col:
            axes_time.append({"ylabel": "RPM", "series": [{"col": rpm_col, "label": "RPM"}]})
            used_cols.add(rpm_col)

        # MAP
        if map_col:
            axes_time.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
            used_cols.add(map_col)

        # Lambda/AFR + alvo
        if lambda_col:
            ser = [{"col": lambda_col, "label": "Lambda"}]
            if lambda_tgt:
                ser.append({"col": lambda_tgt, "label": "Lambda Target"})
                used_cols.add(lambda_tgt)
            axes_time.append({"ylabel": "Lambda", "series": ser})
            used_cols.add(lambda_col)
        elif afr_col:
            ser = [{"col": afr_col, "label": "AFR"}]
            if afr_tgt:
                ser.append({"col": afr_tgt, "label": "AFR Target"})
                used_cols.add(afr_tgt)
            axes_time.append({"ylabel": "AFR", "series": ser})
            used_cols.add(afr_col)

        # PW1..4
        if pw_cols:
            axes_time.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
            used_cols.update(pw_cols)

        # Duty + GammaE (+ correções principais)
        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if gwarm: duty_series.append({"col": gwarm, "label": "Gwarm (%)"})
        if gego: duty_series.append({"col": gego, "label": "Gego (%)"})
        if gbatt: duty_series.append({"col": gbatt, "label": "Gbattery (%)"})
        if duty_series:
            axes_time.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
            used_cols.update([s["col"] for s in duty_series])

        # VE
        if ve_col:
            axes_time.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
            used_cols.add(ve_col)

        # TPS + IAC
        tps_iac_series = []
        if tps_col: tps_iac_series.append({"col": tps_col, "label": "TPS (%)"})
        if iac_col: tps_iac_series.append({"col": iac_col, "label": "IAC"})
        if tps_iac_series:
            axes_time.append({"ylabel": "TPS / IAC", "series": tps_iac_series})
            used_cols.update([s["col"] for s in tps_iac_series])

        # Ignição
        if adv_col or dwell_col:
            ign_series = []
            if adv_col: ign_series.append({"col": adv_col, "label": "Advance (°)"})
            if dwell_col: ign_series.append({"col": dwell_col, "label": "Dwell (ms)"})
            axes_time.append({"ylabel": "Ignição", "series": ign_series})
            used_cols.update([s["col"] for s in ign_series])

        # Temperaturas + Bateria
        temp_series = []
        if clt_col: temp_series.append({"col": clt_col, "label": "CLT (°C)"})
        if iat_col: temp_series.append({"col": iat_col, "label": "IAT (°C)"})
        if temp_series:
            axes_time.append({"ylabel": "Temperatura (°C)", "series": temp_series})
            used_cols.update([s["col"] for s in temp_series])
        if batt_v:
            axes_time.append({"ylabel": "Battery (V)", "series": [{"col": batt_v, "label": "Battery V"}]})
            used_cols.add(batt_v)

        # Carga e afins
        fl_series = []
        if fuel_load: fl_series.append({"col": fuel_load, "label": "FuelLoad"})
        if ign_load: fl_series.append({"col": ign_load, "label": "IgnitionLoad"})
        if mapxrpm: fl_series.append({"col": mapxrpm, "label": "MAP×RPM"})
        if fl_series:
            axes_time.append({"ylabel": "Carga/Produtos", "series": fl_series})
            used_cols.update([s["col"] for s in fl_series])

        # Boost
        if boost_col:
            axes_time.append({"ylabel": "Boost (PSI)", "series": [{"col": boost_col, "label": "Boost PSI"}]})
            used_cols.add(boost_col)

        # Velocidade / Potência / Torque
        if spd_col:
            axes_time.append({"ylabel": "Velocidade (u)", "series": [{"col": spd_col, "label": "Vehicle Speed"}]})
            used_cols.add(spd_col)
        pt_series = []
        if power_col: pt_series.append({"col": power_col, "label": "Power"})
        if torque_col: pt_series.append({"col": torque_col, "label": "Torque"})
        if pt_series:
            axes_time.append({"ylabel": "Potência/Torque", "series": pt_series})
            used_cols.update([s["col"] for s in pt_series])

        render_figure(time_col, axes_time, "X = Time")

    # 2) X = RPM
    if rpm_col:
        axes_rpm = []
        # MAP vs RPM
        if map_col:
            axes_rpm.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
        # Lambda/AFR vs RPM
        if lambda_col or afr_col:
            series = []
            if lambda_col:
                series.append({"col": lambda_col, "label": "Lambda"})
                if lambda_tgt: series.append({"col": lambda_tgt, "label": "Lambda Target"})
            else:
                series.append({"col": afr_col, "label": "AFR"})
                if afr_tgt: series.append({"col": afr_tgt, "label": "AFR Target"})
            axes_rpm.append({"ylabel": "Mistura", "series": series})
        # PW vs RPM
        if pw_cols:
            axes_rpm.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
        # Duty/GammaE vs RPM
        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if duty_series:
            axes_rpm.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
        # Advance vs RPM
        if adv_col:
            axes_rpm.append({"ylabel": "Advance (°)", "series": [{"col": adv_col, "label": "Advance"}]})
        # VE vs RPM
        if ve_col:
            axes_rpm.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
        # Power/Torque vs RPM
        pt_series = []
        if power_col: pt_series.append({"col": power_col, "label": "Power"})
        if torque_col: pt_series.append({"col": torque_col, "label": "Torque"})
        if pt_series:
            axes_rpm.append({"ylabel": "Potência/Torque", "series": pt_series})
        render_figure(rpm_col, axes_rpm, "X = RPM")

    # 3) X = Load (MAP ou FuelLoad)
    load_x = map_col or fuel_load
    if load_x:
        axes_load = []
        # VE vs Load
        if ve_col:
            axes_load.append({"ylabel": "VE (Current)", "series": [{"col": ve_col, "label": ve_col}]})
        # Mistura vs Load
        if lambda_col or afr_col:
            series = []
            if lambda_col:
                series.append({"col": lambda_col, "label": "Lambda"})
                if lambda_tgt: series.append({"col": lambda_tgt, "label": "Lambda Target"})
            else:
                series.append({"col": afr_col, "label": "AFR"})
                if afr_tgt: series.append({"col": afr_tgt, "label": "AFR Target"})
            axes_load.append({"ylabel": "Mistura", "series": series})
        # PW vs Load
        if pw_cols:
            axes_load.append({"ylabel": "PW (ms)", "series": [{"col": c, "label": c} for c in pw_cols]})
        # Advance vs Load
        if adv_col:
            axes_load.append({"ylabel": "Advance (°)", "series": [{"col": adv_col, "label": "Advance"}]})
        # Duty/GammaE vs Load
        duty_series = []
        if duty_col: duty_series.append({"col": duty_col, "label": "Duty (%)"})
        if gammae_col: duty_series.append({"col": gammae_col, "label": "GammaE (%)"})
        if duty_series:
            axes_load.append({"ylabel": "Correções/Duty (%)", "series": duty_series})
        render_figure(load_x, axes_load, "X = Load")

    # 4) X = Vehicle Speed (se existir)
    if spd_col:
        axes_spd = []
        if rpm_col:
            axes_spd.append({"ylabel": "RPM", "series": [{"col": rpm_col, "label": "RPM"}]})
        if map_col:
            axes_spd.append({"ylabel": "MAP (kPa)", "series": [{"col": map_col, "label": "MAP"}]})
        if lambda_col or afr_col:
            if lambda_col:
                axes_spd.append({"ylabel": "Lambda", "series": [{"col": lambda_col, "label": "Lambda"}]})
            else:
                axes_spd.append({"ylabel": "AFR", "series": [{"col": afr_col, "label": "AFR"}]})
        if power_col or torque_col:
            ser = []
            if power_col: ser.append({"col": power_col, "label": "Power"})
            if torque_col: ser.append({"col": torque_col, "label": "Torque"})
            axes_spd.append({"ylabel": "Potência/Torque", "series": ser})
        render_figure(spd_col, axes_spd, "X = Vehicle Speed")

    # 5) Remanescentes vs Time
    if time_col:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # já usados + eixos X principais
        exclude = used_cols.union({time_col, rpm_col, map_col, load_col, spd_col})
        leftovers = [c for c in numeric_cols if c and c not in exclude]
        # tira também colunas óbvias que são índices/constantes
        leftovers = [c for c in leftovers if df[c].nunique(dropna=True) > 1]
        axes_left = [{"ylabel": col, "series": [{"col": col, "label": col}]} for col in leftovers]
        if axes_left:
            render_figure(time_col, axes_left, "Variáveis remanescentes (X = Time)")