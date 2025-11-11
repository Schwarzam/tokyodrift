import pandas as pd
from pathlib import Path

def open_log(path) -> pd.DataFrame:
    # 1) Descobrir a linha do cabeçalho e o separador
    if isinstance(path, str):
        path = Path(path)
    
    header_idx = None
    sep = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("Time"):          # início da tabela
                header_idx = i
                if "\t" in line:
                    sep = "\t"
                elif "," in line:
                    sep = ","
                else:
                    sep = r"\s+"              # fallback: espaços
                break

    if header_idx is None:
        raise RuntimeError("Não encontrei a linha de cabeçalho (que começa com 'Time').")

    # 2) Ler a partir do cabeçalho
    df = pd.read_csv(
        path,
        sep=sep,
        skiprows=header_idx,   # pula tudo antes de "Time"
        engine="python",
        on_bad_lines="skip",   # pula linhas quebradas (ex.: cabeçalhos repetidos no meio do log)
    )

    # 3) Tentar converter colunas numéricas automaticamente
    for c in df.columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df