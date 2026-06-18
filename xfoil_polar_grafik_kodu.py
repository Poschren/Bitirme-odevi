import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1) XFOIL POLAR DOSYASI OKUMA FONKSIYONU
# ============================================================

def read_xfoil_polar(filename):
    """
    XFOIL polar dosyasını okur.
    Beklenen sütunlar:
    alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
    """

    rows = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            # XFOIL veri satırları genelde 7 sayısal değerden olusur:
            # alpha CL CD CDp CM Top_Xtr Bot_Xtr
            if len(parts) >= 7:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    cdp = float(parts[3])
                    cm = float(parts[4])
                    top_xtr = float(parts[5])
                    bot_xtr = float(parts[6])

                    rows.append([alpha, cl, cd, cdp, cm, top_xtr, bot_xtr])

                except ValueError:
                    # Başlık veya metin satırlarını atla
                    pass

    df = pd.DataFrame(
        rows,
        columns=["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    )

    if df.empty:
        return df

    # CL/CD hesabı
    df["CL_CD"] = df["CL"] / df["CD"]

    return df


# ============================================================
# 2) DOSYALARI TANIMLA
# ============================================================

polar_files = {
    "Constrained PARSEC": "polar_constrained_range_new.txt",
    "NACA 0012": "polar_naca0012_range.txt",
    "NACA 2412": "polar_naca2412_range.txt",
}


# ============================================================
# 3) DOSYALARI OKU
# ============================================================

data = {}

for name, filename in polar_files.items():
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"{filename} bulunamadi. Dosyanin Python dosyasi ile ayni klasorde oldugundan emin ol."
        )

    df = read_xfoil_polar(filename)

    if df.empty:
        raise ValueError(
            f"{filename} okunamadi veya icinde gecerli XFOIL polar verisi yok."
        )

    data[name] = df

    print("\n" + name)
    print("-" * 50)
    print(df.head())
    print("Satir sayisi:", len(df))
    print("Alpha araligi:", df["alpha"].min(), "to", df["alpha"].max())


# ============================================================
# 4) GRAFİK AYARLARI
# ============================================================

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def plot_comparison(y_column, y_label, title, output_name):
    """
    Verilen aerodinamik katsayıyı alpha'ya göre çizer.
    """

    plt.figure(figsize=(7.5, 5))

    for name, df in data.items():
        plt.plot(
            df["alpha"],
            df[y_column],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=name
        )

    plt.xlabel("Angle of Attack, α [deg]")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches="tight")
    plt.show()

    print(f"Kaydedildi: {output_name}")


# ============================================================
# 5) GRAFİKLERİ ÇİZ VE KAYDET
# ============================================================

plot_comparison(
    y_column="CL",
    y_label="$C_L$",
    title="$C_L$ vs Angle of Attack",
    output_name="CL_vs_alpha.png"
)

plot_comparison(
    y_column="CD",
    y_label="$C_D$",
    title="$C_D$ vs Angle of Attack",
    output_name="CD_vs_alpha.png"
)

plot_comparison(
    y_column="CL_CD",
    y_label="$C_L/C_D$",
    title="$C_L/C_D$ vs Angle of Attack",
    output_name="CL_CD_vs_alpha.png"
)

plot_comparison(
    y_column="CM",
    y_label="$C_M$",
    title="$C_M$ vs Angle of Attack",
    output_name="CM_vs_alpha.png"
)


# ============================================================
# 6) ALPHA = 5 DERECE KARŞILAŞTIRMA TABLOSU
# ============================================================

comparison_rows = []

target_alpha = 5.0

for name, df in data.items():
    # Alpha = 5 dereceye en yakin satiri bul
    row = df.iloc[(df["alpha"] - target_alpha).abs().argsort()[:1]].iloc[0]

    comparison_rows.append({
        "Airfoil": name,
        "alpha [deg]": row["alpha"],
        "CL": row["CL"],
        "CD": row["CD"],
        "CM": row["CM"],
        "CL/CD": row["CL_CD"],
        "Top_Xtr": row["Top_Xtr"],
        "Bot_Xtr": row["Bot_Xtr"],
    })

comparison_df = pd.DataFrame(comparison_rows)

print("\nAlpha = 5 derece karsilastirma tablosu")
print("=" * 80)
print(comparison_df)

comparison_df.to_csv("alpha_5_comparison.csv", index=False)

print("\nKaydedildi: alpha_5_comparison.csv")


# ============================================================
# 7) ALPHA = 5 İÇİN KISA YORUM
# ============================================================

print("\nAlpha = 5 derece icin ozet:")
print("-" * 80)

for _, row in comparison_df.iterrows():
    print(
        f"{row['Airfoil']}: "
        f"CL = {row['CL']:.4f}, "
        f"CD = {row['CD']:.5f}, "
        f"CM = {row['CM']:.4f}, "
        f"CL/CD = {row['CL/CD']:.2f}"
    )