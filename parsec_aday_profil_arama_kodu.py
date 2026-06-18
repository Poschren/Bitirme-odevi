"""
parsec_conservative_search.py

XFOIL kararlı airfoil profilleri için kısıtlı PARSEC rastgele araması
Amaç:
    Amaç fonksiyonu hâlâ CL/CD maksimumdur.
    Ancak seçim yalnızca geometrik ve aerodinamik olarak makul adaylar arasından yapılır.

    Yani:
        maximize CL/CD

    Şu kısıtlar altında:
        - geometri XFOIL ile uyumlu olmalı
        - kalınlık kabul edilebilir aralıkta olmalı
        - camber (eğrilik) sınırlı olmalı
        - CD (sürükleme katsayısı) gerçekçi olmalı
        - CM aşırı derecede negatif olmamalı
        - aday profil, önceki şüpheli profillerden daha az agresif olmalı

Gereksinimler:
    pip install aerosandbox numpy pandas matplotlib
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import aerosandbox as asb
except ImportError as exc:
    raise ImportError(
        "AeroSandbox is required. Install with: pip install aerosandbox"
    ) from exc


# ============================================================
# KULLANICI AYARLARI
# ============================================================

N_CANDIDATES = 10000
ALPHA_DEG = 5.0
RE = 1_000_000
MACH = 0.0
N_POINTS_PER_SIDE = 160
RANDOM_SEED = 42

OUTPUT_DIR = Path("parsec_conservative_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# None bırak. Hızlı test için örn. 200 yapılabilir.
MAX_NEURALFOIL_EVALS: Optional[int] = None


# ============================================================
# PARSEC MODEL
# ============================================================

@dataclass
class ParsecParams:
    r_le_u: float
    r_le_l: float

    x_u: float
    z_u: float
    z_xx_u: float

    x_l: float
    z_l: float
    z_xx_l: float

    z_te_mid: float
    t_te: float

    alpha_te: float
    beta_te: float


def _poly_terms(x: float) -> np.ndarray:
    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    return x ** powers


def _poly_derivative_terms(x: float) -> np.ndarray:
    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    return powers * x ** (powers - 1.0)


def _poly_second_derivative_terms(x: float) -> np.ndarray:
    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    return powers * (powers - 1.0) * x ** (powers - 2.0)


def _solve_surface_coefficients(
    r_le: float,
    x_crest: float,
    z_crest: float,
    z_xx_crest: float,
    z_te: float,
    dzdx_te: float,
    is_upper: bool,
) -> np.ndarray:
    """
    PARSEC yüzey katsayılarını çözer.

    z(x) = a0*x^(1/2) + a1*x^(3/2) + ... + a5*x^(11/2)
    """
    a0 = math.sqrt(2.0 * r_le)

    if not is_upper:
        a0 *= -1.0

    A = []
    b = []

    # Firar kenarı konumu
    A.append(_poly_terms(1.0)[1:])
    b.append(z_te - a0 * _poly_terms(1.0)[0])

    # Firar kenarı eğimi
    A.append(_poly_derivative_terms(1.0)[1:])
    b.append(dzdx_te - a0 * _poly_derivative_terms(1.0)[0])

    # Tepe noktası konumu
    A.append(_poly_terms(x_crest)[1:])
    b.append(z_crest - a0 * _poly_terms(x_crest)[0])

    # Tepe noktası eğimi = 0
    A.append(_poly_derivative_terms(x_crest)[1:])
    b.append(0.0 - a0 * _poly_derivative_terms(x_crest)[0])

    # Tepe noktası eğriliği
    A.append(_poly_second_derivative_terms(x_crest)[1:])
    b.append(z_xx_crest - a0 * _poly_second_derivative_terms(x_crest)[0])

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    try:
        rest = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("PARSEC coefficient system is singular.") from exc

    return np.concatenate([[a0], rest])


def parsec_to_coordinates(
    p: ParsecParams,
    n_points_per_side: int = N_POINTS_PER_SIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    XFOIL formatında koordinat döndürür:
    üst yüzey TE -> LE, sonra alt yüzey LE -> TE
    """
    z_te_upper = p.z_te_mid + 0.5 * p.t_te
    z_te_lower = p.z_te_mid - 0.5 * p.t_te

    alpha = math.radians(p.alpha_te)
    beta = math.radians(p.beta_te)

    dzdx_te_upper = math.tan(alpha - 0.5 * beta)
    dzdx_te_lower = math.tan(alpha + 0.5 * beta)

    a_u = _solve_surface_coefficients(
        r_le=p.r_le_u,
        x_crest=p.x_u,
        z_crest=p.z_u,
        z_xx_crest=p.z_xx_u,
        z_te=z_te_upper,
        dzdx_te=dzdx_te_upper,
        is_upper=True,
    )

    a_l = _solve_surface_coefficients(
        r_le=p.r_le_l,
        x_crest=p.x_l,
        z_crest=p.z_l,
        z_xx_crest=p.z_xx_l,
        z_te=z_te_lower,
        dzdx_te=dzdx_te_lower,
        is_upper=False,
    )

    theta = np.linspace(0.0, math.pi, n_points_per_side)
    x = 0.5 * (1.0 - np.cos(theta))

    # x = 0 türevlerde singular davranabileceği için çok küçük pozitif değer veriyoruz.
    x[0] = 1e-6
    x[-1] = 1.0

    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    X = x[:, None] ** powers[None, :]

    y_u = X @ a_u
    y_l = X @ a_l

    x_upper = x[::-1]
    y_upper = y_u[::-1]

    x_lower = x[1:]
    y_lower = y_l[1:]

    coords = np.column_stack(
        [
            np.concatenate([x_upper, x_lower]),
            np.concatenate([y_upper, y_lower]),
        ]
    )

    return coords, x, y_u, y_l


# ============================================================
# Temkinli Örnekleme ve Geometri Kısıtları
# ============================================================

def sample_conservative_params(rng: np.random.Generator) -> ParsecParams:
    """
    Daha konservatif PARSEC aralıkları.

    Burada amaç:
        - çok keskin leading edge üretmemek
        - aşırı kamburluk üretmemek
        - çok agresif eğrilik üretmemek
        - XFOIL'in seveceği daha sakin profiller üretmek
    """
    return ParsecParams(
        r_le_u=float(rng.uniform(0.0140, 0.0220)),
        r_le_l=float(rng.uniform(0.0140, 0.0230)),

        x_u=float(rng.uniform(0.32, 0.45)),
        z_u=float(rng.uniform(0.060, 0.075)),
        z_xx_u=float(rng.uniform(-0.30, -0.16)),

        x_l=float(rng.uniform(0.32, 0.47)),
        z_l=float(rng.uniform(-0.032, -0.022)),
        z_xx_l=float(rng.uniform(0.14, 0.26)),

        z_te_mid=float(rng.uniform(-0.0010, 0.0020)),
        t_te=float(rng.uniform(0.0020, 0.0050)),

        alpha_te=float(rng.uniform(-1.5, 1.0)),
        beta_te=float(rng.uniform(10.0, 13.5)),
    )


def geometry_metrics(x: np.ndarray, y_u: np.ndarray, y_l: np.ndarray) -> Dict[str, float]:
    thickness = y_u - y_l
    camber = 0.5 * (y_u + y_l)

    t_max_index = int(np.argmax(thickness))

    return {
        "t_max": float(np.max(thickness)),
        "x_t_max": float(x[t_max_index]),
        "t_min": float(np.min(thickness)),
        "camber_max_abs": float(np.max(np.abs(camber))),
        "upper_min": float(np.min(y_u)),
        "lower_max": float(np.max(y_l)),
        "leading_edge_thickness": float(thickness[0]),
        "trailing_edge_thickness": float(thickness[-1]),
    }


def is_xfoil_friendly_geometry(
    coords: np.ndarray,
    x: np.ndarray,
    y_u: np.ndarray,
    y_l: np.ndarray,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    XFOIL'de TRCHEK2 / NaN / convergence hatası çıkarabilecek geometrileri eler.
    """
    m = geometry_metrics(x, y_u, y_l)

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    if not np.all(np.isfinite(coords)):
        return False, "non_finite_coordinates", m

    if np.any(y_u <= y_l):
        return False, "surface_crossing", m

    # Daha gerçekçi kalınlık aralığı
    if m["t_max"] < 0.105:
        return False, "too_thin", m

    if m["t_max"] > 0.130:
        return False, "too_thick", m

    # Maksimum kalınlığın chord üzerindeki yeri çok öne/arkaya kaçmasın
    if m["x_t_max"] < 0.25:
        return False, "tmax_too_forward", m

    if m["x_t_max"] > 0.55:
        return False, "tmax_too_aft", m

    # Kamburluk kontrollü olsun
    if m["camber_max_abs"] > 0.035:
        return False, "too_much_camber", m

    # Üst yüzey gereksiz yere aşağı düşmesin
    if m["upper_min"] < -0.004:
        return False, "upper_surface_dips_below_zero_too_much", m

    # Alt yüzey yukarıda garip çıkıntı yapmasın
    if m["lower_max"] > 0.004:
        return False, "lower_surface_positive_bump", m

    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    if np.max(segment_lengths) > 0.08:
        return False, "large_panel_jump", m

    if np.any(segment_lengths < 1e-7):
        return False, "duplicate_points", m

    return True, "ok", m


# ============================================================
# NEURALFOIL DEĞERLENDİRMESİ
# ============================================================

def evaluate_with_neuralfoil(coords: np.ndarray) -> Dict[str, float]:
    """
    Aday profili AeroSandbox/NeuralFoil ile değerlendirir.
    """
    airfoil = asb.Airfoil(name="parsec_candidate", coordinates=coords)

    try:
        aero = airfoil.get_aero_from_neuralfoil(
            alpha=ALPHA_DEG,
            Re=RE,
            mach=MACH,
        )
    except Exception:
        return {
            "CL": np.nan,
            "CD": np.nan,
            "CM": np.nan,
            "CL_CD": np.nan,
        }

    CL = float(np.asarray(aero["CL"]).reshape(-1)[0])
    CD = float(np.asarray(aero["CD"]).reshape(-1)[0])
    CM = float(np.asarray(aero["CM"]).reshape(-1)[0]) if "CM" in aero else np.nan

    if not np.isfinite(CL) or not np.isfinite(CD) or CD <= 0:
        ld = np.nan
    else:
        ld = CL / CD

    return {
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "CL_CD": float(ld),
    }


# ============================================================
# DIŞA AKTARMA VE GRAFİK ÇİZİMİ
# ============================================================

def export_dat(
    coords: np.ndarray,
    path: Path,
    name: str = "PARSEC constrained conservative candidate",
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(name + "\n")
        for xi, yi in coords:
            f.write(f"{xi:.8f} {yi:.8f}\n")


def plot_airfoil_geometry(coords: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(coords[:, 0], coords[:, 1], "-")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x/c")
    plt.ylabel("z/c")
    plt.title("Best constrained conservative PARSEC airfoil geometry")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# ANA ARAMA
# ============================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    rows: List[Dict[str, float | int | str]] = []

    accepted = 0
    evaluated = 0
    constrained_accepted = 0

    best_row: Optional[Dict[str, float | int | str]] = None
    best_coords: Optional[np.ndarray] = None

    for i in range(N_CANDIDATES):
        p = sample_conservative_params(rng)

        try:
            coords, x, y_u, y_l = parsec_to_coordinates(p)
        except Exception as exc:
            rows.append(
                {
                    "idx": i,
                    "status": "parsec_fail",
                    "reason": str(exc),
                    **asdict(p),
                }
            )
            continue

        ok, reason, metrics = is_xfoil_friendly_geometry(coords, x, y_u, y_l)

        if not ok:
            rows.append(
                {
                    "idx": i,
                    "status": "geometry_rejected",
                    "reason": reason,
                    **metrics,
                    **asdict(p),
                }
            )
            continue

        accepted += 1

        if MAX_NEURALFOIL_EVALS is not None and evaluated >= MAX_NEURALFOIL_EVALS:
            rows.append(
                {
                    "idx": i,
                    "status": "accepted_not_evaluated",
                    "reason": "MAX_NEURALFOIL_EVALS reached",
                    **metrics,
                    **asdict(p),
                }
            )
            continue

        aero = evaluate_with_neuralfoil(coords)
        evaluated += 1

        status = "evaluated"

        if not np.isfinite(aero["CL_CD"]):
            status = "neuralfoil_fail"

        row = {
            "idx": i,
            "status": status,
            "reason": reason,
            **aero,
            **metrics,
            **asdict(p),
        }

        if status == "evaluated":
            ld = float(row["CL_CD"])
            cl = float(row["CL"])
            cd = float(row["CD"])
            cm = float(row["CM"])
            tmax = float(row["t_max"])
            camber = float(row["camber_max_abs"])

            # ============================================================
            # KISITLI OPTİMİZASYON
            # ============================================================
            # Amaç fonksiyonu:
            #     maximize CL/CD
            #
            # Ancak yalnızca aşağıdaki kısıtları sağlayan adaylar kabul edilir.
            # Böylece yüksek CL/CD veren ama geometrik/aerodinamik olarak şüpheli
            # profiller elenir.
            # ============================================================

            is_constrained_candidate = (
                0.60 <= cl <= 0.88
                and 0.0080 <= cd <= 0.0135
                and -0.080 <= cm <= -0.020
                and 0.105 <= tmax <= 0.130
                and 0.010 <= camber <= 0.032
                and ld <= 100.0
            )

            if is_constrained_candidate:
                constrained_accepted += 1
                row["constraint_status"] = "accepted"

                # Burada hâlâ amaç fonksiyonu CL/CD maksimumdur.
                if best_row is None or ld > float(best_row["CL_CD"]):
                    best_row = row
                    best_coords = coords.copy()

                    print(
                        f"New constrained best idx={i:04d} | "
                        f"CL/CD={ld:.2f} | "
                        f"CL={cl:.4f} CD={cd:.5f} CM={cm:.4f} | "
                        f"t_max={tmax:.4f} camber={camber:.4f}"
                    )
            else:
                row["constraint_status"] = "rejected"

        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "conservative_search_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n================ SUMMARY ================")
    print(f"Total candidates:          {N_CANDIDATES}")
    print(f"Geometry accepted:         {accepted}")
    print(f"NeuralFoil evaluated:      {evaluated}")
    print(f"Constrained accepted:      {constrained_accepted}")
    print(f"Results CSV:               {csv_path}")

    if best_row is None or best_coords is None:
        print(
            "\nNo valid constrained candidate found.\n"
            "Try relaxing one of these constraints:\n"
            "  - tmax lower bound: 0.110 -> 0.105\n"
            "  - CL upper bound: 0.85 -> 0.88\n"
            "  - camber upper bound: 0.028 -> 0.032\n"
        )
        return

    best_dat_path = OUTPUT_DIR / "best_parsec_conservative.dat"
    best_json_path = OUTPUT_DIR / "best_parsec_conservative_params.json"
    best_png_path = OUTPUT_DIR / "best_parsec_conservative_geometry.png"

    export_dat(best_coords, best_dat_path, name="best_parsec_conservative")

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    plot_airfoil_geometry(best_coords, best_png_path)

    print("\n================ BEST CANDIDATE ================")
    print(f"idx      : {best_row['idx']}")
    print(f"CL       : {best_row['CL']:.6f}")
    print(f"CD       : {best_row['CD']:.6f}")
    print(f"CM       : {best_row['CM']:.6f}")
    print(f"CL/CD    : {best_row['CL_CD']:.3f}")
    print(f"t_max    : {best_row['t_max']:.6f}")
    print(f"x_t_max  : {best_row['x_t_max']:.6f}")
    print(f"camber   : {best_row['camber_max_abs']:.6f}")
    print(f"DAT file : {best_dat_path}")
    print(f"Params   : {best_json_path}")
    print(f"Geometry : {best_png_path}")

    print("\nNext XFOIL manual test:")
    print("  copy parsec_conservative_results\\best_parsec_conservative.dat best_parsec_conservative.dat")
    print("  xfoil")
    print("  LOAD best_parsec_conservative.dat")
    print("  PANE")
    print("  OPER")
    print(f"  VISC {RE}")
    print("  ITER 300")
    print("  PACC")
    print("  polar_constrained_alpha5.txt")
    print("  ")
    print(f"  ALFA {ALPHA_DEG}")
    print("  PACC")
    print("  QUIT")


if __name__ == "__main__":
    main()