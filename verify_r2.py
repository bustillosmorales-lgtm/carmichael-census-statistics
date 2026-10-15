"""Second revision: compare the numbers quoted in the revised manuscript with the outputs in results/.

The table cells are copied from the manuscript source by tools/extract_quoted_r2.py into
verify_r2_quoted.json. Each quoted cell is compared with the corresponding output rounded to the
number of decimals printed. Called from verify_all.py."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
J = lambda name: json.load(open(os.path.join(R, name)))


def run(check):
    Q = json.load(open(os.path.join(HERE, "verify_r2_quoted.json"), encoding="utf-8"))["tables"]

    def num(s):
        return float(s.replace(",", "").replace("−", "-"))

    def same(label, q, v):
        """q is the quoted string, v the computed value; equal after rounding to the decimals of q."""
        q = q.strip().replace(",", "")
        d = len(q.split(".")[1]) if "." in q else 0
        check(label, q, f"{v:.{d}f}" if isinstance(v, float) else str(v))

    def interval(label, cell, level, ci):
        m = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", cell)
        same(label + " rate", m.group(1), float(level))
        same(label + " lower", m.group(2), float(ci[0]))
        same(label + " upper", m.group(3), float(ci[1]))

    # ---------------------------------------------------------------- Table 2, actual level
    v = J("12_validation.json")["partD"]
    names = ["D_100", "D_1000", "D_5000", "D_20000", "D_ratio", "m_ratio", "rho_1", "rho_2"]
    for row, name in zip(Q["tab:level"], names):
        for cell, gen in zip(row[1:], ("inverse", "nested", "rank")):
            interval(f"T2 level {name} {gen}", cell, v[name][gen]["level"], v[name][gen]["level_ci"])
    ve = J("12_validation.json")["partE"]
    check("T2 D_cf row present", True, "D_{cf}" in Q["tab:level"][8][0])
    check("T2 D_cf part E realizations and labelings", (1000, 40), (ve["R"], ve["K"]))
    for cell, gen in zip(Q["tab:level"][8][1:], ("inverse", "nested", "rank")):
        interval(f"T2 level D_cf {gen}", cell, ve["D_cf"][gen]["level"], ve["D_cf"][gen]["level_ci"])
    lv = [ve["D_cf"][g]["level"] for g in ("inverse", "nested", "rank")]
    same("2.3 D_cf level lowest (%)", "26", 100 * min(lv)); same("2.3 D_cf level highest (%)", "31", 100 * max(lv))
    same("2.3 D_cf null mean under the reference model", "0.00118", ve["D_cf"]["inverse"]["null_mean"])
    same("2.3 D_cf mean under the true process", "0.00127", ve["D_cf"]["inverse"]["true_mean"])

    # ---------------------------------------------------------------- Table 3, resolution rule
    rr = J("13_resolution_rule.json")
    S = None
    for row in Q["tab:rule"]:
        S = row[0].replace(",", "") or S
        truth = "pchip" if row[1] == "C_0" else "poly"
        for cell, W in zip(row[2:], ("100", "1000", "5000", "20000")):
            lv = rr["levels"][f"{truth}|S={S}|W={W}"]
            interval(f"T3 {truth} S={S} W={W}", cell.rstrip("*"), lv["level"], lv["ci"])
            check(f"T3 asterisk {truth} S={S} W={W}", cell.endswith("*"), rr["rule"][W] == int(S))

    # ---------------------------------------------------------------- Table 4, multiscale columns
    ms = J("20_secondary_inversion.json")
    for row in Q["tab:windows"]:
        W = row[0].replace(",", "")
        if W == "20000":
            same("T4 W=20000 descriptive D_W", row[6], ms["multiscale_W20000_descriptive"]["obs"])
            check("T4 W=20000 not tested", row[7:], ["--"] * 4)
            check("T4 rule gives no S at W=20000", True, rr["rule"]["20000"] is None)
            continue
        m = ms["multiscale"][W]
        same(f"T4 W={W} D_W", row[6], m["obs"])
        same(f"T4 W={W} S", row[7], m["S"])
        same(f"T4 W={W} q95", row[8], m["q95"])
        same(f"T4 W={W} p", row[9], m["p"])
        same(f"T4 W={W} Holm", row[10], m["p_holm"])
        check(f"T4 W={W} R", 2000, m["R"])

    # ---------------------------------------------------------------- Table 6, primary family of four, D_cf descriptive
    pf = J("11b_primary_family.json")
    p4 = J("11c_primary_family_four.json")
    for row, name in zip(Q["tab:tests"], ("D_1000", "D_ratio", "m_ratio", "Q_10", "D_cf")):
        f = pf["family"][name]
        for cell, key in zip(row[2:6], ("obs", "null_mean", "q025", "q975")):
            same(f"T6 {name} {key}", cell, f[key])
        if name == "D_cf":
            check("T6 D_cf descriptive, no p and no Holm", ["--", "--"], row[6:8])
        else:
            g = p4["family"][name]
            p, se = re.match(r"([\d.]+) \(([\d.]+)\)", row[6]).groups()
            same(f"T6 {name} p", p, g["p"]); same(f"T6 {name} se", se, g["p_se"])
            same(f"T6 {name} Holm over four", row[7], g["p_holm"])
        same(f"T6 {name} lattice", row[8], ms["lattice"][name]["value"])
        check(f"T6 {name} lattice at smallest p 1/2001", True, abs(ms["lattice"][name]["p"] - 1 / 2001) < 1e-12)
    check("T6 R", 2000, pf["R"]); check("T6 labelings of D_cf", 40, pf["K"])
    same("T6 Westfall-Young global p over four", "0.753", p4["global_minp"]["p_global"])

    # ---------------------------------------------------------------- Table 5, interval counts by inversion
    ic = J("05b_interval_counts_inversion.json")
    check("T5 R", 200, ic["R"]); check("T5 calibration K and R_cal", {"K": 50, "R_cal": 20}, ic["calibration"])
    check("T5 resolutions", {"100": 20000, "1000": 2000, "5000": 1000, "20000": 1000}, ic["table_S"])
    for row in Q["tab:counts"]:
        for k, W in enumerate(("100", "1000", "5000", "20000")):
            c = ic["table"][f"{row[0]},{W}"]
            same(f"T5 l={row[0]} W={W} data", row[1 + 2 * k], c["data"])
            same(f"T5 l={row[0]} W={W} control", row[2 + 2 * k], c["control"])
    rat = lambda W: [ic["table"][f"{l},{W}"]["ratio"] for l in (1, 2, 5, 20, 100)]
    same("3.2 ratio W=100 lowest", "0.997", min(rat(100))); same("3.2 ratio W=100 highest", "1.019", max(rat(100)))
    same("3.2 ratio W=1000 lowest", "1.001", min(rat(1000))); same("3.2 ratio W=1000 highest", "1.034", max(rat(1000)))
    same("3.2 ratio W=5000 highest", "1.110", max(rat(5000))); same("3.2 ratio W=20000 highest", "1.214", max(rat(20000)))
    check("3.2 cells outside the band of the reference model", 13, sum(not c["inside"] for c in ic["table"].values()))
    check("3.2 cells inside the true-Poisson range", 20, sum(c["ratio_inside_poisson"] for c in ic["table"].values()))
    same("3.2 largest true-Poisson mean ratio", "1.225", max(c["poisson_ratio_mean"] for c in ic["table"].values()))
    check("3.2 it is at l=20, W=20000", "20,20000", max(ic["table"], key=lambda k: ic["table"][k]["poisson_ratio_mean"]))

    # ---------------------------------------------------------------- 3.4 sensitivity by inversion
    se25 = J("25_sensitivity_inversion.json")
    check("3.4 sensitivity R", 2000, se25["R"])
    ps25 = [se25["resolution"][S]["1000"]["p"] for S in ("2000", "5000", "20000", "100000")]
    same("3.4 D_1000 over resolutions lowest", "0.315", min(ps25)); same("3.4 D_1000 over resolutions highest", "0.367", max(ps25))
    same("3.4 counting function distance", "0.00056", se25["counting_function"]["D"])
    same("3.4 counting function p", "0.583", se25["counting_function"]["p"])

    # ---------------------------------------------------------------- Table 7, generators
    g = J("11_reference_model.json")
    names = ["D_100", "D_1000", "D_5000", "D_20000", "D_ratio", "m_ratio", "D_cf", "rho_1", "rho_2"]
    for row, name in zip(Q["tab:generators"], names):
        for cell, gen in zip(row[1:], ("inverse", "nested", "rank")):
            same(f"T7 {name} {gen}", cell, g["stats"][name][gen]["p"])
    check("T7 R", {"inverse": 2000, "nested": 2000, "rank": 1000}, g["R"])
    gens = ("inverse", "nested", "rank")
    spread = max(max(g["stats"][n][a]["p"] for a in gens) - min(g["stats"][n][a]["p"] for a in gens) for n in names)
    check("3.4 p-values differ by at most 0.035 between generators", True, spread <= 0.035)
    check("3.4 nested within 0.023 of inversion", True,
          max(abs(g["stats"][n]["nested"]["p"] - g["stats"][n]["inverse"]["p"]) for n in names) <= 0.023)
    rel = max(abs(g["stats"][n][a]["mean"] / g["stats"][n]["inverse"]["mean"] - 1)
              for n in names if n not in ("D_20000", "rho_2") for a in gens)
    check("3.4 null means within 1.5% except D_20000 and rho_2", True, rel <= 0.015)
    check("3.4 rho_2 null mean of order 1e-5", True, all(abs(g["stats"]["rho_2"][a]["mean"]) < 1e-4 for a in gens))
    for a, q in zip(gens, ("0.00166", "0.00127", "0.00185")):
        same(f"3.4 D_20000 null mean {a}", q, g["stats"]["D_20000"][a]["mean"])

    # ---------------------------------------------------------------- Table 8, decades
    holm_min = 1.0
    for row in Q["tab:decades"]:
        k = re.findall(r"10\^\{?(\d+)\}?", row[0])
        dec = ms["decades"][f"1e{k[0]}-1e{k[1]}"]
        same(f"T8 {row[0]} count", row[1], dec["count"])
        for cell, key in zip(row[2:5], ("mean", "sd", "median")):
            same(f"T8 {row[0]} {key}", cell, dec[key])
        for cell, key in zip(row[5:8], ("D_1000", "D_ratio", "m_ratio")):
            same(f"T8 {row[0]} p {key}", cell, dec["stats"][key]["p"])
            holm_min = min(holm_min, dec["stats"][key]["p_holm"])
    same("T8 smallest Holm over the decades", "0.879", holm_min)

    # ---------------------------------------------------------------- Table 9, classes below 1e18
    st = J("17_strata_factor_count.json")["strata"]
    for row in Q["tab:strata"]:
        s = st[row[0]]
        same(f"T9 r={row[0]} count", row[1], s["N"])
        same(f"T9 r={row[0]} eps50", row[2], s["slow_variation_W1000"]["50"])
        same(f"T9 r={row[0]} D_1000", row[3], s["all"]["D_1000"]["p"])
        same(f"T9 r={row[0]} D_1000 n>=1e12", row[4], s["ge1e12"]["D_1000"]["p"])
        same(f"T9 r={row[0]} D_ratio", row[5], s["all"]["D_ratio"]["p"])
        same(f"T9 r={row[0]} mean ratio", row[6], s["all"]["m_ratio"]["p"])
        same(f"T9 r={row[0]} Q_10", row[7], s["all"]["Q_10"]["p"])

    # ---------------------------------------------------------------- Table 10, confirmatory 1e18-1e24
    ex = J("19_census_1e24_report.json")["by_range"]["all"]["stats"]
    e2 = J("19_census_1e24_report_R200.json")["by_range"]["all"]["stats"]
    for row, name in zip(Q["tab:external"], ("D_1000", "D_ratio", "m_ratio", "Q_10", "rho_1_excess")):
        same(f"T10 {name} value", row[1], ex[name]["value"])
        same(f"T10 {name} R of the plan", row[2], e2[name]["R"])
        same(f"T10 {name} p of the plan", row[3], e2[name]["p"])
        same(f"T10 {name} Holm of the plan", row[4], e2[name]["p_holm"])
        check(f"T10 {name} R beyond the plan", 1000, ex[name]["R"])
        same(f"T10 {name} p with R=1000", row[5], ex[name]["p"])
        same(f"T10 {name} Holm with R=1000", row[6], ex[name]["p_holm"])

    # ---------------------------------------------------------------- Table 11, classes 1e18-1e24
    sv = J("23_slow_variation_1e24_by_r.json")
    for row in Q["tab:external-r"]:
        r = row[0]
        same(f"T11 r={r} count", row[1], J(f"19_census_1e24_observed_r{r}.json")["N"])
        same(f"T11 r={r} eps50", row[2], sv[r]["median"])
        rep = J(f"19_census_1e24_report_r{r}.json")["by_range"]["all"]["stats"]
        for cell, name in zip(row[3:], ("D_1000", "D_ratio", "m_ratio", "Q_10", "rho_1_excess")):
            same(f"T11 r={r} {name}", cell, rep[name]["p"])
            check(f"T11 r={r} {name} R", 1000, rep[name]["R"])
    r3 = J("19_census_1e24_report_r3.json")["by_range"]["all"]["stats"]
    same("3.7 r=3 D_1000 p", "0.004", r3["D_1000"]["p"]); same("3.7 r=3 D_1000 Holm", "0.020", r3["D_1000"]["p_holm"])
    same("3.7 r=3 D_ratio p", "0.682", r3["D_ratio"]["p"]); same("3.7 r=3 mean ratio p", "0.905", r3["m_ratio"]["p"])
    adj = [(J(f"19_census_1e24_report_r{r}.json")["by_range"]["all"]["stats"][n]["p_holm"], r, n) for r in range(3, 13)
           for n in ("D_1000", "D_ratio", "m_ratio", "Q_10", "rho_1_excess")]
    check("3.7 only Holm value at or below 0.05 is r=3 D_1000", [(3, "D_1000")], [(r, n) for a, r, n in adj if a <= 0.05])

    # ---------------------------------------------------------------- Tables 12 and 13, leading digits
    for lab, f, tag in (("tab:digits", "16_digits_bootstrap.json", "T12"), ("tab:digits24", "22_digits_1e24.json", "T13")):
        dg = J(f)
        for row, key in zip(Q[lab], ("observed", "piecewise_prediction", "smooth_prediction")):
            for i, cell in enumerate(row[1:10]):
                same(f"{tag} {key} digit {i + 1}", cell, dg[key][i])
        same(f"{tag} chi-square piecewise", Q[lab][1][10], dg["chi_square"])
        same(f"{tag} chi-square smooth", Q[lab][2][10], dg["chi_square_smooth"])
        same(f"{tag} p smooth", Q[lab][2][11], dg["p_boot_smooth"])
        if Q[lab][1][11].startswith("<"):
            check(f"{tag} p piecewise below 0.0001", True, dg["p_boot"] < 1e-4)
        else:
            same(f"{tag} p piecewise", Q[lab][1][11], dg["p_boot"])

    # ---------------------------------------------------------------- Table A1, power
    pw = J("14b_power_table.json")["table"]
    fams = {"Gamma": "gamma", "AR(1)": "ar1", "fGn": "fgn", "Rare": "cluster", "Repulsion": "hardcore",
            "Change": "breaks", "Modulated": "oscillation", "Antithetic": "antithetic"}
    fam = None
    for row in Q["tab:power"]:
        if row[0]:
            fam = fams[row[0].split()[0].rstrip(",")]
        key = next(k for k in pw if k.split("|")[0] == fam and abs(float(k.split("|")[1]) - num(row[1])) < 1e-12)
        for cell, name in zip(row[2:], ("D_1000", "D_ratio", "m_ratio", "D_cf", "Q_10")):
            same(f"TA1 {key} {name}", cell, pw[key][name]["rate"])
    check("TA1 settings", 37, len(Q["tab:power"]))
    rate = lambda k, n: pw[k][n]["rate"]
    keys = lambda f: [k for k in pw if k.split("|")[0] == f]
    dep_d = [rate(k, "D_1000") for f in ("ar1", "fgn", "antithetic") for k in keys(f) if k != "fgn|0.55"]
    same("3.6 D_1000 under dependence, lowest", "0.044", min(dep_d)); same("3.6 D_1000 under dependence, highest", "0.092", max(dep_d))
    same("3.6 D_1000 at H=0.55", "0.170", rate("fgn|0.55", "D_1000"))
    same("3.6 rho=0.002 at most (%)", "18", 100 * max(rate("ar1|0.002", n) for n in ("D_1000", "D_ratio", "m_ratio", "D_cf", "Q_10")))
    same("3.6 rho=0.005 D_ratio (%)", "69", 100 * rate("ar1|0.005", "D_ratio")); same("3.6 rho=0.005 mean ratio (%)", "73", 100 * rate("ar1|0.005", "m_ratio"))
    same("3.6 rho=0.0075 (%)", "97", 100 * rate("ar1|0.0075", "D_ratio"))
    every = [k for k in ("gamma|0.99", "gamma|1.01", "cluster|0.002", "cluster|0.005", "hardcore|0.002", "hardcore|0.005",
                         "hardcore|0.01", "hardcore|0.02", "breaks|0.02")]
    check("3.6 rejected in every realization by D_1000, D_ratio, mean ratio", True,
          all(rate(k, n) == 1.0 for k in every for n in ("D_1000", "D_ratio", "m_ratio")))
    same("3.6 change of shape 0.01, at least (%)", "98.7", 100 * min(rate("breaks|0.01", n) for n in ("D_1000", "D_ratio", "m_ratio")))
    same("3.6 modulation at most (%)", "11", 100 * max(rate(k, n) for k in keys("oscillation") for n in ("D_1000", "D_ratio", "m_ratio", "Q_10")))
    same("3.6 D_cf at a=0.1 (%)", "91", 100 * rate("oscillation|0.1", "D_cf"))
    cells = [(s[d][n]["p"], r, d, n) for r, s in st.items() for d in ("1e15-1e16", "1e16-1e17", "1e17-1e18") if d in s
             for n in ("D_1000", "D_ratio", "m_ratio", "rho_1", "rho_2", "Q_10")]
    check("3.5 class-decade cells below 1e18", 120, len(cells))
    check("3.5 of these below 0.05", 5, sum(c[0] < 0.05 for c in cells))
    pmin = min(cells)
    same("3.5 smallest cell p", "0.005", pmin[0]); check("3.5 smallest at r=9 rho_1 1e16-1e17", ("9", "1e16-1e17", "rho_1"), pmin[1:])
    holm = sorted(c[0] for c in cells)
    holm_adj = max(min(1.0, (len(holm) - i) * p) for i, p in enumerate(holm[:1]))
    same("3.5 its Holm value over 120", "0.60", holm_adj)
    hw = max((c["ci"][1] - c["ci"][0]) / 2 for t in pw.values() for n, c in t.items() if n != "R")
    check("TA1 half-width at most 0.031", True, hw <= 0.0315)

    # ---------------------------------------------------------------- numbers quoted in the text
    fr = J("11_reference_model.json")["slow_variation_W1000_quantiles"]
    same("2.2 slow variation median", "0.003", fr["50"]); same("2.2 slow variation 99%", "0.095", fr["99"])
    same("2.2 slow variation max", "0.161", fr["max"])
    same("2.3 single-labeling sd of D_cf", "0.00043", pf["D_cf_single_labeling"]["sd"])
    dep = J("15_dependence.json")
    same("3.5 excess lag 1", "0.0014", dep["profile"]["excess"][0]); same("3.5 excess lag 2", "0.0020", dep["profile"]["excess"][1])
    same("3.5 z lag 1", "1.73", dep["profile"]["z"][0]); same("3.5 z lag 2", "2.19", dep["profile"]["z"][1])
    same("3.5 max |z| lags 1-10", "2.19", dep["max_abs_z_10"]["obs"]); same("3.5 critical 1-10", "2.77", dep["max_abs_z_10"]["crit95"])
    same("3.5 p max 1-10", "0.24", dep["max_abs_z_10"]["p"])
    same("3.5 max |z| lags 1-50", "2.63", dep["max_abs_z_50"]["obs"]); check("3.5 argmax lag 18", 18, dep["max_abs_z_50"]["argmax_lag"])
    same("3.5 critical 1-50", "3.30", dep["max_abs_z_50"]["crit95"]); same("3.5 p max 1-50", "0.35", dep["max_abs_z_50"]["p"])
    same("3.5 Q_10", "0.40", dep["Q_10"]["p"]); same("3.5 Q_50", "0.21", dep["Q_50"]["p"]); same("3.5 S_10", "0.34", dep["S_10"]["p"])
    zs = [dep["decades"][k][l]["z"] for k in dep["decades"] for l in ("lag1", "lag2")]
    same("3.5 largest |z| over decades", "1.81", max(abs(z) for z in zs))
    check("3.5 lag 2 positive in every decade", True, all(dep["decades"][k]["lag2"]["excess"] > 0 for k in dep["decades"]))
    check("3.5 lag 1 positive in four decades", 4, sum(dep["decades"][k]["lag1"]["excess"] > 0 for k in dep["decades"]))
    uv = dep["u_variants"]
    for lab, key, q1, q2 in (("W=100", "W=100", "-0.91", "0.28"), ("W=1000", "W=1000", "-0.68", "0.52"),
                             ("W=5000", "W=5000", "0.80", "2.08"), ("C0", "C0", "-2.38", "-1.19")):
        for q, l in ((q1, "lag1"), (q2, "lag2")):
            same(f"3.5 normalized gaps {lab} {l} z", q, uv[key][l]["excess"] / uv[key][l]["null_sd"])
    e = J("19_census_1e24_report.json")["by_range"]["all"]["excess"]
    same("3.7 excess lag 1", "-0.000093", e["lag1"]["value"]); same("3.7 excess lag 2", "-0.000030", e["lag2"]["value"])
    same("3.7 z lag 1", "-1.67", e["lag1"]["z"]); same("3.7 z lag 2", "-0.48", e["lag2"]["z"])
    same("3.7 null sd lag 1", "0.000056", e["lag1"]["null_sd"]); same("3.7 null sd lag 2", "0.000062", e["lag2"]["null_sd"])
    same("3.7 MC se of null mean", "0.000002", e["lag1"]["mc_se_of_null_mean"])
    same("3.7 0.0014 in null sd", "27", (0.0014 - e["lag1"]["value"]) / e["lag1"]["null_sd"])
    by = J("19_census_1e24_report.json")["by_range"]
    decs = [k for k in by if k != "all"]
    ps = [(by[k]["stats"][n]["p"], k, n, by[k]["stats"][n]["p_holm"]) for k in decs for n in by[k]["stats"]]
    check("3.7 decade p-values", 30, len(ps))
    pmin = min(ps)
    same("3.7 smallest decade p", "0.031", pmin[0]); check("3.7 smallest at Q_10 1e22-1e23", ("1e22-1e23", "Q_10"), pmin[1:3])
    same("3.7 its Holm value", "0.155", pmin[3])
    zl = [by[k]["excess"]["lag1"]["z"] for k in decs]
    same("3.7 lag-1 z lowest over decades", "-1.9", min(zl)); same("3.7 lag-1 z highest over decades", "-0.2", max(zl))
    obs = J("19_census_1e24_observed.json")
    check("3.7 numbers 1e18-1e24", 306878295, obs["N"])
    check("3.7 numbers 1e23-1e24", 184897957, obs["decade_bounds"][6] - obs["decade_bounds"][5])
    mu = J("24_multiplicity_by_r.json")
    check("3.7 tests over classes and decades", 350, mu["all_cells"]["tests"])
    check("3.7 below 0.05", 14, mu["all_cells"]["below_005"]); check("3.7 at most 0.01", 4, mu["all_cells"]["at_most_001"])
    same("3.7 smallest BH", "0.525", mu["all_cells"]["min_bh"])
    same("3.7 smallest p over the cells", "0.002", mu["all_cells"]["smallest"][0][3])
    check("3.7 it is r=12 1e19-1e20 D_1000", [12, "1e19-1e20", "D_1000"], mu["all_cells"]["smallest"][0][:3])
    check("3.7 tests in cells of at least 100,000", 270, mu["cells_ge_100000"]["tests"])
    check("3.7 below 0.05 there", 11, mu["cells_ge_100000"]["below_005"]); same("3.7 smallest BH there", "0.54", mu["cells_ge_100000"]["min_bh"])
    s3 = J("23_slow_variation_1e24_by_r_decade.json")["3"][0]
    same("3.7 r=3 first decade share above 0.05 (%)", "32", 100 * s3[2])
    svr = J("23_slow_variation_1e24_by_r.json")
    same("3.7 r=3 share above 0.05 (%)", "0.6", 100 * svr["3"]["share_above_005"])
    same("3.7 r=4 share above 0.05 (%)", "8.0", 100 * svr["4"]["share_above_005"])
    check("3.7 no share above 0.05 for r=5..12", True, all(svr[str(r)]["share_above_005"] == 0 for r in range(5, 13)))
    check("3.7 classes hold 306,862,906", 306862906, sum(J(f"19_census_1e24_observed_r{r}.json")["N"] for r in range(3, 13)))
    integ = J("18_census_1e24_integrity.json")
    check("2.1 lines of the 1e24 file", 308279939, integ["lines"])
    check("2.1 no failed record", 0, integ["bad_product"] + integ["bad_repeat"] + integ["bad_korselt"] + integ["bad_order"])
    check("2.1 counts agree with OEIS", True, all(integ["counts_match_oeis"].values()))
    check("2.1 below 1e18 equal to Pinch", True, integ["below_1e18_equals_pinch"] and integ["below_1e18_factor_counts_equal"])
    d16 = J("16_digits_bootstrap.json")
    same("4.1 bootstrap mean", "8.03", d16["boot_mean"]); same("4.1 bootstrap q95", "15.57", d16["boot_q95"])
    same("4.1 p se", "0.0004", d16["p_boot_se"]); same("4.1 smooth bootstrap mean", "5.81", d16["boot_smooth_mean"])
    same("4.1 smooth bootstrap q95", "12.16", d16["boot_smooth_q95"])
    same("4.1 total variation", "0.0016", d16["total_variation"]); same("4.1 Cohen w", "0.0036", d16["cohen_w"])
    d22 = J("22_digits_1e24.json")
    same("4.1 1e24 bootstrap mean", "8.05", d22["boot_mean"]); same("4.1 1e24 smooth bootstrap mean", "5.83", d22["boot_smooth_mean"])
    same("4.1 1e24 smooth q95", "12.15", d22["boot_smooth_q95"])
    same("4.1 1e24 smooth TV", "0.00008", d22["total_variation_smooth"]); same("4.1 1e24 smooth Cohen w", "0.00021", d22["cohen_w_smooth"])
    same("4.1 1e24 piecewise TV", "0.00088", d22["total_variation"])
    same("4.1 1e24 smooth max deviation", "0.00004", max(abs(a - b) for a, b in zip(d22["observed"], d22["smooth_prediction"])))
    same("4.1 1e24 piecewise max deviation", "0.00057", max(abs(a - b) for a, b in zip(d22["observed"], d22["piecewise_prediction"])))
    check("4.1 1e24 p piecewise is 1/100,001", True, abs(d22["p_boot"] - 1 / 100001) < 1e-12)
    same("4.1 1e24 smooth p se", "0.0006", (d22["p_boot_smooth"] * (1 - d22["p_boot_smooth"]) / d22["R_smooth"]) ** 0.5)
    vd = J("12_validation.json")
    for key, q in (("B1 mean ratio = 2 log 2 - 1", "0.38650"), ("B2 ratio law 2x/(1+x)", "0.239"),
                   ("B4 supremum distance follows the Kolmogorov law", "0.047"),
                   ("B4b supremum distance follows the Kolmogorov law, 20,000 replicates", "0.185"),
                   ("C2 [inversion] time-rescaled gaps Exp(1)", "0.797"), ("C2 [thinning] time-rescaled gaps Exp(1)", "0.666"),
                   ("C5 inversion and thinning give the same rescaled-gap law", "0.859")):
        check(f"3.1 {key}", True, vd[key]["ok"] and q in vd[key]["detail"])
    check("3.1 every validation check passes", True, all(x["ok"] for k, x in vd.items() if isinstance(x, dict) and "ok" in x))
