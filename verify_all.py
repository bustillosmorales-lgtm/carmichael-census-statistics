"""Compare every number quoted in the article with the outputs in results/.
Each line prints the quoted value, the computed value and OK or DIFFERS. Exit status 1 on any difference."""
import json
import math
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
J = lambda name: json.load(open(os.path.join(R, name)))
ok_all = True


def check(label, quoted, computed, tol=0.0):
    global ok_all
    if isinstance(quoted, float) or isinstance(computed, float):
        good = abs(float(quoted) - float(computed)) <= tol
    else:
        good = quoted == computed
    ok_all &= good
    print(f"{'OK     ' if good else 'DIFFERS'} {label:62s} quoted={quoted!s:>14} computed={computed!s:>22}")


c = J("01_census_validation.json")
check("census total", 1401644, c["total"])
check("lines below 1e16", 246683, c["lines"]["carmichael-16"])
check("lines 1e16-1e17", 338672, c["lines"]["carmichael17"])
check("lines 1e17-1e18", 816289, c["lines"]["carmichael18"])
check("sixteen checkpoints match", True, c["all_checkpoints_match"])
check("Korselt failures", 0, c["bad_korselt"])
check("product mismatches", 0, c["bad_product"])
check("repeated primes", 0, c["bad_squarefree"])

s = J("02_spacing_tests.json")
check("gaps in Table 1 caption", 1401643, 1401644 - 1)
for W in ("1000", "5000"):
    check(f"D_W below q95 at W={W}", True, s["windows"][W]["D"] < s["windows"][W]["null_q95"])
check("D_W above q95 at W=20000", True, s["windows"]["20000"]["D"] > s["windows"]["20000"]["null_q95"])
check("no test rejects at 5%", True, all(v["p"] > 0.05 for v in s["tests"].values()))
check("gamma 0.99 and 1.01 rejected in every realization (D, ratio, mean)", True,
      all(s["power"][k][t] == 1.0 for k in ("gamma 0.99", "gamma 1.01") for t in ("D_1000", "D_rat", "m_rat")))
check("Poisson mean ratio 2 log 2 - 1", 0.3863, round(2 * math.log(2) - 1, 4))
check("tabulation to 1e24 holds about 220 times the points", 220, round(308279939 / 1401644))
check("standard error factor about 15", 15, round(math.sqrt(308279939 / 1401644)))

se = J("03_serial_dependence.json")
check("excess at lags 1 and 2 of order 1e-3", True, all(1e-4 < abs(se["all"]["excess"][i]) < 1e-2 for i in (0, 1)))

d = J("06_leading_digits.json")
obs = [0.2153, 0.1546, 0.1255, 0.1074, 0.0943, 0.0850, 0.0781, 0.0723, 0.0676]
pred = [0.2161, 0.1552, 0.1254, 0.1071, 0.0944, 0.0850, 0.0778, 0.0719, 0.0671]
for i in range(9):
    check(f"observed digit {i + 1}", obs[i], round(d["observed"][i], 4), 1e-12)
    check(f"predicted digit {i + 1}", pred[i], round(d["predicted"][i], 4), 1e-12)
check("max absolute deviation 0.0008", 0.0008, round(d["max_abs_dev"], 4), 1e-12)
check("max relative deviation 0.69%", 0.69, round(100 * d["max_rel_dev"], 2), 1e-12)
check("chi-square 18.37", 18.37, round(d["chi_square"], 2), 1e-12)
check("critical value 15.51", 15.51, round(d["critical_5pct"], 2), 1e-12)
check("p = 0.019", 0.019, round(d["p_value"], 3), 1e-12)
alphas = {3: 0.264, 4: 0.174, 5: 0.225, 6: 0.306, 7: 0.396, 8: 0.515, 9: 0.661, 10: 0.853}
counts = {3: 16047, 4: 14927, 5: 77356, 6: 220354, 7: 288319, 8: 160362, 9: 36158, 10: 2718}
for k in alphas:
    check(f"alpha r={k}", alphas[k], round(d["alpha_by_r_decade17"][str(k)]["alpha"], 3), 1e-12)
    check(f"count r={k}", counts[k], d["alpha_by_r_decade17"][str(k)]["count"])
check("r=11 count 48", 48, d["alpha_by_r_decade17"]["11"]["count"])
check("r=11 interval lower 0.63", 0.63, round(d["alpha_by_r_decade17"]["11"]["ci95"][0], 2), 1e-12)
check("r=11 interval upper 2.17", 2.17, round(d["alpha_by_r_decade17"]["11"]["ci95"][1], 2), 1e-12)
check("eight columns cover 816,241", 816241, sum(counts.values()))
check("decade total 816,289", 816289, d["decade17_total"])
check("consecutive intervals do not overlap from r=4", True,
      all(d["alpha_by_r_decade17"][str(k)]["ci95"][1] < d["alpha_by_r_decade17"][str(k + 1)]["ci95"][0] for k in range(4, 10)))
check("aggregate alpha 0.3810", 0.3810, round(d["alpha_aggregate"], 4), 1e-12)
check("growth exponent 0.3766", 0.3766, round(d["theta_bar"], 4), 1e-12)
check("aggregate gap 0.0044", 0.0044, round(d["gap_aggregate"], 4), 1e-12)
check("numbers with r = 3..6", 668994, d["count_r3to6"])
check("restricted alpha 0.2905", 0.2905, round(d["alpha_r3to6"], 4), 1e-12)
check("prediction covers all but 561", 1401643, d["n_predicted"])

t = J("07_tuples_and_sieve.json")
table = {3: (35586, 19040, [1, 2, 3], 783), 4: (40685, 40092, [1, 3, 6, 10], 18), 5: (178063, 178038, None, None),
         6: (414660, 414660, None, None)}
for k, (num, dist, top, occ) in table.items():
    check(f"tuples k={k} numbers", num, t["table"][str(k)]["numbers"])
    check(f"tuples k={k} distinct", dist, t["table"][str(k)]["distinct"])
    if top:
        check(f"tuples k={k} most frequent", (top, occ), tuple(t["table"][str(k)]["top"][0]))
check("k=5 at most 3 occurrences", 3, t["table"]["5"]["top"][0][1])
check("k=6 every tuple once", 1, t["table"]["6"]["top"][0][1])
check("share of (1,2,3) 2.20%", 2.20, round(100 * 783 / 35586, 2), 1e-12)
nxt = [[[1, 2, 15], 330], [[1, 2, 5], 308], [[1, 3, 4], 294], [[1, 3, 5], 288], [[1, 3, 10], 281]]
check("next five tuples for k=3", nxt, t["table"]["3"]["top"][1:6])
check("numbers with at most six factors", 668994, t["total_at_most_six"])
check("tuples with LCM <= 44", 105, t["lcm44_tuples"])
check("of which three-factor", 28, t["lcm44_three_factor"])
check("sieve agrees on all", 105, t["lcm44_agree"])
extract = {"(1, 2, 3)": 783, "(1, 2, 5)": 308, "(1, 3, 4)": 294, "(1, 3, 5)": 288, "(1, 3, 8)": 217,
           "(1, 2, 15)": 330, "(2, 3, 5)": 243, "(1, 6, 7)": 149}
for a, v in extract.items():
    check(f"sieve extract {a}", [v, v], t["extract"][a])
check("(1,2,3) below 1e18", 783, t["sieve_123_1e18"])
check("(1,2,3) below 1e24", 30601, t["sieve_123_1e24"])
check("run to 1e24 under one minute", True, t["seconds_1e24"] < 60)
check("Chernick closed form on all tuples", 2866, t["chernick_closed_form_ok"])
check("pairwise coprime tuples below 40", 2866, t["chernick_tuples"])

o = J("08_counting_obstruction.json")
chron = [(0.8284, 0.0812), (0.4490, 0.2609), (0.3500, 0.3078), (0.3174, 0.3232), (0.3032, 0.3299), (0.2961, 0.3333),
         (0.2843, 0.3389)]
for row, (th, ex) in zip(o["chronology"], chron):
    check(f"theta {row['year']}", th, row["theta_4"], 1e-12)
    check(f"exponent {row['year']}", ex, row["exponent_4"], 1e-12)
check("record 0.33895...", 0.33895, math.floor(o["chronology"][-1]["exponent"] * 1e5) / 1e5, 1e-12)
check("observed exponent at 1e24", 0.3537, round(o["observed_exponent_1e24"], 4), 1e-12)
eq = {5040: (23, 1152, 12, "8.0e-03", 31, 144, 67), 55440: (40, 11520, 60, "5.5e-05", 63, 3600, 375),
      720720: (75, 138240, 60, "1.6e-08", 127, 3600, 525), 4324320: (128, 829440, 360, None, 127, 129600, 3147)}
for L, (r, phi, lam, dev, idx2, sq, cl) in eq.items():
    e = o["equid"][str(L)]
    check(f"L={L} r, phi, lambda", (r, phi, lam), (e["r"], e["phi"], e["lam"]))
    if dev:
        check(f"L={L} deviation", dev, f"{e['max_rel_dev']:.1e}")
    else:
        check(f"L={L} deviation below 1e-14", True, e["max_rel_dev"] < 1e-14)
    check(f"L={L} index-two subgroups", idx2, e["index_two_subgroups"])
    check(f"L={L} lambda squared", sq, e["lam_sq"])
    check(f"L={L} classical bound", cl, round(e["classical"]))
    check(f"L={L} guarantee exceeds supply", True, e["classical"] > e["r"])
check("shortfall factor 1,012", 1012, round(o["equid"]["4324320"]["shortfall_sq"]))
check("function-field cases", 163, o["function_field_cases"])
check("function-field agreement", 163, o["function_field_agree"])

w = J("09_withdrawn_measurements.json")
i2 = w["item2"]["1.1e+11"]
check("item ii data 15.58", 15.58, round(i2["data"], 2), 1e-12)
check("item ii inhomogeneous 16.11", 16.11, round(i2["inhomogeneous_mean"], 2), 1e-12)
check("item ii range 15.68", 15.68, round(i2["inhomogeneous_min"], 2), 1e-12)
check("item ii range 16.48", 16.48, round(i2["inhomogeneous_max"], 2), 1e-12)
check("item ii homogeneous 1.00", 1.00, round(i2["homogeneous_mean"], 2), 1e-12)
i3 = w["item3"]
check("item iii tuples", 1231, i3["total"])
check("item iii without solution", 1113, i3["empty"])
check("item iii admissible and all occur", (118, 118), (i3["admissible"], i3["occurring"]))
check("item iii at most one class", 1, i3["max_classes"])

print("\n--- second revision ---")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_r2
verify_r2.run(check)

print("\nALL QUOTED NUMBERS MATCH" if ok_all else "\nSOME QUOTED NUMBERS DIFFER")
sys.exit(0 if ok_all else 1)
