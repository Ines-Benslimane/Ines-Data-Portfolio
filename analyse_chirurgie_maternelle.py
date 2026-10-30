#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chirurgie bariatrique maternelle et poids de naissance des enfants
===================================================================
Analyse exploratoire de deux extraits de données hospitalières pseudonymisées
(fichier MÈRES : séjour de chirurgie + accouchements ; fichier ENFANTS : naissances).

Question : le poids de naissance des enfants diffère-t-il selon que la mère a été
opérée AVANT ou APRÈS la conception ?

Usage :
    python analyse_chirurgie_maternelle.py ENFANT_CHIR_M2DMS.csv MERE_CHIR_M2DMS.csv sortie/

Sorties (dans le dossier `sortie/`) : resultats.json, base_analyse.csv (sans identifiants), fig1/2/3.png
Dépendances : pandas, numpy, scipy, statsmodels, matplotlib
NB : ne jamais publier les fichiers sources ni la base d'analyse (données de santé).
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")

# Couleurs (validées daltonisme) : avant = ocre, après = bleu ; texte en encre sombre
C_AVANT, C_APRES, INK, INK2, GRID = "#b5651d", "#1f6fb2", "#3b2f2f", "#6b5644", "#e4dcd4"
SEUIL_PETIT_POIDS, SEUIL_MACROSOMIE = 2500, 4000
MAX_ENFANTS_PAR_IDENTIFIANT = 4   # au-delà : identifiant « mère » manifestement partagé


# ============================================================================
# 1. CONSTRUCTION DE LA BASE (une ligne = un enfant)
# ============================================================================
def construire_base(enfant_csv, mere_csv):
    log = {}
    e = pd.read_csv(enfant_csv, sep=";", dtype=str)
    m = pd.read_csv(mere_csv, sep=";", dtype=str)
    dt = lambda s: pd.to_datetime(s, format="%d/%m/%Y")   # dates tronquées au mois (jour = 01)
    m["date_hosp"], m["date_codegross"] = dt(m.date_hosp), dt(m.date_codegross)
    m["age_gross"] = m.age_gross.astype(int)
    e["date_naiss"] = dt(e.date_naiss)
    for c in ("sexe", "age_gest", "poids"):
        e[c] = e[c].astype(int)

    log["lignes_fichier_enfants"], log["lignes_fichier_meres"] = len(e), len(m)
    log["enfants_bruts"] = int(e.anonyme.nunique())
    log["meres_fichier"] = int(m.anonyme_mereNN.nunique())

    # --- une ligne par enfant (le fichier source a une ligne par diagnostic) ---
    ch = e.groupby("anonyme").agg(
        mere=("anonyme_mereNN", "first"), n_mere=("anonyme_mereNN", "nunique"),
        sexe=("sexe", "first"), ag=("age_gest", "first"), n_ag=("age_gest", "nunique"),
        poids=("poids", "first"), n_p=("poids", "nunique"),
        naiss=("date_naiss", "first"), n_dn=("date_naiss", "nunique")).reset_index()
    codes = e.groupby("anonyme").diag.apply(set)
    ch["pag_code"] = ch.anonyme.map(lambda a: bool({"P050", "P051"} & codes[a]))   # petit pour l'âge gest.
    ch["gag_code"] = ch.anonyme.map(lambda a: bool({"P080", "P081"} & codes[a]))   # gros pour l'âge gest.

    # --- exclusions : attributs contradictoires pour un même enfant ---
    conflit = (ch.n_mere > 1) | (ch.n_ag > 1) | (ch.n_p > 1) | (ch.n_dn > 1)
    log["exclus_attributs_contradictoires"] = int(conflit.sum())
    ch = ch[~conflit].copy()
    # --- exclusions : identifiants « mère » partagés (jusqu'à 291 enfants pour un identifiant) ---
    nb = ch.groupby("mere").size()
    agreges = nb[nb > MAX_ENFANTS_PAR_IDENTIFIANT].index
    log["identifiants_meres_agreges"] = int(len(agreges))
    log["exclus_identifiants_agreges"] = int(nb[nb > MAX_ENFANTS_PAR_IDENTIFIANT].sum())
    ch = ch[~ch.mere.isin(agreges)].copy()
    m = m[~m.anonyme_mereNN.isin(agreges)]

    # --- chirurgie : premier séjour opératoire de la mère ---
    m1 = m.sort_values("date_hosp").drop_duplicates("anonyme_mereNN")
    types = {"HFFC0180": "Sleeve", "HFCC0030": "Bypass", "HFMC0070": "Anneau", "HFMC0060": "Anneau"}
    chir = pd.DataFrame({"mere": m1.anonyme_mereNN.values, "date_chir": m1.date_hosp.values,
                         "acte": m1.acte.values})
    chir["type_chir"] = chir.acte.map(types).fillna("Autre")
    ch = ch.merge(chir, on="mere", how="left")

    # --- accouchement correspondant (même mère, même mois) : âge maternel, complications ---
    ob = m.assign(
        diab_gest=m.diag_gross.str.startswith("O244"),                                   # diabète gestationnel
        hta=m.diag_gross.str[:3].isin(["O10", "O11", "O13", "O14", "O15", "O16"]),        # troubles hypertensifs
        obesite=m.typ_ob_gro.fillna("").str.startswith("E66"))                           # obésité codée au séjour
    liv = ob.groupby(["anonyme_mereNN", "date_codegross"]).agg(
        age_mere=("age_gross", "first"), diab_gest=("diab_gest", "max"),
        hta=("hta", "max"), obesite=("obesite", "max")).reset_index()
    ch = ch.merge(liv, left_on=["mere", "naiss"], right_on=["anonyme_mereNN", "date_codegross"], how="left")
    log["enfants_sans_accouchement_apparie"] = int(ch.age_mere.isna().sum())
    ch = ch.drop(columns=["anonyme_mereNN", "date_codegross", "n_mere", "n_ag", "n_p", "n_dn"])

    # --- naissances multiples et rang de naissance (parmi les naissances du fichier) ---
    ch["multiple"] = ch.groupby(["mere", "naiss"]).anonyme.transform("size") > 1
    ev = ch[["mere", "naiss"]].drop_duplicates().sort_values(["mere", "naiss"])
    ev["rang"] = ev.groupby("mere").cumcount() + 1
    ch = ch.merge(ev, on=["mere", "naiss"])
    ch["rang3"] = ch.rang.clip(upper=3)

    # --- EXPOSITION : moment de la conception par rapport à la chirurgie ---
    ch["mois_chir_naiss"] = ((ch.naiss.dt.year - ch.date_chir.dt.year) * 12
                             + (ch.naiss.dt.month - ch.date_chir.dt.month))
    ch["mois_chir_concep"] = ch.mois_chir_naiss - ch.ag * 7 / 30.44    # mois entre chirurgie et conception

    def classer(r):
        if r.mois_chir_naiss < 0:
            return "0 Avant"                    # né avant la chirurgie de sa mère
        if r.mois_chir_naiss == 0 or r.mois_chir_concep < 0:
            return "x Ambigu"                   # chirurgie pendant la grossesse / même mois
        if r.mois_chir_concep < 12:
            return "1 <12 mois"
        if r.mois_chir_concep < 24:
            return "2 12-23 mois"
        return "3 >=24 mois"
    ch["expo4"] = ch.apply(classer, axis=1)
    ch["apres"] = ch.expo4.map({"0 Avant": 0, "1 <12 mois": 1, "2 12-23 mois": 1, "3 >=24 mois": 1})
    ch["annee_naiss"] = ch.naiss.dt.year

    # --- critères de jugement ---
    ch["petit_poids"] = ch.poids < SEUIL_PETIT_POIDS
    ch["macrosomie"] = ch.poids >= SEUIL_MACROSOMIE
    ch["premature"] = ch.ag < 37
    ch["fille"] = (ch.sexe == 2).astype(int)
    ch["mult"] = ch.multiple.astype(int)
    log["enfants_analysables"] = int(len(ch))
    log["meres_analysables"] = int(ch.mere.nunique())
    return ch, log


# ============================================================================
# 2. OUTILS STATISTIQUES
# ============================================================================
def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return float(lo), float(hi)


def coef(res, nom, expo=False):
    b, (lo, hi), p = res.params[nom], res.conf_int().loc[nom], res.pvalues[nom]
    f = np.exp if expo else (lambda x: x)
    return {"est": float(f(b)), "lo": float(f(lo)), "hi": float(f(hi)), "p": float(p)}


def ols_cluster(formule, d):
    return smf.ols(formule, d).fit(cov_type="cluster", cov_kwds={"groups": d.mere})


def logit_cluster(formule, d):
    return smf.logit(formule, d).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": d.mere})


AJUST = "fille + age_mere + C(rang3) + mult + C(annee_naiss) + C(type_chir)"


# ============================================================================
# 3. ANALYSES
# ============================================================================
def analyser(ch):
    R = {}
    base = ch[ch.apres.notna() & ch.age_mere.notna()].copy()
    base["apres"] = base.apres.astype(int)
    R["n_modele"] = {"enfants": int(len(base)), "meres": int(base.mere.nunique()),
                     "avant": int((base.apres == 0).sum()), "apres": int((base.apres == 1).sum()),
                     "ambigus_exclus": int((ch.expo4 == "x Ambigu").sum())}
    av, ap = base[base.apres == 0], base[base.apres == 1]

    # --- tableau descriptif ---
    def resume(d):
        return {
            "n": int(len(d)),
            "poids_moy": float(d.poids.mean()), "poids_sd": float(d.poids.std()),
            "ag_moy": float(d.ag.mean()), "age_mere_moy": float(d.age_mere.mean()),
            "pct_filles": float(100 * d.fille.mean()), "pct_multiples": float(100 * d.multiple.mean()),
            "pct_petit_poids": float(100 * d.petit_poids.mean()),
            "pct_macrosomie": float(100 * d.macrosomie.mean()),
            "pct_premature": float(100 * d.premature.mean()),
            "pct_pag": float(100 * d.pag_code.mean()), "pct_gag": float(100 * d.gag_code.mean()),
            "pct_diab_gest": float(100 * d.diab_gest.mean()), "pct_hta": float(100 * d.hta.mean()),
            "pct_obesite_codee": float(100 * d.obesite.mean()),
            "pct_sleeve": float(100 * (d.type_chir == "Sleeve").mean()),
            "pct_bypass": float(100 * (d.type_chir == "Bypass").mean()),
            "pct_anneau": float(100 * (d.type_chir == "Anneau").mean()),
            "annee_moy": float(d.annee_naiss.mean())}
    R["descriptif"] = {"avant": resume(av), "apres": resume(ap)}
    R["repartition_delai"] = {k: int(v) for k, v in base.expo4.value_counts().sort_index().items()}

    # --- poids de naissance : différence moyenne (g) ---
    t = stats.ttest_ind(ap.poids, av.poids, equal_var=False)
    R["poids"] = {
        "brut_welch": {"diff": float(ap.poids.mean() - av.poids.mean()), "p": float(t.pvalue)},
        "brut": coef(ols_cluster("poids ~ apres", base), "apres"),
        "ajuste": coef(ols_cluster("poids ~ apres + " + AJUST, base), "apres"),
        "ajuste_ag": coef(ols_cluster("poids ~ apres + " + AJUST + " + C(ag)", base), "apres")}

    # --- comparaison intra-mère (frères et sœurs) : mères ayant des naissances avant ET après ---
    both = base.groupby("mere").apres.nunique()
    fe = base[base.mere.isin(both[both == 2].index)]
    R["intra_mere"] = {"meres": int(fe.mere.nunique()), "enfants": int(len(fe)),
                       "sans_ag": coef(ols_cluster("poids ~ apres + fille + mult + C(rang3) + C(mere)", fe), "apres"),
                       "avec_ag": coef(ols_cluster("poids ~ apres + fille + mult + C(rang3) + C(ag) + C(mere)", fe), "apres")}

    # --- selon le délai chirurgie -> conception ---
    c = base[base.expo4 != "x Ambigu"]
    r = ols_cluster("poids ~ C(expo4, Treatment('0 Avant')) + " + AJUST, c)
    nom = lambda k: f"C(expo4, Treatment('0 Avant'))[T.{k}]"
    R["poids_delai"] = {k: {**coef(r, nom(k)), "n": int((c.expo4 == k).sum())}
                        for k in ("1 <12 mois", "2 12-23 mois", "3 >=24 mois")}

    # --- critères binaires : prévalences, OR bruts et ajustés (erreurs robustes par mère) ---
    R["binaires"] = {}
    for y, lib in (("petit_poids", "Poids < 2500 g"), ("macrosomie", "Poids ≥ 4000 g"),
                   ("premature", "Naissance < 37 SA"), ("pag_code", "Petit pour l'âge gest. (codé)"),
                   ("gag_code", "Gros pour l'âge gest. (codé)")):
        base[y + "_i"] = base[y].astype(int)
        r0 = logit_cluster(f"{y}_i ~ apres", base)
        r1 = logit_cluster(f"{y}_i ~ apres + " + AJUST, base)
        k0, k1 = int(av[y].sum()), int(ap[y].sum())
        R["binaires"][y] = {
            "libelle": lib,
            "avant": {"k": k0, "n": len(av), "pct": 100 * k0 / len(av), "ic": [100 * x for x in wilson(k0, len(av))]},
            "apres": {"k": k1, "n": len(ap), "pct": 100 * k1 / len(ap), "ic": [100 * x for x in wilson(k1, len(ap))]},
            "or_brut": coef(r0, "apres", True), "or_ajuste": coef(r1, "apres", True)}

    # --- maternelles (descriptif complémentaire : cohérence avec la littérature) ---
    R["maternel"] = {}
    for y, lib in (("diab_gest", "Diabète gestationnel"), ("hta", "Troubles hypertensifs")):
        base[y + "_i"] = base[y].astype(int)
        r1 = logit_cluster(f"{y}_i ~ apres + age_mere + C(rang3) + C(annee_naiss) + C(type_chir)", base)
        R["maternel"][y] = {"libelle": lib, "avant": float(100 * av[y].mean()),
                            "apres": float(100 * ap[y].mean()), "or_ajuste": coef(r1, "apres", True)}

    # --- analyses de sensibilité (différence de poids ajustée, g) ---
    sens = {
        "Analyse principale": base,
        "Naissances uniques seulement": base[base.mult == 0],
        "Naissances à terme (≥ 37 SA)": base[base.ag >= 37],
        "Sleeve gastrectomie seule": base[base.type_chir == "Sleeve"],
        "Bypass gastrique seul": base[base.type_chir == "Bypass"]}
    R["sensibilite"] = {}
    for nom_s, d in sens.items():
        f = AJUST if nom_s not in ("Sleeve gastrectomie seule", "Bypass gastrique seul") \
            else AJUST.replace(" + C(type_chir)", "")
        R["sensibilite"][nom_s] = {**coef(ols_cluster("poids ~ apres + " + f, d), "apres"), "n": int(len(d))}
    return R, base


# ============================================================================
# 4. FIGURES
# ============================================================================
def style():
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def fr(x, d=1):
    """Nombre au format français (virgule décimale, espace pour les milliers)."""
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def fig1(base, R, out):
    """Distribution du poids de naissance, avant / après."""
    style()
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=200)
    bins = np.arange(500, 5500, 150)
    for d, col in ((base[base.apres == 0], C_AVANT), (base[base.apres == 1], C_APRES)):
        ax.hist(d.poids, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.28, lw=0)
        ax.hist(d.poids, bins=bins, density=True, histtype="step", color=col, lw=2.2)
        ax.axvline(d.poids.mean(), color=col, lw=1.8, ls="--")
    ax.axvline(SEUIL_PETIT_POIDS, color=GRID, lw=1.4, zorder=0)
    ax.axvline(SEUIL_MACROSOMIE, color=GRID, lw=1.4, zorder=0)
    ax.text(SEUIL_PETIT_POIDS - 50, 0.97, "2 500 g", ha="right", va="top", color=INK2, fontsize=12,
            transform=ax.get_xaxis_transform())
    ax.text(SEUIL_MACROSOMIE + 50, 0.97, "4 000 g", ha="left", va="top", color=INK2, fontsize=12,
            transform=ax.get_xaxis_transform())
    a, p = R["descriptif"]["avant"], R["descriptif"]["apres"]
    ax.text(0, 1.20, f"Nés avant la chirurgie : moyenne {fr(a['poids_moy'], 0)} g (n = {fr(a['n'], 0)})",
            transform=ax.transAxes, color=C_AVANT, fontsize=13, fontweight="bold")
    ax.text(0, 1.10, f"Conçus après la chirurgie : moyenne {fr(p['poids_moy'], 0)} g (n = {fr(p['n'], 0)})",
            transform=ax.transAxes, color=C_APRES, fontsize=13, fontweight="bold")
    ax.set_xlabel("Poids de naissance (g)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(500, 5400)
    fig.tight_layout()
    fig.savefig(out + "/fig1_distribution_poids.png", facecolor="white")
    plt.close(fig)


def fig2(R, out):
    """Prévalence des critères, avant / après (IC 95 % de Wilson)."""
    style()
    cles = ["petit_poids", "pag_code", "premature", "gag_code", "macrosomie"]
    libelles = {"petit_poids": "Poids\n< 2 500 g", "pag_code": "Petit pour l'âge\ngestationnel (codé)",
                "premature": "Naissance\n< 37 SA", "gag_code": "Gros pour l'âge\ngestationnel (codé)",
                "macrosomie": "Poids\n≥ 4 000 g"}
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    h, pos = 0.36, np.arange(len(cles))[::-1]
    for i, k in enumerate(cles):
        b = R["binaires"][k]
        for dy, grp, col in ((+h / 2, "avant", C_AVANT), (-h / 2, "apres", C_APRES)):
            v = b[grp]
            ax.barh(pos[i] + dy, v["pct"], height=h - 0.04, color=col)
            ax.errorbar(v["pct"], pos[i] + dy, xerr=[[v["pct"] - v["ic"][0]], [v["ic"][1] - v["pct"]]],
                        color=INK2, lw=1.3, capsize=3)
            ax.text(v["ic"][1] + 0.3, pos[i] + dy, fr(v["pct"]) + " %", va="center", color=INK, fontsize=12)
    ax.set_yticks(pos)
    ax.set_yticklabels([libelles[k] for k in cles])
    ax.set_xlabel("Proportion d'enfants (%)")
    ax.set_xlim(0, 16.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_AVANT, label="Nés avant la chirurgie"),
                       Patch(color=C_APRES, label="Conçus après la chirurgie")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out + "/fig2_criteres.png", facecolor="white")
    plt.close(fig)


def fig3(R, out):
    """Écart de poids de naissance (g) par rapport aux enfants nés avant la chirurgie."""
    style()
    groupes = [
        ("Modèles principaux", [("Principal (ajusté)", R["poids"]["ajuste"]),
                                ("À âge gest. égal", R["poids"]["ajuste_ag"]),
                                (f"Intra-mère ({R['intra_mere']['meres']} mères)", R["intra_mere"]["avec_ag"])]),
        ("Délai chirurgie → conception", [("< 12 mois", R["poids_delai"]["1 <12 mois"]),
                                          ("12 à 23 mois", R["poids_delai"]["2 12-23 mois"]),
                                          ("≥ 24 mois", R["poids_delai"]["3 >=24 mois"])]),
        ("Sensibilité", [("Naissances uniques", R["sensibilite"]["Naissances uniques seulement"]),
                         ("À terme (≥ 37 SA)", R["sensibilite"]["Naissances à terme (≥ 37 SA)"]),
                         ("Sleeve seule", R["sensibilite"]["Sleeve gastrectomie seule"]),
                         ("Bypass seul", R["sensibilite"]["Bypass gastrique seul"])])]
    fig, ax = plt.subplots(figsize=(7, 6.4), dpi=200)
    y, ys, labs = 0.0, [], []
    for titre, items in groupes:
        ax.text(0.0, y, titre, transform=ax.get_yaxis_transform(), color=INK2, fontsize=11.5,
                style="italic", va="center", ha="left")
        y -= 0.9
        for nom, c in items:
            ax.plot([c["lo"], c["hi"]], [y, y], color=C_APRES, lw=2.4, solid_capstyle="round")
            ax.plot(c["est"], y, "o", color=C_APRES, ms=9, mec="white", mew=1.6, zorder=3)
            ax.text(c["hi"] + 14, y, fr(c["est"], 0) + " g", va="center", color=INK, fontsize=12)
            ys.append(y); labs.append(nom); y -= 1
        y -= 0.35
    ax.axvline(0, color=INK2, lw=1.1)
    ax.set_yticks(ys); ax.set_yticklabels(labs)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Écart de poids de naissance (g, IC à 95 %)\npar rapport aux enfants nés avant la chirurgie")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-620, 70)
    ax.set_xticks([-600, -400, -200, 0])
    ax.set_xticklabels(["−600", "−400", "−200", "0"])
    ax.set_ylim(min(ys) - 0.8, 0.6)
    fig.tight_layout()
    fig.savefig(out + "/fig3_ecarts_poids.png", facecolor="white")
    plt.close(fig)


# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    enfant_csv, mere_csv, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    ch, log = construire_base(enfant_csv, mere_csv)
    R, base = analyser(ch)
    R["flux"] = log
    fig1(base, R, out); fig2(R, out); fig3(R, out)
    cols = ["sexe", "ag", "poids", "annee_naiss", "type_chir", "age_mere", "rang", "multiple", "expo4",
            "mois_chir_concep", "diab_gest", "hta", "obesite"]
    base[cols].to_csv(out + "/base_analyse_SANS_identifiants.csv", index=False)   # à ne pas publier
    json.dump(R, open(out + "/resultats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))
