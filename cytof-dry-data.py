#!/usr/bin/env python3
"""
Simulate small 5v5 CyTOF FCS files (WGP vs Untrained) for a fast pipeline dry run.

Mimics a 63-parameter Helios panel: 6 acquisition/QC channels (Time, Event_length,
Center, Offset, Width, Residual), DNA1/DNA2/Viability, and 54 antibody markers
covering CD8 T, CD4 T, B, myeloid, neutrophil, NK and DC lineages.

WGP is simulated with a "trained immunity" phenotype relative to Untrained:
higher myeloid/DC abundance and higher CD86/MHCII/Ki67/pSTAT3/pS6 in those
lineages. Per-animal random effects are added on top so the 5 vs 5 replicates
have realistic between-mouse variance.

~65% of events per file are simulated as debris/doublets/dead/instrument-noise,
so gating retention lands in the ~25-35% range seen in the real data.

Writes real (minimal) FCS3.0 files -- no extra FCS-writing dependency needed,
only numpy. Also writes a per-file <name>_ground_truth.csv with the true
population label for every event, for validating clustering/gating output.
"""
import argparse
import os
import numpy as np

# ----------------------------- CONFIG (edit here or override via CLI) -----
CONFIG = dict(
    out_dir="./cytof_dryrun_fcs",
    n_events=5000,          # small on purpose -- real files are 400,000 events
    conditions=("treatment", "control"),
    n_replicates=5,
    seed=42,
    cofactor=5.0,           # arcsinh cofactor used to go back to raw ion-count scale
    filename_template="dry_{condition}_{rep}_01_1.fcs",
)

# ----------------------------- Panel definition ----------------------------
NON_MARKER_CHANNELS = ["Time", "Event_length", "Center", "Offset", "Width", "Residual"]
DNA_VIABILITY = [("Ir191Di", "DNA1"), ("Ir193Di", "DNA2"), ("Pt195Di", "Viability")]

MARKERS = [
    "CD45", "CD3e", "CD4", "CD8a", "CD19", "B220", "CD11b", "CD11c", "Ly6G", "Ly6C",
    "NK1.1", "MHCII", "F4_80", "CD64", "CD86", "CD25", "CD69", "FoxP3", "Ki67",
    "pSTAT3", "pSTAT5", "pERK1_2", "pS6", "CXCR4", "CCR7", "CD44", "CD62L", "PD1",
    "PDL1", "CTLA4", "ICOS", "CD127", "CD103", "CD24", "CD115", "SiglecF", "CD49b",
    "TCRb", "CD1d", "TCRgd", "CD21_35", "CD23", "IgD", "IgM", "CD138", "CD38", "CD5",
    "CD9", "CD27", "CD28", "CD117", "CD150", "Tbet", "GATA3",
]
ISOTOPES = [
    "Y89", "Pd102", "Pd104", "Pd105", "Pd106", "Pd108", "In113", "In115", "La139",
    "Ce140", "Pr141", "Nd142", "Nd143", "Nd144", "Nd145", "Nd146", "Sm147", "Nd148",
    "Sm149", "Nd150", "Eu151", "Sm152", "Eu153", "Sm154", "Gd155", "Gd156", "Gd158",
    "Tb159", "Gd160", "Dy161", "Dy162", "Dy163", "Dy164", "Ho165", "Er166", "Er167",
    "Er168", "Tm169", "Er170", "Yb171", "Yb172", "Yb173", "Yb174", "Lu175", "Yb176",
    "Pt198", "Bi209", "Cd111", "Cd112", "Cd114", "Cd116", "Rh103", "Ru101", "Ru104",
]
assert len(ISOTOPES) >= len(MARKERS)

LINEAGES = {
    "CD8_T":       dict(pos=["CD45", "CD3e", "CD8a", "TCRb", "CD44", "CCR7", "CD62L"]),
    "CD4_T":       dict(pos=["CD45", "CD3e", "CD4", "TCRb", "CD44", "CCR7", "CD62L"]),
    "B":           dict(pos=["CD45", "CD19", "B220", "IgD", "IgM", "CD21_35", "CD23", "MHCII"]),
    "Myeloid":     dict(pos=["CD45", "CD11b", "F4_80", "CD64", "MHCII", "CD86", "CD115"]),
    "Neutrophil":  dict(pos=["CD45", "CD11b", "Ly6G", "Ly6C"]),
    "NK":          dict(pos=["CD45", "NK1.1", "CD49b", "CD27"]),
    "DC":          dict(pos=["CD45", "CD11c", "MHCII", "CD86", "CD24"]),
}
BASE_LINEAGE_PROPORTIONS = {
    "CD8_T": 0.12, "CD4_T": 0.20, "B": 0.28, "Myeloid": 0.18,
    "Neutrophil": 0.12, "NK": 0.06, "DC": 0.04,
}
# WGP = trained-immunity phenotype: more myeloid/DC, activated state in those lineages
WGP_ABUNDANCE_SHIFT = {"Myeloid": +0.06, "DC": +0.02, "CD4_T": -0.05, "B": -0.03}
WGP_STATE_MARKERS = ["CD86", "MHCII", "Ki67", "pSTAT3", "pS6"]
WGP_STATE_SHIFT = 1.2  # arcsinh-space bump applied in Myeloid/DC for WGP

JUNK_FRACTIONS = dict(debris=0.25, doublet=0.15, dead=0.15, noise=0.10)  # sums to 0.65

# ----------------------------- FCS3.0 writer (numpy only) ------------------
def write_fcs(filepath, channel_names, data, delimiter="/"):
    data = np.ascontiguousarray(data, dtype="<f4")
    n_events, n_params = data.shape

    kv = {}
    kv["$BEGINANALYSIS"] = "0"
    kv["$ENDANALYSIS"] = "0"
    kv["$BEGINSTEXT"] = "0"
    kv["$ENDSTEXT"] = "0"
    kv["$BYTEORD"] = "1,2,3,4"
    kv["$DATATYPE"] = "F"
    kv["$MODE"] = "L"
    kv["$NEXTDATA"] = "0"
    kv["$PAR"] = str(n_params)
    kv["$TOT"] = str(n_events)
    for i, (short, long_) in enumerate(channel_names, start=1):
        col_max = float(np.max(data[:, i - 1])) if n_events else 1.0
        kv[f"$P{i}B"] = "32"
        kv[f"$P{i}E"] = "0,0"
        kv[f"$P{i}N"] = short
        kv[f"$P{i}R"] = str(int(np.ceil(col_max)) + 1)
        kv[f"$P{i}S"] = long_

    def build_text(kv):
        parts = []
        for k, v in kv.items():
            parts.append(k)
            parts.append(v)
        return delimiter + delimiter.join(parts) + delimiter

    header_len = 58
    kv["$BEGINDATA"] = "0"
    kv["$ENDDATA"] = "0"
    text_len = len(build_text(kv))
    data_len = n_events * n_params * 4

    # iterate in case digit-count of the offsets changes text length
    for _ in range(4):
        text_start = header_len
        text_end = text_start + text_len - 1
        data_start = text_end + 1
        data_end = data_start + data_len - 1
        kv["$BEGINDATA"] = str(data_start)
        kv["$ENDDATA"] = str(data_end)
        new_len = len(build_text(kv))
        if new_len == text_len:
            break
        text_len = new_len

    text_str = build_text(kv)

    def fmt_offset(v, width=8):
        s = str(v)
        return s.rjust(width) if len(s) <= width else "0" * width

    header = "FCS3.0" + "    "
    header += fmt_offset(text_start) + fmt_offset(text_end)
    header += fmt_offset(data_start) + fmt_offset(data_end)
    header += fmt_offset(0) + fmt_offset(0)
    assert len(header) == 58

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(text_str.encode("ascii"))
        f.write(data.tobytes(order="C"))


# ----------------------------- simulation helpers ---------------------------
def arcsinh_to_raw(x, cofactor, rng, bg=2.0):
    raw = np.sinh(x) * cofactor
    raw = raw + rng.poisson(bg, size=x.shape)
    return np.clip(raw, 0, None)


def make_marker_matrix(n, pos_markers, rng, animal_shift, extra_state=None):
    """arcsinh-space marker matrix: baseline noise everywhere, elevated on pos_markers."""
    x = rng.normal(0, 0.25, size=(n, len(MARKERS)))
    idx = {m: j for j, m in enumerate(MARKERS)}
    for m in pos_markers:
        mean = rng.uniform(3.2, 5.5)
        x[:, idx[m]] = rng.normal(mean, 0.5, size=n)
    if extra_state:
        for m, shift in extra_state.items():
            x[:, idx[m]] = x[:, idx[m]] + shift
    x = x + animal_shift[np.newaxis, :]
    return x


def simulate_sample(condition, rep, n_events, cofactor, rng):
    animal_shift = rng.normal(0, 0.15, size=len(MARKERS))  # per-mouse random effect

    proportions = dict(BASE_LINEAGE_PROPORTIONS)
    if condition == "WGP":
        for lin, delta in WGP_ABUNDANCE_SHIFT.items():
            proportions[lin] += delta
    jitter = {k: max(0.01, v + rng.normal(0, 0.015)) for k, v in proportions.items()}
    tot = sum(jitter.values())
    proportions = {k: v / tot for k, v in jitter.items()}

    n_junk = int(round(n_events * sum(JUNK_FRACTIONS.values())))
    n_good = n_events - n_junk

    lineage_names = list(proportions.keys())
    counts = [int(round(n_good * proportions[l])) for l in lineage_names]
    counts[-1] += n_good - sum(counts)  # fix rounding

    rows_marker, rows_time_meta, labels = [], [], []

    for lin, n in zip(lineage_names, counts):
        if n <= 0:
            continue
        extra_state = None
        if condition == "WGP" and lin in ("Myeloid", "DC"):
            extra_state = {m: WGP_STATE_SHIFT for m in WGP_STATE_MARKERS}
        x = make_marker_matrix(n, LINEAGES[lin]["pos"], rng, animal_shift, extra_state)
        raw = arcsinh_to_raw(x, cofactor, rng)
        dna = rng.normal(150, 15, size=n).clip(20, None)
        dna2 = dna * rng.normal(0.95, 0.05, size=n)
        viability = rng.exponential(4, size=n)  # live: low signal
        evlen = rng.normal(15, 2, size=n).clip(5, None)
        center = rng.normal(30, 4, size=n)
        offset = rng.normal(0, 1, size=n)
        width = rng.normal(30, 3, size=n)
        residual = rng.normal(0, 2, size=n)
        rows_marker.append(raw)
        rows_time_meta.append(np.column_stack([evlen, center, offset, width, residual, dna, dna2, viability]))
        labels += [lin] * n

    def junk_block(kind, n):
        if n <= 0:
            return None, None, None
        if kind == "debris":
            x = rng.normal(0, 0.2, size=(n, len(MARKERS)))
            raw = arcsinh_to_raw(x, cofactor, rng, bg=0.5)
            dna = rng.normal(8, 4, size=n).clip(0, None)
            dna2 = dna * rng.normal(0.9, 0.1, size=n)
            viability = rng.exponential(3, size=n)
            evlen = rng.normal(6, 1.5, size=n).clip(1, None)
            center, offset, width, residual = (rng.normal(30, 4, n), rng.normal(0, 1, n),
                                                rng.normal(30, 3, n), rng.normal(0, 2, n))
        elif kind == "doublet":
            lin_a = rng.choice(list(LINEAGES.keys()), size=n)
            lin_b = rng.choice(list(LINEAGES.keys()), size=n)
            xa = np.stack([make_marker_matrix(1, LINEAGES[a]["pos"], rng, animal_shift)[0] for a in lin_a])
            xb = np.stack([make_marker_matrix(1, LINEAGES[b]["pos"], rng, animal_shift)[0] for b in lin_b])
            raw = arcsinh_to_raw(xa, cofactor, rng) + arcsinh_to_raw(xb, cofactor, rng)
            dna = rng.normal(150, 15, size=n) * rng.normal(1.9, 0.1, size=n)
            dna2 = dna * rng.normal(0.95, 0.05, size=n)
            viability = rng.exponential(4, size=n)
            evlen = rng.normal(30, 3, size=n).clip(15, None)
            center, offset, width, residual = (rng.normal(30, 4, n), rng.normal(0, 1, n),
                                                rng.normal(45, 4, n), rng.normal(0, 2, n))
        elif kind == "dead":
            x = rng.normal(0.3, 0.3, size=(n, len(MARKERS)))
            raw = arcsinh_to_raw(x, cofactor, rng)
            dna = rng.normal(140, 25, size=n).clip(10, None)
            dna2 = dna * rng.normal(0.9, 0.1, size=n)
            viability = rng.normal(80, 15, size=n).clip(30, None)
            evlen = rng.normal(15, 3, size=n).clip(5, None)
            center, offset, width, residual = (rng.normal(30, 4, n), rng.normal(0, 1, n),
                                                rng.normal(30, 3, n), rng.normal(0, 2, n))
        else:  # noise -- instrument/gaussian-parameter outliers
            x = rng.normal(0, 0.3, size=(n, len(MARKERS)))
            raw = arcsinh_to_raw(x, cofactor, rng)
            dna = rng.normal(150, 15, size=n).clip(20, None)
            dna2 = dna * rng.normal(0.95, 0.05, size=n)
            viability = rng.exponential(4, size=n)
            evlen = rng.normal(15, 2, size=n).clip(5, None)
            center = rng.uniform(0, 60, size=n)
            offset = rng.normal(0, 15, size=n)
            width = rng.uniform(0, 90, size=n)
            residual = rng.normal(0, 20, size=n)
        meta = np.column_stack([evlen, center, offset, width, residual, dna, dna2, viability])
        return raw, meta, [f"junk_{kind}"] * n

    for kind, frac in JUNK_FRACTIONS.items():
        n = int(round(n_junk * (frac / sum(JUNK_FRACTIONS.values()))))
        raw, meta, lab = junk_block(kind, n)
        if raw is not None:
            rows_marker.append(raw)
            rows_time_meta.append(meta)
            labels += lab

    marker_mat = np.vstack(rows_marker)
    meta_mat = np.vstack(rows_time_meta)
    n_total = marker_mat.shape[0]

    # order: acquire in a random shuffled order, then assign monotonic Time
    order = rng.permutation(n_total)
    marker_mat = marker_mat[order]
    meta_mat = meta_mat[order]
    labels = [labels[i] for i in order]

    inter_arrival = rng.exponential(1.0 / 400.0, size=n_total)  # ~400 events/sec
    time_col = np.cumsum(inter_arrival)

    evlen, center, offset, width, residual, dna, dna2, viability = meta_mat.T
    full = np.column_stack([time_col, evlen, center, offset, width, residual, dna, dna2, viability, marker_mat])
    return full, labels


def build_channel_list():
    channels = [(c, c) for c in NON_MARKER_CHANNELS]
    channels += list(DNA_VIABILITY)
    channels += [(f"{iso}Di", marker) for iso, marker in zip(ISOTOPES, MARKERS)]
    return channels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=CONFIG["out_dir"])
    p.add_argument("--n-events", type=int, default=CONFIG["n_events"])
    p.add_argument("--n-replicates", type=int, default=CONFIG["n_replicates"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--cofactor", type=float, default=CONFIG["cofactor"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    channels = build_channel_list()
    assert len(channels) == 63, f"expected 63 channels, got {len(channels)}"

    master_seed = np.random.SeedSequence(args.seed)
    child_seeds = master_seed.spawn(len(CONFIG["conditions"]) * args.n_replicates)
    seed_i = 0

    print(f"{'file':45s} {'n_events':>9s} {'n_good':>7s} {'retained_%':>10s}")
    for condition in CONFIG["conditions"]:
        for rep in range(1, args.n_replicates + 1):
            rng = np.random.default_rng(child_seeds[seed_i]); seed_i += 1
            data, labels = simulate_sample(condition, rep, args.n_events, args.cofactor, rng)
            fname = CONFIG["filename_template"].format(condition=condition, rep=rep)
            fpath = os.path.join(args.out_dir, fname)
            write_fcs(fpath, channels, data)

            n_good = sum(1 for l in labels if not l.startswith("junk_"))
            print(f"{fname:45s} {len(labels):9d} {n_good:7d} {100*n_good/len(labels):9.1f}%")

            gt_path = fpath.replace(".fcs", "_ground_truth.csv")
            with open(gt_path, "w") as f:
                f.write("event_index,true_population,condition,replicate\n")
                for i, l in enumerate(labels):
                    f.write(f"{i},{l},{condition},{rep}\n")

    print(f"\nWrote {len(CONFIG['conditions']) * args.n_replicates} FCS files "
          f"({args.n_events} events, 63 params each) to {args.out_dir}")


if __name__ == "__main__":
    main()
