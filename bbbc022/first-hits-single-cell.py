import pandas as pd
import numpy as np
import seaborn as sb
import os
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("../profiling/")
import profiling

PROJECT_ROOT = "/dgx1nas1/cellpainting-datasets/BBBC022/"
EXP = "cp_dataset"
NUM_FEATURES = 672

OUTPUT_BASE_NAME = 'cp_dataset'

PERT_NAME = "Treatment"
CTRL_NAME = "DMSO@0"

REG_PARAM = 1e-2


def find_first_hits(features, feats, pert_name, control):
    results = []

    gen = features[features[pert_name] != control].groupby(
        ["Metadata_Plate", "Metadata_Well", pert_name])["Replicate_Use"].count().reset_index().iterrows()
    for k, r in tqdm(gen):
        # Select samples in a well
        well = features.query(f"Metadata_Plate == {r.Metadata_Plate} & Metadata_Well == '{r.Metadata_Well}'").index
        A = np.asarray(features.loc[well,feats])

        # Get cells in other wells
        others = features.query(f"Metadata_Plate != {r.Metadata_Plate} | Metadata_Well != '{r.Metadata_Well}'").index
        B = np.asarray(features.loc[others,feats])

        # Compute cosine similarity
        C = np.dot(A, B.T)
        An = np.linalg.norm(A, axis=1)
        Bn = np.linalg.norm(B, axis=1)
        cos = C / (An[:,np.newaxis] @ Bn[:,np.newaxis].T)

        # Rank cells in other wells
        ranking = np.argsort(-cos, axis=1)

        # Find first hits
        H = np.asarray(features.loc[others, pert_name]  == r[pert_name], dtype=np.uint8)
        for h in range(len(well)):
            hit = np.where(H[ranking[h]] == 1)[0][0]
            results.append({"Metadata_Plate":r.Metadata_Plate, 
                            "Metadata_Well":r.Metadata_Well, 
                            "Treatment":r.Treatment,
                            "first_hit": hit,
                           })
    return pd.DataFrame(data=results)


def summarize(results):
    summary = results.groupby([PERT_NAME])["first_hit"].mean().reset_index()
    summary["std"] = results.groupby([PERT_NAME])["first_hit"].std().reset_index()["first_hit"]
    summary["top_percent"] = (summary["first_hit"] / len(results))*100
    summary["percent_group"] = np.ceil(summary["top_percent"])
    summary["coef_var"] = summary["std"] / summary["first_hit"]
    summary["signal_noise"] = summary["first_hit"] / summary["std"]
    return summary


def visualize(summary):
    plt.figure(figsize=(10,5))
    summary = summary.sort_values("first_hit",na_position='last')
    sb.barplot(data=summary, x=PERT_NAME, y="top_percent")
    print("Treatments with hits in the top 1%:", summary[summary["top_percent"] <= 1].shape[0])
    plt.show()
    return summary


def request_controls(control_meta_subset):
    control_meta_subset.reset_index(inplace=True, drop=True)
    sc_control_features_array = np.zeros((control_meta_subset.shape[0], NUM_FEATURES))
    for i in tqdm(control_meta_subset.index):
        filename = PROJECT_ROOT + "outputs/" + EXP + "/features/{}/{}/{}.npz"
        filename = filename.format(
            control_meta_subset.loc[i, "Metadata_Plate"],
            control_meta_subset.loc[i, "Metadata_Well"],
            control_meta_subset.loc[i, "Metadata_Site"],
        )
        if os.path.isfile(filename):
            with open(filename, "rb") as data:
                info = np.load(data)
                sc_control_features_array[i, :] = info["features"][control_meta_subset.loc[i, "SCID"]]

    return pd.merge(control_meta_subset, pd.DataFrame(data=sc_control_features_array), how="left", left_index=True,
                    right_index=True)


def request_treatments(treatments_meta_subset):
    treatments_meta_subset.reset_index(inplace=True, drop=True)
    sc_treatment_features_array = np.zeros((treatments_meta_subset.shape[0], NUM_FEATURES))
    for i in tqdm(treatments_meta_subset.index):
        filename = PROJECT_ROOT + "outputs/" + EXP + "/features/{}/{}/{}.npz"
        filename = filename.format(
            treatments_meta_subset.loc[i, "Metadata_Plate"],
            treatments_meta_subset.loc[i, "Metadata_Well"],
            treatments_meta_subset.loc[i, "Metadata_Site"],
        )
        if os.path.isfile(filename):
            with open(filename, "rb") as data:
                info = np.load(data)
                sc_treatment_features_array[i, :] = info["features"][treatments_meta_subset.loc[i, "SCID"]]

    return pd.merge(treatments_meta_subset, pd.DataFrame(data=sc_treatment_features_array), how="left", left_index=True,
                    right_index=True)


if __name__ == "__main__":
    # Load metadata
    metadata = pd.read_csv(os.path.join(PROJECT_ROOT, "inputs/metadata/index_after_qc_trimmed_maxconc.csv"))
    Y = pd.read_csv("data/BBBC022_MOA_MATCHES_fixed_trimmed_filtered_v2.csv")
    profiles = pd.merge(metadata, Y, left_on="Compound", right_on="Var1")
    meta = pd.concat((profiles, metadata[metadata[PERT_NAME] == CTRL_NAME]), axis=0).reset_index()

    total_single_cells = 0
    sc_meta_idx = []
    sc_sc_idx = []
    for i in tqdm(meta.index):
        filename = PROJECT_ROOT + "outputs/" + EXP + "/features/{}/{}/{}.npz"
        filename = filename.format(
            meta.loc[i, "Metadata_Plate"],
            meta.loc[i, "Metadata_Well"],
            meta.loc[i, "Metadata_Site"],
        )
        if os.path.isfile(filename):
            with open(filename, "rb") as data:
                info = np.load(data)
                cells = info["features"].shape[0]
                sc_meta_idx += [i]*cells
                for c in range(cells):
                    sc_sc_idx.append(c)
                total_single_cells += cells

    cols = ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Plate_Map_Name",
            "broad_sample_Replicate", "Treatment", "Compound", "Concentration", "Replicate_Use"]
    sc_meta = pd.merge(pd.DataFrame(sc_meta_idx, columns=["ID"]), meta[cols], left_on="ID", right_index=True)
    sc_meta = pd.merge(pd.DataFrame(sc_sc_idx, columns=["SCID"]), sc_meta, left_index=True, right_index=True)

    sc_controls = sc_meta[sc_meta[PERT_NAME] == CTRL_NAME]
    sc_treatments = sc_meta[sc_meta[PERT_NAME] != CTRL_NAME]

    feats = [i for i in range(NUM_FEATURES)]

    for i in range(10):
        sc_control_features = request_controls(sc_controls.sample(50000, replace=False))
        sc_sample = []
        gen = sc_treatments.groupby(["Metadata_Plate", "Metadata_Well", PERT_NAME])["Replicate_Use"].count().reset_index().iterrows()
        for k, r in tqdm(gen):
            if r.Replicate_Use >= 10:
                sc_sample.append(sc_treatments.query(
                    f"Metadata_Plate == {r.Metadata_Plate} & Metadata_Well == '{r.Metadata_Well}'").sample(10))
        sc_sample = pd.concat(sc_sample)
        whN = profiling.WhiteningNormalizer(sc_control_features[feats], reg_param=REG_PARAM)
        sc_sample_features = request_treatments(sc_sample)
        whD = whN.normalize(sc_sample_features[feats])
        sc_sample_features[feats] = whD
        sc_results = find_first_hits(sc_sample_features, feats, PERT_NAME, CTRL_NAME)
        sc_summary = summarize(sc_results)
        sc_summary = visualize(sc_summary)
        sc_summary.to_csv(OUTPUT_BASE_NAME + '_single_cell_level_sample_{}.csv'.format(i))
