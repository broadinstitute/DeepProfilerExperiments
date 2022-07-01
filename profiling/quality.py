import numpy as np
import scipy

# Enrichment analysis
def enrichment_analysis(similarities, moa_matches, percentile):
    threshold = np.percentile(similarities, percentile)

    v11 = np.sum(np.logical_and(similarities > threshold, moa_matches)) 
    v12 = np.sum(np.logical_and(similarities > threshold, np.logical_not(moa_matches)))
    v21 = np.sum(np.logical_and(similarities <= threshold, moa_matches))
    v22 = np.sum(np.logical_and(similarities <= threshold, np.logical_not(moa_matches)))

    V = np.asarray([[v11, v12], [v21, v22]])
    r = scipy.stats.fisher_exact(V, alternative="greater")
    result = {"percentile": percentile, "threshold": threshold, "ods_ratio": r[0], "p-value": r[1]}
    if np.isinf(r[0]):
        result["ods_ratio"] = v22
    return result

# Fraction strong test
def fraction_strong_test(treatment_corr, null, num_treatments):
    null.sort()
    p95 = null[ int( 0.95*len(null) ) ]
    fraction = np.sum([m > p95 for m in treatment_corr])/num_treatments
    print("Treatments tested:", num_treatments)
    print("At 95th percentile of the null")
    print("Fraction strong: {:5.2f}%".format(fraction*100))
    print("Null threshold: {:6.4f}".format(p95))