import numpy as np
import scipy

# Matrix form of the enrichment analysis
def enrichment_analysis_matrix(corr_matrix, moa_matches, percentile):
    threshold = np.percentile(corr_matrix, percentile)

    v11 = np.sum(np.logical_and(corr_matrix > threshold, moa_matches)) 
    v12 = np.sum(np.logical_and(corr_matrix > threshold, np.logical_not(moa_matches)))
    v21 = np.sum(np.logical_and(corr_matrix <= threshold, moa_matches))
    v22 = np.sum(np.logical_and(corr_matrix <= threshold, np.logical_not(moa_matches)))

    V = np.asarray([[v11, v12], [v21, v22]])
    print(percentile, threshold)
    print(V, np.sum(V))
    r = scipy.stats.fisher_exact(V, alternative="greater")
    result = {"percentile": percentile, "threshold": threshold, "ods_ratio": r[0], "p-value": r[1]}
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