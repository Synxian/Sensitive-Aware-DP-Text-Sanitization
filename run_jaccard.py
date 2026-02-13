def compute_jaccard_similarity(original, new):
    print("\n--- calculating Jaccard Similarity ---")
    set_orig = set(original.lower().split())
    set_new = set(new.lower().split())

    intersection = len(set_orig.intersection(set_new))
    union = len(set_orig.union(set_new))
    return intersection / union if union > 0 else 0
