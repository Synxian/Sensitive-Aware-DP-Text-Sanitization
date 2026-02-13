from bert_score import score


def compute_bert_score(original, new, device):
    print("\n--- calculating BERTScore ---")

    P, R, F1 = score(new, original, lang="en", verbose=True, device=device)

    return F1.mean().item()
