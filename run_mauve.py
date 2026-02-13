import mauve


def compute_mauve(originales, sanitizados, device):
    print("\n--- calculating MAUVE ---")
    try:
        out = mauve.compute_mauve(
            p_text=originales,
            q_text=sanitizados,
            device_id=0 if device == "cuda" else -1,
            max_text_length=512,
            verbose=False,
        )
        return out.mauve
    except Exception as e:
        print(f"Error computing MAUVE: {e}")
        return None
