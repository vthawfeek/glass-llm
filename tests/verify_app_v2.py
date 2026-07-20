"""Boot app_v2.py under Streamlit's AppTest across several knob combinations and assert
no exceptions render. Run after the zoo build completes:  python tests/verify_app_v2.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streamlit.testing.v1 import AppTest
import ui2

K = ui2.KNOBS
DOM_D, DOM_G = K["domain"]["options"]         # domain, generic
LO, MD, HI = K["volume"]["options"]           # volume labels (carry MB suffixes)
H4, H8 = K["heads"]["options"]
COMBOS = [
    # (domain, volume, heads, finetune, rag)
    (DOM_D, HI, H8, "Yes", "No"),
    (DOM_G, LO, H4, "No", "No"),
    (DOM_D, MD, H4, "Yes", "Yes"),
    (DOM_G, HI, H8, "No", "Yes"),
    (DOM_D, LO, H4, "No", "No"),
]


def set_knob(at, key, val):
    for sb in at.selectbox:
        if sb.key == key:
            sb.set_value(val)
            return
    raise AssertionError(f"selectbox {key} not found")


def set_radio(at, key, val):
    for r in at.radio:
        if r.key == key:
            r.set_value(val)
            return
    raise AssertionError(f"radio {key} not found")


def run_combo(combo, embed_choice=None):
    dom, vol, heads, ft, rg = combo
    at = AppTest.from_file(str(ROOT / "app_v2.py"), default_timeout=180)
    at.run()
    if at.exception:
        return f"initial run exception: {at.exception}"
    set_knob(at, "k_domain", dom)
    set_knob(at, "k_volume", vol)
    set_knob(at, "k_heads", heads)
    set_knob(at, "k_ft", ft)
    set_knob(at, "k_rag", rg)
    at.run()
    if at.exception:
        return f"exception: {at.exception}"
    if rg == "Yes" and embed_choice is not None:
        try:
            set_radio(at, "k_rag_embed", embed_choice)
        except AssertionError as e:
            return str(e)  # e.g. MiniLM index not built -- radio only has one option
        at.run()
        if at.exception:
            return f"exception after embed_choice={embed_choice!r}: {at.exception}"
    return None


def main():
    fails = 0
    for combo in COMBOS:
        err = run_combo(combo)
        label = " · ".join(str(c) for c in combo)
        if err:
            fails += 1
            print(f"FAIL  {label}\n      {err}")
        else:
            print(f"OK    {label}")
    # RAG embedding-source selector: exercise both branches explicitly on one combo
    for choice in ["MiniLM (recommended)", "Our model (from-scratch)"]:
        err = run_combo((DOM_D, MD, H4, "Yes", "Yes"), embed_choice=choice)
        label = f"RAG embed source = {choice}"
        if err:
            fails += 1
            print(f"FAIL  {label}\n      {err}")
        else:
            print(f"OK    {label}")
    total = len(COMBOS) + 2
    print(f"\n{total-fails}/{total} combinations rendered cleanly.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
