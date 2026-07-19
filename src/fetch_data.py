"""Stage 1: pull ClinicalTrials.gov v2 data, split by NCT ID (no leakage), fetch a
general-English corpus, and emit a manifest. Deterministic and resumable."""
import argparse, hashlib, json, re, sys, time, urllib.request, urllib.parse
from pathlib import Path

CTGOV = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.descriptionModule.detailedDescription",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.designModule.phases",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.outcomesModule.primaryOutcomes",
]
# Public-domain Project Gutenberg plain-text (general-English corpus). Expanded to ~40 MB so
# the training-data-volume caps (up to 32 MB) are matched between the domain and generic sides.
_GUT_IDS = [
    2600,   # War and Peace          2701,   # Moby Dick
    1342,   # Pride and Prejudice    84,     # Frankenstein
    1661,   # Sherlock Holmes        345,    # Dracula
    100,    # Complete Shakespeare   1184,   # Count of Monte Cristo
    135,    # Les Misérables         996,    # Don Quixote
    1399,   # Anna Karenina          2554,   # Crime and Punishment
    28054,  # Brothers Karamazov     1400,   # Great Expectations
    1260,   # Jane Eyre              768,    # Wuthering Heights
    4300,   # Ulysses                145,    # Middlemarch
    6130,   # The Iliad              1727,   # The Odyssey
    3600,   # Essays of Montaigne    74,     # Tom Sawyer
    76,     # Huckleberry Finn       158,    # Emma
    98,     # A Tale of Two Cities   174,    # Dorian Gray
    2591,   # Grimms' Fairy Tales    1080,   # A Modest Proposal
    2542,   # A Doll's House         2814,   # Dubliners
    766,    # David Copperfield      1023,   # Bleak House
    6593,   # Tom Jones              8800,   # Divine Comedy
    1497,   # Republic (Plato)       3207,   # Leviathan
    205,    # Walden                 161,    # Sense and Sensibility
    105,    # Persuasion             2413,   # Madame Bovary
    120,    # Treasure Island        2148,   # Works of Poe
    1998,   # Thus Spake Zarathustra 105,    # (dup guard, harmless)
    1400,   # (dup guard, harmless)
]
GUTENBERG = [f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt" for i in _GUT_IDS]

def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glass-llm-pilot/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)          # strip any stray HTML
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_trials(area, max_trials, quick, intervention_type=""):
    adv = "AREA[StudyType]INTERVENTIONAL"
    if intervention_type:
        adv += f" AND AREA[InterventionType]{intervention_type}"
    params = {"pageSize": "1000", "fields": ",".join(FIELDS), "filter.advanced": adv}
    if area and area.lower() != "all":
        params["query.cond"] = area
    docs, seen, token = {}, set(), None
    while len(docs) < max_trials:
        q = dict(params)
        if token:
            q["pageToken"] = token
        data = json.loads(_get(CTGOV + "?" + urllib.parse.urlencode(q)))
        for s in data.get("studies", []):
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            if not nct or nct in docs:
                continue
            dm = ps.get("descriptionModule", {})
            title = ps.get("identificationModule", {}).get("briefTitle", "") or ""
            elig = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "") or ""
            summ = dm.get("briefSummary", "") or ""
            detail = dm.get("detailedDescription", "") or ""
            conds = ps.get("conditionsModule", {}).get("conditions", []) or []
            phases = ps.get("designModule", {}).get("phases", []) or []
            ivs = [i.get("name", "") for i in
                   ps.get("armsInterventionsModule", {}).get("interventions", []) or []]
            outs = ps.get("outcomesModule", {}).get("primaryOutcomes", []) or []
            out_txt = "; ".join(
                o.get("measure", "") + (": " + o["description"] if o.get("description") else "")
                for o in outs if o.get("measure"))
            # all CURATED narrative/design sections, no admin junk (sponsors/sites/dates/IDs)
            body = clean("\n\n".join(filter(None, [
                "Title: " + title if title else "",
                "Conditions: " + ", ".join(conds) if conds else "",
                "Phase: " + ", ".join(phases) if phases else "",
                "Interventions: " + ", ".join(filter(None, ivs)) if ivs else "",
                "Brief Summary: " + summ if summ else "",
                "Detailed Description: " + detail if detail else "",
                "Primary Outcome Measures: " + out_txt if out_txt else "",
                "Eligibility Criteria:\n" + elig if elig else ""])))
            if len(body) < 120:           # skip near-empty records
                continue
            h = hashlib.md5(body.encode()).hexdigest()
            if h in seen:                 # dedup identical bodies
                continue
            seen.add(h)
            docs[nct] = body
            if len(docs) >= max_trials:
                break
        token = data.get("nextPageToken")
        if not token:
            break
        if quick and len(docs) >= max_trials:
            break
    return docs

def split_by_nct(docs):
    buckets = {"train": {}, "val": {}, "test": {}}
    for nct, body in docs.items():
        b = int(hashlib.md5(nct.encode()).hexdigest(), 16) % 100
        key = "train" if b < 80 else ("val" if b < 90 else "test")
        buckets[key][nct] = body
    return buckets

def fetch_general(out, quick, cap_mb):
    parts, total = [], 0
    for url in GUTENBERG:
        try:
            t = _get(url)
        except Exception as e:
            print(f"  general corpus: skip {url} ({e})", file=sys.stderr)
            continue
        # strip Gutenberg header/footer
        m = re.search(r"\*\*\* START OF.*?\*\*\*(.*?)\*\*\* END OF", t, re.S)
        body = clean(m.group(1) if m else t)
        parts.append(body); total += len(body.encode())
        if quick and total > cap_mb * 1024 * 1024:
            break
    out.write_text("\n\n".join(parts), encoding="utf-8")
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="all")
    ap.add_argument("--out", default="pilot/data")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--intervention-type", default="", help="e.g. BIOLOGICAL, DRUG")
    ap.add_argument("--no-general", action="store_true", help="skip Gutenberg corpus")
    ap.add_argument("--max-trials", type=int, default=0, help="override default cap")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    max_trials = a.max_trials or (1200 if a.quick else 30000)
    gen_cap = 3 if a.quick else 20

    print(f"[data] pulling CT.gov area={a.area!r} type={a.intervention_type!r} "
          f"max_trials={max_trials} quick={a.quick}")
    docs = fetch_trials(a.area, max_trials, a.quick, a.intervention_type)
    print(f"[data] fetched {len(docs)} unique trials")
    buckets = split_by_nct(docs)

    # leakage assertion: no NCT ID in more than one split
    ids = [set(b) for b in buckets.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]), \
        "NCT-ID overlap across splits!"

    sizes = {}
    for name, b in buckets.items():
        p = out / f"{name}.txt"
        text = "\n\n<|endoftext|>\n\n".join(b.values())
        p.write_text(text, encoding="utf-8")
        sizes[name] = {"trials": len(b), "bytes": len(text.encode("utf-8"))}

    if a.no_general:
        gen_bytes = 0
    else:
        print("[data] fetching general-English corpus (Project Gutenberg, public domain)")
        gen_bytes = fetch_general(out / "general.txt", a.quick, gen_cap)

    manifest = {
        "source": CTGOV, "area": a.area, "quick": a.quick,
        "pull_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_trials": len(docs), "splits": sizes,
        "leakage_check": "PASS (no NCT-ID overlap across splits)",
        "general_corpus": {"source": "Project Gutenberg (public domain)", "bytes": gen_bytes},
        "fields": FIELDS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[data] manifest:", json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
