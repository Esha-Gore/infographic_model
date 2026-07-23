import argparse
import glob
import os
import re
import sys

HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

GLOBS = [
    os.path.join(HF_HOME, "hub", "models--microsoft--Florence-2-base-ft",
                 "snapshots", "*", "modeling_florence2.py"),
    os.path.join(HF_HOME, "modules", "transformers_modules", "microsoft",
                 "*", "*", "modeling_florence2.py"),
]

IMPORT_RE = re.compile(r"^(\s*)from flash_attn")


def find_files():
    files = []
    for pattern in GLOBS:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def patch_file(path, check_only=False):
    with open(path) as f:
        lines = f.readlines()
    active = [i for i, ln in enumerate(lines) if IMPORT_RE.match(ln)]
    if not active:
        return 0  # already clean
    if check_only:
        return len(active)
    for i in active:
        indent = IMPORT_RE.match(lines[i]).group(1)
        lines[i] = f"{indent}pass  # PATCHED-OUT flash_attn: {lines[i].strip()}\n"
    with open(path, "w") as f:
        f.writelines(lines)
    return len(active)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit non-zero if any active flash_attn import remains")
    args = ap.parse_args()

    files = find_files()
    if not files:
        print(f"No cached modeling_florence2.py found under {HF_HOME}.")
        print("Load the model once (or run the pipeline) so it downloads, then re-run this.")
        # Not an error: nothing to patch means nothing will crash yet.
        return 0

    total_active = 0
    for path in files:
        n = patch_file(path, check_only=args.check)
        total_active += n
        state = ("has %d active import(s)" % n) if n else "clean"
        verb = "would patch" if (args.check and n) else ("patched %d line(s)" % n if n else "skipped")
        print(f"[{verb}] {state}: {path}")

    if args.check:
        if total_active:
            print(f"\n{total_active} active flash_attn import(s) remain; run without --check to patch.")
            return 1
        print("\nAll cached modeling files are clean.")
        return 0

    print(f"\nDone. Neutralized flash_attn imports across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
