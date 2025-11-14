import os
import glob
import time
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
recon_path = "/ceph/submit/data/user/a/anton100/datasets/recon3D/"
labels_path = "/ceph/submit/data/user/a/anton100/datasets/labels/"
output_path = "/ceph/submit/data/user/a/anton100/output_v3"   # ← NEW CLEAN OUTPUT

Path(output_path).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Find recon files
# -----------------------------
recon_files = sorted(glob.glob(os.path.join(recon_path, "recon3D_*.parquet")))
assert len(recon_files) > 0, "No recon files found!"

print(f"Found {len(recon_files)} recon shards")
bad_files = []

# -----------------------------
# Helper: safe read with retry
# -----------------------------
def safe_read_parquet(path, retries=5):
    for attempt in range(retries):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"⚠️ READ FAIL [{attempt+1}/{retries}] for {path}: {e}")
            time.sleep(1)

    print(f"❌ Giving up on {path}")
    return None

# -----------------------------
# Helper: safe write (atomic)
# -----------------------------
def safe_write_parquet(df, out_path):
    tmp_path = out_path + ".tmp"

    try:
        df.to_parquet(tmp_path, index=False)
        # Atomic rename (safe on CEPH)
        os.replace(tmp_path, out_path)
    except Exception as e:
        print(f"❌ WRITE FAIL for {out_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    # verify after write
    try:
        pq.read_table(out_path)
        return True
    except Exception as e:
        print(f"❌ VERIFICATION FAIL for {out_path}: {e}")
        return False

# -----------------------------
# MAIN LOOP — sequential (CEPH-safe)
# -----------------------------
for idx, recon_file in enumerate(recon_files):
    base = os.path.basename(recon_file).replace("recon3D_", "").replace(".parquet", "")
    label_file = os.path.join(labels_path, f"labels_{base}.parquet")

    out_file = os.path.join(output_path, f"part.{idx:04d}.parquet")

    print(f"\n🔹 Processing shard {idx+1}/{len(recon_files)}")
    print(f"   recon:  {recon_file}")
    print(f"   labels: {label_file}")

    if not os.path.exists(label_file):
        print(f"❌ SKIP — missing labels for {recon_file}")
        bad_files.append((recon_file, "missing labels"))
        continue

    # ---- read both files safely ----
    df_recon = safe_read_parquet(recon_file)
    df_labels = safe_read_parquet(label_file)

    if df_recon is None or df_labels is None:
        bad_files.append((recon_file, "read failure"))
        continue

    # ---- check alignment ----
    if len(df_recon) != len(df_labels):
        print(f"❌ ROW MISMATCH: recon={len(df_recon)}, labels={len(df_labels)}")
        bad_files.append((recon_file, "row mismatch"))
        continue

    # ---- merge ----
    df = pd.concat([df_recon.reset_index(drop=True),
                    df_labels.reset_index(drop=True)], axis=1)

    # ---- write safely ----
    ok = safe_write_parquet(df, out_file)
    if not ok:
        bad_files.append((recon_file, "write/verify fail"))
        continue

    print(f"   ✓ Saved: {out_file}")

print("\n✔ Done")
print("Bad files:")
for b in bad_files:
    print("  ", b)
