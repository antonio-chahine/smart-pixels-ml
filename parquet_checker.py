# parquet_checker_full.py
import os, glob
import pyarrow.parquet as pq

base = "/ceph/submit/data/user/a/anton100/output_v2"
files = sorted(glob.glob(f"{base}/part.*.parquet"))

bad = []

for f in files:
    try:
        pq.read_table(f, columns=[0])  # smallest read
    except Exception as e:
        print("❌ BAD:", f)
        print("    ", e)
        bad.append(f)

print("Done.")
print("Bad files:", bad)
