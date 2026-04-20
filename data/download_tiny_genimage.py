"""
Download TheKernel01/Tiny-GenImage và tổ chức theo đúng layout GenImage:

  data/raw/GenImage/<Generator Name>/train/{ai,nature}/
  data/raw/GenImage/<Generator Name>/val/{ai,nature}/
  data/metadata/metadata.jsonl
  data/splits/{train,val,test_id,test_ood}.jsonl

Split strategy (theo Dataset Guide):
  train    : Stable Diffusion V1.4 / train / ai + nature
  val      : 80% của Stable Diffusion V1.4 / val / ai + nature
  test_id  : 20% còn lại của Stable Diffusion V1.4 / val
  test_ood : val của tất cả generator khác (+ một ít train nếu val nhỏ)
"""

import argparse
import io
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ID = "TheKernel01/Tiny-GenImage"

TARGET_FILES = [
    "data/train-00000-of-00014.parquet",
    "data/train-00001-of-00014.parquet",
    "data/train-00002-of-00014.parquet",
    "data/train-00003-of-00014.parquet",
    "data/train-00004-of-00014.parquet",
    "data/train-00005-of-00014.parquet",
    "data/validation-00000-of-00004.parquet",
    "data/validation-00001-of-00004.parquet",
]

# Generator chuẩn hoá → tên thư mục GenImage (giống dataset gốc)
GENERATOR_TO_FOLDER = {
    "stable_diffusion_v14": "Stable Diffusion V1.4",
    "stable_diffusion_v15": "Stable Diffusion V1.5",
    "midjourney":           "Midjourney",
    "biggan":               "BigGAN",
    "adm":                  "ADM",
    "glide":                "GLIDE",
    "vqdm":                 "VQDM",
    "wukong":               "Wukong",
    "real":                 "real",          # ảnh thật, không có generator riêng
}

# Mapping số nguyên → generator key  (theo dataset card)
#   generator field là ClassLabel: 0=Real,1=ADM,2=BigGAN,3=GLIDE,
#   4=Midjourney,5=SD14,6=SD15,7=VQDM,8=Wukong
INT_TO_KEY = {
    0: "real",
    1: "adm",
    2: "biggan",
    3: "glide",
    4: "midjourney",
    5: "stable_diffusion_v14",
    6: "stable_diffusion_v15",
    7: "vqdm",
    8: "wukong",
}

# Fallback: chuẩn hoá tên string → key
STR_TO_KEY = {
    "real":                  "real",
    "adm":                   "adm",
    "biggan":                "biggan",
    "glide":                 "glide",
    "midjourney":            "midjourney",
    "sd14":                  "stable_diffusion_v14",
    "stable diffusion v1.4": "stable_diffusion_v14",
    "stable_diffusion_v1.4": "stable_diffusion_v14",
    "sd15":                  "stable_diffusion_v15",
    "stable diffusion v1.5": "stable_diffusion_v15",
    "stable_diffusion_v1.5": "stable_diffusion_v15",
    "vqdm":                  "vqdm",
    "wukong":                "wukong",
}

TRAIN_GENERATOR = "stable_diffusion_v14"   # generator dùng cho train/val/test_id


def normalise_generator(raw) -> str:
    """Chấp nhận cả int lẫn string."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return INT_TO_KEY.get(int(raw), "unknown")
    key = str(raw).lower().strip()
    return STR_TO_KEY.get(key, key.replace(" ", "_"))


# ---------------------------------------------------------------------------
# Bước 1: Download parquet (skip nếu đã có)
# ---------------------------------------------------------------------------

def download_parquets(repo_id: str, parquet_dir: Path, hf_token: str | None) -> list[Path]:
    from huggingface_hub import hf_hub_download

    parquet_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    print(f"\n[1/3] Kiểm tra {len(TARGET_FILES)} file parquet:\n")
    for filename in TARGET_FILES:
        basename = Path(filename).name
        dest = parquet_dir / basename

        if dest.exists():
            mb = dest.stat().st_size / 1024 / 1024
            print(f"  SKIP  {basename}  ({mb:.0f} MB — đã có)")
            downloaded.append(dest)
            continue

        print(f"  TẢI   {basename} ...", end="", flush=True)
        kwargs = {"repo_id": repo_id, "repo_type": "dataset", "filename": filename}
        if hf_token:
            kwargs["token"] = hf_token
        try:
            local = hf_hub_download(**kwargs)   # dùng HF_HUB_CACHE đã set ở trên
            shutil.copy2(local, dest)
            mb = dest.stat().st_size / 1024 / 1024
            print(f" xong ({mb:.0f} MB)")
            downloaded.append(dest)
        except Exception as e:
            print(f" LỖI: {e}")

    return downloaded


# ---------------------------------------------------------------------------
# Bước 2: Giải nén ảnh → GenImage/<Generator Name>/<split>/{ai,nature}/
# ---------------------------------------------------------------------------

def process_parquets(parquet_files: list[Path], genimage_dir: Path) -> list[dict]:
    """
    Lưu ảnh theo đúng layout GenImage:
      genimage_dir/
        Stable Diffusion V1.4/train/ai/
        Stable Diffusion V1.4/train/nature/
        Stable Diffusion V1.4/val/ai/
        ...
        Midjourney/val/ai/
        ...
        real/train/nature/      (ảnh thật của SD1.4 split; cũng copy sang mỗi generator nếu cần)
    """
    import pandas as pd
    from PIL import Image

    records: list[dict] = []
    total_saved = 0

    print(f"\n[2/3] Giải nén ảnh vào {genimage_dir} ...\n")

    for pfile in parquet_files:
        fname = pfile.name
        hf_split = "train" if fname.startswith("train") else "val"

        print(f"  {fname}  (hf_split={hf_split}) ...", end="", flush=True)
        df = pd.read_parquet(pfile)
        saved_this = 0

        for i, row in df.iterrows():
            # --- Label ---
            # label là ClassLabel int: 0=real, 1=fake
            label_raw = row.get("label", row.get("class", 1))
            if label_raw in (0, "0", "real", "nature"):
                label  = "real"
                subdir = "nature"
            else:
                label  = "synthetic"
                subdir = "ai"

            # --- Generator key ---
            # generator là ClassLabel int: 0=Real,1=ADM,...,8=Wukong
            gen_raw = row.get("generator", row.get("source", row.get("model", None)))
            gen_key = normalise_generator(gen_raw) if gen_raw is not None else "unknown"
            if label == "real":
                gen_key = "real"

            # --- Tên thư mục ---
            folder_name = GENERATOR_TO_FOLDER.get(gen_key, gen_key)

            # --- Image ---
            img_data = row.get("image", row.get("img"))
            if img_data is None:
                continue
            try:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                elif hasattr(img_data, "save"):
                    img = img_data
                else:
                    img = Image.open(io.BytesIO(bytes(img_data)))
            except Exception:
                continue

            # --- Lưu ảnh ---
            # Layout: GenImage/<Generator Name>/<split>/{ai,nature}/
            dest_dir = genimage_dir / folder_name / hf_split / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            img_path = dest_dir / f"{fname.split('-')[0]}_{i:06d}.png"

            if not img_path.exists():
                img.save(str(img_path))
                saved_this += 1

            records.append({
                "image_path": str(img_path.resolve()),
                "label":     label,
                "generator": gen_key,
                "source":    "tiny_genimage",
                "hf_split":  hf_split,
                "folder":    folder_name,
            })

        total_saved += saved_this
        print(f" {len(df)} rows, {saved_this} ảnh mới lưu")

    print(f"\n  Tổng ảnh mới: {total_saved}  |  Tổng records: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Bước 3: Build splits theo Dataset Guide
# ---------------------------------------------------------------------------

def build_splits(
    records:        list[dict],
    splits_dir:     Path,
    train_gen:      str   = TRAIN_GENERATOR,
    test_id_frac:   float = 0.2,       # % của val SD1.4 dùng làm test_id
    ood_train_cap:  int   = 50,        # lấy tối đa N ảnh train của OOD gen nếu val nhỏ
    seed:           int   = 42,
) -> None:
    """
    train    : train/ai + train/nature của train_gen  (+ real/train)
    val      : (1-test_id_frac) × val của train_gen
    test_id  : test_id_frac × val của train_gen
    test_ood : val của mọi gen khác (+ tối đa ood_train_cap ảnh train nếu val nhỏ)

    Ảnh "real" (label=real) được phân theo hf_split, không theo generator.
    """
    rng = random.Random(seed)

    # Index theo (generator, hf_split, label)
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        idx[(r["generator"], r["hf_split"], r["label"])].append(r)

    def fetch(gen, split, label=None):
        if label:
            rows = list(idx.get((gen, split, label), []))
        else:
            rows = list(idx.get((gen, split, "real"), [])) + \
                   list(idx.get((gen, split, "synthetic"), []))
        rng.shuffle(rows)
        return rows

    # ---- train ----
    train_rows = (
        fetch(train_gen, "train", "synthetic") +
        fetch("real",    "train", "real")
    )
    rng.shuffle(train_rows)

    # ---- val + test_id (từ val SD1.4) ----
    sd_val = (
        fetch(train_gen, "val", "synthetic") +
        fetch("real",    "val", "real")
    )
    rng.shuffle(sd_val)
    cut        = int(len(sd_val) * (1 - test_id_frac))
    val_rows     = sd_val[:cut]
    test_id_rows = sd_val[cut:]

    # ---- test_ood (tất cả generator khác) ----
    all_gen_keys = {r["generator"] for r in records} - {"real", train_gen}
    test_ood_rows: list[dict] = []
    for gen in sorted(all_gen_keys):
        ood_val = fetch(gen, "val")
        test_ood_rows.extend(ood_val)
        # Nếu val nhỏ, bổ sung thêm từ train
        if len(ood_val) < ood_train_cap:
            extra = fetch(gen, "train")[:ood_train_cap - len(ood_val)]
            test_ood_rows.extend(extra)
        # Cũng lấy real tương ứng theo hf_split
    # Thêm real/val cho OOD (đã lấy real/train cho train, real/val dùng trong val+test_id)
    rng.shuffle(test_ood_rows)

    # ---- Ghi file ----
    splits_dir.mkdir(parents=True, exist_ok=True)
    keep_keys = {"image_path", "label", "generator", "source"}

    print(f"\n[3/3] Ghi split files vào {splits_dir}:\n")
    for name, rows in [
        ("train",    train_rows),
        ("val",      val_rows),
        ("test_id",  test_id_rows),
        ("test_ood", test_ood_rows),
    ]:
        out = splits_dir / f"{name}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k in keep_keys}) + "\n")

        # Thống kê nhanh
        n_real = sum(1 for r in rows if r["label"] == "real")
        n_fake = sum(1 for r in rows if r["label"] == "synthetic")
        gens   = sorted({r["generator"] for r in rows if r["generator"] != "real"})
        print(f"  {name}.jsonl  →  {len(rows):>5} rows  "
              f"(real={n_real}, fake={n_fake})")
        if gens:
            print(f"    generators: {', '.join(gens)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Tiny-GenImage và build splits theo Dataset Guide."
    )
    parser.add_argument("--repo-id",    default=REPO_ID)
    parser.add_argument("--data-dir",   default=str(Path(__file__).parent),
                        help="Root data/ directory (default: thư mục chứa script)")
    parser.add_argument("--hf-token",   default=None,
                        help="HuggingFace token (hoặc set HF_TOKEN env var)")
    parser.add_argument("--train-gen",  default=TRAIN_GENERATOR,
                        help="Generator dùng cho train/val/test_id (default: stable_diffusion_v14)")
    parser.add_argument("--test-id-frac", type=float, default=0.2,
                        help="Tỉ lệ val của train_gen dùng làm test_id (default: 0.2)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--hf-cache",   default="/workspace/cache",
                        help="Thư mục cache HuggingFace (default: /workspace/cache)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Bỏ qua giải nén ảnh, chỉ build splits từ metadata đã có")
    args = parser.parse_args()

    hf_token    = args.hf_token or os.environ.get("HF_TOKEN")

    # Chuyển cache HF sang /workspace/cache để tránh đầy root
    hf_cache = Path(args.hf_cache).resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]      = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    print(f"  hf_cache    : {hf_cache}")

    data_dir    = Path(args.data_dir).resolve()
    parquet_dir = data_dir / "raw" / "_parquet"
    genimage_dir = data_dir / "raw" / "GenImage"
    meta_dir    = data_dir / "metadata"
    splits_dir  = data_dir / "splits"
    meta_path   = meta_dir / "metadata.jsonl"

    print("=" * 60)
    print(f"  data_dir    : {data_dir}")
    print(f"  train_gen   : {args.train_gen}")
    print(f"  test_id_frac: {args.test_id_frac}")
    print(f"  seed        : {args.seed}")
    print("=" * 60)

    # ---- Bước 1 ----
    parquet_files = download_parquets(args.repo_id, parquet_dir, hf_token)
    if not parquet_files:
        print("[error] Không tải được file nào. Kiểm tra kết nối hoặc HF_TOKEN.")
        return

    # ---- Bước 2 ----
    if args.skip_extract and meta_path.exists():
        print(f"\n[2/3] skip-extract — đọc metadata từ {meta_path} ...")
        records = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
        # Thêm lại hf_split từ image_path nếu thiếu
        for r in records:
            if "hf_split" not in r:
                p = r["image_path"].lower()
                r["hf_split"] = "val" if "/val/" in p else "train"
        print(f"  {len(records)} records đọc xong.")
    else:
        meta_dir.mkdir(parents=True, exist_ok=True)
        records = process_parquets(parquet_files, genimage_dir)
        with open(meta_path, "w") as f:
            for r in records:
                entry = {k: v for k, v in r.items() if k != "hf_split"}
                f.write(json.dumps(entry) + "\n")
        print(f"\n  metadata.jsonl → {len(records)} rows")

    if not records:
        print("[error] Không có ảnh nào. Kiểm tra lại parquet files.")
        return

    # ---- Bước 3 ----
    build_splits(
        records,
        splits_dir,
        train_gen=args.train_gen,
        test_id_frac=args.test_id_frac,
        seed=args.seed,
    )

    print("\n✓ Xong! Layout kết quả:")
    print(f"  {genimage_dir}/")
    print(f"    <Generator Name>/train/{{ai,nature}}/")
    print(f"    <Generator Name>/val/{{ai,nature}}/")
    print(f"  {splits_dir}/train.jsonl  val.jsonl  test_id.jsonl  test_ood.jsonl")


if __name__ == "__main__":
    main()