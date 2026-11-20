import argparse
from pathlib import Path


def tambah_angka_ke_file(
    folder_path: str = ".",
    start: int = 1,
    padding: int = 3,
    dry_run: bool = False,
    auto_confirm: bool = False,
) -> None:
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        print(f"[ERROR] Folder tidak ditemukan: {folder}")
        return

    self_path = Path(__file__).resolve()
    files = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.resolve() != self_path),
        key=lambda x: x.name.lower(),
    )

    if not files:
        print(f"Tidak ada file untuk diubah di: {folder}")
        return

    rencana = []
    for index, file in enumerate(files, start=start):
        new_name = f"{index:0{padding}d}_{file.name}"
        target = file.with_name(new_name)
        rencana.append((file, target))

    print(f"Folder: {folder}\n")
    for file, target in rencana:
        print(f"  {file.name} -> {target.name}")

    if dry_run:
        print("\n[DRY-RUN] Tidak ada file yang diubah.")
        return

    if not auto_confirm:
        jawab = input(f"\nLanjutkan mengubah {len(rencana)} file? (y/n): ").strip().lower()
        if jawab not in ("y", "ya", "yes"):
            print("Dibatalkan.")
            return

    print()
    for file, target in rencana:
        if target.exists():
            print(f"[LEWATI] {target.name} sudah ada.")
        else:
            file.rename(target)
            print(f"[OK] {file.name} -> {target.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Menambahkan nomor urut di awal nama semua file dalam folder.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Path folder target (default: folder saat ini)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Angka awal penomoran (default: 1)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=3,
        help="Jumlah digit angka dengan nol di depan (default: 3 -> 001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya tampilkan preview tanpa mengubah nama file",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Langsung eksekusi tanpa konfirmasi",
    )

    args = parser.parse_args()
    tambah_angka_ke_file(
        folder_path=args.folder,
        start=args.start,
        padding=args.padding,
        dry_run=args.dry_run,
        auto_confirm=args.yes,
    )
