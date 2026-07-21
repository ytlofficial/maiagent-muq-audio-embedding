#!/usr/bin/env python3
"""Build a local SQLite database from maimai maidata chart folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path


DIFFICULTIES = {
    5: "Master",
    6: "Re:Master",
}

FIELD_RE = re.compile(r"^&([^=\s]+)=(.*)$")
CHART_VARIANT_RE = re.compile(r"\s+\[(DX|ST)\]$")
CHART_VERSION_RE = re.compile(r"^(\d+)\.")

CHARTER_ALIASES = [
    ["はっぴー", "緑風 犬三郎", "緑風犬三郎", "原田ひろゆき", "はっぴー(INFP)"],
    ["Jack", "JAQ"],
    ["譜面-100号"],
    ["チャン＠DP皆伝", "チャン@DP皆伝"],
    ["某S氏"],
    ["ロシェ＠ペンギン", "ロシェ@ペンギン"],
    ["Techno Kitchen"],
    ["ぴちネコ", "ロシアンブラック", "Pizzicat", "CODE:CatastroPhe", "Panther", "MSИ-00100"],
    ["Moon Strix"],
    ["玉子豆腐"],
    ["小鳥遊さん", "Phoenix", "小鳥遊さん fused with Phoenix", "譜面男子学院 中堅 小鳥 遊"],
    ["ものくろっく", "一ノ瀬 リズ"],
    ["すきやき奉行"],
    ["サファ太", "-ZONE- SaFaRi", "Safari", "さふぁた", "Saf", "モリモリさふぁた", "DANCE TIME(サファ太)", "ZONE:SaFaRi", "サファ太 vs -ZONE- SaFaRi"],
    ["華火職人"],
    ["シチミヘルツ", "7.3Hz", "7.3GHz", "7.3GHz -Før The Legends-", "KOP7th -FiNAL BATTLE- by 7.3GHz"],
    ["うさぎランドリー"],
    ["アマリリス", "アマリリスせんせえ"],
    ["群青リコリス"],
    ["隅田川星人", "The ALiEN"],
    ["アミノハバキリ"],
    ["Redarrow"],
    ["翠楼屋", "作譜：翠楼屋", "翡翠マナ", "翡翠マナ（推测）", "KOP3rd with 翡翠マナ", "翡翠マナ -Memoir-"],
    ["あまくちジンジャー", "EL DiABLO", "あまくちジンジャー＠やべー新人"],
    ["じゃこレモン", "僕の檸檬本当上手"],
    ["カマボコ君"],
    ["メロンポップ", "ずんだポップ"],
    ["みそかつ侍"],
    ["鳩ホルダー", "The Dove", "The Dove（推测）", "7sRef -DOVE-"],
    ["rintaro soma"],
    ["Luxizhel", "BELiZHEL", "るしえる"],
    ["Ruby"],
    ["PG-NAKAGAWA"],
    ["ミニミライト", "Twinrook"],
    ["りんご Full Set"],
    ["きょむりん"],
    ["まぐランド"],
    ["せめんともり"],
    ["mai-Star"],
    ["ニャイン"],
    ["rioN"],
    ["Revo@LC"],
    ["LabiLabi"],
    ["未署名"],
]

COLLABORATION_CREDITS = {
    "三枝明那とはっぴー": ["三枝明那", "はっぴー"],
    "Luxizhel & Jack": ["Luxizhel", "Jack"],
    "“H”ack": ["Jack", "小鳥遊さん"],
    "“H”ack underground": ["Jack", "小鳥遊さん"],
    "いぬっくまとボコっくま": ["はっぴー／緑風 犬三郎", "カマボコ君"],
    "アマリリスせんせえ with ぴちネコせんせえ": ["アマリリス", "ぴちネコ"],
    "サぴぴぴぴちネファ太太太太コ": ["サファ太", "ぴちネコ"],
    "せめんともり & カマボコ君": ["せめんともり", "カマボコ君"],
    "Safata.Hz": ["サファ太", "シチミヘルツ"],
    "サファ太 vs 翠楼屋": ["サファ太", "翠楼屋"],
    "鳩ホルぴー": ["鳩ホルダー", "はっぴー"],
    "Luxiいぬ": ["Luxizhel", "はっぴー／緑風 犬三郎"],
    "みぞれヤナギ&サファ太": ["みぞれヤナギ", "サファ太"],
    "Jack&アマリリス": ["Jack", "アマリリス"],
    "Luxizhel＆はっぴー": ["Luxizhel", "はっぴー"],
    "カマボコホルダー": ["カマボコ君", "鳩ホルダー"],
    "きょむりん＆はっぴー": ["きょむりん", "はっぴー"],
    "7.3Hz＋Jack": ["シチミヘルツ", "Jack"],
    "譜面-100号とはっぴー": ["譜面-100号", "はっぴー"],
    "ゲキ*チュウマイ Fumen Team": [],
    "七味星人": ["シチミヘルツ", "隅田川星人"],
    "しちみりこりす": ["シチミヘルツ", "群青リコリス"],
    "チェシャ猫とハートのジャック": ["ぴちネコ", "Jack"],
    "はぴネコ(はっぴー&ぴちネコ)": ["はっぴー", "ぴちネコ"],
    "Jack vs あまくちジンジャー": ["Jack", "あまくちジンジャー"],
    "チャン＠DP皆伝 vs シチミヘルツ": ["チャン＠DP皆伝", "シチミヘルツ"],
    "チャン@DP皆伝 vs シチミヘルツ": ["チャン＠DP皆伝", "シチミヘルツ"],
    "サファ太&ぴちネコ": ["サファ太", "ぴちネコ"],
    "Jack vs サファ太": ["Jack", "サファ太"],
    "鳩ホルダー＆Luxizhel": ["鳩ホルダー", "Luxizhel"],
    "Luxizhel vs サファ太": ["Luxizhel", "サファ太"],
    "Safazhel": ["サファ太", "Luxizhel"],
    "jacK on Phoenix": ["Jack", "小鳥遊さん"],
    "シチミッピー": ["シチミヘルツ", "はっぴー"],
    "7.3GHz vs Phoenix": ["シチミヘルツ", "小鳥遊さん"],
    "Sukiyaki vs Happy": ["すきやき奉行", "はっぴー"],
    "ﾚよ†ょ／Ｕヽ” ┠ (十,3、了ﾅﾆ": ["華火職人", "サファ太"],
    "隅田川華火大会": ["隅田川星人", "華火職人"],
    "7.3連発華火": ["シチミヘルツ", "華火職人"],
    "SHICHIMI☆CAT": ["シチミヘルツ", "ぴちネコ"],
    "-ZONE-Phoenix": ["サファ太", "小鳥遊さん"],
    "Jack & Licorice Gunjyo": ["Jack", "群青リコリス"],
    "超七味星人": ["シチミヘルツ", "隅田川星人"],
    "はっぴー星人": ["はっぴー", "隅田川星人"],
    "Jack & はっぴー vs からめる & ぐるん": ["Jack", "はっぴー", "からめる", "ぐるん"],
    "maimai Fumen All-Stars": [],
    "ネコトリサーカス団": [],
    "red phoenix": ["Redarrow", "小鳥遊さん"],
    "Redarrow VS 翠楼屋": ["Redarrow", "翠楼屋"],
    "舞舞10年ズ ～ファイナル～": ["チャン＠DP皆伝", "はっぴー"],
    "サファ太 vs じゃこレモン": ["サファ太", "じゃこレモン"],
    "ボコ太": ["カマボコ君", "サファ太"],
    "jacK on Phoenix & -ZONE- SaFaRi": ["Jack", "小鳥遊さん", "サファ太"],
    "jacK on Phoenix vs -ZONE- SaFaRi": ["Jack", "小鳥遊さん", "サファ太"],
    "はっぴー & サファ太": ["はっぴー", "サファ太"],
    "鳩ホルダー & Luxizhel": ["鳩ホルダー", "Luxizhel"],
    "Safata.GHz": ["サファ太", "シチミヘルツ"],
    "鳩サファzhel": ["鳩ホルダー", "サファ太", "Luxizhel"],
    "maimai TEAM DX": [],
    "R-blacX of JacQ": ["ぴちネコ／ロシアンブラック", "Jack"],
    "みぞれヤナギ＆Jack": ["みぞれヤナギ", "Jack"],
    "BELiZHEL vs 7.3GHz": ["Luxizhel", "シチミヘルツ"],
    "サファ太 vs Luxizhel": ["サファ太", "Luxizhel"],
    "BELiZHEL vs Safari": ["Luxizhel", "サファ太"],
    "SAFARI☆CAT": ["サファ太", "ぴちネコ"],
    "SΛFΛRI/RΦCHER": ["サファ太", "ロシェ＠ペンギン"],
    "きょむりん vs Luxizhel": ["きょむりん", "Luxizhel"],
    "Jack + Soma": ["Jack", "rintaro soma"],
    "Luxizhel+カマボコ君+はっぴー": ["Luxizhel", "カマボコ君", "はっぴー"],
    "The ALiEN vs. Phoenix": ["隅田川星人", "小鳥遊さん"],
    "小鳥遊さん vs 華火職人": ["小鳥遊さん", "華火職人"],
    "小鳥遊チミ": ["小鳥遊さん", "シチミヘルツ"],
    "Hz-R.Arrow": ["シチミヘルツ", "Redarrow"],
    "翠楼屋 vs あまくちジンジャー": ["翠楼屋", "あまくちジンジャー"],
    "小鳥遊さん×アミノハバキリ": ["小鳥遊さん", "アミノハバキリ"],
    "たかなっぴー": ["小鳥遊さん", "はっぴー"],
    "あまくちヘルツ": ["あまくちジンジャー", "シチミヘルツ"],
    "譜面ボーイズからの挑戦状": [],
    "Twinrook & Safari": ["ミニミライト", "サファ太"],
    "サファ太 ＆ 鳩ホルダー": ["サファ太", "鳩ホルダー"],
    "メロンポップ vs Luxizhel": ["メロンポップ", "Luxizhel"],
    "Luxizhel & 鳩ホルダー": ["Luxizhel", "鳩ホルダー"],
    "サファ太＆メロンホップ": ["サファ太", "メロンポップ"],
    "舞舞10年ズ (チャンとはっぴー)": ["チャン＠DP皆伝", "はっぴー"],
    "舞舞10年ズ 〜ファイナル〜": ["チャン＠DP皆伝", "はっぴー"],
    "ものくロシェ": ["ものくろっく", "ロシェ＠ペンギン"],
    "safaTAmago": ["サファ太", "玉子豆腐"],
    "はっぴー respects for 某S氏": ["はっぴー", "某S氏"],
    "ﾚよ†ょ／∪ヽ”┠  (十,3､了ﾅﾆ": ["華火職人", "サファ太"],
    "合作だよ": [],
    "maimai TEAM": [],
    "しろいろ": [],
    "如月 ゆかり": [],
    "Garakuta Scramble!": [],
    "PANDORA BOXXX": [],
    "PANDORA PARADOXXX": [],
    "“Carpe diem” ＊ HAN∀BI": [],
    "みんなでマイマイマー": [],
    "畳返し": [],
}

UNKNOWN_CHARTER_CREDITS = {
    "BEYOND THE MEMORIES",
    "Xaleid◆scopiX",
    "Anomaly Labyrinth",
    "廻屋捗",
    "Starlight Disco Festa",
    "project_raputa",
    "BLaCK rOSE dIsEASe pATENT",
    "KALEIDXSCOPE",
}

ALIAS_TO_MAIN = {
    alias: aliases[0]
    for aliases in CHARTER_ALIASES
    for alias in aliases
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build songs and charts tables from chartdata maidata files."
    )
    parser.add_argument(
        "--chartdata",
        default="chartdata-rebuilt",
        type=Path,
        help="Directory containing one folder per chart file. Default: chartdata-rebuilt",
    )
    parser.add_argument(
        "--db",
        default="maimai_charts.db",
        type=Path,
        help="SQLite database path to create. Default: maimai_charts.db",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop and recreate songs/charts before importing.",
    )
    parser.add_argument(
        "--on-duplicate-music-id",
        choices=("error", "skip", "suffix"),
        default="error",
        help=(
            "How to handle multiple folders with the same music_id. "
            "Default: error"
        ),
    )
    return parser.parse_args()


def parse_maidata(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = FIELD_RE.match(line)
        if match:
            if current_key is not None:
                fields[current_key] = "\n".join(current_lines).strip()
            current_key = match.group(1)
            current_lines = [match.group(2)]
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        fields[current_key] = "\n".join(current_lines).strip()

    return fields


def first_existing_chart_file(song_dir: Path) -> Path | None:
    for name in ("maidata.txt", "majdata.txt"):
        path = song_dir / name
        if path.is_file():
            return path
    return None


def stable_music_id(fields: dict[str, str], song_dir: Path) -> str:
    short_id = fields.get("shortid", "").strip()
    if short_id:
        return short_id

    title = fields.get("title", "").strip() or song_dir.name
    digest = hashlib.sha1(f"{title}\n{song_dir.name}".encode("utf-8")).hexdigest()
    return f"local_{digest[:12]}"


def suffixed_music_id(music_id: str, song_dir: Path) -> str:
    digest = hashlib.sha1(str(song_dir).encode("utf-8")).hexdigest()
    return f"{music_id}__{digest[:8]}"


def content_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def normalize_display(value: str | None) -> str:
    return unicodedata.normalize("NFC", (value or "").strip())


def canonical_song_title(title: str) -> str:
    return CHART_VARIANT_RE.sub("", normalize_display(title))


def canonical_artist_key(artist: str | None) -> str:
    value = normalize_display(artist)
    value = value.replace("＋", "+").replace("＆", "&")
    return re.sub(r"[\s+&・/／]+", "", value).lower()


def chart_kind(fields: dict[str, str], folder_name: str) -> str:
    title = normalize_display(fields.get("title") or folder_name)
    folder = normalize_display(folder_name)
    if title.endswith("[ST]") or folder.endswith("[ST]"):
        return "ST"
    if title.endswith("[DX]") or folder.endswith("[DX]"):
        return "DX"
    if normalize_display(fields.get("cabinet")).upper() == "DX":
        return "DX"
    if chart_version_sort_key(fields.get("chart_version"))[0] >= 14:
        return "DX"
    return "ST"


def chart_version_sort_key(version: str | None) -> tuple[int, str]:
    version = normalize_display(version)
    match = CHART_VERSION_RE.match(version)
    if match:
        return int(match.group(1)), version
    return 9999, version


def song_group_key(fields: dict[str, str], folder_name: str) -> tuple[str, str]:
    title = canonical_song_title(fields.get("title") or folder_name)
    artist_key = canonical_artist_key(fields.get("artist"))
    return title, artist_key


def canonicalize_participant(name: str) -> str:
    name = name.strip()
    if "／" in name:
        name = name.split("／", 1)[0].strip()
    return ALIAS_TO_MAIN.get(name, name)


def classify_charter(charter: str | None) -> tuple[str | None, str]:
    if not charter:
        return "个人谱师", json.dumps(["未署名"], ensure_ascii=False)

    charter = charter.strip()
    if charter in ALIAS_TO_MAIN:
        category = "个人谱师"
        main_names = [ALIAS_TO_MAIN[charter]]
    elif charter in COLLABORATION_CREDITS:
        category = "多人合作谱"
        participants = COLLABORATION_CREDITS[charter]
        main_names = [canonicalize_participant(name) for name in participants]
        if not main_names:
            main_names = [charter]
    elif charter in UNKNOWN_CHARTER_CREDITS:
        category = "多人合作谱"
        main_names = [charter]
    else:
        category = "未识别"
        main_names = [charter]

    return category, json.dumps(main_names, ensure_ascii=False)


def create_schema(conn: sqlite3.Connection, replace: bool) -> None:
    if replace:
        conn.execute("DROP TABLE IF EXISTS charts")
        conn.execute("DROP TABLE IF EXISTS songs")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT,
            artist_id TEXT,
            bpm TEXT,
            genre TEXT,
            cabinet TEXT,
            version TEXT,
            source_folders TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS charts (
            song_id INTEGER NOT NULL,
            chart_kind TEXT NOT NULL,
            chart_version TEXT NOT NULL,
            difficulty_index INTEGER NOT NULL,
            difficulty_name TEXT NOT NULL,
            level TEXT,
            charter TEXT,
            charter_category TEXT,
            charter_main_names TEXT NOT NULL,
            chart_content TEXT NOT NULL,
            has_chart INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            created_from_file TEXT NOT NULL,
            PRIMARY KEY (song_id, chart_kind, difficulty_index),
            FOREIGN KEY (song_id) REFERENCES songs (song_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_title ON songs (title)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_version ON songs (version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_charts_chart_version ON charts (chart_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_charts_kind ON charts (chart_kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_charts_level ON charts (level)")


def collect_records(chartdata: Path) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    skipped_dirs: list[str] = []

    for song_dir in sorted(p for p in chartdata.iterdir() if p.is_dir()):
        maidata_path = first_existing_chart_file(song_dir)
        if maidata_path is None:
            skipped_dirs.append(str(song_dir))
            continue

        fields = parse_maidata(maidata_path)
        title = normalize_display(fields.get("title") or song_dir.name)
        base_title = canonical_song_title(title)
        version = normalize_display(fields.get("chart_version"))
        if not version:
            version = normalize_display(fields.get("version")) or "未标注"

        records.append(
            {
                "fields": fields,
                "path": maidata_path,
                "folder_name": song_dir.name,
                "song_key": song_group_key(fields, song_dir.name),
                "base_title": base_title,
                "artist": normalize_display(fields.get("artist")) or None,
                "artist_id": normalize_display(fields.get("artistid")) or None,
                "bpm": normalize_display(fields.get("wholebpm")) or None,
                "genre": normalize_display(fields.get("genre")) or None,
                "cabinet": normalize_display(fields.get("cabinet")) or None,
                "chart_kind": chart_kind(fields, song_dir.name),
                "chart_version": version,
            }
        )

    return records, skipped_dirs


def import_records(conn: sqlite3.Connection, records: list[dict[str, object]]) -> tuple[int, int]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(record["song_key"], []).append(record)  # type: ignore[index]

    song_groups = sorted(
        grouped.values(),
        key=lambda rows: (
            min(chart_version_sort_key(str(row["chart_version"])) for row in rows),
            str(rows[0]["base_title"]),
            str(rows[0].get("artist") or ""),
        ),
    )

    imported_songs = 0
    imported_charts = 0

    for song_id, rows in enumerate(song_groups, start=1):
        rows = sorted(
            rows,
            key=lambda row: (
                chart_version_sort_key(str(row["chart_version"])),
                str(row["chart_kind"]),
                str(row["folder_name"]),
            ),
        )
        first = rows[0]
        earliest_version = min(
            (str(row["chart_version"]) for row in rows),
            key=chart_version_sort_key,
        )
        source_folders = json.dumps(
            [str(row["folder_name"]) for row in rows],
            ensure_ascii=False,
        )

        conn.execute(
            """
            INSERT INTO songs (
                song_id, title, artist, artist_id, bpm, genre, cabinet, version,
                source_folders
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                song_id,
                first["base_title"],
                first.get("artist"),
                first.get("artist_id"),
                first.get("bpm"),
                first.get("genre"),
                first.get("cabinet"),
                earliest_version,
                source_folders,
            ),
        )
        imported_songs += 1

        seen_chart_keys: set[tuple[str, int]] = set()
        for row in rows:
            fields = row["fields"]  # type: ignore[assignment]
            default_charter = normalize_display(fields.get("des"))  # type: ignore[union-attr]
            for difficulty_index, difficulty_name in DIFFICULTIES.items():
                level = normalize_display(fields.get(f"lv_{difficulty_index}"))  # type: ignore[union-attr]
                chart_content = fields.get(f"inote_{difficulty_index}", "").strip()  # type: ignore[union-attr]
                if not level and not chart_content:
                    continue

                chart_key = (str(row["chart_kind"]), difficulty_index)
                if chart_key in seen_chart_keys:
                    raise SystemExit(
                        "duplicate chart key for song "
                        f"{song_id} {first['base_title']}: {chart_key}"
                    )
                seen_chart_keys.add(chart_key)

                charter = normalize_display(fields.get(f"des_{difficulty_index}")) or default_charter  # type: ignore[union-attr]
                charter_category, charter_main_names = classify_charter(charter)
                conn.execute(
                    """
                    INSERT INTO charts (
                        song_id, chart_kind, chart_version, difficulty_index,
                        difficulty_name, level, charter, charter_category,
                        charter_main_names, chart_content, has_chart, content_hash,
                        created_from_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        song_id,
                        row["chart_kind"],
                        row["chart_version"],
                        difficulty_index,
                        difficulty_name,
                        level or None,
                        charter or None,
                        charter_category,
                        charter_main_names,
                        chart_content,
                        1 if chart_content else 0,
                        content_hash(chart_content),
                        str(row["path"]),
                    ),
                )
                imported_charts += 1

    return imported_songs, imported_charts


def main() -> None:
    args = parse_args()
    chartdata = args.chartdata
    if not chartdata.is_dir():
        raise SystemExit(f"chartdata directory not found: {chartdata}")

    args.db.parent.mkdir(parents=True, exist_ok=True)

    records, skipped_dirs = collect_records(chartdata)

    with sqlite3.connect(args.db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn, args.replace)
        imported_songs, imported_charts = import_records(conn, records)
        conn.commit()

    print(f"database={args.db}")
    print(f"source_chart_files={len(records)}")
    print(f"imported_songs={imported_songs}")
    print(f"imported_charts={imported_charts}")
    print(f"skipped_dirs_without_maidata={len(skipped_dirs)}")
    if skipped_dirs:
        print("first_skipped_dirs:")
        for skipped in skipped_dirs[:10]:
            print(f"  {skipped}")


if __name__ == "__main__":
    main()
