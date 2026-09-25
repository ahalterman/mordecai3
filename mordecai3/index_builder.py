"""
Build the GeoNames Elasticsearch index that Mordecai resolves places against.

Run it through the CLI: `mordecai3 index build` (see `mordecai3 index --help`).

Ported from NGEC's `elasticsearch/es_geonames/load_geonames_es.py`, which in
turn descends from openeventdata/es-geonames. The `documents()` generator is
unchanged, so the documents are field-for-field what those tools produce and an
index built here is interchangeable with the prebuilt download. Differences:

- The cluster's disk watermarks are relaxed *before* the index is created. On a
  disk more than 95% full, Elasticsearch otherwise marks the new index
  read-only the moment it exists and the load dies with a 429.
- Accent stripping is inlined (textacy's `accents(fast=False)`, verbatim), so
  building an index needs nothing beyond mordecai3's own dependencies.
- Only the `geonames` index is ever deleted or created, so it is safe to point
  at a node that also holds other indices (NGEC's `wiki`, for one).
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import unicodedata
import zipfile
from datetime import date, datetime
from importlib.resources import files
from pathlib import Path
from urllib.request import urlretrieve

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

INDEX = "geonames"
GEONAMES_BASE = "https://download.geonames.org/export/dump"
GAZETTEER_FILES = ["allCountries.zip", "admin1CodesASCII.txt", "admin2Codes.txt"]

csv.field_size_limit()


def remove_accents(text):
    """textacy.preprocessing.remove.accents(text, fast=False), inlined."""
    return "".join(char for char in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(char))


# ISO alpha-2 -> alpha-3, verbatim from es-geonames so the documents match.
ISO3 = {"AD":"AND", "AE":"ARE", "AF":"AFG", "AG":"ATG", "AI":"AIA",
            "AL":"ALB", "AM":"ARM", "AO":"AGO", "AQ":"ATA", "AR":"ARG",
            "AS":"ASM", "AT":"AUT", "AU":"AUS", "AW":"ABW", "AX":"ALA",
            "AZ":"AZE", "BA":"BIH", "BB":"BRB", "BD":"BGD", "BE":"BEL",
            "BF":"BFA", "BG":"BGR", "BH":"BHR", "BI":"BDI", "BJ":"BEN",
            "BL":"BLM", "BM":"BMU", "BN":"BRN", "BO":"BOL", "BQ":"BES",
            "BR":"BRA", "BS":"BHS", "BT":"BTN", "BV":"BVT", "BW":"BWA",
            "BY":"BLR", "BZ":"BLZ", "CA":"CAN", "CC":"CCK", "CD":"COD",
            "CF":"CAF", "CG":"COG", "CH":"CHE", "CI":"CIV", "CK":"COK",
            "CL":"CHL", "CM":"CMR", "CN":"CHN", "CO":"COL", "CR":"CRI",
            "CU":"CUB", "CV":"CPV", "CW":"CUW", "CX":"CXR", "CY":"CYP",
            "CZ":"CZE", "DE":"DEU", "DJ":"DJI", "DK":"DNK", "DM":"DMA",
            "DO":"DOM", "DZ":"DZA", "EC":"ECU", "EE":"EST", "EG":"EGY",
            "EH":"ESH", "ER":"ERI", "ES":"ESP", "ET":"ETH", "FI":"FIN",
            "FJ":"FJI", "FK":"FLK", "FM":"FSM", "FO":"FRO", "FR":"FRA",
            "GA":"GAB", "GB":"GBR", "GD":"GRD", "GE":"GEO", "GF":"GUF",
            "GG":"GGY", "GH":"GHA", "GI":"GIB", "GL":"GRL", "GM":"GMB",
            "GN":"GIN", "GP":"GLP", "GQ":"GNQ", "GR":"GRC", "GS":"SGS",
            "GT":"GTM", "GU":"GUM", "GW":"GNB", "GY":"GUY", "HK":"HKG",
            "HM":"HMD", "HN":"HND", "HR":"HRV", "HT":"HTI", "HU":"HUN",
            "ID":"IDN", "IE":"IRL", "IL":"ISR", "IM":"IMN", "IN":"IND",
            "IO":"IOT", "IQ":"IRQ", "IR":"IRN", "IS":"ISL", "IT":"ITA",
            "JE":"JEY", "JM":"JAM", "JO":"JOR", "JP":"JPN", "KE":"KEN",
            "KG":"KGZ", "KH":"KHM", "KI":"KIR", "KM":"COM", "KN":"KNA",
            "KP":"PRK", "KR":"KOR", "XK":"XKX", "KW":"KWT", "KY":"CYM",
            "KZ":"KAZ", "LA":"LAO", "LB":"LBN", "LC":"LCA", "LI":"LIE",
            "LK":"LKA", "LR":"LBR", "LS":"LSO", "LT":"LTU", "LU":"LUX",
            "LV":"LVA", "LY":"LBY", "MA":"MAR", "MC":"MCO", "MD":"MDA",
            "ME":"MNE", "MF":"MAF", "MG":"MDG", "MH":"MHL", "MK":"MKD",
            "ML":"MLI", "MM":"MMR", "MN":"MNG", "MO":"MAC", "MP":"MNP",
            "MQ":"MTQ", "MR":"MRT", "MS":"MSR", "MT":"MLT", "MU":"MUS",
            "MV":"MDV", "MW":"MWI", "MX":"MEX", "MY":"MYS", "MZ":"MOZ",
            "NA":"NAM", "NC":"NCL", "NE":"NER", "NF":"NFK", "NG":"NGA",
            "NI":"NIC", "NL":"NLD", "NO":"NOR", "NP":"NPL", "NR":"NRU",
            "NU":"NIU", "NZ":"NZL", "OM":"OMN", "PA":"PAN", "PE":"PER",
            "PF":"PYF", "PG":"PNG", "PH":"PHL", "PK":"PAK", "PL":"POL",
            "PM":"SPM", "PN":"PCN", "PR":"PRI", "PS":"PSE", "PT":"PRT",
            "PW":"PLW", "PY":"PRY", "QA":"QAT", "RE":"REU", "RO":"ROU",
            "RS":"SRB", "RU":"RUS", "RW":"RWA", "SA":"SAU", "SB":"SLB",
            "SC":"SYC", "SD":"SDN", "SS":"SSD", "SE":"SWE", "SG":"SGP",
            "SH":"SHN", "SI":"SVN", "SJ":"SJM", "SK":"SVK", "SL":"SLE",
            "SM":"SMR", "SN":"SEN", "SO":"SOM", "SR":"SUR", "ST":"STP",
            "SV":"SLV", "SX":"SXM", "SY":"SYR", "SZ":"SWZ", "TC":"TCA",
            "TD":"TCD", "TF":"ATF", "TG":"TGO", "TH":"THA", "TJ":"TJK",
            "TK":"TKL", "TL":"TLS", "TM":"TKM", "TN":"TUN", "TO":"TON",
            "TR":"TUR", "TT":"TTO", "TV":"TUV", "TW":"TWN", "TZ":"TZA",
            "UA":"UKR", "UG":"UGA", "UM":"UMI", "US":"USA", "UY":"URY",
            "UZ":"UZB", "VA":"VAT", "VC":"VCT", "VE":"VEN", "VG":"VGB",
            "VI":"VIR", "VN":"VNM", "VU":"VUT", "WF":"WLF", "WS":"WSM",
            "YE":"YEM", "YT":"MYT", "ZA":"ZAF", "ZM":"ZMB", "ZW":"ZWE",
            "CS":"SCG", "AN":"ANT"}


def read_codes(path):
    """admin1CodesASCII.txt / admin2Codes.txt -> {code: name}."""
    with open(path, encoding="utf-8") as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t")}


def documents(reader, adm1_dict, adm2_dict, expand_ascii=True, bad_codes=None):
    """Yield one bulk action per GeoNames row.

    With `expand_ascii`, every alternative name with accents also gets an
    accent-stripped copy ("Ḩadīqat ash Shahbā" -> "Hadiqat ash Shahba");
    non-Latin scripts are left alone.
    """
    if bad_codes is None:
        bad_codes = set()
    todays_date = datetime.today().strftime("%Y-%m-%d")
    adm2_missing = 0   # benign: admin2 code has no name; row still indexed
    row_errors = 0     # real: row failed to parse and was skipped
    for row in tqdm(reader, total=13_300_000, unit=" rows"):
        try:
            coords = row[4] + "," + row[5]
            country_code3 = ISO3.get(row[8])
            if country_code3 is None:
                bad_codes.add(row[8])
                country_code3 = "NA"
            alt_names = list(set(row[3].split(",")))
            if str(row[0]) == "6252001":
                alt_names.append("US")
                alt_names.append("U.S.")
            if str(row[0]) == "239880":
                alt_names.append("C.A.R.")
            alt_name_length = len(alt_names)
            if expand_ascii:
                alt_names = list(set(alt_names + [remove_accents(i) for i in alt_names]))
            admin1_name = ""
            if row[10]:
                admin1_name = adm1_dict.get(f"{row[8]}.{row[10]}", "")
            admin2_name = ""
            if row[11]:
                admin2_name = adm2_dict.get(f"{row[8]}.{row[10]}.{row[11]}", "")
                if not admin2_name:
                    adm2_missing += 1
            doc = {"geonameid": row[0],
                   "name": row[1],
                   "asciiname": row[2],
                   "alternativenames": alt_names,
                   "coordinates": coords,
                   "feature_class": row[6],
                   "feature_code": row[7],
                   "country_code3": country_code3,
                   "admin1_code": row[10],
                   "admin1_name": admin1_name,
                   "admin2_code": row[11],
                   "admin2_name": admin2_name,
                   "admin3_code": row[12],
                   "admin4_code": row[13],
                   "population": row[14],
                   "alt_name_length": alt_name_length,
                   "modification_date": todays_date}
            yield {"_index": INDEX, "_id": doc["geonameid"], "_source": doc}
        except Exception as e:
            print(e, row)
            row_errors += 1
    print(f"admin2 name not found (benign, record still indexed): {adm2_missing}")
    print(f"rows skipped due to errors: {row_errors}")


def download(data_dir):
    """Download and unpack the GeoNames gazetteer (~400 MB zip, 1.8 GB unpacked)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fn in GAZETTEER_FILES:
        print(f"Downloading {fn} ...")
        urlretrieve(f"{GEONAMES_BASE}/{fn}", data_dir / fn)
    print("Unpacking allCountries.zip ...")
    with zipfile.ZipFile(data_dir / "allCountries.zip") as z:
        z.extractall(data_dir)


def connect(es_url):
    return Elasticsearch(es_url, timeout=60, max_retries=2, retry_on_timeout=True)


def count(es):
    """Documents in the geonames index, or None if there is no index."""
    if not es.indices.exists(index=INDEX):
        return None
    es.indices.refresh(index=INDEX)
    return es.count(index=INDEX)["count"]


def provenance(es):
    """The `_meta` a build stamped on the index ({} for older indices)."""
    if not es.indices.exists(index=INDEX):
        return None
    mapping = es.indices.get_mapping(index=INDEX)
    return next(iter(mapping.values()))["mappings"].get("_meta", {})


def recreate(es):
    """Delete ONLY the geonames index and recreate it from the packaged mapping."""
    # Watermarks first: on a >95%-full disk the default flood stage would make
    # the new index read-only before a single document arrives.
    es.cluster.put_settings(body={"persistent": {
        "cluster.routing.allocation.disk.watermark.low": "10gb",
        "cluster.routing.allocation.disk.watermark.high": "5gb",
        "cluster.routing.allocation.disk.watermark.flood_stage": "4gb",
    }})
    if es.indices.exists(index=INDEX):
        print(f"Deleting existing '{INDEX}' index ...")
        es.indices.delete(index=INDEX)
    mapping = json.loads((files("mordecai3") / "assets" / "geonames_mapping.json")
                         .read_text(encoding="utf-8"))
    print(f"Creating '{INDEX}' index ...")
    es.indices.create(index=INDEX, body=mapping)


def load(es, data_dir):
    """Bulk-load the gazetteer in `data_dir`. About 15-25 minutes."""
    data_dir = Path(data_dir)
    t = time.time()
    bad_codes = set()
    adm1 = read_codes(data_dir / "admin1CodesASCII.txt")
    adm2 = read_codes(data_dir / "admin2Codes.txt")
    with open(data_dir / "allCountries.txt", encoding="utf-8") as f:
        actions = documents(csv.reader(f, delimiter="\t"), adm1, adm2,
                            bad_codes=bad_codes)
        helpers.bulk(es, actions, chunk_size=500)
    es.indices.refresh(index=INDEX)
    print(f"Loaded in {(time.time() - t) / 60:.1f} minutes")
    if bad_codes:
        print(f"Unrecognised ISO codes (indexed as 'NA'): {sorted(bad_codes)}")


def stamp(es, data_dir, doc_count):
    """Record build provenance on the index itself, in the mapping's `_meta`.

    GeoNames carries no version stamp, so the gazetteer file's download date is
    the best answer to "how stale is this index?".
    """
    from . import __version__
    gazetteer = Path(data_dir) / "allCountries.txt"
    meta = {
        "source": GEONAMES_BASE,
        "gazetteer_file": "allCountries.txt",
        "dump_date": date.fromtimestamp(gazetteer.stat().st_mtime).isoformat(),
        "build_date": date.today().isoformat(),
        "doc_count": doc_count,
        "builder": f"mordecai3 {__version__} (mordecai3 index build)",
    }
    es.indices.put_mapping(index=INDEX, body={"_meta": meta})
    return meta


# ---------------------------------------------------------------- prebuilt index
#
# The prebuilt index is paired with the packaged checkpoint (candidate features
# come from the index), so the archive is pinned by checksum, and a new index
# means a new filename and checksum here. Mirrors are tried in order;
# MORDECAI_INDEX_URL puts another one in front (a private mirror, or a file://
# path to an archive already on disk).

PREBUILT_NAME = "mordecai3_geonames_index_2026-09-24.tar.gz"
PREBUILT_SHA256 = "7fe3d44337e423d4cb467ef2f7cbd40eb9c73eee16c0a750ccae8bb6865d1979"
PREBUILT_MIRRORS = [f"https://andrewhalterman.com/files/{PREBUILT_NAME}"]
PREBUILT_TOP = "geonames_index"
PREBUILT_NEEDS_BYTES = 4 * 1024**3   # 1.5 GB archive + 2.2 GB unpacked, with headroom


class FetchError(Exception):
    """The prebuilt index could not be downloaded, verified, or unpacked."""

    def __init__(self, failures, fallback_ok=True):
        self.failures = failures          # [(source, reason)]
        self.fallback_ok = fallback_ok    # False when building would fail too (disk)
        super().__init__("; ".join(f"{s}: {r}" for s, r in failures))


def prebuilt_urls():
    extra = os.environ.get("MORDECAI_INDEX_URL")
    return ([extra] if extra else []) + PREBUILT_MIRRORS


def _download_verified(url, dest, sha256, progress=True):
    """Stream `url` to `dest`, hashing on the way. Raises on any failure."""
    from urllib.request import urlopen
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    try:
        with urlopen(url, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or None
            with tqdm(total=total, unit="B", unit_scale=True, disable=not progress,
                      desc=PREBUILT_NAME) as bar:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    bar.update(len(chunk))
        if h.hexdigest() != sha256:
            raise ValueError(f"checksum mismatch (got {h.hexdigest()[:12]}..., "
                             f"expected {sha256[:12]}...) -- incomplete or different file")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def _safe_extract(archive, dest_dir):
    """Unpack, refusing any member outside `<dest_dir>/geonames_index/`."""
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != PREBUILT_TOP \
                    or m.issym() or m.islnk():
                raise ValueError(f"unexpected archive member {m.name!r}")
        if sys.version_info >= (3, 12):
            tar.extractall(dest_dir, filter="data")
        else:
            tar.extractall(dest_dir)


def fetch_prebuilt(dest_dir, urls=None, sha256=None, keep_archive=False,
                   progress=True):
    """Download, verify and unpack the prebuilt index into `dest_dir/geonames_index`.

    Raises FetchError listing what went wrong at each mirror.
    """
    sha256 = sha256 or PREBUILT_SHA256
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / PREBUILT_TOP
    if target.exists() and any(target.iterdir()):
        raise FetchError([(str(target), "already exists and is not empty; "
                           "remove it or pass another --dir")], fallback_ok=False)
    free = shutil.disk_usage(dest_dir).free
    if free < PREBUILT_NEEDS_BYTES:
        raise FetchError([(str(dest_dir), f"only {free / 1024**3:.1f} GB free; "
                           f"need about {PREBUILT_NEEDS_BYTES / 1024**3:.0f} GB")],
                         fallback_ok=False)
    archive = dest_dir / PREBUILT_NAME
    failures = []
    for url in (urls or prebuilt_urls()):
        try:
            _download_verified(url, archive, sha256, progress=progress)
            break
        except Exception as e:                      # network, HTTP, checksum, disk
            failures.append((url, str(e) or type(e).__name__))
    else:
        raise FetchError(failures)
    try:
        _safe_extract(archive, dest_dir)
    except Exception as e:
        shutil.rmtree(target, ignore_errors=True)   # it was empty or absent before
        raise FetchError([(str(archive), f"could not unpack: {e}")])
    finally:
        if not keep_archive:
            archive.unlink(missing_ok=True)
    return target
