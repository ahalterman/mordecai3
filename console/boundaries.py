"""Administrative boundary polygons for resolved toponyms.

Mordecai resolves a mention to a GeoNames record, which is a *point* -- even
when the record is a country or a province, where a point is a poor
representation of what the text was talking about. This module attaches the
polygon when there is a sensible one to attach, from the geoBoundaries CGAZ
composites built into SQLite by `build_boundaries.py`.

The join is the interesting part. geoBoundaries carries only `shapeGroup`
(ISO3) and `shapeName`; there is no GeoNames id in it, and no admin code. The
obvious approach -- fuzzy-match `admin1_name` against `shapeName` -- loses to
transliteration ("Tillabéri" / "Tillaberi"), to administrative renaming, and to
the "Province of X" / "X Region" / "X" family of near-synonyms.

So the primary join is geometric: **which polygon contains the coordinate the
ranker already committed to.** That needs no string heuristics, is exact where
it fires, and is cheap because the bounding boxes in the store reject
essentially every wrong polygon before any geometry is parsed. Name matching is
kept, but only as a fallback for the cases where the GeoNames coordinate falls
outside its own unit -- which happens: an ADM2 seat can sit on the far side of a
disputed border, and CGAZ is clipped to a different border set than GeoNames
uses.

CGAZ stops at ADM2, so this covers countries, first-order and second-order
divisions. A city, a river, or an airport gets no polygon and stays a point,
which is the right answer rather than a limitation.
"""

import json
import logging
import sqlite3
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

import jellyfish
from shapely.geometry import Point, shape

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "geoboundaries" / "boundaries.sqlite"

# GeoNames feature code -> CGAZ admin level. Only the A (administrative) class
# appears here: everything else is a point by nature.
#
# PCLI is the ordinary sovereign state; the rest of the PCL* family covers
# dependencies, freely-associated states and the leftovers, all of which CGAZ
# carries at ADM0. ADMD ("administrative division, unspecified") is genuinely
# ambiguous between levels and is handled separately below.
FEATURE_CODE_LEVEL = {
    "PCLI": 0, "PCL": 0, "PCLD": 0, "PCLF": 0, "PCLS": 0, "PCLIX": 0, "TERR": 0,
    "ADM1": 1, "ADM1H": 1,
    "ADM2": 2, "ADM2H": 2,
}

# ADMD could be either; try the more specific level first, since a unit that is
# only recorded at ADM1 will still be found on the second pass.
AMBIGUOUS_LEVELS = {"ADMD": (2, 1), "ADM3": (2,), "ADM4": (2,)}


def normalize_name(name: str) -> str:
    """Casefold and strip diacritics. Mirrors `build_boundaries.normalize_name`."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


# Words that name a *kind* of administrative unit rather than which one.
# Stripped only for the comparison, never from what is displayed.
#
# The list is deliberately multilingual: the two gazetteers pick different
# languages for the same unit more or less at random, and an untranslated type
# word is pure noise that drags the similarity of a correct pair down --
# "Kupiškis" against "Kupiškio rajono savivaldybė" scores 0.859 with the
# Lithuanian left in and 1.0 with it removed.
#
# What is deliberately NOT here: "city", "urban", "rural", "metropolitan",
# "greater", "north/south/east/west". They look like the same sort of word and
# are not -- they are the entire difference between "Osh City" and "Osh
# Region", or "Hwanghae-namdo" and "Hwanghae-bukto", which are distinct units
# that nest or sit side by side. Stripping them would turn correct rejections
# into confident wrong answers.
_ADMIN_NOISE = frozenset({
    # English / French / Spanish / Portuguese / Italian
    "region", "province", "provincia", "provincie", "provinsi", "prefecture",
    "governorate", "district", "distrito", "county", "state", "department",
    "departement", "division", "cercle", "municipality", "municipio",
    "commune", "comune", "comuna", "canton", "territory", "area", "zone",
    "regione", "arrondissement", "subprefecture", "subdistrict",
    # Slavic / post-Soviet
    "oblast", "oblasti", "raion", "rayon", "rayonu", "okrug", "kraj", "krai",
    "powiat", "gmina", "opstina", "obshtina", "zupanija", "judet", "judetul",
    "miskrada", "munitsipaliteti", "munitsipalitet", "savivaldybe",
    "rajono", "rajonas", "vald", "maakond",
    # Germanic / Nordic
    "gewest", "landkreis", "kreis", "bezirk", "amt", "fylke", "kommun",
    "kommune", "lan", "shire",
    # Middle East / South Asia / Africa / East Asia
    "wilayat", "wilayet", "wilayah", "muhafazah", "mintaqah", "imarat",
    "nahiya", "woreda", "upazila", "tehsil", "taluk", "taluka", "mandal",
    "aimag", "somon", "kabupaten", "shi", "sheng",
    # Articles and connectives
    "of", "the", "de", "du", "des", "la", "le", "el", "al", "da", "do",
    "di", "van", "von",
})


# How similar two administrative name cores have to be to count as the same
# unit. Exact equality is too strict -- the two gazetteers transliterate
# independently, and CGAZ spells Idlib "Idleb", Koulikoro "Koulikouro" -- while
# anything much looser starts accepting neighbouring units. Calibrated in
# `validate_boundaries.py` against every ADM1 and ADM2 record in the GeoNames
# index; see console/README.md for the measured separation.
NAME_AGREE_THRESHOLD = 0.86


def name_similarity(a: str, b: str) -> float:
    """Jaro-Winkler over two name cores.

    Jaro-Winkler rather than raw edit distance because it weights a shared
    prefix, which is exactly how transliteration variants differ: they agree at
    the front and drift at the vowels.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return jellyfish.jaro_winkler_similarity(a, b)


def _name_core(name: str) -> str:
    """The identifying part of an administrative name.

    "Cercle de Bourem", "Bourem Cercle" and "Bourem" all reduce to "bourem" --
    which is what makes the fallback usable at all, since GeoNames and
    geoBoundaries pick different members of that family more or less at random.
    """
    tokens = [t for t in normalize_name(name).split() if t not in _ADMIN_NOISE]
    return " ".join(tokens) or normalize_name(name)


class BoundaryStore:
    """Read-only lookups against the SQLite boundary store.

    Thread-safe by giving each thread its own connection: SQLite connections
    are not shareable across threads, and the server serves requests from a
    pool.
    """

    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = Path(db_path)
        self.available = self.db_path.exists()
        self._local = threading.local()
        if not self.available:
            logger.warning(
                "boundary store %s not found -- resolved places will be points "
                "only. Run console/fetch_boundaries.sh then "
                "console/build_boundaries.py.", self.db_path)
            self.meta = {}
            self.shape_count = 0
        else:
            with self._connect() as conn:
                self.meta = dict(conn.execute(
                    "SELECT key, value FROM meta").fetchall())
                self.shape_count = conn.execute(
                    "SELECT COUNT(*) FROM shapes").fetchone()[0]
            logger.info("boundary store: %d shapes from %s",
                        self.shape_count, self.db_path)

    def _connect(self):
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    @property
    def conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    # ------------------------------------------------------------------ query

    def lookup(self, resolved):
        """The boundary for one resolved entity, or None.

        Parameters
        ----------
        resolved : dict
            A Mordecai result with at least `feature_code`, `country_code3`,
            `lat` and `lon`. `name`, `admin1_name` and `admin2_name` are used
            by the name fallback when present.

        Returns
        -------
        dict or None
            `{"level", "name", "iso3", "match": "iso3"|"point-in-polygon"|"name",
              "name_score": float or None, "geometry": <GeoJSON geometry>,
              "bbox": [minx, miny, maxx, maxy]}`
        """
        if not self.available:
            return None
        code = (resolved.get("feature_code") or "").upper()
        iso3 = (resolved.get("country_code3") or "").upper()
        if not iso3 or iso3 == "NULL":
            return None

        if code in FEATURE_CODE_LEVEL:
            levels = (FEATURE_CODE_LEVEL[code],)
        elif code in AMBIGUOUS_LEVELS:
            levels = AMBIGUOUS_LEVELS[code]
        else:
            return None

        # Whether the record *is* the unit we are looking for, as opposed to a
        # place sitting inside it. It decides whether the name check below has
        # anything to compare against.
        self_naming = code in FEATURE_CODE_LEVEL

        # Read once: the geometric join needs it below, and every branch needs
        # it to pick the part of the shape a map should frame on.
        lat, lon = resolved.get("lat"), resolved.get("lon")

        for level in levels:
            # ADM0 is unambiguous: one shape per country, and the country code
            # is a real key rather than a name. No geometry test needed, and
            # doing one would only introduce a way to fail.
            if level == 0:
                hit = self._only_shape(0, iso3)
                if hit:
                    return self._as_boundary(hit, "iso3", lon=lon, lat=lat)
                continue

            hit = None
            if lat is not None and lon is not None:
                hit = self._containing(level, iso3, float(lon), float(lat))

            if hit is not None:
                # Point-in-polygon always answers, which is the problem: when
                # GeoNames and geoBoundaries disagree about a country's
                # administrative structure it answers *plausibly and wrongly*.
                # Mali split Ménaka out of Gao in 2016; GeoNames records Ménaka
                # as an ADM1, CGAZ still carries the older nine-region layout,
                # and the containing polygon for Ménaka's coordinate is Gao.
                # So when the record names its own unit, the name has to agree
                # before the polygon is trusted -- a point on the map is a much
                # better answer than the wrong province drawn confidently.
                sim = (self._name_agreement(hit[0], resolved, level)
                       if self_naming else None)
                if sim is None or sim >= NAME_AGREE_THRESHOLD:
                    return self._as_boundary(hit, "point-in-polygon", sim,
                                             lon=lon, lat=lat)
                logger.info(
                    "boundary rejected: %r (%s) resolves inside ADM%d %r -- "
                    "the two gazetteers disagree about this unit",
                    resolved.get("name"), code, level, hit[0])
                continue

            hit = self._by_name(level, iso3, resolved)
            if hit:
                return self._as_boundary(
                    hit, "name", self._name_agreement(hit[0], resolved, level),
                    lon=lon, lat=lat)
        return None

    @staticmethod
    def _name_agreement(shape_name, resolved, level):
        """How well the containing polygon's name matches the record's, 0-1.

        Returned rather than thresholded internally so the console can show it:
        a boundary matched at 0.87 and one matched at 1.0 are not equally
        trustworthy, and the honest thing is to say which this was rather than
        to draw both the same way.

        Compared on `_name_core`, so "Cercle de Bourem" agrees with "Bourem"
        and "Tillabéri Region" with "Tillaberi", while "Ménaka Region" does not
        agree with "Gao". The record's own admin name is accepted too: a unit
        GeoNames files under a local name often carries the CGAZ spelling in
        `admin1_name`.
        """
        target = _name_core(shape_name)
        if not target:
            return False
        candidates = [resolved.get("name"),
                      resolved.get("admin2_name") if level == 2
                      else resolved.get("admin1_name")]
        return max((name_similarity(_name_core(c), target)
                    for c in candidates if c and c != "NULL"), default=0.0)

    def _only_shape(self, level, iso3):
        return self.conn.execute(
            "SELECT name, iso3, level, minx, miny, maxx, maxy, geom"
            "  FROM shapes WHERE level = ? AND iso3 = ? LIMIT 1",
            (level, iso3)).fetchone()

    def _containing(self, level, iso3, lon, lat):
        """The shape at `level` in `iso3` whose polygon contains (lon, lat).

        The bbox test is done in SQL so the vast majority of the country's
        shapes are rejected without their geometry ever being read off disk --
        for an ADM2 lookup in a country with 600 units that is typically one or
        two candidates parsed instead of 600.
        """
        rows = self.conn.execute(
            "SELECT name, iso3, level, minx, miny, maxx, maxy, geom"
            "  FROM shapes"
            " WHERE level = ? AND iso3 = ?"
            "   AND minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ?",
            (level, iso3, lon, lon, lat, lat)).fetchall()
        if not rows:
            return None
        point = Point(lon, lat)
        for row in rows:
            if shape(json.loads(row[7])).contains(point):
                return row
        return None

    def _by_name(self, level, iso3, resolved):
        """Fallback: match the unit's name, stripped to its identifying core.

        Tries the record's own name first, then the admin name GeoNames records
        for it -- "Ménaka Region" resolves against `name`, while a town whose
        ADM2 we want resolves against `admin2_name`.
        """
        wanted = [resolved.get("name")]
        wanted.append(resolved.get("admin2_name") if level == 2
                      else resolved.get("admin1_name"))
        cores = {_name_core(w) for w in wanted if w and w != "NULL"}
        if not cores:
            return None
        rows = self.conn.execute(
            "SELECT name, iso3, level, minx, miny, maxx, maxy, geom"
            "  FROM shapes WHERE level = ? AND iso3 = ?", (level, iso3)
        ).fetchall()
        for row in rows:
            if _name_core(row[0]) in cores:
                return row
        return None

    @staticmethod
    def _focus_bbox(geometry, full, lon, lat):
        """The extent of the one part of this shape the mention's point sits in.

        `bbox` is the true extent of the whole unit, and for a country with
        scattered overseas territory that extent is close to useless as a map
        frame. France runs from Wallis to Guadeloupe; Russia, the United
        States, New Zealand, Fiji and Kiribati wrap the antimeridian and come
        back as a bbox spanning the entire globe. Eight of the 218 ADM0 shapes
        in the store are in that state, and they include four of the most
        frequently mentioned countries there are -- so a map that frames on
        `bbox` shows the whole world the moment a document says "Russia", and
        every polygon on it is a few pixels wide. That reads as the boundary
        layer not working.

        So the payload also carries the extent of the single part containing
        the coordinate the ranker already committed to: metropolitan France,
        the Russian mainland, the lower 48. The off-frame parts are still
        drawn -- nothing is hidden, they simply do not get a vote on the
        framing.

        When the point falls outside every part -- an offshore gazetteer
        centroid, or the two gazetteers clipping a coastline differently --
        the nearest part wins, which is still a far better frame than the
        globe.

        The cost, stated rather than hidden: for an archipelago whose full
        extent was never pathological, this frames tighter than it needs to.
        Indonesia's GeoNames centroid lands on Sulawesi, so a document saying
        only "Indonesia" frames on Sulawesi with the rest of the archipelago
        drawn spilling off the edges, where the old behaviour framed the whole
        country. That is a worse frame in one case against an unreadable one
        in eight, and the shape is still drawn either way.
        """
        try:
            geom = shape(geometry)
        except Exception:                     # pragma: no cover - defensive
            return full
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        if len(parts) < 2:
            return full
        point = Point(lon, lat)
        for part in parts:
            if part.contains(point):
                return list(part.bounds)
        return list(min(parts, key=lambda p: p.distance(point)).bounds)

    @classmethod
    def _as_boundary(cls, row, match, name_score=None, lon=None, lat=None):
        name, iso3, level, minx, miny, maxx, maxy, geom = row
        geometry = json.loads(geom)
        full = [minx, miny, maxx, maxy]
        focus = full
        if lon is not None and lat is not None:
            focus = cls._focus_bbox(geometry, full, float(lon), float(lat))
        return {"level": level,
                "name": name,
                "iso3": iso3,
                "match": match,
                # None when there was nothing to compare -- an ADM0 keyed on
                # the country code, or a city whose containing unit we looked
                # up rather than the unit itself. Not the same as 0.0.
                "name_score": (round(name_score, 3)
                               if name_score is not None else None),
                "bbox": full,
                # What a map should frame on; see `_focus_bbox`. Equal to
                # `bbox` for every single-part shape, which is most of them.
                "focus_bbox": focus,
                "geometry": geometry}


@lru_cache(maxsize=1)
def get_store(db_path=None):
    """The process-wide store. Cached because opening it reads the meta table."""
    return BoundaryStore(db_path or DEFAULT_DB)
