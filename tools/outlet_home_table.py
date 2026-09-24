"""Curated outlet-domain -> newsroom-home table, and its gazetteer geocoder.

PROVENANCE / LEAK PROTOCOL
==========================

This table is the whole experiment.  A feature that says "the gold answer tends
to sit near the newspaper's home town" is only worth measuring if the home town
was not itself read off the answer key, so every row here was written from *the
domain string plus outside knowledge of which newspaper that domain belongs to*
and nothing else.  Concretely, the rules the table was built under:

1. The only corpus-derived input is the set of ``<domain>`` strings.  No article
   text, no ``<toponym>``, no ``<gaztag>``, no geonameid, no per-domain gold
   statistic of any kind was consulted while writing the ``HOME`` entries -- not
   from the held-out split and not from the training split either.  (Training
   labels would have been permissible under the brief; they were not used, which
   makes the mapping split-independent and therefore identical for a train
   document and a held-out one.)
2. Each value names a *place*, in words -- "Richmond, Indiana, USA" -- never a
   geonameid.  The words are turned into coordinates by :func:`geocode_homes`,
   which queries the same GeoNames index the ranker retrieves candidates from.
   So the numbers that reach the model come from the gazetteer, not from a
   hand-picked identifier that could have been copied out of the key.
3. ``scope`` is a judgement about the *outlet*, not about its articles: "local"
   means a city/metro paper, broadcaster or campus paper whose newsroom serves a
   named place; "national" means a national or international outlet whose
   coverage has no home town.  National outlets get **no** home, so the feature
   is null for them -- inventing a city for Reuters would put a locality prior on
   a wire service.
4. Where the outlet behind a domain was genuinely unclear (``reporter.net``,
   ``dailytribune.net``, ``theintelligencer.com``, ``nlhnews.co.uk``,
   ``goldentrianglenewspapers.com``), the row is ``None`` rather than a guess.
   All five are training-split-only domains, so the choice cannot flatter the
   held-out numbers either way.

The pleasing consequence, and the reason the feature is not just a restatement
of the population prior: a local paper called *Paris* is evidence **against**
Paris, France.  ``parispi.net`` is the Paris Post-Intelligencer of Paris,
**Tennessee**; ``theparisnews.com`` is The Paris News of Paris, **Texas**;
``sentinel-echo.com`` is the Sentinel-Echo of London, **Kentucky**;
``themercury.com`` is the Manhattan Mercury of Manhattan, **Kansas**.  Those
readings come from knowing the mastheads, and each one inverts what a
population-ranked gazetteer lookup of the domain's tokens would return.

``tools/outlet_leak_audit.py`` re-checks properties 1 and 2 mechanically.
"""

import json
import os

# domain -> (place, admin1, ISO3, scope) | None
#
# ``place`` is the newsroom's home town as a gazetteer would name it, ``admin1``
# disambiguates it (the whole point: Richmond *Indiana*, Paris *Tennessee*), and
# ``scope`` is "local" or "national" per rule 3 above.  ``None`` is rule 4.
HOME = {
    #
    #   LGL: 85 domains.  Overwhelmingly US local dailies, plus a dozen UK/
    #   Canadian locals and a group of Middle Eastern / Russian / Georgian
    #   papers.
    #
    "ajc.com": ("Atlanta", "Georgia", "USA", "local"),                        # Atlanta Journal-Constitution
    "alextimes.com": ("Alexandria", "Virginia", "USA", "local"),              # Alexandria Times
    "alligator.org": ("Gainesville", "Florida", "USA", "local"),              # Independent Florida Alligator (UF)
    "athensreview.com": ("Athens", "Texas", "USA", "local"),                  # Athens Daily Review
    "bclocalnews.com": ("British Columbia", None, "CAN", "local"),            # Black Press BC Local News (province-wide)
    "cambridge-news.co.uk": ("Cambridge", "England", "GBR", "local"),         # Cambridge News
    "chronicle.augusta.com": ("Augusta", "Georgia", "USA", "local"),          # Augusta Chronicle
    "civil.ge": (None, None, "GEO", "national"),                              # Civil Georgia
    "columbiadailyherald.com": ("Columbia", "Tennessee", "USA", "local"),     # The Daily Herald
    "columbuslocalnews.com": ("Columbus", "Ohio", "USA", "local"),            # Columbus (OH) suburban weeklies
    "columbustelegram.com": ("Columbus", "Nebraska", "USA", "local"),         # Columbus Telegram
    "concordmonitor.com": ("Concord", "New Hampshire", "USA", "local"),       # Concord Monitor
    "courant.com": ("Hartford", "Connecticut", "USA", "local"),               # Hartford Courant
    "daily-chronicle.com": ("DeKalb", "Illinois", "USA", "local"),            # Daily Chronicle
    "daily-jeff.com": ("Cambridge", "Ohio", "USA", "local"),                  # The Daily Jeffersonian
    "dailyherald.com": ("Arlington Heights", "Illinois", "USA", "local"),     # Daily Herald (Chicago suburbs)
    "dailymail.com": ("Charleston", "West Virginia", "USA", "local"),         # Charleston Daily Mail (NOT the UK paper)
    "dailynews.com": ("Los Angeles", "California", "USA", "local"),           # Los Angeles Daily News
    "dailypostathenian.com": ("Athens", "Tennessee", "USA", "local"),         # The Daily Post-Athenian
    "dailystar.com.lb": ("Beirut", None, "LBN", "local"),                     # The Daily Star (Beirut)
    "dailytribune.net": None,                                                 # rule 4: several "Daily Tribune"s
    "dallasnews.com": ("Dallas", "Texas", "USA", "local"),                    # Dallas Morning News
    "dispatch.com": ("Columbus", "Ohio", "USA", "local"),                     # The Columbus Dispatch
    "echopress.com": ("Alexandria", "Minnesota", "USA", "local"),             # Echo Press
    "gainesville.com": ("Gainesville", "Florida", "USA", "local"),            # The Gainesville Sun
    "gainesvilleregister.com": ("Gainesville", "Texas", "USA", "local"),      # Gainesville Daily Register
    "gainesvilletimes.com": ("Gainesville", "Georgia", "USA", "local"),       # The Times
    "goldentrianglenewspapers.com": None,                                     # rule 4: TX or MS "Golden Triangle"
    "haaretz.com": (None, None, "ISR", "national"),                           # Haaretz
    "heraldonline.com": ("Rock Hill", "South Carolina", "USA", "local"),      # The Herald
    "independent.ie": (None, None, "IRL", "national"),                        # Irish Independent
    "insidebayarea.com": ("Oakland", "California", "USA", "local"),           # Inside Bay Area / ANG
    "itemonline.com": ("Huntsville", "Texas", "USA", "local"),                # The Huntsville Item
    "jordannews.com": (None, None, "JOR", "national"),                        # Jordan News
    "jpost.com": ("Jerusalem", None, "ISR", "local"),                         # The Jerusalem Post
    "kansascity.com": ("Kansas City", "Missouri", "USA", "local"),            # The Kansas City Star
    "lansingstatejournal.com": ("Lansing", "Michigan", "USA", "local"),       # Lansing State Journal
    "lasvegassun.com": ("Las Vegas", "Nevada", "USA", "local"),               # Las Vegas Sun
    "latimes.com": ("Los Angeles", "California", "USA", "local"),             # Los Angeles Times
    "ldnews.com": ("Lebanon", "Pennsylvania", "USA", "local"),                # Lebanon Daily News
    "ledger-enquirer.com": ("Columbus", "Georgia", "USA", "local"),           # Columbus Ledger-Enquirer
    "lfpress.ca": ("London", "Ontario", "CAN", "local"),                      # London Free Press
    "masslive.com": ("Springfield", "Massachusetts", "USA", "local"),         # MassLive / The Republican
    "mercurynews.com": ("San Jose", "California", "USA", "local"),            # San Jose Mercury News
    "messenger.com.ge": (None, None, "GEO", "national"),                      # The Messenger (Tbilisi)
    "miamiherald.com": ("Miami", "Florida", "USA", "local"),                  # Miami Herald
    "middletownpress.com": ("Middletown", "Connecticut", "USA", "local"),     # The Middletown Press
    "moscowtimes.ru": ("Moscow", None, "RUS", "local"),                       # The Moscow Times
    "myinrich.com": ("Richmond", "Indiana", "USA", "local"),                  # "my IN rich" = Richmond, Indiana
    "newarkadvocate.com": ("Newark", "Ohio", "USA", "local"),                 # The Advocate
    "news-leader.com": ("Springfield", "Missouri", "USA", "local"),           # Springfield News-Leader
    "news.postbulletin.com": ("Rochester", "Minnesota", "USA", "local"),      # Post-Bulletin
    "norfolkdailynews.com": ("Norfolk", "Nebraska", "USA", "local"),          # Norfolk Daily News
    "onlineathens.com": ("Athens", "Georgia", "USA", "local"),                # Athens Banner-Herald
    "oxfordpress.com": ("Oxford", "Ohio", "USA", "local"),                    # Oxford Press (Cox Ohio)
    "pal-item.com": ("Richmond", "Indiana", "USA", "local"),                  # Palladium-Item
    "palestineherald.com": ("Palestine", "Texas", "USA", "local"),            # Palestine Herald-Press
    "parisbeacon.com": ("Paris", "Illinois", "USA", "local"),                 # Paris Beacon-News
    "parispi.net": ("Paris", "Tennessee", "USA", "local"),                    # Paris Post-Intelligencer
    "philly.com": ("Philadelphia", "Pennsylvania", "USA", "local"),           # Philadelphia Inquirer / Daily News
    "post-gazette.com": ("Pittsburgh", "Pennsylvania", "USA", "local"),       # Pittsburgh Post-Gazette
    "postandcourier.com": ("Charleston", "South Carolina", "USA", "local"),   # The Post and Courier
    "recordernewspapers.com": ("New Jersey", None, "USA", "local"),           # Recorder Community Newspapers (NJ)
    "registercitizen.com": ("Torrington", "Connecticut", "USA", "local"),     # The Register Citizen
    "reporter.net": None,                                                     # rule 4
    "richmond.com": ("Richmond", "Virginia", "USA", "local"),                 # Richmond Times-Dispatch
    "richmondandtwickenhamtimes.co.uk": ("Richmond", "England", "GBR", "local"),  # Richmond & Twickenham Times
    "richmondregister.com": ("Richmond", "Kentucky", "USA", "local"),         # The Richmond Register
    "sentinel-echo.com": ("London", "Kentucky", "USA", "local"),              # The Sentinel-Echo
    "springfieldnewssun.com": ("Springfield", "Ohio", "USA", "local"),        # Springfield News-Sun
    "sptimes.ru": ("Saint Petersburg", None, "RUS", "local"),                 # The St. Petersburg Times
    "star-telegram.com": ("Fort Worth", "Texas", "USA", "local"),             # Fort Worth Star-Telegram
    "thecolumbiastar.com": ("Columbia", "South Carolina", "USA", "local"),    # The Columbia Star
    "theday.com": ("New London", "Connecticut", "USA", "local"),              # The Day
    "theintelligencer.com": None,                                             # rule 4
    "thelancasterandmorecambecitizen.co.uk": ("Lancaster", "England", "GBR", "local"),  # Lancaster & Morecambe Citizen
    "themercury.com": ("Manhattan", "Kansas", "USA", "local"),                # The Manhattan Mercury
    "theparisnews.com": ("Paris", "Texas", "USA", "local"),                   # The Paris News
    "thestate.com": ("Columbia", "South Carolina", "USA", "local"),           # The State
    "thetowntalk.com": ("Alexandria", "Louisiana", "USA", "local"),           # The Town Talk
    "thisisoxfordshire.co.uk": ("Oxford", "England", "GBR", "local"),         # Oxford Mail
    "timesdispatch.com": ("Richmond", "Virginia", "USA", "local"),            # Richmond Times-Dispatch
    "weekly.ahram.org.eg": (None, None, "EGY", "national"),                   # Al-Ahram Weekly
    "woodstocksentinelreview.com": ("Woodstock", "Ontario", "CAN", "local"),  # Woodstock Sentinel-Review
    "wvgazette.com": ("Charleston", "West Virginia", "USA", "local"),         # Charleston Gazette
    #
    #   TR-News: 35 domains.  Mostly national broadcasters and wires, which get
    #   no home; the local stations and small-town papers do.
    #
    "amarillo.com": ("Amarillo", "Texas", "USA", "local"),                    # Amarillo Globe-News
    "bnonews.com": (None, None, None, "national"),                            # BNO News (wire)
    "brantnews.com": ("Brantford", "Ontario", "CAN", "local"),                # Brant News
    "edmontonjournal.com": ("Edmonton", "Alberta", "CAN", "local"),           # Edmonton Journal
    "engineeringnews.co.za": (None, None, "ZAF", "national"),                 # Engineering News
    "eparisextra.com": ("Paris", "Texas", "USA", "local"),                    # eParis Extra
    "gazette.com": ("Colorado Springs", "Colorado", "USA", "local"),          # The Gazette
    "globalnews.ca": (None, None, "CAN", "national"),                         # Global News
    "katv.com": ("Little Rock", "Arkansas", "USA", "local"),                  # KATV
    "wcluradio.com": ("Glasgow", "Kentucky", "USA", "local"),                 # WCLU Radio
    "www.abc.net.au": (None, None, "AUS", "national"),                        # ABC Australia
    "www.bbc.com": (None, None, None, "national"),                            # BBC (global desk)
    "www.cbc.ca": (None, None, "CAN", "national"),                            # CBC
    "www.chinapost.com.tw": (None, None, "TWN", "national"),                  # The China Post
    "www.cnn.com": (None, None, None, "national"),                            # CNN
    "www.ctvnews.ca": (None, None, "CAN", "national"),                        # CTV News
    "www.enfield-today.co.uk": ("Enfield", "England", "GBR", "local"),        # Enfield Today
    "www.freshplaza.com": (None, None, None, "national"),                     # FreshPlaza (trade wire)
    "www.huntingtonnews.net": ("Huntington", "West Virginia", "USA", "local"),  # Huntington News
    "www.independent.co.uk": (None, None, "GBR", "national"),                 # The Independent
    "www.innisfailprovince.ca": ("Innisfail", "Alberta", "CAN", "local"),     # The Innisfail Province
    "www.kwwl.com": ("Waterloo", "Iowa", "USA", "local"),                     # KWWL
    "www.lemarssentinel.com": ("Le Mars", "Iowa", "USA", "local"),            # Le Mars Sentinel
    "www.lfpress.com": ("London", "Ontario", "CAN", "local"),                 # London Free Press
    "www.marketwired.com": (None, None, None, "national"),                    # Marketwired (PR wire)
    "www.missourifarmertoday.com": ("Missouri", None, "USA", "local"),        # Missouri Farmer Today (state-wide)
    "www.moberlymonitor.com": ("Moberly", "Missouri", "USA", "local"),        # Moberly Monitor-Index
    "www.nationalpost.com": (None, None, "CAN", "national"),                  # National Post
    "www.nlhnews.co.uk": None,                                                # rule 4
    "www.nytimes.com": (None, None, "USA", "national"),                       # The New York Times
    "www.qconline.com": ("Moline", "Illinois", "USA", "local"),               # Quad-Cities Online
    "www.registercitizen.com": ("Torrington", "Connecticut", "USA", "local"),  # The Register Citizen
    "www.reuters.com": (None, None, None, "national"),                        # Reuters
    "www.wbko.com": ("Bowling Green", "Kentucky", "USA", "local"),            # WBKO
    "www.wkyt.com": ("Lexington", "Kentucky", "USA", "local"),                # WKYT
}


#
#   Turning the words into coordinates
#

# Prefer a real settlement, then an administrative area; within a class, the
# most populous match wins.  A newsroom sits in a town, so P beats A when the
# curated name is ambiguous between the two ("Richmond" the city vs "Richmond"
# the county) -- except when the curated name is itself an admin unit, which is
# what the ADM-only pass at the end is for.
_P_CODES = ["PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPL", "PPLX"]
_A_CODES = ["ADM1", "ADM2", "ADM3", "PCLI"]


def _search(es, name, admin1, iso3, classes):
    """Best gazetteer row for a curated home name, or None."""
    must = [{"match_phrase": {"asciiname": name}}]
    if iso3:
        must.append({"term": {"country_code3": iso3}})
    if admin1:
        must.append({"match_phrase": {"admin1_name": admin1}})
    body = {
        "size": 50,
        "query": {"bool": {"must": must,
                           "filter": [{"terms": {"feature_code": classes}}]}},
    }
    resp = es.search(index="geonames", body=body, request_timeout=60)
    hits = [h["_source"] for h in resp["hits"]["hits"]]
    # Exact (case-insensitive) name match only; match_phrase is a recall net.
    hits = [h for h in hits
            if str(h.get("asciiname", "")).lower() == name.lower()
            or str(h.get("name", "")).lower() == name.lower()]
    if not hits:
        return None

    def pop(h):
        try:
            return int(h.get("population") or 0)
        except (TypeError, ValueError):
            return 0

    hits.sort(key=lambda h: (-pop(h), _rank(h, classes)))
    return hits[0]


def _rank(h, classes):
    try:
        return classes.index(str(h.get("feature_code")))
    except ValueError:
        return len(classes)


def _country_row(es, iso3):
    """The PCLI row for a country code, for national outlets."""
    body = {"size": 5,
            "query": {"bool": {"must": [{"term": {"country_code3": iso3}}],
                               "filter": [{"terms": {"feature_code": ["PCLI", "PCL"]}}]}}}
    resp = es.search(index="geonames", body=body, request_timeout=60)
    hits = [h["_source"] for h in resp["hits"]["hits"]]
    return hits[0] if hits else None


def researched_table(path=None):
    """The independently researched table (e53), in :data:`HOME`'s shape.

    Built by two web research passes that saw only the 120 domain strings; the
    data and its per-row source URLs live in
    ``tools/data/outlet_homes_researched.tsv``.  Convention difference worth
    knowing: a row with a place gets a point-level home even when its scope is
    "national", so Haaretz resolves to Tel Aviv here and to Israel-the-country
    in the curated table.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "outlet_homes_researched.tsv")
    out = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = (line.rstrip("\n").split("\t") + [""] * 7)[:7]
            domain, place, admin1, iso3, scope = cols[0], cols[1], cols[2], cols[3], cols[4]
            place = None if place.strip().upper() in ("", "NONE") else place.strip()
            out[domain.strip()] = (place, admin1.strip() or None,
                                   iso3.strip() or None,
                                   "local" if place else "national")
    return out


def geocode_homes(es, verbose=True, table=None):
    """domain -> resolved home dict, built purely from :data:`HOME` + GeoNames.

    The resolved dict is what the feature code reads:

    ``lat``/``lon``
        the newsroom point, or ``None`` for a national outlet
    ``country_code3``
        the home country, set for local and national outlets alike
    ``admin1``
        ``(country_code3, admin1_code)`` of the home point, or ``None``
    ``level``
        ``"point"`` or ``"country"``
    """
    out = {}
    for domain, entry in sorted((table if table is not None else HOME).items()):
        if entry is None:
            continue
        place, admin1, iso3, scope = entry
        if scope == "national" or not place:
            if not iso3:
                continue          # a wire service: genuinely no home at all
            row = _country_row(es, iso3)
            out[domain] = {"lat": None, "lon": None, "country_code3": iso3,
                           "admin1": None, "level": "country",
                           "resolved": (row or {}).get("name")}
            continue
        row = (_search(es, place, admin1, iso3, _P_CODES)
               or _search(es, place, admin1, iso3, _A_CODES))
        if row is None:
            if verbose:
                print(f"  UNRESOLVED {domain}: {place}, {admin1}, {iso3}")
            continue
        lat, _, lon = str(row["coordinates"]).partition(",")
        a1 = str(row.get("admin1_code") or "").strip()
        out[domain] = {
            "lat": float(lat), "lon": float(lon),
            "country_code3": str(row.get("country_code3") or ""),
            "admin1": (str(row.get("country_code3") or ""), a1) if a1 else None,
            "level": "point",
            "resolved": row.get("name"),
            "geonameid": row.get("geonameid"),
            "feature_code": row.get("feature_code"),
            "admin1_name": row.get("admin1_name"),
        }
    return out


def load_or_build(es=None, cache_path=None, verbose=True, table=None):
    """Resolved homes, from ``cache_path`` if present, else geocoded and saved."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf8") as f:
            return json.load(f)
    if es is None:
        raise ValueError("no cache at {} and no ES client to build one".format(cache_path))
    homes = geocode_homes(es, verbose=verbose, table=table)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf8") as f:
            json.dump(homes, f, indent=1, sort_keys=True)
    return homes
