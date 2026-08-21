"""Read-only helpers for talking to the shared geonames index.

HARD RULE for this experiment: the live `geonames` index is shared with other
running agents.  Everything here is a GET/_mget/_search.  Nothing in this file
(or anything that imports it) writes, deletes, updates or reindexes.
"""
import json
import urllib.request

ES = "http://localhost:9200"
INDEX = "geonames"


def _post(path, body):
    req = urllib.request.Request(
        ES + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def mget(gids, fields=None):
    """{geonameid: _source} for a list of ids, in chunks."""
    out = {}
    gids = [str(g) for g in dict.fromkeys(gids) if g and str(g) != "NULL"]
    for i in range(0, len(gids), 1000):
        chunk = gids[i:i + 1000]
        body = {"docs": [{"_index": INDEX, "_id": g} for g in chunk]}
        if fields is not None:
            body["_source"] = fields
        resp = _post("/_mget", body)
        for d in resp["docs"]:
            if d.get("found"):
                out[d["_id"]] = d["_source"]
    return out


def phrase_search(name, size=100, sort_altname=True):
    """The exact query GeonamesService.build_name_search sends (post-cleaning).

    Kept byte-compatible with mordecai3/geonames.py so a prototype measured
    here means the same thing in the serving path.
    """
    body = {
        "query": {"multi_match": {"query": name,
                                  "fields": ["name", "asciiname",
                                             "alternativenames"],
                                  "type": "phrase"}},
        "size": size,
    }
    if sort_altname:
        body["sort"] = [{"alt_name_length": {"order": "desc"}}]
    resp = _post("/%s/_search" % INDEX, body)
    return [h["_source"] for h in resp["hits"]["hits"]]


def count(query):
    return _post("/%s/_count" % INDEX, {"query": query})["count"]
