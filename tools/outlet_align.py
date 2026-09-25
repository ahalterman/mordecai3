"""Align the entities in a Gritta-corpus pickle back to their source articles.

`tools/train.py:data_formatter` walks the corpus XML article by article and, for
each article, emits one entity per toponym that survives two filters (the
toponym's character span has to overlap a spaCy GPE/LOC token, and the entity
has to end up with a non-None ``correct_geonamesid``).  Everything downstream --
`format_source`, the enrichment, the positional train/held-out split -- preserves
that order, so the pickle is the corpus flattened: article after article, and
within an article, toponym after toponym, with gaps where a toponym was dropped.

Nothing in the pickle records which article an entity came from.  Two facts let
us put it back:

* entities from one article share a byte-identical ``doc_tensor`` (it is the mean
  over that document's token tensors), so consecutive runs of equal
  ``doc_key`` are exactly the surviving entities of one article; and
* an entity's ``search_name`` / ``correct_geonamesid`` are copied verbatim from
  the toponym's ``phrase`` / ``gaztag/@geonameid``.

So we walk the doc groups and the articles in lockstep and accept an article for
a group when the group's ``(phrase, geonameid)`` sequence is a subsequence of the
article's.  Articles that contributed nothing are skipped over.  The alignment is
rejected loudly rather than guessed at: :func:`align` raises if any group fails
to match, which is what makes the outlet join safe to build a feature on.
"""

import hashlib
import os

import xmltodict


CORPUS_PATHS = {
    "lgl": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml",
    "tr": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
}


def document_key(entity):
    """The same hash tools/enrich_pickles.py:document_key uses."""
    return hashlib.sha1(entity["doc_tensor"].tobytes()).hexdigest()


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def read_articles(path):
    """[(domain, [(phrase, geonameid), ...]), ...] in file order."""
    with open(path, "rb") as f:
        parsed = xmltodict.parse(f.read())
    out = []
    for art in _as_list(parsed["articles"]["article"]):
        topos = []
        toponyms = art.get("toponyms") or {}
        for topo in _as_list(toponyms.get("toponym")):
            gaz = topo.get("gaztag") or {}
            gid = gaz.get("@geonameid")
            topos.append((str(topo.get("phrase")), str(gid) if gid else None))
        out.append((art.get("domain"), topos))
    return out


def doc_groups(data):
    """Consecutive runs of equal doc_key -> [(doc_key, [indices]), ...]."""
    groups = []
    prev = None
    for i, ent in enumerate(data):
        key = document_key(ent)
        if key != prev:
            groups.append((key, []))
            prev = key
        groups[-1][1].append(i)
    return groups


def _is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(item in it for item in needle)


def align(data, articles, key="phrase+gid"):
    """entity index -> article index.  Raises if the alignment is not exact.

    Both sequences are in corpus order, so this is a single forward pass: each
    doc group must match the next article that can host it, and an article can
    host it only if the group's key sequence appears inside the article's in
    order.

    ``key`` selects the join key.  ``"phrase+gid"`` (the default) is the more
    constrained one; ``"phrase"`` uses only the mention strings and so touches no
    gold label at all.  ``tools/outlet_leak_audit.py`` runs both and reports
    where they differ -- on LGL, the source this feature is aimed at, they agree
    on every one of the 3,245 entities, which is what makes the outlet join
    demonstrably independent of the answer key.
    """
    groups = doc_groups(data)
    mapping = {}
    a = 0
    for gi, (_key, idxs) in enumerate(groups):
        if key == "phrase":
            want = [str(data[i]["search_name"]) for i in idxs]
        else:
            want = [(str(data[i]["search_name"]),
                     str(data[i]["correct_geonamesid"])) for i in idxs]
        matched = None
        while a < len(articles):
            if key == "phrase":
                have = [p for p, _g in articles[a][1]]
            else:
                have = [(p, g) for p, g in articles[a][1]]
            if _is_subsequence(want, have):
                matched = a
                a += 1
                break
            a += 1
        if matched is None:
            raise ValueError(
                f"doc group {gi} ({len(idxs)} entities, first "
                f"{want[:3]}) matched no article; ran off the end at {a}")
        for i in idxs:
            mapping[i] = matched
    return mapping


def entity_domains(data, source, data_dir, key="phrase+gid"):
    """entity index -> the outlet domain of the article it came from.

    Returns ``{}`` for a source with no domain metadata in its corpus file.
    """
    rel = CORPUS_PATHS.get(source)
    if rel is None:
        return {}
    path = os.path.join(data_dir, rel)
    if not os.path.exists(path):
        return {}
    articles = read_articles(path)
    mapping = align(data, articles, key=key)
    out = {}
    for i, ai in mapping.items():
        dom = articles[ai][0]
        if dom:
            out[i] = str(dom).strip().lower()
    return out
