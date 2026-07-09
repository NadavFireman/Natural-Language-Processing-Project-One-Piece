"""
data_collection.py
One Piece NLP Project — data collection layer (collection only).

This module ONLY collects raw data. All parsing / cutting (bounty, infobox,
flags, clean text) happens later in the main notebook / features.py.

Output per character: name | pageid | url | description
  description = the full raw wikitext of the page (single source of truth).

Stages:
  A.   fetch_all_pages(categories)  -> list of {pageid, name, url}
       Recursively walks sub-categories so a top category like "Characters"
       (which holds only sub-categories) yields every character page.
  B.   fetch_descriptions(pages)    -> adds 'description' (raw wikitext)
  B+.  fetch_tabs_infoboxes(pages)  -> recover the off-page infobox of Tabs pages
  B++. fetch_subpages(pages)        -> append body prose from subpages of Tabs pages
  C.   drop_noise(pages)            -> drop galleries / sub-pages / empty redirects

All network access goes through _request(), which retries with back-off and
honours maxlag, so a single bad response can neither kill a crawl nor silently
fill a batch with empty descriptions.
"""

import re
import time
import requests

BASE_URL = "https://onepiece.fandom.com/api.php"
USER_AGENT = "HIT-NLP-Project/1.0 (academic research)"
SLEEP_TIME = 0.2

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


# ----------------------------------------------------------------------
# Robust GET — retry + back-off + maxlag, JSON-guarded
# ----------------------------------------------------------------------
def _request(params, retries=3, sleep_time=SLEEP_TIME):
    """GET the API with retry/back-off, guarding against the HTML error pages
    Fandom sometimes returns instead of JSON. Returns {} only after giving up,
    so callers can tell a hard failure apart from a genuinely empty page."""
    params = {**params, "maxlag": 5}
    for attempt in range(retries):
        try:
            r = _SESSION.get(BASE_URL, params=params, timeout=30)
            if r.status_code == 429 or "MediaWiki-API-Error" in r.headers:
                time.sleep(sleep_time * (2 ** attempt) + 1)
                continue
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(sleep_time * (2 ** attempt) + 1)
    return {}


# ----------------------------------------------------------------------
# Low-level category query
# ----------------------------------------------------------------------
def _category_members(category, member_type, sleep_time=SLEEP_TIME):
    """
    Return members of a category. member_type is 'page' or 'subcat'.
    Follows continuation. Each item: {'pageid', 'title', 'type'}.
    """
    out, params = [], {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": member_type,
        "cmlimit": "500",
        "format": "json",
    }
    if member_type == "page":
        params["cmnamespace"] = "0"
    while True:
        data = _request(params, sleep_time=sleep_time)
        out.extend(data.get("query", {}).get("categorymembers", []))
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(sleep_time)
        else:
            break
    return out


# ----------------------------------------------------------------------
# Stage A — recursive page list
# ----------------------------------------------------------------------
def fetch_all_pages(categories, max_depth=2, sleep_time=SLEEP_TIME, verbose=True):
    """
    Stage A — collect a de-duplicated page list, walking sub-categories
    recursively up to max_depth. A top category such as "Characters"
    (only sub-categories, no direct pages) therefore yields every
    character page underneath it (Male, Female, Okama, ...).

    Returns list of dicts: {'pageid': int, 'name': str, 'url': str}.
    """
    seen_pages, seen_cats, pages = set(), set(), []

    def walk(category, depth):
        if category in seen_cats:
            return
        seen_cats.add(category)

        members = _category_members(category, "page", sleep_time)
        new = 0
        for m in members:
            if m["pageid"] not in seen_pages:
                seen_pages.add(m["pageid"])
                title = m["title"]
                pages.append({
                    "pageid": m["pageid"],
                    "name": title,
                    "url": "https://onepiece.fandom.com/wiki/" + title.replace(" ", "_"),
                })
                new += 1
        if verbose:
            indent = "  " * (depth + 1)
            print(f"{indent}{category:<34} {len(members):>5} pages, {new:>5} new")
        time.sleep(sleep_time)

        if depth < max_depth:
            for sub in _category_members(category, "subcat", sleep_time):
                sub_name = sub["title"].replace("Category:", "")
                walk(sub_name, depth + 1)

    for cat in categories:
        walk(cat, 0)

    if verbose:
        print(f"Total unique pages: {len(pages)}")
    return pages


# ----------------------------------------------------------------------
# Stage B — raw wikitext (the 'description')
# ----------------------------------------------------------------------
def _get_wikitext_batch(pageids):
    """Fetch raw wikitext for up to 50 pages in one request. {pageid: text}.
    An empty dict (no 'query') means the request itself failed — distinct from
    a real redirect, which comes back empty under a SUCCESSFUL response. With
    redirects=1 a redirect's text is served under its target's id, so the
    redirect's own id stays empty and drop_noise treats it as a redirect."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "pageids": "|".join(str(p) for p in pageids),
        "format": "json",
        "redirects": 1,
    }
    data = _request(params)
    if "query" not in data:                       # hard failure, NOT redirects
        return {}
    out = {}
    for pid, page in data["query"].get("pages", {}).items():
        try:
            out[int(pid)] = page["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            out[int(pid)] = ""
    return out


def fetch_descriptions(pages, batch_size=50, sleep_time=SLEEP_TIME, verbose=True):
    """
    Stage B — add 'description' (full raw wikitext) to every page, 50 per
    request. Pages from a batch that HARD-FAILED are retried once at the end,
    so a transient error never masquerades as an empty redirect and vanishes
    in drop_noise. Returns the same list with 'description' added.
    """
    by_id = {p["pageid"]: p for p in pages}
    ids = list(by_id.keys())
    failed = []
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        texts = _get_wikitext_batch(chunk)
        if not texts:                             # whole batch failed -> retry
            failed.extend(chunk)
        for pid in chunk:
            by_id[pid]["description"] = texts.get(pid, by_id[pid].get("description", ""))
        if verbose:
            print(f"  fetched {min(i + batch_size, len(ids)):>5} / {len(ids)}")
        time.sleep(sleep_time)

    if failed:
        if verbose:
            print(f"  retrying {len(failed)} pages from failed batches")
        for i in range(0, len(failed), batch_size):
            chunk = failed[i:i + batch_size]
            texts = _get_wikitext_batch(chunk)
            for pid in chunk:
                if texts.get(pid):
                    by_id[pid]["description"] = texts[pid]
            time.sleep(sleep_time)
    return list(by_id.values())


# ----------------------------------------------------------------------
# Stage B+ — recover off-page Char Box infoboxes (Tabs pages)
# ----------------------------------------------------------------------
_TABS_RE = re.compile(r"\{\{([^{}]*Tabs Top)\}\}")
# subpages that hold only images -> skipped by Stage B++
_SKIP_SUBPAGE_RE = re.compile(r"/(Gallery|Image Gallery)\b", re.I)


def _get_wikitext_by_titles(titles):
    """Fetch raw wikitext for up to 50 page TITLES in one request.
    Returns {requested_title: text}. Accounts for the title normalization the
    API applies, so the returned key matches the title that was requested."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": "|".join(titles),
        "format": "json",
    }
    out = {}
    data = _request(params)
    q = data.get("query", {})
    norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
    for _, page in q.get("pages", {}).items():
        title = page.get("title", "")
        orig = norm.get(title, title)
        try:
            out[orig] = page["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            out[orig] = ""
    return out


def fetch_tabs_infoboxes(pages, batch_size=50, sleep_time=SLEEP_TIME, verbose=True):
    """
    Recover the off-page Char Box for "Tabs" character pages (Luffy, Garp, Big
    Mom, Buggy, ...). A Tabs page holds no infobox of its own; its wikitext only
    transcludes one via {{<Name> Tabs Top}}, a separate template page that Stage
    B never fetched. As a result those characters' structured fields (full
    bounty history, occupation, affiliation, age, status) were missing. This
    stage finds every Tabs page, fetches the matching 'Template:<Name> Tabs Top'
    wikitext, and APPENDS it to that page's description so the parser recovers
    the infobox like any normal page. Returns the same list with descriptions
    enriched in place.
    """
    need = {}
    for p in pages:
        m = _TABS_RE.search(p.get("description", "") or "")
        if m:
            need.setdefault("Template:" + m.group(1), []).append(p["pageid"])
    if verbose:
        print(f"Tabs pages: {sum(len(v) for v in need.values())}, "
              f"unique templates: {len(need)}")

    by_id = {p["pageid"]: p for p in pages}
    titles = list(need.keys())
    for i in range(0, len(titles), batch_size):
        chunk = titles[i:i + batch_size]
        texts = _get_wikitext_by_titles(chunk)
        for t in chunk:
            body = texts.get(t, "")
            for pid in need[t]:
                if body:
                    by_id[pid]["description"] = (
                        (by_id[pid].get("description", "") or "") + "\n\n" + body)
        if verbose:
            print(f"  fetched {min(i + batch_size, len(titles)):>5} / {len(titles)} templates")
        time.sleep(sleep_time)
    return list(by_id.values())


# ----------------------------------------------------------------------
# Stage B++ — recover body prose that lives on character SUBPAGES (Tabs pages)
# ----------------------------------------------------------------------
def _list_subpages(title, sleep_time=SLEEP_TIME):
    """Return every namespace-0 subpage title under '<title>/...', skipping
    redirect subpages so we don't pull in '#REDIRECT' stubs."""
    out, params = [], {
        "action": "query",
        "list": "allpages",
        "apprefix": title + "/",
        "apnamespace": "0",
        "apfilterredir": "nonredirects",
        "aplimit": "500",
        "format": "json",
    }
    while True:
        data = _request(params, sleep_time=sleep_time)
        out.extend(p["title"] for p in
                   data.get("query", {}).get("allpages", []))
        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
            time.sleep(sleep_time)
        else:
            break
    return out


def fetch_subpages(pages, batch_size=50, sleep_time=SLEEP_TIME, verbose=True):
    """
    For "Tabs" (major) characters the body — Personality and Relationships,
    History, Abilities and Powers, Misc. — is NOT on the main page. It lives on
    uncategorized SUBPAGES ('<Name>/History', ...), which Stage A never collects
    and drop_noise would discard, so those characters came back as just an intro
    line + (after Stage B+) the infobox. This stage uses the same {{<Name> Tabs
    Top}} marker to find the major characters, enumerates each one's content
    subpages, fetches their wikitext, and APPENDS it to the parent's description
    so 'description' becomes the character's COMPLETE wiki page. Gallery / image
    subpages are skipped. No new rows are added. Returns the list enriched in
    place.
    """
    tabs_pages = [p for p in pages
                  if _TABS_RE.search(p.get("description", "") or "")]
    if verbose:
        print(f"Tabs (major) characters needing subpages: {len(tabs_pages)}")

    # 1) discover content subpages, remembering each one's parent
    parent_of, to_fetch = {}, []
    for p in tabs_pages:
        for sub in _list_subpages(p["name"], sleep_time):
            if _SKIP_SUBPAGE_RE.search(sub):
                continue
            parent_of[sub] = p["pageid"]
            to_fetch.append(sub)
        time.sleep(sleep_time)
    if verbose:
        print(f"  content subpages found: {len(to_fetch)}")

    # 2) fetch wikitext by title (50 at a time) and append onto the parent
    by_id = {p["pageid"]: p for p in pages}
    for i in range(0, len(to_fetch), batch_size):
        chunk = to_fetch[i:i + batch_size]
        texts = _get_wikitext_by_titles(chunk)
        for t in chunk:
            body = texts.get(t, "")
            if body:
                pid = parent_of[t]
                by_id[pid]["description"] = (
                    (by_id[pid].get("description", "") or "")
                    + f"\n\n== {t} ==\n" + body)
        if verbose:
            done = min(i + batch_size, len(to_fetch))
            print(f"  fetched {done:>5} / {len(to_fetch)} subpages")
        time.sleep(sleep_time)
    return list(by_id.values())


# ----------------------------------------------------------------------
# Stage C — drop pure-noise pages (collect a clean CSV)
# ----------------------------------------------------------------------
def drop_noise(pages, min_len=5, verbose=True):
    """
    Remove pure-noise pages so the saved CSV is clean:
      - sub-pages whose name contains '/' (image galleries like
        'Monkey D. Luffy/Gallery', and satellite sub-pages like
        'Vegapunk/Shaka') — these carry no usable descriptive text.
      - redirect stubs: pages whose wikitext came back empty. With
        redirects=1 the API serves a redirect's target under the target's
        pageid, so the redirect's own id gets no text; an empty page is
        therefore a redirect, and its real content already exists in the
        CSV under the target page.

    Character AND crew/organization pages are KEPT. The character vs.
    non-character split (and keeping crews for the graph) is done later in
    features.drop_non_characters, since crew pages such as 'Marines' or
    'Buggy Pirates' are needed as graph nodes.
    """
    def is_noise(p):
        name = p.get("name", "")
        desc = (p.get("description") or "").strip()
        return ("/" in name) or (len(desc) < min_len)

    before = len(pages)
    kept = [p for p in pages if not is_noise(p)]
    if verbose:
        n_sub = sum(1 for p in pages if "/" in p.get("name", ""))
        n_empty = sum(1 for p in pages
                      if len((p.get("description") or "").strip()) < min_len
                      and "/" not in p.get("name", ""))
        print(f"drop_noise: removed {before - len(kept)} "
              f"({n_sub} sub-pages/galleries, {n_empty} empty redirects); "
              f"{len(kept)} pages kept")
    return kept