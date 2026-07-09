"""
features.py
One Piece NLP Project — parsing & feature engineering layer.

Stage 1 (this version): parsing.
  Opens the Char Box infobox dictionary, extracts the bounty (with edge-case
  handling), and produces clean description text. The whole infobox is opened
  into ib_* columns so no information is lost; downstream code chooses which
  fields to use as features.

Entry point:
  parse_dataframe(raw_df) -> parsed DataFrame (one row per character)
"""

import re
import numpy as np
import pandas as pd
import mwparserfromhell


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------
def find_char_box(wikicode):
    """Return the One Piece 'Char Box' infobox template, or None."""
    for tpl in wikicode.filter_templates():
        if str(tpl.name).strip().lower().replace("_", " ") == "char box":
            return tpl
    return None


# Cross Guild bounties are placed BY the Cross Guild ON the Marines (the Fleet
# Admiral, the three Admirals, Garp) — the INVERSE of a World-Government bounty.
# In the Char Box bounty field they appear as a star rating {{C|N}}, e.g.
# {{C|3}} = 3 stars = 3,000,000,000. Ordinary (WG) bounties use {{B}} instead.
_CG_RE = re.compile(r"\{\{\s*C\s*\|\s*(\d+)")


def cross_guild_stars(raw_text):
    """Return the Cross Guild star count N when the character's bounty is a
    Cross Guild one ({{C|N}} in the bounty field), else 0. The 'bounty' column
    keeps the figure for everyone; this lets a CG bounty be told apart by its
    stars (and N billion ~= the figure). Fast-skips pages with no {{C| marker."""
    raw_text = str(raw_text)
    if not _CG_RE.search(raw_text):          # cheap skip: no Cross Guild marker
        return 0
    box = find_char_box(mwparserfromhell.parse(raw_text))
    if box is None or not box.has("bounty"):
        return 0
    m = _CG_RE.search(str(box.get("bounty").value))
    return int(m.group(1)) if m else 0


# Star-rated World-Government "bounties" — the INVERSE of a pirate wanted
# bounty. Two template forms appear in the Char Box bounty field:
#   {{B|s}} - Marine officer ratings (★ = 100,000,000): Smoker, Koby, Hina, ...
#   {{C|N}} - Cross Guild ratings on the admirals (★ = 1,000,000,000).
# Both are power valuations, not "wanted" sums, and on the One Piece wiki carry
# a {{Qref|special=marinebounties}} citation. They are flagged so they can be
# told apart from — or excluded from — the real wanted bounties.
_STAR_BOUNTY_RE = re.compile(r"\{\{\s*[Bb]\s*\|\s*s\b|\{\{\s*C\s*\|\s*\d+")


def is_star_bounty(raw_text):
    """True when the character's bounty is a star-rating WG valuation
    ({{B|s}} Marine rating or {{C|N}} Cross Guild rating) rather than a plain
    {{B}} wanted bounty. Reads the Char Box bounty field only, so prose mentions
    of other characters' bounties never trigger it."""
    box = find_char_box(mwparserfromhell.parse(str(raw_text)))
    if box is None or not box.has("bounty"):
        return False
    return bool(_STAR_BOUNTY_RE.search(str(box.get("bounty").value)))


def split_links(value):
    """Wiki-link targets in a value -> list of names (for relationship fields)."""
    names = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", value)
    return [n.strip() for n in names if n.strip()]


def clean_value(value):
    """Clean one infobox value to plain text: drop templates, unwrap links, strip markup."""
    value = re.sub(r"\{\{[Nn]ihongo\|([^|}]*).*?\}\}", r"\1", value, flags=re.DOTALL)
    while re.search(r"\{\{[^{}]*\}\}", value):
        value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", value).strip(" ;,").strip()


# ----------------------------------------------------------------------
# Field extractors
# ----------------------------------------------------------------------
def parse_char_box(raw_text):
    """Open the whole Char Box into a dict of cleaned fields, keys prefixed 'ib_'."""
    box = find_char_box(mwparserfromhell.parse(raw_text))
    out = {}
    if box is None:
        return out
    for p in box.params:
        key = "ib_" + p.name.strip().lower().replace(" ", "_")
        out[key] = clean_value(str(p.value))
    return out


def extract_affiliations(raw_text):
    """Affiliation names as a list (for the relationship graph)."""
    box = find_char_box(mwparserfromhell.parse(raw_text))
    if box is None or not box.has("affiliation"):
        return []
    return split_links(str(box.get("affiliation").value))


def extract_categories(raw_text):
    """
    All [[Category:...]] tags as a list. These exist on EVERY page — including
    Tabs pages (Luffy, Law, ...) whose Char Box is empty — so they recover the
    structured info (crew, race, role, Haki, Devil Fruit, canon status) that the
    infobox misses for the biggest characters. Sort-key suffixes ('|...') are
    stripped.
    """
    cats = re.findall(r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", raw_text)
    return [c.strip() for c in cats if c.strip()]


# Category-name suffixes that denote crew/organization membership. Used to
# recover graph edges for Tabs characters whose infobox affiliation is empty.
_GRAPH_CAT_SUFFIXES = (" Members", " Pirates", " Crew")

# Large organizations: a character tagged with one of these (even via a rank
# category like 'Marine Vice Admirals') should be linked to the organization
# in the graph, so figures like Garp are not isolated nodes. Maps a detector
# substring -> canonical node name.
_GRAPH_ORGS = {
    "Marine": "Marines",
    "Revolutionary": "Revolutionary Army",
    "Cipher Pol": "Cipher Pol",
    "Celestial Dragon": "World Nobles",
    "World Noble": "World Nobles",
    "Five Elders": "World Government",
    "Warlords of the Sea": "Seven Warlords of the Sea",
}
_GRAPH_ORG_EXCLUDE = ("prisoner", "slave", "victim", "captive", "ally", "allies",
                      "enemy", "enemies", "hunter", "non-canon", "marineford")


def graph_affiliations(affiliations, categories):
    """
    Merge infobox affiliations with crew/org membership inferred from
    categories, so every character (including Tabs pages) has graph edges.
    Two sources from categories: (1) crew categories like 'Straw Hat Pirates
    Members' -> 'Straw Hat Pirates'; (2) membership in a large organization
    via a rank category like 'Marine Vice Admirals' -> 'Marines', so solo
    figures (Garp, warlords) are linked rather than isolated. Excludes
    prisoner/slave/ally tags. Returns a de-duplicated list.
    """
    out = list(affiliations) if affiliations else []
    for c in categories or []:
        name = c
        if name.endswith(" Members"):
            name = name[:-len(" Members")]
        if (c.endswith(_GRAPH_CAT_SUFFIXES)
                and "Non-Canon" not in c and "by " not in c):
            if name not in out:
                out.append(name)
    # link to large organizations from rank/role categories
    for c in categories or []:
        cl = c.lower()
        if any(x in cl for x in _GRAPH_ORG_EXCLUDE):
            continue
        for needle, node in _GRAPH_ORGS.items():
            nl = needle.lower()
            is_word = (cl == nl or cl.startswith(nl + " ")
                       or cl.endswith(" " + nl) or (" " + nl + " ") in cl)
            if is_word and node not in out:
                out.append(node)
    # de-duplicate while preserving order (the infobox can list a crew twice)
    seen, deduped = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _bounty_numbers(text):
    """All numbers written right after a {{B}}/{{b}}/{{Bounty}} beli template
    (the reliable bounty marker on the One Piece wiki).

    FORMER bounties are written with a strikethrough between the template and
    the figure, e.g. {{B}}<s>438,000,000</s>; HTML tags (<s>, </s>, <br/>) are
    therefore stripped FIRST, otherwise the regex stops at '<' and only the
    current (first) figure is captured — losing the whole history (this is why
    bounty_min used to equal bounty_max for almost every character). Wiki
    bold/italic markup ('' / ''') is likewise stripped, because editors
    sometimes bold part of a figure (e.g. {{B}}5,0'''46,000,000), which would
    otherwise truncate it."""
    text = re.sub(r"<[^>]+>", "", text)                # drop <s>, </s>, <br/>, ...
    text = text.replace("'''", "").replace("''", "")
    out = []
    for m in re.findall(r"\{\{[Bb](?:ounty)?(?:\|[^}]*)?\}\}\s*([\d,]+)", text):
        try:
            v = int(m.replace(",", ""))
            if v > 0:
                out.append(v)
        except ValueError:
            continue
    return out


# Top characters whose Char Box (the infobox holding the bounty field) is on a
# transcluded sub-page that is not part of our dump, so their bounty appears
# only in the article prose. For these — and only these — the current bounty is
# taken as the highest {{B}} figure their OWN page cites (verified against the
# wiki to be theirs). No bounty values are hardcoded; the figures are read from
# the text. Detected structurally, never used as a feature.
_BODY_BOUNTY_NAMES = {
    "Shanks", "Kaidou", "Edward Newgate", "Buggy",
    "Portgas D. Ace", "Crocodile", "Monkey D. Garp", "Sabo",
}


def _all_bounty_numbers(raw_text):
    """Every {{B}} figure anywhere on the page (used only for the handful of
    body-bounty characters above)."""
    return sorted(set(_bounty_numbers(raw_text)))


def _strip_possessor_bounties(text):
    """Blank out OTHER characters' bounties so a page does not inherit them,
    e.g. on a relative's page: "Luffy's bounty of {{B}}3,000,000,000" or
    "[[Luffy]]'s ... {{B}}3,000,000,000". Only explicit possessives (an
    apostrophe-s right after a name or link) are removed, so the subject's own
    bounty ("with a bounty of {{B}}N", "his bounty is {{B}}N") is kept."""
    pat = r"(?:\[\[[^\]]+\]\]|[A-Z][\w.\- ]{1,30})'s[^.{}]{0,30}?\{\{[Bb][^}]*\}\}\s*[\d,]+"
    return re.sub(pat, " ", text)


def own_bounty_values(raw_text):
    """All of THIS character's own bounty figures (the full history the page
    records), as a sorted list of distinct values.

    Source (high precision — a wrong label poisons training, a miss only drops
    a row):
      1. Char Box infobox 'bounty' field — lists every value the character has
         held (current + frozen/former). Former values are struck through
         (<s>...</s>) and are KEPT, because they are the character's own past
         bounties and form the history (min / number of changes); the strike
         marks "superseded / or deceased", never "wrong".
      2. Tabs pages with no infobox bounty field: the {{B}} numbers in the LEAD
         section (before the first '==' header), after removing other people's
         (possessive) bounties. The lead describes this character, so a top
         character's own current bounty is stated there, while crewmates'/
         relatives' figures (which live in the body) are not picked up. This is
         why a page like Garp's — which mentions Luffy's bounty only in the body
         — correctly yields no bounty for Garp.
    Returns [] when nothing is found.
    """
    box = find_char_box(mwparserfromhell.parse(raw_text))
    if box is not None and box.has("bounty"):
        vals = _bounty_numbers(str(box.get("bounty").value))
        if vals:
            return sorted(set(vals))
    lead = re.split(r"\n==", raw_text, maxsplit=1)[0]
    vals = _bounty_numbers(_strip_possessor_bounties(lead))
    return sorted(set(vals))


def extract_bounty(raw_text):
    """Current (highest) bounty for this character = max of own_bounty_values.
    None when no own-bounty is found (the character drops from the supervised
    target). Use parse_dataframe(..., overrides=...) to supply manually-verified
    values for the few top characters whose bounty lives in the page body."""
    vals = own_bounty_values(raw_text)
    return max(vals) if vals else None

def _scrub_bounty_leakage(text):
    """Remove explicit bounty figures from body prose so the text-only model
    cannot read the target. Strips comma-grouped numbers (1,000,000,000),
    long bare digit runs, and the beli/berry currency words. The word
    'bounty' is kept (it carries danger semantics, not the value)."""
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", " ", text)      # 1,000,000,000
    text = re.sub(r"\b\d{4,}\b", " ", text)                    # 50000000
    text = re.sub(r"\b(?:beli|belly|berries|berry)\b", " ", text, flags=re.IGNORECASE)
    return text


def clean_text(raw_text):
    """Strip wikitext to plain readable prose for language models."""
    wikicode = mwparserfromhell.parse(raw_text)
    for tpl in reversed(list(wikicode.filter_templates())):
        try:
            wikicode.remove(tpl)
        except ValueError:
            continue
    text = wikicode.strip_code().strip()
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = _scrub_bounty_leakage(text)
    return re.sub(r"\s+", " ", text).strip()


def has_d_in_name(name):
    """'Will of D.' naming pattern (a metadata feature)."""
    return bool(re.search(r"\bD\.", str(name)))


# ----------------------------------------------------------------------
# Entry point — parse the whole raw DataFrame
# ----------------------------------------------------------------------
def is_crew_or_org(raw_text):
    """True if the page is a crew/organization page (Crew Box / Organization
    Box), whose 'bounty' is an aggregate crew total, NOT an individual value.
    Such pages must be excluded from the supervised target (but can stay as
    graph nodes)."""
    for tpl in mwparserfromhell.parse(raw_text).filter_templates():
        nm = str(tpl.name).strip().lower().replace("_", " ")
        if nm in ("crew box", "organization box", "company box"):
            return True
    return False


def clean_raw_dataframe(raw_df):
    """Clean the raw collection BEFORE parsing.

    The wiki dump contains non-character pages that pollute the dataset:
      * Marine base pages ("153rd Branch", "16th Branch", ...) — locations, not
        characters. Dropped (any name containing the word "Branch").
      * Sub-pages written as "Name/Gallery", "Name/Atlas", etc. A "/Gallery"
        page is just a block of images (no prose) and duplicates its base
        character, so these are dropped — e.g. "Tony Tony Chopper/Gallery" goes
        away while "Tony Tony Chopper" stays. (Merging them in would only add
        image-file names as noise to the text.)

    Returns a cleaned copy with one row per real page.
    """
    df = raw_df.copy()
    df["description"] = df["description"].fillna("").astype(str)
    is_branch = df["name"].str.contains(r"\bBranch\b", case=False, na=False)
    is_subpage = df["name"].str.contains("/", na=False)
    df = df[~(is_branch | is_subpage)].reset_index(drop=True)
    return df


def drop_non_characters(df):
    """Remove pages that are not individual characters: merchandise, video
    games, items, weapons, tribes, units and other group/object pages that
    slipped into the dump (e.g. "One Piece Miracle Battle Carddass", "Pacifista",
    "Bazooka Unit").

    A row is KEPT when any of these holds:
      * it has a character category (a "... Characters" category, plural), or
      * it has a bounty (top characters on Tabs pages list only Warlord/Ruler
        categories and have no "Characters" category, but clearly are
        characters — e.g. Boa Hancock), or
      * it is a crew/organization page (is_crew — kept and flagged, handled
        separately by the modelling, never used as a supervised target).
    Everything else is dropped. No bounty-bearing character is removed.
    """
    char_cat = df["categories"].apply(
        lambda cs: any("Characters" in c for c in (cs or [])))
    is_crew = df["is_crew"] if "is_crew" in df.columns else False
    keep = char_cat | df["bounty"].notna() | is_crew
    return df[keep].reset_index(drop=True)


def parse_dataframe(raw_df, overrides=None):
    """
    Parse every row of the raw collection DataFrame.
    Input columns:  name | pageid | url | description (raw wikitext)
    Output: one row per character with:
      name, pageid, url, bounty, has_D, affiliations, clean_text, raw_length,
      and all ib_* infobox fields (the opened dictionary).
    """
    records = []
    for _, row in raw_df.iterrows():
        raw = row.get("description", "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        name = row["name"]
        is_crew = is_crew_or_org(raw)

        if is_crew:
            bounty, bvals = None, []
        else:
            bvals = own_bounty_values(raw)
            bounty = max(bvals) if bvals else None
            # top characters whose infobox lives off-page: read their current
            # (highest) bounty from their own prose
            if bounty is None and name in _BODY_BOUNTY_NAMES:
                allv = _all_bounty_numbers(raw)
                if allv:
                    bounty = max(allv)
                    bvals = [bounty]
        if overrides and name in overrides:      # optional manual override
            bounty = overrides[name]
            bvals = [bounty]

        cg_stars = 0 if is_crew else cross_guild_stars(raw)
        star_bounty = False if is_crew else is_star_bounty(raw)

        rec = {
            "name": name,
            "pageid": row["pageid"],
            "url": row.get("url", ""),
            "is_crew": is_crew,
            "bounty": bounty,
            "bounty_values": bvals,            # bounty history (min/max/changes)
            "cross_guild_stars": cg_stars,     # N from {{C|N}}, else 0
            "is_cross_guild_bounty": cg_stars > 0,  # bounty placed ON a Marine
            "is_star_bounty": star_bounty,     # star-rated WG valuation ({{B|s}} or {{C|N}})
            "has_D": has_d_in_name(name),
            "affiliations": extract_affiliations(raw),
            "categories": extract_categories(raw),
            "clean_text": clean_text(raw),
            "raw_length": len(raw),
        }
        rec["graph_affiliations"] = graph_affiliations(
            rec["affiliations"], rec["categories"])
        rec.update(parse_char_box(raw))
        records.append(rec)
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Post-parse helpers
# ----------------------------------------------------------------------
def add_non_canon_flag(df):
    """
    Add boolean 'is_non_canon'. Combines reliable signals:
    (1) the character's BASE category is non-canon ('Non-Canon Humans',
        'Non-Canon Male/Female Characters', etc.) — marks the character itself
        as non-canon, unlike trait tags ('Non-Canon Swordsmen') on otherwise
        canon characters, which must NOT trigger the flag;
    (2) the opening of the description states it (non-canon / anime-only /
        filler / movie-only);
    (3) cross-franchise guests from Shonen Jump crossover games (e.g. the
        Dragon Ball 'Caramel Man' robots) — these are not One Piece characters
        at all. They are identified by an 'Unnamed Jump World' origin or a
        'One-Shots' / 'Crossover' category, and have no bounty.
    """
    df = df.copy()

    base_nc = ("non-canon humans", "non-canon male characters",
               "non-canon female characters", "non-canon characters",
               "non-canon fish-men", "non-canon giants", "non-canon minks",
               "non-canon dwarves", "non-canon animals")

    def from_cats(cats):
        cats = cats or []
        if any(c.lower() in base_nc for c in cats):
            return True
        # cross-franchise: a 'One-Shots' or 'Crossover' category
        if any(("one-shot" in c.lower() or "crossover" in c.lower())
               for c in cats):
            return True
        nc = [c for c in cats if "non-canon" in c.lower()]
        if nc and any("crossover" in c.lower() for c in cats):
            return True
        return False

    def from_origin(origin):
        o = str(origin or "").lower()
        return "jump world" in o or "jump super" in o

    pat = r"non-canon|anime-only|anime only|filler|movie-only|movie only|game-only|novel-only"
    opening = df["clean_text"].fillna("").str.lower().str[:200]
    flag = (df["categories"].apply(from_cats)
            | opening.str.contains(pat, regex=True, na=False))
    if "ib_origin" in df.columns:
        flag = flag | df["ib_origin"].apply(from_origin)
    df["is_non_canon"] = flag
    return df


# Columns worth keeping for modeling / analysis (the rest of the 60+ ib_*
# fields are voice actors, colours, image names, etc. — pure noise).
MODEL_COLUMNS = [
    "name", "pageid", "url",          # identifiers
    "bounty",                          # target
    "bounty_min", "bounty_max", "bounty_n_values", "bounty_changes",  # bounty history
    "cross_guild_stars", "is_cross_guild_bounty", "is_star_bounty",  # identify star-rated WG valuations — analysis only
    "clean_text",                      # text features (NLP)
    "affiliations", "graph_affiliations", "categories",   # graph
    "is_non_canon",                    # flag
    "has_D", "raw_length",             # metadata
    # organization
    "is_world_government", "is_marine", "is_cipher_pol", "is_celestial",
    "is_gorosei", "is_pirate", "is_revolutionary", "is_bounty_hunter", "is_no_group",
    # special status
    "is_yonko", "is_warlord", "is_worst_generation", "is_captain",
    "is_grand_fleet", "is_yonko_crew",
    # abilities
    "has_devil_fruit", "df_type", "df_awakened",
    "has_haki", "has_armament_haki", "has_observation_haki", "has_conqueror_haki",
    # personal meta
    "race", "gender", "is_royalty", "is_deceased", "epithet_present", "origin_sea",
    # raw infobox kept for reference
    "ib_occupation", "ib_affiliation", "ib_origin", "ib_age", "ib_status",
]


def select_model_columns(df, columns=None):
    """Return a slim DataFrame with only the columns useful for modeling.
    Missing columns are skipped so it never errors on a different parse."""
    cols = columns if columns is not None else MODEL_COLUMNS
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


# ======================================================================
# Feature engineering from categories + infobox + text
# All flags derive from the data only (no external One Piece knowledge).
# Values are categorical strings/booleans for now; can be ranked to
# numbers later in the tuning stage.
# ======================================================================

def _has_cat(cats, *needles):
    """True if any category contains any of the needle substrings (case-insensitive)."""
    low = [c.lower() for c in (cats or [])]
    return any(any(n.lower() in c for c in low) for n in needles)


def _has_cat_word(cats, *needles):
    """True if any category EQUALS or ends with a needle (avoids 'Marineford' matching 'Marine')."""
    low = [c.lower() for c in (cats or [])]
    out = []
    for n in needles:
        nl = n.lower()
        out.append(any(c == nl or c.endswith(nl) or c.startswith(nl + " ") for c in low))
    return any(out)


def _cat_marks_org(cats, org_word, exclude=()):
    """
    True if the character is a MEMBER of an organization identified by org_word.
    A category counts only if org_word appears as a whole word and the category
    is not an exclusion (prisoner of / slave of / victim / ally, etc.).
    e.g. 'Marine Vice Admirals' -> marine; 'Marine Prisoners' -> NOT marine.
    """
    base_exclude = ("prisoner", "slave", "victim", "captive", "ally", "allies",
                    "enemy", "enemies", "alliance", "hunter", "hunters")
    exclude = tuple(e.lower() for e in exclude) + base_exclude
    ow = org_word.lower()
    for c in (cats or []):
        cl = c.lower()
        # word match: equals, starts with "word ", ends with " word", or " word "
        is_word = (cl == ow or cl.startswith(ow + " ")
                   or cl.endswith(" " + ow) or (" " + ow + " ") in cl)
        if not is_word:
            continue
        if any(x in cl for x in exclude):
            continue
        return True
    return False


def add_org_flags(df):
    """Organization membership flags (group A). From categories + graph
    affiliations. Excludes prisoner/slave/victim/ally categories so e.g. a
    pirate jailed by the Marines ('Marine Prisoners') is not mislabeled. Also
    checks crew names (a member of 'Blackbeard Pirates' is a pirate even if no
    'Pirates' category is present, as with former Admiral Kuzan)."""
    df = df.copy()
    cats = df["categories"]
    affs = df["graph_affiliations"]

    def aff_has(a, *words):
        return any(any(w.lower() in x.lower() for w in words) for x in (a or []))

    df["is_marine"]        = cats.apply(lambda c: _cat_marks_org(c, "Marine") or _cat_marks_org(c, "Marines"))
    df["is_cipher_pol"]    = cats.apply(lambda c: _has_cat(c, "cipher pol", "cp0", "cp9", "cp-0", "cp-9"))
    df["is_celestial"]     = cats.apply(lambda c: _has_cat(c, "celestial dragon", "world noble"))
    df["is_gorosei"]       = cats.apply(lambda c: _has_cat(c, "five elders"))
    df["is_world_government"] = (df["is_marine"] | df["is_cipher_pol"]
                                 | df["is_celestial"] | df["is_gorosei"]
                                 | cats.apply(lambda c: _cat_marks_org(c, "World Government")))
    # pirate: 'Pirates' category (not the Ninja-Pirate alliance), a pirate
    # role category, OR membership in a crew whose name ends in 'Pirates'.
    df["is_pirate"] = (
        cats.apply(lambda c: _cat_marks_org(c, "Pirates", exclude=("ninja-pirate", "mink-samurai")))
        | cats.apply(lambda c: _has_cat(c, "pirate captains", "pirate officers", "pirate apprentices"))
        | affs.apply(lambda a: aff_has(a, "Pirates"))
    )
    df["is_revolutionary"] = (cats.apply(lambda c: _cat_marks_org(c, "Revolutionary Army")
                                                   or _has_cat(c, "revolutionaries"))
                              | affs.apply(lambda a: aff_has(a, "Revolutionary")))
    df["is_bounty_hunter"] = cats.apply(lambda c: _has_cat(c, "bounty hunter"))
    df["is_no_group"]      = ~(df["is_world_government"] | df["is_pirate"]
                               | df["is_revolutionary"] | df["is_bounty_hunter"])
    return df


def add_special_status(df):
    """High-profile status flags (group C). From categories."""
    df = df.copy()
    cats = df["categories"]
    # EXACT 'Four Emperors' / 'Yonko' category only. A substring match is wrong
    # here: 'Four Emperors Officers' (Cross Guild officers like Crocodile and
    # Mihawk) contains 'Four Emperors' but is NOT a Yonko.
    df["is_yonko"]            = cats.apply(
        lambda c: any(x.lower() in ("four emperors", "yonko") for x in (c or [])))
    df["is_warlord"]          = cats.apply(lambda c: _has_cat(c, "Seven Warlords", "Warlords of the Sea"))
    df["is_worst_generation"] = cats.apply(lambda c: _has_cat(c, "Worst Generation", "Super Rookies"))
    df["is_captain"]          = cats.apply(lambda c: _has_cat(c, "Pirate Captains"))
    df["is_grand_fleet"]      = df["graph_affiliations"].apply(
        lambda a: any("Grand Fleet" in x for x in (a or [])))
    # crew belongs to a Yonko
    yonko_crews = ("Beasts Pirates", "Big Mom Pirates", "Red Hair Pirates",
                   "Blackbeard Pirates", "Whitebeard Pirates")
    df["is_yonko_crew"] = df["graph_affiliations"].apply(
        lambda a: any(x in yonko_crews for x in (a or [])))
    return df


def add_ability_features(df):
    """Devil Fruit & Haki features (group D). From categories + infobox."""
    df = df.copy()
    cats = df["categories"]

    # Devil Fruit
    df["has_devil_fruit"] = (df["ib_dftype"].notna()
                             | cats.apply(lambda c: _has_cat(c, "Devil Fruit Users")))
    df["df_type"] = df["ib_dftype"].where(df["ib_dftype"].notna(), None)
    df["df_awakened"] = cats.apply(lambda c: _has_cat(c, "Awakened"))

    # Haki (one column per type, as requested)
    df["has_armament_haki"]   = cats.apply(lambda c: _has_cat(c, "Armament Haki"))
    df["has_observation_haki"] = cats.apply(lambda c: _has_cat(c, "Observation Haki"))
    df["has_conqueror_haki"]  = cats.apply(lambda c: _has_cat(c, "Supreme King Haki"))
    df["has_haki"] = (df["has_armament_haki"] | df["has_observation_haki"]
                      | df["has_conqueror_haki"]
                      | cats.apply(lambda c: _has_cat(c, "Haki Users")))
    return df


_RACE_CATS = ["Humans", "Fish-Men", "Merfolk", "Giants", "Dwarves", "Mink Tribe",
              "Lunarians", "Longarms", "Longlegs", "Cyborgs", "Skypieans"]
_SEA_CATS = ["East Blue", "West Blue", "North Blue", "South Blue", "Grand Line", "Sky Island"]


def add_meta_features(df):
    """Personal metadata (group E). From categories + infobox."""
    df = df.copy()
    cats = df["categories"]

    def first_match(c, options):
        for opt in options:
            if _has_cat(c, opt):
                return opt
        return None

    df["race"] = cats.apply(lambda c: first_match(c, _RACE_CATS))
    df["gender"] = cats.apply(
        lambda c: "Transgender" if _has_cat(c, "Transgender")
        else "Female" if _has_cat(c, "Female Characters")
        else "Male" if _has_cat(c, "Male Characters") else None)
    df["is_royalty"] = cats.apply(lambda c: _has_cat(
        c, "Kings", "Queens", "Princes", "Princesses", "Royalty", "Monarchs"))
    df["is_deceased"] = cats.apply(lambda c: _has_cat(c, "Deceased"))
    df["epithet_present"] = df["ib_epithet"].notna() if "ib_epithet" in df.columns else False
    # origin sea: from infobox origin text first, else categories
    def sea_from(row):
        org = str(row.get("ib_origin") or "")
        for s in _SEA_CATS:
            if s in org:
                return s
        return None
    df["origin_sea"] = df.apply(sea_from, axis=1)
    return df


def add_bounty_features(df):
    """Bounty-history features derived purely from the data (group B).

    From the list of the character's own bounty values (bounty_values):
      bounty_min        - lowest figure ever recorded (their first/frozen one)
      bounty_max        - highest figure (== the target 'bounty')
      bounty_n_values   - count of distinct figures the page records
      bounty_changes    - number of bounty "stages" the page records = the
                          count of distinct figures, +1 for a Seven Warlord
                          (a Warlord's bounty is frozen on appointment and only
                          reinstated after the system is abolished — an extra
                          stage the figures alone don't show). The struck "0"
                          (nobody) figure is not counted (only values > 0).
                          0 when the character has no bounty. Warlord status is
                          detected FROM THE DATA (the 'Seven Warlords of the
                          Sea' category, reliable and false for non-members),
                          never a hardcoded name list.

    These describe a character's bounty trajectory, not its level: bounty_max is
    the target, and bounty_min can equal it for single-bounty characters, so
    NEITHER is fed to the model (kept for reference only). The model uses only
    bounty_n_values and bounty_changes (see STRUCTURED_NUM in modeling.py).
    """
    df = df.copy()
    vals = df["bounty_values"] if "bounty_values" in df.columns else None
    if vals is None:
        df["bounty_min"] = df["bounty_max"] = df["bounty_n_values"] = df["bounty_changes"] = np.nan
        return df

    # When the page text yields no value list but a (manually verified) target
    # bounty exists, treat that single figure as the known history point.
    def value_list(row):
        v = list(row["bounty_values"] or [])
        if not v and pd.notna(row.get("bounty")):
            v = [int(row["bounty"])]
        return v
    vlist = df.apply(value_list, axis=1)

    df["bounty_min"]      = vlist.apply(lambda v: min(v) if v else np.nan)
    df["bounty_max"]      = vlist.apply(lambda v: max(v) if v else np.nan)
    df["bounty_n_values"] = vlist.apply(len).astype(int)

    # is_warlord may already exist (add_special_status); compute from category
    # here too so the function is self-contained and order-independent.
    warlord = (df["is_warlord"] if "is_warlord" in df.columns
               else df["categories"].apply(
                   lambda c: _has_cat(c, "Seven Warlords", "Warlords of the Sea")))
    # changes = number of bounty STAGES = count of distinct figures recorded,
    # +1 for a Seven Warlord. The struck "0" (nobody) figure is not counted
    # because _bounty_numbers keeps only values > 0.
    df["bounty_changes"] = (df["bounty_n_values"]
                            + warlord.astype(int)).astype(int)
    # characters with no bounty have no trajectory
    df.loc[df["bounty_n_values"] == 0, "bounty_changes"] = 0
    return df


def add_all_features(df):
    """Run the full feature-engineering pipeline (groups A, B, C, D, E)."""
    df = add_org_flags(df)
    df = add_special_status(df)
    df = add_ability_features(df)
    df = add_meta_features(df)
    df = add_bounty_features(df)    # needs is_warlord from add_special_status
    return df

# ======================================================================
# Graph features (network model)
# Build a character–character graph: two characters are connected if they
# share a graph-affiliation (same crew / unit / organization). Edge weight
# = number of shared groups. Centrality on this projection gives each
# character its "position in the world", independent of its description.
# ======================================================================
import networkx as nx


def build_character_graph(df, min_group_size=2, max_group_size=400):
    """Bipartite character<->group membership, projected to a weighted
    character graph. Huge hub groups (e.g. 'Marines' with hundreds of members)
    are capped out so they don't make the graph a single clique."""
    G = nx.Graph()
    names = list(df["name"])
    G.add_nodes_from(names)

    # group -> members
    groups = {}
    for name, affs in zip(df["name"], df["graph_affiliations"]):
        for g in (affs or []):
            groups.setdefault(g, set()).add(name)

    for g, members in groups.items():
        n = len(members)
        if n < min_group_size or n > max_group_size:
            continue
        members = list(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    return G


def add_graph_features(df):
    """Add network-topology columns computed on the character graph.
    These are the ONLY inputs to the network-only model."""
    df = df.copy()
    G = build_character_graph(df)

    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
    clust = nx.clustering(G, weight="weight") if G.number_of_edges() else {}
    core = nx.core_number(G) if G.number_of_edges() else {}
    # betweenness is O(VE); approximate with k samples for speed
    k = min(300, G.number_of_nodes()) if G.number_of_nodes() else 0
    btw = nx.betweenness_centrality(G, k=k, weight="weight", seed=42) if k > 2 else {}

    df["g_degree"]        = df["name"].map(deg).fillna(0).astype(float)
    df["g_weighted_deg"]  = df["name"].map(wdeg).fillna(0).astype(float)
    df["g_pagerank"]      = df["name"].map(pr).fillna(0.0)
    df["g_betweenness"]   = df["name"].map(btw).fillna(0.0)
    df["g_clustering"]    = df["name"].map(clust).fillna(0.0)
    df["g_core"]          = df["name"].map(core).fillna(0).astype(float)
    return df


GRAPH_FEATURE_COLS = [
    "g_degree", "g_weighted_deg", "g_pagerank",
    "g_betweenness", "g_clustering", "g_core",
]