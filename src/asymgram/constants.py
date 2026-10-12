from .conllu import Token

# ==== Punctuations ====

CHINESE_PUNCT_FORMS = "，。？！…（）；：、+—=“”‘’《》"

CHINESE_COMMA = Token({
    "id": -1,  # Temporary id, will be reassigned by TokenList.refresh_ids()
    "form": "，",
    "lemma": "_",
    "upos": "PUNCT",
    "xpos": "_",
    "feats": "_",
    "head": -1,  # Temporary head, will be reassigned after attachment
    "deprel": "punct",
    "deps": "_",
    "misc": "_"
})

# ==== UD Dependency Relations ====
# Core Deprels
CORE_DEPRELS = (
    "nsubj",    # nominal subject
    "obj",      # object
    "iobj",     # indirect object
    "obl",      # oblique nominal
    "vocative", # vocative
    "expl",     # expletive
    "dislocated",  # dislocated elements (topic, afterthoughts)
    # "list",     # list
    "advcl",    # adverbial clause
    "advmod",   # adverbial modifier
    "discourse",  # discourse particle
    "aux",      # auxiliary
    "cop",      # copula
    "mark",     # marker (subordinating conjunction)
    "nmod",     # nominal modifier
    "amod",     # adjectival modifier
    "appos",    # apposition
    "det",      # determiner
    "clf",      # classifier
    "case",     # adposition case marker
    "compound", # compound
    "conj",     # conjunct
    "cc",       # coordinating conjunction
    "parataxis", # sentence coordination
    "punct", # punctuation
    "root", # root node
    "ccomp", # clausal complement
    "xcomp" # non-finite clausal complement
)

# nominal argument relations
NOMINAL_ARGUMENTS = ("nsubj", "obj", "iobj", "obl")

# adverbial relations
ADVERBIAL_DEPRELS = ("advcl", "advmod", "obl")

# punctuation relation
PUNCTUATION_DEPRELS = ("punct",)

# coordination-related relations
COORDINATION_DEPRELS = ("conj", "cc", "parataxis")

# ad-modifier relations
MODIFIERS = ("amod", "nmod", "compound", "det", "case", "clf", "appos")

# functional relations
FUNCTIONAL_DEPRELS = ("aux", "cop", "mark")

# multiword expressions
MULTIWORD_EXPRESSIONS = ("flat", "fixed", "compound")

# ==== Extended dependency relation groups ====

STANDARD_UD_DEPRELS = (
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
)

# Non-standard but useful labels used by asymgram conversions.
ASYMGRAM_CUSTOM_DEPRELS = (
    "adp",
    "pobj",
)

ALL_KNOWN_DEPRELS = tuple(dict.fromkeys(STANDARD_UD_DEPRELS + ASYMGRAM_CUSTOM_DEPRELS))

# Argument-like relations.
CORE_ARGUMENT_DEPRELS = (
    "nsubj",
    "obj",
    "iobj",
    "csubj",
    "ccomp",
    "xcomp",
)

NOMINAL_ARGUMENT_DEPRELS = (
    "nsubj",
    "obj",
    "iobj",
)

CLAUSAL_ARGUMENT_DEPRELS = (
    "csubj",
    "ccomp",
    "xcomp",
)

OBLIQUE_ARGUMENT_DEPRELS = (
    "obl",
    "obl:arg",
    "obl:agent",
    "pobj",
)

ARGUMENT_DEPRELS = tuple(dict.fromkeys(CORE_ARGUMENT_DEPRELS + OBLIQUE_ARGUMENT_DEPRELS))

# Modifier and adjunct relations.
NOMINAL_MODIFIER_DEPRELS = (
    "nmod",
    "nummod",
    "amod",
    "acl",
    "appos",
)

CLAUSAL_MODIFIER_DEPRELS = (
    "advcl",
    "acl",
)

ADVERBIAL_MODIFIER_DEPRELS = (
    "advmod",
    "advcl",
    "obl",
)

NOMINAL_INTERNAL_DEPRELS = (
    "det",
    "clf",
    "case",
    "nmod",
    "nummod",
    "amod",
    "appos",
    "compound",
    "acl",
)

MODIFIER_DEPRELS = tuple(dict.fromkeys(
    NOMINAL_MODIFIER_DEPRELS
    + CLAUSAL_MODIFIER_DEPRELS
    + ADVERBIAL_MODIFIER_DEPRELS
))

# Function-word and marker relations.
AUXILIARY_DEPRELS = (
    "aux",
    "aux:pass",
)

FUNCTION_WORD_DEPRELS = (
    "aux",
    "aux:pass",
    "cop",
    "mark",
    "case",
    "cc",
    "det",
    "clf",
)

FUNCTION_HEAD_CONVERSION_DEPRELS = (
    "case",
    "aux",
    "aux:pass",
    "mark",
    "cop",
)

# Clause-related relations.
CLAUSE_DEPRELS = (
    "root",
    "csubj",
    "ccomp",
    "xcomp",
    "advcl",
    "acl",
    "acl:relcl",
    "parataxis",
    "conj",
)

SUBORDINATION_DEPRELS = (
    "mark",
    "advcl",
    "acl",
    "ccomp",
    "csubj",
    "xcomp",
)

FINITE_CLAUSE_LIKE_DEPRELS = (
    "root",
    "ccomp",
    "csubj",
    "advcl",
    "acl",
    "parataxis",
)

NONFINITE_CLAUSE_LIKE_DEPRELS = (
    "xcomp",
    "acl",
    "advcl",
)

# Coordination and parataxis.
COORDINATION_MARKER_DEPRELS = (
    "cc",
)

COORDINATE_MEMBER_DEPRELS = (
    "conj",
)

COORDINATION_STRUCTURE_DEPRELS = (
    "cc",
    "conj",
    "parataxis",
    "list",
)

# Multiword, lexicalized, and name-like structures.
FIXED_EXPRESSION_DEPRELS = (
    "fixed",
    "flat",
    "compound",
)

NAME_LIKE_DEPRELS = (
    "flat",
    "flat:name",
    "appos",
)

WORD_FORM_PROBLEM_DEPRELS = (
    "goeswith",
    "reparandum",
)

# Discourse, repair, and extra-sentential material.
DISCOURSE_DEPRELS = (
    "discourse",
    "vocative",
    "dislocated",
)

REPAIR_DEPRELS = (
    "reparandum",
    "goeswith",
    "orphan",
)

SPECIAL_DEPRELS = (
    "dep",
    "orphan",
    "list",
    "reparandum",
    "goeswith",
)

ROOT_DEPRELS = (
    "root",
)

# Content/function split for coarse analyses.
CONTENT_DEPRELS = tuple(dict.fromkeys(
    ARGUMENT_DEPRELS
    + MODIFIER_DEPRELS
    + COORDINATE_MEMBER_DEPRELS
    + ("root", "parataxis", "dep", "dislocated", "vocative")
))

NON_CONTENT_DEPRELS = tuple(dict.fromkeys(
    FUNCTION_WORD_DEPRELS
    + PUNCTUATION_DEPRELS
    + FIXED_EXPRESSION_DEPRELS
    + REPAIR_DEPRELS
))

# Group registry for programmatic lookup.
DEPREL_GROUPS = {
    "standard_ud": STANDARD_UD_DEPRELS,
    "asymgram_custom": ASYMGRAM_CUSTOM_DEPRELS,
    "argument": ARGUMENT_DEPRELS,
    "core_argument": CORE_ARGUMENT_DEPRELS,
    "nominal_argument": NOMINAL_ARGUMENT_DEPRELS,
    "clausal_argument": CLAUSAL_ARGUMENT_DEPRELS,
    "oblique_argument": OBLIQUE_ARGUMENT_DEPRELS,
    "modifier": MODIFIER_DEPRELS,
    "nominal_modifier": NOMINAL_MODIFIER_DEPRELS,
    "clausal_modifier": CLAUSAL_MODIFIER_DEPRELS,
    "adverbial_modifier": ADVERBIAL_MODIFIER_DEPRELS,
    "nominal_internal": NOMINAL_INTERNAL_DEPRELS,
    "function_word": FUNCTION_WORD_DEPRELS,
    "auxiliary": AUXILIARY_DEPRELS,
    "function_head_conversion": FUNCTION_HEAD_CONVERSION_DEPRELS,
    "clause": CLAUSE_DEPRELS,
    "subordination": SUBORDINATION_DEPRELS,
    "finite_clause_like": FINITE_CLAUSE_LIKE_DEPRELS,
    "nonfinite_clause_like": NONFINITE_CLAUSE_LIKE_DEPRELS,
    "coordination": COORDINATION_STRUCTURE_DEPRELS,
    "coordination_marker": COORDINATION_MARKER_DEPRELS,
    "coordinate_member": COORDINATE_MEMBER_DEPRELS,
    "multiword": FIXED_EXPRESSION_DEPRELS,
    "name_like": NAME_LIKE_DEPRELS,
    "word_form_problem": WORD_FORM_PROBLEM_DEPRELS,
    "discourse": DISCOURSE_DEPRELS,
    "repair": REPAIR_DEPRELS,
    "special": SPECIAL_DEPRELS,
    "punctuation": PUNCTUATION_DEPRELS,
    "root": ROOT_DEPRELS,
    "content": CONTENT_DEPRELS,
    "non_content": NON_CONTENT_DEPRELS,
}


def base_deprel(deprel: str | None) -> str:
    """Return the base dependency relation before the first subtype colon."""
    return str(deprel).split(":", 1)[0] if deprel else ""


def deprel_in_group(deprel: str | None, group: str, match_subtypes: bool = True) -> bool:
    """Return whether a dependency relation belongs to a named group."""
    labels = DEPREL_GROUPS[group]
    if deprel in labels:
        return True
    return match_subtypes and base_deprel(deprel) in labels


def deprel_groups(deprel: str | None, match_subtypes: bool = True) -> tuple[str, ...]:
    """Return all group names containing a dependency relation."""
    return tuple(
        group
        for group in DEPREL_GROUPS
        if deprel_in_group(deprel, group, match_subtypes=match_subtypes)
    )
