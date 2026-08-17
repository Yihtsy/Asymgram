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
PUNCTUATION_DEPRELS = ("punct")

# coordination-related relations
COORDINATION_DEPRELS = ("conj", "cc", "parataxis")

# ad-modifier relations
MODIFIERS = ("amod", "nmod", "compound", "det", "case", "clf", "appos")

# functional relations
FUNCTIONAL_DEPRELS = ("aux", "cop", "mark")

# multiword expressions
MULTIWORD_EXPRESSIONS = ("flat", "fixed", "compound")