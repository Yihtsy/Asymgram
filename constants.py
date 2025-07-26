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
    "dislocated", 
    "advcl",    # adverbial clause
    "advmod",   # adverbial modifier
    "discourse", 
    "aux",      # auxiliary
    "cop",      # copula
    "mark",     # marker
    "nmod",     # nominal modifier
    "amod",     # adjectival modifier
    "appos",    # apposition
    "det",      # determiner
    "clf",      # classifier
    "case",     # adposition case marker
    "compound", # compound
    "conj",     # conjunct
    "cc",       # coordinating conjunction
    "parataxis",
    "punct",
    "root",
    "ccomp",
    "xcomp"
)

NOMINAL_ARGUMENTS = ("nsubj", "obj", "iobj", "obl")

# 状语修饰
ADVERBIAL_DEPRELS = {"advcl", "advmod", "obl"}

# 标点
PUNCTUATION_DEPRELS = {"punct"}

# 句法连接
COORDINATION_DEPRELS = {"conj", "cc", "parataxis"}

# 附加修饰
MODIFIERS = {"amod", "nmod", "compound", "det", "case", "clf", "appos"}

# 功能词
FUNCTIONAL_DEPRELS = {"aux", "cop", "mark"}

MULTIWORD_EXPRESSIONS = {"flat", "fixed", "compound"}