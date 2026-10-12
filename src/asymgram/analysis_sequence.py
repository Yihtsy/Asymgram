"""Composable descriptions of annotation schemes and parser phenotypes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AnalysisLocus:
    code: str
    domain: str
    question: str
    alleles: tuple[str, ...]
    weight: float = 1.0


ANALYSIS_LOCI = (
    AnalysisLocus("U.WORD", "unitization", "Word segmentation authority", ("UD_GSD", "CTB", "CHAR", "CUSTOM"), 2.0),
    AnalysisLocus("U.AFFIX", "unitization", "Affixes merged or split", ("MERGE", "SPLIT", "MIXED"), 2.0),
    AnalysisLocus("U.VV", "unitization", "Verb complements merged or split", ("MERGE", "SPLIT", "MIXED"), 2.0),
    AnalysisLocus("U.VO", "unitization", "Verb-object compounds merged or split", ("MERGE", "SPLIT", "MIXED"), 2.0),
    AnalysisLocus("U.MWT", "unitization", "Multiword-token rows", ("KEEP", "DROP", "NONE")),
    AnalysisLocus("U.EMPTY", "unitization", "Empty nodes", ("ALLOW", "DROP", "NONE")),
    AnalysisLocus("P.AFFIX", "category", "Expanded affix category", ("PART", "NOUN", "AFFIX", "CONTEXT", "NA")),
    AnalysisLocus("P.LOC", "category", "Localizer category", ("ADP", "NOUN", "CONTEXT")),
    AnalysisLocus("P.ADPV", "category", "Adposition-verb ambiguity", ("ADP", "VERB", "CONTEXT")),
    AnalysisLocus("P.AUXV", "category", "Auxiliary-verb ambiguity", ("AUX", "VERB", "CONTEXT")),
    AnalysisLocus("H.ADP", "head", "Adposition head policy", ("CONTENT", "FUNCTION", "CONTEXT"), 2.0),
    AnalysisLocus("H.LOC", "head", "Localizer head policy", ("CONTENT", "FUNCTION", "CONTEXT"), 2.0),
    AnalysisLocus("H.MARK", "head", "Subordinator head policy", ("CONTENT", "FUNCTION", "CONTEXT"), 2.0),
    AnalysisLocus("H.COP", "head", "Copula head policy", ("CONTENT", "FUNCTION", "CONTEXT"), 2.0),
    AnalysisLocus("H.AUX", "head", "Auxiliary head policy", ("CONTENT", "FUNCTION", "CONTEXT"), 2.0),
    AnalysisLocus("H.CLF", "head", "Classifier head policy", ("NUMERAL", "CLASSIFIER", "NOMINAL", "CONTEXT"), 2.0),
    AnalysisLocus("H.COORD", "head", "Coordination head policy", ("FIRST", "LAST", "CONJUNCTION", "SYMMETRIC", "CONTEXT"), 2.0),
    AnalysisLocus("H.FLAT", "head", "Flat-expression head policy", ("FIRST", "LAST", "LEXICAL", "CONTEXT")),
    AnalysisLocus("T.REL", "topology", "Relative-clause topology", ("NOMINAL_HEAD", "CLAUSE_HEAD", "ENHANCED", "CONTEXT")),
    AnalysisLocus("T.CTRL", "topology", "Control topology", ("BASIC_XCOMP", "ENHANCED_SUBJECT", "FUNCTION_HEAD", "CONTEXT")),
    AnalysisLocus("T.CAUS", "topology", "Causative topology", ("BASIC_TREE", "ENHANCED_SHARED", "CONTEXT")),
    AnalysisLocus("T.SERIAL", "topology", "Serial-verb analysis", ("ADVCL", "COMPOUND", "COORD", "CONTEXT")),
    AnalysisLocus("T.BA", "topology", "BA construction", ("OBL_PATIENT", "BA_HEAD", "CONSTRUCTION", "CONTEXT")),
    AnalysisLocus("T.BEI", "topology", "BEI construction", ("AUX_PASS", "ADP_AGENT", "BEI_HEAD", "CONTEXT")),
    AnalysisLocus("R.DE", "relation", "的/地/得/之 policy", ("FUNCTION_SPLIT", "GENERIC", "LEXICAL", "CONTEXT")),
    AnalysisLocus("R.VV", "relation", "Verb-verb relation", ("SUBTYPES", "COMPOUND", "CLAUSAL", "CONTEXT")),
    AnalysisLocus("R.VO", "relation", "Verb-object relation", ("COMPOUND_VO", "OBJ", "TOKEN", "CONTEXT")),
    AnalysisLocus("R.TIME", "relation", "Temporal relation", ("TMOD_SUBTYPES", "OBL", "NMOD", "CONTEXT")),
    AnalysisLocus("R.TOPIC", "relation", "Topic relation", ("DISLOCATED", "CORE_ROLE", "ENHANCED", "CONTEXT")),
    AnalysisLocus("G.BASIC", "graph", "Basic graph", ("TREE", "DAG")),
    AnalysisLocus("G.ENH", "graph", "Enhanced graph", ("NONE", "PARTIAL", "FULL")),
    AnalysisLocus("G.COREF", "graph", "Shared argument representation", ("NONE", "DEPS", "EMPTY", "CONTEXT")),
)

_LOCI_BY_CODE = {locus.code: locus for locus in ANALYSIS_LOCI}


@dataclass
class AnalysisSequence:
    sequence_version: str
    identity: dict[str, Any]
    alleles: dict[str, str]
    phenotype: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        unknown = set(self.alleles) - set(_LOCI_BY_CODE)
        if unknown:
            raise ValueError(f"Unknown loci: {sorted(unknown)}")
        for code, allele in self.alleles.items():
            if allele == "?" or allele.startswith("MIX{"):
                continue
            allowed = _LOCI_BY_CODE[code].alleles
            if allele not in allowed:
                raise ValueError(f"Invalid allele {allele!r} for {code}; expected {allowed}")

    def encode(self) -> str:
        self.validate()
        values = (f"{locus.code}={self.alleles.get(locus.code, '?')}" for locus in ANALYSIS_LOCI)
        return f"SAS/{self.sequence_version};" + ";".join(values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sequence"] = self.encode()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisSequence":
        result = cls(
            sequence_version=str(data["sequence_version"]),
            identity=dict(data.get("identity", {})),
            alleles=dict(data.get("alleles", {})),
            phenotype=dict(data.get("phenotype", {})),
            evidence=dict(data.get("evidence", {})),
        )
        result.validate()
        return result


def compare_analysis_sequences(left: AnalysisSequence, right: AnalysisSequence) -> dict[str, Any]:
    """Compare observed loci with a weighted normalized Hamming distance."""
    mutations = []
    difference = observed = 0.0
    for locus in ANALYSIS_LOCI:
        left_value = left.alleles.get(locus.code, "?")
        right_value = right.alleles.get(locus.code, "?")
        if "?" in {left_value, right_value}:
            continue
        observed += locus.weight
        if left_value != right_value:
            difference += locus.weight
            mutations.append({
                "locus": locus.code,
                "domain": locus.domain,
                "left": left_value,
                "right": right_value,
                "weight": locus.weight,
            })
    return {
        "left": left.identity,
        "right": right.identity,
        "weighted_distance": round(difference / observed, 6) if observed else 0.0,
        "mutations": mutations,
    }


def read_analysis_sequence(path: str | Path) -> AnalysisSequence:
    """Read and validate a JSON analysis-sequence profile."""
    return AnalysisSequence.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_analysis_sequence(sequence: AnalysisSequence, path: str | Path) -> None:
    """Validate and write a JSON analysis-sequence profile."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sequence.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
