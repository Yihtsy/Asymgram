import os
from conllu import SentenceList as BaseSentenceList
from typing import Optional, Union, List, Iterator
from conllu.models import TokenList as BaseTokenList, Token as BaseToken
from conllu.models import Metadata
from conllu import parse as base_parse
from conllu import parse_incr as base_parse_incr
from collections import Counter, OrderedDict

class Token(BaseToken):
    """
    An enhanced Token that stores a reference to the TokenList it belongs to,
    allowing contextual operations like get_headtoken().
    """

    def __init__(self, data: dict):
        """
        Parameters:
            data (dict): Token fields (e.g., id, form, head, deprel, etc.)
            tokenlist (list): The list of tokens (TokenList) this token belongs to.
        """
        super().__init__(data)
        self.tokenlist = None  # Initialize as None, to be assigned later

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Token' object has no attribute '{key}'")

    def set_tokenlist(self, tokenlist: "TokenList") -> None:
        if not isinstance(tokenlist, TokenList):
            raise TypeError("tokenlist must be an instance of TokenList")
        self.tokenlist = tokenlist

    def get_headtoken(self) -> Optional["Token"]:
        """
        Return the head token (if any) from the same tokenlist.
        Returns None if head is 0 or invalid.
        """
        head_id = self.get("head")
        if isinstance(head_id, int) and head_id != 0:
            index = head_id - 1
            if 0 <= index < len(self.tokenlist):
                return self.tokenlist[index]
        return None

    def get_deptokens(
            self,
            exclude_deprels: List[str] | str | None = None,
            include_punct: bool = False
        ) -> List["Token"]:
            """
            Return a list of tokens that are dependents (children) of the current token.

            Parameters:
                exclude_deprels (str or list of str or None): 
                    Dependency relation(s) to exclude. If None, no exclusion is applied.
                    If a string is given, it is treated as a single deprel.
                include_punct (bool): 
                    If False (default), 'punct' will be added to the exclude list automatically.

            Returns:
                List[Token]: All dependents of this token, excluding specified relations.
            """
            # Normalize exclude_deprels to a set
            if exclude_deprels is None:
                exclude_set = set()
            elif isinstance(exclude_deprels, str):
                exclude_set = {exclude_deprels}
            else:
                exclude_set = set(exclude_deprels)

            # Exclude punctuation if requested
            if not include_punct:
                exclude_set.add("punct")

            # Collect direct dependents (children)
            dependents = []
            for tok in self.tokenlist:
                if tok.head == self.id and tok.deprel not in exclude_set:
                    dependents.append(tok)

            return dependents

    def get_deptoken(
        self,
        target_deprel: str | None = None,
        include_punct: bool = False
    ) -> Optional["Token"]:
        """
        Return the unique dependent token under the current token.

        Parameters:
            target_deprel (str or None): 
                If specified, return the only dependent with this dependency relation.
                If None, return the only dependent (of any relation) after filtering.
            include_punct (bool): 
                If False (default), exclude 'punct' dependents.

        Returns:
            Token or None: The unique dependent token, or None if no dependents exist.

        Raises:
            ValueError: If more than one matching dependent is found.
        """
        # Get filtered dependents
        dependents = self.get_deptokens(include_punct=include_punct)

        if target_deprel is None:
            if len(dependents) == 1:
                return dependents[0]
            elif len(dependents) > 1:
                print(f"[Warning] Multiple dependents found at {self.position()}: {len(dependents)} dependents")
            return None
        else:
            matching = [tok for tok in dependents if tok.deprel == target_deprel]
            if len(matching) == 1:
                return matching[0]
            elif len(matching) > 1:
                print(f"[Warning] Multiple '{target_deprel}' dependents at {self.position()}: {len(matching)} matches")
            return None

    def position(self) -> tuple[str, str, int]:
        """
        Return a tuple (newdoc_id, sent_id, token_id) indicating the token's position.
        If metadata or ID is missing, use fallback defaults.
        """
        newdoc_id = self.tokenlist.sentencelist.metadata.get("newdoc id", "N/A")
        sent_id = self.tokenlist.metadata.get("sent_id", "N/A")
        token_id = self.id if isinstance(self.id, int) else -1
        return (newdoc_id, sent_id, token_id)

    def get_subtree_span(
        self,
        exclude_left_punct: bool = True,
        exclude_right_punct: bool = False
    ) -> tuple[int, int]:
        """
        Return the (min_id, max_id) span of this token's dependency subtree.

        Parameters:
            exclude_left_punct (bool): If True, exclude the leftmost token if it's punctuation.
            exclude_right_punct (bool): If True, exclude the rightmost token if it's punctuation.

        Returns:
            (int, int): Span of subtree token IDs (inclusive).

        Raises:
            ValueError: If token is not in a TokenList, or subtree is not continuous.
        """
        if not self.tokenlist:
            raise ValueError("Token is not part of any TokenList (missing token.tokenlist).")

        tokenlist = self.tokenlist
        root_id = self["id"]

        # Collect all token IDs in the subtree
        subtree_ids = set()
        queue = [root_id]
        while queue:
            current = queue.pop()
            subtree_ids.add(current)
            for t in tokenlist:
                if t.get("head") == current:
                    queue.append(t["id"])

        # Sort subtree tokens by ID
        subtree_tokens = sorted(
            (t for t in tokenlist if t["id"] in subtree_ids),
            key=lambda t: t["id"]
        )

        # Exclude punctuation at edges
        if exclude_left_punct and subtree_tokens and subtree_tokens[0].get("upos") == "PUNCT":
            subtree_tokens = subtree_tokens[1:]
        if exclude_right_punct and subtree_tokens and subtree_tokens[-1].get("upos") == "PUNCT":
            subtree_tokens = subtree_tokens[:-1]

        if not subtree_tokens:
            raise ValueError("Subtree became empty after punctuation exclusion.")

        span_ids = [t["id"] for t in subtree_tokens]

        expected = list(range(min(span_ids), max(span_ids) + 1))
        if span_ids != expected:
            raise ValueError(
                f"Subtree rooted at token ID {root_id} is not continuous after punctuation exclusion: {span_ids}"
            )

        return (min(span_ids), max(span_ids))

    def is_leaf_node(self) -> bool:
        """
        Return True if this token is a leaf node (i.e., no token takes it as head).
        """
        if not hasattr(self, "tokenlist"):
            raise ValueError("Token is not associated with a TokenList.")
        return all(tok.headtoken != self for tok in self.tokenlist if isinstance(tok.get("id"), int))

class TokenList(BaseTokenList):
    """
    Enhanced TokenList that wraps enhanced Token objects (with tokenlist references).
    """

    def __init__(self, token_dicts, metadata: Optional[Metadata] = None, build_refs: bool = True):
        """
        Parameters:
            token_dicts (list of dict): List of token dictionaries (as parsed)
            metadata (dict): Optional CoNLL-U comment metadata
        """
        # Wrap each token dict into Token and assign tokenlist reference
        if token_dicts and isinstance(token_dicts[0], Token):
            tokens = token_dicts  # pass in Tokens
        else:
            tokens = [Token(tok) for tok in token_dicts]  # pass in dicts
        super().__init__(tokens, metadata)
        for tok in self:
            tok.tokenlist = self
        # Only assign .headtoken reference if requested
        if build_refs:
            self._build_intertoken_refs()

    def append(self, token: Token):
        """Override append to automatically assign tokenlist reference."""
        if not isinstance(token, Token):
            raise TypeError("Only Token instances can be appended.")
        super().append(token)
        token.set_tokenlist(self)

    def _build_intertoken_refs(self):
        """
        Internal method to resolve token.headtoken references from token["head"].
        """
        for tok in self:
            head_id = tok.get("head")
            if isinstance(head_id, int) and head_id != 0:
                tok.headtoken = self._id2token(head_id)
            else:
                tok.headtoken = None  # root

    def refresh_ids(self):
        """
        Refresh the 'id' field of each token in this TokenList to ensure continuous numbering from 1.

        Side Effects:
            Each token's 'id' field will be reassigned sequentially starting from 1.
        """
        for idx, token in enumerate(self, 1):
            if isinstance(token.get("id"), int):
                token["id"] = idx

    def refresh_heads(self):
        """
        Refresh the 'head' field of each token in this TokenList based on its .headtoken reference.

        Side Effects:
            Each token's 'head' field will be updated to match its .head['id'], or 0 if root.
        """
        for token in self:
            if not isinstance(token["id"], int):
                continue  # Skip non-token lines (e.g., multiword tokens, empty lines)

            if getattr(token, "head", None) is None:
                token["head"] = 0  # Root
            else:
                token["head"] = token.head["id"]

    def refresh_metadata_text(self, consider_spaceafter: bool = False):
        """
        Refresh the metadata["text"] field by reconstructing the sentence from token forms.

        Parameters:
            consider_spaceafter (bool): Whether to honor SpaceAfter=No in the MISC field.
                                        If False, tokens are concatenated without spaces.

        Side Effects:
            Updates self.metadata["text"] with the reconstructed string.
        """
        parts = []
        for i, tok in enumerate(self):
            if not isinstance(tok.get("id"), int):
                continue  # Skip non-tokens (e.g., multiword or empty lines)

            parts.append(tok["form"])

            if consider_spaceafter:
                # If not last token and no SpaceAfter=No, add space
                misc = tok.get("misc")
                if not (isinstance(misc, str) and "SpaceAfter=No" in misc):
                    parts.append(" ")

        # Remove possible trailing space
        new_text = "".join(parts).rstrip()
        self.metadata["text"] = new_text

    def position(self) -> tuple[str, str]:
        """
        Return a tuple (newdoc_id, sent_id) indicating the position of this TokenList.
        If metadata is missing, fallback to 'N/A'.
        """
        newdoc_id = self.sentencelist.metadata.get("newdoc id", "N/A")
        sent_id = self.metadata.get("sent_id", "N/A")
        return (newdoc_id, sent_id)

    def get_roottoken(self) -> Token:
        """
        Return the unique root token of the TokenList.
        A valid root token must satisfy both:
            - token["head"] == 0
            - token["deprel"] == "root"

        If errors occur, print a warning including the position and return None.
        """
        head0_tokens = [tok for tok in self if tok.get("head") == 0]
        root_tokens = [tok for tok in self if tok.get("deprel") == "root"]

        pos_info = self.position()  # (newdoc_id, sent_id)

        if len(head0_tokens) != 1:
            print(f"[Warning] {pos_info}: Expected 1 token with head=0, but found {len(head0_tokens)}: {[t['id'] for t in head0_tokens]}")
            return None

        if len(root_tokens) != 1:
            print(f"[Warning] {pos_info}: Expected 1 token with deprel='root', but found {len(root_tokens)}: {[t['id'] for t in root_tokens]}")
            return None

        if head0_tokens[0] is not root_tokens[0]:
            print(f"[Warning] {pos_info}: Head=0 token (id={head0_tokens[0]['id']}) and root-deprel token (id={root_tokens[0]['id']}) do not match")
            return None

        return head0_tokens[0]

    def _id2token(self, id_value: int) -> Token:
        """
        Retrieve the token object with the given id, ensuring uniqueness.

        Parameters:
            id_value (int): The ID of the token to retrieve.

        Returns:
            Token: The token with matching ID.

        Raises:
            ValueError: If no token with this ID exists, or if multiple tokens have the same ID.
        """
        matches = [tok for tok in self if tok.get("id") == id_value]

        if len(matches) == 0:
            return None # dangling head
        if len(matches) > 1:
            raise ValueError(f"Multiple tokens found with id={id_value}, indicating inconsistent TokenList.")
        
        return matches[0]

    def to_text(self):
        return self.metadata.get("text")

class SentenceList(BaseSentenceList):
    """
    An extended version of conllu.SentenceList with added methods.
    """

    def __init__(self, sentences, metadata: Optional[dict] = None):
        super().__init__(sentences)
        self.metadata = metadata or {}  # document-level metadata
        # Link each TokenList back to this SentenceList
        for sent in self:
            sent.sentencelist = self

    def serialize(self) -> str:
        """
        Serialize the entire SentenceList into CoNLL-U format, including document-level metadata.

        Returns:
            str: Serialized CoNLL-U string.
        """
        lines = []

        # Add document-level metadata at the top (e.g., newdoc id, source, etc.)
        for key, value in self.metadata.items():
            lines.append(f"# {key} = {value}")

        # Serialize each TokenList (sentence)
        for sentence in self:
            lines.append(sentence.serialize().strip() + "\n")  # Avoid extra blank lines

        return "\n".join(lines)

    def to_conllu(self, output_path):
        """
        Save the SentenceList to a .conllu file using its serialize() method.

        Parameters:
            output_path (str or Path): Target file path. Must end with '.conllu'.

        Raises:
            ValueError: If the output file does not end with '.conllu'.
        """
        output_path = os.fspath(output_path)
        if not output_path.endswith(".conllu"):
            raise ValueError(f"Output file must have a '.conllu' extension: got '{output_path}'")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.serialize())

    def to_text(self, line_by_line: bool = False) -> str:
        """
        Convert the SentenceList to plain text using metadata['text'] if available,
        otherwise fall back to concatenating tokens without space.

        Parameters:
            line_by_line (bool): Whether to return one sentence per line.

        Returns:
            str: The reconstructed text.
        """
        lines = []
        for sentence in self:
            sent_text = sentence.metadata.get("text") if hasattr(sentence, "metadata") else None
            if sent_text:
                lines.append(sent_text)
            else:
                tokens = [tok["form"] for tok in sentence if isinstance(tok["id"], int)]
                lines.append("".join(tokens))
        return "\n".join(lines) if line_by_line else "".join(lines)

    def to_segmented_text(self) -> str:
        """
        Concatenate tokens within each sentence (no spaces), with each sentence on a new line.

        Returns:
            str: Plain text where each sentence is on a new line and token forms are joined without spaces.
        """
        lines = []
        for sentence in self:
            tokens = [tok["form"] for tok in sentence if isinstance(tok["id"], int)]
            lines.append("".join(tokens))
        return "\n".join(lines)

    def to_tokenized_text(self) -> str:
        """
        Concatenate token forms with spaces between tokens and newlines between sentences.

        Returns:
            str: Plain text where each sentence is on a new line and tokens are space-separated.
        """
        lines = []
        for sentence in self:
            tokens = [tok["form"] for tok in sentence if isinstance(tok["id"], int)]
            lines.append(" ".join(tokens))
        return "\n".join(lines)

class SentenceLists(list):
    """
    A wrapper class for a list of ag.SentenceList objects.
    """

    def __init__(self, sentence_lists: List[SentenceList]):
        """
        Initialize a SentenceLists object.

        Parameters:
            sentence_lists (List[ag.SentenceList]): A list of ag.SentenceList instances.
        """
        if not all(isinstance(s, SentenceList) for s in sentence_lists):
            raise TypeError("All elements must be instances of ag.SentenceList")
        super().__init__(sentence_lists)

    def flatten(self) -> SentenceList:
        """
        Flatten all SentenceLists into a single SentenceList.

        Returns:
            SentenceList: A single SentenceList containing all TokenLists.
        """
        all_sentences = [sent for sent_list in self for sent in sent_list]
        return SentenceList(all_sentences)

    def count_deprel_freq(self) -> OrderedDict:
        """
        Count the frequency of each deprel label across all TokenLists.

        Returns:
            OrderedDict: deprel -> count, sorted by frequency in descending order.
        """
        counter = Counter()

        for sentence in self.flatten():
            for token in sentence:
                if not isinstance(token["id"], int):
                    continue
                deprel = token.get("deprel")
                if deprel:
                    counter[deprel] += 1

        return OrderedDict(counter.most_common())

def parse(text: str) -> SentenceList:
    """
    Parse CoNLL-U text and return enhanced SentenceList with doc-level metadata.
    """
    base_tokenlists = base_parse(text)
    sentence_objects = []

    # Extract SentenceList-level metadata
    doc_metadata = {}

    for sent in base_tokenlists:
        metadata = getattr(sent, "metadata", {})
        # Promote newdoc id and newpar id to document level
        for key in ("newdoc id", "newpar id"):
            if key in metadata:
                doc_metadata[key] = metadata.pop(key)
        sentence_objects.append(TokenList(sent, metadata))

    return SentenceList(sentence_objects, metadata=doc_metadata)

def parse_incr(file_like) -> Iterator[TokenList]:
    """
    Incrementally parse a CoNLL-U file-like object, returning enhanced TokenList objects.

    Parameters:
        file_like: A file-like object (must support .readline())

    Yields:
        TokenList: Enhanced sentence-level token list
    """
    for base_sent in base_parse_incr(file_like):
        metadata = getattr(base_sent, "metadata", None)
        yield TokenList(base_sent, metadata)

def to_sentencelist(data):
    """Ensure input is SentenceList; wrap TokenList if needed."""
    if isinstance(data, SentenceList):
        return data
    elif isinstance(data, TokenList):
        return SentenceList([data])
    else:
        raise TypeError("Input must be a SentenceList or TokenList")