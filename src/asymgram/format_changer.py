import os
import shutil
from .conllu import SentenceLists, SentenceList, parse, parse_incr
from typing import Generator

def conllu_to_text(conllu_file_path, text_file_path, remove_punct=False, style="cont"):
    # Read CoNLL-U files
    with open(conllu_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Parsing CoNLL-U data
    #sentences = conllu.parse(data)
    sentences = parse(data)
    
    # Extract plain text
    with open(text_file_path, 'w', encoding='utf-8') as f:
        if style == "vert": # one token per line
            for sentence in sentences:
                for token in sentence:
                    if remove_punct == True: # skip PUNCT
                        if token['upostag'] == 'PUNCT' or token['deprel'] == 'punct':
                            continue
                    f.write(token['form'] + '\n')
                f.write('\n')  # A blank line after each sentence
        elif style == "hori": # horizontal style
            pass
        elif style == "cont": # continuous style
            pass

def read_conllu(file_path: str) -> SentenceList:
    """
    Parse a single .conllu file into a SentenceList.

    Parameters:
        file_path (str): Path to the .conllu file

    Returns:
        SentenceList: Parsed list of sentences
    """
    file_path = os.fspath(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    #return SentenceList(parse(content))
    return parse(content)

def write_conllu(sentences, output_path):
    """
    Write SentenceList to a .conllu file.
    """
    output_path = os.fspath(output_path)
    if not output_path.endswith(".conllu"):
        raise ValueError(f"Output file must have a '.conllu' extension: got '{output_path}'")
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.serialize())

def conllu_folder2sentencelists(folder_path: str) -> Generator[SentenceList, None, None]:
    """
    Yield SentenceList objects from all .conllu files in a folder, in sorted order by filename.

    Parameters:
        folder_path (str): Path to the folder containing .conllu files

    Yields:
        SentenceList: Parsed list of sentences from each .conllu file
    """
    folder_path = os.fspath(folder_path)
    conllu_files = sorted(
        f for f in os.listdir(folder_path) if f.endswith(".conllu")
    )

    for filename in conllu_files:
        full_path = os.path.join(folder_path, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            yield from parse_incr(f)

def load_sentence_lists(data) -> SentenceLists:
    """
    Load data and convert to a SentenceLists object.
    
    Parameters:
        data (str or SentenceList): directory, file path, or already parsed SentenceList
    
    Returns:
        SentenceLists: A wrapper around a list of SentenceList objects
    """
    sentence_lists = []

    if isinstance(data, str) and os.path.isdir(data):
        conllu_files = [f for f in os.listdir(data) if f.endswith(".conllu")]
        if not conllu_files:
            raise ValueError(f"No .conllu files found in directory: {data}")
        for fname in conllu_files:
            with open(os.path.join(data, fname), "r", encoding="utf-8") as f:
                sentence_lists.append(SentenceList(parse(f.read())))

    elif isinstance(data, str) and data.endswith(".conllu"):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        with open(data, "r", encoding="utf-8") as f:
            sentence_lists.append(SentenceList(parse(f.read())))

    elif isinstance(data, SentenceList):
        sentence_lists.append(data)

    else:
        raise TypeError("Input must be a SentenceList, a .conllu file path, or a directory path.")

    return SentenceLists(sentence_lists)

def load_sentence_list(file_path: str) -> SentenceList:
    """
    Load a single .conllu file and convert it to a SentenceList object.

    Parameters:
        file_path (str): Path to the .conllu file.

    Returns:
        SentenceList: Parsed sentence list from the given file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the input is not a .conllu file.
    """
    if not isinstance(file_path, str):
        raise TypeError("Input must be a file path string.")
    
    if not file_path.endswith(".conllu"):
        raise ValueError("Only .conllu files are supported.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        sentences = parse(data)
        return SentenceList(sentences)

def conllu_to_txt(folderpath: str):
    """
    Convert all .conllu files in the specified folder to .txt files
    and save them in a new folder with the suffix '_txt'.

    Parameters:
        folderpath (str): The path to the folder containing .conllu files.
    """
    if not folderpath or not os.path.isdir(folderpath):
        print("Invalid folder path.")
        return

    # Determine the parent folder and the new target folder path
    parent_folder = os.path.dirname(folderpath)
    new_folder_name = os.path.basename(folderpath) + "_txt"
    target_folder = os.path.join(parent_folder, new_folder_name)

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Process each .conllu file in the source folder
    for filename in os.listdir(folderpath):
        if filename.endswith(".conllu"):
            source_path = os.path.join(folderpath, filename)
            new_filename = os.path.splitext(filename)[0] + ".txt"
            target_path = os.path.join(target_folder, new_filename)

            # Copy the file content to a .txt file
            shutil.copyfile(source_path, target_path)
            print(f"Converted: {filename} -> {new_filename}")

    print(f"\nAll .conllu files have been converted and saved in: {target_folder}")

def spacy_doc2conllu_style_text(doc, filename: str = "", sent_index: int = 0, sent_id_length: int = 3):
    """
    Convert spaCy documentation to CoNLL format
    - The HEAD column of the ROOT node is 0
    - Add meta information: # newdoc id (only added in the first sentence), # sent_id, # text
    - The default sent_id is three digits
    """
    conllu_lines = []
    for token in doc:
        head = token.head.i + 1 if token.head != token else 0
        if token.dep_ == "punct" and token.head == token:
            head = 0
        conllu_lines.append(
            f"{token.i+1}\t{token.text}\t_\t{token.pos_}\t_\t_\t{head}\t{token.dep_}\t_\t_"
        )
    
    lines = []
    if sent_index == 0:
        lines.append(f"# newdoc id = {filename}")
    sent_id = f"{filename}_{str(sent_index+1).zfill(sent_id_length)}"
    lines.append(f"# sent_id = {sent_id}")
    lines.append(f"# text = {doc.text}")
    lines.extend(conllu_lines)

    return "\n".join(lines) + "\n"


def stanza_doc2conllu_style_text(doc, filename: str = None, sent_start_index: int = 0, sent_id_length: int = 3) -> str:
    """
    Convert a Stanza Document into CoNLL-U formatted string.
    - Add "# newdoc id" to the first sentence
    - Add "# sent_id" and "# text" for each sentence; text is the original sentence with spaces removed
    - Support start index offset (for continuous sentence numbering)
    """
    lines = []
    for i, sentence in enumerate(doc.sentences):
        global_sent_index = sent_start_index + i
        
        if global_sent_index == 0 and filename:
            lines.append(f"# newdoc id = {filename}")
        
        if filename:
            sent_id = f"{filename}_{str(global_sent_index + 1).zfill(sent_id_length)}"
        else:
            sent_id = f"{str(global_sent_index + 1).zfill(sent_id_length)}"
        clean_text = sentence.text.replace(" ", "")  # Remove spaces to restore original sentence
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {clean_text}")

        for word in sentence.words:
            head = word.head if word.head is not None else 0
            line = f"{word.id}\t{word.text}\t_\t{word.upos}\t_\t_\t{head}\t{word.deprel}\t_\t_"
            lines.append(line)
        
        lines.append("")  # Blank line between sentences

    return "\n".join(lines)


def hanlp_doc2conllu_style_text(doc: dict, filename: str = None, sent_start_index: int = 0, sent_id_length: int = 3) -> str:
    """
    Convert HanLP dependency parsing output into CoNLL-U formatted text.
    - Uses "tok/fine" for tokenization, "pos/ctb" for POS tags, and "dep" for dependencies.
    - POS tags are written into the XPOS column (5th), leaving UPOS (4th) as underscore.
    - Adds "# newdoc id", "# sent_id", and "# text" for each sentence.
    - Supports offset and customizable sent_id formatting.
    """
    lines = []

    tokens = doc['tok/fine']
    pos_tags = doc['pos/ctb']
    dependencies = doc['dep']

    for i, (words, pos, deps) in enumerate(zip(tokens, pos_tags, dependencies)):
        global_sent_index = sent_start_index + i

        if global_sent_index == 0 and filename:
            lines.append(f"# newdoc id = {filename}")

        sent_id = f"{filename + '_' if filename else ''}{str(global_sent_index + 1).zfill(sent_id_length)}"
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {''.join(words)}")

        for idx, (word, xpos, (head, rel)) in enumerate(zip(words, pos, deps)):
            head = int(head)
            line = f"{idx + 1}\t{word}\t_\t_\t{xpos}\t_\t{head}\t{rel}\t_\t_"
            lines.append(line)

        lines.append("")

    return "\n".join(lines)

def extract_all_deprels(
    conllu_folder: str,
    include_punct: bool = False,
    include_root: bool = False
) -> list[str]:
    """
    Extract all unique dependency relation types (deprel) from .conllu files in a folder.

    Parameters:
        conllu_folder (str): Path to the folder containing .conllu files.
        include_punct (bool): Whether to include 'punct' deprel.
        include_root (bool): Whether to include 'root' deprel.

    Returns:
        List[str]: A sorted list of unique deprel labels.
    """
    all_deprels = set()

    for filename in os.listdir(conllu_folder):
        if filename.endswith(".conllu"):
            filepath = os.path.join(conllu_folder, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    sentences = parse(f.read())
                    for sentence in sentences:
                        for token in sentence:
                            deprel = token.get("deprel")
                            if deprel is None:
                                continue
                            if not include_punct and deprel == "punct":
                                continue
                            if not include_root and deprel == "root":
                                continue
                            all_deprels.add(deprel)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return sorted(all_deprels)

def conllu_to_latex_standalone(conllu_data):
    latex_code = "\\documentclass{standalone}\n"
    latex_code += "\\usepackage{tikz-dependency}\n"
    latex_code += "\\usepackage{fontspec}\n"
    latex_code += "\\setmainfont{Times New Roman} % Replace with the font to be used\n"
    latex_code += "\\begin{document}\n"
    latex_code += "    \\begin{dependency}\n"
    latex_code += "        \\begin{deptext}[column sep=3em]\n"

    lines = conllu_data.strip().split("\n")
    forms_line = ""
    upos_line = ""
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        form, upos = parts[1], parts[3]
        forms_line += f"{form} \\& "
        upos_line += f"{upos} \\& "
    
    forms_line = forms_line.rstrip(" \\&")  # Remove the last column separator
    upos_line = upos_line.rstrip(" \\&")    # Remove the last column separator
    latex_code += f"            {forms_line} \\\\\n"
    latex_code += f"            {upos_line} \\\\\n"
    latex_code += "        \\end{deptext}\n"

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        head, deprel = parts[6], parts[7]
        if head == '0':
            continue
        latex_code += f"        \\depedge{{{head}}}{{{parts[0]}}}{{\\textbf{{{deprel}}}}}\n"
    
    latex_code += "    \\end{dependency}\n"
    latex_code += "\\end{document}"

    return latex_code

'''
# Example ConLL-U data
conllu_data = """
# newpar
# sent_id = art-sjadziba-17841
# text = На лекцыі Аляксандр распавядзе хітрыкі працы на канцэртах, колькі трэба рабіць здымкаў, як на наступны дзень пасля канцэрта выпусціць якасны рэпартаж, колькі часу сыходзіць на адзін канцэрт і шмат чаго яшчэ.
# genre = social-media
1   На  на  ADP IN  _   2   case    2:case  _
2   лекцыі  лекцыя  NOUN    NN  Animacy=Inan|Case=Loc|Gender=Fem|Number=Sing    4   obl 4:obl:на:loc    _
3   Аляксандр   Аляксандр   PROPN   NNP Animacy=Anim|Case=Nom|Gender=Masc|NameType=Giv|Number=Sing  4   nsubj   4:nsubj _
4   распавядзе  распавесці  VERB    VBC Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin|Voice=Act  0   root    0:root  _
5   хітрыкі хітрыка NOUN    NN  Animacy=Inan|Case=Acc|Gender=Fem|Number=Plur    4   obj 4:obj|11:conj   _
6   працы   праца   NOUN    NN  Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing    5   nmod    5:nmod:gen  _
7   на  на  ADP IN  _   8   case    8:case  _
8   канцэртах   канцэрт NOUN    NN  Animacy=Inan|Case=Loc|Gender=Masc|Number=Plur   6   nmod    6:nmod:на:loc   SpaceAfter=No
9   ,   ,   PUNCT   PUNCT   _   10  punct   10:punct    _
10  колькі  колькі  NUM CD  Case=Acc    13  nummod:gov  13:nummod:gov   _
11  трэба   трэба   VERB    PRED    _   5   acl 5:acl   _
12  рабіць  рабіць  VERB    VB  Aspect=Imp|VerbForm=Inf|Voice=Act   11  csubj   11:csubj    _
13  здымкаў здымак  NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Plur   12  obj 12:obj  SpaceAfter=No
14  ,   ,   PUNCT   PUNCT   _   21  punct   21:punct    _
15  як  як  ADV WRB Degree=Pos  21  advmod  21:advmod   _
16  на  на  ADP IN  _   18  case    18:case _
17  наступны    наступны    ADJ JJL Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing    18  amod    18:amod _
18  дзень   дзень   NOUN    NN  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing   21  obl 21:obl:на:acc   _
19  после   после   ADP IN  _   20  case    20:case _
20  канцэрта    канцэрт NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing   21  obl 21:obl:пасля:gen    _
21  выпусціць   выпусціць   VERB    VB  Aspect=Perf|VerbForm=Inf|Voice=Act  11  conj    5:acl|11:conj   _
22  якасны  якасны  ADJ JJL Case=Nom|Degree=Pos|Gender=Masc|Number=Sing 23  amod    23:amod _
23  рэпартаж    рэпартаж    NOUN    NN  Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing   21  nsubj:pass  21:nsubj:pass   SpaceAfter=No
24  ,   ,   PUNCT   PUNCT   _   27  punct   27:punct    _
25  колькі  колькі  NUM CD  Animacy=Inan|Case=Acc   26  nummod:gov  26:nummod:gov   _
26  часу    час NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing   27  nsubj   27:nsubj    _
27  сыходзіць   сыходзіць   VERB    VBC Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act  23  acl 23:acl  _
28  на  на  ADP IN  _   30  case    30:case _
29  адзін   адзін   NUM CD  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|NumType=Card  30  nummod  30:nummod   _
30  канцэрт канцэрт NOUN    NN  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing   27  obl 27:obl:на:acc   _
31  і   і   CCONJ   CC  _   33  cc  33:cc   _
32  шмат    шмат    ADV RB  Degree=Pos  33  advmod  33:advmod   _
33  чаго    што PRON    WP  Animacy=Inan|Case=Gen|Gender=Neut|Number=Sing|PronType=Int  11  conj    5:ref   _
34  яшчэ    яшчэ    ADV RB  Degree=Pos  33  advmod  33:advmod   SpaceAfter=No
35  .   .   PUNCT   PUNCT   _   4   punct   4:punct _
"""

latex_code = conllu_to_latex_standalone(conllu_data)
print(latex_code)
'''
