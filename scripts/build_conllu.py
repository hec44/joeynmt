import stanza
from stanza.utils.conll import CoNLL



def doc_2_conll(input_file, output_file, lang='en'):
    """process a file with text to CoNLL-U format as described in: https://universaldependencies.org/format.html

    Args:
        input_file (String): input file with ### format
        output_file (String): file that will be generated with CoNLL-U format
        lang (str, optional): Language of input file. Defaults to 'es'.
    """

    nlp = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse')
    output_lines = []
    input_doc = open(input_file, 'r')
    doc_lines = input_doc.readlines()
    input_doc.close()
    for line in doc_lines:
        doc = nlp(line)
        dicts = doc.to_dict() 
        conll = CoNLL.convert_dict(dicts)
        if len(conll) > 1:
            print("PROBLEM with sentence: "+line)
            continue
        for word in conll[0]:
            conll_line = "\t".join(word)
            output_lines.append(conll_line)

        output_lines.append("")
    output_doc = open(output_file, 'w+')
    output_doc.writelines("\n".join(output_lines))
    output_doc.close()

if __name__ == "__main__":
    doc_2_conll("/home/hec44/Documents/encodingRepresentations/experiments/ggnnBaseline/joeynmt/test/data/toy/dev.en",\
        "/home/hec44/Documents/encodingRepresentations/experiments/ggnnBaseline/joeynmt/test/data/toy/dev.conll")
