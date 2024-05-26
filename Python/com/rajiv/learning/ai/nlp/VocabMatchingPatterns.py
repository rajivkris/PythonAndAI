import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
import chardet

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
doc = nlp(u"When I was young solar Powered calculator were very expensive. Now, solar---powered calculators are very cheap!")

pattern1 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': "*"}, {'LOWER': 'powered'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'powered'}]

matcher.add('SolarPower', [pattern1, pattern2], on_match=None)

found_matches = matcher(doc)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)

with open('TextFiles/reaganomics.txt', 'rb') as f:
    result = chardet.detect(f.read())
    file_encoding = result['encoding']
    print(file_encoding)

# Use the detected encoding to read the file
with open('TextFiles/reaganomics.txt', encoding=file_encoding) as f:
    file_doc = nlp(f.read())

phraseMatcher = PhraseMatcher(nlp.vocab)
phrases = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
phraseDocs = [nlp(text) for text in phrases]
phraseMatcher.add('VoddooEcnomics', None, *phraseDocs)
matches = phraseMatcher(file_doc)

sentence_map = {}

for sentence in file_doc.sents:
    sentence_map[(sentence.start, sentence.end)] = sentence.text

def get_match_sentence(sentence_map, start, end):
    for start_end, sentence in sentence_map.items():
        if start_end[0] <= start and start_end[1] >= end:
            return sentence
    return None
    
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = file_doc[start:end]
    match_sentence = get_match_sentence(sentence_map, start, end)
    print(string_id, span.text, match_sentence)



