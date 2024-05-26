import spacy
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(u'Apple is looking at buying a large factory work INR 10 billion in New Delhi with a email apple@apple.com!')

# Print each token separately
for token in doc:
    print(token.text, end=' | ')

print('\n')

# Print different entities in the Doc
for entity in doc.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print('\n')

# Print Noun chunks in the Doc
for chunk in doc.noun_chunks:
    print(chunk.text)