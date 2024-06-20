import spacy
from scipy import spatial

nlp = spacy.load('en_core_web_lg')

print(nlp("lion").vector)

tokens = nlp("lion cat pet")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

king = nlp("king").vector
man = nlp("man").vector
woman = nlp("woman").vector

new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])

