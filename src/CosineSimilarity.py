if __name__ == "__main__":
    d1 = "It is going to rain today"
    d2 = "Today I am not going outside"
    d3 = "NLP is an interesting topic"
    d4 = "NLP includes ML, DL topics too"
    d5 = "I am going to complete NLP homework, today"

data = [d1, d2, d3, d4, d5]

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
vector_matrix = count_vectorizer.fit_transform(data)
print(vector_matrix)

tokens = count_vectorizer.get_feature_names()
print(tokens)

vector_matrix.toarray()

import pandas as pd

def create_dataframe(matrix, tokens):

    doc_names = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return(df)

create_dataframe(vector_matrix.toarray(),tokens)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity_matrix = cosine_similarity(vector_matrix)
print(create_dataframe(cosine_similarity_matrix, ['doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5']))