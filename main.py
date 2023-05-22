import numpy as np
from scipy.optimize import linear_sum_assignment

def load_glove_embeddings(file_path):
    word_vectors = {}
    with open(file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors

def find_field_matches(field_list_a, field_list_b, glove_model_path):
    # Step 1: Load GloVe model
    glove_model = load_glove_embeddings(glove_model_path)

    # Step 2: Create cost matrix using GloVe similarity scores
    n = len(field_list_a)
    m = len(field_list_b)
    cost_matrix = np.zeros((n, m))

    for i, field_a in enumerate(field_list_a):
        for j, field_b in enumerate(field_list_b):
            vector_a = glove_model.get(field_a)
            vector_b = glove_model.get(field_b)
            if vector_a is not None and vector_b is not None:
                similarity_score = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
                cost_matrix[i, j] = 1 - similarity_score  # Convert similarity to distance

    # Step 3: Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Step 4: Generate the field matches
    field_matches = []
    for i, j in zip(row_ind, col_ind):
        field_matches.append((field_list_a[i], field_list_b[j]))

    return field_matches

# Example usage
field_list_a = [ "a", "b", "c" ]
field_list_b = [ "1", "2", "3" ]
glove_model_path = "glove.6B.300d.txt"

matches = find_field_matches(field_list_a, field_list_b, glove_model_path)
for match in matches:
    print(match)
