import math

# Calculate the Cosine Similarity
def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vocab)
    vec1_length = math.sqrt(sum(vec1.get(term, 0)**2 for term in vocab))
    vec2_length = math.sqrt(sum(vec2.get(term, 0)**2 for term in vocab))

    if vec1_length == 0 or vec2_length == 0:
        return 0.0

    return dot_product / (vec1_length * vec2_length)

def find_most_similar(tfidf_vectors, vocab):
    max_similarity = -1
    most_similar_pair = None

    n = len(tfidf_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocab)
            print(f"Similarity between Document {i+1} and Document {j+1}: {similarity:.4f}")

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (i+1, j+1)

    if most_similar_pair:
        print(f"\nThe most similar documents are Document {most_similar_pair[0]} and Document {most_similar_pair[1]} with similarity: {max_similarity:.4f}")
    else:
        print("\nNo similar documents found.")
