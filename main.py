from tf_raw import compute_tf
from tf_idf import compute_idf, compute_tfidf
from cosine_similarity import find_most_similar
import wikipedia
import re
from nltk.tokenize import word_tokenize  

# List of wiki pages
wiki_pages = [
    "Astrology",
    "Virgo",
    "Libra",
    "August",
    "September"
]

print("Wiki Pages (Checking if available):")

# Prepare valid tokenized documents
tokenized_pages = []

for page_title in wiki_pages:
    try:
        page = wikipedia.page(page_title)
        text = page.content[:500]  # Limit to 1000 characters

        # Clean: remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text).lower()

        # Tokenize
        tokens = word_tokenize(text)

        tokenized_pages.append(tokens)

        print(f"✅ Page found: {page_title}")

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Disambiguation page for '{page_title}', skipping. Options: {e.options}")
    except wikipedia.exceptions.PageError:
        print(f"❌ Page not found: {page_title}")

if not tokenized_pages:
    print("\nNo valid pages found. Exiting.")
    exit()

# Create vocabulary from tokenized pages
vocabulary = set(word for tokens in tokenized_pages for word in tokens)

# Compute the term frequency for each document
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_pages]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Document {i+1}: {tf_vector}")

# Compute the Inverse Document Frequency (IDF)
idf = compute_idf(tokenized_pages, vocabulary)
print("\nInverse Document Frequency:")
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")

# Compute TF-IDF vectors
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

print("\nTF-IDF Vectors:")
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"Document {i+1}: {tfidf_vector}")

# Prints the most similar document pair
find_most_similar(tfidf_vectors, vocabulary)
