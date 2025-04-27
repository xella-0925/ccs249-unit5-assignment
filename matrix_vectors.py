import wikipedia
import re
from nltk.tokenize import word_tokenize  
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

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
        text = page.content[:500]  # Limit to 500 characters

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

# Train the Word2Vec model
model = gensim.models.Word2Vec(sentences=tokenized_pages, vector_size=100, window=5, min_count=1, workers=4)

# Function to get average word vector for a document
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Create document vectors by averaging word vectors
doc_vectors = np.array([get_doc_vector(doc, model) for doc in tokenized_pages])

# Corresponding labels for classification (one per document)
labels = [0, 1, 2, 3, 4]  # Each document has a unique label

# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(doc_vectors, labels)

# Predict the same labels 
predictions = classifier.predict(doc_vectors)

# Output the classification report
print("Classification Report:\n")
print(classification_report(labels, predictions, zero_division=1))

accuracy = accuracy_score(labels, predictions)
conf_matrix = confusion_matrix(labels, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
