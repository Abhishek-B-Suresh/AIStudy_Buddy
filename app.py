from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def answer_question(question, context):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question, context])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return context, float(similarity[0][0])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]
    context = data["context"]

    answer, score = answer_question(question, context)

    return jsonify({
        "answer": answer,
        "score": round(score, 2)
    })

if __name__ == "_main_":
    app.run(debug=True)