from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    if request.method == "POST":
        # image prediction code
        pass
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
