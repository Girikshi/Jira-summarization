from flask import Flask, render_template, request, jsonify

from jira_summary import predictor

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    jira_id = request.form["jira-id"]
    person_info = predictor(jira_id)

    return jsonify(
        {
            "jira_id": person_info.jira_id,
            "title": person_info.title,
            "assignee": person_info.assignee,
            "status": person_info.status,
            "summary": person_info.summary,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
