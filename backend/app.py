"""
Flask REST API for the Interview Trainer Agent.

Endpoints:
  POST /api/generate   — generate tailored interview questions
  POST /api/feedback   — evaluate a candidate's answer
  POST /api/chat       — free-form conversation with the agent
  GET  /api/roles      — list all available roles in the corpus
  GET  /api/health     — health check
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from prompt_builder import build_chat_prompt, build_feedback_prompt, build_question_prompt
from rag_engine import retrieve
from watsonx_client import generate_response

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend (same origin or localhost)

@app.route("/")
def index():
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    return send_from_directory(frontend_dir, "index.html")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Interview Trainer Agent"})


@app.get("/api/test_key")
def test_key():
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    key = os.getenv("WATSONX_API_KEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    
    masked_key = f"{key[:3]}...{key[-3:]}" if len(key) > 6 else "Too Short"
    
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        credentials = Credentials(url=url, api_key=key)
        model = ModelInference(
            model_id="ibm/granite-8b-code-instruct",
            credentials=credentials,
            project_id=project_id
        )
        res = model.generate_text(prompt="Say hi in 1 word")
        status = f"Success! Response: {res.strip()}"
    except Exception as e:
        status = f"Failed! Error: {e}"
        
    return jsonify({
        "url": url,
        "key_length": len(key),
        "masked_key": masked_key,
        "project_id": project_id,
        "status": status
    })



# ---------------------------------------------------------------------------
# GET /api/roles  — return unique roles from the corpus
# ---------------------------------------------------------------------------

@app.get("/api/roles")
def get_roles():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "interview_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    roles = sorted({entry["role"] for entry in corpus})
    return jsonify({"roles": roles})


# ---------------------------------------------------------------------------
# POST /api/generate — generate tailored interview questions
# ---------------------------------------------------------------------------

@app.post("/api/generate")
def generate_questions():
    """
    Request body:
      {
        "role": "Software Engineer",
        "experience": "2 years",
        "focus_area": "system design and algorithms"
      }
    """
    data = request.get_json(force=True) or {}

    role = data.get("role", "").strip()
    experience = data.get("experience", "").strip()
    focus_area = data.get("focus_area", "").strip()

    if not role:
        return jsonify({"error": "Field 'role' is required."}), 400

    # Build RAG query from user inputs
    rag_query = f"{role} {experience} {focus_area}"
    context = retrieve(rag_query, top_k=5)

    prompt = build_question_prompt(role, experience, focus_area, context)
    response = generate_response(prompt, max_new_tokens=900)

    return jsonify({
        "role": role,
        "experience": experience,
        "focus_area": focus_area,
        "questions": response,
        "sources_used": len(context),
    })


# ---------------------------------------------------------------------------
# POST /api/feedback — evaluate a candidate's answer
# ---------------------------------------------------------------------------

@app.post("/api/feedback")
def get_feedback():
    """
    Request body:
      {
        "role": "Data Scientist",
        "question": "What is the bias-variance trade-off?",
        "answer": "It's about balancing how well the model fits training data..."
      }
    """
    data = request.get_json(force=True) or {}

    role = data.get("role", "").strip()
    question = data.get("question", "").strip()
    user_answer = data.get("answer", "").strip()

    if not question or not user_answer:
        return jsonify({"error": "Fields 'question' and 'answer' are required."}), 400

    rag_query = f"{role} {question}"
    context = retrieve(rag_query, top_k=3)

    prompt = build_feedback_prompt(role, question, user_answer, context)
    response = generate_response(prompt, max_new_tokens=700)

    return jsonify({
        "role": role,
        "question": question,
        "feedback": response,
    })


# ---------------------------------------------------------------------------
# POST /api/chat — free-form conversation
# ---------------------------------------------------------------------------

@app.post("/api/chat")
def chat():
    """
    Request body:
      { "message": "What are the most common React interview questions?" }
    """
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    context = retrieve(message, top_k=4)
    prompt = build_chat_prompt(message, context)
    response = generate_response(prompt, max_new_tokens=600)

    return jsonify({"reply": response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting Interview Trainer Agent on port {port}…")
    app.run(host="0.0.0.0", port=port, debug=debug)
