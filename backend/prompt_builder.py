"""
Prompt builder — constructs structured prompts for IBM Granite
using retrieved RAG context.
"""


def build_question_prompt(
    role: str,
    experience: str,
    focus_area: str,
    rag_context: list[dict],
) -> str:
    """
    Build a prompt asking Granite to generate tailored interview questions.
    """
    context_block = _format_context(rag_context)

    prompt = f"""<|system|>
You are an expert interview coach. Your job is to generate a set of tailored interview questions for a candidate, along with model answers and personalised improvement tips. Be specific, practical, and encouraging.
<|user|>
Generate a personalised interview preparation plan for the following candidate:

- Job Role: {role}
- Experience Level: {experience}
- Focus Area / Topic: {focus_area}

Use the reference Q&A examples below to calibrate difficulty and style:

--- REFERENCE EXAMPLES ---
{context_block}
--- END EXAMPLES ---

Provide exactly 5 interview questions with:
1. The question text
2. A concise model answer (3-5 sentences)
3. One specific improvement tip for this role and experience level

Format each question as:
**Question N:** <question>
**Model Answer:** <answer>
**Tip:** <tip>

After all 5 questions, add a short "Overall Preparation Strategy" section (3-4 bullet points).
<|assistant|>"""

    return prompt


def build_feedback_prompt(
    role: str,
    question: str,
    user_answer: str,
    rag_context: list[dict],
) -> str:
    """
    Build a prompt asking Granite to evaluate and improve a candidate's answer.
    """
    context_block = _format_context(rag_context[:2])

    prompt = f"""<|system|>
You are an expert interview coach providing constructive, honest, and encouraging feedback on a candidate's answer.
<|user|>
Role: {role}
Interview Question: {question}
Candidate's Answer: {user_answer}

Reference model answer for calibration:
{context_block}

Please provide:
1. **Score** (out of 10) with a brief justification
2. **Strengths** — what was done well (2-3 points)
3. **Improvements** — specific, actionable suggestions (2-3 points)
4. **Revised Answer** — a polished version of the candidate's answer
<|assistant|>"""

    return prompt


def build_chat_prompt(user_message: str, rag_context: list[dict]) -> str:
    """
    Build a general conversational prompt with RAG context for free-form
    interview questions (e.g. 'What questions do companies ask for ML roles?').
    """
    context_block = _format_context(rag_context)

    prompt = f"""<|system|>
You are a helpful, knowledgeable interview preparation assistant. You help candidates prepare for job interviews by providing guidance, sample questions, and coaching tips. Use the provided reference material to ground your answers.
<|user|>
{user_message}

Relevant reference material:
{context_block}
<|assistant|>"""

    return prompt


def _format_context(entries: list[dict]) -> str:
    if not entries:
        return "(No specific reference material found for this query.)"

    lines = []
    for i, entry in enumerate(entries, 1):
        lines.append(
            f"[{i}] Role: {entry.get('role', 'General')} | "
            f"Category: {entry.get('category', 'General')}\n"
            f"    Q: {entry['question']}\n"
            f"    A: {entry['model_answer']}\n"
            f"    Tip: {entry.get('tips', '')}"
        )
    return "\n\n".join(lines)
