## Self-Correcting RAG with Gemini & LangGraph
A smart RAG system that uses an AI Judge to critique and improve its own answers.

## Core Concept

This project mimics how a person thinks twice before answering. It generates an initial answer, has an independent "Judge" AI review it against the source material, and if the answer is weak, it automatically tries again from a different angle.

## The Logic Flowchart

Here's the step-by-step journey of a question through our system:

1. Start: The Question ❓
A user asks a question, like "What are the ethical concerns of AI?".

2. Retrieve: Gather Evidence 📚
The system searches its knowledge base (a Wikipedia article) for all documents relevant to the question.

3. Generate: The First Draft ✍️
An AI "Generator" creates an initial answer based only on the evidence it was given.

4. Judge: The Quality Check ⚖️
A separate, impartial "Judge" AI receives the question, the evidence, and the first-draft answer. It scores the answer from 0.0 to 1.0 based on two strict rules:

Faithfulness: Does the answer make things up? (Is it free of hallucinations?)

Relevance: Does it actually answer the original question?

5. Decide: The Fork in the Road 🚦
The system checks the Judge's score.

If the score is high (≥ 0.8): The answer is good. The process moves directly to the end.

If the score is low (< 0.8): The answer is not good enough. The system enters the correction loop.

6. Correct: A New Approach (If Needed) 🔄
The system gives the original question to an AI "Rephraser" which formulates a new, slightly different question (e.g., "What are the primary societal risks of AI?"). The process then jumps back to Step 2 (Retrieve) with this new question, hoping to find better evidence.

7. Finish: The Final Answer ✅
A high-quality, verified answer is presented to the user.

## Key Technologies

Logic: LangGraph

AI Models: Google Gemini

Embeddings: Google Generative AI

Vector Storage: FAISS

## Quick Start

Install Dependencies: 

Bash
'''pip install -r requirements.txt'''
Set API Key:
Set your GOOGLE_API_KEY in your environment.

Run the Project:

Bash
'''python main.py'''
