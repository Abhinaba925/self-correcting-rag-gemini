# 🧠 Self-Correcting RAG with Gemini & LangGraph

> A smart RAG system that uses an AI Judge to critique and improve its own answers.

This project mimics how a person thinks twice before answering. It generates an initial answer, has an independent "Judge" AI review it against the source material, and if the answer is weak, it automatically tries again from a different angle.

---

## ⚙️ How It Works: The Self-Correction Loop

Here's the step-by-step journey of a question through our system:

1.  **❓ Start: The Question**
    A user asks a question, like "What are the ethical concerns of AI?".

2.  **📚 Retrieve: Gather Evidence**
    The system searches its knowledge base (e.g., a Wikipedia article) for all documents relevant to the question.

3.  **✍️ Generate: The First Draft**
    An AI "Generator" creates an initial answer based *only* on the evidence it was given.

4.  **⚖️ Judge: The Quality Check**
    A separate, impartial "Judge" AI receives the question, evidence, and the first-draft answer. It scores the answer from `0.0` to `1.0` based on two strict rules:
    * **Faithfulness:** Is the answer free of hallucinations and grounded in the source material?
    * **Relevance:** Does it actually answer the original question?

5.  **🚦 Decide: The Fork in the Road**
    The system checks the Judge's score.
    * **If the score is high (`≥ 0.8`):** The answer is good. The process moves directly to the end.
    * **If the score is low (`< 0.8`):** The answer is not good enough. The system enters the correction loop.

6.  **🔄 Correct: A New Approach (If Needed)**
    The system gives the original question to an AI "Rephraser" which formulates a new, slightly different question (e.g., "What are the primary societal risks of AI?"). The process then jumps back to Step 2 (Retrieve) with this new question, hoping to find better evidence.

7.  **✅ Finish: The Final Answer**
    A high-quality, verified answer is presented to the user.

---

## 🛠️ Tech Stack

* **AI Models**: Google Gemini
* **Orchestration**: LangGraph
* **Embeddings**: Google Generative AI
* **Vector Storage**: FAISS

---

## 🚀 Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key**
    Set your `GOOGLE_API_KEY` in your environment.

3.  **Run the Project**
    ```bash
    python main.py
    ```

    # 🤖💬 A Conversation in Self-Correction: The Smarter RAG

Imagine two AI agents, **Generator (Gen)** and **Judge (Jud)**, collaborating to answer a user's question.

---

**User:** What are the ethical concerns of AI?

---

**Gen:** (Excitedly) Okay, based on the retrieved Wikipedia article, here's my first attempt: "AI ethics involves discussions about job displacement and the potential for superintelligence."

---

**Jud:** (Scrutinizing the answer and the source material) Hmm, let's see... While job displacement is mentioned, the definition of "superintelligence" wasn't really detailed in the context of ethical *concerns* in the provided documents. Also, it feels a bit narrow.

**Jud:** (Scores the answer internally) *Faithfulness: 0.7 (Slightly off-topic on superintelligence)*, *Relevance: 0.6 (Doesn't fully capture the breadth of ethical concerns)*. Overall score: **0.65**. Not quite there!

---

**System:** (Notices the low score and activates the Rephraser) Time to rethink! Let's ask our Rephraser for a new angle on the original question.

---

**Rephraser:** (Thinking) The user asked about ethical concerns... perhaps a more specific angle would yield better results. How about: "What are the primary societal risks associated with the development and deployment of artificial intelligence?"

---

**System:** (Using the rephrased question to retrieve information again) Let's try this again!

---

**Gen:** (After another retrieval round, with more focused evidence) Okay, take two! My answer is: "Key ethical concerns of AI include potential biases in algorithms leading to unfair outcomes, the lack of transparency in complex AI models making accountability difficult, and the implications for privacy due to the vast amounts of data AI systems often require."

---

**Jud:** (Reviewing the new answer against the source) Much better! This aligns well with the discussion of bias, transparency, and data privacy in the provided text.

**Jud:** (Scores the answer) *Faithfulness: 0.9 (Strongly supported by the source)*, *Relevance: 0.85 (Directly addresses ethical risks)*. Overall score: **0.88**. Excellent!

---

**System:** (Presents the improved answer to the user)

**✅ Final Answer:** Key ethical concerns of AI include potential biases in algorithms leading to unfair outcomes, the lack of transparency in complex AI models making accountability difficult, and the implications for privacy due to the vast amounts of data AI systems often require.

---

**Gen:** (Smiling) Nailed it! Sometimes you need a second opinion to get things right.

**Jud:** (Nodding) Indeed! It's all about ensuring accuracy and relevance.

---

This little "chat" illustrates the core concept: our RAG system uses a critical "Judge" to ensure the "Generator" provides high-quality, well-supported answers through a process of self-correction.
