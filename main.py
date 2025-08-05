import os
import sqlite3
from typing import List, TypedDict

# MODIFICATION: Imported WikipediaLoader instead of WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# --- Environment Setup ---
# Set your Google API key 
#os.environ["GOOGLE_API_KEY"] = "" 
# os.environ["LANGCHAIN_API_KEY"] = "ls__..."  # Optional, for LangSmith tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Optional, for LangSmith tracing


# --- LLM and Embedding Model Initialization ---
generator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- 1. Document Loading and Vector Store Setup ---

# This will search for "Artificial intelligence" on Wikipedia and load the content
# from the top search result.
print("--- 📚 Loading data from Wikipedia... ---")
loader = WikipediaLoader(query="Artificial intelligence", load_max_docs=1)
docs = loader.load()
print("--- ✅ Data loaded successfully. ---")


# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)

# Create a vector store from the document splits using Google's embeddings
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

# Create a retriever
retriever = vectorstore.as_retriever(k=4)


# --- 2. The Answer Generator Chain ---
GENERATE_PROMPT = """
You are an expert question-answering assistant. Your task is to answer the user's question based *only* on the provided context.

If the context does not contain the information needed to answer the question, state that you cannot answer. Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
generate_prompt_template = PromptTemplate(
    input_variables=["context", "question"], template=GENERATE_PROMPT
)
generator_chain = generate_prompt_template | generator_llm | StrOutputParser()


# --- 3. The Judge: Evaluator for Faithfulness and Relevance ---
class Evaluation(BaseModel):
    """Evaluation of the generated answer based on the retrieved context."""
    score: float = Field(
        description="A score from 0.0 to 1.0, where 1.0 is the best. The score represents the overall quality, combining faithfulness and relevance.", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="A detailed explanation of the score, citing specific parts of the context and the answer to justify the evaluation."
    )

JUDGE_PROMPT = """
You are an impartial and meticulous AI evaluator. Your task is to assess a generated answer based on a given context and question.

Your evaluation should be based on two criteria:
1.  **Faithfulness**: Does the answer strictly adhere to the information present in the context? It should not add any information or make any assumptions not supported by the context.
2.  **Relevance**: Does the answer directly and completely address the user's question?

Here is the data for evaluation:
CONTEXT:
{context}

QUESTION:
{question}

GENERATED ANSWER:
{answer}

Provide your evaluation based on the criteria above.
"""
judge_prompt_template = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template=JUDGE_PROMPT,
)
structured_llm_judge = judge_llm.with_structured_output(Evaluation)
judge_chain = judge_prompt_template | structured_llm_judge


# --- 4. The Corrector: Question Re-phrasing Chain ---
REPHRASE_PROMPT = """
You are an expert at rephrasing questions to be more specific and clear.
Your task is to take a user's question and reformulate it. The new question should be different from the original but retain the core intent.
This helps in retrieving more relevant documents from a knowledge base.

Original Question:
{question}

Rephrased Question:
"""
rephrase_prompt_template = PromptTemplate(
    input_variables=["question"], template=REPHRASE_PROMPT
)
rephrase_chain = rephrase_prompt_template | generator_llm | StrOutputParser()


# --- 5. Defining the Graph State and Nodes ---
class GraphState(TypedDict):
    original_question: str
    current_question: str
    context: str
    answer: str
    evaluation: Evaluation
    retry_count: int

def retrieve_node(state):
    print("--- 🧠 RETRIEVING DOCUMENTS ---")
    question = state["current_question"]
    documents = retriever.invoke(question)
    context_str = "\n\n".join([doc.page_content for doc in documents])
    return {"context": context_str, "retry_count": state.get("retry_count", 0) + 1}

def generate_node(state):
    print("--- 📝 GENERATING ANSWER ---")
    context = state["context"]
    question = state["current_question"]
    answer = generator_chain.invoke({"context": context, "question": question})
    return {"answer": answer}

def evaluate_node(state):
    print("--- ⚖️ EVALUATING ANSWER ---")
    context = state["context"]
    question = state["original_question"]
    answer = state["answer"]
    evaluation = judge_chain.invoke({"context": context, "question": question, "answer": answer})
    log_evaluation_to_db(state, evaluation)
    print(f"--- 🧐 JUDGE'S SCORE: {evaluation.score:.2f} ---")
    print(f"--- 🤔 JUDGE'S REASONING: {evaluation.reasoning} ---")
    return {"evaluation": evaluation}

def rephrase_node(state):
    print("--- ✍️ REPHRASING QUESTION ---")
    question = state["original_question"]
    new_question = rephrase_chain.invoke({"question": question})
    print(f"--- ❓ NEW QUESTION: {new_question} ---")
    return {"current_question": new_question}


# --- 6. Defining the Conditional Edges ---
def decide_edge(state):
    evaluation = state["evaluation"]
    retry_count = state["retry_count"]
    if evaluation.score >= 0.8:
        print("--- ✅ DECISION: Answer is good. Ending graph. ---")
        return "end"
    elif retry_count >= 2:
        print("--- ⚠️ DECISION: Max retries reached. Ending graph. ---")
        return "end"
    else:
        print("--- 🔄 DECISION: Answer is not good enough. Retrying. ---")
        return "rephrase"

# --- 7. Database Logging Setup ---
def setup_database():
    conn = sqlite3.connect("evaluation_logs_gemini.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_question TEXT,
        answer TEXT,
        score REAL,
        reasoning TEXT,
        context TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_evaluation_to_db(state, evaluation):
    conn = sqlite3.connect("evaluation_logs_gemini.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO evaluation_logs (original_question, answer, score, reasoning, context) VALUES (?, ?, ?, ?, ?)",
        (
            state["original_question"],
            state["answer"],
            evaluation.score,
            evaluation.reasoning,
            state["context"],
        ),
    )
    conn.commit()
    conn.close()

setup_database()


# --- 8. Assembling and Running the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("rephrase", rephrase_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    decide_edge,
    {"rephrase": "rephrase", "end": END}
)
workflow.add_edge("rephrase", "retrieve")

app = workflow.compile()


initial_question = "What are the ethical concerns surrounding artificial intelligence?"

# Invoke the graph with a question
inputs = {"original_question": initial_question, "current_question": initial_question}
final_state = app.invoke(inputs)

print("\n\n--- ✨ FINAL RESULT ---")
print(f"Original Question: {final_state['original_question']}")
print(f"Final Answer: {final_state['answer']}")
print(f"Final Score: {final_state['evaluation'].score:.2f}")