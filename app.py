import streamlit as st
from transformers import pipeline
import torch
import random
import sqlparse


st.title("SmolLM2-FT-SQL: NL2SQL Demo")

# Check device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # For Apple Silicon (M1/M2) with PyTorch MPS support
else:
    device = "cpu"


@st.cache_resource(show_spinner=True)
def load_generator():
    return pipeline(
        "text-generation",
        model="thiborose/SmolLM2-FT-SQL",
        device=device,
    )


generator = load_generator()

RANDOM_QUESTIONS = [
    "List all employees who joined after 2020.",
    "Show all orders shipped to France.",
    "Find the average price of products in the Electronics category.",
    "Get all customers with more than 3 purchases.",
    "Show the total sales for each month in 2024.",
    "Which products have never been ordered?",
    "List the top 10 customers by total purchase amount in the last year.",
    "Show all orders that include more than 3 different products.",
    "Find employees who have not submitted any expense reports in 2023.",
    "Get the monthly active users for each region in 2024.",
    "List all suppliers who delivered late more than twice.",
    "Show the average delivery time for each shipping method.",
    "Find all students who are enrolled in both Math and Science courses.",
    "List the top 5 products with the highest return rate.",
    "Show all invoices that are overdue by more than 30 days.",
    "Get the total number of orders and total revenue per customer.",
    "Find the most common job title among employees in the IT department.",
    "List all projects that have not started yet but have assigned team members.",
    "Show the change in inventory levels for each product over the last 6 months.",
    "Find all customers who have placed orders in every quarter of 2023.",
]

if "question_text" not in st.session_state:
    st.session_state["question_text"] = ""


def insert_random_question():
    st.session_state["question_text"] = random.choice(RANDOM_QUESTIONS)


st.button("Insert random question", on_click=insert_random_question)

question = st.text_area("Enter your question:", key="question_text")

if st.button("Generate SQL"):
    with st.spinner("Generating..."):
        output = generator(
            [{"role": "user", "content": question}],
            max_new_tokens=128,
            return_full_text=False,
        )[0]

        formatted_sql = sqlparse.format(
            output["generated_text"],
            reindent=True,  # Adds line breaks and indentations
            keyword_case="upper",  # Makes SQL keywords uppercase
        )

        # Display the generated SQL in a styled box
        st.code(formatted_sql, language="sql", line_numbers=True, wrap_lines=True)
