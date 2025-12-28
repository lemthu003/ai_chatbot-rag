import os, time, shutil, warnings
import streamlit as st
import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")

client = OpenAI(api_key = '') # write API key here
model = "gpt-4o-mini"

st.set_page_config(page_title="Helpful Paper Reader", layout="wide")
st.title("📚 Helpful Paper Reader")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def wait_for_run_completion(thread_id, run_id):
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        if run_status.status == "completed":
            break
        time.sleep(1)

@st.cache_resource
def setup_assistant():
    assistant = client.beta.assistants.create(
        name="(Thu) Helpful Paper Reader",
        instructions="Use your vector-store files to answer questions with citations.",
        model=model,
        tools=[{"type": "file_search"}],
    )
    vector_store = client.vector_stores.create(name="Paper Assistant Store")
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )
    return assistant, vector_store

assistant, vector_store = setup_assistant()

st.sidebar.header("📂 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Upload to Vector Store"):
    paths = []
    for f in uploaded_files:
        path = os.path.join(DATA_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        paths.append(path)

    streams = [open(p, "rb") for p in paths]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=streams
    )
    for s in streams:
        s.close()

    st.sidebar.success("Files uploaded!")

st.header("💬 Ask a Question")
question = st.text_area(
    "Enter your question:",
    "What's the most efficient way to train an LLM?"
)

if st.button("Ask Assistant"):
    with st.spinner("Thinking..."):
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        wait_for_run_completion(thread.id, run.id)

        messages = list(
            client.beta.threads.messages.list(
                thread_id=thread.id,
                run_id=run.id
            )
        )
        answer = messages[0].content[0].text.value
        st.markdown("### ✅ Answer")
        st.markdown(answer)
