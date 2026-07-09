import streamlit as st
from anthropic import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from rag import answer, get_collection

st.set_page_config(page_title="Shiryō RAG", page_icon="📚")
st.title("📚 Shiryō RAG")
st.caption(
    "Ask the Temple University Japan student handbooks anything — "
    "every answer is grounded in the documents, with citations."
)

if get_collection().count() == 0:
    st.error("The document index is empty. Run `python ingest.py` first — see README.")
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="e.g. How many hours per week can international students work part-time?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the handbooks..."):
        try:
            result = answer(question)

            st.divider()
            if not result["answerable"]:
                st.warning(f"🤷 {result['answer']}")
                st.caption(
                    "The system refuses to guess: if it isn't in the handbooks, it says so."
                )
            else:
                st.markdown(result["answer"])

                chunk_by_id = {c["id"]: c for c in result["chunks"]}
                if result["citations"]:
                    st.markdown("#### Sources")
                    for cit in result["citations"]:
                        chunk = chunk_by_id.get(cit["chunk_id"])
                        where = (
                            f"{chunk['source']}, page {chunk['page']}"
                            if chunk
                            else cit["chunk_id"]
                        )
                        st.markdown(f"> \"{cit['quote']}\"\n>\n> — *{where}*")

            with st.expander("Retrieved excerpts (what the model saw)"):
                for c in result["chunks"]:
                    st.markdown(f"**[{c['id']}]** {c['source']}, p.{c['page']}")
                    st.text(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

            u = result["usage"]
            st.caption(f"Tokens: {u['input']} in / {u['output']} out")

        except AuthenticationError:
            st.error("API key was rejected. Check the key in your .env file.")
        except RateLimitError:
            st.error("Rate limited — wait a minute and try again.")
        except APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
        except APIConnectionError:
            st.error("Couldn't reach the API. Check your internet connection.")
