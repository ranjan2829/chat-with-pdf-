import streamlit as st
import PyPDF2
from transformers import pipeline

# Function to extract text from a PDF page
def extract_text_from_page(page):
    return page.extract_text()

# Function to summarize text using the Hugging Face summarization model
def generate_summary(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to count the total number of pages in the PDF
def count_pages(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return len(pdf_reader.pages)

# Streamlit app
def main():
    st.set_page_config(page_title="PDF Page Summarizer", page_icon=":page_with_curl:")

    st.title("PDF Page Summarizer")

    # File upload
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file is not None:
        # Total number of pages in the PDF
        total_pages = count_pages(pdf_file)
        st.sidebar.subheader("Navigation")
        page_number = st.sidebar.number_input("Enter page number", min_value=1, max_value=total_pages, value=1)

        # Read PDF and display page content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page = pdf_reader.pages[page_number - 1]
        page_text = extract_text_from_page(page)

        st.subheader(f"Page {page_number}")
        st.write(page_text)

        # Summarize page content
        summary = generate_summary(page_text)
        st.subheader("Summary")
        st.write(summary)

        # Page navigation
        st.sidebar.markdown("---")
        st.sidebar.write(f"Total pages: {total_pages}")
        if st.sidebar.button("Next Page"):
            page_number = min(page_number + 1, total_pages)
            st.experimental_rerun()
        if st.sidebar.button("Previous Page"):
            page_number = max(page_number - 1, 1)
            st.experimental_rerun()

if __name__ == "__main__":
    main()
