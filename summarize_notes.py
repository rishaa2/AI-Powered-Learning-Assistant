import os
import asyncio
from typing import List, Dict, Any, Tuple
from functools import partial

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize the LLM
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

# Document processing function
async def process_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Process a single document to extract summary and QA pairs"""
    try:
        # Use ProcessPoolExecutor for CPU-bound operations
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            _process_document_sync, 
            file_path, 
            filename
        )
        return result
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "summary": f"Error processing file: {str(e)}",
            "qa_pairs": []
        }

def _process_document_sync(file_path: str, filename: str) -> Dict[str, Any]:
    """Synchronous document processing function to be run in a separate process"""
    # Load document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)
    
    # Get LLM
    llm = get_llm()
    
    # Create summary chain
    summary_template = """
    ###Do not process PDF files unrelated to Engineering and Scietific domains.
    You are a helpful assistant that summarizes college engineering notes focusing on domains like- CSE, Mechanical Engineering, Electrical Engineering, Information Technology, Electronics and Communication Engineering.
    Please Summarize the following content in a concise way, capturing the key concepts:
    
    {text}
    
    SUMMARY:
    """
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["text"])
    summary_chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=summary_prompt
    )
    
    # Create QA generation chain
    qa_template = """
    You are a helpful assistant that creates exam preparation questions for college engineering students.
    Based on the following engineering notes, create 40 important questions and answers that might appear in an exam.
    Format each Q&A pair as a JSON object with "question" and "answer" fields.
    
    NOTES:
    {text}
    
    QUESTIONS AND ANSWERS (in JSON format):
    """
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["text"])
    qa_chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt
    )
    
    # Generate summary
    summary = summary_chain.run(docs)
    
    # Generate QA pairs
    qa_text = qa_chain.run(docs)
    
    # Process QA pairs from text to structured format
    import json
    import re
    
    # Clean up QA text to make it valid JSON
    qa_text = qa_text.strip()
    # Remove any text before the first [
    if "[" in qa_text:
        qa_text = qa_text[qa_text.find("["):]
    # Remove any text after the last ]
    if "]" in qa_text:
        qa_text = qa_text[:qa_text.rfind("]") + 1]
    
    try:
        qa_pairs = json.loads(qa_text)
    except json.JSONDecodeError:
        # Handle case where output is not valid JSON
        # Extract Q&A pairs using regex
        qa_pattern = r'"question"\s*:\s*"([^"]*)"\s*,\s*"answer"\s*:\s*"([^"]*)"'
        matches = re.findall(qa_pattern, qa_text)
        qa_pairs = [{"question": q, "answer": a} for q, a in matches]
        
        if not qa_pairs:
            # Fallback if regex doesn't work
            qa_pairs = [
                {"question": "What are the key concepts in this document?", 
                 "answer": "Please refer to the summary for key concepts."}
            ]
    
    # Return results
    return {
        "filename": filename,
        "summary": summary.strip(),
        "qa_pairs": qa_pairs
    }

async def process_documents(file_paths: List[str], file_names: List[str]) -> List[Dict[str, Any]]:
    """Process multiple documents in parallel"""
    tasks = []
    for file_path, filename in zip(file_paths, file_names):
        task = asyncio.create_task(process_document(file_path, filename))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results