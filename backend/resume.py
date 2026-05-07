# job_description
# resume -> technical skills, academic

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from docx import Document
import os

load_dotenv()


def get_chat():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

class ResumeSchema(BaseModel):
    skills: list[str]
    project: list[str]
    experience: list[str]
    academic: list[str]


class ComparisonSchema(BaseModel):
    match_percentage: str
    matching_skills: list[str]
    missing_skills: list[str]
    suggestions: str

output_parser = JsonOutputParser(
    pydantic_object=ResumeSchema
)

comparison_parser = JsonOutputParser(
    pydantic_object=ComparisonSchema
)

formatted_instructions = output_parser.get_format_instructions()

comparison_format = comparison_parser.get_format_instructions()

def extract_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    resume_text = "\n".join(
        [doc.page_content for doc in docs]
    )

    return resume_text


def extract_from_document(file_path):
    doc = Document(file_path)

    resume_text = "\n".join(
        [para.text for para in doc.paragraphs]
    )

    return resume_text


def extract_text_from_doc(file_path):

    if file_path.endswith(".pdf"):
        resume_text = extract_from_pdf(file_path)

    elif file_path.endswith(".docx"):
        resume_text = extract_from_document(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {file_path}. "
            f"Only .pdf and .docx are supported."
        )

    return resume_text

def get_formatted_resume_info(
    file_path,
    format_instruction=formatted_instructions
):

    resume_text = extract_text_from_doc(file_path)

    prompt = ChatPromptTemplate.from_template(
        """
        Extract structured information from the given resume.

        Resume:
        {resume}

        {format_instruction}
        """
    )

    chain = prompt | get_chat() | output_parser

    response = chain.invoke({
        "resume": resume_text,
        "format_instruction": format_instruction
    })

    return response

def get_formatted_job_desc(
    job_description,
    format_instruction=formatted_instructions
):

    prompt = ChatPromptTemplate.from_template(
        """
        Extract structured information from the given job description.

        Job Description:
        {job_desc}

        {format_instruction}
        """
    )

    chain = prompt | get_chat() | output_parser

    response = chain.invoke({
        "job_desc": job_description,
        "format_instruction": format_instruction
    })

    return response

def comparison_function(file_path, job_description):

    try:

        resume_info = get_formatted_resume_info(file_path)

        job_desc_info = get_formatted_job_desc(job_description)

        prompt = ChatPromptTemplate.from_template(
            """
            Compare the resume and the job description.

            Resume:
            {resume}

            Job Description:
            {JD}

            Return:
            - match percentage
            - matching skills
            - missing skills
            - suggestions for improvement

            {format_instruction}
            """
        )

        chain = prompt | get_chat() | comparison_parser

        response = chain.invoke({
            "resume": str(resume_info),
            "JD": str(job_desc_info),
            "format_instruction": comparison_format
        })

        return response

    except ValueError as e:
        print(f"❌ Error: {e}")
        raise e

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e