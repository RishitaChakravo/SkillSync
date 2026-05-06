# job_description
# resume -> technical skills, academic
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from constant import GROQ_API_KEY
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from docx import Document

def get_chat():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

file_path = ""
skill_schema = ResponseSchema(name="skills", description="List of all the skills, tools, frameworks, programming languages the candidate has mentioned in the resume")
project_schema = ResponseSchema(name="project", description="List of all projects with the tools used, a little description and technologies used")
experience_schema = ResponseSchema(name="experience", description="Work experience in a particular industry with the role, company and number of years of service")
academic_schema = ResponseSchema(name="academic", description="List of all the education information related to degrees etc.")

response_schemas = [skill_schema, project_schema, experience_schema, academic_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
formatted_instructions = output_parser.get_format_instructions()

comparison_schemas = [
    ResponseSchema(name="match_percentage", description="Percentage match between resume and job description"),
    ResponseSchema(name="matching_skills", description="Skills present in both resume and job description"),
    ResponseSchema(name="missing_skills", description="Skills missing from resume but required in job description"),
    ResponseSchema(name="suggestions", description="genuine suggestions which can help to improve the resume based on what is missing and what can be added")
]

comparison_parser = StructuredOutputParser.from_response_schemas(comparison_schemas)
comparison_format = comparison_parser.get_format_instructions()

def extract_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    resume_text = "\n".join([doc.page_content for doc in docs])
    return resume_text

def extract_from_document(file_path):
    doc = Document(file_path)
    resume_text = "\n".join([para.text for para in doc.paragraphs])
    return resume_text

def extract_text_from_doc(file_path):
    if file_path.endswith(".pdf"):
        resume_text = extract_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        resume_text = extract_from_document(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Only .pdf and .docx are supported.")
    return resume_text

def get_formatted_resume_info(file_path, format_instruction=formatted_instructions):
    resume_text = extract_text_from_doc(file_path)

    prompt = ChatPromptTemplate.from_template("""Extract structured information from the given resume
                                              Resume: {resume}
                                              {format_instruction}
                                              """)
    chain = prompt | get_chat() | output_parser
    response = chain.invoke({"resume": resume_text, "format_instruction": format_instruction})
    return response

def get_formatted_job_desc(job_description, format_instruction=formatted_instructions):
    prompt = ChatPromptTemplate.from_template("""Extract structured information from the given job description
                                              Job_Description: {job_desc}

                                              {format_instruction}
                                              """)
    chain = prompt | get_chat() | output_parser
    response = chain.invoke({"job_desc": job_description, "format_instruction": format_instruction})
    return response

def comparison_function(file_path, job_description):
    try:
        resume_info = get_formatted_resume_info(file_path)
        job_desc_info = get_formatted_job_desc(job_description)

        prompt = ChatPromptTemplate.from_template("""
        Compare the job description and resume find the all the below informations from the given 
            Resume:{resume}
            Job Description :{JD}

            {format_instruction}                                    
        """)

        chain = prompt | get_chat() | comparison_parser
        response = chain.invoke({"resume": str(resume_info), "JD": str(job_desc_info), "format_instruction": comparison_format})
        return response
    except ValueError as e:
        print(f"❌ Error: {e}")
        raise e
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e


