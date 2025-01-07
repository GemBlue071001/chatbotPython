from typing import List, Union, Dict
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_community.llms.ollama import Ollama
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import uvicorn
from typing import List, Union, Dict
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class InputChat(BaseModel):
   messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
       ..., 
       description="Chat messages"
   )

class CVData(BaseModel):
   status: str
   data: Dict
   file_info: Dict
   parser_used: str
   message: str

def get_exact_skill_match(skill_query):
   skill_mapping = {
       "java": {
           "exact": ["SKILL_EXACT_Java"],
           "exclude": ["SKILL_EXACT_JavaScript", "SKILL_EXACT_Java Script"]
       },
       "javascript": {
           "exact": ["SKILL_EXACT_JavaScript", "SKILL_EXACT_Java Script"],
           "exclude": ["SKILL_EXACT_Java"]
       },
       "react": {
           "exact": ["SKILL_EXACT_React", "SKILL_EXACT_ReactJS"],
           "exclude": ["SKILL_EXACT_React Native"]
       },
       "reactnative": {
           "exact": ["SKILL_EXACT_React Native"],
           "exclude": ["SKILL_EXACT_React", "SKILL_EXACT_ReactJS"]
       },
       "python": {
           "exact": ["SKILL_EXACT_Python"],
           "exclude": []
       },
       "nodejs": {
           "exact": ["SKILL_EXACT_NodeJS", "SKILL_EXACT_Node.js"],
           "exclude": []
       },
       "typescript": {
           "exact": ["SKILL_EXACT_TypeScript"],
           "exclude": []
       }
   }
   
   normalized_query = skill_query.lower().replace(" ", "")
   
   if normalized_query in skill_mapping:
       return {
           "include": skill_mapping[normalized_query]["exact"],
           "exclude": skill_mapping[normalized_query]["exclude"]
       }
   return {"include": [skill_query], "exclude": []}

def load_and_split_documents(file_path):
   try:
       path = Path(file_path)
       logger.info(f"Loading document from: {path.absolute()}")
       
       with open(path, 'r', encoding='utf-8') as f:
           content = f.read()
       
       data = json.loads(content)
       documents = []
       
       if isinstance(data, dict) and 'result' in data:
           jobs = data['result']
           for job in jobs:
               exact_skills = [f"SKILL_EXACT_{skill}" for skill in job.get('skillSets', [])]
               
               doc_text = f"""
               ID: {job.get('id')}
               Title: {job.get('jobTitle')}
               Link: {job.get('link')}
               Location: {', '.join(job.get('jobLocationCities', []))}
               Address: {', '.join(job.get('jobLocationAddressDetail', []))}
               Skills: {', '.join(job.get('skillSets', []))}
               Exact_Skills: {', '.join(exact_skills)}
               Salary: {job.get('salary')}
               MinSalary: {job.get('minsalary')}
               ExpiryDate: {job.get('expiryDate')}
               Company: {job.get('companyName')}
               JobType: {job.get('jobType', {}).get('name')}
               """
               documents.append(Document(page_content=doc_text))
               
       logger.info(f"Created {len(documents)} documents")
       return documents
       
   except Exception as e:
       logger.error(f"Error loading document: {str(e)}", exc_info=True)
       raise RuntimeError(f"Failed to load document: {str(e)}")

def initialize_vectorstore(documents):
   try:
       embeddings = GPT4AllEmbeddings()
       vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
       logger.info("Successfully initialized vector store")
       return vectorstore
   except Exception as e:
       logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
       raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

system_prompt = """
You are a job search assistant specializing in helping users find job opportunities from our database. You will analyze and provide information about job postings based on users' requirements and queries.

Present job listings in this EXACT format:

For queries with skill:
Found [NUMBER] [SKILL] jobs in [LOCATION]:

For queries without skill (return top 5 jobs):
Found [NUMBER] jobs in [LOCATION]:

If no jobs found:
EN: "No jobs found matching your criteria."
VN: "Không tìm thấy công việc phù hợp với yêu cầu của bạn."

Job listing format:
Job #1:
- ID: [ID]
- Title: [TITLE]
- Company: [COMPANY]
- Location: [CITY]
- Address: [ADDRESS]
- Job Type: [JOBTYPE_NAME]
- Skills: [SKILLS_LIST]
- Salary Range: [MINSALARY] - [SALARY] VND
- Expires: [EXPIRYDATE]
- Link: [LINK]

[Additional jobs in same format]

Rules:
1. Technology matching:
  - Java: only SKILL_EXACT_Java
  - JavaScript: only SKILL_EXACT_JavaScript
  - React: only SKILL_EXACT_React
  - React Native: only SKILL_EXACT_React Native

2. Non-job queries:
  EN: "I can only answer questions about jobs and job postings."
  VN: "Tôi chỉ trả lời câu hỏi về việc làm và tin tuyển dụng."

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
   ("system", system_prompt),
   MessagesPlaceholder(variable_name="messages")
])

model = Ollama(model="llama3")
parser = StrOutputParser()

BASE_DIR = Path(__file__).resolve().parent
news_file_path = os.getenv("JOBPOST_PATH", str(BASE_DIR / "jobpost.txt"))

def format_chat_messages(input_data):
   messages = input_data.get("messages", []) if isinstance(input_data, dict) else input_data.messages
   last_human_message = next((msg for msg in reversed(messages) 
                            if isinstance(msg, HumanMessage)), None)
   if last_human_message is None:
       return {"messages": messages, "context": ""}
       
   content = last_human_message.content
   location_filter = None
   skill_matches = {"include": [], "exclude": []}
   
   # Extract location
   locations = {
       "ha noi": "HA NOI",
       "hanoi": "HA NOI", 
       "ho chi minh": "HO CHI MINH",
       "hcm": "HO CHI MINH",
       "saigon": "HO CHI MINH",
       "sai gon": "HO CHI MINH"
   }
   
   content_lower = content.lower()
   for loc_key, loc_value in locations.items():
       if loc_key in content_lower:
           location_filter = loc_value
           break
   
   # Check for skill
   has_skill = False
   for skill in ["Java", "JavaScript", "React", "React Native", "Python", "Node.js", "TypeScript"]:
       if skill.lower() in content_lower:
           has_skill = True
           skill_matches = get_exact_skill_match(skill)
           content = f"{content} INCLUDE:{','.join(skill_matches['include'])} EXCLUDE:{','.join(skill_matches['exclude'])}"
           break
           
   relevant_docs = vectorstore.similarity_search(content)
   
   # Apply filters
   filtered_docs = []
   for doc in relevant_docs:
       doc_content = doc.page_content
       location_match = not location_filter or location_filter in doc_content
       skill_match = not has_skill or (
           any(include in doc_content for include in skill_matches['include']) and 
           not any(exclude in doc_content for exclude in skill_matches['exclude'])
       )
       
       if location_match and skill_match:
           filtered_docs.append(doc)
   
   # Limit to top 5 if no skill specified
   if not has_skill and filtered_docs:
       filtered_docs = filtered_docs[:5]
   
   formatted_context = format_docs(filtered_docs)
   logger.info(f"Found {len(filtered_docs)} relevant documents after filtering")
   return {
       "messages": messages,
       "context": formatted_context
   }

def transform_input(input_data):
   formatted_data = format_chat_messages(input_data)
   return {
       "context": formatted_data["context"],
       "messages": formatted_data["messages"]
   }

chain = (
   RunnablePassthrough()
   | transform_input
   | prompt 
   | model 
   | parser
)

app = FastAPI(
   title="Job Search Assistant",
   version="1.0",
   description="A job search assistant API using LangChain's Runnable interfaces"
)

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
   expose_headers=["*"],
)

@app.post("/compare-cv/")
async def compare_cv(cv_data: CVData):
    try:
        skills = cv_data.data["professional"]["technical_skills"]
        experience = cv_data.data["professional"]["experience"][0]["years"][0] if cv_data.data["professional"]["experience"] else "0"
        education = cv_data.data["professional"]["education"][0]["qualification"][0] if cv_data.data["professional"]["education"] else None
        
        cv_text = f"""
        Skills: {', '.join(skills)}
        Experience: {experience} years
        Education: {education}
        """
        
        for skill in skills:
            skill_matches = get_exact_skill_match(skill)
            cv_text += f" INCLUDE:{','.join(skill_matches['include'])} EXCLUDE:{','.join(skill_matches['exclude'])}"
        
        relevant_docs = vectorstore.similarity_search(cv_text)
        filtered_docs = [
            doc for doc in relevant_docs 
            if any(skill.lower() in doc.page_content.lower() for skill in skills)
        ]
        
        job_matches = []
        for doc in filtered_docs[:5]:
            lines = doc.page_content.strip().split('\n')
            job_details = {}
            for line in lines:
                line = line.strip()
                key, *value = line.split(':', 1)
                if value:
                    job_details[key.lower()] = value[0].strip()
            job_matches.append(job_details)
        
        return {
            "message": "CV compared successfully",
            "cv_analysis": {
                "skills": skills,
                "years_experience": float(experience),
                "education": education
            },
            "matches": job_matches
        }

    except Exception as e:
        logger.error(f"Error comparing CV: {str(e)}")
        return {"error": str(e)}

@app.post("/update-jobpost/")
async def update_jobpost(jobpost_data: Dict):
   try:
       file_path = os.getenv("JOBPOST_PATH", str(BASE_DIR / "jobpost.txt"))
       
       if isinstance(jobpost_data, dict) and 'result' in jobpost_data:
           filtered_jobs = []
           for job in jobpost_data['result']:
               filtered_job = {
                   'id': job.get('id'),
                   'link': job.get('link'),
                   'jobTitle': job.get('jobTitle'),
                   'salary': job.get('salary'),
                   'minsalary': job.get('minsalary'),
                   'expiryDate': job.get('expiryDate'),
                   'companyName': job.get('companyName'),
                   'jobType': job.get('jobType'),
                   'jobLocationCities': job.get('jobLocationCities'),
                   'jobLocationAddressDetail': job.get('jobLocationAddressDetail'),
                   'skillSets': job.get('skillSets')
               }
               filtered_jobs.append(filtered_job)
           
           jobpost_data['result'] = filtered_jobs

       with open(file_path, 'w', encoding='utf-8') as f:
           json.dump(jobpost_data, f, ensure_ascii=False, indent=4)
           
       return {
           "message": "Jobpost updated successfully",
           "file_path": file_path
       }
       
   except Exception as e:
       logger.error(f"Error updating jobpost: {str(e)}")
       return {"error": str(e)}

try:
   documents = load_and_split_documents(news_file_path)
   vectorstore = initialize_vectorstore(documents)
   logger.info("Successfully loaded documents and initialized vector store")
except Exception as e:
   logger.error(f"Startup error: {str(e)}", exc_info=True)
   raise

add_routes(
   app,
   chain.with_types(input_type=InputChat),
   enable_feedback_endpoint=False,
   enable_public_trace_link_endpoint=False,
   playground_type="chat"
)

# if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8080)