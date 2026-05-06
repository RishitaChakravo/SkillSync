from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from pydantic import BaseModel
from resume import comparison_function
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("origins"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    file_path: str
    job_description: str

@app.post('/')
async def submit(file: UploadFile = File(...), jd: str = Form(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())
    try:
        response = comparison_function(
            file_path=file_location,
            job_description=jd
        )
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return response

if __name__== "__main__":
    uvicorn.run(app, host="localhost", port=8000)