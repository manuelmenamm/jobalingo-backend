from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from adzuna import AdzunaJobParser

app = FastAPI()

# Configure CORS to allow your frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", 'http://localhost:8080'], # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Adzuna parser
adzuna_parser = AdzunaJobParser()

@app.get("/api/jobs")
async def get_jobs(
    query: Optional[str] = None,
    industry: Optional[List[str]] = Query(None),
    location: Optional[List[str]] = Query(None),
    experience: Optional[List[str]] = Query(None),
    jobType: Optional[List[str]] = Query(None)
):
    """Fetch jobs from Adzuna API with optional filters"""
    
    # Print received parameters for debugging
    print(f"Received filters - Query: {query}, Industry: {industry}, Location: {location}, Experience: {experience}, JobType: {jobType}")
    
    # Prepare parameters for Adzuna API
    params = {}
    
    if query:
        params['query'] = query
        
    if location:
        params['location'] = location
    
    if industry:
        params['industry'] = industry
    
    if experience:
        params['experience'] = experience
    
    if jobType:
        params['jobType'] = jobType
    
    # Fetch jobs from Adzuna
    jobs = adzuna_parser.fetch_jobs(params)
    
    print(f"Returning {len(jobs)} jobs after filtering")
    
    return {"jobs": jobs}

@app.get("/")
async def root():
    return {"message": "Welcome to the JobAligo API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)