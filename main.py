from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def home():
    return JSONResponse({"message": "FastAPI running on Vercel!"})