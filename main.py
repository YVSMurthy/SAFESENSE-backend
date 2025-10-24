from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from classification import classifier
import tempfile
import shutil

app = FastAPI()

@app.get("/")
def home():
    return JSONResponse({"message": "FastAPI running on Vercel!"})

@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = classifier.classify_audio(temp_path)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)