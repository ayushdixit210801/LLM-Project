# RAG Application

This application allows you to upload PDFs and ask questions about their content using Google's Gemini Pro LLM.

## Prerequisites

- Python 3.8 or higher
- Gemini API key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Setup and Running

1. Create a `.env` file in the root directory with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
uvicorn backend.main:app --reload
```

The application will be running at `http://localhost:8000`

You can access the interactive API documentation and test the endpoints using the Swagger UI at:
```
http://127.0.0.1:8000/docs
```

## How to Use

### Using the Swagger UI
1. Open `http://127.0.0.1:8000/docs` in your browser
2. Use the interactive interface to:
   - Upload PDFs through the `/api/upload` endpoint
   - Ask questions about the documents using the `/api/query` endpoint

### Using curl commands
1. Upload a PDF:
```bash
curl -X POST -F "file=@your_file.pdf" http://localhost:8000/api/upload
```

2. Ask questions about the PDF:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"query":"What is this document about?"}' \
     http://localhost:8000/api/query
```
