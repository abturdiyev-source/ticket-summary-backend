from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Ticket Summary API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    text: str

class SummaryRequest(BaseModel):
    messages: list[Message]
    model: str = "gemini-2.5-flash"

class SummaryResponse(BaseModel):
    summary: str
    inputTokens: int
    outputTokens: int
    thinkingTokens: int
    totalTokens: int

@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI service not configured")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array is required")

    conversation = "\n".join([
        f"{'Client' if m.role == 'client' else 'Support'}: {m.text}"
        for m in request.messages
    ])

    prompt = f"""Ты — AI, который делает краткое саммари тикета.
Суммаризуй диалог между клиентом и оператором в 2–3 предложениях.
Выдели основную проблему клиента и итоговое решение.
Вот диалог:

{conversation}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent?key={api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30.0
        )

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if response.status_code == 402:
        raise HTTPException(status_code=402, detail="Payment required")
    if not response.is_success:
        raise HTTPException(status_code=500, detail="Failed to generate summary")

    data = response.json()

    summary = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No summary")
    usage = data.get("usageMetadata", {})

    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    total_tokens = usage.get("totalTokenCount", 0)
    thinking_tokens = total_tokens - input_tokens - output_tokens

    return SummaryResponse(
        summary=summary,
        inputTokens=input_tokens,
        outputTokens=output_tokens,
        thinkingTokens=thinking_tokens,
        totalTokens=total_tokens
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
