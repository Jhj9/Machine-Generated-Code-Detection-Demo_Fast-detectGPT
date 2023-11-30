from fastapi import FastAPI, Request
from router import punctuation
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from pathlib import Path


origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://3.38.211.186:3000" #FIXME: production url
]
templates = Jinja2Templates(directory="templates")


app = FastAPI()
app.include_router(punctuation.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "backend/static"),
    name="static",
)

@app.get('/')
async def mainPage(request: Request):
    return templates.TemplateResponse('demo.html', {'request': request})