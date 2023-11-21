from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline
from pydantic import BaseModel
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import random

class Item(BaseModel):

    text: str


templates = Jinja2Templates(directory="templates")

config_annotation = {
    "max_length": random.randint(250, 400),
    # "min_length":30,
    "temperature": 1.1,
    "top_p": 2.,
    "num_beams": 10,
    "repetition_penalty": 1.5,
    "num_return_sequences": 9,
    "no_repeat_ngram_size": 2,
    "do_sample": True
}

config_intro = {
    "max_length": random.randint(600, 800),
    # "min_length":30,
    "temperature": 1.1,
    "num_beams": 5,
    "repetition_penalty": 1.5,
    "num_return_sequences": 4,
    "no_repeat_ngram_size": 2,
    "do_sample": True
}


# load model and tokenizer
model_name = 'ai-forever/rugpt3large_based_on_gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)

generation = pipeline('text-generation', model=model,
                      tokenizer=tokenizer, device=-1)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_class=HTMLResponse)
async def create_item(request: Request, text: str = Form(...), config: str = Form('intro')):
    if config == 'intro':
        config_use = config_intro
        limit_words = 200
    elif config == 'annotation':
        config_use = config_annotation
        limit_words = 160
    else:
        config_use = config_intro  # use intro as default
        limit_words = 500  # ну допустим
    output = generation(text, **config_use)[0]['generated_text']
    output_words = output.split()[:limit_words]
    output_limited = ' '.join(output_words)

    # Find the first occurrence of "<s>"
    cut_index = output_limited.find('<s>')
    # If "<s>" was found, trim the string to this point
    if cut_index != -1:
        output_limited = output_limited[:cut_index]

    # Find the last occurrence of "."
    last_period = output_limited.rfind(".")
    # If a period was found, trim the string to this point
    if last_period != -1:
        output_limited = output_limited[:last_period+1]

    return templates.TemplateResponse("index.html", {"request": request, "generated_text": output_limited})