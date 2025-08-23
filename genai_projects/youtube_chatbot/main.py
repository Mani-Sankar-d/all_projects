from fastapi import FastAPI
from ytchat import run
from pydantic import BaseModel
app=FastAPI()

class Req(BaseModel):
    query:str
    id:str

class Res(BaseModel):
    response:str

@app.get('/')
def home():
    return {'message':'home'}
@app.post('/chat',response_model=Res)
def give(req:Req):
    answer = run(req.query,req.id)
    return Res(response=answer)