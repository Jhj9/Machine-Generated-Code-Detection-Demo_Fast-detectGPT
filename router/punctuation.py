from fastapi import APIRouter, File, UploadFile, Depends
from service import PunctuationGenerator
from fastapi_restful.inferring_router import InferringRouter
from fastapi_utils.cbv import cbv
from pydantic import BaseModel


class TextInferenceInput(BaseModel):
    text: str

router = InferringRouter()

@cbv(router)
class Punctuation:

    print('**** setting Punctuation models ****')
    svc = PunctuationGenerator()

    @router.get('/punc/hello')
    async def hello(self):
        return 'hello'

    @router.post('/punc/text')
    async def inference_text(self, input: TextInferenceInput):

        #input.text = input.text.replace('\n', '')
        #result = await self.svc.preprocessing(input.text)
        print(input.text)
        result = await self.svc.generate(input.text)
        print(result)
        return result