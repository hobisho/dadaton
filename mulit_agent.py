# app.py
import os
import re
import base64
from typing import Optional
import requests
from PIL import Image
from io import BytesIO
import io
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import binascii
from  edge_tts import Communicate
from langdetect import detect
from translate import Translator


TEXT_FOLDER = "mydata"
IMAGE_FOLDER = "images"
MODEL_NAME = "llama3.2:1b"
STABLE_DIFFUSION_URL = "http://127.0.0.1:7860"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

app = FastAPI()
app.mount("/assets", StaticFiles(directory="harmonycarebu/dist/assets"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def read_index():
    return FileResponse('harmonycarebu/dist/index.html')

class CustomE5Embedding(Embeddings):
    def __init__(self, model_path=EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.instruction = "query: "

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        formatted_text = self.instruction + text
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings.tolist()

def query_ollama(prompt, model=MODEL_NAME):
    url = f"{OLLAMA_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        return response.json().get("response")
    else:
        print("Ollama 錯誤：", response.text)
        return "本地模型發生錯誤，請檢查是否有啟動。"

def generate_image(prompt):
    payload = {
        "prompt": prompt,
        "steps": 20,
        "cfg_scale": 12,
        "width": 512,
        "height": 512,
    }
    try:
        response = requests.post(f"{STABLE_DIFFUSION_URL}/sdapi/v1/txt2img", json=payload)
        r = response.json()
        if 'images' in r:
            return r['images'][0]
        else:
            return None
    except Exception as e:
        print("圖片生成錯誤：", e)
        return None

def extract_image_prompt(text):
    match = re.search(r"\s*Image\s*Prompt\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    print("match:", match)
    return match.group(1).strip() if match else None

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def is_valid_base64_image(data: str) -> bool:
    try:
        base64.b64decode(data, validate=True)
        return True
    except binascii.Error:
        return False
    
def get_image_caption(base64_str):
    try:
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print("圖像描述錯誤：", e)
        return "無法辨識圖片內容"

# def get_voice(lang_code: str) -> str:
#     voice = {
#         "zh-tw": "zh-TW-XiaoxiaoNeural",
#         "en" : "en-US-JennyNeural",
#         "id" : "id-ID-DianNeural",
#         "vi" : "vi-VN-HaNeural",
#     }
#     lang_code = lang_code.lower()
#     if lang_code.startswith("zh"):
#         return voice["zh-tw"]
#     elif lang_code.startswith("en"):
#         return voice["en"]
#     elif lang_code.startswith("id"):
#         return voice["id"]
#     elif lang_code.startswith("vi"):
#         return voice["vi"]
#     else:
#         return voice["zh-tw"]

# === 多代理設定 ===
AGENT_FILES = {
    "elderly_recipes": {
        "file": "老人食譜_RAG格式_含效果標籤.txt",
        "desc": "Nutritional meal plans for the elderly, dietary restrictions, and cooking recommendations suitable for patients, including explanations of ingredient benefits.",
    },
    "sea_culture": {
        "file": "東南亞文化風俗_RAG格式.txt",
        "desc": "Insights into the culture, taboos, traditions, and religious beliefs of Southeast Asian countries, such as Vietnam and Indonesia.",
    },
    "care_vocab": {
        "file": "東南亞照顧詞彙資料.txt",
        "desc": "Comparison of frequently used care vocabulary in the field, featuring Chinese-Vietnamese and Chinese-Indonesian equivalents to facilitate communication between caregivers and the elderly.",
    },
    "rehab_guide": {
        "file": "復健動作_RAG格式_加強版.txt",
        "desc": "Instructions and guidelines for common rehabilitation exercises, including details and precautions for post-stroke, dementia, and strength training movements.",
    },
    "migrant_guide": {
        "file": "外籍移工須知_RAG格式.txt",
        "desc": "Guidelines and regulations for foreign caregivers working and living in Taiwan, including information on hiring, accommodation, and lifestyle rules.",
    },
    "general": {
        "file": None,
        "desc": "Handles everyday conversations or topics without a specific theme, such as greetings or simple chats.",
    },
}

embedding_function = CustomE5Embedding()
retrievers = {}

def init_retrievers():
    for key, meta in AGENT_FILES.items():
        file = meta["file"]
        if not file:
            continue
        with open(os.path.join(TEXT_FOLDER, file), encoding="utf-8") as f:
            text = f.read()
        chunks = re.split(r"\n{2,}|---+", text)
        docs = [p.strip() for p in chunks if p.strip()]
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = splitter.create_documents(docs)
        vectorstore = Chroma.from_documents(docs, embedding=embedding_function)
        retrievers[key] = vectorstore.as_retriever()

print("📄 載入多代理資料中...")
init_retrievers()
translator= Translator(from_lang="zh", to_lang="en")

class QueryInput(BaseModel):
    query: str
    image: Optional[str] = None

@app.post("/ask")
async def ask(input: QueryInput):
    query = input.query.strip().lower()
    simple_greetings = ["hi", "hello", "你好", "嗨", "哈囉", "hey", "halo"]
        
    mention_image = any(word in query for word in ["圖片", "圖像", "畫", "繪製", "生成圖", "畫面", "圖","照片","image","picture"])
    print("mention_image:",mention_image)
    image_caption = None
    if input.image and is_valid_base64_image(input.image):
        image_caption = get_image_caption(input.image)
        translator_T= Translator(from_lang="en", to_lang="zh")
        print("📷 圖像描述：", image_caption)
        query += f"這張圖片的內容是：{ translator_T.translate(image_caption)}\n\n使用者問題：{query}"
    elif input.image:
        print("⚠️ 圖像資料無效，已忽略")

    if query in simple_greetings:
        return {"answer": "你好，我是 Harmonycare，有需要幫忙照顧長輩或文化溝通的問題都可以問我喔！", "image": None}

    agent_descriptions = "\n".join([f"{key}: {meta['desc']}" for key, meta in AGENT_FILES.items() if key != "general"])
    agent_select_prompt = f"""
        你是一個分類助手，只負責判斷問題屬於哪一個主題領域，並選參考描述選擇一個最適合得key。請務必**只回傳下列 key 名稱中的一個，不可輸出說明、標點、重複描述或多個 key**。

        === 可選 key 與主題描述 ===
        {agent_descriptions}

        有的key名稱:
        "migrant_guide"
        "elderly_recipes"
        "sea_culture"
        "care_vocab"
        "rehab_guide"
        "general"
        
        現在請判斷以下問題要使用哪個agent，然後回傳上述中六個key中最貼近內容的一個key，(只要名稱不要其他描述)：
        使用者問題：{query}
        
        請你一定只輸出上述 key 中的一個，**完全不要輸出任何其他文字、說明、符號或標點符號**，否則會導致錯誤。
        """
    selected_agent_raw = query_ollama(agent_select_prompt)
    print("🤖 選擇的代理：", selected_agent_raw)

    valid_keys = list(AGENT_FILES.keys())
    match = re.search(r'\b(' + '|'.join(valid_keys) + r')\b', selected_agent_raw.lower())
    selected_agent = match.group(1) if match else "general"

    print("🤖 修正後代理 key：", selected_agent)
    context = ""
    if selected_agent != "general":
        docs = retrievers[selected_agent].get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

    
    # 共用部分：背景與介紹
    base_prompt = f"""
    你叫 Harmonycare，是一位專業的老人照護與文化溝通助手，幫助外籍看護與台灣老人更好地溝通與照顧。
    你擁有跨語言、跨文化與長照知識，熟悉各種復健動作、疾病飲食與文化禁忌。
    請依照使用者的需求回答問題，並根據下方資料 {"與圖片內容" if input.image and is_valid_base64_image(input.image) else ""} 來補充說明或提供具體建議。
    以下是可參考的照護與文化資料：
    {context}
    """

    # 如果有有效圖片，則加入圖片描述
    image_section = ""
    if input.image and is_valid_base64_image(input.image):
        image_section = "以下是使用者上傳的圖片內容描述：" + image_caption + "\n\n"

    # 加入使用者問題
    question_section = f"使用者問題：{query}\n\n"

    # 預設回答要求（條列式重點）
    answer_instruction = "請輸出條列式重點，內容要簡潔明確，只需輸出一個最佳答案，避免列出多個版本或思考過程。\n\n"

    # 決定是否需要要求輸出圖片描述：只有當查詢中提及圖片，且(若有 generate_image 參數則為 True)時，才附加圖片描述指示
    image_prompt_instruction = ""
    if mention_image and (not hasattr(input, "generate_image") or input.generate_image):
        image_prompt_instruction = (
            "因為使用者想要生成圖片，所以請以參考知識資料生成一份英文的圖片描述，這份描述是為了讓生圖模型能夠更加了解這個圖片有什麼內容"
            "重要!!!  \n 這份英文描述，請一定要遵循以下格式（僅一次）：\n"
            "Image Prompt :"
            "< \"......\">\n"
            "請不要輸出多組 Image Prompt: 或其他備註。\n"
        )

    # 組合完整 prompt
    full_prompt = base_prompt + image_section + question_section + answer_instruction + image_prompt_instruction

    # 呼叫模型取得回答
    answer = query_ollama(full_prompt)
    print("🧠 模型回答：", answer)

    img_prompt = extract_image_prompt(answer) if mention_image else None
    translator= Translator(from_lang="zh", to_lang="en")
    
    print("img_prompt:", img_prompt)
    print("query:", query)
    
    image_base64 = generate_image(translator.translate(img_prompt)  if img_prompt is not None else  translator.translate(query)) if (img_prompt or mention_image) else None

    # if image_base64:
    #     print("🖼️ 已產生圖片 (base64)")
    # elif mention_image:
    #     print("⚠️ 提及圖片但未成功產生，可能是模型未提供描述")
    # lang = detect(answer)
    # voice = get_voice(lang)
    # print("語音選擇：", voice) 
    # communicate = Communicate(answer, voice=voice)
    # stream = io.BytesIO()
    # async for chunk in communicate.stream():
    #     if chunk["type"] == "audio":
    #         stream.write(chunk["data"])
    # stream.seek(0)
    
    return {"answer": answer, "image": image_base64}
    # return {"answer": answer, "image": image_base64, StreamingResponse: StreamingResponse(stream, media_type="audio/mpeg")}

# @app.post("/tts")
# async def tts(text: str):
#     lang = detect(text)
#     voice = get_voice(lang)
#     print("語音選擇：", voice)
#     communicate = Communicate(text, voice=voice)
#     stream = io.BytesIO()
#     async for chunk in communicate.stream():
#         if chunk["type"] == "audio":
#             stream.write(chunk["data"])
#     stream.seek(0)
#     return StreamingResponse(stream, media_type="audio/mpeg")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
