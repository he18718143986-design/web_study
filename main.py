from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

import pandas as pd
import io

# -------------------
# 数据库配置
# -------------------
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class NameEntry(Base):
    __tablename__ = "names"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)

class ComputeEntry(Base):
    __tablename__ = "computes"
    id = Column(Integer, primary_key=True, index=True)
    number = Column(Float)
    square = Column(Float)
    cube = Column(Float)

Base.metadata.create_all(bind=engine)

# -------------------
# FastAPI 配置
# -------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------
# 首页
# -------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, name_page: int = Query(1), compute_page: int = Query(1), msg: str = ""):
    db = SessionLocal()

    # 分页配置
    page_size = 5
    name_history = db.query(NameEntry).order_by(NameEntry.id.desc()).offset((name_page-1)*page_size).limit(page_size).all()
    compute_history = db.query(ComputeEntry).order_by(ComputeEntry.id.desc()).offset((compute_page-1)*page_size).limit(page_size).all()

    # 总条数
    name_count = db.query(NameEntry).count()
    compute_count = db.query(ComputeEntry).count()

    db.close()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Python Web Starter Day 6",
            "msg": msg,
            "name_history": name_history,
            "compute_history": compute_history,
            "name_page": name_page,
            "compute_page": compute_page,
            "name_pages": (name_count + page_size - 1) // page_size,
            "compute_pages": (compute_count + page_size - 1) // page_size
        }
    )


# -------------------
# 提交名字
# -------------------
@app.post("/submit", response_class=HTMLResponse)
def submit(request: Request, username: str = Form(...)):
    if not username.strip():
        return home(request, msg="请输入有效名字！")
    db = SessionLocal()
    entry = NameEntry(username=username.strip())
    db.add(entry)
    db.commit()
    db.close()
    return home(request, msg=f"名字 '{username}' 提交成功！")


# -------------------
# 数字计算
# -------------------
@app.post("/compute", response_class=HTMLResponse)
def compute(request: Request, number: float = Form(...)):
    db = SessionLocal()
    square = number ** 2
    cube = number ** 3
    entry = ComputeEntry(number=number, square=square, cube=cube)
    db.add(entry)
    db.commit()
    db.close()
    return home(request, msg=f"数字 {number} 计算完成！")


# -------------------
# 下载数据
# -------------------
@app.get("/download/names")
def download_names():
    db = SessionLocal()
    data = db.query(NameEntry).all()
    db.close()
    df = pd.DataFrame([{"id": d.id, "username": d.username} for d in data])
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=names.csv"})


@app.get("/download/computes")
def download_computes():
    db = SessionLocal()
    data = db.query(ComputeEntry).all()
    db.close()
    df = pd.DataFrame([{"id": d.id, "number": d.number, "square": d.square, "cube": d.cube} for d in data])
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=computes.csv"})


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
