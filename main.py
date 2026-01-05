from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# -------------------
# 数据库配置
# -------------------
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 数据表
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

# 创建表
Base.metadata.create_all(bind=engine)

# -------------------
# FastAPI 配置
# -------------------
app = FastAPI()

# 模板与静态文件配置
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------
# 首页
# -------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    db = SessionLocal()
    # 获取历史记录
    name_history = db.query(NameEntry).order_by(NameEntry.id.desc()).limit(5).all()
    compute_history = db.query(ComputeEntry).order_by(ComputeEntry.id.desc()).limit(5).all()
    db.close()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Python Web Starter",
            "message": "欢迎来到 Day 5 网站，历史记录如下",
            "name_history": name_history,
            "compute_history": compute_history
        }
    )


# -------------------
# 提交名字
# -------------------
@app.post("/submit", response_class=HTMLResponse)
def submit(request: Request, username: str = Form(...)):
    db = SessionLocal()
    entry = NameEntry(username=username)
    db.add(entry)
    db.commit()
    db.close()
    return home(request)


# -------------------
# 数字计算
# -------------------
@app.post("/compute", response_class=HTMLResponse)
def compute(request: Request, number: float = Form(...)):
    # Python 功能：计算平方和立方
    square = number ** 2
    cube = number ** 3

    db = SessionLocal()
    entry = ComputeEntry(number=number, square=square, cube=cube)
    db.add(entry)
    db.commit()
    db.close()

    return home(request)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
