from fastapi import FastAPI

from routers.bbox import router

app = FastAPI(
    title="SAM3 Watch Segmentation API",
    description="API for Watch Segmentation using SAM3"
)
#This tells FastAPI When someone sends a POST request to /bbox/predict, execute this function.
app.router.include_router(router)

@app.get("/")
def root():
    return {
        "message": "SAM3 Watch Segmentation API is running!"
    }