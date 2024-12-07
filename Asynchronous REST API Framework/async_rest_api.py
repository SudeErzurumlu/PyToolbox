from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str
    price: float

DATABASE_URL = "postgresql://user:password@localhost:5432/mydatabase"

async def init_db():
    """
    Initializes the database connection.
    """
    app.state.pool = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("startup")
async def startup():
    await init_db()

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()

@app.post("/items/")
async def create_item(item: Item):
    """
    Creates a new item in the database.
    """
    async with app.state.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO items (name, description, price) VALUES ($1, $2, $3)",
            item.name, item.description, item.price
        )
    return {"message": "Item created successfully"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """
    Retrieves an item by ID.
    """
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        return dict(row)

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """
    Deletes an item by ID.
    """
    async with app.state.pool.acquire() as conn:
        result = await conn.execute("DELETE FROM items WHERE id = $1", item_id)
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Item not found")
        return {"message": "Item deleted successfully"}
