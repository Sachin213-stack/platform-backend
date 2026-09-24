import asyncio
import httpx
from app.main import app


async def test_redis_down():
    print("Testing app lifespan with Redis stopped...")
    async with app.router.lifespan_context(app):
        print("Lifespan entered! Creating AsyncClient...")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            res_root_health = await client.get("/health")
            print("GET /health ->", res_root_health.status_code, res_root_health.json()["status"])
            assert res_root_health.status_code == 200

            res_api_health = await client.get("/api/health")
            print("GET /api/health ->", res_api_health.status_code, res_api_health.json()["status"])
            assert res_api_health.status_code == 200
            
            res_v1_health = await client.get("/api/v1/health")
            print("GET /api/v1/health ->", res_v1_health.status_code, res_v1_health.json()["status"])
            assert res_v1_health.status_code == 200

    print("Lifespan exited cleanly after Redis down test!")


if __name__ == "__main__":
    asyncio.run(test_redis_down())
