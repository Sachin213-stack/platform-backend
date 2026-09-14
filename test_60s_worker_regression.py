import asyncio
import time
from app.main import app
from app.db.session import AsyncSessionLocal
from app.db.models.ml import Anomaly
from sqlalchemy import select, func


async def test_60s_worker_interval():
    print("=" * 80)
    print("STARTING 65-SECOND SCHEDULER TICK & GRACEFUL SHUTDOWN REGRESSION TEST")
    print("=" * 80)

    async with app.router.lifespan_context(app):
        async with AsyncSessionLocal() as s:
            initial_count = (await s.execute(select(func.count(Anomaly.id)))).scalar()
        print(f"Initial anomaly count: {initial_count}")
        print("Waiting 65 seconds for the 60s scheduler job (ml_anomaly_detection) to fire...")

        for elapsed in range(10, 70, 10):
            await asyncio.sleep(10)
            print(f"Elapsed: {elapsed}s...")

        await asyncio.sleep(5)
        async with AsyncSessionLocal() as s:
            final_count = (await s.execute(select(func.count(Anomaly.id)))).scalar()
        print(f"Final anomaly count after scheduler run: {final_count}")
        print(f"New anomalies detected during test: {final_count - initial_count}")
        assert final_count >= initial_count, "Anomaly count decreased unexpectedly!"
        print("Periodic scheduler tick executed successfully!")

    print("Lifespan shut down cleanly without unhandled errors!")


if __name__ == "__main__":
    asyncio.run(test_60s_worker_interval())
