import asyncio
from fn_src.sewage_spills import collect_sewage_spills
from fn_src.weather_forecast import collect_weather_forecast

from firebase_admin import initialize_app
import logging

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s")
console_handler.setFormatter(formatter)

loggers = ["sewage_spills", "weather_forecast"]
loggers = [logging.getLogger(logger) for logger in loggers]
for logger in loggers:
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)


initialize_app(options={"storageBucket": "durham-river-level.appspot.com"})


async def run_both():
    await asyncio.gather(collect_sewage_spills(), collect_weather_forecast())


def collect_data(_event):
    # asyncio.run(run_both())
    asyncio.run(collect_sewage_spills())


if __name__ == "__main__":
    collect_data(None)
