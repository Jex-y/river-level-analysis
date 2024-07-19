from pathlib import Path


def get_dotenv():
    with open(Path(__file__).parent / '../.env') as f:
        data = f.read()

    return {
        x[0].strip(): x[1].strip()
        for x in (line.split('=') for line in data.split('\n') if line.strip())
    }
