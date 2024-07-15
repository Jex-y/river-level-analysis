from pathlib import Path

from firebase_admin import credentials, storage, initialize_app

cred = credentials.Certificate(
    Path(__file__).parent / '../../firebase-service-account-key.json'
)
app = initialize_app(cred, {'storageBucket': 'durham-river-level.appspot.com'})

bucket = storage.bucket()
