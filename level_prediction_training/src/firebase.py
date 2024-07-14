from pathlib import Path

import firebase_admin
from firebase_admin import credentials, storage, 

cred = credentials.Certificate(
    Path(__file__).parent / '../../firebase-service-account-key.json'
)
app = firebase_admin.initialize_app(cred, {'storageBucket': 'durham-river-level.appspot.com'})

bucket = storage.bucket()


