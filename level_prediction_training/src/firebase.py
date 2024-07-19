from firebase_admin import initialize_app, storage

app = initialize_app(options={"storageBucket": "durham-river-level.appspot.com"})

bucket = storage.bucket()
