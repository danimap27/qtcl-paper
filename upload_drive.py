"""Sube el proyecto QTCL a Google Drive en una carpeta nueva."""
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json, pathlib, mimetypes

TOKEN = pathlib.Path('/home/quantum-nas/jarvi-bot/google_token.json')
BASE  = pathlib.Path('/home/quantum-nas/qtcl-paper')

data  = json.load(open(TOKEN))
creds = Credentials(
    token=data.get('token'), refresh_token=data.get('refresh_token'),
    token_uri=data.get('token_uri'), client_id=data.get('client_id'),
    client_secret=data.get('client_secret'),
)
svc = build('drive', 'v3', credentials=creds)

def make_folder(name, parent_id=None):
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        meta['parents'] = [parent_id]
    f = svc.files().create(body=meta, fields='id').execute()
    return f['id']

def upload_file(path: pathlib.Path, parent_id: str):
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or 'application/octet-stream'
    media = MediaFileUpload(str(path), mimetype=mime, resumable=False)
    meta  = {'name': path.name, 'parents': [parent_id]}
    f = svc.files().create(body=meta, media_body=media, fields='id,name').execute()
    print(f"  ✓ {path.relative_to(BASE)} → Drive/{f['name']}")
    return f['id']

print("Creando carpeta QTCL-Paper en Drive...")
root_id = make_folder('QTCL-Paper')

# Subcarpetas
code_id    = make_folder('code',    root_id)
figures_id = make_folder('figures', root_id)
paper_id   = make_folder('paper',   root_id)

# Archivos raíz
for f in ['README.md', 'results.csv']:
    p = BASE / f
    if p.exists():
        upload_file(p, root_id)

# Código
for f in (BASE / 'code').iterdir():
    if f.is_file():
        upload_file(f, code_id)

# Figuras (PDF + PNG)
for f in sorted((BASE / 'figures').iterdir()):
    if f.is_file():
        upload_file(f, figures_id)

# Paper
for f in (BASE / 'paper').iterdir():
    if f.is_file() and f.suffix in ('.tex', '.pdf', '.bib'):
        upload_file(f, paper_id)

# Obtener enlace compartido
svc.permissions().create(fileId=root_id,
    body={'role': 'reader', 'type': 'anyone'}).execute()
link = f"https://drive.google.com/drive/folders/{root_id}"
print(f"\n✅ Subida completada.")
print(f"🔗 Enlace: {link}")
