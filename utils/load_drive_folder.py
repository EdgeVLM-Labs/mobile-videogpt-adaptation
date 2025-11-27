import os
import io
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Only requesting read access to your Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    """Authenticates the user and returns the Drive API service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("❌ Error: 'credentials.json' not found. Please download it from Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def list_all_files(service, folder_id):
    """Lists all files in the folder using pagination."""
    files = []
    page_token = None

    print(f"🔍 Scanning folder ID: {folder_id}...")

    while True:
        # q parameter filters for files inside the specific folder and not in trash
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageSize=100, # API fetches 100 at a time
            pageToken=page_token
        ).execute()

        items = response.get('files', [])
        files.extend(items)

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return files

def download_file(service, file_id, file_name, output_folder):
    """Downloads a single file from Drive."""
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(output_folder, file_name)

    # Create file and start download
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    print(f"⬇️  Downloading: {file_name}")
    try:
        while done is False:
            status, done = downloader.next_chunk()
            # Optional: Print progress for large files
            # print(f"Download {int(status.progress() * 100)}%.")
    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Download all files from a Google Drive folder via API.")
    parser.add_argument("folder_url", help="The Google Drive Folder URL")
    parser.add_argument("--output", default="downloads", help="Output directory")

    args = parser.parse_args()

    # Extract Folder ID from URL
    # URL format: https://drive.google.com/drive/folders/18fnifZ0uh7rySsW6wLeZD2zz2mmIAtcK...
    if "folders/" in args.folder_url:
        folder_id = args.folder_url.split("folders/")[1].split("?")[0]
    else:
        # Assume user passed the ID directly
        folder_id = args.folder_url

    # Setup Output Dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 1. Authenticate
    service = authenticate()
    if not service:
        return

    # 2. List Files (Bypassing the 50 file limit via pagination)
    files = list_all_files(service, folder_id)
    print(f"✅ Found {len(files)} files.")

    # 3. Download Files
    for file in files:
        # skip folders inside the folder (recursive download requires extra logic)
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"⏩ Skipping subfolder: {file['name']}")
            continue

        download_file(service, file['id'], file['name'], args.output)

    print("\n🎉 All downloads complete.")

if __name__ == '__main__':
    main()