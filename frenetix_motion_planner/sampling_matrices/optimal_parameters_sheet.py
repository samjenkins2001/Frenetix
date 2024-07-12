import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

def upload_csv_to_google_sheets(csv_file_path, sheet_name):
    # Define the scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('frenetix-959ef12b57ee.json', scope)

    # Authorize the clientsheet
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open(sheet_name)

    # Select the first sheet
    worksheet = sheet.get_worksheet(0)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Clear the existing content in the sheet
    worksheet.clear()

    # Update the sheet with the DataFrame values
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

if __name__ == "__main__":
    csv_file_path = 'dense_optimal.csv'
    sheet_name = 'Dense Optimal Sampling Matrix'
    upload_csv_to_google_sheets(csv_file_path, sheet_name)
