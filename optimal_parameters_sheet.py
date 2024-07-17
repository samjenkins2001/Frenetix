import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from config import SCENARIO_NAME

def upload_csv_to_google_sheets(csv_file_path, sheet_name):
    # Define the scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('frenetix_motion_planner/sampling_matrices/frenetix-959ef12b57ee.json', scope)

    # Authorize the clientsheet
    client = gspread.authorize(creds)
    

    spreadsheet = client.create(f'Sampling Matrices/{sheet_name}')

    spreadsheet.share('samjenkins1255@gmail.com', perm_type='user', role='writer')

    worksheet = spreadsheet.get_worksheet(0)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Clear the existing content in the sheet
    worksheet.clear()

    headers = ['initial time', 'optimal time', 'initial longitudinal position', 'initial longitudinal velocity', 'initial longitudinal acceleration', 'optimal longitudinal velocity', 'optimal longitudinal acceleration', 'initial lateral position', 'initial lateral velocity'
               'initial lateral acceleration', 'optimal lateral position', 'optimal lateral velocity', 'optimal lateral acceleration']
    worksheet.append_row(headers)

    # Update the sheet with the DataFrame values
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

if __name__ == "__main__":
    filename = f'{SCENARIO_NAME}_1'
    csv_file_path = f'frenetix_motion_planner/sampling_matrices/dense/{filename}'
    sheet_name = f'{filename}'
    upload_csv_to_google_sheets(csv_file_path, sheet_name)
