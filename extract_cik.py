import sys
import requests
import zipfile
import json
from io import BytesIO

def get_cik_data(cik_number):
    # URL of the ZIP file
    url = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
    
    # SEC requires a proper User-Agent header
    headers = {
        'User-Agent': 'Kerry Back, kerryback@gmail.com',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    
    # Construct the target filename
    target_file = f"CIK{cik_number}.json"
    
    try:
        # Download the ZIP file
        print(f"Downloading ZIP file from {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Create a BytesIO object from the downloaded content
        zip_data = BytesIO(response.content)
        
        # Open the ZIP file in memory
        print("Reading file from ZIP archive...")
        with zipfile.ZipFile(zip_data) as zip_ref:
            # Check if the target file exists in the ZIP
            if target_file in zip_ref.namelist():
                # Read the JSON data directly from the ZIP
                with zip_ref.open(target_file) as json_file:
                    data = json.load(json_file)
                    print(f"Successfully loaded {target_file}")
                    return data
            else:
                print(f"Error: {target_file} not found in ZIP file")
                sys.exit(1)
                
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Invalid ZIP file")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def main():
    # Check if CIK number was provided
    if len(sys.argv) != 2:
        print("Usage: python extract_cik.py <10-digit-CIK-number>")
        sys.exit(1)
    
    # Get CIK number from command line argument
    cik_number = sys.argv[1]
    
    # Validate CIK number
    if not (cik_number.isdigit() and len(cik_number) == 10):
        print("Error: Please provide a 10-digit CIK number")
        sys.exit(1)
    
    # Get and process the data
    data = get_cik_data(cik_number)
    
    # Here you can work with the data directly
    # For example, print it or process it further
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main() 