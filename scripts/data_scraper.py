import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

def fetch_world_bank_data(indicator="NY.GDP.MKTP.CD", countries=["US", "GB", "CN"], per_page=100):
 
    #Fetches economic data (e.g., GDP) from the World Bank API.
  
    base_url = "http://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page={}"
    url = base_url.format(",".join(countries), indicator, per_page)

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        # Ensure valid response format
        if len(data) > 1 and isinstance(data[1], list):
            records = []
            for entry in data[1]:
                records.append({
                    "Country": entry["country"]["value"],
                    "Year": int(entry["date"]),
                    "GDP": entry["value"]  # GDP can be None if data is missing
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Drop rows where GDP is missing
            df.dropna(subset=["GDP"], inplace=True)
            
            return df
        else:
            print("⚠️ No valid data found.")
            return None
    else:
        print(f"❌ Failed to fetch data. HTTP Status Code: {response.status_code}")
        return None
    
# ------------------------------
# 2️⃣ Fetch Technological Data (IEA Reports & Patents)
# ------------------------------
def fetch_iea_data():
    # Scrapes IEA website for energy technology trends 
    url = "https://www.iea.org/reports"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    reports = []
    for report in soup.find_all("h3", class_="title"):
        reports.append(report.text.strip())

    return reports

# ------------------------------
# 3️⃣ Fetch Political Risk Data (Geopolitical Risk Index & News)
# ------------------------------
def fetch_geopolitical_risk():
    # Fetches Geopolitical Risk Index data from Harvard 
    url = "https://www.matteoiacoviello.com/gpr.htm"  # Harvard GPR page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract recent risk summary
    risk_summary = soup.find("p").text.strip()
    return risk_summary

# ------------------------------
# 4️⃣ Fetch Brent Oil Prices (Yahoo Finance)
# ------------------------------
def fetch_brent_prices():
    # Fetches Brent crude oil prices from Yahoo Finance 
    brent_ticker = "BOP"
    brent_data = yf.download(brent_ticker, start="1987-05-20", end="2022-11-14", interval="1d")
    brent_data.to_csv("brent_oil_prices.csv")
    return brent_data

import datetime

# ------------------ 1. Fetch Brent Oil Prices from Yahoo Finance ------------------
def fetch_brent_oil_prices(start_date="1987-05-20", end_date="2022-11-14"):
    brent_ticker = "BOP"
    brent_data = yf.download(brent_ticker, start=start_date, end=end_date, interval="1d")
    brent_data.ffill(inplace=True)  # Fill missing values
    brent_data.to_csv("brent_oil_prices.csv")
    print("✅ Brent oil prices saved to brent_oil_prices.csv")
    return brent_data

# ------------------ 2. Fetch Economic Indicators from World Bank API ------------------
def fetch_world_bank_data(indicator="NY.GDP.MKTP.CD", countries=["US", "GB", "CN"], start_year=1987, end_year=2022):
    url = f"http://api.worldbank.org/v2/country/{','.join(countries)}/indicator/{indicator}?format=json&per_page=100"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1 and isinstance(data[1], list):
            records = []
            for entry in data[1]:
                year = int(entry["date"])
                if start_year <= year <= end_year:
                    records.append({
                        "Country": entry["country"]["value"],
                        "Year": entry["date"],
                        "Value": entry["value"]
                    })

            df = pd.DataFrame(records)
            filename = f"world_bank_{indicator}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ World Bank data ({indicator}) saved to {filename}")
            return df
    print(f"❌ Failed to fetch World Bank data ({indicator})")
    return None

# ------------------ 3. Fetch IMF Economic Indicators ------------------
def fetch_imf_data():
    imf_url = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/Q..PCPIPCH"
    response = requests.get(imf_url)

    if response.status_code == 200:
        imf_data = response.json()
        print("✅ IMF data fetched successfully")
        return imf_data  # Processing can be added
    else:
        print("❌ Failed to fetch IMF data")
        return None

# ------------------ 4. Fetch Geopolitical Risk Data ------------------
def fetch_geopolitical_risk_index():
    # Sample URL for geopolitical risk index (Replace with a real API if available)
    geopolitics_url = "https://www.mindsdb.com/dataset/geopolitical-risk-index.json"

    response = requests.get(geopolitics_url)
    if response.status_code == 200:
        risk_data = response.json()
        df = pd.DataFrame(risk_data)  # Assume JSON structure supports conversion
        df.to_csv("geopolitical_risk_index.csv", index=False)
        print("✅ Geopolitical Risk Index saved to geopolitical_risk_index.csv")
        return df
    else:
        print("❌ Failed to fetch Geopolitical Risk Index")
        return None

# ------------------ 5. Fetch Technological Reports from IEA ------------------
def fetch_iea_reports():
    iea_url = "https://www.iea.org/api/reports"  # Placeholder URL (Replace with real API)
    response = requests.get(iea_url)

    if response.status_code == 200:
        reports_data = response.json()
        df = pd.DataFrame(reports_data)  # Assume JSON structure supports conversion
        df.to_csv("iea_reports.csv", index=False)
        print("✅ IEA Reports saved to iea_technology_reports.csv")
        return df
    else:
        print("❌ Failed to fetch IEA reports")
        return None

