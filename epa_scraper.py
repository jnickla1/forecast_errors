import pandas as pd
import requests

class epa_fetch:
    def __init__(self, email: str, api_key: str):
        """
        Initialize the analyzer with EPA API credentials
        
        Args:
            email: Email for EPA API access
            api_key: API key for EPA API access
        """
        self.email = email
        self.api_key = api_key
        
        # EPA monitoring stations in Providence
        self.stations = {
##            '440070030': {'name': 'Near Road',
##                'location': 'Corner of Park/Hayes Streets',
##                'lat': 41.829523,
##                'lon': -71.417584},
            '440070022': {
                'name': 'CCRI Liston Campus',
                'location': '1 Hilton Street',
                'lat': 41.807523,
                'lon': -71.413920
            }
        }
        self.stat_suffixes = {k[-4:] for k in self.stations.keys()}


    def fetch_epa_observations(self, start_date: str, end_date: str, 
                              param: str = '62101') -> pd.DataFrame:
        """
        Fetch EPA AQS observations for temperature
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            param: EPA parameter code (62101 for temperature, 62201 for humidity)
            
        Returns:
            DataFrame with hourly observations
        """
        print(f"Fetching EPA observations from {start_date} to {end_date}...")
        
        url = "https://aqs.epa.gov/data/api/sampleData/byState"
        
        all_data = []
        
        # Fetch data for each station
        for station_id in self.stations.keys():
            params = {
                'email': self.email,
                'key': self.api_key,
                'param': param,
                'bdate': start_date,
                'edate': end_date,
                'state': '44',  # Rhode Island
             #   'site_number': station_id[-4:]  # Last 4 digits of station ID
            }
            response = requests.get(url, params=params, timeout=(5, 30),
                    proxies={"http": None, "https": None}, ) # ignore env proxies)

            if response.status_code == 200:
                data = response.json()
                if 'Data' in data and data['Data']:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['Data'])
                    
                    # Extract only the columns we need
                    df_clean = df[['date_gmt', 'time_gmt', 'sample_measurement', 
                                  'site_number', 'county_code']].copy()
                    df_clean = df_clean[df_clean['site_number'].isin(self.stat_suffixes)]
                    
                    # Create station ID to match your format (e.g., 440070022)
                    df_clean['station_id'] = '44' + df_clean['county_code'] + df_clean['site_number']
                    
                    # Add station names
                    #df_clean['station_name'] = df_clean['station_id'].map(
                    #    {sid: info['name'] for sid, info in self.stations.items()})
                    
                    # Drop unnecessary columns
                    df_clean = df_clean[['date_gmt', 'time_gmt', 'sample_measurement', 
                                        'station_id']]
                    # Rename for clarity
                    #df_clean.rename(columns={'sample_measurement': 'temperature'}, inplace=True)
                    
                return df_clean
            else:
                print(f"Warning: Failed to fetch data for all of RI")
