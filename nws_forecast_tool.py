"""
NWS Forecast Accuracy Analysis Tool for Providence, RI
Analyzes forecast accuracy and bias for daily maximum temperatures and heat indices
"""
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import pdb
from datetime import datetime, timedelta
import pytz
warnings.filterwarnings('ignore')
import os, re, json, shutil
from pathlib import Path





class NWSForecastAnalyzer:
    """Main class for analyzing NWS forecast accuracy against EPA observations"""
    
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

        self.DATE_FMT = "%Y-%m-%d"
        self.CACHE_DIR = Path()
        print(self.CACHE_DIR)
        self.FORECAST_RE = re.compile(r"forecasts_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.npz$")
        self.OBS_RE = re.compile(r"temp_obs_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
        self.OBS_HU = re.compile(r"humid_obs_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
        
    def fetch_nws_forecasts(self, req_date: str, time_of_day:int =7) -> pd.DataFrame:
        """
        Fetch NWS forecast bulletins from Iowa State archive
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with parsed forecast data, and the list with max(today), min(overnight), max, min, max
        """
        utc_off =4 #4 or 5 hrs
        stime=(time_of_day+utc_off-1)
        etime=(time_of_day+utc_off+2)
        start_date = req_date + f"T{stime:02d}:00Z"
        end_date = req_date + f"T{etime:02d}:00Z"
        
        print(f"Fetching NWS forecasts from {start_date} to {end_date}...")
        
        url = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
        params = {
            'limit': 9999,
            'pil': 'PFMBOX',
            'fmt': 'text',
            'sdate': start_date,
            'edate': end_date,
            'dl': 0
        }
        
        all_ms = False
        response = requests.get(url, params=params)
        if response.status_code != 200 or response.text.startswith("ERROR"):
            end_date = req_date + f"T{(etime+6):02d}:00Z" #try 6 hrs later
            params2 = {'limit': 9999,'pil': 'PFMBOX','fmt': 'text',
                'sdate': start_date,'edate': end_date,'dl': 0}
            response = requests.get(url, params=params2)
        else:
            #all MM's - no idea what this error is, need to check for it
            lines = response.text.split('\n')
            all_ms = lines[21].endswith(" MM"*16)
        if all_ms: #try the next report
            start_date = req_date + f"T{(stime+3):02d}:00Z"
            end_date = req_date + f"T{(etime+3):02d}:00Z" #try 6 hrs later
            params2 = {'limit': 9999,'pil': 'PFMBOX','fmt': 'text',
                'sdate': start_date,'edate': end_date,'dl': 0}
            response = requests.get(url, params=params2)
        #didn't work on the second try
        if response.status_code != 200 or response.text.startswith("ERROR"):
            print("NO FORECAST FOUND")
            lhblank = (np.arange(1,24,3).tolist()*4)[3:25]
            utcblank = (np.arange(0,24,3).tolist()*4)[3:25]
            forecast_blank = pd.DataFrame({'local_hour': lhblank,'utc_hour': utcblank,'temperature': [np.nan] * 22,'dewpoint': [np.nan] * 22,
                                    'rel_humidity': [np.nan] * 22,'wind_speed': [np.nan] * 22})
            return "",forecast_blank,[np.nan]*5
        
        # Parse the text bulletins
        return self._parse_forecast_bulletins(response.text)
    
    def _parse_forecast_bulletins(self, text: str) -> List[Dict]:
        """
        Parse NWS forecast bulletins to extract temperature predictions
        
        Args:
            text: Raw bulletin text
            
        Returns:
            List of parsed forecast records
            
        """

        def parse_fixed_ints(line: str, width: int = 3, starts: int =13) -> list[int]:
            chunks = [line[i:i+width] for i in range(starts, len(line), width)]
            retlist=[]
            for c in chunks:
                if c.strip():
                    try:
                        retlist.append(int(c.strip()))
                    except:
                        print(line)
                        retlist.append(np.nan)
                else:
                    retlist.append(np.nan)
            return retlist  #[ int(c.strip())if c.strip() else np.nan for c in chunks ]

        bulletins = text.split('\n\n')
        nexti=0
        date_match =""
        local_times = [] ;utc_times = [] ; temp=[]; dewpt=[]; relh=[]; tmaxmin=[];wind=[]
        for bulletin in bulletins:
            if ('Foster-Providence RI' not in bulletin) and nexti!=1 and nexti!=2 :
                continue
            
            # Extract date from bulletin
            if nexti==0:
                date_match = re.search(r'(\d{3})\s+(AM|PM)\s+E(S|D)T\s+\w{3}\s+(\w{3})\s+(\d{1,2})\s+(\d{4})', bulletin)
            
            # Parse forecast temperatures
            # Look for patterns like "HIGH TEMPERATURE... 85 TO 90"
            if nexti==1:
                lines=bulletin.split('\n')
                local_times = [int(x) for x in lines[1].split()[2:]]
                utc_times = [int(x) for x in lines[2].split()[2:]]  # Skip "UTC" and "3hrly"

            if nexti==2:
                lines=bulletin.split('\n')
                temp =parse_fixed_ints(lines[1], width=3)
                dewpt=parse_fixed_ints(lines[2], width=3)
                relh =parse_fixed_ints(lines[3], width=3)
                tmaxmin = [int(x) for x in lines[0].split()[1:]]
                wind = parse_fixed_ints(lines[5], width=3)
            #if len(utc_times) > len(dewpt):
            #    utc_times = utc_times[1:]
            nexti=nexti+1 
        forecast = pd.DataFrame({'local_hour': local_times,'utc_hour': utc_times,'temperature': temp,'dewpoint': dewpt,
                                    'rel_humidity': relh,'wind_speed': wind})
        return date_match,forecast,tmaxmin
    
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
        

    
    def calculate_heat_index(self, temp_f: float, rh: float) -> float:
        """
        Calculate heat index using NWS formula
        
        Args:
            temp_f: Temperature in Fahrenheit
            rh: Relative humidity (0-100)
            
        Returns:
            Heat index in Fahrenheit
        """
        if temp_f < 80:
            return temp_f
            
        # Rothfusz regression
        hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh 
              - 0.22475541 * temp_f * rh - 0.00683783 * temp_f * temp_f 
              - 0.05481717 * rh * rh + 0.00122874 * temp_f * temp_f * rh 
              + 0.00085282 * temp_f * rh * rh - 0.00000199 * temp_f * temp_f * rh * rh)
        
        # Adjustments
        if rh < 13 and 80 <= temp_f <= 112:
            adjustment = ((13 - rh) / 4) * np.sqrt((17 - abs(temp_f - 95)) / 17)
            hi -= adjustment
        elif rh > 85 and 80 <= temp_f <= 87:
            adjustment = ((rh - 85) / 10) * ((87 - temp_f) / 5)
            hi += adjustment
            
        return hi
    
    def process_daily_maxima(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily maximum temperatures from hourly observations
        
        Args:
            obs_df: DataFrame with hourly observations
            
        Returns:
            DataFrame with daily maximum values
        """
        if obs_df.empty:
            return pd.DataFrame()
        
        # Convert date columns to datetime
        obs_df['date'] = pd.to_datetime(obs_df['date_local'] if 'date_local' in obs_df.columns else obs_df['date'])
        
        # Group by date and station, get daily max
        daily_max = obs_df.groupby(['date', 'station_id']).agg({
            'sample_measurement': 'max'
        }).reset_index()
        
        daily_max.columns = ['date', 'station_id', 'max_temp']
        
        return daily_max


    def hi_max_from_full(self, all_full_forecasts):
        IDX_HOUR, IDX_TEMP, IDX_RH = 0, 2, 4

        # Flatten to ordered rows: (hour, temp, rh)
        hi_daily_max=[]
        
        for day_block in all_full_forecasts:
            rows = [];
            for rec in day_block:
                hr = float(rec[IDX_HOUR])
                t  = float(rec[IDX_TEMP])
                rh = float(rec[IDX_RH])
                rows.append((hr, t, rh))

            # Split into 3 day-groups by detecting hour wrap (e.g., 22 -> 01)
            hi_by_day = [[], [], []]

            day = 0
            prev_hr = None
            
            for hr, t, rh in rows:
                if prev_hr is not None and hr < prev_hr:
                    day += 1
                    if day > 2: break  # keep only 3 days
                prev_hr = hr

                hi = self.calculate_heat_index(t, rh)
                hi_by_day[day].append(hi)
                #rh_by_day[day].append(rh)

            # Daily max RH (3 lists or a single list of 3 values—pick what you need)
            #rh_daily_max_lists = [[max(d)] if d else [np.nan] for d in rh_by_day]  # three 1-element lists
            # Or as plain values:
            
            hi_daily_max.append([np.nanmax(d) for d in hi_by_day])
        return hi_daily_max



    
    def calculate_metrics(self, forecast_df: pd.DataFrame, obs_df: pd.DataFrame, mt_threshold: float) -> pd.DataFrame:
        """
        Calculate accuracy metrics comparing forecasts to observations
        
        Args:
            forecast_df: DataFrame with forecast data
            obs_df: DataFrame with observation data
            
        Returns:
            DataFrame with metrics
        """
        import matplotlib.pyplot as plt
        # Merge forecasts with observations to ensure alignment
        merged = pd.merge(
            forecast_df,
            obs_df,
            left_on='bulletin_date',
            right_on='date',
            how='inner'
        ).reset_index(drop=True)
        merged['bulletin_date'] = pd.to_datetime(merged['bulletin_date'])
        # --- compute metrics + collect plotting data ---
        rows = []
        plot_data = {}  # h -> dict with 'date','f','o','err'
        for h, col in enumerate(['max0', 'max1', 'max2']):
            pair0 = pd.DataFrame({
                'date_plot': merged['bulletin_date'] + pd.to_timedelta(h, unit='D'),
                'forecast': merged[col],
                'obs': merged['max_temp'].shift(-h)
            }).dropna()

            warm = pair0['obs'] >= mt_threshold
            pair = pair0.loc[warm]
            if pair.empty:
                rows.append({"lead_days": h, "rmse": np.nan, "mae": np.nan,
                             "bias(Obs-F)": np.nan, "rmseNB": np.nan,
                             "n_samples": 0, "correlation": np.nan})
                plot_data[h] = {"date": [], "f": [], "o": [], "err": []}
                continue

            errors = pair['obs'] - pair['forecast']
            bias = float(np.nanmean(errors))
            corr = np.corrcoef(pair['forecast'], pair['obs'])[0, 1] if len(pair) > 1 else np.nan

            rows.append({
                "lead_days": h,
                "rmse": float(np.sqrt(np.nanmean(errors**2))),
                "mae": float(np.nanmean(np.abs(errors))),
                "bias(Obs-F)": bias,
                "rmseNB": float(np.sqrt(np.nanmean((errors - bias)**2))),
                "n_samples": int(len(pair)),
                "correlation": float(corr),
            })

            plot_data[h] = {
                "date": pair['date_plot'].to_list(),
                "f": pair['forecast'].to_numpy(),
                "o": pair['obs'].to_numpy(),
                "err": (-pair['forecast'].to_numpy() + pair['obs'].to_numpy()),  # F - Obs for plotting
            }

        metrics_df = pd.DataFrame(rows)

        # --- plotting (two axes; points only; side PDFs on bottom axis) ---
        if any(len(d["date"]) for d in plot_data.values()):
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
            )

            # x-range (pad to the right so PDFs fit)
            all_dates = pd.to_datetime(
                sum([d["date"] for d in plot_data.values()], [])
            )
            if len(all_dates):
                x_min = all_dates.min()
                x_max = all_dates.max() + pd.Timedelta(days=10)
                ax1.set_xlim(x_min, x_max)

            # Top: obs in black, forecasts in color (points)
            # Plot obs points only where any horizon had that obs (use union of dates)
            # We already have o per-horizon on aligned dates (date_plot)
            seen_obs = {}
            labelsf=["Same-day forecast", "Tomorrow's forecast", "2-day forecast"]
            for h in range(3):
                d = plot_data[h]
                if len(d["date"]):
                    # Forecast points
                    ax1.plot(pd.to_datetime(d["date"]) + pd.Timedelta(days=0.2*h), d["f"], marker='o', linestyle='None', label=labelsf[h], color=colors[h])
                    # Collect obs by date (avoid re-plotting duplicates)
                    for dt, o in zip(d["date"], d["o"]):
                        seen_obs.setdefault(dt, o)
            if seen_obs:
                ax1.plot(list(seen_obs.keys()), list(seen_obs.values()),
                         marker='x', linestyle='None', color='black', label='Observed (CCRI Liston)')

            ax1.set_ylabel("Heat Index °F")
            ax1.legend(loc='best', fontsize=9)

            # Bottom: differences (F-Obs) over time
            for h in range(3):
                d = plot_data[h]
                if len(d["date"]):
                    ax2.plot(pd.to_datetime(d["date"])+ pd.Timedelta(days=0.2*h), d["err"], marker='o', linestyle='None', color=colors[h], label=f'diff max{h}')

            ax2.axhline(0.0, color='0.6', linewidth=1)
            ax2.set_ylabel("Error in Heat Index Δ°F")
            ax2.set_xlabel("Date")
            ax2.set_title("Difference to Observed Temperatures")

            # Side PDFs (smoothed histograms) drawn along the right side of ax2
            # Use simple Gaussian-smoothed histogram
            from scipy.stats import gaussian_kde
            
##            def smooth_pdf(vals, bins=25, sigma=1.5):
##                vals = np.asarray(vals, float)
##                vals = vals[np.isfinite(vals)]
##                if vals.size == 0:
##                    return np.array([]), np.array([])
##                hist, edges = np.histogram(vals, bins=bins, density=True)
##                centers = 0.5 * (edges[:-1] + edges[1:])
##                # Gaussian smoothing
##                from math import exp
##                ksz = max(3, int(3*sigma))
##                kernel = np.array([exp(-0.5*(i/sigma)**2) for i in range(-ksz, ksz+1)])
##                kernel /= kernel.sum()
##                smooth = np.convolve(hist, kernel, mode='same')
##                return centers, smooth

            # Anchor PDFs at the right; scale to small width
            if len(all_dates):
                x_anchor = x_max - pd.Timedelta(days=6)
                x_left   = x_max - pd.Timedelta(days=12.0)  # width used to scale densities
                # plot each density as a curve from x_left..x_anchor
                for h in range(3):
                    e = plot_data[h]["err"]
                    if len(e):
                        #y, dens = smooth_pdf(e)
                        kde = gaussian_kde(e)
                        y_range = np.linspace(min(e)-1, max(e)+1, 100)
                        
                        #if len(dens):
                        dens=kde(y_range)
                        dens = dens *10 #/ (dens.max() or 1.0)  # normalize 0..1
                            #xs = pd.Series(dens).apply(lambda z: x_left + (x_anchor - x_left)*z)
                        xs= x_left + (x_anchor - x_left)*dens
                        ax2.plot(xs+ pd.Timedelta(days=2*h), y_range, color=colors[h], linewidth=2.5)
                        ax2.plot([x_left + pd.Timedelta(days=2*h),x_anchor + pd.Timedelta(days=2*h)],
                                 [np.nanmean(e)]*2, color=colors[h], linewidth=2.5)

            # Title with year(s)
            years = sorted({d.year for d in pd.to_datetime(all_dates)}) if len(all_dates) else []
            yr_txt = years[0] if len(years) == 1 else (f"{years[0]}–{years[-1]}" if years else "")
            fig.suptitle(f"Warm-day RI Forecast Performance ({yr_txt})", fontsize=12)

            fig.tight_layout()
            plt.savefig(f"warm_forecast{yr_txt}.png",dpi=500,bbox_inches='tight')

        return metrics_df
    
    from typing import Dict, Tuple, Callable, List
    def identify_heat_events(self,
        forecast_df: pd.DataFrame,
        obs_df: pd.DataFrame, biases: pd.DataFrame,
        metric: str = "advisory",
        *,
        # thresholds (°F)
        warning_threshold: float = 105.0,
        advisory_low_range: Tuple[float, float] = (95.0, 100),
        advisory_high_range: Tuple[float, float] = (100.0, 105.0),
        heatwave_threshold: float = 90.0,
    ) -> pd.DataFrame:
        """
        For each forecast horizon (max0, max1, max2), align observations by horizon,
        create event masks per `metric`, count forecasted events (runs) and the %
        that match any observed event, and add an 'observed' summary row.

        Returns a tidy DataFrame with columns:
          ['forecast', 'lead_days', 'n_forecast_events', 'pct_matched', 'n_observed_events']
        """

        # column names
        forecast_cols= ("max0", "max1", "max2")
        obs_col= "max_temp"
        merge_left_on = "bulletin_date"
        merge_right_on= "date"
        # ---- helpers -------------------------------------------------------------

        def _runs_from_mask(mask: pd.Series) -> List[Tuple[int, int]]:
            """Return list of (start_idx, end_idx) inclusive runs where mask is True."""
            idx = mask.index.to_numpy()
            vals = mask.to_numpy().astype(bool)
            runs = []
            if len(vals) == 0:
                return runs
            in_run = False
            start = None
            for i, v in enumerate(vals):
                if v and not in_run:
                    in_run = True
                    start = i
                elif not v and in_run:
                    in_run = False
                    runs.append((start, i - 1))
            if in_run:
                runs.append((start, len(vals) - 1))
            # map back to positional indices (0..n-1); we operate positionally
            return runs

        def _mark_runs(mask: pd.Series, min_len: int) -> pd.Series:
            """Given a boolean mask, mark all days that belong to runs with length >= min_len."""
            out = pd.Series(False, index=mask.index)
            for s, e in _runs_from_mask(mask):
                if (e - s + 1) >= min_len:
                    out.iloc[s:e + 1] = True
            return out

        def _warning_mask(series: pd.Series) -> pd.Series:
            # With daily data, approximate “≥105°F for 2 consecutive hours” as daily max ≥ threshold.
            return (series >= warning_threshold)

        def _advisory_mask(series: pd.Series) -> pd.Series:
            lo, hi = advisory_low_range  # inclusive
            lo_mask = (series >= lo) & (series < hi)
            lo_marked = _mark_runs(lo_mask, min_len=2)  # needs 2 consecutive days
            a_lo, a_hi = advisory_high_range
            hi_mask = (series >= a_lo) & (series < a_hi)  # 1-day condition
            return lo_marked | hi_mask

        def _heatwave_mask(series: pd.Series) -> pd.Series:
            base = (series >= heatwave_threshold)
            return _mark_runs(base, min_len=3)  # mark all days within 3+ day runs

        metric_map: Dict[str, Callable[[pd.Series], pd.Series]] = {
            "warning": _warning_mask,
            "advisory": _advisory_mask,
            "heatwave": _heatwave_mask,
        }
        if metric.lower() not in metric_map:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(metric_map)}")

        mask_fn = metric_map[metric.lower()]

        # ---- merge forecasts with observations -----------------------------------

        merged = pd.merge(
            forecast_df.copy(),
            obs_df.copy(),
            left_on=merge_left_on,
            right_on=merge_right_on,
            how="inner",
            validate="one_to_one",
        ).reset_index(drop=True)

        # Ensure numeric
        for c in list(forecast_cols) + [obs_col]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

        # ---- compute per-horizon stats with proper lag alignment -----------------

        rows = []

        # Build observed mask (no shift), but we’ll also need horizon-shifted versions
        obs_series_full = merged[obs_col]

        for h, fcol in enumerate(forecast_cols):
            # Align: compare forecast at t with obs at t+h
            f = merged[fcol]
            o = obs_series_full.shift(-h)
            bias = biases[h]

            pair = pd.DataFrame({"f": f, "o": o}).dropna()
            if pair.empty:
                rows.append({
                    "forecast": fcol, "n_observed_days": 0,
                    "n_forecast_days": 0, "n_matched_fore":0,
                    "n_foreCorr_days": 0, "n_matched_forC":0,  
                })
                continue

            fmask = mask_fn(pair["f"])
            fmaskC = mask_fn(pair["f"]+bias)
            omask = mask_fn(pair["o"])

            # Count forecast days (runs where fmask True)
            n_f = np.sum(fmask)
            n_fC = np.sum(fmaskC)

            # Count observed days (runs where omask True) within the aligned window
            n_o = np.sum(omask)

            # A forecast day is "matched" if both days are True
            matched = fmask & omask
            overlap = np.sum(matched)
            matchedC = fmaskC & omask
            overlapC = np.sum(matchedC)

            rows.append({
                "lead_days": h, "n_observed_days": int(n_o),
                "n_forecast_days": int(n_f),"n_matched_fore": int(overlap),
                "n_foreCorr_days": int(n_fC),"n_matched_forC": int(overlapC),
            })

        return pd.DataFrame(rows)


            # ----------------------------- Cache helpers ---------------------------------


    def _ensure_dirs(self):
        FORECAST_DIR = self.CACHE_DIR / "forecasts"
        OBS_DIR = self.CACHE_DIR / "obs"
        ARCHIVE_DIR = self.CACHE_DIR / "old"
        for d in (FORECAST_DIR, OBS_DIR, ARCHIVE_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def _span_key(self, start_dt: datetime, end_dt: datetime) -> str:
        return f"{start_dt.strftime(self.DATE_FMT)}_{end_dt.strftime(self.DATE_FMT)}"

    def _list_span_files(self, dirpath: Path, regex: re.Pattern) -> list[tuple[Path, datetime, datetime]]:
        out = []
        for p in dirpath.glob("*"):
            m = regex.match(p.name)
            if not m:
                continue
            s, e = datetime.strptime(m.group(1), self.DATE_FMT), datetime.strptime(m.group(2), self.DATE_FMT)
            out.append((p, s, e))
        return out

    def _cover_status(self,start_dt: datetime, end_dt: datetime, spans: list[tuple[Path, datetime, datetime]]):
        """Return (covering_files, overlapping_files, missing_ranges) for requested [start,end]."""
        # normalize to inclusive daily grid
        # 1) collect any file that fully covers [start,end]
        covering = [p for (p, s, e) in spans if s <= start_dt and e >= end_dt]
        if covering:
            return covering, [], []

        # 2) overlapping parts
        overlapping = [(p, s, e) for (p, s, e) in spans if not (e < start_dt or s > end_dt)]
        # compute missing set on daily basis
        need_days = pd.date_range(start_dt, end_dt, freq="D")
        have = set()
        for _, s, e in overlapping:
            have.update(pd.date_range(max(s, start_dt), min(e, end_dt), freq="D").to_pydatetime())
        missing_days = [d for d in need_days.to_pydatetime() if d not in have]

        # make contiguous missing ranges
        missing_ranges = []
        if missing_days:
            run_start = missing_days[0]
            prev = missing_days[0]
            for d in missing_days[1:]:
                if (d - prev).days == 1:
                    prev = d
                    continue
                missing_ranges.append((run_start, prev))
                run_start = d
                prev = d
            missing_ranges.append((run_start, prev))

        return [], overlapping, missing_ranges

    def _load_forecast_npz(self, path: Path):
        with np.load(path, allow_pickle=True) as npz:
            dates_iso = npz["dates"].astype(str).tolist()
            dates = [datetime.strptime(s, self.DATE_FMT) for s in dates_iso]
            mmaxtemps = npz["mmaxtemps"]  # 2D
            full_forecasts = npz["full_forecasts"].tolist()  # list of objects/strings
        return dates, mmaxtemps, full_forecasts

    def _save_forecast_npz(self, path: Path, dates: list[datetime], mmaxtemps: np.ndarray, full_forecasts: list):
        dates_iso = np.array([d.strftime(self.DATE_FMT) for d in dates], dtype=object)
        np.savez_compressed(path, dates=dates_iso, mmaxtemps=mmaxtemps, full_forecasts=np.array(full_forecasts, dtype=object))

    def _concat_forecasts(self, chunks: list[tuple[list[datetime], np.ndarray, list]]):
        """Stitch by date; if duplicates, keep first occurrence."""
        all_dates, all_m, all_full = [], [], []
        seen = set()
        for dates, m, full in sorted(chunks, key=lambda x: min(x[0])):
            for i, d in enumerate(dates):
                if d in seen:
                    continue
                seen.add(d)
                all_dates.append(d)
                all_m.append(m[i])
                all_full.append(full[i])
        # sort by date
        order = np.argsort(np.array(all_dates, dtype="datetime64[ns]"))
        all_dates = [all_dates[i] for i in order]
        all_m = np.array([all_m[i] for i in order])
        all_full = [all_full[i] for i in order]
        return all_dates, all_m, all_full

    def _load_obs_parquet(self,path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _save_obs_parquet(self,path: Path, df: pd.DataFrame):
        df.to_parquet(path, index=False)

    def _concat_obs(self,dfs: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates().sort_values("date_gmt").reset_index(drop=True)
        return df

    def _archive_files(self,paths: list[Path]):
        for p in paths:
            dest = self.CACHE_DIR / 'old' / p.name
            try:
                shutil.move(str(p), str(dest))
            except Exception:
                pass

    # ------------------------ Use within your retrieval flow ----------------------

    def fetch_with_cache(self, start_date: str, end_date: str):
        """
        Returns:
          dates_forecast: list[datetime]
          all_mmaxtemps: np.ndarray (N x >=5)
          all_full_forecasts: list
          temp_obs: pd.DataFrame (with 'date' matching forecast dates)
        """
        self._ensure_dirs()
        start_dt = datetime.strptime(start_date, self.DATE_FMT)
        end_dt   = datetime.strptime(end_date, self.DATE_FMT)

        # ---------- Forecasts: check cache ----------
        f_spans = self._list_span_files(self.CACHE_DIR / "forecasts", self.FORECAST_RE)
        f_cover, f_overlap, f_missing = self._cover_status(start_dt, end_dt, f_spans)

        if f_cover:
            # Perfect coverage by one file
            dates_forecast, all_mmaxtemps, all_full_forecasts = self._load_forecast_npz(f_cover[0])
            # crop to requested span
            mask = [(start_dt <= d <= end_dt) for d in dates_forecast]
            dates_forecast = [d for d, keep in zip(dates_forecast, mask) if keep]
            all_mmaxtemps = all_mmaxtemps[mask]
            all_full_forecasts = [f for f, keep in zip(all_full_forecasts, mask) if keep]
        else:
            # Load partial overlaps (if any)
            loaded_chunks = []
            for p, s, e in f_overlap:
                loaded_chunks.append(self._load_forecast_npz(p))
            # Fetch missing ranges day-by-day (your original loop)
            fetched_chunks = []
            for (ms, me) in f_missing or []:
                current_dt = ms
                retrieved_days, all_full, all_mm = [], [], []
                while current_dt <= me:
                    date_str = current_dt.strftime(self.DATE_FMT)
                    day, daily_forecast, tmm = self.fetch_nws_forecasts(date_str)
                    if day is not None:
                        try:
                            retrieved_days.append(day.group())
                        except:
                            time_str = "709 AM EDT"
                            rest_str = current_dt.strftime("%a %b %-d %Y") 
                            retrieved_days.append(f"{time_str} {rest_str}")

                        all_full.append(daily_forecast.values.tolist()) #save everything as lists
                        all_mm.append(tmm)
                    current_dt += timedelta(days=1)
                if all_mm:
                    dates_block = [datetime.strptime(d[11:], "%a %b %d %Y") for d in retrieved_days]
                    fetched_chunks.append((dates_block, np.array(all_mm), all_full))

            # Stitch all parts
            if loaded_chunks or fetched_chunks:
                dates_forecast, all_mmaxtemps, all_full_forecasts = self._concat_forecasts(loaded_chunks + fetched_chunks)
            else:
                # Nothing cached → fetch entire span
                current_dt = start_dt
                retrieved_days, all_full_forecasts, all_mmaxtemps = [], [], []
                while current_dt <= end_dt:
                    day, daily_forecast, tmm = self.fetch_nws_forecasts(current_dt.strftime(self.DATE_FMT))
                    if day is not None:
                        retrieved_days.append(day.group())
                        all_full_forecasts.append(daily_forecast)
                        all_mmaxtemps.append(tmm)
                    current_dt += timedelta(days=1)
                dates_forecast = [datetime.strptime(d[11:], "%a %b %d %Y") for d in retrieved_days]

            # Save consolidated continuous file for full span; archive pieces we used
            f_path = self.CACHE_DIR / "forecasts" / f"forecasts_{self._span_key(start_dt, end_dt)}.npz"
            self._save_forecast_npz(f_path, dates_forecast, all_mmaxtemps, all_full_forecasts)
            self._archive_files([p for (p, *_ ) in f_overlap])


        # ---------- Observations in Temperature: check cache ----------
        o_spans = self._list_span_files(self.CACHE_DIR / "obs", self.OBS_RE)
        end_dto =  end_dt + timedelta(days=3)
        o_cover, o_overlap, o_missing = self._cover_status(start_dt, end_dto, o_spans)

        if o_cover:
            temp_obs = self._load_obs_parquet(o_cover[0])
                # crop back to the requested window
            mask = (pd.to_datetime(temp_obs['date']) >= start_dt) & \
                   (pd.to_datetime(temp_obs['date']) <= end_dto)
            temp_obs = temp_obs.loc[mask].reset_index(drop=True)
        else:
            loaded_obs = []
            for p, s, e in o_overlap:
                loaded_obs.append(self._load_obs_parquet(p))

            # Fetch missing ranges in 5-day chunks (your existing EPA logic)
            fetched_obs = []
            for (ms, me) in o_missing or []:
                cur = ms
                chunk_days = 5
                dfs = []
                while cur <= me:
                    chunk_end = min(cur + timedelta(days=chunk_days - 1), me)
                    epa_start = cur.strftime("%Y%m%d")
                    epa_end   = (chunk_end ).strftime("%Y%m%d")
                    df_chunk = self.fetch_epa_observations(epa_start, epa_end, '62101')
                    if df_chunk is not None and not df_chunk.empty:
                        dfs.append(df_chunk)
                    cur = chunk_end + timedelta(days=1)
                if dfs:
                    fetched_obs.append(pd.concat(dfs, ignore_index=True))

            if loaded_obs or fetched_obs:
                temp_obs = self._concat_obs(loaded_obs + fetched_obs)
            else:
                # Nothing cached → fetch entire span
                cur = start_dt
                chunk_days = 5
                dfs = []
                while cur <= end_dt:
                    chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
                    epa_start = cur.strftime("%Y%m%d")
                    epa_end   = (chunk_end ).strftime("%Y%m%d")
                    df_chunk = self.fetch_epa_observations(epa_start, epa_end, '62101')
                    if df_chunk is not None and not df_chunk.empty:
                        dfs.append(df_chunk)
                    cur = chunk_end + timedelta(days=1)
                temp_obs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

            # Normalize and save/archive
            if not temp_obs.empty:
                temp_obs = temp_obs.drop_duplicates().sort_values("date_gmt").reset_index(drop=True)
                temp_obs['datetime_utc'] = pd.to_datetime(
                    temp_obs['date_gmt'] + ' ' + temp_obs['time_gmt'],
                    format='%Y-%m-%d %H:%M'
                ).dt.tz_localize('UTC')
                temp_obs['datetime_local'] = temp_obs['datetime_utc'].dt.tz_convert('US/Eastern')
                temp_obs['date'] = temp_obs['datetime_local'].dt.strftime('%Y-%m-%d')
                temp_obs['time_local'] = temp_obs['datetime_local'].dt.strftime('%H:%M')

            o_path = self.CACHE_DIR / "obs" / f"temp_obs_{self._span_key(start_dt, end_dto)}.parquet"
            self._save_obs_parquet(o_path, temp_obs)
            self._archive_files([p for (p, *_ ) in o_overlap])


        # ---------- Observations in Humidity: check cache ----------
        oh_spans = self._list_span_files(self.CACHE_DIR / "obs", self.OBS_HU)
        oh_cover, oh_overlap, oh_missing = self._cover_status(start_dt, end_dto, oh_spans)

        if oh_cover:
            humid_obs = self._load_obs_parquet(oh_cover[0])
                # crop back to the requested window
            mask = (pd.to_datetime(humid_obs['date']) >= start_dt) & \
                   (pd.to_datetime(humid_obs['date']) <= end_dto)
            humid_obs = humid_obs.loc[mask].reset_index(drop=True)
        else:
            loaded_obs = []
            for p, s, e in oh_overlap:
                loaded_obs.append(self._load_obs_parquet(p))

            # Fetch missing ranges in 5-day chunks (your existing EPA logic)
            fetched_obs = []
            for (ms, me) in oh_missing or []:
                cur = ms
                chunk_days = 5
                dfs = []
                while cur <= me:
                    chunk_end = min(cur + timedelta(days=chunk_days - 1), me)
                    epa_start = cur.strftime("%Y%m%d")
                    epa_end   = (chunk_end ).strftime("%Y%m%d")
                    df_chunk = self.fetch_epa_observations(epa_start, epa_end, '62201')
                    if df_chunk is not None and not df_chunk.empty:
                        dfs.append(df_chunk)
                    cur = chunk_end + timedelta(days=1)
                if dfs:
                    fetched_obs.append(pd.concat(dfs, ignore_index=True))

            if loaded_obs or fetched_obs:
                humid_obs = self._concat_obs(loaded_obs + fetched_obs)
            else:
                # Nothing cached → fetch entire span
                cur = start_dt
                chunk_days = 5
                dfs = []
                while cur <= end_dt:
                    chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
                    epa_start = cur.strftime("%Y%m%d")
                    epa_end   = (chunk_end ).strftime("%Y%m%d")
                    df_chunk = self.fetch_epa_observations(epa_start, epa_end, '62201')
                    if df_chunk is not None and not df_chunk.empty:
                        dfs.append(df_chunk)
                    cur = chunk_end + timedelta(days=1)
                humid_obs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

            # Normalize and save/archive
            if not humid_obs.empty:
                humid_obs = humid_obs.drop_duplicates().sort_values("date_gmt").reset_index(drop=True)
                humid_obs['datetime_utc'] = pd.to_datetime(
                    humid_obs['date_gmt'] + ' ' + humid_obs['time_gmt'],
                    format='%Y-%m-%d %H:%M'
                ).dt.tz_localize('UTC')
                humid_obs['datetime_local'] = humid_obs['datetime_utc'].dt.tz_convert('US/Eastern')
                humid_obs['date'] = humid_obs['datetime_local'].dt.strftime('%Y-%m-%d')
                humid_obs['time_local'] = humid_obs['datetime_local'].dt.strftime('%H:%M')

            o_path = self.CACHE_DIR / "obs" / f"humid_obs_{self._span_key(start_dt, end_dto)}.parquet"
            self._save_obs_parquet(o_path, humid_obs)
            self._archive_files([p for (p, *_ ) in oh_overlap])

        # Ensure obs dates align to forecast dates if you need a strict match
        # (Optional) filter temp_obs to only forecast dates:
        #if not temp_obs.empty:
        #    keep_dates = set(d.strftime(self.DATE_FMT) for d in dates_forecast)
        #    temp_obs = temp_obs[temp_obs['date'].isin(keep_dates)].copy()

        return dates_forecast, all_mmaxtemps, all_full_forecasts, temp_obs, humid_obs



    def run_analysis(self, start_date: str, end_date: str, 
                    min_temp_threshold: float = 80,
                    use_heat_index: bool = False) -> Dict:
        """
        Run complete analysis pipeline
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_temp_threshold: Minimum temperature to include in analysis
            use_heat_index: Whether to calculate and use heat index instead of temperature
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Fetch forecasts
        print("\n" + "="*40)
        print("FETCHING DATA...")
        print("="*40)
        dates_forecast, all_mmaxtemps, all_full_forecasts, temp_obs, humid_obs = self.fetch_with_cache( start_date, end_date)
        minmaxar = np.array(all_mmaxtemps)


        # Calculate metrics for each station (just one at the moment)
        print("\n" + "="*40)
        print("ANALYZING BY STATION...")
        print("="*40)
        

        if not use_heat_index:
            pred_condensed = pd.DataFrame({ "bulletin_date": dates_forecast, "max0": minmaxar[:, 0],
                                            "max1": minmaxar[:, 2], "max2": minmaxar[:, 4], })
            # Process daily maxima
            daily_max = self.process_daily_maxima(temp_obs)
            print("Currently ONLY analyzing daily Tmax (plan to do Heat Index)")
        else:
            df_comb_obs = pd.merge(
                temp_obs[['date_gmt','time_gmt','station_id','sample_measurement',
                          'datetime_utc','datetime_local','date','time_local']],
                humid_obs[['date_gmt','time_gmt','station_id','sample_measurement']],
                on=['date_gmt','time_gmt','station_id'],
                how='inner',
                suffixes=('_temp','_rh'))
            ufunc_hi = np.frompyfunc(lambda t, rh: self.calculate_heat_index(float(t), float(rh)), 2, 1)
            hi = pd.to_numeric(ufunc_hi(df_comb_obs['sample_measurement_temp'], df_comb_obs['sample_measurement_rh']), errors='coerce')

            # assemble in the same column order as temp_obs
            heat_obs = df_comb_obs[['date_gmt','time_gmt','station_id','datetime_utc','datetime_local','date','time_local']].copy()
            heat_obs['sample_measurement'] = hi
            heat_obs = heat_obs[['date_gmt','time_gmt','sample_measurement','station_id',
                                 'datetime_utc','datetime_local','date','time_local']]
            daily_max = self.process_daily_maxima(heat_obs)
            
            himaxes = np.array(self.hi_max_from_full(all_full_forecasts))
            pred_condensed = pd.DataFrame({ "bulletin_date": dates_forecast,"max0": himaxes[:, 0],
                                            "max1": himaxes[:, 1], "max2": himaxes[:, 2], })
            #calculate_heat_index(
            print("Currently analyzing Heat Index on hourly basis, max each day")
        

        # Filter for days above threshold
        #warm_days = daily_max['max_temp'] >= min_temp_threshold
        #daily_max_filtered = daily_max[warm_days]
        #pred_condensed_filtered = pred_condensed[warm_days]
        #print(f"✓ Found {len(daily_max_filtered)} days above {min_temp_threshold}°F threshold")
        

        
        
        for station_id in self.stations.keys():
            station_data = daily_max[daily_max['station_id'] == station_id]
            
            if not station_data.empty:
                print(f"\nProcessing {self.stations[station_id]['name']}...")
                print(f"table below only considers obs T>{min_temp_threshold}°F")
                station_metrics = self.calculate_metrics(pred_condensed, station_data,min_temp_threshold)
                print(station_metrics.to_string(index=False))
                
                print("\n Warning (just hottest 1 hr coded now)")
                heat_events = self.identify_heat_events(pred_condensed,station_data,station_metrics['bias(Obs-F)'],metric='warning')
                print(heat_events.to_string(index=False))
                
                print("\n Advisory (both options)")
                heat_events = self.identify_heat_events(pred_condensed,station_data,station_metrics['bias(Obs-F)'])
                print(heat_events.to_string(index=False))

                print("\n Heatwave (>=90°F)")
                heat_events = self.identify_heat_events(pred_condensed,station_data,station_metrics['bias(Obs-F)'],metric='heatwave')
                print(heat_events.to_string(index=False))

                
##                results[station_id] = {
##                    'station_name': self.stations[station_id]['name'],
##                    'metrics': station_metrics,
##                    'heat_events': heat_events,
##                    'data': station_data  # Store for potential further analysis
##                }
                print(f"  ✓ Calculated metrics for {len(pred_condensed)} days: {start_date} to {end_date}")
            else:
                print(f"\n⚠ No data above threshold for {self.stations[station_id]['name']}")
        
        return results
    

import argparse

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your EPA credentials
    analyzer = NWSForecastAnalyzer(
        email="john_nicklas@brown.edu",
        api_key="mauvebird36"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="Year to process")
    args = parser.parse_args()
    year = args.year

    # Run analysis for summer 2024 as an example
    analyzer.run_analysis(
        start_date=f"{year}-04-05",
        end_date=f"{year}-09-30", #2025 not yet available on EPA site 04-01 to 2024-09-30
        min_temp_threshold=75, use_heat_index=True,
    )
    

