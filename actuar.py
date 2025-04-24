import pandas as pd
import numpy as np
import datetime
# Removed plotting libraries

# --- Default Configuration Values ---
# These can be overridden by parameters passed to functions
DEFAULT_N_CLAIMS = 8000
DEFAULT_START_YEAR = 2014
DEFAULT_END_YEAR = 2023
DEFAULT_MAX_DEV_PERIODS = 10
DEFAULT_ANOMALY_THRESHOLD = 2.0

# --- 1. Synthetic Data Generation ---

def generate_synthetic_claims(n_claims, start_year, end_year, data_end_date):
    """Generates a DataFrame of synthetic claims data."""
    print(f"Generating {n_claims} synthetic claims from {start_year} to {end_year}, valuation date {data_end_date}...")
    np.random.seed(42) # for reproducibility

    # Generate Accident Dates (more claims in later years - simple growth)
    years = np.arange(start_year, end_year + 1)
    if len(years) == 0:
        print("Warning: start_year > end_year. No claims generated.")
        return pd.DataFrame(columns=[
            'accident_date', 'claim_id', 'accident_year', 'report_date',
            'ultimate_cost', 'payment_date', 'payment_amount'
        ])

    # Basic claim count allocation
    claim_counts_per_year = (np.linspace(0.8, 1.2, len(years)) * n_claims / len(years)).astype(int)
    # Adjust to ensure total is exactly n_claims
    diff = n_claims - claim_counts_per_year.sum()
    indices = np.random.choice(len(years), abs(diff), replace=True)
    claim_counts_per_year[indices] += np.sign(diff)
    claim_counts_per_year = np.maximum(0, claim_counts_per_year) # Ensure non-negative
    # Final check if adjustment failed (e.g., all counts were 0)
    if claim_counts_per_year.sum() != n_claims and n_claims > 0:
         # Simple redistribution if adjustment failed
         claim_counts_per_year[:] = 0
         claim_counts_per_year[:n_claims % len(years)] = n_claims // len(years) + 1
         claim_counts_per_year[n_claims % len(years):] = n_claims // len(years)


    accident_dates = []
    for year, count in zip(years, claim_counts_per_year):
        if count <= 0: continue # Skip if no claims for this year
        days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
        start_of_year = datetime.date(year, 1, 1)
        random_days = np.random.randint(0, days_in_year, count)
        accident_dates.extend([start_of_year + datetime.timedelta(days=int(d)) for d in random_days])

    if not accident_dates:
         print("Warning: No accident dates generated based on inputs.")
         return pd.DataFrame(columns=[
            'accident_date', 'claim_id', 'accident_year', 'report_date',
            'ultimate_cost', 'payment_date', 'payment_amount'
        ])

    claims_df = pd.DataFrame({'accident_date': accident_dates})
    claims_df['accident_date'] = pd.to_datetime(claims_df['accident_date'])
    claims_df['claim_id'] = range(1, len(claims_df) + 1)
    claims_df['accident_year'] = claims_df['accident_date'].dt.year

    # Simulate Reporting Lag
    reporting_lag_days = np.random.exponential(scale=30, size=len(claims_df)).astype(int) + 1
    claims_df['report_date'] = claims_df['accident_date'] + pd.to_timedelta(reporting_lag_days, unit='D')

    # Simulate Ultimate Claim Cost with Inflation
    base_mu = np.log(5000)
    min_acc_year = claims_df['accident_year'].min()
    inflation_factor = (claims_df['accident_year'] - min_acc_year) * 0.02 # Use min actual year as base
    mu = base_mu + inflation_factor
    sigma = 0.8
    claims_df['ultimate_cost'] = np.random.lognormal(mean=mu, sigma=sigma, size=len(claims_df))

    # Simulate Payments
    payments = []
    max_dev_years_for_payment = 5 # Internal detail: most paid within 5 years
    data_end_timestamp = pd.Timestamp(data_end_date)

    for _, claim in claims_df.iterrows():
        paid_so_far = 0
        payment_profile = np.random.dirichlet(np.ones(max_dev_years_for_payment)*0.8 + np.array([3, 2, 1, 0.5, 0.2]))
        payment_date = claim['report_date']

        for dev_year in range(max_dev_years_for_payment):
            payment_delay_days = np.random.randint(30, 365)
            payment_date = payment_date + datetime.timedelta(days=int(payment_delay_days))

            if payment_date > data_end_timestamp: break # Stop payments after valuation date

            payment_amount = claim['ultimate_cost'] * payment_profile[dev_year]
            payment_amount = max(0, payment_amount + np.random.normal(0, payment_amount * 0.1))

            if paid_so_far + payment_amount >= claim['ultimate_cost'] * 0.999:
                payment_amount = claim['ultimate_cost'] - paid_so_far
                if payment_amount < 0: payment_amount = 0

            if payment_amount > 0:
                 payments.append({
                     'claim_id': claim['claim_id'],
                     'payment_date': payment_date,
                     'payment_amount': payment_amount
                 })
                 paid_so_far += payment_amount

            if paid_so_far >= claim['ultimate_cost']: break

    payments_df = pd.DataFrame(payments)

    # Combine claim details with payments
    if payments_df.empty:
        full_claims_df = claims_df.copy()
        full_claims_df['payment_date'] = pd.NaT
        full_claims_df['payment_amount'] = 0.0
    else:
        full_claims_df = pd.merge(claims_df, payments_df, on='claim_id', how='left')

    full_claims_df['payment_amount'] = full_claims_df['payment_amount'].fillna(0)
    if 'payment_date' not in full_claims_df.columns:
         full_claims_df['payment_date'] = pd.NaT

    print("Synthetic data generation complete.")
    return full_claims_df

# --- 2. Triangle Aggregation ---

def aggregate_to_triangle(claims_df, max_dev_periods, data_end_date, value_col='payment_amount', date_col='payment_date'):
    """Aggregates granular claims data into a cumulative development triangle."""
    print("Aggregating data into cumulative triangle...")

    # Ensure claims_df is not empty
    if claims_df.empty or date_col not in claims_df.columns:
        print(f"Warning: Input DataFrame is empty or missing '{date_col}'. Returning empty triangle.")
        return pd.DataFrame(columns=range(1, max_dev_periods + 1))

    # Ensure date columns are datetime objects, handling potential NaTs
    claims_df_copy = claims_df.copy() # Work on a copy
    claims_df_copy[date_col] = pd.to_datetime(claims_df_copy[date_col], errors='coerce')
    claims_df_copy['accident_date'] = pd.to_datetime(claims_df_copy['accident_date'], errors='coerce')

    # Filter out rows where dates couldn't be parsed or are missing
    claims_df_copy.dropna(subset=[date_col, 'accident_date'], inplace=True)

    # Ensure data_end_date is datetime
    data_end_timestamp = pd.Timestamp(data_end_date)

    # Filter payments up to the valuation date
    claims_df_filtered = claims_df_copy[claims_df_copy[date_col] <= data_end_timestamp].copy()

    if claims_df_filtered.empty:
        print("Warning: No claims data on or before the data end date. Returning empty triangle.")
        acc_years = sorted(claims_df_copy['accident_year'].unique()) if 'accident_year' in claims_df_copy else []
        return pd.DataFrame(index=pd.Index(acc_years, name='accident_year'),
                          columns=range(1, max_dev_periods + 1))

    # Calculate development period in years
    claims_df_filtered['dev_days'] = (claims_df_filtered[date_col] - claims_df_filtered['accident_date']).dt.days
    claims_df_filtered['dev_period'] = ((claims_df_filtered['dev_days'] / 365.25).fillna(-1) + 1).astype(int)

    # Keep only valid development periods (>= 1)
    claims_df_filtered = claims_df_filtered[claims_df_filtered['dev_period'] >= 1].copy()

    if claims_df_filtered.empty:
        print("Warning: No data found within valid development periods. Returning empty triangle.")
        acc_years = sorted(claims_df_copy['accident_year'].unique()) if 'accident_year' in claims_df_copy else []
        return pd.DataFrame(index=pd.Index(acc_years, name='accident_year'),
                          columns=range(1, max_dev_periods + 1))


    # Group by accident year and development period
    grouped = claims_df_filtered.groupby(['accident_year', 'dev_period'])[value_col].sum().reset_index()

    # Pivot to create incremental triangle
    triangle = grouped.pivot(index='accident_year', columns='dev_period', values=value_col)

    # Ensure all accident years from original (pre-filter) data are included
    all_accident_years = sorted(claims_df_copy['accident_year'].unique())
    triangle = triangle.reindex(index=all_accident_years) # Add missing years

    # Ensure all development periods up to max are included
    triangle = triangle.reindex(columns=range(1, max_dev_periods + 1), fill_value=0) # Add missing columns, fill with 0
    triangle.fillna(0, inplace=True) # Fill NaNs for years with no claims in certain periods

    # Create cumulative triangle
    cumulative_triangle = triangle.cumsum(axis=1)

    # Mask future periods based on valuation date
    latest_year = data_end_timestamp.year
    for year_idx in cumulative_triangle.index:
        year = int(year_idx)
        max_dev_for_year = latest_year - year + 1
        if max_dev_for_year < max_dev_periods:
             cumulative_triangle.loc[year_idx, max_dev_for_year + 1:] = np.nan

    print("Triangle aggregation complete.")
    return cumulative_triangle


# --- 3. Chain-Ladder Calculation ---

def calculate_chain_ladder(triangle, max_dev_periods):
    """Performs the Paid Chain-Ladder calculation."""
    print("Calculating Chain-Ladder...")
    if triangle.empty:
        print("Warning: Input triangle is empty. Cannot calculate Chain-Ladder.")
        # Return structure with empty/default values
        return {
            "cumulative_triangle": pd.DataFrame(),
            "ata_factors_selected": pd.DataFrame(),
            "all_ata_factors": {},
            "cdfs": pd.Series(dtype=float),
            "latest_paid": pd.Series(dtype=float),
            "ultimate_losses": pd.Series(dtype=float),
            "ibnr": pd.Series(dtype=float),
            "projected_triangle": pd.DataFrame()
        }

    # Ensure triangle uses max_dev_periods columns, preserving existing NaNs
    cumulative_triangle = triangle.reindex(columns=range(1, max_dev_periods + 1)).copy()
    rows, cols = cumulative_triangle.shape

    if cols == 0:
        print("Warning: Triangle has no development periods. Cannot calculate Chain-Ladder.")
        return { # Return empty structure
            "cumulative_triangle": cumulative_triangle,
            "ata_factors_selected": pd.DataFrame(), "all_ata_factors": {}, "cdfs": pd.Series(dtype=float),
            "latest_paid": pd.Series(dtype=float, index=triangle.index),
            "ultimate_losses": pd.Series(dtype=float, index=triangle.index),
            "ibnr": pd.Series(dtype=float, index=triangle.index),
            "projected_triangle": cumulative_triangle
        }

    # Calculate Age-to-Age (ATA) factors
    ata_factors = pd.DataFrame(index=['Factors'], columns=range(1, cols), dtype=float)
    all_ata = {}

    for j in range(cols - 1): # Develops from period j+1 to j+2
        dev_from = j + 1
        dev_to = j + 2
        col_j = cumulative_triangle.iloc[:, j]
        col_j_plus_1 = cumulative_triangle.iloc[:, j + 1]

        valid_mask = col_j.notna() & col_j_plus_1.notna() & (col_j != 0)

        if not valid_mask.any():
             print(f"Warning: No valid pairs for Dev {dev_from}-{dev_to}. Factor set to 1.0")
             selected_factor = 1.0
             all_ata[dev_from] = pd.Series(dtype=float) # Store index for dev_from
        else:
            individual_factors = col_j_plus_1[valid_mask] / col_j[valid_mask]
            all_ata[dev_from] = individual_factors # Store index for dev_from

            sum_col_j_plus_1 = col_j_plus_1[valid_mask].sum()
            sum_col_j = col_j[valid_mask].sum()

            if sum_col_j == 0:
                selected_factor = 1.0
                print(f"Warning: Sum of valid data in Dev {dev_from} is zero. Factor {dev_from}-{dev_to} set to 1.0")
            else:
                selected_factor = sum_col_j_plus_1 / sum_col_j

            if pd.isna(selected_factor): selected_factor = 1.0 # Handle potential NaN result

        ata_factors.loc['Factors', dev_from] = selected_factor # Assign factor for dev_from -> dev_to

    # Select Tail Factor
    if cols > 1 and not ata_factors.empty:
        last_valid_col_idx = ata_factors.iloc[0].last_valid_index()
        if last_valid_col_idx is not None:
            tail_factor = ata_factors.loc['Factors', last_valid_col_idx]
            if pd.isna(tail_factor) or tail_factor < 1.0:
                # Try average of last 3 valid factors >= 1.0
                valid_tail_cands = ata_factors.iloc[0][ata_factors.iloc[0] >= 1.0].dropna()
                if len(valid_tail_cands) >= 3:
                    tail_factor = valid_tail_cands.iloc[-3:].mean()
                elif not valid_tail_cands.empty:
                    tail_factor = valid_tail_cands.iloc[-1] # Use last valid one
                else:
                    tail_factor = 1.0 # Fallback
            if tail_factor < 1.0: # Final check
                 print(f"Warning: Selected tail factor ({tail_factor:.4f}) < 1.0. Overriding to 1.0.")
                 tail_factor = 1.0
        else:
            tail_factor = 1.0 # No valid factors calculated
            print("Warning: No valid ATA factors calculated. Tail factor set to 1.0.")
    else:
        tail_factor = 1.0 # Not enough columns
        print("Warning: Not enough columns for tail factor calculation. Tail factor set to 1.0.")

    ata_factors[cols] = tail_factor # Development from col to col+1 (ultimate)
    print("Age-to-Age Factors (Selected):")
    print(ata_factors.to_string(float_format="%.4f"))

    # Calculate Cumulative Development Factors (CDFs) to Ultimate
    cdfs = ata_factors.iloc[0].sort_index(ascending=False).cumprod().sort_index(ascending=True)
    cdfs.name = "CDFs_to_Ultimate"
    print("\nCumulative Development Factors (LDFs to Ultimate):")
    print(cdfs.to_string(float_format="%.4f"))

    # Project Ultimate Losses
    ultimate_losses = pd.Series(index=cumulative_triangle.index, dtype=float)
    latest_paid_actual = pd.Series(index=cumulative_triangle.index, dtype=float)

    for year in cumulative_triangle.index:
        row_data = cumulative_triangle.loc[year].dropna()
        if row_data.empty:
            latest_paid = 0
            dev_periods_so_far = 0
        else:
            latest_paid = row_data.iloc[-1]
            dev_periods_so_far = row_data.index[-1] # Last column index with data

        latest_paid_actual[year] = latest_paid

        if dev_periods_so_far >= cols: # Fully developed
            cdf_for_year = 1.0
        elif dev_periods_so_far == 0: # No payments yet
             latest_paid = 0
             cdf_for_year = cdfs.get(1, 1.0) # Use CDF from 1 to ultimate
        else:
             # CDF needed is for development from dev_periods_so_far to ultimate
             cdf_for_year = cdfs.get(dev_periods_so_far, 1.0) # Get the right CDF

        ultimate_losses[year] = latest_paid * cdf_for_year

    ultimate_losses.name = "Projected_Ultimate_Loss"
    print("\nProjected Ultimate Losses:")
    print(ultimate_losses.to_string(float_format="%.0f"))

    # Fill triangle to ultimate (Projected Triangle)
    projected_triangle = cumulative_triangle.copy()
    for year in projected_triangle.index:
        last_valid_dev = projected_triangle.loc[year].last_valid_index()
        if last_valid_dev is None: # No data for this year
            if ultimate_losses.get(year, 0) > 0 and 1 in cdfs and cdfs[1] > 0:
                # Estimate first period payment based on ultimate and CDF
                # This is a rough estimation
                estimated_first_period = ultimate_losses[year] / cdfs[1]
                projected_triangle.loc[year, 1] = estimated_first_period
                last_valid_dev = 1 # Start projection from here
            else:
                 continue # Skip years with no data and no projection possible

        current_val = projected_triangle.loc[year, last_valid_dev]
        for dev_period in range(int(last_valid_dev) + 1, cols + 1):
             ata = ata_factors.loc['Factors', dev_period -1] if dev_period -1 in ata_factors.columns else 1.0
             current_val = current_val * ata
             projected_triangle.loc[year, dev_period] = current_val

    # Calculate IBNR (Ultimate - Latest Paid)
    ibnr = ultimate_losses - latest_paid_actual
    ibnr = ibnr.clip(lower=0) # Ensure non-negative
    ibnr.name = "IBNR_Estimate"
    print("\nIBNR Estimate:")
    print(ibnr.to_string(float_format="%.0f"))
    total_ibnr = ibnr.sum()
    print(f"\nTotal IBNR Estimate: {total_ibnr:,.0f}")

    results = {
        "cumulative_triangle": cumulative_triangle,
        "ata_factors_selected": ata_factors,
        "all_ata_factors": all_ata, # Keys are dev period start (e.g., 1 for 1-2 factor)
        "cdfs": cdfs,
        "latest_paid": latest_paid_actual,
        "ultimate_losses": ultimate_losses,
        "ibnr": ibnr,
        "projected_triangle": projected_triangle,
        "total_ibnr": total_ibnr
    }
    print("Chain-Ladder calculation complete.")
    return results

# --- 4. Basic Anomaly Detection ---

def detect_anomalies(all_ata_factors, threshold=DEFAULT_ANOMALY_THRESHOLD):
    """Detects anomalies in individual Age-to-Age factors using Z-score."""
    print(f"\n--- Anomaly Detection (Z-score Threshold: {threshold}) ---")
    anomalies = []
    # `all_ata_factors` keys are the start of the dev period (e.g., 1 for 1-2)
    for dev_period_start, factors in all_ata_factors.items():
        if len(factors) < 3: continue # Need sufficient data points

        mean = factors.mean()
        std = factors.std()

        if std == 0 or pd.isna(std): continue # Avoid division by zero or NaN std dev

        z_scores = (factors - mean) / std

        outliers = factors[abs(z_scores) > threshold]
        for year, factor in outliers.items():
            anomalies.append({
                "Accident Year": year,
                "Development Lag": f"{dev_period_start}-{dev_period_start+1}",
                "Factor": factor,
                "Z-Score": z_scores.loc[year],
                "Mean Factor (Lag)": mean,
                "Std Dev (Lag)": std
            })

    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        print(f"Detected {len(anomalies)} potential anomalies:")
        # Ensure specific column order for better readability
        cols_ordered = ["Accident Year", "Development Lag", "Factor", "Z-Score", "Mean Factor (Lag)", "Std Dev (Lag)"]
        print(anomaly_df[cols_ordered].to_string(float_format="%.4f"))
        return anomaly_df
    else:
        print("No significant anomalies detected.")
        return pd.DataFrame(columns=["Accident Year (שנת אירוע)", "Development Lag (פיגור התפתחות)", "Factor (מקדם)", "Z-Score (ציון תקן)", "Mean Factor (Lag) (מקדם ממוצע לפיגור)", "Std Dev (Lag) (סטיית תקן לפיגור)"])


# --- 5. Bornhuetter-Ferguson (BF) Method ---

def generate_apriori_estimates(latest_paid, cdfs, method='cl_ultimate_proxy', **kwargs):
    """
    Generates a plausible 'a priori' ultimate loss estimate for BF method (MVP placeholder).

    Args:
        latest_paid (pd.Series): Series of latest paid amounts by accident year.
        cdfs (pd.Series): Series of Cumulative Development Factors (CDFs) to ultimate.
        method (str): Method to use ('cl_ultimate_proxy', 'avg_cdf_extrapolation', etc.).
        **kwargs: Additional arguments depending on the method.

    Returns:
        pd.Series: A priori ultimate loss estimates indexed by accident year.
    """
    print(f"Generating A Priori estimates using method: {method}")
    apriori_ultimates = pd.Series(index=latest_paid.index, dtype=float)

    if method == 'cl_ultimate_proxy':
        # Simple proxy: Use the Chain Ladder ultimate if available
        # Requires passing cl_ultimate_losses in kwargs
        cl_ultimates = kwargs.get('cl_ultimate_losses')
        if cl_ultimates is not None and isinstance(cl_ultimates, pd.Series):
             print("Using Chain Ladder ultimate as A Priori proxy.")
             return cl_ultimates.copy()
        else:
             print("Warning: CL Ultimates not provided for 'cl_ultimate_proxy'. Falling back.")
             # Fallback to another simple method if CL ultimate isn't passed
             method = 'avg_cdf_extrapolation' # Example fallback

    if method == 'avg_cdf_extrapolation':
        print("Using simple extrapolation based on latest paid and average development.")
        # Very basic: Use an overall average CDF applied to latest paid
        # This isn't a true 'a priori' but serves as a placeholder calculation
        if not cdfs.empty:
             avg_cdf = cdfs.mean() # Crude average CDF
             if avg_cdf > 0:
                 apriori_ultimates = latest_paid * avg_cdf
             else:
                 apriori_ultimates = latest_paid # Fallback if avg CDF is zero
        else:
             apriori_ultimates = latest_paid # Fallback if no CDFs

    # Ensure non-negative
    apriori_ultimates = apriori_ultimates.fillna(0).clip(lower=0)
    print("A Priori Estimates (Placeholder):")
    print(apriori_ultimates.to_string(float_format="%.0f"))
    return apriori_ultimates


def calculate_bornhuetter_ferguson(cumulative_triangle, cdfs, apriori_ultimate_losses):
    """
    Performs the Bornhuetter-Ferguson (BF) calculation.

    Args:
        cumulative_triangle (pd.DataFrame): Cumulative paid triangle.
        cdfs (pd.Series): Cumulative Development Factors (CDFs) to ultimate.
        apriori_ultimate_losses (pd.Series): A priori ultimate loss estimates by accident year.

    Returns:
        dict: Dictionary containing BF results (IBNR, Ultimate).
    """
    print("\nCalculating Bornhuetter-Ferguson...")
    if cumulative_triangle.empty or cdfs.empty or apriori_ultimate_losses.empty:
        print("Warning: Input data (triangle, cdfs, or a priori) is empty. Cannot calculate BF.")
        return {
            "bf_ibnr": pd.Series(dtype=float),
            "bf_ultimate_losses": pd.Series(dtype=float),
            "pct_unreported": pd.Series(dtype=float),
            "apriori_ultimate_losses": apriori_ultimate_losses # Return input
        }

    # Ensure consistent indexing
    latest_paid = pd.Series(index=cumulative_triangle.index, dtype=float)
    dev_periods_so_far_map = pd.Series(index=cumulative_triangle.index, dtype=int)

    for year in cumulative_triangle.index:
        row_data = cumulative_triangle.loc[year].dropna()
        if row_data.empty:
            latest_paid[year] = 0
            dev_periods_so_far_map[year] = 0
        else:
            latest_paid[year] = row_data.iloc[-1]
            dev_periods_so_far_map[year] = row_data.index[-1] # Last column index with data

    # Calculate Percentage Unreported (based on CDFs)
    # % Unreported = 1 - % Reported = 1 - (1 / CDF)
    pct_reported = pd.Series(index=cumulative_triangle.index, dtype=float)
    for year, dev_period in dev_periods_so_far_map.items():
         if dev_period == 0:
             # Assume 0% reported if no payments yet
             cdf_for_period = cdfs.get(1, 0) # Need CDF from 1 to ultimate
             pct_reported[year] = 0.0
         elif dev_period >= len(cdfs): # Fully developed based on available CDFs
             pct_reported[year] = 1.0
         else:
             cdf_for_period = cdfs.get(dev_period, 0)
             if cdf_for_period > 0:
                 pct_reported[year] = 1 / cdf_for_period
             else:
                 pct_reported[year] = 0 # Undefined if CDF is zero, assume 0% reported

    pct_unreported = 1 - pct_reported
    pct_unreported = pct_unreported.clip(lower=0.0, upper=1.0) # Bounds [0, 1]
    pct_unreported.name = "Percent_Unreported (אחוז_לא_מדווח)"

    # Calculate BF IBNR = A Priori Ultimate * % Unreported
    bf_ibnr = apriori_ultimate_losses * pct_unreported
    bf_ibnr.name = "BF_IBNR_Estimate (תתו\"ע לפי ב\"פ)"

    # Calculate BF Ultimate = Latest Paid + BF IBNR
    bf_ultimate_losses = latest_paid + bf_ibnr
    bf_ultimate_losses.name = "BF_Ultimate_Loss (נזק_סופי_לפי_ב\"פ)"

    print("BF Calculation Complete.")
    print("\nBF IBNR Estimate:")
    print(bf_ibnr.to_string(float_format="%.0f"))
    print("\nBF Projected Ultimate Losses:")
    print(bf_ultimate_losses.to_string(float_format="%.0f"))

    total_bf_ibnr = bf_ibnr.sum()
    print(f"\nTotal BF IBNR Estimate: {total_bf_ibnr:,.0f}")

    results = {
        "bf_ibnr": bf_ibnr,
        "bf_ultimate_losses": bf_ultimate_losses,
        "pct_unreported": pct_unreported,
        "apriori_ultimate_losses": apriori_ultimate_losses, # Include the input A Priori
        "total_bf_ibnr": total_bf_ibnr
    }
    return results

# --- Plotting Functions (Consider moving to a separate file or app layer) ---
# Removing plotting from this core actuarial logic file.