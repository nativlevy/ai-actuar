import streamlit as st
import pandas as pd
import numpy as np
import datetime
import actuar # Import the refactored functions

st.set_page_config(layout="wide") # Use wider layout for tables

st.title("Actuarial Chain Ladder Analysis")

# --- Sidebar for Configuration ---
st.sidebar.header("Configuration Parameters")

# Use default values from actuar.py if needed, but define them here for clarity
default_n_claims = 8000
default_start_year = 2014
default_end_year = 2023
default_max_dev_periods = 10
default_anomaly_threshold = 2.0
current_year = datetime.date.today().year

n_claims = st.sidebar.number_input(
    "Number of Synthetic Claims",
    min_value=100,
    max_value=100000,
    value=default_n_claims,
    step=100,
    help="Total number of claims to simulate."
)

start_year = st.sidebar.number_input(
    "Start Accident Year",
    min_value=1980,
    max_value=current_year - 1,
    value=default_start_year,
    step=1,
    help="First year for which claims occur."
)

end_year = st.sidebar.number_input(
    "End Accident Year (Valuation Year)",
    min_value=start_year, # End year must be >= start year
    max_value=current_year,
    value=default_end_year,
    step=1,
    help="Last accident year included. Calculations use data up to Dec 31st of this year."
)

max_dev_periods = st.sidebar.number_input(
    "Maximum Development Periods (Years)",
    min_value=1,
    max_value=30,
    value=default_max_dev_periods,
    step=1,
    help="The maximum number of development years to track and project."
)

anomaly_threshold = st.sidebar.slider(
    "Anomaly Detection Z-Score Threshold",
    min_value=1.0,
    max_value=5.0,
    value=default_anomaly_threshold,
    step=0.1,
    help="Z-score threshold to flag individual ATA factors as potential anomalies."
)

# Add a button to trigger the analysis
run_button = st.sidebar.button("Run Analysis", type="primary")

# --- Main Area for Results ---
st.subheader("Analysis Results")

# Placeholder for results - only run when button is clicked
if 'results' not in st.session_state:
    st.session_state.results = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'bf_results' not in st.session_state: # Add BF results placeholder
    st.session_state.bf_results = None
if 'log_messages' not in st.session_state: # Initialize log message list
    st.session_state.log_messages = []

if run_button:
    # Clear previous logs on new run
    st.session_state.log_messages = []
    # Calculate valuation date based on selected end year
    data_end_date = datetime.date(end_year, 12, 31)

    # Display progress/info
    with st.spinner(f"Running analysis for {start_year}-{end_year} with {n_claims} claims..."):
        st.write("--- Log (יומן) --- ") # Simple log area
        log_placeholder = st.empty()

        # Define function to append log messages
        def log_message(msg):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.log_messages.append(f"[{timestamp}] {msg}")
            # Display intermediate log state (optional, can be removed if too fast)
            # log_placeholder.text_area("Calculation Log (יומן חישובים)", value="\n".join(st.session_state.log_messages), height=150)
            print(msg) # Also print to console

        # --- Start Calculations ---
        try:
            # 1. Generate Data
            log_message(f"Generating {n_claims} claims...")
            raw_claims_data = actuar.generate_synthetic_claims(n_claims, start_year, end_year, data_end_date)
            log_message("Synthetic data generated.")

            if not raw_claims_data.empty:
                # 2. Aggregate into Triangle
                log_message("Aggregating triangle...")
                cumulative_paid_triangle = actuar.aggregate_to_triangle(
                    raw_claims_data,
                    max_dev_periods=max_dev_periods,
                    data_end_date=data_end_date
                )
                log_message("Triangle aggregated.")

                # 3. Calculate Chain Ladder & IBNR
                log_message("Calculating Chain Ladder...")
                cl_results = actuar.calculate_chain_ladder(cumulative_paid_triangle, max_dev_periods)
                st.session_state.results = cl_results # Store results in session state
                log_message("Chain Ladder calculation complete.")

                # 4. Detect Anomalies
                log_message("Detecting anomalies...")
                detected_anomalies = actuar.detect_anomalies(cl_results['all_ata_factors'], threshold=anomaly_threshold)
                st.session_state.anomalies = detected_anomalies # Store anomalies
                log_message("Anomaly detection complete.")

                # 5. Generate A Priori & Calculate Bornhuetter-Ferguson (BF)
                if cl_results: # Only run BF if CL was successful
                    log_message("Generating A Priori Estimates for BF...")
                    apriori_estimates = actuar.generate_apriori_estimates(
                        latest_paid=cl_results['latest_paid'],
                        cdfs=cl_results['cdfs'],
                        method='cl_ultimate_proxy',
                        cl_ultimate_losses=cl_results['ultimate_losses']
                    )
                    log_message("A Priori Estimates Generated.")

                    log_message("Calculating Bornhuetter-Ferguson...")
                    bf_calc_results = actuar.calculate_bornhuetter_ferguson(
                        cumulative_triangle=cumulative_paid_triangle,
                        cdfs=cl_results['cdfs'],
                        apriori_ultimate_losses=apriori_estimates
                    )
                    st.session_state.bf_results = bf_calc_results # Store BF results
                    log_message("Bornhuetter-Ferguson calculation complete.")
                else:
                    log_message("Skipping Bornhuetter-Ferguson due to Chain Ladder failure.")
                    st.session_state.bf_results = None

                log_message("Analysis complete. Displaying results.")
                st.success("Analysis Complete! (ניתוח הושלם!)")
            else:
                log_message("Failed to generate claims data based on input parameters. Cannot proceed.")
                st.error("Failed to generate claims data based on input parameters. Cannot proceed. (כשל ביצירת נתונים)")
                st.session_state.results = None
                st.session_state.anomalies = None
                st.session_state.bf_results = None

        except Exception as e:
             log_message(f"ERROR during analysis: {e}")
             st.error(f"An error occurred during the analysis: {e}")
             # Clear potentially partial results
             st.session_state.results = None
             st.session_state.anomalies = None
             st.session_state.bf_results = None

        # Display final log messages in the placeholder
        log_placeholder.text_area("Calculation Log (יומן חישובים)", value="\n".join(st.session_state.log_messages), height=200)

# Display results if they exist in session state
if st.session_state.results:
    results = st.session_state.results
    anomalies = st.session_state.anomalies
    bf_results = st.session_state.bf_results # Get BF results

    st.markdown("--- --- ---") # Separator
    st.header("Summary (סיכום)")

    # Use columns for summary metrics
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        # Display Total IBNR as a metric (Chain Ladder)
        total_ibnr_cl = results.get('total_ibnr', 0)
        st.metric("Total CL IBNR (סך תתו\"ע לפי שרשרת)", f"${total_ibnr_cl:,.0f}")
    with col_sum2:
        # Display Total IBNR as a metric (Bornhuetter-Ferguson)
        if bf_results:
            total_ibnr_bf = bf_results.get('total_bf_ibnr', 0)
            st.metric("Total BF IBNR (סך תתו\"ע לפי ב\"פ)", f"${total_ibnr_bf:,.0f}")
        else:
            st.metric("Total BF IBNR (סך תתו\"ע לפי ב\"פ)", "N/A")


    st.markdown("--- --- ---") # Separator
    st.header("Detailed Results (תוצאות מפורטות)")

    # Use tabs for different results sections
    tab_list = [
        "Cumulative Triangle (משולש מצטבר)",
        "Projected Triangle (CL) (משולש חזוי)",
        "ATA Factors (מקדמי התפתחות)",
        "CDFs (מקדמי התפתחות מצטברים)",
        "CL Ultimates & IBNR (נזק סופי ותתו\"ע - שרשרת)",
        "BF Ultimates & IBNR (נזק סופי ותתו\"ע - ב\"פ)", # New Tab for BF
        "Anomalies (חריגים)"
    ]
    tabs = st.tabs(tab_list)

    # Assign tabs dynamically
    tab_cl_tri, tab_cl_proj, tab_ata, tab_cdf, tab_cl_ult, tab_bf_ult, tab_anom = tabs

    with tab_cl_tri:
        st.subheader("Cumulative Paid Triangle (משולש תשלומים מצטבר)")
        st.dataframe(results['cumulative_triangle'].style.format("{:,.0f}", na_rep=""))

    with tab_cl_proj:
        st.subheader("Projected Cumulative Triangle (CL) (משולש חזוי מצטבר - שרשרת)")
        st.dataframe(results['projected_triangle'].style.format("{:,.0f}", na_rep="-"))

    with tab_ata:
        st.subheader("Selected Age-to-Age (ATA) Factors (מקדמי התפתחות נבחרים)")
        st.dataframe(results['ata_factors_selected'].style.format("{:.4f}"))
        # Optionally display individual factors (can be large)
        # st.subheader("Individual ATA Factors (by Accident Year)")
        # st.write(results['all_ata_factors']) # Might be better formatted

    with tab_cdf:
        st.subheader("Cumulative Development Factors (CDFs to Ultimate) (מקדמי התפתחות מצטברים לנזק סופי)")
        st.dataframe(results['cdfs'].to_frame().style.format("{:.4f}")) # Use to_frame for better display

    with tab_cl_ult:
        st.subheader("Chain Ladder Ultimate Loss and IBNR Estimate by Accident Year (נזק סופי ותתו\"ע לפי שנת אירוע - שרשרת)")
        # Recreate summary_df to include diagnostics if they exist
        cl_summary_data = {
            'Latest Paid (תשלום אחרון)': results['latest_paid'],
            'CL Ultimate Loss (נזק סופי - שרשרת)': results['ultimate_losses'],
            'CL IBNR Estimate (תתו\"ע - שרשרת)': results['ibnr']
        }
        # Add diagnostics if they were calculated and stored
        if 'ibnr_pct_ultimate' in results:
            cl_summary_data['IBNR % Ultimate (תתו\"ע % נזק סופי)'] = results['ibnr_pct_ultimate']
        if 'ibnr_pct_paid' in results:
             cl_summary_data['IBNR % Paid (תתו\"ע % תשלום נוכחי)'] = results['ibnr_pct_paid']

        summary_df_cl = pd.DataFrame(cl_summary_data)
        st.dataframe(summary_df_cl.style.format("{:,.0f}", na_rep="").format({ # Apply multiple formats
              'IBNR % Ultimate (תתו\"ע % נזק סופי)': "{:.1%}",
              'IBNR % Paid (תתו\"ע % תשלום נוכחי)': "{:.1%}"
        }, na_rep="-"))


        st.subheader("Charts (תרשימים) - Chain Ladder")
        # Use columns for side-by-side charts
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(summary_df_cl[['Latest Paid (תשלום אחרון)', 'CL Ultimate Loss (נזק סופי - שרשרת)']])
            st.caption("Latest Paid vs. CL Ultimate (תשלום אחרון מול נזק סופי - שרשרת)")
        with col2:
            st.bar_chart(summary_df_cl['CL IBNR Estimate (תתו\"ע - שרשרת)'])
            st.caption("CL IBNR Estimate by Accident Year (תתו\"ע לפי שנת אירוע - שרשרת)")

    with tab_bf_ult: # Display BF Results
        st.subheader("Bornhuetter-Ferguson Ultimate Loss and IBNR Estimate by Accident Year (נזק סופי ותתו\"ע לפי שנת אירוע - ב\"פ)")
        if bf_results:
            bf_summary_df = pd.DataFrame({
                 'Latest Paid (תשלום אחרון)': results['latest_paid'], # Use latest paid from CL
                 'A Priori Ultimate (אומדן אפריורי)': bf_results['apriori_ultimate_losses'],
                 '% Unreported (אחוז לא מדווח)': bf_results['pct_unreported'],
                 'BF Ultimate Loss (נזק סופי - ב\"פ)': bf_results['bf_ultimate_losses'],
                 'BF IBNR Estimate (תתו\"ע - ב\"פ)': bf_results['bf_ibnr']
            })
            st.dataframe(bf_summary_df.style.format("{:,.0f}", na_rep="-").format({
                 '% Unreported (אחוז לא מדווח)': '{:.1%}'
            }))

            st.subheader("Charts (תרשימים) - Bornhuetter-Ferguson")
            col_bf1, col_bf2 = st.columns(2)
            with col_bf1:
                 st.line_chart(bf_summary_df[['Latest Paid (תשלום אחרון)', 'BF Ultimate Loss (נזק סופי - ב\"פ)']])
                 st.caption("Latest Paid vs. BF Ultimate (תשלום אחרון מול נזק סופי - ב\"פ)")
            with col_bf2:
                 st.bar_chart(bf_summary_df['BF IBNR Estimate (תתו\"ע - ב\"פ)'])
                 st.caption("BF IBNR Estimate by Accident Year (תתו\"ע לפי שנת אירוע - ב\"פ)")

        else:
            st.info("Bornhuetter-Ferguson results are not available.")


    with tab_anom:
        st.subheader("Potential Factor Anomalies (מקדמים חריגים אפשריים)")
        st.write(f"Based on Z-Score threshold (סף ציון תקן): > {anomaly_threshold}")
        if anomalies is not None and not anomalies.empty:
            st.dataframe(anomalies.style.format({
                "Factor (מקדם)": "{:.4f}",
                "Z-Score (ציון תקן)": "{:.2f}",
                "Mean Factor (Lag) (מקדם ממוצע לפיגור)": "{:.4f}",
                "Std Dev (Lag) (סטיית תקן לפיגור)": "{:.4f}"
            }))
        else:
            st.write("No significant anomalies detected at this threshold. (לא זוהו חריגות משמעותיות בסף זה)")

else:
    st.info("Click 'Run Analysis' in the sidebar to generate and view results. (לחץ על 'הרץ ניתוח' בסרגל הצד כדי להפיק ולצפות בתוצאות)") 