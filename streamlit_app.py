import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Experiment Impact Summary Report")
st.write("### Key Business Metrics Summary")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("1MG_Test_and_control_report_transformed (2).csv", parse_dates=['date'])
    return df

df = load_data()

# Ensure necessary columns exist
required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
has_recency_data = 'Recency' in df.columns

if not required_columns.issubset(df.columns):
    st.write("Missing required columns in the CSV file.")
    st.stop()

# Define control group and test groups
control_group = "Control Set"
test_groups = [g for g in df['data_set'].unique() if g != control_group]

# Test start dates
test_start_dates = {
    "resp": pd.Timestamp("2025-03-05"),
    "cardiac": pd.Timestamp("2025-03-18"),
    "diabetic": pd.Timestamp("2025-03-06"),
    "derma": pd.Timestamp("2025-03-18")
}

# Cohort selection with default "All Cohorts" option
cohort_options = ["All Cohorts"] + list(df['cohort'].unique())
selected_cohort = st.sidebar.selectbox("Select Cohort", cohort_options)

# Recency selector with default "All Recency" option
if has_recency_data:
    recency_values = sorted(df['Recency'].unique())
    recency_options = ["All Recency"] + list(recency_values)
    selected_recency = st.sidebar.selectbox("Select Recency", recency_options)
else:
    selected_recency = "All Recency"

# Filter data based on selections
def filter_data(df, cohort, recency):
    # Start with base filtered data
    if cohort != "All Cohorts":
        filtered_df = df[df['cohort'] == cohort]
        start_date = test_start_dates.get(cohort, df['date'].min())
    else:
        filtered_df = df.copy()
        # For all cohorts, use the earliest test start date
        start_date = min([date for date in test_start_dates.values()])
    
    filtered_df = filtered_df[filtered_df['date'] >= start_date]
    
    # Apply recency filter if needed
    if recency != "All Recency" and has_recency_data:
        filtered_df = filtered_df[filtered_df['Recency'] == recency]
    
    return filtered_df

filtered_df = filter_data(df, selected_cohort, selected_recency)

# Calculate total metrics and extrapolations
st.write(f"## Impact Summary for {selected_cohort if selected_cohort != 'All Cohorts' else 'All Cohorts'}")
if selected_recency != "All Recency":
    st.write(f"### Recency: {selected_recency}")

# Initialize metrics dictionary
metrics_summary = {}

# Function to calculate metrics for a cohort
def calculate_cohort_metrics(df, cohort):
    if cohort != "All Cohorts":
        cohort_df = df[df['cohort'] == cohort]
        cohort_start_date = test_start_dates.get(cohort, df['date'].min())
        cohort_df = cohort_df[cohort_df['date'] >= cohort_start_date]
    else:
        cohort_df = df.copy()
    
    # Get control and test data
    control_df = cohort_df[cohort_df['data_set'] == control_group]
    test_dfs = {test_group: cohort_df[cohort_df['data_set'] == test_group] for test_group in test_groups if test_group in cohort_df['data_set'].unique()}
    
    metrics = {}
    
    # Calculate metrics for each test group against control
    for test_group, test_df in test_dfs.items():
        # Get dates that exist in both test and control
        common_dates = set(test_df['date']).intersection(set(control_df['date']))
        if not common_dates:
            continue
            
        test_df_filtered = test_df[test_df['date'].isin(common_dates)]
        control_df_filtered = control_df[control_df['date'].isin(common_dates)]
        
        # Calculate metrics - using 70/30 split and extrapolation as requested
        # Total raw metrics
        test_total_gmv = test_df_filtered['gmv'].sum()
        control_total_gmv = control_df_filtered['gmv'].sum()
        test_total_app_opens = test_df_filtered['app_opens'].sum()
        control_total_app_opens = control_df_filtered['app_opens'].sum()
        test_total_orders = test_df_filtered['orders'].sum()
        control_total_orders = control_df_filtered['orders'].sum()
        test_total_transactors = test_df_filtered['transactors'].sum()
        control_total_transactors = control_df_filtered['transactors'].sum()
        
        # Adjusting for 70/30 split and calculating incremental impact
        # Formula: (((total_metric_test_group)*100/70)-((total_metric_control_group)*100/30))
        incremental_gmv = ((test_total_gmv*100/70) - (control_total_gmv*100/30))
        incremental_app_opens = ((test_total_app_opens*100/70) - (control_total_app_opens*100/30))
        incremental_orders = ((test_total_orders*100/70) - (control_total_orders*100/30))
        incremental_transactors = ((test_total_transactors*100/70) - (control_total_transactors*100/30))
        
        # Calculate percentage lifts
        if control_total_gmv > 0 and control_df_filtered['audience_size'].sum() > 0 and test_df_filtered['audience_size'].sum() > 0:
            gmv_lift_percent = ((test_total_gmv/test_df_filtered['audience_size'].sum()) - 
                               (control_total_gmv/control_df_filtered['audience_size'].sum())) / \
                               (control_total_gmv/control_df_filtered['audience_size'].sum()) * 100
        else:
            gmv_lift_percent = 0
            
        if control_total_app_opens > 0 and control_df_filtered['audience_size'].sum() > 0 and test_df_filtered['audience_size'].sum() > 0:
            app_opens_lift_percent = ((test_total_app_opens/test_df_filtered['audience_size'].sum()) - 
                                     (control_total_app_opens/control_df_filtered['audience_size'].sum())) / \
                                     (control_total_app_opens/control_df_filtered['audience_size'].sum()) * 100
        else:
            app_opens_lift_percent = 0
            
        if control_total_orders > 0 and control_df_filtered['audience_size'].sum() > 0 and test_df_filtered['audience_size'].sum() > 0:
            orders_lift_percent = ((test_total_orders/test_df_filtered['audience_size'].sum()) - 
                                  (control_total_orders/control_df_filtered['audience_size'].sum())) / \
                                  (control_total_orders/control_df_filtered['audience_size'].sum()) * 100
        else:
            orders_lift_percent = 0
            
        if control_total_transactors > 0 and control_df_filtered['audience_size'].sum() > 0 and test_df_filtered['audience_size'].sum() > 0:
            transactors_lift_percent = ((test_total_transactors/test_df_filtered['audience_size'].sum()) - 
                                      (control_total_transactors/control_df_filtered['audience_size'].sum())) / \
                                      (control_total_transactors/control_df_filtered['audience_size'].sum()) * 100
        else:
            transactors_lift_percent = 0
        
        # Total audience sizes
        test_audience = test_df_filtered['audience_size'].sum()
        control_audience = control_df_filtered['audience_size'].sum()
        
        # Store metrics
        metrics[test_group] = {
            'test_total_gmv': test_total_gmv,
            'control_total_gmv': control_total_gmv,
            'incremental_gmv': incremental_gmv,
            'gmv_lift_percent': gmv_lift_percent,
            
            'test_total_app_opens': test_total_app_opens,
            'control_total_app_opens': control_total_app_opens,
            'incremental_app_opens': incremental_app_opens,
            'app_opens_lift_percent': app_opens_lift_percent,
            
            'test_total_orders': test_total_orders,
            'control_total_orders': control_total_orders,
            'incremental_orders': incremental_orders,
            'orders_lift_percent': orders_lift_percent,
            
            'test_total_transactors': test_total_transactors,
            'control_total_transactors': control_total_transactors,
            'incremental_transactors': incremental_transactors,
            'transactors_lift_percent': transactors_lift_percent,
            
            'test_audience': test_audience,
            'control_audience': control_audience,
            'test_dates': len(common_dates)
        }
    
    return metrics

# Calculate metrics based on selections
if selected_cohort == "All Cohorts":
    # Calculate for each cohort individually
    for cohort in df['cohort'].unique():
        cohort_df = filtered_df[filtered_df['cohort'] == cohort]
        if len(cohort_df) > 0:
            metrics_summary[cohort] = calculate_cohort_metrics(cohort_df, cohort)
else:
    # Calculate for selected cohort
    metrics_summary[selected_cohort] = calculate_cohort_metrics(filtered_df, selected_cohort)

# Display results in a nice format
st.write("## 🚀 Business Impact Summary")

# Summary table
summary_rows = []

for cohort, test_groups_metrics in metrics_summary.items():
    for test_group, metrics in test_groups_metrics.items():
        summary_rows.append({
            'Cohort': cohort,
            'Test Group': test_group,
            'Test Duration (Days)': metrics['test_dates'],
            'Incremental GMV (₹)': round(metrics['incremental_gmv'], 2),
            'GMV Lift (%)': round(metrics['gmv_lift_percent'], 2),
            'Incremental App Opens': round(metrics['incremental_app_opens'], 0),
            'App Opens Lift (%)': round(metrics['app_opens_lift_percent'], 2),
            'Incremental Orders': round(metrics['incremental_orders'], 0),
            'Orders Lift (%)': round(metrics['orders_lift_percent'], 2),
            'Incremental Transactors': round(metrics['incremental_transactors'], 0),
            'Transactors Lift (%)': round(metrics['transactors_lift_percent'], 2)
        })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    
    # Just display the dataframe without applying style
    st.dataframe(summary_df)
    
    # Create visualizations
    st.write("## 📊 Visual Impact Analysis")
    
    # Bar chart for GMV Impact by Cohort
    fig_gmv = px.bar(summary_df, x='Cohort', y='Incremental GMV (₹)', 
                     color='Test Group', barmode='group',
                     title="Incremental GMV by Cohort")
    st.plotly_chart(fig_gmv, use_container_width=True)
    
    # Bar chart for App Opens Impact
    fig_opens = px.bar(summary_df, x='Cohort', y='Incremental App Opens', 
                       color='Test Group', barmode='group',
                       title="Incremental App Opens by Cohort")
    st.plotly_chart(fig_opens, use_container_width=True)
    
    # Lift comparison
    lift_data = summary_df.melt(id_vars=['Cohort', 'Test Group'], 
                               value_vars=['GMV Lift (%)', 'App Opens Lift (%)', 
                                          'Orders Lift (%)', 'Transactors Lift (%)'],
                               var_name='Metric', value_name='Lift Percentage')
    
    fig_lift = px.bar(lift_data, x='Metric', y='Lift Percentage',
                     color='Cohort', barmode='group', facet_col='Test Group',
                     title="Lift Percentage by Metric and Test Group")
    st.plotly_chart(fig_lift, use_container_width=True)
    
    # Annualized projections
    st.write("## 📈 Annual Projections")
    
    # Calculate annualized impact (multiplying by 365/test_days)
    annual_rows = []
    total_annual_gmv = 0
    
    for row in summary_rows:
        days = row['Test Duration (Days)']
        if days > 0:
            annual_multiplier = 365 / days
            annual_projected_gmv = row['Incremental GMV (₹)'] * annual_multiplier
            annual_projected_app_opens = row['Incremental App Opens'] * annual_multiplier
            annual_projected_orders = row['Incremental Orders'] * annual_multiplier
            annual_projected_transactors = row['Incremental Transactors'] * annual_multiplier
            
            total_annual_gmv += annual_projected_gmv
            
            annual_rows.append({
                'Cohort': row['Cohort'],
                'Test Group': row['Test Group'],
                'Annual Projected GMV (₹)': round(annual_projected_gmv, 2),
                'Annual Projected App Opens': int(annual_projected_app_opens),
                'Annual Projected Orders': int(annual_projected_orders),
                'Annual Projected Transactors': int(annual_projected_transactors)
            })
    
    if annual_rows:
        annual_df = pd.DataFrame(annual_rows)
        st.dataframe(annual_df)
        
        # Total impact across all cohorts
        st.metric("Total Annual Projected Incremental GMV", f"₹{total_annual_gmv:,.2f}")
    
    # ROI calculation if we have campaign cost data
    st.write("## 💰 Return on Investment")
    
    campaign_cost = st.number_input("Enter total campaign cost (₹)", min_value=0.0, step=1000.0)
    
    if campaign_cost > 0:
        roi = (total_annual_gmv - campaign_cost) / campaign_cost * 100
        st.metric("Projected Annual ROI", f"{roi:.2f}%")
        
        # Payback period in days
        if total_annual_gmv > 0:
            daily_gmv = total_annual_gmv / 365
            payback_days = campaign_cost / daily_gmv if daily_gmv > 0 else float('inf')
            
            if payback_days != float('inf'):
                st.metric("Estimated Payback Period", f"{payback_days:.0f} days")
            else:
                st.metric("Estimated Payback Period", "N/A")
else:
    st.write("No data available for the selected filters.")

# If recency data is available, add recency-specific analysis
if has_recency_data and selected_cohort != "All Cohorts" and selected_recency == "All Recency":
    st.write("## 📅 Recency Segment Analysis")
    
    # Calculate metrics for each recency segment
    recency_metrics = {}
    
    for recency in df['Recency'].unique():
        recency_df = filtered_df[filtered_df['Recency'] == recency]
        if len(recency_df) > 0:
            recency_metrics[recency] = calculate_cohort_metrics(recency_df, selected_cohort)
    
    # Prepare data for visualization
    recency_rows = []
    
    for recency, test_groups_metrics in recency_metrics.items():
        for test_group, metrics in test_groups_metrics.items():
            recency_rows.append({
                'Recency': recency,
                'Test Group': test_group,
                'GMV Lift (%)': metrics['gmv_lift_percent'],
                'App Opens Lift (%)': metrics['app_opens_lift_percent'],
                'Orders Lift (%)': metrics['orders_lift_percent'],
                'Transactors Lift (%)': metrics['transactors_lift_percent']
            })
    
    if recency_rows:
        recency_df = pd.DataFrame(recency_rows)
        
        # Create heatmap for lift by recency segment
        for metric in ['GMV Lift (%)', 'App Opens Lift (%)', 'Orders Lift (%)', 'Transactors Lift (%)']:
            # Reshape data for heatmap
            pivot_df = recency_df.pivot(index='Recency', columns='Test Group', values=metric)
            if not pivot_df.empty:
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    colorscale='RdBu_r',
                    zmid=0,  # Center colorscale at zero
                    text=[[f"{val:.2f}%" for val in row] for row in pivot_df.values],
                    texttemplate="%{text}",
                    textfont={"size":12}
                ))
                
                fig.update_layout(
                    title=f"{metric} by Recency Segment",
                    xaxis_title="Test Group",
                    yaxis_title="Recency"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Download button for CSV export
csv = None
if 'summary_df' in locals():
    csv = summary_df.to_csv(index=False).encode('utf-8')
    
if csv:
    st.download_button(
        label="Download Full Report as CSV",
        data=csv,
        file_name="experiment_impact_summary.csv",
        mime="text/csv",
    )

# Executive summary
st.write("## 📋 Executive Summary")

if 'summary_df' in locals() and not summary_df.empty:
    # Calculate overall metrics
    total_inc_gmv = summary_df['Incremental GMV (₹)'].sum()
    avg_gmv_lift = summary_df['GMV Lift (%)'].mean()
    total_inc_opens = int(summary_df['Incremental App Opens'].sum())
    avg_opens_lift = summary_df['App Opens Lift (%)'].mean()
    
    # Find best performing cohort
    if len(summary_df) > 0:
        best_cohort_idx = summary_df['GMV Lift (%)'].idxmax()
        if best_cohort_idx is not None:
            best_cohort = summary_df.loc[best_cohort_idx, 'Cohort']
            best_cohort_lift = summary_df.loc[best_cohort_idx, 'GMV Lift (%)']
            
            # Generate executive summary
            st.markdown(f"""
            Based on our experiment results, we have successfully demonstrated significant positive impact:
            
            - **Total Incremental GMV**: ₹{total_inc_gmv:,.2f}
            - **Average GMV Lift**: {avg_gmv_lift:.2f}%
            - **Total Incremental App Opens**: {total_inc_opens:,}
            - **Average App Opens Lift**: {avg_opens_lift:.2f}%
            
            The **{best_cohort}** cohort showed the strongest performance with a GMV lift of **{best_cohort_lift:.2f}%**.
            
            If projected annually, this initiative could generate approximately **₹{total_annual_gmv:,.2f}** in incremental GMV.
            """)
        else:
            st.write("No best cohort found in the data.")
    else:
        st.write("No data available for generating an executive summary based on current selections.")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# This code should be inserted after the existing code in your Streamlit app
# Add it at the end of the file

# Pre vs Post Test Analysis Section
st.write("## 🔄 Pre vs Post Test Analysis (Test Groups Only)")
st.write("This analysis compares test group performance before and after the test implementation, ignoring control groups.")

# Function to calculate pre vs post metrics for a test group
def calculate_pre_post_metrics(df, cohort, pre_days=7):
    """Calculate metrics for test groups before and after test implementation"""
    if cohort != "All Cohorts":
        cohort_df = df[df['cohort'] == cohort]
        start_date = test_start_dates.get(cohort, df['date'].min())
    else:
        cohort_df = df.copy()
        # For all cohorts, use the earliest test start date
        start_date = min([date for date in test_start_dates.values()])
    
    # Only include test groups, exclude control
    test_df = cohort_df[cohort_df['data_set'] != control_group]
    
    # Determine pre-period (n days before test start)
    pre_start_date = start_date - timedelta(days=pre_days)
    pre_end_date = start_date - timedelta(days=1)
    
    # Filter data for pre and post periods
    pre_df = test_df[(test_df['date'] >= pre_start_date) & (test_df['date'] <= pre_end_date)]
    post_df = test_df[test_df['date'] >= start_date]
    
    pre_post_metrics = {}
    
    # Calculate metrics for each test group
    for test_group in test_df['data_set'].unique():
        test_group_pre = pre_df[pre_df['data_set'] == test_group]
        test_group_post = post_df[post_df['data_set'] == test_group]
        
        # Skip if we don't have data for either period
        if len(test_group_pre) == 0 or len(test_group_post) == 0:
            continue
        
        # Calculate average daily metrics for both periods
        pre_days_count = (test_group_pre['date'].max() - test_group_pre['date'].min()).days + 1
        post_days_count = (test_group_post['date'].max() - test_group_post['date'].min()).days + 1
        
        # Avoid division by zero
        pre_days_count = max(pre_days_count, 1)
        post_days_count = max(post_days_count, 1)
        
        # Pre-period metrics (daily average)
        pre_gmv = test_group_pre['gmv'].sum() / pre_days_count
        pre_app_opens = test_group_pre['app_opens'].sum() / pre_days_count
        pre_orders = test_group_pre['orders'].sum() / pre_days_count
        pre_transactors = test_group_pre['transactors'].sum() / pre_days_count
        
        # Post-period metrics (daily average)
        post_gmv = test_group_post['gmv'].sum() / post_days_count
        post_app_opens = test_group_post['app_opens'].sum() / post_days_count
        post_orders = test_group_post['orders'].sum() / post_days_count
        post_transactors = test_group_post['transactors'].sum() / post_days_count
        
        # Calculate percentage changes
        gmv_pct_change = ((post_gmv - pre_gmv) / pre_gmv * 100) if pre_gmv > 0 else float('inf')
        app_opens_pct_change = ((post_app_opens - pre_app_opens) / pre_app_opens * 100) if pre_app_opens > 0 else float('inf')
        orders_pct_change = ((post_orders - pre_orders) / pre_orders * 100) if pre_orders > 0 else float('inf')
        transactors_pct_change = ((post_transactors - pre_transactors) / pre_transactors * 100) if pre_transactors > 0 else float('inf')
        
        # Calculate absolute increases
        gmv_increase = post_gmv - pre_gmv
        app_opens_increase = post_app_opens - pre_app_opens
        orders_increase = post_orders - pre_orders
        transactors_increase = post_transactors - pre_transactors
        
        # Store metrics
        pre_post_metrics[test_group] = {
            'pre_gmv': pre_gmv,
            'post_gmv': post_gmv,
            'gmv_increase': gmv_increase,
            'gmv_pct_change': gmv_pct_change,
            
            'pre_app_opens': pre_app_opens,
            'post_app_opens': post_app_opens,
            'app_opens_increase': app_opens_increase,
            'app_opens_pct_change': app_opens_pct_change,
            
            'pre_orders': pre_orders,
            'post_orders': post_orders,
            'orders_increase': orders_increase,
            'orders_pct_change': orders_pct_change,
            
            'pre_transactors': pre_transactors,
            'post_transactors': post_transactors,
            'transactors_increase': transactors_increase,
            'transactors_pct_change': transactors_pct_change,
            
            'pre_days': pre_days_count,
            'post_days': post_days_count
        }
    
    return pre_post_metrics

# Set the number of days to look back for pre-test analysis
pre_test_days = st.slider("Number of days before test to analyze (Pre-period)", min_value=3, max_value=30, value=7)

# Calculate pre-post metrics based on selections
pre_post_summary = {}

if selected_cohort == "All Cohorts":
    # Calculate for each cohort individually
    for cohort in df['cohort'].unique():
        cohort_df = filtered_df[filtered_df['cohort'] == cohort]
        if len(cohort_df) > 0:
            pre_post_summary[cohort] = calculate_pre_post_metrics(cohort_df, cohort, pre_test_days)
else:
    # Calculate for selected cohort
    pre_post_summary[selected_cohort] = calculate_pre_post_metrics(filtered_df, selected_cohort, pre_test_days)

# Display results in a nice format
st.write("### 📊 Pre vs Post Test Metrics Summary (Daily Average)")

# Summary table
pre_post_rows = []

for cohort, test_groups_metrics in pre_post_summary.items():
    for test_group, metrics in test_groups_metrics.items():
        # Handle infinite percentage changes
        for metric in ['gmv_pct_change', 'app_opens_pct_change', 'orders_pct_change', 'transactors_pct_change']:
            if metrics[metric] == float('inf'):
                metrics[metric] = "N/A (zero in pre-period)"
            elif isinstance(metrics[metric], (int, float)):
                # Keep the value as is
                pass
    
        pre_post_rows.append({
            'Cohort': cohort,
            'Test Group': test_group,
            'Pre-Period (Days)': metrics['pre_days'],
            'Post-Period (Days)': metrics['post_days'],
            'Pre GMV (Daily Avg ₹)': round(metrics['pre_gmv'], 2),
            'Post GMV (Daily Avg ₹)': round(metrics['post_gmv'], 2),
            'GMV Change (%)': metrics['gmv_pct_change'] if isinstance(metrics['gmv_pct_change'], str) else round(metrics['gmv_pct_change'], 2),
            'Pre App Opens (Daily Avg)': round(metrics['pre_app_opens'], 1),
            'Post App Opens (Daily Avg)': round(metrics['post_app_opens'], 1),
            'App Opens Change (%)': metrics['app_opens_pct_change'] if isinstance(metrics['app_opens_pct_change'], str) else round(metrics['app_opens_pct_change'], 2),
            'Pre Orders (Daily Avg)': round(metrics['pre_orders'], 1),
            'Post Orders (Daily Avg)': round(metrics['post_orders'], 1),
            'Orders Change (%)': metrics['orders_pct_change'] if isinstance(metrics['orders_pct_change'], str) else round(metrics['orders_pct_change'], 2),
            'Pre Transactors (Daily Avg)': round(metrics['pre_transactors'], 1),
            'Post Transactors (Daily Avg)': round(metrics['post_transactors'], 1),
            'Transactors Change (%)': metrics['transactors_pct_change'] if isinstance(metrics['transactors_pct_change'], str) else round(metrics['transactors_pct_change'], 2)
        })

if pre_post_rows:
    pre_post_df = pd.DataFrame(pre_post_rows)
    
    # Just display the dataframe
    st.dataframe(pre_post_df)
    
    # Create visualizations
    st.write("### 📈 Pre vs Post Test Visual Comparison")
    
    # Prepare data for visualization - melt to long format
    metrics_to_plot = [
        ('GMV (Daily Avg ₹)', 'Pre GMV (Daily Avg ₹)', 'Post GMV (Daily Avg ₹)'),
        ('App Opens (Daily Avg)', 'Pre App Opens (Daily Avg)', 'Post App Opens (Daily Avg)'),
        ('Orders (Daily Avg)', 'Pre Orders (Daily Avg)', 'Post Orders (Daily Avg)'),
        ('Transactors (Daily Avg)', 'Pre Transactors (Daily Avg)', 'Post Transactors (Daily Avg)')
    ]
    
    for metric_name, pre_col, post_col in metrics_to_plot:
        # Create a visualization for each metric
        viz_data = []
        
        for _, row in pre_post_df.iterrows():
            viz_data.append({
                'Cohort': row['Cohort'],
                'Test Group': row['Test Group'],
                'Period': 'Pre-Test',
                metric_name: row[pre_col]
            })
            viz_data.append({
                'Cohort': row['Cohort'],
                'Test Group': row['Test Group'],
                'Period': 'Post-Test',
                metric_name: row[post_col]
            })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create grouped bar chart
        fig = px.bar(
            viz_df, 
            x='Test Group', 
            y=metric_name,
            color='Period',
            barmode='group',
            facet_col='Cohort' if len(pre_post_df['Cohort'].unique()) > 1 else None,
            title=f"Pre vs Post {metric_name} by Test Group",
            color_discrete_map={'Pre-Test': '#636EFA', 'Post-Test': '#EF553B'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Percentage change visualization
    change_metrics = [
        ('GMV Change (%)', 'GMV'),
        ('App Opens Change (%)', 'App Opens'),
        ('Orders Change (%)', 'Orders'),
        ('Transactors Change (%)', 'Transactors')
    ]
    
    # Prepare data for visualization - remove non-numeric values
    change_data = []
    for _, row in pre_post_df.iterrows():
        for col, label in change_metrics:
            if isinstance(row[col], (int, float)):
                change_data.append({
                    'Cohort': row['Cohort'],
                    'Test Group': row['Test Group'],
                    'Metric': label,
                    'Percentage Change': row[col]
                })
    
    if change_data:
        change_df = pd.DataFrame(change_data)
        
        # Create grouped bar chart for percentage changes
        fig_change = px.bar(
            change_df, 
            x='Metric', 
            y='Percentage Change',
            color='Test Group',
            barmode='group',
            facet_col='Cohort' if len(pre_post_df['Cohort'].unique()) > 1 else None,
            title="Percentage Change in Metrics (Pre vs Post)",
            text_auto='.1f'
        )
        
        # Add a horizontal line at y=0
        fig_change.add_shape(
            type="line",
            x0=-0.5,
            x1=3.5,
            y0=0,
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        st.plotly_chart(fig_change, use_container_width=True)
    
    # Download button for CSV export
    pre_post_csv = pre_post_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Pre vs Post Analysis as CSV",
        data=pre_post_csv,
        file_name="pre_post_test_analysis.csv",
        mime="text/csv",
    )
    
    # Summary insights
    st.write("### 💡 Key Insights from Pre vs Post Analysis")
    
    # Extract key insights
    insights = []
    
    for cohort, test_groups_metrics in pre_post_summary.items():
        for test_group, metrics in test_groups_metrics.items():
            # Check for significant changes (>10% change)
            significant_metrics = []
            
            if isinstance(metrics['gmv_pct_change'], (int, float)) and abs(metrics['gmv_pct_change']) > 10:
                direction = "increase" if metrics['gmv_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"GMV showed a {abs(metrics['gmv_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['app_opens_pct_change'], (int, float)) and abs(metrics['app_opens_pct_change']) > 10:
                direction = "increase" if metrics['app_opens_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"App Opens showed a {abs(metrics['app_opens_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['orders_pct_change'], (int, float)) and abs(metrics['orders_pct_change']) > 10:
                direction = "increase" if metrics['orders_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"Orders showed a {abs(metrics['orders_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['transactors_pct_change'], (int, float)) and abs(metrics['transactors_pct_change']) > 10:
                direction = "increase" if metrics['transactors_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"Transactors showed a {abs(metrics['transactors_pct_change']):.1f}% {direction}")
            
            if significant_metrics:
                insights.append(f"**{cohort} - {test_group}**: {'; '.join(significant_metrics)}.")
    
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.write("No significant changes detected between pre and post periods.")
else:
    st.write("No data available for pre vs post analysis with the selected filters.")
