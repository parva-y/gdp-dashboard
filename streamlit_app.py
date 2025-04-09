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
        if control_total_gmv > 0:
            gmv_lift_percent = ((test_total_gmv/test_df_filtered['audience_size'].sum()) - 
                               (control_total_gmv/control_df_filtered['audience_size'].sum())) / \
                               (control_total_gmv/control_df_filtered['audience_size'].sum()) * 100
        else:
            gmv_lift_percent = 0
            
        if control_total_app_opens > 0:
            app_opens_lift_percent = ((test_total_app_opens/test_df_filtered['audience_size'].sum()) - 
                                     (control_total_app_opens/control_df_filtered['audience_size'].sum())) / \
                                     (control_total_app_opens/control_df_filtered['audience_size'].sum()) * 100
        else:
            app_opens_lift_percent = 0
            
        if control_total_orders > 0:
            orders_lift_percent = ((test_total_orders/test_df_filtered['audience_size'].sum()) - 
                                  (control_total_orders/control_df_filtered['audience_size'].sum())) / \
                                  (control_total_orders/control_df_filtered['audience_size'].sum()) * 100
        else:
            orders_lift_percent = 0
            
        if control_total_transactors > 0:
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
    
    # Add styling to highlight positive values
    def style_positive_values(val):
        if isinstance(val, (int, float)):
            if 'Lift' in val.name and val > 0:
                return 'color: green; font-weight: bold'
            elif 'Incremental' in val.name and val > 0:
                return 'color: green; font-weight: bold'
        return ''
    
    st.dataframe(summary_df.style.applymap(style_positive_values))
    
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
    for row in summary_rows:
        days = row['Test Duration (Days)']
        if days > 0:
            annual_multiplier = 365 / days
            row['Annual Projected GMV (₹)'] = row['Incremental GMV (₹)'] * annual_multiplier
            row['Annual Projected App Opens'] = row['Incremental App Opens'] * annual_multiplier
            row['Annual Projected Orders'] = row['Incremental Orders'] * annual_multiplier
            row['Annual Projected Transactors'] = row['Incremental Transactors'] * annual_multiplier
    
    annual_df = pd.DataFrame(summary_rows)[['Cohort', 'Test Group', 'Annual Projected GMV (₹)', 
                                           'Annual Projected App Opens', 'Annual Projected Orders',
                                           'Annual Projected Transactors']]
    
    # Format numbers for better readability
    for col in annual_df.columns:
        if 'GMV' in col:
            annual_df[col] = annual_df[col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else x)
        elif any(metric in col for metric in ['Opens', 'Orders', 'Transactors']):
            annual_df[col] = annual_df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else x)
    
    st.dataframe(annual_df)
    
    # Total impact across all cohorts
    total_annual_gmv = sum([row['Incremental GMV (₹)'] * (365 / row['Test Duration (Days)']) 
                             for row in summary_rows if row['Test Duration (Days)'] > 0])
    
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
            heatmap_data = recency_df.pivot(index='Recency', columns='Test Group', values=metric)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdBu_r',
                zmid=0,  # Center colorscale at zero
                text=[[f"{val:.2f}%" for val in row] for row in heatmap_data.values],
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
    best_cohort_idx = summary_df['GMV Lift (%)'].idxmax()
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
    st.write("No data available for generating an executive summary based on current selections.")
