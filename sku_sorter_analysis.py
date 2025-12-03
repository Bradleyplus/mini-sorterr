import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ---------------------- Page Configuration ----------------------
st.set_page_config(
    page_title="SKU Sorting Analysis System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Global Parameter Configuration ----------------------
# Mushiny 3D Sorter Mini Product Parameters
SORTER_SPECS = {
    "Small": {
        "length_range": (50, 240),  # mm
        "width_range": (50, 170),  # mm
        "height_range": (5, 100),  # mm
        "weight_range": (20, 5000),  # g (0.02~5kg)
        "bin_width": 240,
        "color": "#FF6B6B"
    },
    "Medium": {
        "length_range": (50, 400),  # mm
        "width_range": (50, 300),  # mm
        "height_range": (5, 200),  # mm
        "weight_range": (20, 10000),  # g (0.02~10kg)
        "bin_width": 400,
        "color": "#4ECDC4"
    },
    "Large": {
        "length_range": (50, 750),  # mm
        "width_range": (50, 500),  # mm
        "height_range": (5, 250),  # mm
        "weight_range": (20, 10000),  # Default same as Medium (not specified in docs)
        "bin_width": 600,
        "color": "#45B7D1"
    },
    "Unmatched": {
        "color": "#95A5A6"
    }
}

# Size Range Configuration
SIZE_BINS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 1000]
SIZE_LABELS = ["0-50mm", "50-100mm", "100-150mm", "150-200mm", "200-250mm",
               "250-300mm", "300-350mm", "350-400mm", "400-450mm", "≥450mm"]


# ---------------------- Tool Functions ----------------------
def load_data(uploaded_file):
    """Load uploaded Excel/CSV file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file!")
            return None
        return df
    except Exception as e:
        st.error(f"File reading failed: {str(e)}")
        return None


def process_dimensions(df, length_col, width_col, height_col, weight_col):
    """Process dimension data: calculate longest/second longest/shortest side, unit conversion"""
    # Select dimension columns
    dim_cols = [length_col, width_col, height_col]

    # Validate dimension data
    for col in dim_cols + [weight_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove missing values
    df_clean = df.dropna(subset=dim_cols + [weight_col]).copy()

    # Sort dimensions in descending order
    dim_matrix = df_clean[dim_cols].values
    sorted_dims = np.sort(dim_matrix, axis=1)[:, ::-1]  # Descending sort

    # Add processed columns
    df_clean['Longest Side (mm)'] = sorted_dims[:, 0]
    df_clean['Second Longest Side (mm)'] = sorted_dims[:, 1]
    df_clean['Shortest Side (mm)'] = sorted_dims[:, 2]
    df_clean['Weight (g)'] = df_clean[weight_col].astype(float)

    # Unit conversion (convert cm to mm if max < 100)
    if df_clean['Longest Side (mm)'].max() < 100:
        df_clean['Longest Side (mm)'] *= 10
        df_clean['Second Longest Side (mm)'] *= 10
        df_clean['Shortest Side (mm)'] *= 10

    return df_clean


def match_sorter(df):
    """Match suitable sorting product based on dimensions and weight (mutually exclusive)"""

    def get_sorter(row):
        length = row['Longest Side (mm)']
        width = row['Second Longest Side (mm)']
        height = row['Shortest Side (mm)']
        weight = row['Weight (g)']

        # Prioritize matching Small
        if (SORTER_SPECS['Small']['length_range'][0] <= length <= SORTER_SPECS['Small']['length_range'][1] and
                SORTER_SPECS['Small']['width_range'][0] <= width <= SORTER_SPECS['Small']['width_range'][1] and
                SORTER_SPECS['Small']['height_range'][0] <= height <= SORTER_SPECS['Small']['height_range'][1] and
                SORTER_SPECS['Small']['weight_range'][0] <= weight <= SORTER_SPECS['Small']['weight_range'][1]):
            return "Small"

        # Match Medium
        elif (SORTER_SPECS['Medium']['length_range'][0] <= length <= SORTER_SPECS['Medium']['length_range'][1] and
              SORTER_SPECS['Medium']['width_range'][0] <= width <= SORTER_SPECS['Medium']['width_range'][1] and
              SORTER_SPECS['Medium']['height_range'][0] <= height <= SORTER_SPECS['Medium']['height_range'][1] and
              SORTER_SPECS['Medium']['weight_range'][0] <= weight <= SORTER_SPECS['Medium']['weight_range'][1]):
            return "Medium"

        # Match Large
        elif (SORTER_SPECS['Large']['length_range'][0] <= length <= SORTER_SPECS['Large']['length_range'][1] and
              SORTER_SPECS['Large']['width_range'][0] <= width <= SORTER_SPECS['Large']['width_range'][1] and
              SORTER_SPECS['Large']['height_range'][0] <= height <= SORTER_SPECS['Large']['height_range'][1] and
              SORTER_SPECS['Large']['weight_range'][0] <= weight <= SORTER_SPECS['Large']['weight_range'][1]):
            return "Large"

        # Unmatched
        else:
            return "Unmatched"

    df['Matched Product'] = df.apply(get_sorter, axis=1)
    return df


def calculate_individual_eligibility(df):
    """单独统计每个型号可分拣的SKU（非互斥匹配）"""
    # 检查是否符合Small的分拣条件
    df['Eligible for Small'] = (
        df['Longest Side (mm)'].between(*SORTER_SPECS['Small']['length_range']) &
        df['Second Longest Side (mm)'].between(*SORTER_SPECS['Small']['width_range']) &
        df['Shortest Side (mm)'].between(*SORTER_SPECS['Small']['height_range']) &
        df['Weight (g)'].between(*SORTER_SPECS['Small']['weight_range'])
    )

    # 检查是否符合Medium的分拣条件
    df['Eligible for Medium'] = (
        df['Longest Side (mm)'].between(*SORTER_SPECS['Medium']['length_range']) &
        df['Second Longest Side (mm)'].between(*SORTER_SPECS['Medium']['width_range']) &
        df['Shortest Side (mm)'].between(*SORTER_SPECS['Medium']['height_range']) &
        df['Weight (g)'].between(*SORTER_SPECS['Medium']['weight_range'])
    )

    # 检查是否符合Large的分拣条件
    df['Eligible for Large'] = (
        df['Longest Side (mm)'].between(*SORTER_SPECS['Large']['length_range']) &
        df['Second Longest Side (mm)'].between(*SORTER_SPECS['Large']['width_range']) &
        df['Shortest Side (mm)'].between(*SORTER_SPECS['Large']['height_range']) &
        df['Weight (g)'].between(*SORTER_SPECS['Large']['weight_range'])
    )
    return df


def calculate_fill_rate(row, tote_size=(600, 400, 120)):
    """Calculate tote fill rate"""
    tote_volume = tote_size[0] * tote_size[1] * tote_size[2]
    product_volume = row['Longest Side (mm)'] * row['Second Longest Side (mm)'] * row['Shortest Side (mm)']
    return min(100, (product_volume / tote_volume) * 100)


# ---------------------- Page Layout ----------------------
st.title("📦 SKU Sorting Analysis System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration Area")
    uploaded_file = st.file_uploader("Upload SKU Data File", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            st.subheader("Column Mapping")
            columns = df_raw.columns.tolist()

            # Column selection
            length_col = st.selectbox("Length Column", columns)
            width_col = st.selectbox("Width Column", columns)
            height_col = st.selectbox("Height Column", columns)
            weight_col = st.selectbox("Weight Column (g)", columns)
            sku_col = st.selectbox("SKU Column", columns)

            # Analysis button
            analyze_btn = st.button("Start Analysis", type="primary", use_container_width=True)
    else:
        st.info("Please upload a data file first")
        analyze_btn = False

# Main Content Area
if uploaded_file and analyze_btn and 'df_raw' in locals():
    # Data processing
    with st.spinner("Processing data..."):
        df_processed = process_dimensions(df_raw, length_col, width_col, height_col, weight_col)
        df_final = match_sorter(df_processed)
        df_final = calculate_individual_eligibility(df_final)  # 新增：单独型号适配性统计
        df_final['Tote Fill Rate (%)'] = df_final.apply(calculate_fill_rate, axis=1)

    st.success(f"✅ Data processing completed! Analyzed {len(df_final)} valid SKUs")
    st.markdown("---")

    # Data Preview (新增单独型号适配列)
    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(
            df_final[[sku_col, 'Longest Side (mm)', 'Second Longest Side (mm)', 'Shortest Side (mm)', 'Weight (g)',
                      'Matched Product', 'Eligible for Small', 'Eligible for Medium', 'Eligible for Large', 'Tote Fill Rate (%)']],
            use_container_width=True,
            height=200
        )

    # 统计单独型号的适配数据
    individual_model_stats = pd.DataFrame({
        "Model": ["Small", "Medium", "Large"],
        "Eligible SKU Count": [
            df_final['Eligible for Small'].sum(),
            df_final['Eligible for Medium'].sum(),
            df_final['Eligible for Large'].sum()
        ],
        "Percentage of Total SKUs (%)": [
            (df_final['Eligible for Small'].sum() / len(df_final)) * 100,
            (df_final['Eligible for Medium'].sum() / len(df_final)) * 100,
            (df_final['Eligible for Large'].sum() / len(df_final)) * 100
        ]
    })

    # Visualization Analysis (新增单独型号适配性标签页)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dimension Distribution Analysis",
        "Weight Distribution Analysis",
        "3D Data Visualization",
        "Product Matching Results",
        "Tote Fill Rate Analysis",
        "Individual Model Eligibility"  # 新增标签页
    ])

    # 1. Dimension Distribution Analysis
    with tab1:
        st.subheader("Longest Side Distribution")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("SKU Dimension Distribution Analysis", fontsize=16, fontweight='bold')

        # Longest Side Histogram
        ax1.hist(df_final['Longest Side (mm)'], bins=SIZE_BINS, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax1.set_title("Longest Side Distribution", fontweight='bold')
        ax1.set_xlabel("Length (mm)")
        ax1.set_ylabel("Number of SKUs")
        ax1.set_xticks(SIZE_BINS[::2])
        ax1.grid(True, alpha=0.3)

        # Second Longest Side Histogram
        ax2.hist(df_final['Second Longest Side (mm)'], bins=SIZE_BINS, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax2.set_title("Second Longest Side Distribution", fontweight='bold')
        ax2.set_xlabel("Length (mm)")
        ax2.set_ylabel("Number of SKUs")
        ax2.set_xticks(SIZE_BINS[::2])
        ax2.grid(True, alpha=0.3)

        # Shortest Side Histogram
        ax3.hist(df_final['Shortest Side (mm)'], bins=range(0, 301, 30), color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax3.set_title("Shortest Side Distribution", fontweight='bold')
        ax3.set_xlabel("Length (mm)")
        ax3.set_ylabel("Number of SKUs")
        ax3.grid(True, alpha=0.3)

        # Longest Side Range Proportion Pie Chart
        size_ranges = pd.cut(df_final['Longest Side (mm)'], bins=SIZE_BINS, labels=SIZE_LABELS, right=False)
        size_counts = size_ranges.value_counts()
        ax4.pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title("Longest Side Range Proportion", fontweight='bold')

        st.pyplot(fig, use_container_width=True)

    # 2. Weight Distribution Analysis
    with tab2:
        st.subheader("SKU Weight Distribution")
        weight_bins = [0, 100, 500, 1000, 2000, 5000, 10000, 20000]
        weight_labels = ["0-100g", "100-500g", "500-1kg", "1-2kg", "2-5kg", "5-10kg", "≥10kg"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Weight Histogram
        ax1.hist(df_final['Weight (g)'], bins=weight_bins, color='#9B59B6', alpha=0.7, edgecolor='black')
        ax1.set_title("Weight Distribution Histogram", fontweight='bold')
        ax1.set_xlabel("Weight (g)")
        ax1.set_ylabel("Number of SKUs")
        ax1.grid(True, alpha=0.3)

        # Weight Range Proportion Pie Chart
        weight_ranges = pd.cut(df_final['Weight (g)'], bins=weight_bins, labels=weight_labels, right=False)
        weight_counts = weight_ranges.value_counts()
        ax2.pie(weight_counts.values, labels=weight_counts.index, autopct='%1.1f%%', startangle=90,
                colors=plt.cm.Set3.colors)
        ax2.set_title("Weight Range Proportion", fontweight='bold')

        st.pyplot(fig, use_container_width=True)

    # 3. 3D Data Visualization
    with tab3:
        st.subheader("SKU 3D Dimension Distribution (Longest × Second Longest × Shortest Side)")

        # 3D Scatter Plot
        fig = px.scatter_3d(
            df_final,
            x='Longest Side (mm)',
            y='Second Longest Side (mm)',
            z='Shortest Side (mm)',
            color='Matched Product',
            color_discrete_map={
                "Small": SORTER_SPECS['Small']['color'],
                "Medium": SORTER_SPECS['Medium']['color'],
                "Large": SORTER_SPECS['Large']['color'],
                "Unmatched": SORTER_SPECS['Unmatched']['color']
            },
            hover_data=[sku_col, 'Weight (g)'],
            title="SKU 3D Dimension Distribution"
        )

        # Set Axes
        fig.update_layout(
            scene=dict(
                xaxis_title='Longest Side (mm)',
                yaxis_title='Second Longest Side (mm)',
                zaxis_title='Shortest Side (mm)',
                xaxis=dict(range=[0, 800]),
                yaxis=dict(range=[0, 600]),
                zaxis=dict(range=[0, 300])
            ),
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

    # 4. Product Matching Results
    with tab4:
        st.subheader("Sorting Product Matching Results")
        match_counts = df_final['Matched Product'].value_counts()

        # Subplot Layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Matching Proportion Pie Chart
        colors = [SORTER_SPECS[cat]['color'] for cat in match_counts.index]
        ax1.pie(match_counts.values, labels=match_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title("Product Matching Proportion", fontweight='bold')

        # Matching Quantity Bar Chart
        bars = ax2.bar(match_counts.index, match_counts.values, color=colors)
        ax2.set_title("Matching Quantity per Product", fontweight='bold')
        ax2.set_ylabel("Number of SKUs")
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # Longest Side Distribution per Product (Box Plot)
        sorter_types = [t for t in ["Small", "Medium", "Large"] if t in match_counts.index]
        if sorter_types:
            data_for_box = [df_final[df_final['Matched Product'] == t]['Longest Side (mm)'].values for t in sorter_types]
            ax3.boxplot(data_for_box, labels=sorter_types, patch_artist=True,
                        boxprops=dict(facecolor='#4ECDC4', alpha=0.7))
            ax3.set_title("Longest Side Distribution per Product", fontweight='bold')
            ax3.set_ylabel("Longest Side (mm)")
            ax3.grid(True, alpha=0.3)

        # Weight Distribution per Product (Box Plot)
        weight_data = [df_final[df_final['Matched Product'] == t]['Weight (g)'].values for t in sorter_types]
        if weight_data:
            ax4.boxplot(weight_data, labels=sorter_types, patch_artist=True,
                        boxprops=dict(facecolor='#9B59B6', alpha=0.7))
            ax4.set_title("Weight Distribution per Product", fontweight='bold')
            ax4.set_ylabel("Weight (g)")
            ax4.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)

    # 5. Tote Fill Rate Analysis
    with tab5:
        st.subheader("Tote Fill Rate Analysis (Default Size: 600×400×120mm)")

        # Fill Rate Ranges
        fill_bins = [0, 20, 40, 60, 80, 100]
        fill_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        df_final['Fill Rate Range'] = pd.cut(df_final['Tote Fill Rate (%)'], bins=fill_bins, labels=fill_labels, right=False)
        fill_counts = df_final['Fill Rate Range'].value_counts().sort_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Fill Rate Distribution Bar Chart
        bars = ax1.bar(fill_counts.index, fill_counts.values, color='#F39C12', alpha=0.7)
        ax1.set_title("Tote Fill Rate Distribution", fontweight='bold')
        ax1.set_xlabel("Fill Rate Range")
        ax1.set_ylabel("Number of SKUs")
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # Fill Rate Range Proportion Pie Chart
        ax2.pie(fill_counts.values, labels=fill_counts.index, autopct='%1.1f%%',
                colors=plt.cm.viridis(np.linspace(0, 1, len(fill_counts))), startangle=90)
        ax2.set_title("Fill Rate Range Proportion", fontweight='bold')

        st.pyplot(fig, use_container_width=True)

    # 6. 新增：单独型号适配性统计与可视化
    with tab6:
        st.subheader("Individual Model Eligibility (Separate Matching)")
        # 展示统计表格
        st.dataframe(individual_model_stats, use_container_width=True)

        # 可视化：数量 + 占比双图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        model_colors = [SORTER_SPECS['Small']['color'], SORTER_SPECS['Medium']['color'], SORTER_SPECS['Large']['color']]

        # 子图1：各型号可分拣SKU数量
        ax1.bar(individual_model_stats['Model'], individual_model_stats['Eligible SKU Count'], color=model_colors)
        ax1.set_title("Eligible SKU Count per Model", fontweight='bold')
        ax1.set_ylabel("Number of SKUs")
        # 添加数值标签
        for i, count in enumerate(individual_model_stats['Eligible SKU Count']):
            ax1.text(i, count + 2, str(count), ha='center', fontweight='bold')

        # 子图2：各型号可分拣SKU占比
        ax2.bar(individual_model_stats['Model'], individual_model_stats['Percentage of Total SKUs (%)'], color=model_colors)
        ax2.set_title("Percentage of Eligible SKUs (vs Total SKUs)", fontweight='bold')
        ax2.set_ylabel("Percentage (%)")
        # 添加数值标签
        for i, pct in enumerate(individual_model_stats['Percentage of Total SKUs (%)']):
            ax2.text(i, pct + 0.5, f"{pct:.1f}%", ha='center', fontweight='bold')

        st.pyplot(fig, use_container_width=True)

    # Result Download
    st.markdown("---")
    st.subheader("📥 Result Download")

    # Prepare download data (新增单独型号适配列)
    download_df = df_final[[
        sku_col, 'Longest Side (mm)', 'Second Longest Side (mm)', 'Shortest Side (mm)', 'Weight (g)',
        'Matched Product', 'Eligible for Small', 'Eligible for Medium', 'Eligible for Large', 'Tote Fill Rate (%)'
    ]].copy()

    # Convert to CSV
    csv_data = download_df.to_csv(index=False, encoding='utf-8-sig')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download Analysis Results (CSV)",
            data=csv_data,
            file_name=f"SKU_Sorting_Analysis_Results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Generate Analysis Report (新增单独型号统计)
    with col2:
        report = f"""
# SKU Sorting Analysis Report
Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Analyzed SKUs: {len(df_final)}

## Product Matching Results (Mutually Exclusive)
"""
        for product, count in df_final['Matched Product'].value_counts().items():
            ratio = (count / len(df_final)) * 100
            report += f"- {product}: {count} items ({ratio:.1f}%)\n"

        report += f"""
## Individual Model Eligibility (Separate Matching)
"""
        for _, row in individual_model_stats.iterrows():
            report += f"- {row['Model']}: {row['Eligible SKU Count']} eligible items ({row['Percentage of Total SKUs (%)']:.1f}% of total SKUs)\n"

        report += f"""
## Dimension Statistics
- Longest Side Range: {df_final['Longest Side (mm)'].min():.1f}~{df_final['Longest Side (mm)'].max():.1f}mm
- Second Longest Side Range: {df_final['Second Longest Side (mm)'].min():.1f}~{df_final['Second Longest Side (mm)'].max():.1f}mm
- Shortest Side Range: {df_final['Shortest Side (mm)'].min():.1f}~{df_final['Shortest Side (mm)'].max():.1f}mm

## Weight Statistics
- Weight Range: {df_final['Weight (g)'].min():.1f}~{df_final['Weight (g)'].max():.1f}g
- Average Weight: {df_final['Weight (g)'].mean():.1f}g

## Tote Fill Rate
- Average Fill Rate: {df_final['Tote Fill Rate (%)'].mean():.1f}%
- Maximum Fill Rate: {df_final['Tote Fill Rate (%)'].max():.1f}%
- Minimum Fill Rate: {df_final['Tote Fill Rate (%)'].min():.1f}%
        """

        st.download_button(
            label="Download Analysis Report (TXT)",
            data=report,
            file_name=f"SKU_Sorting_Analysis_Report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

elif uploaded_file and not analyze_btn:
    st.info("Please complete column mapping in the sidebar, then click [Start Analysis]")

else:
    # Initial Page
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h3>Welcome to the SKU Sorting Analysis System</h3>
        <p style="color: #666; margin-top: 20px;">
            1. Upload an Excel/CSV file containing SKU, length, width, height, and weight data<br>
            2. Complete column mapping configuration in the sidebar<br>
            3. Click "Start Analysis" to automatically generate visual charts and matching results<br>
            4. Support result export and report download
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display Data Format Example
    st.markdown("---")
    st.subheader("📋 Data Format Example")
    sample_data = pd.DataFrame({
        "SKU": ["SKU001", "SKU002", "SKU003", "SKU004"],
        "Length (mm)": [150, 300, 500, 200],
        "Width (mm)": [100, 200, 300, 150],
        "Height (mm)": [50, 100, 150, 80],
        "Weight (g)": [500, 2000, 8000, 1200]
    })
    st.dataframe(sample_data, use_container_width=True)