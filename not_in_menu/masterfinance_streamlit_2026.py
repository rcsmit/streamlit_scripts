import streamlit as st
import pandas as pd
import plotly.graph_objects as go
# from datetime import datetime

# Page config
st.set_page_config(
    page_title="Financial Sankey Diagram", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stMetric label {
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.75rem !important;}
    .node-label {
        fill: rgb(0, 0, 0) !important;
        text-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)
        
# st.markdown("""
#     <style>
#     .node-label {
#         fill: rgb(0, 0, 0) !important;
#         text-shadow: none !important;
#     }
    
    
# File path
FILE = r"C:\Users\rcxsm\Documents\xls\masterfinance_2023.xlsx"

@st.cache_data
def load_data():
    """Load and prepare the financial data"""
    df = pd.read_excel(FILE, sheet_name="INVOER", header=0)
    return df

def process_data_for_year(df, year, threshold):
    """Process data for a specific year"""
    # Filter for the selected year
    df_year = df[df['year'] == year].copy()
    
    # Separate income and expenses
    df_income = df_year[df_year["income_expenses"] == "IN"].copy()
    df_expenses = df_year[df_year["income_expenses"].isin(['UIT_AZIE', 'UIT', 'UIT_VL', 'UIT_AZIE_VL',  'UT' , 'UIT_AZIE_VLIEGTICKETS'])].copy()
    
    # Make expenses positive for visualization
    df_expenses['Bedrag_abs'] = df_expenses['Bedrag'].abs()
    
    # Calculate totals by category
    income_by_category = df_income.groupby('main_category')['Bedrag'].sum()
    
    # Group expenses by main_category first
    expenses_by_main = df_expenses.groupby('main_category')['Bedrag'].sum().abs()
    
    # Group expenses by both main and subcategory for the flow
    expenses_hierarchical = df_expenses.groupby(['main_category', 'category'])['Bedrag'].sum().abs().reset_index()
    
    # Group by subcategory only
    expenses_by_subcategory = df_expenses.groupby('category')['Bedrag'].sum().abs()
    
    # Calculate balance and add deficit/surplus
    total_income = income_by_category.sum()
    total_expenses = expenses_by_main.sum()
    balance = total_income - total_expenses
    
    if balance < 0:
        # Deficit: add to income
        income_by_category['Deficit'] = abs(balance)
    else:
        # Surplus: add to expenses
        expenses_by_main['Surplus/Savings'] = balance
        expenses_by_subcategory['Surplus/Savings'] = balance
        surplus_row = pd.DataFrame({
            'main_category': ['Surplus/Savings'],
            'category': ['Surplus/Savings'],
            'Bedrag': [balance]
        })
        expenses_hierarchical = pd.concat([expenses_hierarchical, surplus_row], ignore_index=True)
    
    return income_by_category, expenses_by_main, expenses_by_subcategory, expenses_hierarchical

def process_data_for_year_(df, year, threshold):
    """Process data for a specific year"""
    # Filter for the selected year
    df_year = df[df['year'] == year].copy()
    #print(df_year["income_expenses"].unique())
    # ['KRUIS' 'UIT_AZIE' 'UIT' 'UIT_VL' 'UIT_AZIE_VL' 'IN' 'UT' , 'UIT_AZIE_VLIEGTICKETS']
    # Separate income and expenses
    df_income = df_year[df_year["income_expenses"] == "IN"].copy()
    df_expenses = df_year[df_year["income_expenses"].isin(['UIT_AZIE', 'UIT', 'UIT_VL', 'UIT_AZIE_VL',  'UT' , 'UIT_AZIE_VLIEGTICKETS'])].copy()
    
    # Make expenses positive for visualization
    df_expenses['Bedrag_abs'] = df_expenses['Bedrag'].abs()
    
    # Calculate totals by category
    income_by_category = df_income.groupby('main_category')['Bedrag'].sum()
    
    # Group expenses by main_category first
    expenses_by_main = df_expenses.groupby('main_category')['Bedrag'].sum().abs()
    
    # Group expenses by both main and subcategory for the flow
    expenses_hierarchical = df_expenses.groupby(['main_category', 'category'])['Bedrag'].sum().abs().reset_index()
    
    # Group by subcategory only
    expenses_by_subcategory = df_expenses.groupby('category')['Bedrag'].sum().abs()
    
    return income_by_category, expenses_by_main, expenses_by_subcategory, expenses_hierarchical

def create_sankey(income_by_category, expenses_by_main, expenses_by_subcategory, expenses_hierarchical,show_amounts, threshold, selected_year):
    """Create Sankey diagram with two-level expense breakdown"""
   
    # Calculate percentage of each subcategory within its main category
    expenses_hierarchical['main_total'] = expenses_hierarchical['main_category'].map(expenses_by_main)
    expenses_hierarchical['percentage'] = expenses_hierarchical['Bedrag'] / expenses_hierarchical['main_total']

    # Mark items below threshold
    expenses_hierarchical['is_small'] = expenses_hierarchical['percentage'] < threshold

    # Group small items into "Other" per main category
    def aggregate_small(group):
        large = group[~group['is_small']]
        small = group[group['is_small']]
        
        if len(small) > 0:
            other_row = pd.DataFrame({
                'main_category': [group['main_category'].iloc[0]],
                'category': ['Other ' + group['main_category'].iloc[0]],
                'Bedrag': [small['Bedrag'].sum()]
            })
            return pd.concat([large[['main_category', 'category', 'Bedrag']], other_row])
        return large[['main_category', 'category', 'Bedrag']]

    expenses_hierarchical = expenses_hierarchical.groupby('main_category', group_keys=False).apply(aggregate_small).reset_index(drop=True)

    # Group by subcategory only
    expenses_by_subcategory = expenses_hierarchical.groupby('category')['Bedrag'].sum().abs()
    # Calculate totals
    total_income = income_by_category.sum()
    total_expenses = expenses_by_main.sum()
    balance = total_income - total_expenses
    
    # Initialize lists for Sankey
    labels = []
    sources = []
    targets = []
    values = []
    colors = []
    
        # Color palettes
    income_colors = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#06B6D4']
    expense_main_colors = ['#F97316', '#EAB308', '#06B6D4', '#8B5CF6', '#EC4899']
    expense_sub_colors = ['#FFA07A', '#FFD93D', '#4D96FF', '#9D84B7', '#FF8C94',
                        '#A8DADC', '#F4A261', '#E9C46A', '#00B4D8', '#90E0EF',
                        '#CAF0F8', '#FF6392', '#FFB830', '#36C9C6', '#624E88',
                        '#A78BFA', '#FBBF24', '#60A5FA', '#C084FC', '#F472B6']
    # Bright colors for surplus/deficit
    surplus_color = '#00FF00'  # Bright green
    deficit_color = '#FF0000'  # Bright red

    # 1. Add income source nodes
    income_start_idx = 0
    for i, (cat, val) in enumerate(income_by_category.items()):
        if show_amounts:
            labels.append(f'{cat} (€{val:,.0f})')
        else:
            labels.append(f'{cat}')
        # Use bright red for deficit, otherwise use income colors
        if cat == 'Deficit':
            colors.append(deficit_color)
        else:
            colors.append(income_colors[i % len(income_colors)])

    # 2. Add "Total Income" middle node
    total_income_idx = len(labels)
    if show_amounts:
        labels.append(f'Total Income (€{total_income:,.0f})')
    else:
        labels.append('Total Income')
    
    colors.append('#2ECC71')

    # 3. Add main expense category nodes (first level)
    expense_main_start_idx = len(labels)
    main_category_indices = {}
    for i, (cat, val) in enumerate(expenses_by_main.items()):
        if show_amounts:
            labels.append(f'{cat} (€{val:,.0f})')
        else:
            labels.append(f'{cat}')
        # Use bright green for surplus, otherwise use expense colors
        if cat == 'Surplus/Savings':
            colors.append(surplus_color)
        else:
            colors.append(expense_main_colors[i % len(expense_main_colors)])
        main_category_indices[cat] = expense_main_start_idx + i

    # 4. Add subcategory expense nodes (second level)
    expense_sub_start_idx = len(labels)
    subcategory_indices = {}
    for i, (cat, val) in enumerate(expenses_by_subcategory.items()):
        if show_amounts:
            labels.append(f'{cat} (€{val:,.0f})')
        else:
            labels.append(f'{cat}')
        # Use bright green for surplus, otherwise use subcategory colors
        if cat == 'Surplus/Savings':
            colors.append(surplus_color)
        else:
            colors.append(expense_sub_colors[i % len(expense_sub_colors)])
        subcategory_indices[cat] = expense_sub_start_idx + i
  
    # # 5. Add balance node if there's surplus or deficit
   
    # balance_idx = -1
    
    
    
               
    # Create links: Income sources → Total Income
    for i, (cat, val) in enumerate(income_by_category.items()):
        sources.append(income_start_idx + i)
        targets.append(total_income_idx)
        values.append(val)
    
    # Create links: Total Income → Main Expense Categories
    for i, (cat, val) in enumerate(expenses_by_main.items()):
        sources.append(total_income_idx)
        targets.append(expense_main_start_idx + i)
        values.append(val)
    
    # Create links: Main Expense Categories → Subcategories
    for _, row in expenses_hierarchical.iterrows():
        main_cat = row['main_category']
        sub_cat = row['category']
        amount = row['Bedrag']
        
        if main_cat in main_category_indices and sub_cat in subcategory_indices:
            sources.append(main_category_indices[main_cat])
            targets.append(subcategory_indices[sub_cat])
            values.append(amount)
    

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.8)", width=1),  # Lighter border
            label=labels,
            color=colors,
            customdata=[f'{label}' for label in labels],
            hovertemplate='<b>%{label}</b><br>€%{value:,.2f}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>€%{value:,.2f}<extra></extra>'
        )
    )])
    
    fig.update_layout(
    title={
        'text': f"Financial Flow Analysis - {selected_year}",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 28, 'color': '#2c3e50', 'family': 'Inter, Segoe UI, -apple-system, system-ui, sans-serif'}
    },
    # font=dict(
    #         size=13, 
    #         family='Inter, Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, Helvetica Neue, Arial, sans-serif',
    #         color='#2c3e50'
    #     ),
    #font=dict(size=10, color='black'),

    font_family="Courier New",
    font_color="blue",
    font_size=14,
    
        height=1600,
        margin=dict(l=20, r=20, t=80, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig, total_income, total_expenses, balance
def main():
    # Main app
    st.title("💰 Financial Flow Sankey Diagram")
    st.markdown("Visualizing income and expense flows with two-level categorization")

    # Load data
    try:
        df = load_data()
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Year selection
        available_years = sorted(df['year'].unique())
        selected_year = st.sidebar.selectbox(
            "Select Year",
            options=available_years,
            index=len(available_years) - 1 if 2025 in available_years else 0
        )
        
        # Threshold selection
        threshold = st.sidebar.slider(
            "Group subcategories below (%)",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Subcategories below this percentage of their main category will be grouped into 'Other'"
        ) / 100
        show_amounts = st.sidebar.checkbox(
            "Show amounts in labels", True)
        # Process data
        income_by_cat, expenses_by_main, expenses_by_subcat, expenses_hier = process_data_for_year(df, selected_year, threshold)
     

        # Create metrics row
        col1, col2, col3 = st.columns(3)

        # Calculate original totals (before adding deficit/surplus)
        total_income_original = income_by_cat.sum()
        total_expenses_original = expenses_by_main.sum()

        # Remove deficit from income if it exists
        if 'Deficit' in income_by_cat.index:
            total_income_original = income_by_cat.drop('Deficit').sum()

        # Remove surplus from expenses if it exists  
        if 'Surplus/Savings' in expenses_by_main.index:
            total_expenses_original = expenses_by_main.drop('Surplus/Savings').sum()

        balance = total_income_original - total_expenses_original

        with col1:
            st.metric(
                label="Total Income",
                value=f"€{total_income_original:,.2f}",
                delta=None
            )

        with col2:
            st.metric(
                label="Total Expenses",
                value=f"€{total_expenses_original:,.2f}",
                delta=None
            )

        with col3:
            st.metric(
                label="Surplus/Deficit",
                value=f"€{balance:,.2f}",
            )
        
        # Create and display Sankey
        fig, tot_inc, tot_exp, bal = create_sankey(income_by_cat, expenses_by_main, expenses_by_subcat, expenses_hier,  show_amounts, threshold, selected_year)
                                    
        st.plotly_chart(fig)
        
        # Show data tables
        if 1==1:
        #with st.expander("📊 View Detailed Data"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Income by Category")
                st.dataframe(
                    income_by_cat.reset_index().rename(
                        columns={'main_category': 'Category', 'Bedrag': 'Amount'}
                    ),
                    width='stretch'
                )
            
            with col2:
                st.subheader("Expenses by Main Category")
                st.dataframe(
                    expenses_by_main.reset_index().rename(
                        columns={'main_category': 'Category', 'Bedrag_abs': 'Amount'}
                    ),
                    width='stretch'
                )
                

                st.subheader("Expenses by Subcategory")
                # expenses_display = expenses_by_subcat.reset_index().rename(
                #     columns={'category': 'Category', 'Bedrag_abs': 'Amount'}
                # )
            
            
                
                expenses_display =expenses_hier.reset_index().rename(
                    columns={'category': 'Category', 'Bedrag_abs': 'Amount'}
                )
            
                # Add percentage column
                expenses_display['Percentage'] = (expenses_display['Bedrag'] / expenses_display['Bedrag'].sum() * 100).round(2)
                expenses_display['Percentage'] = expenses_display['Percentage'].astype(str) + '%'
                st.dataframe(
                    expenses_display,
                    width='stretch'
                )


        # Add this after the Sankey diagram display:

        st.markdown("---")
        st.subheader("📊 Winst- en Verliesrekening")

        # Create two-column P&L statement
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Inkomsten")
            income_df = income_by_cat.copy()
            if 'Deficit' in income_df.index:
                income_df = income_df.drop('Deficit')
            
            for cat, amount in income_df.items():
                st.markdown(f"**{cat}**: €{amount:,.2f}")
            
            st.markdown("---")
            st.markdown(f"**Totaal Inkomsten**: €{income_df.sum():,.2f}")

        with col_right:
            st.markdown("#### Uitgaven")
            expenses_df = expenses_by_main.copy()
            if 'Surplus/Savings' in expenses_df.index:
                expenses_df = expenses_df.drop('Surplus/Savings')
            
            for cat, amount in expenses_df.items():
                st.markdown(f"**{cat}**: €{amount:,.2f}")
            
            st.markdown("---")
            st.markdown(f"**Totaal Uitgaven**: €{expenses_df.sum():,.2f}")

        # Show balance at bottom
        st.markdown("---")
        balance_display = income_df.sum() - expenses_df.sum()
        if balance_display >= 0:
            st.markdown(f"### ✅ **Winst (Surplus)**: €{balance_display:,.2f}")
        else:
            st.markdown(f"### ❌ **Verlies (Deficit)**: €{abs(balance_display):,.2f}")
    except FileNotFoundError:
        st.error(f"❌ File not found: {FILE}")
        st.info("Please update the FILE path in the code to match your file location.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()