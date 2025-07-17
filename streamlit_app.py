import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from pathlib import Path
import json 
import re

PROJ_DIR = Path.cwd()
RES_DIR = Path(PROJ_DIR, "res.8")
DATA_DIR = Path(PROJ_DIR, "data")

# 1. Download conda: https://conda-forge.org/download/
# 2. Create a new conda environment from terminal on mac or conda prompt on Windows:
# conda create -n dashboard python=3.10 streamlit pandas numpy openpyxl
# 3. Activate the environment:
# conda activate dashboard

#region Utility Functions
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df, {}

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()
    filter_params = {}
    
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
                filter_params[column] = user_cat_input
            elif is_numeric_dtype(df[column]):
                if "P-value" in column or "p-value" in column:
                    # For p-values, we want to filter out values that are not significant, enter threshold
                    threshold = right.number_input(
                        f"Significance threshold for {column}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.05,
                        step=0.01,
                    )
                    df = df[df[column] < threshold]
                else:
                    # For numeric columns, we want to filter by a range
                    # Get min and max values for the column
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                    filter_params[column] = user_num_input
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
                    filter_params[column] = user_date_input
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
                filter_params[column] = user_text_input
    # return filter parameters
    return df, filter_params

def flatten_multi_index(index: pd.MultiIndex | pd.Index, seperator="|") -> list:
    """
    Take a MultiIndex and combine the index level values into a string seperated by the seperator. For a single
    level index will just return that index unchanged. The result of this function can be assigned to a DataFrame
    to turn a MultiIndex into single Index DataFrame, df. columns = flatten_multi_index(df.columns)

    :param index: Index of MultiIndex
    :param seperator: seperator to use between column levels
    :return: list of flattened column names
    """
    if isinstance(index, pd.MultiIndex):
        return [seperator.join(col).strip() for col in index.values]
    else:
        return [x for x in index.values]


def flatten_df(df: pd.DataFrame, seperator="|") -> pd.DataFrame:
    """
    Convenience wrapper around flatten_multi_index to flatten the column MultiIndex and return the DataFrame with
    the new column labels
    :param df: pd.DataFrame
    :param seperator: seperator to use between column levels
    :return: pd.DataFrame
    """
    _df = df.copy()
    _df.columns = flatten_multi_index(_df.columns, seperator=seperator)
    return _df
#endregion
#region Streamlit Functions

def get_excel_path(analysisType: str = "Pre-transplant") -> Path:
    """
    Get the path to the Excel file based on the analysis type and pairwise type.
    """
    if analysisType == "Pre-transplant":
        fileName = "pre-transplant.tables.xlsx"
    elif analysisType == "Post-transplant":
        fileName = "post-transplant.tables.xlsx"
    
    return Path(RES_DIR, fileName)

def load_data(analysisType: str = "Pre-transplant", sheetName: str = "Sheet1") -> pd.DataFrame:
    """ Load data from the Excel file based on the analysis type and pairwise type.
    Args:
        analysisType (str): Type of analysis, e.g., "Pairwise", "Regression", "NMA".
        pairwiseType (str): Type of pairwise analysis, e.g., "Separate", "Combined".
    Returns:
        pd.DataFrame: Flattened DataFrame with the data from the specified Excel file.
    """
    filePath = get_excel_path(analysisType)
    
    indexCol = [0, 1, 2, 3, 4]
    if sheetName == "MLN":
        df = pd.read_excel(filePath, index_col=indexCol, sheet_name=sheetName)
        return df.reset_index().replace(np.nan, "")
    else:
        df = pd.read_excel(filePath, index_col=indexCol, header=[0, 1], sheet_name=sheetName)
        return flatten_df(df, seperator = ": ").reset_index().replace(np.nan, "")
        
    # return flatten_df(df.reset_index(), seperator="|").set_index("index")
    
@st.cache_data
def get_sheet_names(analysisType: str = "Pre-transplant") -> list:
    """
    Get the sheet names from the Excel file based on the analysis type and pairwise type.
    """
    filePath = get_excel_path(analysisType)
    xl = pd.ExcelFile(filePath)
    return xl.sheet_names


def is_significant(p_value: float | str) -> bool:
    """
    Check if the p-value is significant (less than 0.05).
    """
    if isinstance(p_value, str):
        p_value = p_value.lower()
        if p_value == "<0.001":
            return True
        try:
            p_value = float(p_value)
            return p_value < 0.05
        except ValueError:
            return False
    elif isinstance(p_value, (int, float)):
        return p_value < 0.05
    else:
        return False

def assess_significance(row: pd.Series) -> bool:
    """
    Assess significance of a row based on p-values.
    Returns True if any p-value in the row is significant (less than 0.05).
    """
    for col in row.index:
        if "P-value" in col or "p-value" in col:
            return is_significant(row[col])
    return False


def create_pairwise_comparison_tables(df_pw):
    """
    Create pairwise comparison tables from meta-analysis data.
    
    Parameters:
    -----------
    outcome_category : str
        The outcome category to filter by
    outcome_label : str
        The outcome label to filter by
    comparison_category : str
        The comparison category to filter by
        
    Returns:
    --------
    dict
        Dictionary containing 'univariable' and 'multivariable' styled DataFrames
    """
    
    # Load the data
    df = df_pw.copy()
    
    
    
    if df.empty:
        return {'univariable': None, 'multivariable': None}
    
    # Get unique groups and references
    groups = sorted(df['Group'].dropna().unique())
    references = sorted(df['Reference'].dropna().unique())
    
    results = {}
    
    # Check if multivariable data exists
    has_multivariable = any('Multivariable' in col for col in df.columns)
    has_univariable = any('Univariable' in col for col in df.columns)
    
    # Generate univariable table
    if has_univariable:
        univariable_df = create_pairwise_df(df, groups, references, 'Univariable')
        results['univariable'] = style_pairwise_table(univariable_df)
    else:
        results['univariable'] = None
    # Generate multivariable table if data exists
    if has_multivariable:
        multivariable_df = create_pairwise_df(df, groups, references, 'Multivariable')
        results['multivariable'] = style_pairwise_table(multivariable_df)
    else:
        results['multivariable'] = None
    
    return results

def create_pairwise_df(filtered_df, groups, references, analysis_type):
    """
    Create the pairwise comparison DataFrame.
    """
    # Create empty DataFrame
    df = pd.DataFrame(index=groups, columns=references)
    
    # Column names based on analysis type
    # or_col = f'{analysis_type}: OR (95% CI)'
    or_col = filtered_df.columns[filtered_df.columns.str.contains(analysis_type) & filtered_df.columns.str.contains("95% CI")][0]
    p_col = f'{analysis_type}: P-value'
    
    # Fill the DataFrame
    for _, row in filtered_df.iterrows():
        group = row['Group']
        reference = row['Reference']
        
        if pd.notna(group) and pd.notna(reference) and pd.notna(row.get(or_col)):
            # Combine OR and p-value
            or_ci = str(row[or_col])
            if (or_ci is not None) and (or_ci != ""):
                
                p_val = row.get(p_col)
                
                if pd.notna(p_val):
                    if isinstance(p_val, str):
                        if p_val.lower() == "<0.001":
                            cell_value = f"{or_ci}\np < 0.001"
                        else:
                            p_val = extract_p_value(p_val)
                            if p_val is not None:
                                cell_value = f"{or_ci}\np = {p_val:.3f}"
                            else:
                                cell_value = f"{or_ci}\n"
                    elif isinstance(p_val, (int, float)):
                        cell_value = f"{or_ci}\np = {p_val:.3f}"
                    else:
                        cell_value = f"{or_ci}\np = {p_val}"
                else:
                    cell_value = f"{or_ci}\np = N/A"
                
                df.loc[group, reference] = cell_value
    
    return df
def extract_or_value(or_string):
    """
    Extract the OR value from strings like "1.23 (0.89-1.67)"
    """
    if pd.isna(or_string):
        return None
    
    match = re.match(r'^([0-9.]+)', str(or_string))
    if match:
        return float(match.group(1))
    return None

def extract_p_value(cell_value):
    """
    Extract p-value from cell content
    """
    if pd.isna(cell_value):
        return None
    
    match = re.search(r'^([0-9.]+)', str(cell_value))
    if match:
        return float(match.group(1))
    elif "< 0.001" in str(cell_value).lower():
        return 0.001
    return None

def get_cell_color(cell_value):
    """
    Determine cell background color based on OR and p-value
    """
    if pd.isna(cell_value):
        return ''
    
    # Extract OR and p-value
    or_match = re.match(r'^([0-9.]+)', str(cell_value))
    p_match = re.search(r'p = ([0-9.]+)', str(cell_value))
    p_less_match = re.search(r'p < 0.001', str(cell_value))
    
    if not or_match and (not p_match or not p_less_match):
        return ''
    
    or_value = float(or_match.group(1))
    
    # Only color if significant (p < 0.05)
    if p_match:
        p_value = float(p_match.group(1))
        if p_value >= 0.05:
            return ''
    
    if or_value < 1:
        # Red for protective effect (OR < 1)
        intensity = min(abs(np.log(or_value)), 2) / 2  # Cap at log(0.135) ≈ 2
        alpha = 0.3 + (intensity * 0.4)  # Range from 0.3 to 0.7
        # st.write(f"OR value: {or_value}, Intensity: {intensity}, Alpha: {alpha}")
        return f'background-color: rgba(239, 68, 68, {alpha})'
    else:
        # Blue for risk effect (OR > 1) - Changed from green
        intensity = min(np.log(or_value), 2) / 2  # Cap at log(7.39) ≈ 2
        alpha = 0.3 + (intensity * 0.4)  # Range from 0.3 to 0.7
        # st.write(f"OR value: {or_value}, Intensity: {intensity}, Alpha: {alpha}")
        return f'background-color: rgba(59, 130, 246, {alpha})'

def style_pairwise_table(df):
    """
    Apply styling to the pairwise comparison table
    """
    if df is None:
        return None
    
    # Apply styling
    styled_df = df.style.applymap(get_cell_color)
    
    # Add table styling
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f3f4f6'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
        {'selector': '.index_name', 'props': [('background-color', '#f9fafb'), ('font-weight', 'bold')]},
        {'selector': '', 'props': [('border-collapse', 'collapse'), ('border', '1px solid #d1d5db')]}
    ])
    
    return styled_df

# Example usage function for streamlit
def display_pairwise_tables_streamlit(df_pw):
    """
    Function to use in streamlit app
    """
    
    # Generate tables
    tables = create_pairwise_comparison_tables(df_pw)
    
    # Display legend
    st.markdown("### Legend:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔴 **Red**: Effect < 1, p < 0.05")
    with col2:
        st.markdown("🔵 **Blue**: Effect > 1, p < 0.05")  
    # st.markdown("*Color intensity represents the magnitude of the effect (brighter = stronger effect)*")
    st.markdown("*Unhighlighted cells: Not significant (p ≥ 0.05)*")
    st.markdown("Row = Group, Column = Reference")
    
    # Display univariable table
    if tables['univariable'] is not None:
        st.markdown("### Univariable Analysis")
        st.dataframe(tables['univariable'], use_container_width=True)
    
    # Display multivariable table if available
    if tables['multivariable'] is not None:
        st.markdown("### Multivariable Analysis")
        st.dataframe(tables['multivariable'], use_container_width=True)
    
    if tables['univariable'] is None and tables['multivariable'] is None:
        st.warning("No data found for the selected combination. Please check your selections.")
    
    return tables

# Helper function to get unique values for dropdowns
def get_unique_values(df_pw):
    """
    Get unique values for dropdown menus in streamlit
    """
    df = df_pw.copy()
    df.columns = df.columns.str.strip()
    
    return {
        'outcome_categories': sorted(df['Outcome Category'].dropna().unique()),
        'outcome_labels': sorted(df['Outcome Label'].dropna().unique()),
        'comparison_categories': sorted(df['Comparison Category'].dropna().unique())
    }
#endregion
#region Main


st.title("Transplant Dashboard")

# Add a selectbox to the sidebar:
analysisTypeOptions = [
    'Pre-transplant',
    'Post-transplant'
]
analysisType = st.sidebar.selectbox(
    'Analysis',
    analysisTypeOptions
)


sheetNames = get_sheet_names(analysisType)
sheetName = st.sidebar.selectbox(
    'Sheet',
    sheetNames,
    # index = sheetNames.index(st.session_state['sheetName']) if isinstance(st.session_state['sheetName'], str) else st.session_state['sheetName']  # Ensure index is valid
)
# st.session_state['sheetName'] = sheetName


df = load_data(analysisType, sheetName)
# Get unique values for dropdowns
# unique_values = get_unique_values(df)
unique_values = df.groupby(['Outcome Category','Outcome Label', "Comparison Category"]).size().reset_index().rename(columns={0:'count'})


# Create selection dropdowns
col1, col2, col3 = st.columns(3)

with col1:
    outcome_category = st.selectbox(
        "Outcome Category",
        # options=unique_values['outcome_categories']
        options = sorted(unique_values['Outcome Category'].dropna().unique()),
    )
with col2:
    outcome_label = st.selectbox(
        "Outcome Label",
        # options=unique_values['outcome_labels']
        options = sorted(unique_values[unique_values['Outcome Category'] == outcome_category]['Outcome Label'].dropna().unique()),
    )
with col3:
    comparison_category = st.selectbox(
        "Comparison Category",
        # options=unique_values['comparison_categories']
        options = sorted(unique_values[unique_values['Outcome Category'] == outcome_category][unique_values['Outcome Label'] == outcome_label]['Comparison Category'].dropna().unique()),
    )

# # Generate and display tables
# if st.button("Generate Tables"):
    # Filter data based on selections
    
df_pw = df[
    (df['Outcome Category'] == outcome_category) &
    (df['Outcome Label'] == outcome_label) &
    (df['Comparison Category'] == comparison_category)
].copy()[
    [column for column in df.columns if column not in ['Outcome Category', 'Outcome Label', 'Comparison Category']]
]

if (sheetName != "MLN"):
    
    display_pairwise_tables_streamlit(df_pw)


filtered_df, filter_params = filter_dataframe(df_pw)


# highlight P-value < 0.05
# columns_to_highlight = [column for column in filtered_df.columns if "P-value" in column]
rows_to_color = filtered_df.index[filtered_df.apply(lambda x: assess_significance(x), axis=1)].tolist()
color = "green"

# Show filter parameters
filter_params = {
    "analysisType": analysisType,
    "sheetName": sheetName,
} | filter_params
st.code(filter_params, language="json")



selectedRow = st.dataframe(
    filtered_df.style.applymap(lambda _: f"background-color: {color}",
                    subset=(df.index[rows_to_color],)),
    selection_mode = "single-row",
    on_select = "rerun"
)
if selectedRow:
    # st.write("Selected row data:", filtered_df.iloc[selectedRow["selection"]["rows"]])
    if (len(selectedRow["selection"]["rows"]) == 1):
        
        tab1, tab2 = st.tabs(["Univariable", "Multivariable"])
        scriptCols = [col for col in filtered_df.columns if "script" in col.lower()]
        effectCols = [col for col in filtered_df.columns if "95%" in col.lower()]

        with tab1:
            univariableScriptCol = [col for col in scriptCols if "univariable" in col.lower()]
            for scriptCol in univariableScriptCol:
                script = filtered_df.iloc[selectedRow["selection"]["rows"]][scriptCol].values.tolist()[0]
                # Copy the script to clipboard
                st.markdown(f"**Script for {scriptCol}**")
                st.code(script, language='python')
            univariableEffectCols = [col for col in effectCols if "univariable" in col.lower()]
            for effectCol in univariableEffectCols:
                effect = filtered_df.iloc[selectedRow["selection"]["rows"]][effectCol].values.tolist()[0]
                st.markdown(f"**Effect for {effectCol}**")
                st.code(effect, language='python')
                
            if "Univariable: Files" in filtered_df.columns:
            
                filesJSON = filtered_df.iloc[selectedRow["selection"]["rows"]][("Univariable: Files")].values.tolist()[0]
                filesJSON = filesJSON.replace("'", "\"")  # Ensure valid JSON format
                if filesJSON:
                    files = json.loads(filesJSON)
                    figures = files.get("figures", {})
                    for figureName, figurePath in figures.items():
                        figurePath = Path(RES_DIR, figurePath)
                        if figurePath.exists():
                            st.image(figurePath, caption=figureName)
                            st.code(figurePath, language='python')
                        else:
                            st.warning(f"Figure {figureName} not found at {figurePath}")
        with tab2:
            multivariableScriptCol = [col for col in scriptCols if "multivariable" in col.lower()]
            for scriptCol in multivariableScriptCol:
                script = filtered_df.iloc[selectedRow["selection"]["rows"]][scriptCol].values.tolist()[0]
                # Copy the script to clipboard
                st.markdown(f"**Script for {scriptCol}**")
                st.code(script, language='python')
            multivariableScriptCol = [col for col in effectCols if "multivariable" in col.lower()]
            for effectCol in multivariableScriptCol:
                effect = filtered_df.iloc[selectedRow["selection"]["rows"]][effectCol].values.tolist()[0]
                st.markdown(f"**Effect for {effectCol}**")
                st.code(effect, language='python')

            if "Multivariable: Files" in filtered_df.columns:
        
                filesJSON = filtered_df.iloc[selectedRow["selection"]["rows"]][("Multivariable: Files")].values.tolist()[0]
                filesJSON = filesJSON.replace("'", "\"")  # Ensure valid JSON format
                if filesJSON:
                    files = json.loads(filesJSON)
                    figures = files.get("figures", {})
                    for figureName, figurePath in figures.items():
                        figurePath = Path(RES_DIR, figurePath)
                        if figurePath.exists():
                            st.image(figurePath, caption=figureName)
                            st.code(figurePath, language='python')
                        else:
                            st.warning(f"Figure {figureName} not found at {figurePath}")
                



# dfTest = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv')

# AgGrid(flatten_df(df.reset_index()))
# st.write("Data shape:", df.shape)   
# st.write("Data columns:", df.columns.tolist())
# st.write("Data types:", df.dtypes)

#endregion