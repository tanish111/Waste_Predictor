import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

if 'forecast_data' not in st.session_state:
    from model import get_mess_data
    st.session_state.forecast_data = get_mess_data()
forecast_data = st.session_state.forecast_data

# File paths
MENU_FILE = 'menu_df.csv'
INVENTORY_FILE = 'inventory_data.csv'
DAILY_ADDITIONS_FILE = 'daily_additions.csv'

# Read the food ingredient dataset
if 'ingredients_df' not in st.session_state:
    try:
        st.session_state.ingredients_df = pd.read_csv('food_ingredient_dataset.csv')
    except FileNotFoundError:
        st.error("The 'food_ingredient_dataset.csv' file was not found.")
        st.stop()
# Load inventory_data from CSV or initialize
if 'inventory_data' not in st.session_state:
    if os.path.exists(INVENTORY_FILE):
        st.session_state.inventory_data = pd.read_csv(INVENTORY_FILE)
    else:
        st.session_state.inventory_data = pd.DataFrame({
            'Material': ['Flour', 'Vegetables', 'Oils', 'Sugar', 'Milk', 'Paneer', 'Spices'],
            'Quantity': [50000, 30000, 20000, 10000, 8000, 5000, 2000],
            'Unit': ['grams', 'grams', 'mL', 'grams', 'mL', 'grams', 'grams']
        })

# Load menu_df from CSV or initialize
if 'menu_df' not in st.session_state:
    if os.path.exists(MENU_FILE):
        st.session_state.menu_df = pd.read_csv(MENU_FILE)
    else:
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        st.session_state.menu_df = pd.DataFrame({
            'Day': days_of_week,
            'Breakfast': ['']*7,
            'Lunch': ['']*7,
            'Dinner': ['']*7
        })

# Replace NaN values with empty strings in menu_df
st.session_state.menu_df.fillna('', inplace=True)

# Load daily additions from CSV or initialize
if 'daily_additions' not in st.session_state:
    if os.path.exists(DAILY_ADDITIONS_FILE):
        daily_additions_df = pd.read_csv(DAILY_ADDITIONS_FILE)
        st.session_state.daily_additions = daily_additions_df.set_index('Material').T.to_dict()
    else:
        st.session_state.daily_additions = {}

# Helper function to parse ingredient columns
def parse_ingredient_column(col_name):
    if '(' in col_name and ')' in col_name:
        ingredient, unit = col_name.split('(')
        ingredient = ingredient.strip()
        unit = unit.strip(')')
        return ingredient, unit
    else:
        return col_name, ''

# Function to calculate daily ingredient usage based on selected messes
def calculate_daily_ingredients_used(menu_df, forecast_data, ingredients_df, day_name, selected_mess, footfall_override=None):
    total_ingredients = {}
    today_menu = menu_df[menu_df['Day'] == day_name]
    if today_menu.empty:
        # If no menu for the day, return empty DataFrame
        return pd.DataFrame()
    else:
        for index, row in forecast_data.iterrows():
            meal = row['Meal']
            # Ensure we are using correct mess names
            mess_footfall_columns = {
                'A Mess': 'A_Footfall',
                'C Mess': 'C_Footfall',
                'D Mess': 'D_Footfall'
            }
            if footfall_override is not None:
                footfall = footfall_override
            else:
                # Sum the footfall across the selected messes (A, C, D)
                footfall = sum([row[mess_footfall_columns[mess]] for mess in selected_mess if mess in mess_footfall_columns])

            dishes_str = today_menu[meal].values[0]
            if isinstance(dishes_str, str) and dishes_str.strip():
                dishes = dishes_str.split('; ')
            else:
                dishes = []
            for dish in dishes:
                if dish.strip() == '':
                    continue
                dish_ingredients = ingredients_df[ingredients_df['Item'] == dish]
                if dish_ingredients.empty:
                    continue
                else:
                    # Multiply ingredient amounts by footfall
                    for col in dish_ingredients.columns:
                        if col != 'Item':
                            ingredient_name, unit = parse_ingredient_column(col)
                            amount_per_person = dish_ingredients.iloc[0][col]
                            if pd.isna(amount_per_person):
                                amount_per_person = 0
                            total_amount = amount_per_person * footfall
                            key = (ingredient_name, unit)
                            if key in total_ingredients:
                                total_ingredients[key] += total_amount
                            else:
                                total_ingredients[key] = total_amount
        # Convert total_ingredients to DataFrame
        total_ingredients_df = pd.DataFrame(
            [(ingredient, unit, amount) for (ingredient, unit), amount in total_ingredients.items()],
            columns=['Ingredient', 'Unit', 'Total Amount']
        )
        return total_ingredients_df


# Sidebar
with st.sidebar:
    # Mess Options Segmented Buttons with Multi-Select
    st.header("Select Mess")
    mess_options = ['A Mess', 'C Mess', 'D Mess']
    selected_mess = st.multiselect('', mess_options, default=mess_options)
    
    # Navigation Buttons
    st.header("Navigation")
    selected_page = option_menu(
        menu_title=None,
        options=["Raw Materials", "Daily Forecast", "Change Menu", "Ask for Help"],
        icons=["box-seam", "calendar", "pencil-square", "question-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    
    # Upload CSV
    st.header("Upload Footfall Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    # Bottom Container with Name and Logout Button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Name:** Your Name")
        st.write("**Email:** your.email@example.com")
    with col2:
        if st.button("Logout"):
            st.write("Logging out...")  # Placeholder for logout functionality

# Main Page Content Based on Selection
if selected_page == "Raw Materials":
    st.title("Raw Materials")
    st.header("Inventory Levels")

    # Display current inventory levels
    inventory_data = st.session_state.inventory_data
    st.table(inventory_data)

    # Add New Raw Material Flow
    st.header("Add Raw Material to Inventory")
    material_options = inventory_data['Material'].tolist()
    selected_material = st.selectbox("Select Material to Add", material_options)
    daily_addition = st.number_input("Amount Added Daily", min_value=0.0, value=0.0)
    addition_days = st.number_input("Number of Days", min_value=1, value=1, step=1)
    if st.button("Add to Inventory"):
        # Store the daily addition information in session state
        if 'daily_additions' not in st.session_state:
            st.session_state.daily_additions = {}
        st.session_state.daily_additions[selected_material] = {
            'daily_amount': daily_addition,
            'days': addition_days
        }
        # Save daily additions to CSV
        daily_additions_df = pd.DataFrame.from_dict(st.session_state.daily_additions, orient='index')
        daily_additions_df.reset_index(inplace=True)
        daily_additions_df.rename(columns={'index': 'Material'}, inplace=True)
        daily_additions_df.to_csv(DAILY_ADDITIONS_FILE, index=False)
        st.success(f"Will add {daily_addition} units of {selected_material} daily for {addition_days} day(s).")

    # Ingredient Forecasts over Next 10 Days
    st.header("Ingredient Forecasts for Next 10 Days")

    # Get the ingredients data
    ingredients_df = st.session_state.get('ingredients_df')
    menu_df = st.session_state.get('menu_df')

    # Use today's predicted footfall for next 10 days (now considering A, C, and D footfalls)
    # Get today's date and create a list of dates for the next 10 days
    today_date = pd.Timestamp('today').normalize()
    dates = [today_date + pd.Timedelta(days=i) for i in range(10)]
    day_names = [date.day_name() for date in dates]

    # Initialize forecast DataFrame
    forecast_dict = {'Date': [], 'Ingredient': [], 'Quantity': []}

    # Starting inventory levels
    initial_inventory = st.session_state.inventory_data.set_index('Material')['Quantity'].to_dict()
    current_inventory = initial_inventory.copy()

    # Initialize daily additions
    daily_additions = st.session_state.get('daily_additions', {}).copy()

    for i, (date, day_name) in enumerate(zip(dates, day_names)):
        # Calculate daily ingredients used, considering selected messes
        daily_ingredients_df = calculate_daily_ingredients_used(menu_df, forecast_data, ingredients_df, day_name, selected_mess)
        # Update inventory levels
        for ingredient in inventory_data['Material']:
            # Add daily addition if applicable
            addition = 0
            if ingredient in daily_additions:
                if daily_additions[ingredient]['days'] > 0:
                    addition = daily_additions[ingredient]['daily_amount']
                    # Decrement the remaining days
                    daily_additions[ingredient]['days'] -= 1
                else:
                    addition = 0
            # Subtract used amount
            used_amount = 0
            if not daily_ingredients_df.empty:
                ingredient_usage = daily_ingredients_df[daily_ingredients_df['Ingredient'] == ingredient]
                if not ingredient_usage.empty:
                    used_amount = ingredient_usage['Total Amount'].values[0]
            # Update current inventory
            current_inventory[ingredient] = current_inventory.get(ingredient, 0) + addition - used_amount
            if current_inventory[ingredient] < 0:
                current_inventory[ingredient] = 0  # Prevent negative inventory
            # Record the inventory level
            forecast_dict['Date'].append(date)
            forecast_dict['Ingredient'].append(ingredient)
            forecast_dict['Quantity'].append(current_inventory[ingredient])

    # Update the session state with the current inventory
    st.session_state.inventory_data['Quantity'] = st.session_state.inventory_data['Material'].map(current_inventory)
    # Save the updated inventory to CSV
    st.session_state.inventory_data.to_csv(INVENTORY_FILE, index=False)

    forecast_df = pd.DataFrame(forecast_dict)

    # Create line plot with different colors for each ingredient
    fig = px.line(
        forecast_df,
        x='Date',
        y='Quantity',
        color='Ingredient',
        title='Inventory Forecast for Next 10 Days',
        labels={'Quantity': 'Amount', 'Date': 'Date', 'Ingredient': 'Ingredient'}
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_page == "Daily Forecast":
    st.title("Daily Forecast")
    st.header("Predicted Footfall and Wastage")
    
    st.table(forecast_data)
    
    # Get the ingredients data
    ingredients_df = st.session_state.get('ingredients_df')
    if ingredients_df is None:
        st.error("Ingredients data not found.")
    else:
        # Get today's day of the week
        today = pd.Timestamp('today').day_name()
        
        # Get today's menu from menu_df
        menu_df = st.session_state.get('menu_df')
        if menu_df is None:
            st.warning("Menu data not found. Please set up the menu in the 'Change Menu' section.")
        else:
            # Calculate ingredients used today based on predicted footfall
            total_ingredients_df = calculate_daily_ingredients_used(
                menu_df, forecast_data, ingredients_df, today, selected_mess
            )
            if total_ingredients_df.empty:
                st.warning(f"No ingredients data for today ({today}).")
            else:
                st.header("Total Ingredients Used Today")
                st.table(total_ingredients_df)

                # Calculate total possible ingredients if all 1487 students attended
                total_possible_ingredients_df = calculate_daily_ingredients_used(
                    menu_df, forecast_data, ingredients_df, today, selected_mess, footfall_override=1487
                )
                
                # Rename columns to distinguish between used and possible amounts
                total_ingredients_df = total_ingredients_df.rename(columns={'Total Amount': 'Used Amount'})
                total_possible_ingredients_df = total_possible_ingredients_df.rename(columns={'Total Amount': 'Possible Amount'})
                
                # Merge the two DataFrames on 'Ingredient' and 'Unit'
                merged_df = pd.merge(total_possible_ingredients_df, total_ingredients_df, on=['Ingredient', 'Unit'], how='outer')
                
                # Fill NaN values with 0
                merged_df[['Possible Amount', 'Used Amount']] = merged_df[['Possible Amount', 'Used Amount']].fillna(0)
                
                # Calculate 'Wasted Amount' = 'Possible Amount' - 'Used Amount'
                merged_df['Wasted Amount'] = merged_df['Possible Amount'] - merged_df['Used Amount']
                
                # Display the 'Wasted Amount' table
                st.header("Total Ingredients Wasted Today")
                st.table(merged_df[['Ingredient', 'Unit', 'Wasted Amount']])


elif selected_page == "Change Menu":
    st.title("Change Menu")
    st.header("Current Menu")

    # Apply custom CSS to wrap text and preserve line breaks in tables
    st.markdown("""
    <style>
    table td {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

    # Process the DataFrame to replace '; ' with '\n' for better display
    display_df = st.session_state.menu_df.copy()
    for meal_col in ['Breakfast', 'Lunch', 'Dinner']:
        display_df[meal_col] = display_df[meal_col].str.replace('; ', '\n')

    # Display current menu with cell wrapping
    st.table(display_df)

    # Option to edit menu
    st.header("Edit Menu")
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day = st.selectbox("Select Day", days_of_week)
    day_index = days_of_week.index(day)

    # Meal selection
    meal_options = ['Breakfast', 'Lunch', 'Dinner']
    meal = st.selectbox("Select Meal", meal_options)

    # Dishes selection based on meal
    if meal in ['Breakfast', 'Lunch', 'Dinner']:
        # Predefined list of dishes
        dishes_list = st.session_state.ingredients_df['Item'].unique().tolist()
        default_dishes = st.session_state.menu_df.loc[st.session_state.menu_df['Day'] == day, meal].values[0].split('; ')
        selected_dishes = st.multiselect("Select Dishes", dishes_list, default=default_dishes if default_dishes != [''] else [])
    else:
        selected_dishes = []

    if st.button("Update Menu"):
        # Update the menu_df in session state
        st.session_state.menu_df.loc[st.session_state.menu_df['Day'] == day, meal] = '; '.join(selected_dishes)
        # Save the updated menu_df to CSV
        st.session_state.menu_df.to_csv(MENU_FILE, index=False)
        st.success(f"{meal} menu for {day} has been updated.")

elif selected_page == "Ask for Help":
    st.title("Ask for Help")
    
    # Define current state that will be sent to the API
    current_state = {
        "inventory": st.session_state.inventory_data.to_dict(),
        "menu": st.session_state.menu_df.to_dict(),
        "daily_additions": st.session_state.daily_additions
    }
    current_state = json.dumps(current_state, indent=4)

    
    # Display the current state to the user (optional, for debugging)
    with st.expander("Current State Information"):
        st.write("Inventory Data:", st.session_state.inventory_data)
        st.write("Menu Data:", st.session_state.menu_df)
        st.write("Daily Additions:", st.session_state.daily_additions)
    
    # Chat interface
    st.header("Chat with Gemini")
    
    # Get user query
    user_query = st.text_input("Enter your query:", "")
    
    if user_query:
        # Send the query and current state to the Gemini API
        st.write(f"Sending query: {user_query}")
        response = send_to_gemini_api(user_query, current_state)
        
        # Display the response
        if "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            st.write("Response from Gemini:", response.get('response', 'No response available'))
    st.write("If you need assistance, please contact the support team.")
    st.write("**Email:** support@example.com")
    st.write("**Phone:** +1234567890")

else:
    st.title("Welcome")
    st.write("Please select an option from the sidebar.")

# Handle Uploaded Footfall Data
if uploaded_file is not None:
    # Read the uploaded CSV file
    footfall_data = pd.read_csv(uploaded_file)
    st.header("Uploaded Footfall Data")
    st.write(footfall_data)
