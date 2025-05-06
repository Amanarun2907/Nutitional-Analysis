
# # app.py
import streamlit as st
import pandas as pd
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import euclidean_distances
import requests

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
models_dir = os.path.join(project_root, 'src', 'models')
data_dir = os.path.join(project_root, 'data', 'processed')

# --- Load Assets ---
try:
    model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
    cleaned_df = pd.read_csv(os.path.join(data_dir, 'cleaned_data.csv'))
    food_names = cleaned_df['name'].unique().tolist()
    categories = cleaned_df['category'].unique().tolist()
    numerical_features = ['serving_size_g', 'calories', 'protein_g', 'fat_total_g', 'carbs_g', 'fiber_g', 'sugar_g', 'calcium_mg', 'iron_mg', 'potassium_mg', 'sodium_mg', 'vitamin_c_mg', 'vitamin_a_iu', 'vitamin_d_iu', 'cholesterol_mg', 'saturated_fat_g', 'trans_fat_g']
    all_features = numerical_features + ['category', 'brand']
    print("Assets loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading assets: {e}")
    sys.exit(1)

# --- Helper Functions ---
def preprocess_single_input(data):
    df = pd.DataFrame([data], columns=all_features)
    df['calories_per_100g'] = df['calories'] / df['serving_size_g'] * 100
    df['protein_per_100g'] = df['protein_g'] / df['serving_size_g'] * 100
    df['fat_per_100g'] = df['fat_total_g'] / df['serving_size_g'] * 100
    df['carbs_per_100g'] = df['carbs_g'] / df['serving_size_g'] * 100
    total_macros = df['protein_g'] + df['fat_total_g'] + df['carbs_g'] + 1e-6
    df['protein_ratio'] = df['protein_g'] / total_macros
    df['fat_ratio'] = df['fat_total_g'] / total_macros
    df['carb_ratio'] = df['carbs_g'] / total_macros
    X_new = df[numerical_features + ['category', 'brand'] + [
        'calories_per_100g', 'protein_per_100g', 'fat_per_100g', 'carbs_per_100g',
        'protein_ratio', 'fat_ratio', 'carb_ratio'
    ]].copy()
    X_new_processed = preprocessor.transform(X_new)
    return X_new_processed

def predict_nutrition_score(processed_data):
    return model.predict(processed_data)[0]

def get_similar_foods(input_data, top_n=5):
    input_df = pd.DataFrame([input_data])
    if not input_df.empty and numerical_features:
        cleaned_numerical = cleaned_df[numerical_features].fillna(cleaned_df[numerical_features].mean())
        input_numerical = input_df[numerical_features].fillna(cleaned_df[numerical_features].mean())
        distances = euclidean_distances(input_numerical, cleaned_numerical)[0]
        similar_indices = distances.argsort()[:top_n]
        similar_foods = cleaned_df.iloc[similar_indices][['name', 'category', 'nutrition_score'] + numerical_features]
        return similar_foods
    return pd.DataFrame()

def analyze_diet(daily_meals_selection):
    total_nutrients = {'calories': 0, 'protein_g': 0, 'fat_total_g': 0, 'carbs_g': 0, 'fiber_g': 0, 'sugar_g': 0, 'sodium_mg': 0}
    meal_details = {}
    if cleaned_df is not None:
        for meal, item_names in daily_meals_selection.items():
            if item_names:
                meal_nutrient = {'calories': 0, 'protein_g': 0, 'fat_total_g': 0, 'carbs_g': 0, 'fiber_g': 0, 'sugar_g': 0, 'sodium_mg': 0}
                meal_items_data = []
                for item_name in item_names:
                    if item_name and item_name != "Select Food":
                        food_df = cleaned_df[cleaned_df['name'] == item_name]
                        if not food_df.empty:
                            food = food_df.iloc[0]
                            calories = food['calories']
                            protein = food['protein_g']
                            fat = food['fat_total_g']
                            carbs = food['carbs_g']
                            fiber = food['fiber_g']
                            sugar = food['sugar_g']
                            sodium = food['sodium_mg']
                            meal_nutrient['calories'] += calories
                            meal_nutrient['protein_g'] += protein
                            meal_nutrient['fat_total_g'] += fat
                            meal_nutrient['carbs_g'] += carbs
                            meal_nutrient['fiber_g'] += fiber
                            meal_nutrient['sugar_g'] += sugar
                            meal_nutrient['sodium_mg'] += sodium
                            meal_items_data.append(food[['name', 'calories', 'protein_g', 'fat_total_g', 'carbs_g', 'fiber_g', 'sugar_g', 'sodium_mg']])
                        else:
                            st.error(f"Food item '{item_name}' not found in the dataset.")
                if meal_items_data:
                    if len(meal_items_data) > 0:
                        meal_details[meal] = pd.concat(meal_items_data)
                        for nutrient, value in meal_nutrient.items():
                            total_nutrients[nutrient] += value
    return total_nutrients, meal_details

def visualize_nutrients(food_name):
    food_data = cleaned_df[cleaned_df['name'] == food_name].iloc[0]
    macros = ['carbs_g', 'protein_g', 'fat_total_g']
    macro_data = food_data[macros].to_dict()
    fig_pie_macros = px.pie(names=macros, values=list(macro_data.values()), title=f"Macronutrient Distribution in {food_name}", hole=0.3)
    fig_pie_macros.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})')
    fig_pie_macros.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    st.plotly_chart(fig_pie_macros)
    st.markdown("This pie chart illustrates the proportion of the three primary macronutrients - carbohydrates, protein, and fats - in the selected food item. Each slice represents one of these nutrients, with its size indicating its percentage contribution to the total weight of these macronutrients.")

    micronutrients = ['calcium_mg', 'iron_mg', 'potassium_mg', 'vitamin_c_mg']
    micro_data = food_data[micronutrients].to_dict()
    fig_bar_micro = px.bar(x=micronutrients, y=list(micro_data.values()), title=f"Key Micronutrients in {food_name}", labels={'y': 'Amount (mg)', 'x': 'Micronutrient'})
    fig_bar_micro.update_traces(hovertemplate='%{x}: %{y:.2f} mg')
    st.plotly_chart(fig_bar_micro)
    st.markdown("This bar chart displays the quantity of several key micronutrients present in the selected food. The height of each bar corresponds to the amount (in milligrams) of the respective micronutrient.")

    detailed_df_viz = pd.DataFrame(food_data[numerical_features].drop(['serving_size_g']).rename(index=lambda x: x.replace('_g', ' (g)').replace('_mg', ' (mg)').replace('_iu', ' (IU)')).to_dict(), index=[food_name]).T.rename(columns={food_name: 'Amount per Serving'})
    st.subheader("Detailed Nutrient Breakdown")
    st.dataframe(detailed_df_viz.style.format(precision=2))
    st.markdown("This table provides a comprehensive list of the nutritional components of the selected food, detailing the amount of each nutrient present in a standard serving.")

    if 'nutrition_score' in food_data:
        st.subheader("Overall Nutritional Score")
        st.metric("Nutrition Score (0-100)", f"{food_data['nutrition_score']:.2f}")
        st.markdown("This score offers a general assessment of the nutritional quality of the food, with higher scores typically indicating a more nutrient-dense profile.")

    st.markdown("---")
    st.subheader("Understanding Key Nutrients")
    st.markdown(
        """
        * **Carbohydrates, Protein, and Fat:** These are the primary macronutrients, serving as the body's main sources of energy. Protein is also crucial for tissue building and repair, while fats are essential for hormone production and nutrient absorption.
        * **Fiber:** A type of carbohydrate that the body cannot digest. It aids in digestion, helps regulate blood sugar and cholesterol levels, and promotes feelings of fullness.
        * **Sugar:** A simple form of carbohydrate that provides quick energy. It's important to consume natural sugars found in whole foods over added sugars.
        * **Calcium:** A mineral vital for strong bones and teeth, as well as muscle function and nerve signaling.
        * **Iron:** Essential for the transport of oxygen in the blood. Deficiency can lead to fatigue and other health issues.
        * **Potassium:** An electrolyte that helps regulate fluid balance, nerve signals, and muscle contractions, including the heart.
        * **Vitamin C:** An antioxidant that supports the immune system, aids in collagen production, and enhances iron absorption.
        """
    )

    with st.expander("Explore Nutrient Distribution Across Categories"):
        nutrient_to_explore = st.selectbox("Select a nutrient to see its distribution across food categories:", numerical_features)
        fig_dist = px.box(cleaned_df, x='category', y=nutrient_to_explore, title=f"{nutrient_to_explore} Distribution by Food Category")
        fig_dist.update_traces(hovertemplate='Category: %{x}<br>Amount: %{y:.2f}')
        st.plotly_chart(fig_dist)
        st.markdown(f"This box plot visualizes how the levels of '{nutrient_to_explore}' vary across different food categories. The box represents the middle 50% of the data, the line inside the box is the median, and the whiskers extend to show the range of values, with potential outliers displayed as individual points.")

def get_portion_guidance(age, sex, activity_level):
    guidance = {}
    if 18 <= age <= 60:
        if sex == "Male":
            if activity_level == "Sedentary":
                guidance['Grains (cups cooked)'] = "6-7 (e.g., rice, roti)"
                guidance['Vegetables (cups)'] = "2.5-3 (mix of colorful veggies)"
                guidance['Fruits (cups)'] = "2 (whole fruits preferred)"
                guidance['Protein Foods (ounces)'] = "5.5-6.5 (lean meats, lentils, beans)"
                guidance['Dairy (cups)'] = "2 (milk, yogurt)"
                guidance['Oils (teaspoons)'] = "6-7"
            elif activity_level == "Lightly Active":
                guidance['Grains (cups cooked)'] = "7-8"
                guidance['Vegetables (cups)'] = "3"
                guidance['Fruits (cups)'] = "2"
                guidance['Protein Foods (ounces)'] = "6"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "7"
            elif activity_level == "Moderately Active":
                guidance['Grains (cups cooked)'] = "8-9"
                guidance['Vegetables (cups)'] = "3-4"
                guidance['Fruits (cups)'] = "2.5"
                guidance['Protein Foods (ounces)'] = "6.5-7"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "7-8"
            elif activity_level == "Very Active":
                guidance['Grains (cups cooked)'] = "9-10"
                guidance['Vegetables (cups)'] = "3.5"
                guidance['Fruits (cups)'] = "2.5"
                guidance['Protein Foods (ounces)'] = "7"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "8"
            elif activity_level == "Extra Active":
                guidance['Grains (cups cooked)'] = "10+"
                guidance['Vegetables (cups)'] = "4+"
                guidance['Fruits (cups)'] = "3"
                guidance['Protein Foods (ounces)'] = "7+"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "8+"
        elif sex == "Female":
            if activity_level == "Sedentary":
                guidance['Grains (cups cooked)'] = "5-6"
                guidance['Vegetables (cups)'] = "2-2.5"
                guidance['Fruits (cups)'] = "1.5"
                guidance['Protein Foods (ounces)'] = "5-5.5"
                guidance['Dairy (cups)'] = "2"
                guidance['Oils (teaspoons)'] = "5-6"
            elif activity_level == "Lightly Active":
                guidance['Grains (cups cooked)'] = "6"
                guidance['Vegetables (cups)'] = "2.5"
                guidance['Fruits (cups)'] = "1.5"
                guidance['Protein Foods (ounces)'] = "5.5"
                guidance['Dairy (cups)'] = "2.5"
                guidance['Oils (teaspoons)'] = "6"
            elif activity_level == "Moderately Active":
                guidance['Grains (cups cooked)'] = "6-7"
                guidance['Vegetables (cups)'] = "2.5-3"
                guidance['Fruits (cups)'] = "2"
                guidance['Protein Foods (ounces)'] = "5.5-6"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "6-7"
            elif activity_level == "Very Active":
                guidance['Grains (cups cooked)'] = "7-8"
                guidance['Vegetables (cups)'] = "3"
                guidance['Fruits (cups)'] = "2"
                guidance['Protein Foods (ounces)'] = "6"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "7"
            elif activity_level == "Extra Active":
                guidance['Grains (cups cooked)'] = "8+"
                guidance['Vegetables (cups)'] = "3.5+"
                guidance['Fruits (cups)'] = "2.5"
                guidance['Protein Foods (ounces)'] = "6.5+"
                guidance['Dairy (cups)'] = "3"
                guidance['Oils (teaspoons)'] = "7+"
        else:
            guidance['Note'] = "Portion guidance is primarily for adults (18-60 years). Consult specific guidelines for other age groups."
    else:
        guidance['Note'] = "Portion guidance is primarily for adults (18-60 years). Consult specific guidelines for other age groups."
    return guidance

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Poshan Darshan: Your Smart Nutrition Hub")
    st.title("Poshan Darshan: Your Smart Nutrition Hub :eyes:")
    st.markdown("<style>h1, h2, h3 {color: #264a3d;}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Explore with Poshan Darshan:")
        app_mode = option_menu(
            "Main Menu",
            ["Analyze Food & Discover", "Compare Foods Intelligently", "Visualize Nutrients Deeply", "Track Your Daily Intake", "Personalized Guidance"],
            icons=['search', 'scale', 'chart -fill', 'list-check', 'person-fill'],
            menu_icon="menu-app",
            default_index=0,
            styles={
                "container": {"padding": "15px !important", "background-color": "#f4f7f6"},
                "icon": {"color": "#38761d", "font-size": "22px"},
                "nav-link": {"font-size": "17px", "text-align": "left", "margin": "7px", "--hover-color": "#e6f0e9", "color": "#1c392b"},
                "nav-link-selected": {"background-color": "#64b752 !important", "color": "white"},
            }
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("Developed with :heart: for a healthier you.")
        st.sidebar.markdown("[Check out our data source](your_data_source_link)") # Replace with your data source

    col_main, _ = st.columns([3, 1])

    with col_main:
        if app_mode == 'Analyze Food & Discover':
            st.subheader("Unlock the Nutritional Secrets of Your Food")
            selected_food = st.selectbox("Select a food item for an in-depth analysis:", food_names)
            if selected_food:
                food_data = cleaned_df[cleaned_df['name'] == selected_food].iloc[0]
                st.markdown(f"### Unveiling the Nutrients in **{selected_food}**")
                cols_metrics = st.columns(3)
                cols_metrics[0].metric("Calories (kcal)", f"{food_data['calories']:.0f}")
                cols_metrics[1].metric("Protein (g)", f"{food_data['protein_g']:.2f}")
                cols_metrics[2].metric("Fat (g)", f"{food_data['fat_total_g']:.2f}")
                cols_metrics[0].metric("Carbs (g)", f"{food_data['carbs_g']:.2f}")
                cols_metrics[1].metric("Fiber (g)", f"{food_data['fiber_g']:.2f}")
                cols_metrics[2].metric("Sugar (g)", f"{food_data['sugar_g']:.2f}")

                key_highlights = []
                if 'fiber_g' in food_data and food_data['fiber_g'] > cleaned_df['fiber_g'].median():
                    key_highlights.append("a good source of fiber")
                if 'protein_g' in food_data and food_data['protein_g'] > cleaned_df['protein_g'].median():
                    key_highlights.append("relatively high in protein")
                if 'vitamin_c_mg' in food_data and food_data['vitamin_c_mg'] > 10:
                    key_highlights.append("contains a notable amount of Vitamin C")

                if key_highlights:
                    st.markdown(f"This food is {', and '.join(key_highlights)}.")

                with st.expander("🔬 Detailed Nutritional Profile"):
                    detailed_df = pd.DataFrame(food_data[numerical_features].drop(['serving_size_g']).rename(index=lambda x: x.replace('_g', ' (g)').replace('_mg', ' (mg)').replace('_iu', ' (IU)')).to_dict(), index=[selected_food]).T.rename(columns={selected_food: 'Amount per Serving'})
                    detailed_df['Role'] = ''
                    if 'calories (kcal)' in detailed_df.index:
                        detailed_df.loc['calories (kcal)', 'Role'] = '(Energy)'
                    if 'protein (g)' in detailed_df.index:
                        detailed_df.loc['protein (g)', 'Role'] = '(Building Blocks)'
                    if 'calcium (mg)' in detailed_df.index:
                        detailed_df.loc['calcium (mg)', 'Role'] = '(Bone Health)'
                    if 'vitamin_c (mg)' in detailed_df.index:
                        detailed_df.loc['vitamin_c (mg)', 'Role'] = '(Immunity)'
                    detailed_df_display = detailed_df.reset_index().rename(columns={'index': 'Nutrient'})
                    st.dataframe(detailed_df_display)
                    st.markdown("This table provides a comprehensive breakdown of the nutrients found in one serving of the selected food item, along with some notes on their primary roles.")

                if 'protein_per_100g' in food_data and 'serving_size_g' in food_data:
                    recommended_protein = 50
                    protein_per_serving = (food_data['protein_per_100g'] / 100) * food_data['serving_size_g']
                    protein_percentage = (protein_per_serving / recommended_protein) * 100
                    st.subheader("Protein Contribution")
                    st.progress(protein_percentage / 100 if protein_percentage <= 100 else 1.0)
                    st.markdown(f"One serving provides approximately {protein_percentage:.0f}% of the example daily recommended intake for protein.")

                st.markdown("---")
                st.subheader("Explore Similar Nutritional Alternatives")
                similar_foods_df = get_similar_foods(food_data.to_dict())
                if not similar_foods_df.empty:
                    st.dataframe(similar_foods_df[['name', 'category', 'nutrition_score']].head().style.set_caption("Top 5 Nutritionally Similar Foods"))
                    st.markdown("These foods have a similar overall nutritional profile based on their macronutrient content and nutrition score.")
                else:
                    st.info("No closely similar foods found in our database.")

        elif app_mode == 'Compare Foods Intelligently':
            st.subheader("Side-by-Side Nutritional Showdown")
            col_select1, col_select2 = st.columns(2)
            food1 = col_select1.selectbox("Select the First Food for Comparison:", ["Select Food"] + food_names)
            food2 = col_select2.selectbox("Choose the Second Food to Compare:", ["Select Food"] + food_names)

            if food1 != "Select Food" and food2 != "Select Food" and st.button("Initiate Comparison"):
                food1_data = cleaned_df[cleaned_df['name'] == food1].iloc[[0]][numerical_features + ['name']].set_index('name')
                food2_data = cleaned_df[cleaned_df['name'] == food2].iloc[[0]][numerical_features + ['name']].set_index('name')
                comparison_df = pd.concat([food1_data, food2_data], axis=1, keys=[food1, food2])

                print("--- Comparison DataFrame Info ---")
                print("Column Multi-Index:")
                print(comparison_df.columns)
                print(f"Type of food1: {type(food1)}, Value: '{food1}'")
                print(f"Type of food2: {type(food2)}, Value: '{food2}'")
                print("--- End of Comparison DataFrame Info ---")

                def highlight_diff(row):
                    '''Highlights differences between two food columns in a row.'''
                    style = pd.Series('', index=row.index)
                    try:
                        val1 = row[(food1, row.name)]
                        val2 = row[(food2, row.name)]
                        diff = val1 - val2
                        if diff > 5:
                            style[(food1, row.name)] = 'background-color: lightgreen; color: black;'
                        elif diff < -5:
                            style[(food2, row.name)] = 'background-color: salmon; color: black;'
                    except KeyError as e:
                        st.error(f"KeyError in nutritional comparison (highlight_diff): {e}")
                        print(f"Key not found in row: {e}")
                        print(f"Row index: {row.index}")
                        print(f"food1: '{food1}', food2: '{food2}', row.name: '{row.name}'")
                        return style # Return style to avoid further errors

                    return style

                styled_comparison_df = comparison_df.style.apply(highlight_diff, axis=1).format(precision=2).set_caption(f"Nutritional Comparison: {food1} vs {food2}")
                st.subheader(f"Nutritional Face-off: {food1} vs {food2}")
                st.dataframe(styled_comparison_df)

                st.subheader("Visualizing the Macronutrient Balance")
                col_pie_compare1, col_pie_compare2 = st.columns(2)
                macros = ['carbs_g', 'protein_g', 'fat_total_g']
                pie1_data = cleaned_df[cleaned_df['name'] == food1].iloc[0][macros].to_dict()
                pie2_data = cleaned_df[cleaned_df['name'] == food2].iloc[0][macros].to_dict()

                fig_pie1 = px.pie(names=macros, values=list(pie1_data.values()), title=f"Macronutrient Mix in {food1}", hole=0.3)
                fig_pie1.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})')
                fig_pie1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                col_pie_compare1.plotly_chart(fig_pie1)
                col_pie_compare1.markdown(f"Distribution of energy-providing nutrients in {food1}.")

                fig_pie2 = px.pie(names=macros, values=list(pie2_data.values()), title=f"Macronutrient Mix in {food2}", hole=0.3)
                fig_pie2.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})')
                fig_pie2.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                col_pie_compare2.plotly_chart(fig_pie2)
                col_pie_compare2.markdown(f"Distribution of energy-providing nutrients in {food2}.")

                fiber_sugar_df = pd.DataFrame({food1: [cleaned_df[cleaned_df['name'] == food1].iloc[0]['fiber_g'], cleaned_df[cleaned_df['name'] == food1].iloc[0]['sugar_g']],
                                              food2: [cleaned_df[cleaned_df['name'] == food2].iloc[0]['fiber_g'], cleaned_df[cleaned_df['name'] == food2].iloc[0]['sugar_g']]},
                                             index=['Fiber (g)', 'Sugar (g)'])
                fig_bar_compare = px.bar(fiber_sugar_df, barmode='group', title="Fiber vs. Sugar Content", labels={'value': 'Amount (g)', 'index': 'Nutrient', 'variable': 'Food'})
                fig_bar_compare.update_traces(hovertemplate='%{variable}: %{y:.2f}g')
                st.plotly_chart(fig_bar_compare)
                st.markdown("A direct comparison of dietary fiber (important for digestion) and sugar (simple carbohydrates) levels.")

                st.subheader("Key Micronutrient Comparison")
                key_micros = ['iron_mg', 'calcium_mg', 'vitamin_c_mg']
                micro_compare_df = pd.DataFrame({
                    food1: cleaned_df[cleaned_df['name'] == food1].iloc[0][key_micros].values,
                    food2: cleaned_df[cleaned_df['name'] == food2].iloc[0][key_micros].values
                }, index=key_micros)
                fig_micro_bar = px.bar(micro_compare_df, barmode='group', title="Comparison of Key Micronutrients", labels={'value': 'Amount (mg)', 'index': 'Micronutrient', 'variable': 'Food'})
                fig_micro_bar.update_traces(hovertemplate='%{variable}: %{y:.2f} mg')
                st.plotly_chart(fig_micro_bar)
                st.markdown("Comparing the levels of essential micronutrients: iron (for oxygen transport), calcium (for bone health), and vitamin C (for immune function).")

                st.subheader("Nutritional Summary")
                summary_text = ""
                if food1 in comparison_df.keys() and food2 in comparison_df.keys():
                    if 'protein_g' in comparison_df[(food1)].index and 'protein_g' in comparison_df[(food2)].index:
                        food1_protein = comparison_df[(food1)].loc['protein_g']
                        food2_protein = comparison_df[(food2)].loc['protein_g']
                        if food1_protein > food2_protein + 5:
                            summary_text += f"{food1} is notably higher in protein than {food2}. "
                        elif food2_protein > food1_protein + 5:
                            summary_text += f"{food2} provides significantly more protein compared to {food1}. "
                    if 'sugar_g' in comparison_df[(food1)].index and 'sugar_g' in comparison_df[(food2)].index:
                        food1_sugar = comparison_df[(food1)].loc['sugar_g']
                        food2_sugar = comparison_df[(food2)].loc['sugar_g']
                        if food1_sugar > food2_sugar + 5:
                            summary_text += f"{food1} contains considerably more sugar than {food2}. "
                        elif food2_sugar > food1_sugar + 5:
                            summary_text += f"{food2} has a much higher sugar content than {food1}. "
                        elif food1_sugar < food2_sugar - 5:
                            summary_text += f"{food1} is significantly lower in sugar than {food2}. "
                        elif food2_sugar < food1_sugar - 5:
                            summary_text += f"{food2} is notably lower in sugar compared to {food1}. "
                    if 'fat_total_g' in comparison_df[(food1)].index and 'fat_total_g' in comparison_df[(food2)].index:
                        food1_fat = comparison_df[(food1)].loc['fat_total_g']
                        food2_fat = comparison_df[(food2)].loc['fat_total_g']
                        if food1_fat > food2_fat + 5:
                            summary_text += f"{food1} has a higher fat content than {food2}. "
                        elif food2_fat > food1_fat + 5:
                            summary_text += f"{food2} is richer in fat than {food1}. "
                    if 'fiber_g' in comparison_df[(food1)].index and 'fiber_g' in comparison_df[(food2)].index:
                        food1_fiber = comparison_df[(food1)].loc['fiber_g']
                        food2_fiber = comparison_df[(food2)].loc['fiber_g']
                        if food1_fiber > food2_fiber + 3:
                            summary_text += f"{food1} provides more dietary fiber than {food2}. "
                        elif food2_fiber > food1_fiber + 3:
                            summary_text += f"{food2} is a better source of fiber than {food1}. "

                if summary_text:
                    st.info(summary_text)
                else:
                    st.info("The two foods have relatively similar macronutrient profiles.")

        elif app_mode == 'Visualize Nutrients Deeply':
            st.subheader("Interactive Nutrient Exploration")
            selected_food_viz = st.selectbox("Select a food item to visualize its nutrient profile:", ["Select Food"] + food_names)
            if selected_food_viz != "Select Food":
                visualize_nutrients(selected_food_viz)

        elif app_mode == 'Track Your Daily Intake':
            st.subheader("Monitor Your Daily Nutritional Journey")
            daily_meals_selection = {}
            for meal in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
                daily_meals_selection[meal] = st.multiselect(f"{meal}: Select Food Items", food_names)

            if st.button("Analyze My Day's Choices"):
                total_nutrients, meal_details = analyze_diet({k: v for k, v in daily_meals_selection.items() if v})
                st.subheader("Your Day's Nutritional Snapshot:")
                summary_df = pd.DataFrame([total_nutrients]).T.rename(columns={0: 'Total'}).style.format(precision=2)
                st.dataframe(summary_df.set_caption("Total Nutrients Consumed Today"))
                st.markdown("This table summarizes the total amount of key nutrients you have selected across all your meals for the day.")

                st.subheader("Meal-by-Meal Nutrient Insights:")
                if meal_details:
                    for meal, items_df in meal_details.items():
                        st.markdown(f"**{meal}:**")
                        if isinstance(items_df, pd.Series):
                            st.dataframe(items_df.to_frame().T.style.format(precision=2).set_caption(f"Nutritional Content for {meal}"))
                        else:
                            st.dataframe(
                                items_df.style.format(precision=2).set_caption(f"Nutritional Content for {meal}"))
                        st.markdown(f"This table details the nutritional content of each food item you selected for {meal}.")
                else:
                    st.info("No meals were tracked today.")

        elif app_mode == 'Personalized Guidance':
            st.subheader("Your Personalized Path to Nutritional Wellness")
            st.info("Adjust the parameters below to get general dietary insights based on your profile.")
            age = st.slider("Your Age (Years):", 10, 80, 30)
            sex = st.radio("Your Gender:", ["Male", "Female"])
            activity_level = st.selectbox("Your Activity Level:", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])
            guidance = None # Initialize guidance

            if st.button("Get Dietary Insights"):
                guidance = get_portion_guidance(age, sex, activity_level)
                st.subheader("General Dietary Insights:")
                if guidance:
                    for food_group, recommendation in guidance.items():
                        st.markdown(f"**{food_group}:** {recommendation}")
                    st.markdown("---")
                    st.markdown(
                        """
                        These insights are based on general nutritional guidelines for adults within a certain age range and activity level.
                        The recommendations aim to provide a balanced intake of essential nutrients.
                        * **Grains:** Provide carbohydrates for energy. Whole grains are preferred for their fiber content.
                        * **Vegetables:** Rich in vitamins, minerals, and fiber. A variety of colors is beneficial.
                        * **Fruits:** Offer vitamins, minerals, and natural sugars. Whole fruits are better than juices.
                        * **Protein Foods:** Essential for muscle building, repair, and various bodily functions. Include lean sources.
                        * **Dairy:** Important for calcium and vitamin D. Choose low-fat or fat-free options.
                        * **Oils:** Provide essential fatty acids but should be consumed in moderation.

                        Activity levels generally correspond to:
                        * **Sedentary:** Minimal physical activity.
                        * **Lightly Active:** Light exercise or activity 1-3 days a week.
                        * **Moderately Active:** Moderate exercise 3-5 days a week.
                        * **Very Active:** Hard exercise 6-7 days a week.
                        * **Extra Active:** Very strenuous activity or daily intense exercise.

                        **Important Note:** These are general suggestions and individual nutritional needs can vary significantly based on health status, specific dietary requirements, and other factors. It is always recommended to consult with a registered dietitian or a healthcare professional for personalized dietary advice.
                        """
                    )
                else:
                    st.warning("Unable to generate specific dietary insights based on the input.")

            st.subheader("Explore Recipes Based on Nutrients (Coming Soon!)")
            st.info("In the future, you'll be able to specify nutrients you're interested in, and we'll suggest relevant recipes!")

if __name__ == '__main__':
    main()

# app.py
# app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import sys
# import plotly.express as px
# import plotly.graph_objects as go
# from streamlit_option_menu import option_menu
# from sklearn.metrics.pairwise import euclidean_distances
# import requests

# # --- Path Configuration ---
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)
# models_dir = os.path.join(project_root, 'src', 'models')
# data_dir = os.path.join(project_root, 'data', 'processed')

# # --- Load Assets ---
# try:
#     model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
#     preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
#     cleaned_df = pd.read_csv(os.path.join(data_dir, 'cleaned_data.csv'))
#     food_names = cleaned_df['name'].unique().tolist()
#     categories = cleaned_df['category'].unique().tolist()
#     numerical_features = ['serving_size_g', 'calories', 'protein_g', 'fat_total_g', 'carbs_g', 'fiber_g', 'sugar_g', 'calcium_mg', 'iron_mg', 'potassium_mg', 'sodium_mg', 'vitamin_c_mg', 'vitamin_a_iu', 'vitamin_d_iu', 'cholesterol_mg', 'saturated_fat_g', 'trans_fat_g']
#     all_features = numerical_features + ['category', 'brand']
#     print("Assets loaded successfully.")
# except FileNotFoundError as e:
#     st.error(f"Error loading assets: {e}")
#     sys.exit(1)

# # --- Helper Functions ---
# def preprocess_single_input(data):
#     df = pd.DataFrame([data], columns=all_features)
#     df['calories_per_100g'] = df['calories'] / df['serving_size_g'] * 100
#     df['protein_per_100g'] = df['protein_g'] / df['serving_size_g'] * 100
#     df['fat_per_100g'] = df['fat_total_g'] / df['serving_size_g'] * 100
#     df['carbs_per_100g'] = df['carbs_g'] / df['serving_size_g'] * 100
#     total_macros = df['protein_g'] + df['fat_total_g'] + df['carbs_g'] + 1e-6
#     df['protein_ratio'] = df['protein_g'] / total_macros
#     df['fat_ratio'] = df['fat_total_g'] / total_macros
#     df['carb_ratio'] = df['carbs_g'] / total_macros
#     X_new = df[numerical_features + ['category', 'brand'] + [
#         'calories_per_100g', 'protein_per_100g', 'fat_per_100g', 'carbs_per_100g',
#         'protein_ratio', 'fat_ratio', 'carb_ratio'
#     ]].copy()
#     X_new_processed = preprocessor.transform(X_new)
#     return X_new_processed

# def predict_nutrition_score(processed_data):
#     return model.predict(processed_data)[0]

# def get_similar_foods(input_data, top_n=5):
#     input_df = pd.DataFrame([input_data])
#     if not input_df.empty and numerical_features:
#         cleaned_numerical = cleaned_df[numerical_features].fillna(cleaned_df[numerical_features].mean())
#         input_numerical = input_df[numerical_features].fillna(cleaned_df[numerical_features].mean())
#         distances = euclidean_distances(input_numerical, cleaned_numerical)[0]
#         similar_indices = distances.argsort()[:top_n]
#         similar_foods = cleaned_df.iloc[similar_indices][['name', 'category', 'nutrition_score'] + numerical_features]
#         return similar_foods
#     return pd.DataFrame()

# def analyze_diet(daily_meals_selection):
#     total_nutrients = {'calories': 0, 'protein_g': 0, 'fat_total_g': 0, 'carbs_g': 0, 'fiber_g': 0, 'sugar_g': 0, 'sodium_mg': 0}
#     meal_details = {}
#     if cleaned_df is not None:
#         for meal, item_names in daily_meals_selection.items():
#             if item_names:
#                 meal_nutrient = {'calories': 0, 'protein_g': 0, 'fat_total_g': 0, 'carbs_g': 0, 'fiber_g': 0, 'sugar_g': 0, 'sodium_mg': 0}
#                 meal_items_data = []
#                 for item_name in item_names:
#                     if item_name and item_name != "Select Food":
#                         food_df = cleaned_df[cleaned_df['name'] == item_name]
#                         if not food_df.empty:
#                             food = food_df.iloc[0]
#                             calories = food['calories']
#                             protein = food['protein_g']
#                             fat = food['fat_total_g']
#                             carbs = food['carbs_g']
#                             fiber = food['fiber_g']
#                             sugar = food['sugar_g']
#                             sodium = food['sodium_mg']
#                             meal_nutrient['calories'] += calories
#                             meal_nutrient['protein_g'] += protein
#                             meal_nutrient['fat_total_g'] += fat
#                             meal_nutrient['carbs_g'] += carbs
#                             meal_nutrient['fiber_g'] += fiber
#                             meal_nutrient['sugar_g'] += sugar
#                             meal_nutrient['sodium_mg'] += sodium
#                             meal_items_data.append(food[['name', 'calories', 'protein_g', 'fat_total_g', 'carbs_g', 'fiber_g', 'sugar_g', 'sodium_mg']])
#                         else:
#                             st.error(f"Food item '{item_name}' not found in the dataset.")
#                 if meal_items_data:
#                     if len(meal_items_data) > 0:
#                         meal_details[meal] = pd.concat(meal_items_data)
#                         for nutrient, value in meal_nutrient.items():
#                             total_nutrients[nutrient] += value
#     return total_nutrients, meal_details

# def visualize_nutrients(food_name):
#     food_data = cleaned_df[cleaned_df['name'] == food_name].iloc[0]
#     macros = ['carbs_g', 'protein_g', 'fat_total_g']
#     macro_data = food_data[macros].to_dict()
#     fig_pie_macros = px.pie(names=macros, values=list(macro_data.values()), title=f"Macronutrient Distribution in {food_name}", hole=0.3)
#     fig_pie_macros.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})')
#     fig_pie_macros.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
#     st.plotly_chart(fig_pie_macros)
#     st.markdown("This pie chart illustrates the proportion of the three primary macronutrients - carbohydrates, protein, and fats - in the selected food item. Each slice represents one of these nutrients, with its size indicating its percentage contribution to the total weight of these macronutrients.")

#     micronutrients = ['calcium_mg', 'iron_mg', 'potassium_mg', 'vitamin_c_mg']
#     micro_data = food_data[micronutrients].to_dict()
#     fig_bar_micro = px.bar(x=micronutrients, y=list(micro_data.values()), title=f"Key Micronutrients in {food_name}", labels={'y': 'Amount (mg)', 'x': 'Micronutrient'})
#     fig_bar_micro.update_traces(hovertemplate='%{x}: %{y:.2f} mg')
#     st.plotly_chart(fig_bar_micro)
#     st.markdown("This bar chart displays the quantity of several key micronutrients present in the selected food. The height of each bar corresponds to the amount (in milligrams) of the respective micronutrient.")

#     detailed_df_viz = pd.DataFrame(food_data[numerical_features].drop(['serving_size_g']).rename(index=lambda x: x.replace('_g', ' (g)').replace('_mg', ' (mg)').replace('_iu', ' (IU)')).to_dict(), index=[food_name]).T.rename(columns={food_name: 'Amount per Serving'})
#     st.subheader("Detailed Nutrient Breakdown")
#     st.dataframe(detailed_df_viz.style.format(precision=2))
#     st.markdown("This table provides a comprehensive list of the nutritional components of the selected food, detailing the amount of each nutrient present in a standard serving.")

#     if 'nutrition_score' in food_data:
#         st.subheader("Overall Nutritional Score")
#         st.metric("Nutrition Score (0-100)", f"{food_data['nutrition_score']:.2f}")
#         st.markdown("This score offers a general assessment of the nutritional quality of the food, with higher scores typically indicating a more nutrient-dense profile.")

#     st.markdown("---")
#     st.subheader("Understanding Key Nutrients")
#     st.markdown(
#         """
#         * **Carbohydrates, Protein, and Fat:** These are the primary macronutrients, serving as the body's main sources of energy. Protein is also crucial for tissue building and repair, while fats are essential for hormone production and nutrient absorption.
#         * **Fiber:** A type of carbohydrate that the body cannot digest. It aids in digestion, helps regulate blood sugar and cholesterol levels, and promotes feelings of fullness.
#         * **Sugar:** A simple form of carbohydrate that provides quick energy. It's important to consume natural sugars found in whole foods over added sugars.
#         * **Calcium:** A mineral vital for strong bones and teeth, as well as muscle function and nerve signaling.
#         * **Iron:** Essential for the transport of oxygen in the blood. Deficiency can lead to fatigue and other health issues.
#         * **Potassium:** An electrolyte that helps regulate fluid balance, nerve signals, and muscle contractions, including the heart.
#         * **Vitamin C:** An antioxidant that supports the immune system, aids in collagen production, and enhances iron absorption.
#         """
#     )

#     with st.expander("Explore Nutrient Distribution Across Categories"):
#         nutrient_to_explore = st.selectbox("Select a nutrient to see its distribution across food categories:", numerical_features)
#         fig_dist = px.box(cleaned_df, x='category', y=nutrient_to_explore, title=f"{nutrient_to_explore} Distribution by Food Category")
#         fig_dist.update_traces(hovertemplate='Category: %{x}<br>Amount: %{y:.2f}')
#         st.plotly_chart(fig_dist)
#         st.markdown(f"This box plot visualizes how the levels of '{nutrient_to_explore}' vary across different food categories. The box represents the middle 50% of the data, the line inside the box is the median, and the whiskers extend to show the range of values, with potential outliers displayed as individual points.")

# def get_portion_guidance(age, sex, activity_level):
#     guidance = {}
#     if 18 <= age <= 60:
#         if sex == "Male":
#             if activity_level == "Sedentary":
#                 guidance['Grains (cups cooked)'] = "6-7 (e.g., rice, roti)"
#                 guidance['Vegetables (cups)'] = "2.5-3 (mix of colorful veggies)"
#                 guidance['Fruits (cups)'] = "2 (whole fruits preferred)"
#                 guidance['Protein Foods (ounces)'] = "5.5-6.5 (lean meats, lentils, beans)"
#                 guidance['Dairy (cups)'] = "2 (milk, yogurt)"
#                 guidance['Oils (teaspoons)'] = "6-7"
#             elif activity_level == "Lightly Active":
#                 guidance['Grains (cups cooked)'] = "7-8"
#                 guidance['Vegetables (cups)'] = "3"
#                 guidance['Fruits (cups)'] = "2"
#                 guidance['Protein Foods (ounces)'] = "6"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "7"
#             elif activity_level == "Moderately Active":
#                 guidance['Grains (cups cooked)'] = "8-9"
#                 guidance['Vegetables (cups)'] = "3-4"
#                 guidance['Fruits (cups)'] = "2.5"
#                 guidance['Protein Foods (ounces)'] = "6.5-7"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "7-8"
#             elif activity_level == "Very Active":
#                 guidance['Grains (cups cooked)'] = "9-10"
#                 guidance['Vegetables (cups)'] = "3.5"
#                 guidance['Fruits (cups)'] = "2.5"
#                 guidance['Protein Foods (ounces)'] = "7"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "8"
#             elif activity_level == "Extra Active":
#                 guidance['Grains (cups cooked)'] = "10+"
#                 guidance['Vegetables (cups)'] = "4+"
#                 guidance['Fruits (cups)'] = "3"
#                 guidance['Protein Foods (ounces)'] = "7+"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "8+"
#         elif sex == "Female":
#             if activity_level == "Sedentary":
#                 guidance['Grains (cups cooked)'] = "5-6"
#                 guidance['Vegetables (cups)'] = "2-2.5"
#                 guidance['Fruits (cups)'] = "1.5"
#                 guidance['Protein Foods (ounces)'] = "5-5.5"
#                 guidance['Dairy (cups)'] = "2"
#                 guidance['Oils (teaspoons)'] = "5-6"
#             elif activity_level == "Lightly Active":
#                 guidance['Grains (cups cooked)'] = "6"
#                 guidance['Vegetables (cups)'] = "2.5"
#                 guidance['Fruits (cups)'] = "1.5"
#                 guidance['Protein Foods (ounces)'] = "5.5"
#                 guidance['Dairy (cups)'] = "2.5"
#                 guidance['Oils (teaspoons)'] = "6"
#             elif activity_level == "Moderately Active":
#                 guidance['Grains (cups cooked)'] = "6-7"
#                 guidance['Vegetables (cups)'] = "2.5-3"
#                 guidance['Fruits (cups)'] = "2"
#                 guidance['Protein Foods (ounces)'] = "5.5-6"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "6-7"
#             elif activity_level == "Very Active":
#                 guidance['Grains (cups cooked)'] = "7-8"
#                 guidance['Vegetables (cups)'] = "3"
#                 guidance['Fruits (cups)'] = "2"
#                 guidance['Protein Foods (ounces)'] = "6"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "7"
#             elif activity_level == "Extra Active":
#                 guidance['Grains (cups cooked)'] = "8+"
#                 guidance['Vegetables (cups)'] = "3.5+"
#                 guidance['Fruits (cups)'] = "2.5"
#                 guidance['Protein Foods (ounces)'] = "6.5+"
#                 guidance['Dairy (cups)'] = "3"
#                 guidance['Oils (teaspoons)'] = "7+"
#         else:
#             guidance['Note'] = "Portion guidance is primarily for adults (18-60 years). Consult specific guidelines for other age groups."
#     else:
#         guidance['Note'] = "Portion guidance is primarily for adults (18-60 years). Consult specific guidelines for other age groups."
#     return guidance

# # --- Main App ---
# def main():
#     st.set_page_config(layout="wide", page_title="Poshan Darshan: Your Smart Nutrition Hub")
#     st.title("Poshan Darshan: Your Smart Nutrition Hub :eyes:")
#     st.markdown("<style>h1, h2, h3 {color: #264a3d;}</style>", unsafe_allow_html=True)

#     with st.sidebar:
#         st.subheader("Explore with Poshan Darshan:")
#         app_mode = option_menu(
#             "Main Menu",
#             ["Analyze Food & Discover", "Compare Foods Intelligently", "Visualize Nutrients Deeply", "Track Your Daily Intake", "Personalized Guidance"],
#             icons=['search', 'scale', 'chart-fill', 'list-check', 'person-fill'],
#             menu_icon="menu-app",
#             default_index=0,
#             styles={
#                 "container": {"padding": "15px !important", "background-color": "#f4f7f6"},
#                 "icon": {"color": "#38761d", "font-size": "22px"},
#                 "nav-link": {"font-size": "17px", "text-align": "left", "margin": "7px", "--hover-color": "#e6f0e9", "color": "#1c392b"},
#                 "nav-link-selected": {"background-color": "#64b752 !important", "color": "white"},
#             }
#         )

#         st.sidebar.markdown("---")
#         st.sidebar.markdown("Developed with :heart: for a healthier you.")
#         st.sidebar.markdown("[Check out our data source](your_data_source_link)") # Replace with your data source

#     col_main, _ = st.columns([3, 1])

#     with col_main:
#         if app_mode == 'Analyze Food & Discover':
#             st.subheader("Unlock the Nutritional Secrets of Your Food", anchor=False)
#             selected_food = st.selectbox("Select a food item for an in-depth analysis:", food_names)
#             if selected_food:
#                 food_data = cleaned_df[cleaned_df['name'] == selected_food].iloc[0]
#                 st.markdown(f"### Unveiling the Nutrients in **{selected_food}**")
#                 cols_metrics = st.columns(3)
#                 cols_metrics[0].metric("Calories (kcal)", f"{food_data['calories']:.0f}")
#                 cols_metrics[1].metric("Protein (g)", f"{food_data['protein_g']:.2f}")
#                 cols_metrics[2].metric("Fat (g)", f"{food_data['fat_total_g']:.2f}")
#                 cols_metrics[0].metric("Carbs (g)", f"{food_data['carbs_g']:.2f}")
#                 cols_metrics[1].metric("Fiber (g)", f"{food_data['fiber_g']:.2f}")
#                 cols_metrics[2].metric("Sugar (g)", f"{food_data['sugar_g']:.2f}")

#                 key_highlights = []
#                 if 'fiber_g' in food_data and food_data['fiber_g'] > cleaned_df['fiber_g'].median():
#                     key_highlights.append("a good source of fiber")
#                 if 'protein_g' in food_data and food_data['protein_g'] > cleaned_df['protein_g'].median():
#                     key_highlights.append("relatively high in protein")
#                 if 'vitamin_c_mg' in food_data and food_data['vitamin_c_mg'] > 10:
#                     key_highlights.append("contains a notable amount of Vitamin C")

#                 if key_highlights:
#                     st.markdown(f"This food is {', and '.join(key_highlights)}.")

#                 with st.expander("🔬 Detailed Nutritional Profile"):
#                     detailed_df = pd.DataFrame(food_data[numerical_features].drop(['serving_size_g']).rename(index=lambda x: x.replace('_g', ' (g)').replace('_mg', ' (mg)').replace('_iu', ' (IU)')).to_dict(), index=[selected_food]).T.rename(columns={selected_food: 'Amount per Serving'})
#                     detailed_df['Role'] = ''
#                     if 'calories (kcal)' in detailed_df.index:
#                         detailed_df.loc['calories (kcal)', 'Role'] = '(Energy)'
#                     if 'protein (g)' in detailed_df.index:
#                         detailed_df.loc['protein (g)', 'Role'] = '(Building Blocks)'
#                     if 'calcium (mg)' in detailed_df.index:
#                         detailed_df.loc['calcium (mg)', 'Role'] = '(Bone Health)'
#                     if 'vitamin_c (mg)' in detailed_df.index:
#                         detailed_df.loc['vitamin_c (mg)', 'Role'] = '(Immunity)'
#                     detailed_df_display = detailed_df.reset_index().rename(columns={'index': 'Nutrient'})
#                     st.dataframe(detailed_df_display)
#                     st.markdown("This table provides a comprehensive breakdown of the nutrients found in one serving of the selected food item, along with some notes on their primary roles.")

#                 if 'protein_per_100g' in food_data and 'serving_size_g' in food_data:
#                     recommended_protein = 50
#                     protein_per_serving = (food_data['protein_per_100g'] / 100) * food_data['serving_size_g']
#                     protein_percentage = (protein_per_serving / recommended_protein) * 100
#                     st.subheader("Protein Contribution")
#                     st.progress(protein_percentage / 100 if protein_percentage <= 100 else 1.0)
#                     st.markdown(f"One serving provides approximately {protein_percentage:.0f}% of the example daily recommended intake for protein.")

#                 st.markdown("---")
#                 st.subheader("Explore Similar Nutritional Alternatives")
#                 similar_foods_df = get_similar_foods(food_data.to_dict())
#                 if not similar_foods_df.empty:
#                     st.dataframe(similar_foods_df[['name', 'category', 'nutrition_score']].head().style.set_caption("Top 5 Nutritionally Similar Foods"))
#                     st.markdown("These foods have a similar overall nutritional profile based on their macronutrient content and nutrition score.")
#                 else:
#                     st.info("No closely similar foods found in our database.")

#         elif app_mode == 'Compare Foods Intelligently':
#             st.subheader("Side-by-Side Nutritional Showdown", anchor=False) # Increased prominence

#             col_select1, col_select2 = st.columns(2)
#             food1 = col_select1.selectbox("Select the First Food for Comparison:", ["Select Food"] + food_names)
#             food2 = col_select2.selectbox("Choose the Second Food to Compare:", ["Select Food"] + food_names)

#             if food1 != "Select Food" and food2 != "Select Food" and st.button("Compare Now"):
#                 food1_data = cleaned_df[cleaned_df['name'] == food1].iloc[0]
#                 food2_data = cleaned_df[cleaned_df['name'] == food2].iloc[0]

#                 st.subheader(f"Nutritional Comparison: **{food1}** vs **{food2}**", anchor=False) # Increased prominence

#                 # --- Macronutrient Balance Visualization ---
#                 st.subheader("Macronutrient Breakdown", anchor=False)
#                 col_macro1, col_macro2 = st.columns(2)
#                 macros = ['carbs_g', 'protein_g', 'fat_total_g']

#                 pie1_data = food1_data[macros].to_dict()
#                 fig_pie1 = px.pie(names=macros, values=list(pie1_data.values()),
#                                   title=f"{food1}",
#                                   hole=0.3,
#                                   labels={'names': 'Macronutrient', 'values': 'Amount (g)'})
#                 fig_pie1.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})',
#                                       textinfo='percent+label', textfont_size=14) # Larger text on slices
#                 fig_pie1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',
#                                       title_font_size=18, legend_font_size=14) # Larger title and legend
#                 col_macro1.plotly_chart(fig_pie1, use_container_width=True) # Ensure it uses the full column width
#                 col_macro1.markdown(f"Macronutrient distribution in **{food1}**.")

#                 pie2_data = food2_data[macros].to_dict()
#                 fig_pie2 = px.pie(names=macros, values=list(pie2_data.values()),
#                                   title=f"{food2}",
#                                   hole=0.3,
#                                   labels={'names': 'Macronutrient', 'values': 'Amount (g)'})
#                 fig_pie2.update_traces(hovertemplate='%{label}: %{value:.2f}g (%{percent:.1%})',
#                                       textinfo='percent+label', textfont_size=14) # Larger text on slices
#                 fig_pie2.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',
#                                       title_font_size=18, legend_font_size=14) # Larger title and legend
#                 col_macro2.plotly_chart(fig_pie2, use_container_width=True) # Ensure it uses the full column width
#                 col_macro2.markdown(f"Macronutrient distribution in **{food2}**.")

#                 st.markdown("---")

#                 # --- Fiber and Sugar Comparison ---
#                 st.subheader("Fiber and Sugar Content", anchor=False)
#                 fiber_sugar_df = pd.DataFrame({
#                     food1: [food1_data['fiber_g'], food1_data['sugar_g']],
#                     food2: [food2_data['fiber_g'], food2_data['sugar_g']]
#                 }, index=['Fiber (g)', 'Sugar (g)'])

#                 fig_bar_compare_fs = px.bar(fiber_sugar_df, barmode='group',
#                                             title="Fiber vs. Sugar",
#                                             labels={'value': 'Amount (g)', 'index': 'Nutrient', 'variable': 'Food'},
#                                             color_discrete_sequence=px.colors.qualitative.Set2)
#                 fig_bar_compare_fs.update_traces(hovertemplate='%{variable}: %{y:.2f}g')
#                 fig_bar_compare_fs.update_layout(title_font_size=18, legend_font_size=14,
#                                                  xaxis_title_font_size=14, yaxis_title_font_size=14,
#                                                  xaxis_tickfont_size=12, yaxis_tickfont_size=12) # Larger fonts
#                 st.plotly_chart(fig_bar_compare_fs, use_container_width=True)
#                 st.markdown("Comparison of fiber and sugar content.")

#                 st.markdown("---")

#                 # --- Key Micronutrient Comparison ---
#                 st.subheader("Key Micronutrient Levels", anchor=False)
#                 key_micros = ['iron_mg', 'calcium_mg', 'vitamin_c_mg']
#                 micro_compare_df = pd.DataFrame({
#                     food1: food1_data[key_micros].values,
#                     food2: food2_data[key_micros].values
#                 }, index=key_micros)
#                 micro_compare_df.index.name = 'Micronutrient'
#                 micro_compare_df = micro_compare_df.reset_index().melt(id_vars='Micronutrient', var_name='Food', value_name='Amount (mg)')

#                 fig_bar_compare_micro = px.bar(micro_compare_df, x='Micronutrient', y='Amount (mg)', color='Food', barmode='group',
#                                                title="Key Micronutrients",
#                                                labels={'Amount (mg)': 'Amount (mg)'},
#                                                color_discrete_sequence=px.colors.qualitative.Pastel1)
#                 fig_bar_compare_micro.update_traces(hovertemplate='%{data.frame[Food][%{curveNumber}]}: %{y:.2f} mg')
#                 fig_bar_compare_micro.update_layout(title_font_size=18, legend_font_size=14,
#                                                     xaxis_title_font_size=14, yaxis_title_font_size=14,
#                                                     xaxis_tickfont_size=12, yaxis_tickfont_size=12) # Larger fonts
#                 st.plotly_chart(fig_bar_compare_micro, use_container_width=True)
#                 st.markdown("Comparison of iron, calcium, and vitamin C levels.")

#                 # --- Nutritional Summary ---
#                 st.subheader("Nutritional Highlights", anchor=False)
#                 summary_text = ""
#                 if 'protein_g' in food1_data and 'protein_g' in food2_data:
#                     protein_diff = food1_data['protein_g'] - food2_data['protein_g']
#                     if abs(protein_diff) > 2:
#                         summary_text += f"**{food1}** has {'more' if protein_diff > 0 else 'less'} protein (+{protein_diff:.2f}g) than **{food2}**. "
#                 if 'sugar_g' in food1_data and 'sugar_g' in food2_data:
#                     sugar_diff = food1_data['sugar_g'] - food2_data['sugar_g']
#                     if abs(sugar_diff) > 3:
#                         summary_text += f"**{food1}** has significantly {'more' if sugar_diff > 0 else 'less'} sugar (+{sugar_diff:.2f}g) than **{food2}**. "
#                 if 'fat_total_g' in food1_data and 'fat_total_g' in food2_data:
#                     fat_diff = food1_data['fat_total_g'] - food2_data['fat_total_g']
#                     if abs(fat_diff) > 2:
#                         summary_text += f"**{food1}** has {'more' if fat_diff > 0 else 'less'} fat (+{fat_diff:.2f}g) than **{food2}**. "
#                 if 'fiber_g' in food1_data and 'fiber_g' in food2_data:
#                     fiber_diff = food1_data['fiber_g'] - food2_data['fiber_g']
#                     if abs(fiber_diff) > 1:
#                         summary_text += f"**{food1}** has {'more' if fiber_diff > 0 else 'less'} fiber (+{fiber_diff:.2f}g) than **{food2}**. "

#                 if summary_text:
#                     st.markdown(f"<p style='font-size: 16px;'>Key Differences: {summary_text}</p>", unsafe_allow_html=True) # Larger summary text
#                 else:
#                     st.info("The nutritional profiles of the two foods are quite similar.", icon="ℹ️")

#         elif app_mode == 'Visualize Nutrients Deeply':
#             st.subheader("Interactive Nutrient Exploration", anchor=False)
#             selected_food_viz = st.selectbox("Select a food item to visualize its nutrient profile:", ["Select Food"] + food_names)
#             if selected_food_viz != "Select Food":
#                 visualize_nutrients(selected_food_viz)

#         elif app_mode == 'Track Your Daily Intake':
#             st.subheader("Monitor Your Daily Nutritional Journey", anchor=False)
#             daily_meals_selection = {}
#             for meal in ["Breakfast", "Lunch", "Snacks", "Dinner"]:
#                 daily_meals_selection[meal] = st.multiselect(f"{meal}: Select Food Items", food_names)

#             if st.button("Analyze My Day's Choices"):
#                 total_nutrients, meal_details = analyze_diet({k: v for k, v in daily_meals_selection.items() if v})
#                 st.subheader("Your Day's Nutritional Snapshot:", anchor=False)
#                 summary_df = pd.DataFrame([total_nutrients]).T.rename(columns={0: 'Total'}).style.format(precision=2)
#                 st.dataframe(summary_df.set_caption("Total Nutrients Consumed Today"))
#                 st.markdown("This table summarizes the total amount of key nutrients you have selected across all your meals for the day.")

#                 st.subheader("Meal-by-Meal Nutrient Insights:", anchor=False)
#                 if meal_details:
#                     for meal, items_df in meal_details.items():
#                         st.markdown(f"**{meal}:**")
#                         if isinstance(items_df, pd.Series):
#                             st.dataframe(items_df.to_frame().T.style.format(precision=2).set_caption(f"Nutritional Content for {meal}"))
#                         else:
#                             st.dataframe(
#                                 items_df.style.format(precision=2).set_caption(f"Nutritional Content for {meal}"))
#                         st.markdown(f"This table details the nutritional content of each food item you selected for {meal}.")
#                 else:
#                     st.info("No meals were tracked today.")

#         elif app_mode == 'Personalized Guidance':
#             st.subheader("Your Personalized Path to Nutritional Wellness", anchor=False)
#             st.info("Adjust the parameters below to get general dietary insights based on your profile.")
#             age = st.slider("Your Age (Years):", 10, 80, 30)
#             sex = st.radio("Your Gender:", ["Male", "Female"])
#             activity_level = st.selectbox("Your Activity Level:", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])
#             guidance = None # Initialize guidance
#         if st.button("Get My Dietary Snapshot"):
#                 guidance = get_portion_guidance(age, sex, activity_level)

#                 if guidance:
#                     st.subheader("General Portion Size Guidance:", anchor=False)
#                     if 'Note' in guidance:
#                         st.warning(guidance['Note'])
#                     else:
#                         guidance_df = pd.DataFrame(list(guidance.items()), columns=['Food Group', 'Recommended Portion']).set_index('Food Group')
#                         st.dataframe(guidance_df)
#                         st.markdown("These are general recommendations for daily portion sizes based on your age, gender, and activity level. Individual needs may vary; consult a healthcare professional or registered dietitian for personalized advice.")

#                     st.subheader("Tips for a Balanced Diet:", anchor=False)
#                     st.markdown(
#                         """
#                         * **Variety is Key:** Aim to include a wide range of fruits, vegetables, grains, proteins, and dairy (or alternatives) in your diet to ensure you get all the necessary nutrients.
#                         * **Prioritize Whole Foods:** Choose whole, unprocessed foods over highly processed items whenever possible. These tend to be more nutrient-dense and lower in unhealthy fats, sugars, and sodium.
#                         * **Hydration Matters:** Drink plenty of water throughout the day.
#                         * **Mindful Eating:** Pay attention to your hunger and fullness cues, and eat at a moderate pace.
#                         * **Limit Added Sugars, Saturated Fats, and Sodium:** These can have negative impacts on your health if consumed in excess.
#                         * **Consider Your Lifestyle:** Your dietary needs can change based on your activity level, health conditions, and life stage.
#                         """
#                     )
#                     st.warning("This guidance is for informational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional for personalized dietary recommendations.")

# if __name__ == "__main__":
#     main()


# streamlit run "C:\Users\aman2\OneDrive\Desktop\4th semester\Nutritional Analysis\nutritional-analysis\src\web\app.py"