import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def get_expanded_food_categories():
    """Define comprehensive food categories and search terms"""
    return {
        'indian_dishes': [
            'butter chicken', 'chicken tikka masala', 'dal makhani', 'palak paneer',
            'biryani', 'tandoori chicken', 'samosa', 'naan', 'dosa', 'idli',
            'chole bhature', 'malai kofta', 'rajma', 'korma', 'vindaloo',
            'aloo gobi', 'bhindi masala', 'dal tadka', 'pav bhaji', 'vada',
            'uttapam', 'dhokla', 'poha', 'upma', 'paratha', 'roti', 'chapati',
            'sambar', 'rasam', 'curry leaf rice', 'lemon rice', 'pulao'
        ],
        'international_dishes': [
            'pizza', 'pasta', 'sushi', 'burger', 'taco', 'pad thai', 'ramen',
            'pho', 'lasagna', 'risotto', 'paella', 'kebab', 'falafel', 'hummus',
            'moussaka', 'gyoza', 'spring roll', 'dim sum', 'fried rice'
        ],
        'proteins': [
            'chicken breast', 'salmon fillet', 'tuna', 'shrimp', 'beef steak',
            'lamb chop', 'pork chop', 'tofu', 'tempeh', 'eggs', 'greek yogurt',
            'cottage cheese', 'whey protein', 'protein powder', 'protein bar',
            'chicken thigh', 'fish fillet', 'ground beef', 'turkey breast'
        ],
        'vegetables_fruits': [
            'spinach', 'kale', 'broccoli', 'cauliflower', 'carrot', 'potato',
            'sweet potato', 'tomato', 'cucumber', 'bell pepper', 'mushroom',
            'apple', 'banana', 'orange', 'mango', 'grape', 'strawberry',
            'blueberry', 'avocado', 'pineapple', 'pomegranate', 'kiwi'
        ],
        'grains_legumes': [
            'white rice', 'brown rice', 'quinoa', 'oats', 'wheat', 'barley',
            'millet', 'lentils', 'chickpeas', 'black beans', 'kidney beans',
            'pinto beans', 'navy beans', 'soybeans', 'split peas'
        ],
        'dairy_alternatives': [
            'milk', 'almond milk', 'soy milk', 'oat milk', 'coconut milk',
            'yogurt', 'cheese', 'butter', 'cream', 'ice cream', 'kefir',
            'buttermilk', 'condensed milk', 'whipped cream'
        ],
        'snacks_desserts': [
            'chocolate', 'cookies', 'cake', 'pie', 'muffin', 'chips',
            'popcorn', 'nuts', 'dried fruits', 'energy bar', 'granola',
            'trail mix', 'crackers', 'pretzels', 'candy'
        ],
        'beverages': [
            'coffee', 'tea', 'green tea', 'smoothie', 'juice', 'soda',
            'energy drink', 'sports drink', 'lemonade', 'milkshake',
            'protein shake', 'herbal tea', 'coconut water'
        ]
    }

def get_usda_data(api_key):
    """Fetch comprehensive nutrition data from USDA API"""
    food_data = []
    base_url = 'https://api.nal.usda.gov/fdc/v1'
    categories = get_expanded_food_categories()
    
    for category, foods in categories.items():
        for food in foods:
            for page in range(3):  # Get multiple pages per food
                try:
                    params = {
                        'api_key': api_key,
                        'query': food,
                        'dataType': ['Foundation', 'SR Legacy', 'Branded'],
                        'pageSize': 200,
                        'pageNumber': page + 1
                    }
                    
                    response = requests.get(f'{base_url}/foods/search', params=params)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('foods', []):
                            try:
                                nutrients = {n['nutrientName']: n['value'] for n in item.get('foodNutrients', [])}
                                food_item = {
                                    'name': item.get('description', ''),
                                    'category': category,
                                    'brand': item.get('brandOwner', 'Generic'),
                                    'serving_size_g': item.get('servingSize', 100),
                                    'calories': nutrients.get('Energy', 0),
                                    'protein_g': nutrients.get('Protein', 0),
                                    'fat_total_g': nutrients.get('Total lipid (fat)', 0),
                                    'carbs_g': nutrients.get('Carbohydrate, by difference', 0),
                                    'fiber_g': nutrients.get('Fiber, total dietary', 0),
                                    'sugar_g': nutrients.get('Sugars, total including NLEA', 0),
                                    'calcium_mg': nutrients.get('Calcium, Ca', 0),
                                    'iron_mg': nutrients.get('Iron, Fe', 0),
                                    'potassium_mg': nutrients.get('Potassium, K', 0),
                                    'sodium_mg': nutrients.get('Sodium, Na', 0),
                                    'vitamin_c_mg': nutrients.get('Vitamin C', 0),
                                    'vitamin_a_iu': nutrients.get('Vitamin A, IU', 0),
                                    'vitamin_d_iu': nutrients.get('Vitamin D', 0),
                                    'cholesterol_mg': nutrients.get('Cholesterol', 0),
                                    'saturated_fat_g': nutrients.get('Fatty acids, total saturated', 0),
                                    'trans_fat_g': nutrients.get('Fatty acids, total trans', 0),
                                    'search_term': food
                                }
                                
                                # Only add items with valid calorie values
                                if food_item['calories'] > 0:
                                    food_data.append(food_item)
                                    
                            except Exception:
                                continue

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    print(f"Error fetching {food}: {str(e)}")
                    continue

    return pd.DataFrame(food_data)