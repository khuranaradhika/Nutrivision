import numpy as np

# Per 100g: [calories, protein_g, carbs_g, fat_g, fiber_g]
# Values approximate from USDA FoodData Central
NUTRITION_PER_100G = {
    'apple_pie':               [237, 2.0,  34.0,  11.0,  1.5],
    'baby_back_ribs':          [290, 24.0,  0.0,  21.0,  0.0],
    'baklava':                 [428,  6.0,  47.0,  25.0,  1.5],
    'beef_carpaccio':          [160, 20.0,   1.0,   9.0,  0.0],
    'beef_tartare':            [175, 22.0,   2.0,   9.0,  0.0],
    'beet_salad':              [ 65,  2.0,  10.0,   2.0,  2.0],
    'beignets':                [350,  5.0,  45.0,  18.0,  1.0],
    'bibimbap':                [140,  8.0,  20.0,   4.0,  2.0],
    'bread_pudding':           [195,  5.0,  30.0,   6.0,  0.5],
    'breakfast_burrito':       [210, 10.0,  25.0,   8.0,  2.0],
    'bruschetta':              [190,  5.0,  28.0,   7.0,  2.0],
    'caesar_salad':            [105,  4.0,   7.0,   8.0,  1.0],
    'cannoli':                 [320,  6.0,  38.0,  17.0,  0.5],
    'caprese_salad':           [150,  8.0,   4.0,  12.0,  0.5],
    'carrot_cake':             [305,  3.0,  45.0,  13.0,  1.0],
    'ceviche':                 [ 90, 16.0,   5.0,   1.5,  1.0],
    'cheese_plate':            [350, 20.0,   2.0,  30.0,  0.0],
    'cheesecake':              [321,  5.0,  26.0,  22.0,  0.3],
    'chicken_curry':           [155, 14.0,   8.0,   8.0,  1.5],
    'chicken_quesadilla':      [260, 15.0,  24.0,  11.0,  1.0],
    'chicken_wings':           [290, 27.0,   0.0,  20.0,  0.0],
    'chocolate_cake':          [371,  4.0,  54.0,  16.0,  1.5],
    'chocolate_mousse':        [240,  4.0,  28.0,  14.0,  1.0],
    'churros':                 [400,  5.0,  50.0,  21.0,  1.5],
    'clam_chowder':            [ 92,  5.0,   9.0,   4.0,  0.5],
    'club_sandwich':           [230, 14.0,  22.0,  10.0,  1.5],
    'crab_cakes':              [195, 13.0,  12.0,  11.0,  0.5],
    'creme_brulee':            [210,  4.0,  20.0,  13.0,  0.0],
    'croque_madame':           [270, 14.0,  22.0,  14.0,  1.0],
    'cup_cakes':               [350,  3.0,  53.0,  14.0,  0.5],
    'deviled_eggs':            [170, 10.0,   1.0,  14.0,  0.0],
    'donuts':                  [380,  5.0,  50.0,  19.0,  1.0],
    'dumplings':               [150,  7.0,  22.0,   4.0,  1.0],
    'edamame':                 [122, 11.0,   9.0,   5.0,  5.0],
    'eggs_benedict':           [220, 12.0,  14.0,  13.0,  0.5],
    'escargots':               [ 90, 16.0,   2.0,   2.0,  0.0],
    'falafel':                 [333, 13.0,  32.0,  18.0,  5.0],
    'filet_mignon':            [225, 26.0,   0.0,  13.0,  0.0],
    'fish_and_chips':          [265, 13.0,  28.0,  12.0,  1.5],
    'foie_gras':               [462, 11.0,   5.0,  44.0,  0.0],
    'french_fries':            [312,  3.5,  41.0,  15.0,  3.5],
    'french_onion_soup':       [ 75,  4.0,   9.0,   2.5,  1.0],
    'french_toast':            [230,  8.0,  26.0,  11.0,  1.0],
    'fried_calamari':          [175, 15.0,  10.0,   8.0,  0.5],
    'fried_rice':              [190,  5.0,  28.0,   7.0,  1.0],
    'frozen_yogurt':           [127,  3.0,  26.0,   1.5,  0.0],
    'garlic_bread':            [330,  7.0,  42.0,  15.0,  2.0],
    'gnocchi':                 [175,  4.0,  34.0,   2.0,  2.0],
    'greek_salad':             [ 80,  3.0,   7.0,   5.0,  2.0],
    'grilled_cheese_sandwich': [330, 12.0,  30.0,  19.0,  1.5],
    'grilled_salmon':          [187, 25.0,   0.0,   9.0,  0.0],
    'guacamole':               [150,  2.0,   8.0,  13.0,  5.0],
    'gyoza':                   [190,  9.0,  22.0,   7.0,  1.5],
    'hamburger':               [295, 17.0,  24.0,  14.0,  1.5],
    'hot_and_sour_soup':       [ 60,  4.0,   6.0,   2.0,  0.5],
    'hot_dog':                 [290, 11.0,  22.0,  17.0,  1.0],
    'huevos_rancheros':        [185, 10.0,  15.0,  10.0,  2.0],
    'hummus':                  [177,  8.0,  20.0,   8.0,  6.0],
    'ice_cream':               [207,  3.5,  24.0,  11.0,  0.0],
    'lasagna':                 [165,  9.0,  16.0,   7.0,  1.0],
    'lobster_bisque':          [105,  6.0,   7.0,   6.0,  0.3],
    'lobster_roll_sandwich':   [280, 16.0,  26.0,  12.0,  1.0],
    'macaroni_and_cheese':     [220,  8.0,  28.0,   9.0,  1.0],
    'macarons':                [410,  6.0,  65.0,  15.0,  1.0],
    'miso_soup':               [ 40,  3.0,   4.0,   1.0,  0.5],
    'mussels':                 [ 86, 12.0,   4.0,   2.0,  0.0],
    'nachos':                  [310,  9.0,  35.0,  16.0,  3.0],
    'omelette':                [155, 11.0,   1.0,  12.0,  0.0],
    'onion_rings':             [340,  5.0,  40.0,  18.0,  2.0],
    'oysters':                 [ 68,  7.0,   4.0,   2.5,  0.0],
    'pad_thai':                [190,  9.0,  28.0,   5.0,  2.0],
    'paella':                  [170, 12.0,  20.0,   5.0,  1.0],
    'pancakes':                [230,  6.0,  38.0,   7.0,  1.5],
    'panna_cotta':             [180,  3.0,  20.0,  10.0,  0.0],
    'peking_duck':             [280, 19.0,   4.0,  21.0,  0.0],
    'pho':                     [ 95,  8.0,  12.0,   1.5,  1.0],
    'pizza':                   [266, 11.0,  33.0,  10.0,  2.0],
    'pork_chop':               [250, 26.0,   0.0,  16.0,  0.0],
    'poutine':                 [260,  8.0,  32.0,  12.0,  2.5],
    'prime_rib':               [280, 25.0,   0.0,  20.0,  0.0],
    'pulled_pork_sandwich':    [290, 18.0,  28.0,  11.0,  1.5],
    'ramen':                   [115,  7.0,  16.0,   2.5,  1.0],
    'ravioli':                 [195,  8.0,  28.0,   6.0,  2.0],
    'red_velvet_cake':         [330,  3.5,  50.0,  13.0,  0.5],
    'risotto':                 [175,  5.0,  28.0,   5.0,  0.5],
    'samosa':                  [262,  5.0,  33.0,  13.0,  3.0],
    'sashimi':                 [130, 20.0,   0.0,   5.0,  0.0],
    'scallops':                [ 88, 17.0,   4.0,   1.0,  0.0],
    'seaweed_salad':           [ 70,  2.0,  11.0,   2.0,  1.5],
    'shrimp_and_grits':        [185, 14.0,  18.0,   7.0,  1.0],
    'spaghetti_bolognese':     [180,  9.0,  22.0,   6.0,  2.0],
    'spaghetti_carbonara':     [250, 10.0,  28.0,  11.0,  1.5],
    'spring_rolls':            [165,  5.0,  22.0,   7.0,  2.0],
    'steak':                   [240, 26.0,   0.0,  15.0,  0.0],
    'strawberry_shortcake':    [270,  3.0,  41.0,  11.0,  1.5],
    'sushi':                   [143,  6.0,  25.0,   2.0,  1.0],
    'tacos':                   [215, 12.0,  20.0,  10.0,  2.0],
    'takoyaki':                [185,  7.0,  24.0,   7.0,  0.5],
    'tiramisu':                [283,  5.0,  28.0,  17.0,  0.5],
    'tuna_tartare':            [140, 22.0,   2.0,   5.0,  0.0],
    'waffles':                 [291,  7.0,  37.0,  14.0,  1.5],
}

# Typical serving sizes in grams
SERVING_SIZE_G = {
    'apple_pie': 125, 'baby_back_ribs': 250, 'baklava': 60, 'beef_carpaccio': 85,
    'beef_tartare': 150, 'beet_salad': 150, 'beignets': 90, 'bibimbap': 400,
    'bread_pudding': 175, 'breakfast_burrito': 250, 'bruschetta': 90,
    'caesar_salad': 200, 'cannoli': 85, 'caprese_salad': 180, 'carrot_cake': 110,
    'ceviche': 200, 'cheese_plate': 120, 'cheesecake': 125, 'chicken_curry': 300,
    'chicken_quesadilla': 200, 'chicken_wings': 200, 'chocolate_cake': 120,
    'chocolate_mousse': 120, 'churros': 100, 'clam_chowder': 300, 'club_sandwich': 300,
    'crab_cakes': 170, 'creme_brulee': 150, 'croque_madame': 250, 'cup_cakes': 75,
    'deviled_eggs': 100, 'donuts': 85, 'dumplings': 200, 'edamame': 155,
    'eggs_benedict': 280, 'escargots': 150, 'falafel': 150, 'filet_mignon': 225,
    'fish_and_chips': 400, 'foie_gras': 80, 'french_fries': 150,
    'french_onion_soup': 350, 'french_toast': 180, 'fried_calamari': 200,
    'fried_rice': 300, 'frozen_yogurt': 200, 'garlic_bread': 80, 'gnocchi': 300,
    'greek_salad': 250, 'grilled_cheese_sandwich': 150, 'grilled_salmon': 200,
    'guacamole': 100, 'gyoza': 200, 'hamburger': 280, 'hot_and_sour_soup': 350,
    'hot_dog': 175, 'huevos_rancheros': 350, 'hummus': 100, 'ice_cream': 145,
    'lasagna': 350, 'lobster_bisque': 350, 'lobster_roll_sandwich': 250,
    'macaroni_and_cheese': 350, 'macarons': 45, 'miso_soup': 250, 'mussels': 300,
    'nachos': 250, 'omelette': 200, 'onion_rings': 150, 'oysters': 175,
    'pad_thai': 350, 'paella': 400, 'pancakes': 200, 'panna_cotta': 150,
    'peking_duck': 200, 'pho': 500, 'pizza': 200, 'pork_chop': 230, 'poutine': 400,
    'prime_rib': 280, 'pulled_pork_sandwich': 300, 'ramen': 500, 'ravioli': 300,
    'red_velvet_cake': 120, 'risotto': 350, 'samosa': 110, 'sashimi': 150,
    'scallops': 200, 'seaweed_salad': 150, 'shrimp_and_grits': 350,
    'spaghetti_bolognese': 400, 'spaghetti_carbonara': 350, 'spring_rolls': 200,
    'steak': 250, 'strawberry_shortcake': 160, 'sushi': 200, 'tacos': 210,
    'takoyaki': 200, 'tiramisu': 150, 'tuna_tartare': 150, 'waffles': 200,
}


def get_nutrition(class_name, serving_g=None):
    key = class_name.replace(' ', '_').lower()
    if key not in NUTRITION_PER_100G:
        return None

    vals = NUTRITION_PER_100G[key]
    per_100g = {
        'calories': vals[0], 'protein_g': vals[1],
        'carbs_g':  vals[2], 'fat_g':     vals[3], 'fiber_g': vals[4],
    }
    if serving_g is None:
        serving_g = SERVING_SIZE_G.get(key, 150)
    factor = serving_g / 100.0
    per_serving = {k: round(v * factor, 1) for k, v in per_100g.items()}

    return {
        'food':        class_name.replace('_', ' ').title(),
        'serving_g':   serving_g,
        'per_100g':    per_100g,
        'per_serving': per_serving,
    }


def format_nutrition_label(class_name, serving_g=None):
    info = get_nutrition(class_name, serving_g)
    if info is None:
        return f'No data for {class_name}'
    s = info['per_serving']
    lines = [
        f"  {info['food']}  (serving: {info['serving_g']}g)",
        f"  Calories:  {s['calories']:.0f} kcal",
        f"  Protein:   {s['protein_g']:.1f} g",
        f"  Carbs:     {s['carbs_g']:.1f} g",
        f"  Fat:       {s['fat_g']:.1f} g",
        f"  Fiber:     {s['fiber_g']:.1f} g",
    ]
    return '\n'.join(lines)


def get_macro_breakdown(class_name):
    info = get_nutrition(class_name)
    if info is None:
        return None
    s = info['per_serving']
    protein_cal = s['protein_g'] * 4
    carbs_cal   = s['carbs_g']   * 4
    fat_cal     = s['fat_g']     * 9
    total       = protein_cal + carbs_cal + fat_cal
    if total == 0:
        return None
    return {
        'protein_pct': round(100 * protein_cal / total, 1),
        'carbs_pct':   round(100 * carbs_cal   / total, 1),
        'fat_pct':     round(100 * fat_cal     / total, 1),
    }


def compare_foods(class_names):
    rows = []
    for name in class_names:
        info = get_nutrition(name)
        if info:
            s = info['per_serving']
            rows.append({
                'food':     info['food'],
                'serving_g': info['serving_g'],
                **s,
            })
    return rows
