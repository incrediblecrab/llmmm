#!/usr/bin/env python3
"""Chinese and Russian ingredient name -> Epicure vocabulary.

The paper builds its 1,790-term vocabulary from a 4.14M multilingual corpus via
an "LLM-augmented pipeline". 37.4% of that corpus is Chinese (XiaChuFang) and
3.5% is Russian (Povarenok), and without them our co-occurrence table is
American home cooking wearing a lab coat -- 176 vocabulary terms never appear at
all, and almost every one of them is Chinese (chu_hou_paste, dried_ganba_mushroom,
chinese_toon).

These maps are hand-built rather than machine-translated on purpose. Machine
translation of ingredient names is unreliable in exactly the places that matter:
生抽 and 老抽 both translate to "soy sauce" but are different ingredients, 料酒
becomes "cooking wine" which is not in the vocabulary, and 粉丝 ("fans") is
glass noodles. Each entry below is a decision, not a lookup.

Coverage targets the head of a Zipfian distribution: ~330 Chinese terms cover
most ingredient mentions in 1.55M recipes, ~260 Russian terms cover ~93% of
mentions in 147k recipes.
"""
from __future__ import annotations

# --- Chinese -------------------------------------------------------------
# Ordered longest-first at match time, so 低筋面粉 (cake flour) wins over 面粉.
ZH = {
    # seasonings and sauces
    "盐": "salt", "食盐": "salt", "精盐": "salt", "细盐": "salt",
    "糖": "sugar", "白糖": "sugar", "细砂糖": "sugar", "绵白糖": "sugar",
    "白砂糖": "sugar", "砂糖": "sugar", "冰糖": "rock_sugar",
    "红糖": "brown_sugar", "黄糖": "brown_sugar", "黑糖": "brown_sugar",
    "生抽": "light_soy_sauce", "老抽": "dark_soy_sauce", "酱油": "soy_sauce",
    "豉油": "soy_sauce", "蚝油": "oyster_sauce", "料酒": "rice_wine",
    "黄酒": "rice_wine", "米酒": "rice_wine", "绍兴酒": "shaoxing_wine",
    "花雕酒": "shaoxing_wine", "白酒": "baijiu", "醋": "vinegar",
    "米醋": "rice_vinegar", "香醋": "black_vinegar", "陈醋": "black_vinegar",
    "白醋": "white_vinegar", "баль": "vinegar",
    "味精": "msg", "鸡精": "msg", "鸡粉": "msg",
    "豆瓣酱": "doubanjiang", "郫县豆瓣": "doubanjiang", "甜面酱": "sweet_bean_sauce",
    "黄豆酱": "soybean_paste", "海鲜酱": "hoisin_sauce", "柱候酱": "chu_hou_paste",
    "沙茶酱": "peanut_sauce", "芝麻酱": "sesame_paste", "腐乳": "tofu",
    "豆豉": "fermented_black_bean", "番茄酱": "ketchup", "沙拉酱": "mayonnaise",
    "蛋黄酱": "mayonnaise", "辣椒酱": "chili_sauce", "辣椒油": "chili_oil",
    "红油": "chili_oil", "香油": "sesame_oil", "麻油": "sesame_oil",
    "芝麻油": "sesame_oil", "花椒油": "chili_oil",
    "鱼露": "fish_sauce", "咖喱": "curry_powder", "咖喱粉": "curry_powder",
    "孜然": "cumin", "孜然粉": "cumin", "五香粉": "five_spice_powder",
    "十三香": "five_spice_powder", "花椒": "sichuan_peppercorn",
    "麻椒": "sichuan_peppercorn", "八角": "star_anise", "大料": "star_anise",
    "桂皮": "cinnamon", "肉桂": "cinnamon", "香叶": "bay_leaf",
    "草果": "cardamom", "丁香": "clove", "茴香": "fennel",
    "小茴香": "fennel", "白胡椒": "white_pepper", "黑胡椒": "black_pepper",
    "胡椒": "black_pepper", "胡椒粉": "black_pepper", "咖喱块": "curry_powder",
    "辣椒粉": "chili_powder", "辣椒面": "chili_powder", "干辣椒": "chili_pepper",
    "小米椒": "chili_pepper", "辣椒": "chili_pepper", "青椒": "bell_pepper",
    "红椒": "red_pepper", "彩椒": "bell_pepper", "柿子椒": "bell_pepper",
    "灯笼椒": "bell_pepper", "杭椒": "chili_pepper", "螺丝椒": "chili_pepper",
    "蜂蜜": "honey", "芥末": "mustard", "沙拉汁": "salad_dressing",
    "高汤": "broth", "鸡汤": "chicken_broth", "骨汤": "meat_stock",
    "水": "water", "清水": "water", "开水": "water", "温水": "water",
    "冰水": "water", "食用油": "vegetable_oil", "油": "vegetable_oil",
    "植物油": "vegetable_oil", "菜籽油": "canola_oil", "花生油": "peanut_oil",
    "橄榄油": "olive_oil", "玉米油": "corn_oil", "调和油": "vegetable_oil",
    "猪油": "lard", "黄油": "butter", "奶油": "cream", "淡奶油": "cream",
    "动物性淡奶油": "cream", "酥油": "ghee",
    # aromatics
    "姜": "ginger", "生姜": "ginger", "姜片": "ginger", "姜末": "ginger",
    "姜丝": "ginger", "老姜": "ginger", "葱": "scallion", "小葱": "scallion",
    "香葱": "scallion", "大葱": "scallion", "葱花": "scallion",
    "洋葱": "onion", "紫洋葱": "red_onion", "红葱头": "shallot",
    "蒜": "garlic", "大蒜": "garlic", "蒜瓣": "garlic", "瓣蒜": "garlic",
    "蒜末": "garlic", "蒜蓉": "garlic", "蒜苗": "garlic",
    "蒜薹": "garlic_scape", "韭菜": "chive", "香菜": "coriander",
    "芫荽": "coriander", "薄荷": "mint", "罗勒": "basil", "紫苏": "perilla",
    "迷迭香": "rosemary", "百里香": "thyme", "牛至": "oregano",
    # vegetables
    "土豆": "potato", "马铃薯": "potato", "红薯": "sweet_potato",
    "地瓜": "sweet_potato", "山药": "yam", "芋头": "taro",
    "胡萝卜": "carrot", "白萝卜": "daikon", "萝卜": "radish",
    "西红柿": "tomato", "番茄": "tomato", "圣女果": "tomato",
    "黄瓜": "cucumber", "茄子": "eggplant", "南瓜": "pumpkin",
    "冬瓜": "winter_melon", "丝瓜": "luffa", "苦瓜": "bitter_melon",
    "西葫芦": "zucchini", "白菜": "napa_cabbage", "大白菜": "napa_cabbage",
    "娃娃菜": "napa_cabbage", "包菜": "cabbage", "卷心菜": "cabbage",
    "圆白菜": "cabbage", "紫甘蓝": "cabbage", "小白菜": "bok_choy",
    "青菜": "bok_choy", "上海青": "bok_choy", "油菜": "bok_choy",
    "生菜": "lettuce", "菠菜": "spinach", "芹菜": "celery",
    "西芹": "celery", "西兰花": "broccoli", "花菜": "cauliflower",
    "菜花": "cauliflower", "莲藕": "lotus_root", "藕": "lotus_root",
    "竹笋": "bamboo_shoot", "笋": "bamboo_shoot", "冬笋": "bamboo_shoot",
    "豆芽": "bean_sprout", "黄豆芽": "soybean_sprout", "豆角": "green_bean",
    "四季豆": "green_bean", "豇豆": "green_bean", "荷兰豆": "snow_pea",
    "豌豆": "pea", "毛豆": "edamame", "玉米": "corn", "甜玉米": "corn",
    "秋葵": "okra", "莴笋": "celtuce", "茭白": "water_bamboo",
    "荸荠": "water_chestnut", "马蹄": "water_chestnut",
    "海带": "kelp", "紫菜": "nori", "木耳": "wood_ear_mushroom",
    "黑木耳": "wood_ear_mushroom", "银耳": "snow_fungus",
    "香菇": "shiitake_mushroom", "冬菇": "shiitake_mushroom",
    "金针菇": "enoki_mushroom", "杏鲍菇": "king_oyster_mushroom",
    "平菇": "oyster_mushroom", "口蘑": "mushroom", "蘑菇": "mushroom",
    "草菇": "straw_mushroom", "茶树菇": "tea_tree_mushroom",
    # proteins
    "鸡蛋": "egg", "蛋": "egg", "土鸡蛋": "egg", "咸蛋": "salted_duck_egg",
    "皮蛋": "century_egg", "鹌鹑蛋": "quail_egg", "蛋清": "egg_white",
    "蛋白": "egg_white", "蛋黄": "egg_yolk",
    "猪肉": "pork", "五花肉": "pork", "里脊": "pork", "梅花肉": "pork",
    "排骨": "pork", "猪蹄": "pork", "培根": "bacon",
    "香肠": "sausage", "腊肠": "chinese_sausage", "火腿": "ham",
    "腊肉": "chinese_cured_pork", "肉馅": "pork", "肉末": "pork",
    "牛肉": "beef", "牛腩": "beef", "牛排": "beef",
    "羊肉": "lamb", "鸡肉": "chicken", "鸡胸肉": "chicken",
    "鸡腿": "chicken", "鸡翅": "chicken", "鸡爪": "chicken",
    "整鸡": "chicken", "鸭": "duck", "鸭肉": "duck", "鹅": "goose",
    "鱼": "fish", "草鱼": "carp", "鲈鱼": "sea_bass",
    "带鱼": "hairtail", "三文鱼": "salmon", "鳕鱼": "cod",
    "虾": "shrimp", "基围虾": "shrimp", "虾仁": "shrimp",
    "蟹": "crab", "螃蟹": "crab", "鱿鱼": "squid", "花甲": "clam",
    "蛤蜊": "clam", "扇贝": "scallop", "牡蛎": "oyster",
    "豆腐": "tofu", "老豆腐": "tofu", "嫩豆腐": "tofu",
    "豆干": "dried_tofu", "腐竹": "tofu_skin", "豆皮": "tofu_skin",
    # staples
    "面粉": "flour", "低筋面粉": "flour", "高筋面粉": "flour",
    "中筋面粉": "flour", "糯米粉": "glutinous_rice_flour",
    "玉米淀粉": "cornstarch", "淀粉": "starch", "生粉": "starch",
    "红薯淀粉": "sweet_potato_starch", "土豆淀粉": "starch",
    "米": "rice", "大米": "rice", "米饭": "rice", "糯米": "glutinous_rice",
    "小米": "millet", "燕麦": "oat", "面条": "noodle", "挂面": "noodle",
    "粉丝": "vermicelli", "米粉": "rice_noodle", "河粉": "rice_noodle",
    "馄饨皮": "dumpling_wrapper", "饺子皮": "dumpling_wrapper",
    "面包": "bread", "吐司": "bread", "面包糠": "bread_crumbs",
    "酵母": "yeast", "泡打粉": "baking_powder", "小苏打": "baking_soda",
    # dairy, fruit, sweets
    "牛奶": "milk", "纯牛奶": "milk", "酸奶": "yogurt", "奶粉": "milk",
    "炼乳": "condensed_milk", "芝士": "cheese", "奶酪": "cheese",
    "马苏里拉": "mozzarella_cheese", "芝士片": "cheese",
    "苹果": "apple", "香蕉": "banana", "橙子": "orange", "橘子": "tangerine",
    "柠檬": "lemon", "青柠": "lime", "草莓": "strawberry",
    "蓝莓": "blueberry", "芒果": "mango", "菠萝": "pineapple",
    "西瓜": "watermelon", "葡萄": "grape", "梨": "pear", "桃": "peach",
    "红枣": "red_date", "大枣": "red_date", "枸杞": "goji_berry",
    "花生": "peanut", "核桃": "walnut", "杏仁": "almond", "腰果": "cashew",
    "芝麻": "sesame_seed", "白芝麻": "sesame_seed", "黑芝麻": "black_sesame_seed",
    "巧克力": "chocolate", "可可粉": "cocoa_powder", "抹茶": "matcha_powder",
    "香草精": "vanilla", "淡奶": "evaporated_milk",
    "红豆": "red_bean", "绿豆": "mung_bean", "黄豆": "soybean",
    "黑豆": "black_bean", "薏米": "barley", "百合": "lily_bulb",
}

# --- Russian -------------------------------------------------------------
RU = {
    "соль": "salt", "сахар": "sugar", "сахарная пудра": "sugar",
    "сахар коричневый": "brown_sugar", "сахар тростниковый": "brown_sugar",
    "ванильный сахар": "vanilla", "ванилин": "vanilla", "ваниль": "vanilla",
    "ванильная эссенция": "vanilla",
    "яйцо куриное": "egg", "яйцо": "egg", "желток яичный": "egg_yolk",
    "белок яичный": "egg_white", "яйцо перепелиное": "quail_egg",
    "мука пшеничная": "flour", "мука": "flour", "мука ржаная": "rye",
    "мука кукурузная": "cornmeal", "мука цельнозерновая": "whole_wheat_flour",
    "лук репчатый": "onion", "лук красный": "red_onion", "лук белый": "onion",
    "лук зеленый": "scallion", "лук-порей": "leek", "чеснок": "garlic",
    "порошок чесночный": "garlic",
    "масло растительное": "vegetable_oil", "масло сливочное": "butter",
    "масло оливковое": "olive_oil", "масло подсолнечное": "sunflower_oil",
    "масло топленое": "ghee", "масло кунжутное": "sesame_oil",
    "маргарин": "margarine", "сало": "lard",
    "перец черный": "black_pepper", "перец белый": "white_pepper",
    "перец душистый": "allspice", "перец болгарский": "bell_pepper",
    "перец сладкий": "bell_pepper", "перец сладкий красный": "bell_pepper",
    "перец красный жгучий": "chili_pepper", "перец чили": "chili_pepper",
    "смесь перцев": "black_pepper", "паприка сладкая": "paprika",
    "вода": "water", "молоко": "milk", "молоко сгущенное": "condensed_milk",
    "сливки": "cream", "сметана": "sour_cream", "кефир": "kefir",
    "йогурт": "yogurt", "творог": "cottage_cheese", "сыворотка": "whey",
    "майонез": "mayonnaise", "кетчуп": "ketchup", "горчица": "mustard",
    "соевый соус": "soy_sauce", "уксус": "vinegar", "бальзамик": "vinegar",
    "томатная паста": "tomato", "аджика": "chili_sauce", "хрен": "horseradish",
    "соус": "brown_sauce", "морковь": "carrot", "картофель": "potato",
    "помидор": "tomato", "помидоры черри": "tomato",
    "томаты в собственном соку": "tomato", "томаты вяленые": "sun_dried_tomato",
    "огурец": "cucumber", "огурец маринованный": "pickled_cucumber",
    "огурец соленый": "pickled_cucumber", "капуста белокочанная": "cabbage",
    "капуста цветная": "cauliflower", "капуста пекинская": "napa_cabbage",
    "капуста квашеная": "sauerkraut", "брокколи": "broccoli",
    "свекла": "beet", "тыква": "pumpkin", "кабачок": "zucchini",
    "цуккини": "zucchini", "баклажан": "eggplant", "редис": "radish",
    "сельдерей черешковый": "celery", "сельдерей корневой": "celeriac",
    "шпинат": "spinach", "листья салата": "lettuce", "руккола": "arugula",
    "зелень": "herb", "укроп": "dill", "петрушка": "parsley",
    "кинза": "coriander", "кориандр": "coriander", "базилик": "basil",
    "мята": "mint", "тимьян": "thyme", "розмарин": "rosemary",
    "орегано": "oregano", "лист лавровый": "bay_leaf", "имбирь": "ginger",
    "куркума": "turmeric", "карри": "curry_powder", "гвоздика": "clove",
    "корица": "cinnamon", "кардамон": "cardamom", "тмин": "cumin",
    "зира": "cumin", "шафран": "saffron", "орех мускатный": "nutmeg",
    "хмели-сунели": "spice_mix", "смесь специй": "spice_mix",
    "специи": "spice_mix", "приправа": "spice_mix", "пряности": "spice_mix",
    "кунжут": "sesame_seed", "мак": "poppy_seed",
    "семечки подсолнуха": "sunflower_seed",
    "рис": "rice", "крупа гречневая": "buckwheat", "крупа манная": "semolina",
    "хлопья овсяные": "oat", "пшено": "millet", "отруби": "bran",
    "чечевица": "lentil", "нут": "chickpea", "фасоль": "bean",
    "фасоль стручковая": "green_bean", "фасоль консервированная": "bean",
    "горошек зеленый": "pea", "кукуруза": "corn",
    "макаронные изделия": "pasta", "хлеб": "bread", "батон": "bread",
    "лаваш": "flatbread", "сухари панировочные": "bread_crumbs",
    "сухарики": "crouton", "блин": "pancake", "печенье": "cookie",
    "тесто слоеное": "puff_pastry", "тесто слоеное бездрожжевое": "puff_pastry",
    "дрожжи": "yeast", "разрыхлитель теста": "baking_powder",
    "сода": "baking_soda", "сода гашеная уксусом": "baking_soda",
    "крахмал": "starch", "крахмал кукурузный": "cornstarch",
    "крахмал картофельный": "starch", "желатин": "gelatin", "желе": "gelatin",
    "закваска": "yeast", "кислота лимонная": "citric_acid",
    "краситель пищевой": "food_coloring", "посыпка кондитерская": "sprinkles",
    "мясо": "meat", "фарш мясной": "meat", "фарш куриный": "chicken",
    "свинина": "pork", "говядина": "beef", "баранина": "lamb",
    "курица": "chicken", "филе куриное": "chicken",
    "грудка куриная": "chicken", "окорочок куриный": "chicken",
    "бедро куриное": "chicken", "голень куриная": "chicken",
    "крылья куриные": "chicken", "печень куриная": "liver",
    "печень говяжья": "liver", "индейка": "turkey", "бекон": "bacon",
    "ветчина": "ham", "колбаса": "sausage", "сосиска": "sausage",
    "грудинка": "beef", "бульон": "broth",
    "рыба": "fish", "филе рыбное": "fish", "сельдь": "herring",
    "семга": "salmon", "лосось": "salmon", "форель": "trout",
    "скумбрия": "mackerel", "тунец": "tuna", "консервы рыбные": "fish",
    "креветки": "shrimp", "кальмар": "squid", "крабовые палочки": "crab_stick",
    "сыр твердый": "cheese", "сыр голландский": "cheese",
    "сыр плавленый": "cheese", "сырок плавленый": "cheese",
    "сыр творожный": "cream_cheese", "сыр сливочный": "cream_cheese",
    "сыр мягкий": "cheese", "пармезан": "parmesan_cheese",
    "моцарелла": "mozzarella_cheese", "брынза": "feta_cheese",
    "фета": "feta_cheese", "маскарпоне": "mascarpone_cheese",
    "рикотта": "ricotta_cheese",
    "яблоко": "apple", "груша": "pear", "банан": "banana",
    "апельсин": "orange", "мандарин": "tangerine", "лимон": "lemon",
    "лайм": "lime", "цедра лимона": "lemon", "цедра апельсина": "orange",
    "сок лимонный": "lemon", "сок апельсиновый": "orange",
    "сок томатный": "tomato", "сок": "fruit_juice",
    "клубника": "strawberry", "малина": "raspberry", "вишня": "cherry",
    "черешня": "cherry", "слива": "plum", "абрикос": "apricot",
    "персик": "peach", "виноград": "grape", "киви": "kiwi",
    "ананас": "pineapple", "гранат": "pomegranate", "авокадо": "avocado",
    "смородина черная": "black_currant", "клюква": "cranberry",
    "черника": "blueberry", "ягода": "berry", "изюм": "raisin",
    "курага": "apricot", "чернослив": "prune", "финик": "date",
    "цукаты": "candied_fruit", "варенье": "jam", "джем": "jam",
    "мед": "honey", "повидло": "jam",
    "орехи грецкие": "walnut", "орехи": "nut", "орехи кедровые": "pine_nut",
    "миндаль": "almond", "фундук": "hazelnut", "арахис": "peanut",
    "стружка кокосовая": "coconut",
    "шоколад темный": "chocolate", "шоколад молочный": "milk_chocolate",
    "шоколад белый": "white_chocolate", "какао-порошок": "cocoa_powder",
    "мороженое": "ice_cream", "грибы": "mushroom", "шампиньоны": "mushroom",
    "вино белое сухое": "white_wine", "вино красное сухое": "red_wine",
    "коньяк": "cognac", "водка": "vodka", "ром": "rum", "ликер": "liqueur",
    "пиво светлое": "beer", "кофе растворимый": "coffee",
    "кофе натуральный": "coffee", "маслины": "black_olive",
    "оливки зеленые": "olive", "каперсы": "caper",
}


def build_maps(vocab):
    """Keep only entries whose target is a real vocabulary term.

    Silently dropping unknown targets would hide typos, so this reports them.
    """
    out, dropped = {}, {}
    for lang, table in (("zh", ZH), ("ru", RU)):
        keep = {k: v for k, v in table.items() if v in vocab}
        out[lang] = keep
        dropped[lang] = sorted({v for v in table.values() if v not in vocab})
    return out, dropped


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_recipe_cooc import load_vocab

    vocab, _ = load_vocab()
    maps, dropped = build_maps(vocab)
    for lang in ("zh", "ru"):
        print(f"{lang}: {len(maps[lang]):>4} usable of "
              f"{len(ZH if lang == 'zh' else RU):>4} entries")
        if dropped[lang]:
            print(f"    targets not in vocabulary ({len(dropped[lang])}): "
                  f"{', '.join(dropped[lang])}")
