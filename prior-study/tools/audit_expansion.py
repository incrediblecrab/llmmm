"""Count recipe-expansion/ corpora and measure net-new vs the frozen recipe/ baseline."""
import csv, glob, json, os, re, sys, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore"); csv.field_size_limit(10**9)
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E=B=(os.environ.get("IM_CORPUS") or os.path.join(os.path.dirname(ROOT),"raw-data"))+"/"

_norm=re.compile(r"[^\w]+", re.UNICODE)
def nz(t): return _norm.sub(" ", str(t).lower()).strip()

def ne(s):
    if s is None: return False
    if isinstance(s,(list,np.ndarray)): return len(s)>0
    t=str(s).strip()
    return t not in ("","[]","nan","None","{}","c()","<table></table>")

def load(p, cols=None):
    if p.endswith(".parquet"): return pd.read_parquet(p, columns=cols)
    if p.endswith((".json",".jsonl")):
        try:
            d=json.load(open(p))
        except Exception:
            d=[json.loads(l) for l in open(p) if l.strip()]
        return pd.DataFrame(d if isinstance(d,list) else d.get("data",d))
    return pd.read_csv(p, engine="python", on_bad_lines="skip")
def cat(pat, cols=None):
    fs=sorted(glob.glob(E+pat)); return pd.concat([load(f,cols) for f in fs], ignore_index=True)

# (key, glob, ingredient_col, title_col, lang, dedup_pool, parquet_cols)
SPECS=[
 ("foodcom-522k","foodcom-522k-canonical/recipes.csv","RecipeIngredientParts","Name","en","en",None),
 ("foodcom-raw-231k","foodcom-raw-231k/food_recipes.parquet","ingredients","name","en","en",None),
 ("povarenok-detail","povarenok-detail/data/*.parquet","ingredients","title","ru","ru",None),
 ("turkish-102k","turkish-102k/data/*.parquet","Malzemeler","Yemek İsmi","tr","tr",None),
 ("thefoodprocessor-74k","thefoodprocessor-74k/data/*.parquet","recipe",None,"en","en",None),
 ("allrecipes-33k","allrecipes-33k/recipe.csv","ingredients","title","en","en",None),
 ("bhuvii-17k","bhuvii-17k/pretokenization.json","Ingredients","Title","en","en",None),
 ("kaggle-food-13k","kaggle-food-13k/*.csv","Cleaned_Ingredients","Title","en","en",None),
 ("hebrew-9.7k","hebrew-9.7k/recipes.parquet","json_ld","title","he",None,None),
 ("indian-7k","indian-7k/Food_Recipe.csv","ingredients_name","name","en-IN","en",None),
 ("persian-6k","persian-6k/train.csv","table","name","fa",None,None),
 ("greek-5k","greek-5k/recipes_greek.csv","Ingredients","name","el",None,None),
 ("moroccan-4.6k","moroccan-4.6k/moroccan_recipes_dataset.csv","prompt",None,"en-MA","en",None),
 ("japanese-3k","japanese-3k/recipe_jap.csv","材料","タイトル","ja",None,None),
 ("filipino-2k","filipino-2k/data/*.parquet","ingredients","recipe_name","fil",None,None),
 ("halal-2k","halal-2k/data/recipes.jsonl","ingredients","title","en","en",None),
 ("thai-1k","thai-1k/thai_food_dataset.csv","วัตถุดิบ (Ingredients)","ชื่อเมนู","th",None,None),
 ("taiwan-1.8k","taiwan-1.8k/data/*.parquet","ingredients","title","zh-TW","zh",None),
 ("romanian-881","romanian-881/*.csv","1","0","ro",None,None),
]

def baseline_titles():
    pools={}
    en=set()
    for ch in pd.read_csv(B+"01-recipenlg/RecipeNLG_dataset.csv", usecols=["title"], chunksize=300_000):
        en.update(nz(t) for t in ch["title"].astype(str))
    pools["en"]=en; print(f"  baseline en titles: {len(en):,}", flush=True)
    ru=set()
    for ch in pd.read_csv(B+"03-povarenok/povarenok.csv", usecols=["name"], chunksize=200_000,
                          engine="python", on_bad_lines="skip"):
        ru.update(nz(t) for t in ch["name"].astype(str))
    pools["ru"]=ru; print(f"  baseline ru titles: {len(ru):,}", flush=True)
    zh=set()
    with open(B+"02-xiachufang/recipe_corpus_full.json") as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            try: zh.add(nz(json.loads(line).get("name","")))
            except Exception: pass
    pools["zh"]=zh; print(f"  baseline zh titles: {len(zh):,}", flush=True)
    tr=set(nz(t) for t in pd.read_parquet(B+"06-turkish/turkish_recipe_v3.parquet").iloc[:,0].astype(str))
    pools["tr"]=tr; print(f"  baseline tr titles: {len(tr):,}", flush=True)
    return pools

def prep_hebrew(df):
    names, ings = [], []
    for s in df["json_ld"]:
        n = i = ""
        try:
            j = json.loads(s)
            if isinstance(j, list): j = j[0]
            if isinstance(j, dict):
                g = j.get("@graph")
                if g: j = next((x for x in g if x.get("@type") == "Recipe"), j)
                n = j.get("name", ""); i = j.get("recipeIngredient", "")
        except Exception: pass
        names.append(n); ings.append(i)
    return pd.DataFrame({"title": names, "ing": ings})

PREP = {"hebrew-9.7k": prep_hebrew}

def main():
    print("Loading baseline titles ...", flush=True)
    pools=baseline_titles()
    rows=[]; seen_exp={}
    for key,pat,ic,tc,lang,pool,pcols in SPECS:
        try:
            df=cat(pat,pcols)
        except Exception as e:
            print(f"{key:22}LOAD FAIL {type(e).__name__} {str(e)[:50]}",flush=True); continue
        if key in PREP:
            df=PREP[key](df); ic,tc="ing","title"
        n_raw=len(df)
        n_ing=int(df[ic].map(ne).sum()) if ic in df.columns else 0
        sub=df[df[ic].map(ne)] if ic in df.columns else df.iloc[0:0]
        if tc and tc in sub.columns:
            titles=[nz(t) for t in sub[tc].astype(str)]
        else:
            titles=[nz(str(v).split("\n")[0][:120]) for v in sub[ic].astype(str)]
        uniq=set(t for t in titles if t)
        base=pools.get(pool,set()) if pool else set()
        overlap_base=len(uniq & base)
        prior=set().union(*seen_exp.values()) if seen_exp else set()
        overlap_exp=len(uniq & prior)
        net=len(uniq - base - prior)
        seen_exp[key]=uniq
        rows.append(dict(source=key,lang=lang,raw_rows=n_raw,with_ingredients=n_ing,
                         unique_titles=len(uniq),dup_vs_baseline=overlap_base,
                         dup_vs_expansion=overlap_exp,net_new=net))
        print(f"{key:22}{lang:7}{n_raw:>10,}{n_ing:>12,}{len(uniq):>11,}{overlap_base:>11,}{overlap_exp:>10,}{net:>11,}", flush=True)
    tot=dict(with_ingredients=sum(r['with_ingredients'] for r in rows), net_new=sum(r['net_new'] for r in rows))
    print("-"*84); print(f"{'TOTAL':29}{tot['with_ingredients']:>12,}{'':11}{'':11}{'':10}{tot['net_new']:>11,}")
    json.dump(dict(sources=rows,total=tot), open(ROOT+"data/derived/expansion_audit.json","w"), indent=2)

if __name__=="__main__":
    print(f"{'source':22}{'lang':7}{'raw':>10}{'w/ing':>12}{'uniqTitle':>11}{'dupBase':>11}{'dupExp':>10}{'NET NEW':>11}")
    main()
