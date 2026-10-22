"""Pair list for the counterfactual data factory (10-22): (garbled string A, target string B, seed, surface) rows.
Targets come from sign-text lists that EXCLUDE the 40 benchmark targets (those are held out; a separate validation list
uses them). A is produced by a garbling model of the generator's errors seen in the benchmark readings: letter
substitution, deletion, insertion, word duplication, word drop (1-3 operations); for Chinese, character substitution
from a pool of common characters, deletion and insertion. Fixed seed; writes JSONL.
Usage: python make_pairs.py <out.jsonl> --n 2000 [--seed 20261022] [--validation]"""
import argparse, json, random

EN = ["BARBER", "PIZZA", "SUSHI BAR", "BOOKS", "MUSIC", "DENTIST", "FLORIST", "BURGERS", "DINER", "LIQUOR", "MARKET", "TAXI", "PARKING", "ENTRANCE", "PUSH", "PULL",
      "SALE", "OPEN 24H", "CLOSED", "FOR RENT", "FOR SALE", "HAPPY HOUR", "LIVE MUSIC", "FREE WIFI", "CASH ONLY", "NO SMOKING", "KEEP OUT", "ONE WAY", "STOP", "DO NOT ENTER",
      "BUS ONLY", "BIKE LANE", "FRESH FISH", "HOT DOGS", "DONUTS", "BAGELS", "NOODLES", "DUMPLINGS", "RAMEN", "TACOS", "VEGAN", "ORGANIC", "CLINIC", "DENTAL", "OPTICAL",
      "BANK", "ATM", "POST", "TOYS", "GIFTS", "JEWELRY", "WATCHES", "SHOES", "HATS", "TAILOR", "CLEANERS", "REPAIR", "PRINT SHOP", "STUDIO", "THEATRE", "MUSEUM",
      "LIBRARY", "SCHOOL", "GARAGE", "MOTEL", "INN", "ROOMS", "VACANCY", "TICKETS", "EXIT ONLY", "GRAND OPENING", "NOW OPEN", "COLD BEER", "WINE BAR", "COCKTAILS",
      "ESPRESSO", "SMOOTHIES", "JUICE", "FROZEN YOGURT", "CANDY", "CHOCOLATE", "BUTCHER", "DELI", "GROCERY", "HARDWARE", "PAINT", "LUMBER", "TIRES", "AUTO PARTS",
      "CAR WASH", "GAS", "DIESEL", "SUBWAY", "TRAIN", "AIRPORT", "ARRIVALS", "DEPARTURES", "BAGGAGE", "CHECK IN", "LOBBY", "RECEPTION", "ELEVATOR", "STAIRS",
      "RESTROOM", "TOILETS", "SHOWERS", "LOCKERS", "GYM", "YOGA", "SPA", "SALON", "NAILS", "TATTOO", "PIERCING", "VINTAGE", "THRIFT", "ANTIQUES", "ART", "CRAFTS",
      "POTTERY", "GLASSWARE", "CAMERA", "PHONES", "COMPUTERS", "GAMES", "COMICS", "RECORDS SHOP", "GUITARS", "PIANO", "DANCE", "KARAOKE", "BILLIARDS", "BOWLING",
      "ARCADE", "CASINO", "LOTTERY", "PAWN", "LOANS", "INSURANCE", "LAWYER", "NOTARY", "TAX", "REAL ESTATE", "APARTMENTS", "CONDOS", "LOFTS", "OFFICE", "WAREHOUSE"]
ZH = ["理发", "超市", "药店", "餐厅", "面馆", "火锅城", "奶茶", "烧烤", "酒吧", "书屋", "花坊", "服装", "鞋店", "眼镜", "牙科", "诊所", "银行", "邮政", "快递", "修车",
      "加油站", "停车", "入口", "营业", "欢迎", "谢谢惠顾", "小心台阶", "禁止吸烟", "请勿触摸", "免费", "特价", "新品上市", "打折", "咖啡馆", "甜品", "蛋糕房", "水果", "蔬菜", "肉铺", "海鲜",
      "米线", "饺子馆", "包子", "粥铺", "早餐", "夜宵", "网吧", "照相馆", "文具", "玩具", "宠物", "美容", "按摩", "健身房", "游泳馆", "旅馆", "公寓", "售楼处", "菜市场", "五金",
      "家具", "灯饰", "窗帘", "地板", "装修", "搬家", "洗车", "汽修", "轮胎", "电动车", "自行车", "手机", "电脑", "维修", "打印", "复印", "翻译", "培训", "幼儿园", "小学",
      "中学", "图书馆", "博物馆", "剧院", "影城", "游乐园", "动物园", "植物园", "码头", "车站", "机场", "候车室", "售票处", "取票", "安检", "出站", "换乘", "洗手间", "母婴室", "吸烟区",
      "茶楼", "酒楼", "小吃", "快餐", "自助餐", "麻辣烫", "串串", "烤鱼", "牛肉面", "兰州拉面", "黄焖鸡", "沙县小吃", "煎饼", "豆浆", "油条", "烤红薯", "糖葫芦", "冰淇淋", "凉茶", "果汁"]
BENCH_EN = {"BUS STOP", "FRESH BREAD", "BOOK STORE", "PHARMACY", "COFFEE SHOP", "TEA HOUSE", "NO PARKING", "FLOWER MARKET", "SALE 50%", "NIGHT MARKET", "ICE CREAM", "RECORDS", "OPEN", "EXIT", "GALLERY", "BAKERY", "HOTEL", "LAUNDRY", "CINEMA", "WELCOME"}
BENCH_ZH = {"面包房", "小心地滑", "邮局", "停车场", "电影院", "书店", "出口", "洗衣店", "咖啡店", "茶馆", "欢迎光临", "营业中", "理发店", "花店", "公园", "火锅", "夜市", "便利店", "药房", "水果店"}
CJK_POOL = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四民先示色内友流"
CJK_POOL = "".join(dict.fromkeys(CJK_POOL))


def garble_en(s, rnd):
    ops = rnd.randint(1, 3); words = s.split(" ")
    for _ in range(ops):
        kind = rnd.choices(["sub", "del", "ins", "dupword", "dropword"], weights=[0.45, 0.2, 0.2, 0.08, 0.07])[0]
        w = rnd.randrange(len(words)); word = list(words[w])
        if kind == "sub" and word: word[rnd.randrange(len(word))] = rnd.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        elif kind == "del" and len(word) > 2: del word[rnd.randrange(len(word))]
        elif kind == "ins": word.insert(rnd.randrange(len(word) + 1), rnd.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        elif kind == "dupword": words.insert(w, words[w]); continue
        elif kind == "dropword" and len(words) > 1: del words[w]; continue
        words[w] = "".join(word)
    out = " ".join(x for x in words if x)
    return out if out and out != s else garble_en(s, rnd)


def garble_zh(s, rnd):
    ops = rnd.randint(1, 2); c = list(s)
    for _ in range(ops):
        kind = rnd.choices(["sub", "del", "ins"], weights=[0.65, 0.15, 0.2])[0]
        if kind == "sub": c[rnd.randrange(len(c))] = rnd.choice(CJK_POOL)
        elif kind == "del" and len(c) > 2: del c[rnd.randrange(len(c))]
        else: c.insert(rnd.randrange(len(c) + 1), rnd.choice(CJK_POOL))
    out = "".join(c)
    return out if out != s else garble_zh(s, rnd)


ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261022)
ap.add_argument("--validation", action="store_true", help="use the 40 benchmark targets instead (held-out validation pairs)")
A = ap.parse_args(); rnd = random.Random(A.seed)
en = sorted(BENCH_EN) if A.validation else [w for w in EN if w not in BENCH_EN]; zh = sorted(BENCH_ZH) if A.validation else [w for w in ZH if w not in BENCH_ZH]
surfaces = ["matte", "glossy", "neon", "glass", "paper", "screen", "metal", "wood"]
with open(A.out, "w") as fh:
    for i in range(A.n):
        cjk = rnd.random() < 0.45; b = rnd.choice(zh if cjk else en); a = garble_zh(b, rnd) if cjk else garble_en(b, rnd)
        fh.write(json.dumps({"id": f"{'v' if A.validation else 'p'}{i:05d}", "seed": A.seed * 100 + i, "text_a": a, "text_b": b, "surface": rnd.choice(surfaces)}, ensure_ascii=False) + "\n")
print("PAIRS", A.n, "->", A.out)
