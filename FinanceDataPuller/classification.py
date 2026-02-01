import pandas as pd

categories = {
    "Politics & Government": [
        "trump", "biden", "election", "impeachment", "senate", "supreme court",
        "government", "shutdown", "congress", "kamala", "bernie", "desantis",
        "prime minister", "boris johnson", "rfk", "democrats", "republican",
        "kevin mccarthy", "charlie kirk", "pope", "department of education",
        "vance", "jd vance", "harris", "approval rating"
    ],

    "Economy & Finance": [
        "stock", "market", "dow", "recession", "inflation", "tariff",
        "housing", "bank", "layoffs", "unemployment", "stimulus",
        "student loan", "economy", "powerball", "lottery", "xrp price"
    ],

    "Public Health & Disease": [
        "covid", "coronavirus", "vaccine", "omicron", "delta variant",
        "bird flu", "measles", "monkeypox", "health", "disease"
    ],

    "War, Geopolitics & Terrorism": [
        "ukraine", "russia", "israel", "palestine", "iran", "hezbollah",
        "taliban", "afghanistan", "china", "taiwan", "north korea",
        "war", "military", "syria", "lebanon", "houthis"
    ],

    "Natural Disasters & Climate": [
        "hurricane", "earthquake", "tsunami", "wildfire", "fires",
        "weather", "climate", "storm", "flood"
    ],

    "Technology & AI": [
        "chatgpt", "artificial intelligence", "ai", "iphone", "tesla",
        "nvidia", "spacex", "cybertruck", "intel", "switch",
        "nintendo", "elon musk twitter"
    ],

    "Crypto & Digital Assets": [
        "bitcoin", "ethereum", "crypto", "dogecoin", "shiba inu",
        "litecoin", "cardano", "ripple", "xrp", "gamestop"
    ],

    "Sports": [
        "nba", "nfl", "lebron", "tom brady", "kyrie", "lakers",
        "warriors", "world cup", "messi", "ronaldo", "chiefs",
        "eagles", "dodgers", "yankees", "olympics", "hamlin"
    ],

    "Entertainment & Celebrities": [
        "taylor swift", "beyonce", "kanye", "kim kardashian",
        "johnny depp", "amber heard", "netflix", "movie", "avengers",
        "harry", "meghan", "queen", "hulk hogan", "drake"
    ]
}

def classify_event(event_name):
    name = event_name.lower()

    for category, keywords in categories.items():
        for kw in keywords:
            if kw in name:
                return category

    return "General / Breaking News"

# Load your dataset
df = pd.read_csv("major_event_frame.csv")

event_names = df.columns

classification = []

for e in event_names:
    classification.append({
        "event_keyword": e,
        "category": classify_event(e)
    })

class_df = pd.DataFrame(classification)

class_df.to_excel("classified_news_events.xlsx", index=False)

print(class_df.head())
