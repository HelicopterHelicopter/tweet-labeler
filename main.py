import os
from openai import OpenAI
from dotenv import load_dotenv
import pymongo
import certifi
import json

load_dotenv()

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"),tlsCAFile=certifi.where())

db = mongo_client["remus"]

label_collection = db["label_test"]
tweets_collection = db["tweets"]

model = "meta/llama-3.3-70b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("nim_api_key")
)

content = "You will be a list of tweets and you have to label based on their tone. Some example tones are Witty, Sarcastic, Humorous, Serious, etc. The output should be a list of labels for each tweet. A tweet can have more than 1 label if necessary. The output should be a JSON array where each object contains the tweetId, tweet_text and its corresponding array of tones. Output only the json and no extra text. \n"

tweets = tweets_collection.find({
   "account":"duolingo"
}, {
    "tweet_text": 1,
    "tweetId": 1
}).limit(10)


for tweet in tweets:
    content += tweet["tweetId"]+ ": "+ tweet["tweet_text"] + "\n"

print(content)

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": content}],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    stream=True
)

response_string = ""

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
    response_string += chunk.choices[0].delta.content


json_data = json.loads(response_string)

for labelled_tweet in json_data:
    label_collection.insert_one({
       "tweet_text": labelled_tweet["tweet_text"],
       "label": labelled_tweet["tones"],
       "tweetId": labelled_tweet["tweetId"],
       "model": model
    })