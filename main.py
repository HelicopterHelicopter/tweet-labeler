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

content = "You will be given a list of tweets and you have to label based on their tone. Some example tones are Witty, Sarcastic, Humorous, Serious, etc. The output should be a list of labels for each tweet. A tweet can have more than 1 label if necessary. The output should be a JSON array string where each object contains the tweetId, tweet_text and its corresponding array of tones. Output only the json and no extra text and do not make it a code block and it should be parsable by json.loads(). The output list should be in the same order as input list. \n"

tweets = list(tweets_collection.find({
    "llama_labels": {"$exists": False}
}).limit(50))


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

for index, tweet in enumerate(tweets):
   print("Tweet: ", tweet["tweetId"])
   tweet["llama_labels"] = json_data[index]["tones"]
   tweets_collection.replace_one({
        "tweetId": tweet["tweetId"]
    }, tweet)