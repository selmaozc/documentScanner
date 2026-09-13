import os # built-in module for interacting with operating system
# reading environment variables, working with file paths, checking if files exist
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import cv2
import re


load_dotenv() # reads env file and injects everything in it into the environment for the programs session

# looks up an env variable named gemini api key and returns its value as string
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # craetes a client object, authenticated connection to gemini

def extract_fields(img, prompt=None):
    # warped image only exists as a np array - raw pixel data
    # gemini's api needs it as actual jpeg file
    success, encoded_img = cv2.imencode(".jpg",
                                        img)  # converts array into jpeg format in memory (without saving to disk)
    if not success:
        raise ValueError("Failed to encode image")
    img_bytes = encoded_img.tobytes()  # gets the raw byte data gemini expects

    if prompt is None:
        prompt = ("Extract the following fields from this receipt image and return ONLY valid JSON, no other text:"
                  "store_name, date, time, total_amount, currency (the ISO currency code e.g. USD, EUR, GBP - infer from symbols, "
                "store location or context if not explicitly printed), items (list of {name, price})")

    try:
        # the api call - sending image + the prompt to gemini
        response = client.models.generate_content(model="gemini-3.6-flash",
                                                  contents=[
                                                      types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                                                      prompt,
                                                      ],
                                                  )

        raw_text = response.text

        if not raw_text:
            raise ValueError("Gemini returned no text")

        match = re.search(r"\{(.*)\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("Failed to extract fields from image")

        json_str = match.group(0)
        return json.loads(json_str)

    except Exception as e:
        print(f"Extraction failed {e}")
        return {
            "store_name" : None,
            "date" : None,
            "time" : None,
            "total_amount" : None,
            "items" : [],
            "error" : str(e)
        }

    #print("Raw gemini response: ", repr(raw_text))

    #print("full response", response)

    #if raw_text.startswith("'''"):
        #raw_text = raw_text.strip("'")
        #raw_text = raw_text.replace("json", "", 1).strip()

    #return json.loads(raw_text)


