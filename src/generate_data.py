"""
Generate synthetic noisy STT transcripts with PII entities
"""
import json
import random
from typing import List, Dict, Tuple

# Sample data pools
FIRST_NAMES = [
    "ramesh", "priya", "amit", "sneha", "rajesh", "kavita", "suresh", "meera",
    "vijay", "anita", "arun", "deepa", "kiran", "pooja", "manoj", "shalini",
    "rahul", "divya", "nitin", "swati", "john", "emily", "michael", "sarah",
    "david", "jessica", "robert", "lisa", "james", "mary", "william", "patricia"
]

LAST_NAMES = [
    "sharma", "kumar", "singh", "patel", "reddy", "nair", "verma", "gupta",
    "rao", "iyer", "menon", "desai", "shah", "joshi", "mehta", "agarwal",
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis"
]

CITIES = [
    "chennai", "bangalore", "mumbai", "delhi", "hyderabad", "pune", "kolkata",
    "ahmedabad", "jaipur", "lucknow", "new york", "los angeles", "chicago",
    "houston", "london", "paris", "tokyo", "singapore", "dubai"
]

LOCATIONS = [
    "anna nagar", "t nagar", "velachery", "adyar", "koramangala", "whitefield",
    "bandra", "andheri", "connaught place", "rajouri garden", "central park",
    "times square", "oxford street", "fifth avenue", "marine drive", "gandhi road"
]

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

EMAIL_DOMAINS = ["gmail", "yahoo", "hotmail", "outlook", "company", "work"]

# Number to word mapping
NUM_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}


def num_to_words(num_str: str) -> str:
    """Convert digit string to spoken words"""
    return ' '.join([NUM_TO_WORD[d] for d in num_str])


def generate_credit_card() -> Tuple[str, str]:
    """Generate credit card in spoken format"""
    # Generate realistic credit card (4 groups of 4)
    groups = []
    for _ in range(4):
        group = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        groups.append(group)
    
    # Convert to spoken
    spoken_groups = [num_to_words(g) for g in groups]
    
    # Add some variations
    if random.random() < 0.5:
        spoken = ' '.join(spoken_groups)
    else:
        spoken = ' '.join([' '.join(g.split()) for g in spoken_groups])
    
    return spoken, "CREDIT_CARD"


def generate_phone() -> Tuple[str, str]:
    """Generate phone number in spoken format"""
    # 10 digit phone
    digits = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    spoken = num_to_words(digits)
    return spoken, "PHONE"


def generate_email() -> Tuple[str, str]:
    """Generate email in spoken format"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    
    # Variations
    variations = [
        f"{first} dot {last} at {domain} dot com",
        f"{first} underscore {last} at {domain} dot com",
        f"{first}{random.randint(1,99)} at {domain} dot com",
        f"{first} at {domain} dot com",
    ]
    
    email = random.choice(variations)
    return email, "EMAIL"


def generate_person_name() -> Tuple[str, str]:
    """Generate person name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    variations = [
        f"{first} {last}",
        f"{first}",
        f"mister {first} {last}",
        f"miss {first} {last}",
        f"doctor {first} {last}",
    ]
    
    name = random.choice(variations)
    return name, "PERSON_NAME"


def generate_date() -> Tuple[str, str]:
    """Generate date in spoken format"""
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    year = random.randint(1980, 2025)
    
    variations = [
        f"{day} {month} {year}",
        f"{month} {day} {year}",
        f"{day}th of {month} {year}",
        f"{month} {year}",
    ]
    
    date = random.choice(variations)
    return date, "DATE"


def generate_city() -> Tuple[str, str]:
    """Generate city name"""
    return random.choice(CITIES), "CITY"


def generate_location() -> Tuple[str, str]:
    """Generate location"""
    return random.choice(LOCATIONS), "LOCATION"


# Templates for different scenarios
TEMPLATES = [
    "my credit card number is {CREDIT_CARD} and my email is {EMAIL}",
    "call me on {PHONE} i live in {CITY}",
    "my name is {PERSON_NAME} and i was born on {DATE}",
    "please send the document to {EMAIL} by {DATE}",
    "contact {PERSON_NAME} at {PHONE} for more details",
    "my card ending in {CREDIT_CARD} was charged yesterday",
    "i am {PERSON_NAME} from {CITY} and my phone is {PHONE}",
    "email me at {EMAIL} or call {PHONE}",
    "the meeting is on {DATE} at {LOCATION} in {CITY}",
    "my details are name {PERSON_NAME} email {EMAIL} phone {PHONE}",
    "i need to update my card {CREDIT_CARD} for the subscription",
    "reach out to {PERSON_NAME} his email is {EMAIL}",
    "i visited {LOCATION} in {CITY} on {DATE}",
    "can you call me at {PHONE} my name is {PERSON_NAME}",
    "the invoice should go to {EMAIL} dated {DATE}",
    "my friend {PERSON_NAME} lives in {CITY} at {LOCATION}",
    "please charge my card {CREDIT_CARD}",
    "i was there on {DATE} at {LOCATION}",
    "{PERSON_NAME} can be reached at {EMAIL} or {PHONE}",
    "send it to {CITY} {LOCATION} by {DATE}",
]

# Add more complex templates
COMPLEX_TEMPLATES = [
    "hi this is {PERSON_NAME} calling from {CITY} my number is {PHONE} and email {EMAIL}",
    "um so my card number is {CREDIT_CARD} yeah and i live at {LOCATION} in {CITY}",
    "okay the meeting with {PERSON_NAME} is scheduled for {DATE} at {LOCATION}",
    "you can reach me anytime at {PHONE} or send mail to {EMAIL} thanks",
    "i need to verify my identity name is {PERSON_NAME} card number {CREDIT_CARD}",
    "the delivery address is {LOCATION} {CITY} and my phone is {PHONE}",
    "please update my email to {EMAIL} and phone to {PHONE} in the system",
    "my colleague {PERSON_NAME} will contact you on {DATE} from {CITY}",
    "uh the payment should be done using card {CREDIT_CARD} before {DATE}",
    "contact details are phone {PHONE} email {EMAIL} located in {CITY}",
]

ALL_TEMPLATES = TEMPLATES + COMPLEX_TEMPLATES


def generate_example(idx: int) -> Dict:
    """Generate one training example"""
    template = random.choice(ALL_TEMPLATES)
    
    entities = []
    text = template
    offset = 0
    
    # Find all placeholders and replace
    generators = {
        "CREDIT_CARD": generate_credit_card,
        "PHONE": generate_phone,
        "EMAIL": generate_email,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    # Process template
    result_text = ""
    last_end = 0
    
    import re
    pattern = r'\{(' + '|'.join(generators.keys()) + r')\}'
    
    for match in re.finditer(pattern, template):
        entity_type = match.group(1)
        entity_text, label = generators[entity_type]()
        
        # Add text before entity
        result_text += template[last_end:match.start()]
        
        # Add entity
        start = len(result_text)
        result_text += entity_text
        end = len(result_text)
        
        entities.append({
            "start": start,
            "end": end,
            "label": label
        })
        
        last_end = match.end()
    
    # Add remaining text
    result_text += template[last_end:]
    
    # Add some noise
    if random.random() < 0.3:
        noise_words = ["um", "uh", "like", "you know", "so", "well", "okay"]
        noise = random.choice(noise_words)
        # Add at beginning
        if random.random() < 0.5:
            result_text = noise + " " + result_text
            # Adjust entity offsets
            offset_adjust = len(noise) + 1
            for ent in entities:
                ent["start"] += offset_adjust
                ent["end"] += offset_adjust
        else:
            result_text = result_text + " " + noise
    
    return {
        "id": f"utt_{idx:04d}",
        "text": result_text.lower(),
        "entities": entities
    }


def generate_dataset(num_examples: int) -> List[Dict]:
    """Generate dataset"""
    examples = []
    for i in range(num_examples):
        examples.append(generate_example(i + 1))
    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--dev_size", type=int, default=150)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    
    print(f"Generating {args.train_size} training examples...")
    train_data = generate_dataset(args.train_size)
    
    print(f"Generating {args.dev_size} dev examples...")
    dev_data = generate_dataset(args.dev_size)
    
    # Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    with open(f"{args.output_dir}/dev.jsonl", "w") as f:
        for ex in dev_data:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Saved to {args.output_dir}/")
    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")


if __name__ == "__main__":
    main()