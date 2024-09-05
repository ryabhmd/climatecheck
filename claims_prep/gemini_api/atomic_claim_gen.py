import re
import ast
import argparse

"""
Script to go over both enlish claims and german claims and extract a list of atomic claims 
using Gemini API. Atomic claims are extracted only from original claims. 
The following arguments are expected:
english_claims_path: path for final English claims data (see final_en_claims.py)
german_claims_path: path for final German claims data (see final_de_claims.py)
google_api_key: API key to use Gemini. 
"""

def extract_claims_list(text):
    # Use regex to find the content after "claims ="
    match = re.search(r'claims\s*=\s*(\[.*?\])', text, re.DOTALL)

    # If a match is found, return the matched content
    if match:
        return match.group(1).strip()
    else:
        return None  # Return None if no match is found

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--english_claims_path", type=str, help="Path for English claims dataset")
    parser.add_argument("--german_claims_path", type=str, help="Path for German claims dataset")
    parser.add_argument("--google_api_key", type=str, help="Google API to use Gemini")

    args = parser.parse_args()

    english_claims = pd.read_pickle(args.english_claims_path)
    german_claims = pd.read_pickle(args.german_claims_path)

    GOOGLE_API_KEY = args.google_api_key

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    data = [english_claims, german_claims]

    for item in data:
        atomic_claims = []
        for idx, row in item.iterrows():
            claim = row['claim']
            if row['generation_method'] == 'original':
                try:
                    response = model.generate_content(f"Atomic Claim Generation is a task that outputs a list of all claims given an input text. Claims are verifiable statements expressing a finding about one and only one aspect of a scientific entity or process, which can be verified from a single source. The output should only split different sentences in the input text in a way where each sentence contains one claim.  ** It is extremely important in this task that the style of the text, including the used words and characters, should not be changed, and the text itself should not be rephrased. Claims should be copy-pasted. ** ** Each claim should be self-contained without needing more context. A claim should have a subject, a predicate and an object. If a sentence in the input text needs more context to be understood completely, it should not be included in the list of answers. ** Perform atomic claim generation on the following input text: {claim}. Give your answer in a list in python code.")
                    claims = ast.literal_eval(extract_claims_list(response.text))
                    atomic_claims.append(claims)
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    atomic_claims.append(f"Gemini Error: {error_type}, {error_message}")
            
            else:
                atomic_claims.append(None)
                
            if (idx+1) % 15 == 0:
                time.sleep(60)
        item['atomic_claims'] = atomic_claims

    data[0].to_pickle('final_english_claims_atomic.pkl')
    data[1].to_pickle('final_german_claims_atomic.pkl')


if __name__ == "__main__":
    main()
