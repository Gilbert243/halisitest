#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import os
import time
from openai import AzureOpenAI
import numpy as np
from dotenv import load_dotenv


# In[2]:


# Load environment variables from the .env file
load_dotenv()


# In[3]:


api_key = os.getenv('AZUREOPENAI_API_KEY')
api_version = os.getenv('AZUREOPENAI_API_VERSION')
azure_endpoint = os.getenv('AZUREOPENAI_API_ENDPOINT')


# In[4]:


# Create Azure OpenAI client
# Make sure the environment variables are created
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint = azure_endpoint
    )


# In[5]:


# Define chat completion function
def completeChat(prompt, style, client, model="gpt-4o-mini"):
    # Execute API call
    result = client.chat.completions.create(
        model=model,
        messages= [
            {
                "role": "system",
                "content": style,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=1000,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        seed=42,
        n=1,
    )

    # Extract the response
    response = result.choices[0].message.content.strip()

    return response


# In[6]:


# Define text embedding function
def embedText(text, client, model="text-embedding-ada-002"):
    # Execute API call
    result = client.embeddings.create(
        model=model,
        input=text
    )

    # Extract and normalize the embeddings
    embedding = np.array(result.data[0].embedding)
    embedding /= np.linalg.norm(embedding)

    return embedding


# In[7]:


# Create prompt

example = """

{
	"Product Info" : {
		"Product Sheet" : {
			"Brand" : "Elola Beauté",
			"Product name" : "Shampoing Bouclés",
			"Marketing Description" : {
				"EN" : "Silk shampoo that regenerates and gently cleanses curly and curly hair"
			},

			"Key ingredients" : {
				"EN" : [
					"Silk"
				]
			},
            "Price (euros)" : "28,99",
			"Quantity (ml)" : "500 ml",
			"Category" : {
				"EN" : "Shampoo"
			},
            "Ages involved" : {
				"EN" : [
					"13-17 years",
					"18-24 years",
					"25-44 years",
					"45-64 years",
					"65 years and over"
				]
			},
            "Suitable for pregnant women?" : {
				"EN" : [
					"Yes"
				]
			},
            "Compatible with allergies?" : {
				"EN" : [
					"Yes"
				]
			},
            "q001" : {
				"EN" : [
					"3A",
					"3B",
					"3C",
					"4A",
					"4B",
					"4C"
				]
			},
            "q002" : {
				"EN" : [
					"Natural"
				]
			},
			"q003" : {
				"EN" : [
					"Curl definition",
					"Healthy hair",
				]
			},
            "q004" : {
				"EN" : [
					"Dryness",
					"Frizz"
				]
			},
            "q005" : {
				"EN" : [
					"Oily",
					"Flaky",
					"Sensitive",
					"Dandruff",
					"Dermatitis",
					"Psoriasis"
				]
			}

		}
	}
}

"""

#Definition du questionnaire dans la variable questionnaire f

questionnaire = """

{
	"questions" : [
		{
			"label" : "q001",
			"question" : {
				"EN" : "Texture(s) concerned"
			},
			"answers" : {
				"EN" : ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C"]
			}
		},
		{
			"label" : "q002",
			"question" : {
				"EN" : "Condition(s)"
			},
			"answers" : {
				"EN": [
					"Natural", 
					"Straightened/chemically treated", 
					"In transition", 
					"Locs", 
					"Braids"]
			}
		},
		{
			"label" : "q003",
			"question" : {
				"EN" : "Desired objective"
			},
			"answers" : {
				"EN": [
					"Curl definition",
					"Length retention",
					"Moisture retention",
					"Shine enhancement",
					"Healthy heat styling",
					"Colour-treated hair care",
					"Manageability",
					"Stronger hair",
					"Volume enhancement",
					"Healthy hair",
					"Shrinkage",
					"None"
				  ]
			}
		},
		{
			"label" : "q004",
			"question" : {
				"EN" : "Problem encountered"
			},
			"answers" : {
				"EN": [
					"Product build-up",
					"Dryness",
					"Greasy hair",
					"Breakage",
					"Frizz",
					"Hair loss",
					"Dull hair",
					"Porous hair",
					"Heat damage",
					"Physical damage(pulling)",
					"Hair transition",
					"Colour change",
					"Manageability",
					"Thinning hair",
					"Weak edges",
					"None"
				  ]
			},
			"max_selections" : 3,
			"importance" : 4,
			"Tag" : "Hair Challenges"
		},
		{
			"label" : "q005",
			"question" : {
				"EN" : "Suitable scalp"
			},
			"answers" : {
				"EN": [
					"Dry", 
					"Oily", 
					"Flaky", 
					"Sensitive", 
					"Dandruff", 
					"Dermatitis", 
					"Alopecia", 
					"Psoriasis", 
					"None"
				]
			}
		}
	],
	"contraindications" : [
		{
			"contraindication" : "Ages involved",
			"answers" : {
				"EN" : [
					"0-1 year",
					"2-5 years",
					"6-12 years",
					"13-17 years",
					"18-24 years",
					"25-44 years",
					"45-64 years",
					"65 years and over"
                ]
			}
		},
		{
			"contraindication" : "Suitable for pregnant women?",
			"answers" : {
				"EN" : [
					"Yes",
					"No"
				]
			}
		},
		{
			"contraindication" : "Compatible with allergies?",
			"answers" : {
				"EN" : [
					"Yes",
					"No"
				]
		}
        }   
	]
}

"""

hair_type_dict_en = """

{
        '1A': 'straight and fine, known for its sleekness and smooth texture but may lack volume and get oily quickly',
        '1B': 'straight with some body, which holds styles better than finer hair and adds a bit more volume',
        '1C': 'straight with texture and body, making it versatile but prone to frizz in humid conditions',
        '2A': 'soft, loose waves that give your hair a gentle texture without too much frizz',
        '2B': 'wavy with more defined curls, giving your hair great texture and body but prone to frizz',
        '2C': 'wavy with thick, textured waves that bring volume and require moisture to maintain definition',
        '3A': 'curly with loose, well-defined curls that offer bounce and texture, requiring hydration for best results',
        '3B': 'curly with tighter ringlets that provide volume and definition but often need moisture to reduce frizz',
        '3C': 'curly with tight, springy curls that offer great texture but can shrink when dry, requiring intense moisture',
        '4A': 'coily with tight, well-defined curls that need deep hydration to avoid dryness and maintain strength',
        '4B': 'coily with less defined curls, offering volume and versatility but requiring moisture for definition',
        '4C': 'coily with very tight, zigzag curls, which thrive on intense moisture and need careful styling'
}

"""

#Declaration de la variable qui contient les informations sur le produit

product_information = f""" 

Devance
Ahead of Cosmetics
PRODUCE
 
RANGES
 
REFERENCE FAIRS
 
PRESS / MEDIA
 
CONTACT
 
PROFESSIONALS
 
The Boutique Ahead
 0.00€

NOURISHING CARE
HomeHair Care Specific Treatments NOURISHING CARE
Novelty

Loading...Loading...Loading...Loading...Loading...
Fullscreen
NOURISHING CARE
Write your comment
In Stock

Sku: DEVNOUR-3G
8.00€ – 48.00€

Rehydration powder – Single dose – 3g sachet – sold in batches.
Natural nourishing treatment, specially designed to strengthen dry and fragile hair. 3-in-1 treatment. Patented process.

Quat ammonium free, sulfate-free, No synthetics, silicone, preservatives or fragrances
Available in packs for a complete routine.
Vegan Product

Packaging: 3g
single dose Made in France / Made in France

Description: Nourishing treatment with organic cocoa butter and phytokeratin. Natural and vegan deep treatment to nourish and strengthen dry and fragile hair. The presence of okra and dictamus moisturizes and structures the hair. Powder treatment to be diluted. Available in a single-dose sachet of 3g (dose for a head of short and thick hair). sold in lots.

What it does: Formulated with tropical plants naturally rich in film-forming compounds and vitamins A & E such as shea, mango and cocoa butter, the treatment coats the hair. Complemented by the botox-like effect of okra and natural ceramides, this subtle blend restores damaged cuticles. Deep action thanks to panthenol and coconut oil that strengthen the hair fiber.

The results: Your hair is detangled, strong, supple and shiny. Hair is easier to style.

The ingredients: The matrix of dictamus and okra mucilage provides softness and hydration. Mango, cocoa and shea butters nourish the hair and reduce moisture loss. Restorative active ingredients such as ceramides and panthenol rebalance the hair fiber for a complete result. Paraben-free, surfactant-free, and fragrance-free, this innovative formula is suitable for irritated or sensitized scalps.

Claims:
Sulfate-free
No preservatives
No essential
oils No surfactants or emulsifiers
No synthetics, silicone or fragrances
Vegan product.

24% of the total ingredients are from Organic Farming. 96% of the total is of natural origin.

Frequency of use: every 15 days (special cure for very dry and damaged hair) or once a month (to strengthen your hair).

"""

style = "You are a cosmetics product data extractor."

prompt = f"""

Your task is to extract and return **only** the following product information from the provided product description text below.  
Use only the information that is **explicitely mentioned**.  
Do **not guess** or infer any data.
Use only the exact values found in the allowed choices from the questionnaire or hair type dictionary. 

Only include these fields in the output:

- Brand (Analyse all the given information in product information and give the real brand of this product.) 

- Product name

- Marketing Description (Extract the full sentence(s) that describe the product’s benefits or function. Do not rephrase or summarize.)
- Key ingredients (Return the list of ingredients exactly as they appear in the INCI (ingredients) list if it is provided 
(usually introduced by a label like "INGREDIENTS:").
If a full INCI list is present, extract all ingredients listed, in the original order and spelling.
If no INCI list is available, extract only the key active ingredients explicitly mentioned in the marketing description, 
benefits, or formulation details.
Do not merge both lists.
Do not infer or guess any ingredient names.
The result must be a list of ingredients (as strings), using the exact wording found in the product description.) 

- Price (euros; CDF or another currency)  

- Quantity (ml)  

- Category 

- Ages involved (Must match one or more of the following fixed age ranges:
0–1 year, 2–5 years, 6–12 years, 13–17 years, 18–24 years, 25–44 years, 45–64 years, 65 years and over
Include only the ranges that are explicitly supported or clearly deducible from statements like:
“For children over 3 years”, “For adults”, “From 6 years old”, etc.
If a minimum age is given (e.g., “from 3 years old”), include all standard age groups that cover or begin at or above 
that minimum age.
For example:
“From 3 years old” → include: "2–5 years", "6–12 years", "13–17 years", "18–24 years", "25–44 years", "45–64 years", "65 years and over"
“From 6 years old” → include: "6–12 years", "13–17 years", "18–24 years", "25–44 years", "45–64 years", "65 years and over"
“For adults” → include: "18–24 years", "25–44 years", "45–64 years", "65 years and over"
If no age is mentioned, exclude all babies groups (i.e., 0–1 year, 2–5 years)
Never assume compatibility based on vague terms like “gentle”, “universal”
Do not create new custom age ranges (e.g., "3–5 years") — only use the fixed questionnaire values.)

- Suitable for pregnant women? (Must be either "Yes" or "No" as per the questionnaire. If the product description explicitly 
states that it is suitable or safe for use during pregnancy, select "Yes". In all other cases — including if no mention is 
made — default to "No". Do not infer suitability based on ingredient naturalness, softness, or general safety claims.)

- Compatible with allergies? (Must be either "Yes" or "No" as per the questionnaire. Select "Yes" only if the product 
information clearly indicates that the product is hypoallergenic, allergy-tested, or formulated to be compatible with allergies.
In the absence of this information, default to "No". Do not assume compatibility based on phrases like “natural ingredients” 
or “dermatologically tested” unless allergies are directly addressed.)

- q001 – Hair texture(s) (Must match exactly one or more of the following values from the hair type dictionary:
1A, 1B, 1C, 2A, 2B, 2C, 3A, 3B, 3C, 4A, 4B, 4C. Include only textures that are explicitly mentioned or clearly targeted in the 
product information. Use only textures that are described directly or strongly implied by the product’s use cases, benefits, 
or suitability for certain hair types (e.g., “curly”, “coily”, “textured”). Do not list all textures unless they are all 
mentioned or clearly intended.)

- q002 – Hair condition(s) (Must match one or more of the following predefined options: Natural, Straightened/chemically treated, 
In transition, Locs, Braids. Include only conditions that are explicitly named or clearly implied in the product usage or 
benefits. Do not infer conditions from general claims — only select what is clearly supported by the text.)

- q003 – Desired objective(s) (Must match one or more of the following from the questionnaire:
Curl definition, Length retention, Moisture retention, Shine enhancement, Healthy heat styling, Colour-treated hair care, 
Manageability, Stronger hair, Volume enhancement, Healthy hair, Shrinkage, None.
Include all that are explicitly mentioned or strongly indicated in the benefits, usage, or marketing description.
Match only those goals that are directly stated (e.g., “helps define curls”, “enhances shine”) or clearly supported by wording 
related to hair change or improvement.
Do not assume benefits based on vague claims like “nourishes” unless clearly linked to one of the valid choices.)

- q004 – Hair problem(s) (Must match exactly one or more of the following options from the questionnaire: Product build-up, 
Dryness, Greasy hair, Breakage, Frizz, Hair loss, Dull hair, Porous hair, Heat damage, Physical damage (pulling), 
Hair transition, Colour change, Manageability, Thinning hair, Weak edges, None. Include only hair problems that are explicitly 
described as issues this product helps to treat, solve, or prevent — based on wording in the benefits, marketing description, 
or instructions for use.)

- q005 – Suitable scalp type(s) (If no scalp types are explicitly mentioned in the product_information, analyze the answers already extracted for questions q001, q002, q003, and q004, and infer the appropriate values for q005 – Suitable scalp type(s) from the following list :
Dry, Oily, Flaky, Sensitive, Dandruff, Dermatitis, Alopecia, Psoriasis, None
Only proceed with this deduction if no scalp types were directly stated in the original product description.
Review the answers given for q001 (Hair texture), q002 (Condition), q003 (Objectives), and q004 (Hair problems).
Based on known correlations between hair/scalp conditions, identify which of the scalp issues above are likely supported.
If no relevant or strongly associated scalp condition can be logically inferred, return only ["None"].
Do not invent, speculate, or guess. Be cautious and evidence-based.)





Return your result in this exact JSON format:

{{
  "Product Info": {{
    "Product Sheet": {{

       "Brand": "...",
      "Product name": "...",
      "Marketing Description": {{
        "EN": "..."
      }},
      "Key ingredients": {{
        "EN": [
          "..."
        ]
      }},
      "Price (euros)": "...",
      "Quantity (ml)": "...",
      "Category": {{
        "EN": "..."
      }},
      "Ages involved": {{
        "EN": [
          "..."
        ]
      }},
      "Suitable for pregnant women?": {{
        "EN": ["Yes" or "No"]
      }},
      "Compatible with allergies?": {{
        "EN": ["Yes" or "No"]
      }},
      "q001": {{
        "EN": [
          "..."
        ]
      }},
      "q002": {{
        "EN": [
          "..."
        ]
      }},
      "q003": {{
        "EN": [
          "..."
        ]
      }},
      "q004": {{
        "EN": [
          "..."
        ]
      }},
      "q005": {{
        "EN": [
          "..."
        ]
      }}
    }}
  }}
}}

Use only values that appear in the following references:
- Hair types (q001): {hair_type_dict_en}
- Questionnaire options: {questionnaire}

Source product description: {product_information}

Return only the JSON output. Do not include comments, explanations, or introductory text. Do not infer or fabricate data.


" 

"""


# In[8]:


# Execute chat completion
start_time = time.time()
completion_model = "gpt-4o-mini" # Make sure it is deployed on Azure AI Studio
response = completeChat(prompt, style, client, completion_model)
elapsed_time = time.time() - start_time

# Print the response
print(response)
print(f"Time taken: {elapsed_time:.2f} seconds")

import json

try:
    response_dict = json.loads(response)  # Convertir le texte en dictionnaire
    with open("prompt_output/product_info_output.json", "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False)
    print("✅ Fichier JSON généré avec succès : product_info_output.json")
except json.JSONDecodeError as e:
    print("❌ Erreur lors de la conversion en JSON :", e)
    print("Contenu brut :", response)


# In[9]:


start_time = time.time()
embedding_model = "text-embedding-ada-002" # Make sure it is deployed on Azure AI Studio
embedding = embedText(response, client, embedding_model)
elapsed_time = time.time() - start_time

# Print the embeddings
print(embedding)
print(embedding.shape)
print(np.linalg.norm(embedding))
print(f"Time taken: {elapsed_time:.2f} seconds")


# In[10]:


start_time = time.time()
first_keyword = "Bayesian Filtering"
second_keyword = "State Estimation"
query = f"Does the response talk about {first_keyword} and {second_keyword}?"
query_embedding = embedText(query, client, embedding_model)
similarity = np.dot(embedding, query_embedding) # Compute cosine similarity via inner product
elapsed_time = time.time() - start_time

# Print the similarity result
print(similarity)
print(f"Time taken: {elapsed_time:.2f} seconds")

