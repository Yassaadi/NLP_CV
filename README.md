# NLP_CV

Yassine Assaadi: NLP and CNN for products (image+text) clustering

# Objectif
The mission is to conduct a feasibility study for a product classification engine in its initial iteration. The engine will be based on an image and a description to automate the assignment of product categories.

# Steps
Preprocess the text or image data, depending on the case.
Extract features from the data.
Reduce the dimensionality to 2D, projecting the products onto a 2D graph where each point's color corresponds to the actual category.
Analyze the graph to determine the feasibility of automatically grouping products of the same category based on descriptions or images.
Perform a quantitative measure to confirm the visual analysis by calculating the similarity between the actual categories and the categories obtained from clustering.
The goal is to demonstrate the feasibility of automatically grouping products of the same category using this approach.

# constraints:
## Text features
To extract text features, the following approaches need to be implemented:

Two bag-of-words approaches: simple word counting and TF-IDF.

A traditional word/sentence embedding approach using Word2Vec (or Glove or FastText).

A word/sentence embedding approach using BERT.

A word/sentence embedding approach using USE (Universal Sentence Encoder).

Attached is an example implementation of these text feature extraction approaches on a different dataset. It is recommended to use it as a starting point to save time.

## Image features
To extract image features, the following need to be implemented:

SIFT algorithm.

A CNN Transfer Learning algorithm.







![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (1)](https://github.com/Yassaadi/NLP_CV/assets/106546639/935766de-fec3-418e-87e6-7052c2da38b6)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (2)](https://github.com/Yassaadi/NLP_CV/assets/106546639/4e4c2192-0cbf-4143-8f40-bc5917bc477d)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (3)](https://github.com/Yassaadi/NLP_CV/assets/106546639/d7ea5610-5485-423a-89df-258d4b95979a)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (4)](https://github.com/Yassaadi/NLP_CV/assets/106546639/74064fd5-a1c4-436f-a2f7-60cc831843f0)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (5)](https://github.com/Yassaadi/NLP_CV/assets/106546639/98ce6d80-1d9b-4e62-88e3-88914399e302)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (6)](https://github.com/Yassaadi/NLP_CV/assets/106546639/76482778-77df-4e34-9b04-262949e253fe)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (7)](https://github.com/Yassaadi/NLP_CV/assets/106546639/9edc9f80-9045-4544-9e9e-8d4c324a7fb6)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (8)](https://github.com/Yassaadi/NLP_CV/assets/106546639/a1c32a2a-eca4-43eb-ab28-7b5e01527582)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (9)](https://github.com/Yassaadi/NLP_CV/assets/106546639/9558d48c-5423-4e77-af92-fcd9f78f6f98)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (10)](https://github.com/Yassaadi/NLP_CV/assets/106546639/398c1eba-2661-47f8-ac86-45558a228aef)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (11)](https://github.com/Yassaadi/NLP_CV/assets/106546639/406b5ea4-35b9-4b30-900f-af69e7e0f414)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (12)](https://github.com/Yassaadi/NLP_CV/assets/106546639/7fb5a3ba-8730-498a-8e34-853afd1b9298)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (13)](https://github.com/Yassaadi/NLP_CV/assets/106546639/eb3ef355-09d1-4a01-b9bc-7f1864dc941a)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (14)](https://github.com/Yassaadi/NLP_CV/assets/106546639/436c8dc6-18ca-45c1-9ea0-bdf3a2f245f7)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (15)](https://github.com/Yassaadi/NLP_CV/assets/106546639/4ad673bd-551a-4459-b7d5-f0089b41a202)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (16)](https://github.com/Yassaadi/NLP_CV/assets/106546639/fe105f8d-1b77-4909-b3a1-894145e65fe7)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (17)](https://github.com/Yassaadi/NLP_CV/assets/106546639/191e442b-78c1-47e0-81f6-29cb87916dc1)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (18)](https://github.com/Yassaadi/NLP_CV/assets/106546639/559140c7-e141-4fad-af31-5526f3ce39e6)
![Assaadi_Yassine_2_pre╠üsentation_022023  pptx (19)](https://github.com/Yassaadi/NLP_CV/assets/106546639/00bcf9e1-a434-49cd-90d5-db63bfae7123)
