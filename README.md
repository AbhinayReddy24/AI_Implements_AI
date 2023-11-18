# AI_Implements_AI

This repository contains AI-Implements-AI team's report for the Generative AI Hackathon hosted by AI Tinkerer.

Project Objective: 

Our project aims to test and evaluate multiple LLMs ability to generate functional code to create a model based on an AI research paper.

Project Outline: 

As prelimianry steps we will choose 2-3 LLMs and research papers. 
Using the latter we will test and evalute the LLMs. 
We will continue our experiement by identifying fine-tuning techniques and prompt to improve our favored model.


# our evaluation metric is how close are we to final goal mentioned in the article
# so first step should be to get the final goal of the paper which is very standard or we can even put it as constant prompt
# we can feed the model with all the possible titanic aritcles 
# RAG is not perfect all the time and it might not be completly reliable all the time
# Instruct fine tunning is kind of path way we need to take based on @daniel analysis
# to develop instruct fine tunning we need data set of the below foramt

# prompt: what are the steps I need to build ML model based on the below mentioned article
# response: we need to ask the following questions:
# Q1: what is this paper trying to achieve or what are the final outcomes
# Q2: what is in the introduction
# Q3: what is the first step taken (this can be data cleaning and managing data imbalances)
# Q4: how first step is implemented (follow up question)
# Q5: what was done after first step or what is the step 2 (this can be exploratory data analysis)
# Q6: what was intrepreted from the step 2 (follow up question)
# Q7: what was done after step 2 or what is step 3 (this can be feature engineering) - (there might not be this step in all 
#       the papers)
# Q8: was step 2 (EDA) had affect on step 3 (this can be achieved by verifying if there are new features intrudouced compared to 
#       initial features)
# Q9: what are the outcomes of step 3 (final features from step 3)
# Q10:what is the done after step 3 or what is step 4 (this can what was the baseline model selected) or (the models they played 
#       with before goining to final model)
# Q11:what are the evaluation metrics (this might not be a step this can asked at any point preferebly before baseline model)
# Q12:what is the main model they fouced on or step 5  - at this point I am not sure to call it step 5 but let's keep it that way
# Q13:how the model is tuned or what parameters that were tuned 
# Q14:what were the final tuned parameters 
# Q15:what is the accuracy of the final model
