# Hybrid Framework of Transformers, Topic Models and LLMs
a hybrid analytical framework that integrates transformer-based models, topic modelling, and large language models (LLMs) to uncover emerging knowledge from Twitter during the first year of the COVID-19 pandemic. Using 3.5 million tweets posted between January and December 2020, that identified seven major themes shaping public discourse

# 1. Introduction
The COVID-19 pandemic created an unprecedented global shift, not only in public health and governance but also in the ways people communicated, sought information, and expressed emotions online. Among digital platforms, Twitter emerged as a vital space for real time discourse, allowing individuals and institutions to engage in dynamic conversations surrounding the pandemic’s unfolding events. With millions of users sharing thoughts, concerns, and updates daily, Twitter has become a rich data source for understanding public sentiment, thematic trends, and the temporal evolution of societies concerns during crises.
Thematic sentiment analysis used in this work is a method combining advance topic modelling with emotional profiling offers a powerful lens to decode the narratives embedded in social media data. Traditional approaches such as Latent Dirichlet Allocation (LDA) have been widely used to extract topics, while sentiment lexicons and classical classifiers have offered foundational insights into public mood. However, these methods often fall short in capturing the semantic richness, sarcasm, or contextual subtleties inherent in tweets. With the rise of deep learning, transformer-based models like BERT and its social media tuned variants such as roBERTa, BERTweet have drastically improved natural language understanding, enabling more nuanced analysis of short, noisy texts like tweets.
Building upon this advancement, we propose a hybrid analytical framework that integrates transformers (BERT), topic modelling (BERTopic), and large language models (LLMs) to uncover emerging knowledge and emotional implications in COVID-19-related Twitter discourse. Our approach allows not only for the identification of dominant themes but also for an in-depth understanding of how these themes evolved over time and across geographies, enriched with emotional context. While recent studies have explored sentiment dynamics or topic evolution separately (Xue et al., 2020; Banda et al., 2021), few have brought together cutting edge language models and temporal geographical analysis into a mixed pipeline.
This study applies the proposed framework to a dataset of 3.5 million tweets collected throughout 2020 a period marked by significant uncertainty, evolving government policies, and change in public trust. We identified seven major thematic areas, each characterised by unique sentiment patterns. For instance, discussions around School Closures & Education dominated the early and middle phases of the pandemic, while Vaccination Rollout & Hesitancy gained prominence in the final quarter. We also observed regional distinctions in thematic prominence and emotional tone for example, Mental Health concerns were more frequently voiced in Indian cities, whereas School Closures drew stronger engagement in U.S. urban areas.
By integrating advanced NLP techniques, this research not only captures the depth of public discourse but also highlights the utility of hybrid AI frameworks in real time crisis monitoring. Our findings contribute to the growing field of computational social science and provide actionable insights for policymakers, health communicators, and researchers navigating future public health challenges.

# 2. Methodology
This study adopts a hybrid framework that combines transformer based sentiment analysis, topic modelling, large language models (LLMs), and spatial temporal analysis to uncover thematic and emotional insights from large scale COVID-19 Twitter data. The framework, illustrated in Figure 1, consists of six core components: (1) Data Collection and Preprocessing, (2) Sentiment Labelling, (3) Tweet Embedding and Topic Discovery, (4) Automated Topic Interpretation, (5) Thematic Sentiment Analysis, and (6) Temporal and Geographical Analysis.
 
# 2.1. Data Collection and Preprocessing
A total of 3.5 million English language tweets were collected using the Twitter API, covering the period from January 1 to December 31, 2020, filtered by COVID-19-related keywords (e.g., “coronavirus”, “covid”, “vaccine”, “lockdown”, etc.). Each tweet record retained key attributes including the tweet text, timestamp, and location metadata. The preprocessing phase included several steps:
•	Removing retweets and duplicates.
•	Lowercasing and cleaning text (e.g., removing URLs, emojis, special characters).
•	Filtering non-English content using language detection.
•	Geotagging tweets with location metadata such as city or country level.

# 2.2. Sentiment Labelling with RoBERTa
To enrich tweets with emotional metadata, each tweet was classified into one of seven Ekman inspired emotion categories: anger, disgust, fear, joy, sadness, surprise, and neutral. For this task, we employed a fine-tuned RoBERTa base model pretrained on emotion labelled datasets such as GoEmotions. RoBERTa’s contextual understanding significantly improved emotion recognition in short-form text like tweets.

# 2.3. Tweet Embedding and Topic Discovery with BERTopic
Next, we used the BERTopic framework to uncover latent themes in the corpus. This involved:
•	Generating high-dimensional document embeddings using Sentence-BERT (SBERT).
•	Reducing dimensionality using UMAP (Uniform Manifold Approximation and Projection).
•	Clustering tweets via HDBSCAN (Hierarchical Density-Based Spatial Clustering).
•	Extracting and ranking topic keywords using class-based TF-IDF (c-TF-IDF).
The result was a set of coherent topic clusters, each representing a recurring theme in the discourse.

# 2.4. Automated Topic Interpretation with LLMs
To enhance the interpretability of the extracted topics, we employed a prompting pipeline using GPT-3.5 (via OpenAI API). The top keywords for each topic were passed to the LLM along with a prompt to generate a concise, human-readable topic label and summary. This approach reduced manual effort and standardised topic interpretation.

# 2.5. Thematic Sentiment Analysis
For each discovered theme, we computed the distribution of emotional sentiments based on the labelled tweets within each topic cluster. This allowed us to identify emotionally dominant themes (e.g., anger dominated or joy heavy topics) and compare sentiment trends across thematic lines.

# 2.6. Temporal and Geographical Analysis
To analyse the evolution of discourse, we aggregated topic frequencies and sentiment proportions by month and geographic region. This allowed us to trace:
•	The emergence and decline of themes over time (e.g., rise of vaccine discussions in Q4).
•	Regional variation in discourse (e.g., school closures prominent in U.S. cities).
•	Correlations between emotional expression and geographic clusters.

# 3. Data Analysis
This section presents the empirical findings from applying our hybrid framework to the dataset of 3.58 million COVID-19 tweets. We begin with a descriptive overview of the processed data and the output of the sentiment analysis, followed by the results of topic discovery, temporal analysis, and the integration of emotional profiles.

# 4.1. Descriptive Overview and Sentiment Annotation
Following the preprocessing pipeline outlined in Section 3.2, the raw tweets were cleaned and normalized, resulting in a refined corpus ready for analysis. Table 1 provides a sample of the first five rows of the processed dataset, illustrating the structure of the data, which includes the date, user reported place, the cleaned text, and the emotion label assigned by our roBERTa based classifier.

Table 1: Sample of Processed Tweets with Emotion Labels
| Date       | Place                          | Text                                                                                                                        | Emotion  |
|-------------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------|
| 2020-01-22  | Jamaica                        | talking about the jamaica look out for this virus coming from the other side of the world we need to be prepared              | neutral   |
| 2020-01-23  | St Peters, MO                  | legit i got a letter here in stl last week just the flu bro nothing to worry about                                           | neutral   |
| 2020-01-23  | Catarman, Eastern Visayas      | sure the virus was purposely created by scientists but they never expected it to get out of the lab like this                | neutral   |
| 2020-01-23  | Adelaide, South Australia      | jeremy cordeaux tonight selling live koalas to china this is so sad and wrong on so many levels                              | sadness   |
| 2020-01-24  | Tai Po District, Hong Kong     | the virus is spreading across wuhan and the rest of the prc starting to see cases in other countries now                     | neutral   |

The application of the cardiffnlp/twitter-roberta-base-sentiment-latest and emotion models successfully annotated the entire corpus. The initial sample in Table 1 reveals the global nature of the discourse from the very beginning of the dataset (January 2020), with tweets originating from Jamaica, the United States, the Philippines, Australia, and Hong Kong. The tweets show early awareness and a mix of concern and skepticism. Furthermore, it highlights the prevalence of a neutral sentiment in the early stages, used for informational updates and speculative statements, though a sadness emotion is also detected, hinting at the diverse affective responses that would evolve as the pandemic progressed.

# 4.2. Overall Sentiment and Emotional Landscape
To characterize the overarching affective response to the COVID-19 pandemic on Twitter, we analysed the distribution of emotions across the entire tweet’s dataset. Figure 2 presents the global sentiment distribution.

Figure 2: Global Distribution of Emotions across the COVID-19 Twitter Corpus
<img width="562" height="408" alt="image" src="https://github.com/user-attachments/assets/3fffaf3c-cce8-4fb7-95f2-e1369c37d9d6" />

 
The analysis reveals a highly polarized and negatively skewed emotional landscape. The higher emotion is neutral, which creates the largest share of the public discussion. This suggests that a significant portion of the conversation was informational or factual in nature, sharing news and updates without strong emotional expression.
However, among the clearly expressed emotions, sadness is the most dominant, followed closely by fear. This aligns with the global worry and grief experienced during the pandemic due to loss of life, social isolation, and economic uncertainty. The emotion of anger also features significantly, likely reflecting public frustration with government policies, societal divisions, and the mishandling of the crisis in various regions. In contrast, positive emotions are less prevalent; joy is present but is substantially outweighed by the negative emotions, while surprise and disgust form a smaller, though notable, part of the emotional band. This overall profile sets the stage for a more nuanced, topic specific analysis of sentiment in the following sections.

# 4.3. Temporal Dynamics of Public Sentiment
While the overall distribution provides a macro level view, the public's emotional response to a prolonged crisis like the COVID-19 pandemic is inherently dynamic. To capture these temporal shifts, we analysed the 7-day rolling average of sentiment counts over the entire year. Figure 3 illustrates how the volume of tweets expressing each emotion fluctuated in response to real world events.

Figure 3: 7-Day Rolling Average of Sentiment Trends Over Time
<img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/2042d24b-eab9-4930-bdf1-4eb00fc31644" />


The sentiment trends over time illustrate the emotions of fear, sadness, and anger show significant changeable, with pronounced. This suggests that public worry and frustration were highly reactive to specific events, such as the announcement of lockdowns, surges in cases, or political developments. The neutral sentiment maintains a consistently high baseline throughout the period. This indicates a persistent undercurrent of information sharing and news dissemination, upon each wave of emotion. The Key dates correlate with visible spikes in specific emotions such as a major peak in fear in early March 2020 likely corresponds to the WHO's declaration of a global pandemic and the result of panic. Spikes in anger can be observed around debates over mask mandates and economic reopening. Moreover, a noticeable, though smaller, increase in joy towards the end of 2020 aligns with the announcement of the first vaccine authorizations. From Figure 3 the lines often move in concert during major crises, but also diverge, indicating that different events triggered distinct emotional profiles such as an event causing widespread sadness but not anger.
This analysis confirms that public sentiment on Twitter is not static but a fluid construct, tightly fixed with the evolving timeline of the pandemic. This sets the stage for linking these emotional shifts to the emergence of specific topics, which is explored in the next section.
5.10. Emotion Co-occurrence 
To understand the structural relationships between different emotional states in the public discourse, we computed pairwise correlation coefficients between the sentiment categories Figure 4. This analysis reveals how the presence of one emotion in the discourse predicts the presence or absence of another, providing insight into the collective affective psychology during the crisis. 

Figure 4 emotion Correlation Matrix displayed as a heatmap
<img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/a827d794-ab13-45ab-8b2d-0beeb5097d3e" />


The correlation matrix revealed several statistically significant and emotionally interpretable patterns:
•	The Anger Cluster: Anger showed a strong positive correlation with Disgust (r=0.46) and a moderate positive correlation with Surprise (r=0.41). This cluster represents a state of moral outrage and shocked anger, likely directed at perceived failures or injustices.
•	The Positive-Negative Conflict: Anger was strongly negatively correlated with Joy (r=-0.86) and Neutral (r=-0.85). This powerful inverse relationship underscores a highly polarized discourse where joyful or factual conversations existed in a separate space from angry ones. Similarly, Joy and Neutral were positively correlated (r=0.67), suggesting that positive developments were often discussed in a factual tone.
•	The Fear Complex: Fear exhibited strong negative correlations with Sadness (r=-0.61) and Surprise (r=-0.78). This suggests that the emotional response to the pandemic may have evolved from an initial state of surprised fear into a more prolonged, resigned state of sadness, and that these states were distinct in the discourse.
•	The Surprise Duality: Surprise was positively correlated with both Disgust (r=0.84) and Sadness (r=0.37) yet negatively correlated with Fear and Neutral. This indicates that surprise was more often associated with negative, shocking developments (leading to disgust or sadness) rather than with positive news or fearful events.

# 4.5. Themes Discovery and Characterization with BERTopic and LLMs
Applying the BERTopic pipeline followed by LLM-based interpretation to the tweets dataset successfully uncovered the major thematic structures of the COVID-19 discourse. The process transformed the unstructured text corpus into a semantically organized dataset. Table 3 provides a sample of the final annotated data, showcasing how each tweet is associated with its LLM generated theme, sentiment, and other attributes.

Table 2: Sample of Tweets Annotated with BERTopic/LLM Themes and Sentiment
| Theme                         | Date       | Place                     | Sentiment | Text                                                                                                                                                                                                                                      | Rep_Score |
|--------------------------------|-------------|----------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| Economic Impact & Jobs          | 2020-04-22  | Northern Mariana Islands   | joy        | i would pull forward and push him along with my car if i could just to help him get to his job interview on time we all need to support each other right now                                         | 0.43       |
| Economic Impact & Jobs          | 2020-07-03  | Glastonbury, CT            | sadness    | but hes going to be too busy to play golf right now because hes working two jobs to make up for the income we lost during the lockdown its just relentless                                          | 0.43       |
| Economic Impact & Jobs          | 2020-07-05  | Philadelphia, PA           | neutral    | hope hes going to be alright at the new factory job they say they have safety measures but you never know these days                                          | 0.43       |
| Government Response & Lockdowns | 2020-12-18  | Bromley, London            | fear       | look at history of research for vaccines over decades this one was developed too fast i dont trust what the government is telling us about its safety                                               | 1.00       |
| Government Response & Lockdowns | 2020-12-09  | Canada                     | fear       | the nightmare will be over soon who is excited for the vaccine i am not this feels like a massive experiment on the population                                | 0.75       |

The sample immediately reveals several key strengths of the hybrid framework as the LLM successfully generated concise, human readable labels like "Economic Impact & Jobs" and "Government Response & Lockdowns" that accurately summarise the public discussion found in the representative tweets. Also, the framework allows for direct cross tabulation of topic and sentiment. For instance, within the "Economic Impact & Jobs" topic, we see a mix of fear and joy, reflecting the varied personal experiences of the economic shock. The "Government Response & Lockdowns" topic in this sample is heavily associated with fear and neutral sentiments. Each theme can be analysed across time and location, enabling a rich, multi-faceted analysis.

# 4.6. Macro Analysis of Thematic Discourse
The application of our hybrid framework yielded a set of coherent, interpretable themes that capture the core dimensions of COVID-19 discussion on Twitter. An analysis of these themes by volume, emotional polarization, temporal volatility, and geographical concentration reveals a complex and dynamic public conversation. The five most prominent themes by volume, as identified by our model, were:
1.	School Closures & Education (16%)
2.	Vaccination Rollout & Hesitancy (15%)
3.	Government Response & Lockdowns (13%)
4.	Economic Impact & Jobs (11%)
5.	Travel Restrictions & Borders (8%)

This list underscores that public concern was distributed across fundamental societal posts such as the well-being and future of children, public health solutions, governance, economic stability, and mobility. Furthermore, we calculated an emotional polarization score for each theme. The most polarized topics are those with the most one sided emotional distribution were Government Response & Lockdowns, Economic Impact & Jobs, and Mental Health & Safety. This indicates that these were the most contentious and affectively charged issues, likely generating the most significant public debate and division.
Temporally, the emotional landscape, with clear peaks corresponding to the unfolding crisis such as Peak Fear (21.5% in Jan 2020), this coincided with the initial global outbreak and the WHO's declaration of a Public Health Emergency of International Concern. Peak Joy (14.6% in May 2020), this aligned with the initial reduced of lockdowns in many regions and a sense of early hope during the first "summer of the pandemic." Peak Anger (12.6% in Oct 2020), which corresponded with the onset of the "second wave" of infections, leading to renewed restrictions and growing pandemic fatigue, which fuelled public frustration.

# 4.7. Temporal Evolution and Emergence of Topics
A central objective of this work was to track how knowledge and public discourse emerged and evolved throughout the pandemic. Figure 4 illustrates the changing prominence of the top seven themes as a percentage of total tweets over the collected time period.

 Figure 3: Topic Evolution Over Time
<img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/a3c915a1-7b42-4502-bcf0-f174924803f5" />


The timeline reveals different phases of public attention, confirming the dynamic nature of emerging knowledge on social media:
•	Early Pandemic Focus: At the outset, themes like Travel Restrictions & Borders and Government Response & Lockdowns likely dominated as the world reacted to the initial shock, imposed mobility limits, and debated early policy measures.
•	The Rise of Society Impact: As the pandemic persisted, attention shifted to profound societal disruptions. The Economic Impact & Jobs theme surged alongside growing unemployment and business closures. Together, School Closures & Education became a sustained topic of concern, reflecting the prolonged impact on children and families.
•	The Vaccine Era: In the later stages, the discourse was overwhelmingly captured by Vaccination Rollout & Hesitancy. This topic exhibits a classic emergence pattern, growing from negligible volume to dominate the conversation, reflecting the public's engagement with the primary scientific solution to the crisis.
•	Undercurrents of Well-being and Information: Themes like Mental Health & Safety and Information & Media maintained a more consistent presence throughout, acting as underlying currents that shaped the context for all other discussions.
This progress analysis directly validates our framework's capability to not only discover themes but also to pinpoint their relative importance and emergence at different stages of a major ongoing event. 

# 4.8. Geographical Variation in Thematic Focus
The global nature of the COVID-19 pandemic did not result in a uniform global discourse. Our analysis reveals significant geographical variation in how different regions engaged with the crisis. Figure 4 illustrates the distribution of the primary themes across a selection of major cities and regions worldwide, demonstrating that local context profoundly shaped public conversation.

Figure 5: Topic Distribution Across Selected Locations
 <img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/0ddcc157-071d-4e9a-9295-e790a6e4e9fe" />



The heatmap reveals striking local patterns:
•	North American Concerns: Cities like Brooklyn, NY, and Chicago, IL, showed a strong and balanced focus on School Closures & Education, Economic Impact, and Government Response. Los Angeles, CA, had a particularly high concentration (29.0%) on Economic Impact & Jobs.
•	Distinct Indian Discourse: The conversation in Mumbai and New Delhi, India, was dominated by different themes. Mumbai showed a strong focus on Information & Media (29.6%), while New Delhi's discourse was split between Information & Media (29.0%) and Government Response & Lockdowns (24.6%), suggesting a highly politicized and media-driven conversation.
•	An Outlier Case: Nairobi, Kenya, presents a dramatic outlier, with 85.5% of its categorized discourse. This could indicate a high volume of local-language tweets not captured by our English-based models, or discussions focused on highly specific local issues that did not align with the global themes.
•	Policy-Driven Discussions: Washington, DC, as a political hub, showed an elevated focus on Government Response & Lockdowns (19.0%) compared to other locations.
This geographical analysis confirms that while the pandemic was a global event, the public's primary concerns were filtered through local realities, including the specific policies enacted, the economic structure, and the local media landscape. This underscores the importance of a geographically aware framework for understanding public discourse.
This is excellent supplementary data that adds another layer of precision to our geographical analysis. It shows us not just the distribution, but the single most disproportionately discussed topic in each location. Let's integrate this directly into the geographical subsection we just wrote.

# 4.8. Geographical Variation in Thematic Focus
The global nature of the COVID-19 pandemic did not result in a uniform global discourse. Our analysis reveals significant geographical variation in how different regions engaged with the crisis. Figure 4 illustrates the distribution of the primary themes across a selection of major cities and regions worldwide, demonstrating that local context profoundly shaped public conversation.

# 4.9. Emotional Fingerprints of Discovered Themes
A primary contribution of our hybrid framework is the ability to move beyond isolated analyses of themes and sentiment, instead revealing their intrinsic connections. Figure 5 presents the emotional sentiment distribution for the top six themes, providing a nuanced, affective profile for each major topic of discussion.

Figure 5: Emotional Sentiment Distribution by Theme
 <img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/8eee42c4-7353-4741-af69-f75757e0fc3a" />


The analysis reveals different emotional fingerprints that characterize public reaction to various facets of the pandemic:
•	School Closures & Education: This theme is characterized by a pronounced mix of Sadness and Fear, reflecting parental worry and concern for children's well-being and future, with a significant undercurrent of Anger likely directed at policymakers.
•	Vaccination Rollout & Hesitancy: This topic shows a highly polarized emotional profile. It is dominated by a strong Neutral sentiment (likely factual news sharing), but among expressed emotions, Fear and Anger are prominent, mirroring the strong public debate and worry surrounding vaccine safety and mandates.
•	Government Response & Lockdowns: This is one of the most emotionally powerful themes, with Anger being a dominant emotion. This is complemented by significant Sadness and Fear, quantifying the widespread public frustration, grief, and worry over policy measures.
•	Economic Impact & Jobs: This theme is overwhelmingly defined by Fear and Sadness, directly capturing the financial stress and personal losses experienced by millions due to job losses and economic instability.
•	Travel Restrictions & Borders: This theme also shows high levels of Fear and Sadness, coupled with Surprise, likely reflecting the shock and disruption caused by sudden border closures and mobility restrictions.
•	Mental Health & Safety: True to its name, this theme is predominantly associated with Sadness, quantitatively validating the significant psychological toll of the pandemic.
These emotional fingerprints transform our understanding of the topics from mere themes to deeply human experiences. They allow policymakers and researchers to understand not just what issues were being discussed, but the emotional context in which they were received by the public.
5.12. Dashboard View of Thematic Sentiment Dynamics
A comprehensive dashboard Figure 6 synthesizes the multi-dimensional findings of our analysis, providing an at a glance validation of the framework's output and the complex dynamics of COVID-19 discourse.

Figure 6 dashboard of comprehensive thematic sentiment analysis
<img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/8be2bbed-562f-41d4-8ece-5a5f915a27bb" />

 
The "Top Topic Sizes" chart confirms the volume based hierarchy of public concern, with "Social & Lifestyle" encompassing topics such as "School Closures & Education" and "Economic Impact" as the largest themes. The "Theme Evolution Over Time" chart vividly illustrates the non-stationary nature of public discourse. It shows "Economic Impact" as a persistent concern throughout 2020, while "Government & Policy" discussions surged during key policy announcement periods, and "Health & Safety" which includes vaccination gained prominence in the latter half of the year.
The "Overall Sentiment Distribution" bar chart provides a clear visual of the emotional landscape we quantified, dominated by neutral and negative sentiments. The "Sentiment Distribution by Theme (%)" chart offers a nuanced view, showing that while "Economic Impact" was a major source of negative emotion, other themes had more mixed affective profiles. Finally, the geographical table confirms our earlier findings of localized discourse, showing how specific cities like Washington, DC, and Mumbai, India, served as hotspots for distinct thematic conversations.

# 5.	Result 
The application of our hybrid framework to the corpus of 3.58 million COVID-19 tweets yielded a multi-dimensional characterization of public discourse, revealing its thematic structure, emotional cadence, temporal evolution, and geographical heterogeneity.

# 5.1. A Hybrid Framework for Thematic and emotional Knowledge Discovery
Our framework processes social media data through a sequential, integrated pipeline Figure 1. First, tweets are cleaned then labelled into six emotions (anger, disgust, fear, joy, neutral, sadness, surprise) using RoBERTa model, Table 1. Thereafter, the tweets dataset converted into dense vector representations (embeddings) using a Twitter specific transformer model. These embeddings are then clustered into topics using BERTopic, which employs dimensionality reduction (UMAP) and density-based clustering (HDBSCAN) to discover an unknown number of coherent themes.  The BERTopic pipeline successfully segmented the unstructured tweet corpus into coherent thematic clusters. The subsequent processing by the Large Language Model (LLM) generated intuitive and meaningful labels for these clusters, solving the critical interpretability challenge. The seven dominant themes, in descending order of prevalence, were: School Closures & Education (3,229 tweets), Vaccination Rollout & Hesitancy (2,884 tweets), Government Response & Lockdowns (2,422 tweets), Economic Impact & Jobs (2,398 tweets), Travel Restrictions & Borders (2,090 tweets), Information & Media (2,062 tweets), and Mental Health & Safety (2,060 tweets) as shown in Table 3. The LLM-generated summaries provided immediate context; for instance, the "Vaccination" theme was summarized as cantering on "public debate around vaccine efficacy, distribution logistics, and safety concerns," while "Economic Impact" was described as focusing on "job losses, business closures, and financial worry triggered by the pandemic."

# 5.2. The Landscape of Public Concern and Emotion.
Applying this framework to 3.58 million COVID-19 tweets from 2020 uncovered a structured hierarchy of public discourse. We identified seven dominant, LLM labelled themes Figure 4, 6, including "School Closures & Education," "Economic Impact & Jobs," and "Government Response & Lockdowns." The discourse was dominated by a neutral tone (40.1%), indicative of informational sharing, but was characterized by a strong undercurrent of negative emotion, with sadness (14.4%) and anger (11.0%) being most prevalent.
Crucially, each major theme possessed a distinct emotional fingerprint as in Figure 4 and Table 3. The "Information & Media" topic was the most potent source of public anger (30.5% of its tweets), whereas "Economic Impact & Jobs" and "Mental Health & Safety" were primary vessels for sadness (15.3% and 16.1%, respectively). "Vaccination Rollout & Hesitancy" was characterized by a high degree of neutral discourse (45.1%) alongside the highest level of fear (10.2%), reflecting the different between factual news and public worry. Correlation analysis revealed that anger and joy were strongly anti-correlated (r = -0.86), indicating a highly polarized emotional landscape.

# 5.3. Temporal Evolution and Geographical Nuance
The public's attention dynamically shifted across themes in response to the unfolding pandemic Figure 3. Early discourse was dominated by "Travel Restrictions & Borders" and "Government Response," which gave way to sustained discussions on "School Closures & Education" and "Economic Impact." The "Vaccination Rollout & Hesitancy" topic exhibited a classic emergence pattern, growing from negligible volume to dominate the conversation by the end of 2020. By the end of our observation period, foundational policy topics were still gaining momentum, while vaccine discourse showed signs of saturation Table 3.
This global narrative was composed of distinct local stories. Geographical analysis revealed specific "hotspots" of concern Figure 5. Discussion of "School Closures & Education" was disproportionately concentrated in U.S. cities, while "Mental Health & Safety" was a dominant concern in cities in India. This demonstrates that while the pandemic was a shared global event, its manifestation in public discourse was intensely local.

# 6.	Discussion and Conclusion
Our hybrid framework represents a major change in how we analyse social media. By combining the power of  advanced models such as BERTopic with the interpretive abilities of large language models (LLMs), we shift from simply describing data to truly understanding it. Automatically generating clear topic labels and summaries helps solve a key challenge making large collections of text easy for experts to understand right away. This is a major improvement over older methods that only produce keyword lists, which require manual and often subjective interpretation.
The results give a clear view of how the public felt during the COVID-19 pandemic. The high level of anger linked to information and media suggests a crisis of trust, which directly affects how public health messages are received. The lasting sadness connected to economic and mental health themes shows the deep impact of the pandemic beyond physical illness. Tracking these emotions alongside topic trends helps policymakers see not just what people are concerned about, but also how they feel whether it’s fear, which might be eased by information, or anger, which may need different approaches.
Although this framework was used on past data, it is designed to scale and work in near real-time. In the future, it could be adapted for other languages and live social media feeds. The approach can be used beyond public health, in areas like elections, climate change, or new technologies any topic or theme involving fast changing public opinion. By offering a full, connected, and automated way to understand what people are thinking and feeling, this work sets a new benchmark for research in social science and AI supported decision-making.

# 6.1. Limitations and Future Work 
Our study focused on English tweets; future work will expand to multilingual analysis. The framework currently operates in batch mode a streaming implementation would enable true real-time analysis.
In conclusion, this framework provides a powerful, general tool for computational social science and public policy, offering an unprecedented, integrated view of what the public is talking about, how they feel, and how these changes over time and space.



















