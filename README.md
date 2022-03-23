# TwitterFakeID-Detection
Ensemble Method for Recognizing fake Identities on Social media platform

1	INTRODUCTION

There are about 4.48 billion people using social media in the world as of July 2021, which is almost 57 percent of the global population Datareportal (2021). Social media is the easiest method of mass communication in the current technological generation and all information is accessible at the fingertip of anyone with a smart device and Internet Gurajala et al. (2016). Since social media touches the lives of more than half of the living population, there is    a need to maintain and regulate the information generated and delivered in their platforms and to protect the identity of everyone Xiao et al. (2015).
Fake ID is a form of identity deception, where an individual creates an account on a social media platform providing false profile details such as name, age, gender, and marital status.
Fake ID can also be accounts that are generated solely for the purpose of promoting false news. The IDs can not only be human generated, but can also be computer generated.
Computer generated and managed IDs are called bots and they are mostly used to spread false information on social media. These Fake IDs will target either a particular individual or a community or an idea. A perfect example that disseminating false information on social media can have a huge impact on everyone is the 2016 US presidential election. On Social media platforms (mainly Facebook) during the elections, the amount of real news was far less than the amount of fake news Mitchell et al. (2015). The information on social media platforms becomes even more significant when surveyed showed more than 61% of targeted users considered Facebook as their primary source of news Budak (2019). Information on social media is so powerful because they reach people on every corner of the globe. Anyone with a smart device and Internet can access the information and make decisions depending on that information. When a topic trends in twitter, it gets the attention of almost everyone and when an individual owns a bulk amount of fake IDs he/she can trend a topic or idea or information that is not legitimate. We live in a world where the number of likes and comments on a Tweet or post decides the genuineness of information. When there is a way that an individual can generate and manage a large number of accounts, they can dispense large amounts of false and deceptive news on social media, which touches the lives of more than half of the global population. There is a strong need to identify
 
and stop it. The main objective of this work is to build a machine learning model that can predict the credibility of accounts on Twitter. We have chosen Twitter as our platform to perform the study because there are 199 million daily active users in twitter and 500 million tweets are sent per day as per the official twitter website Aslam (2021). The objective of this work is to construct a classification model for identifying Fake IDs on social media platforms.


2	Background Theory

The previous research work done shows that fake IDs can  be identified using filtering Jiang et al. (2016b), rules Alishahi et al. (2015) and machine learning Cresci et al. (2015) .
In filtering, a list of users will be maintained and all the activities from that list will be blocked. It is the most reactive method to identify fake IDs. Only after a user has been identified and verified as a malicious user, the user will be blocked out from the social media platform. This method along with other methods like blacklisting malicious URLs and quarantining bots is used in twitter to control the dissemination of false information Thomas et al. (2011). In rule based identification, rules are established to identify fake accounts.
Rules can be simple rules like presence of a particular word or a phrase or presence of a particular number of words or can be complicated rules based on the activity and type of activities on platform Jiang et al. (2016a); Hayati et al. (2009). These rules are applied to Social media platforms to detect  and eliminate fake accounts Benevenuto et al. (2010). These rules might get outdated and we  might need to adapt to more dynamic rules that depend  on human behaviour Kwak et al. (2010). But, human behaviour is random and deemed to be uncertain and it is harder to form rules that can cover the uncertainties of human behaviour Radziwill and Benton (2016); Egele et al. (2013).
Supervised machine learning models have also proved to be a good way to identify fake accounts. Supervised machine learning models require labelled data. Model learns from this labelled training data and it predicts whether the given account is real or fake. Supervised machine learning models like support vector, naïve Bayes and decision are excellently serving this purpose Tuteja (2016).
Semi-Supervised machine learning models have been used to identify fake IDs on social media. Ebrahimi et al. Tuteja (2016) has proposed a way to identify fake IDs and has shown considerably good results Ebrahimi et al. (2016).
Unsupervised machine learning has been used for detection of fake accounts on Social media platforms  by  Gu  et  al.  (2008),  Wu et al. (2018), and Yahyazadeh and Abadi (2012). They have used clustering to detect fake accounts. Clustering is a commonly used unsupervised machine learning method. In unsupervised machine learning the
 
dataset is not labelled. They are grouped based on their characteristics. This method has proven to be better at identifying bots on twitter.
Using reinforcement learning to train the machine learning model for iden- tifying fake accounts has been proposed by Venkatesan et al (2017). Reinforcement learning tries trial and er- ror methods to come up with the best solution for the problem, but it requires a complex environment. Currently such an environment is not available on social media platforms Bellinger et al. (2012).
Given all these methods to identify fake accounts, the Supervised machine learning model provides the best results based on the previous research work that is done Beleites et al. (2011). Hence we will work on the supervised machine learning model to develop a model aiming at providing better accuracy than the existing ones for identifying fake accounts on social media platforms.

3	Methodology

This section discusses the process involved in identifying fake accounts using supervised machine learning. The proposed methodology involves dataset collection, dataset pre-processing, selection of feature set for constructing the model and ensembling techniques that are used in solving this classification problem.

3.1	Dataset collection

Initial collection of data was done using twitter API. Over 1000 twitter account’s profile data were obtained, out of which 500 were profiles of verified users and the remaining 500 were fake accounts. The ID of verified accounts were obtained from twitter’s official website and ID of fake accounts were obtained from the previous research work Venkatesan et al. (2017). The data from the API response was in JSON format. With the help of Online JSON to excel converter, JSON data was converted to excel format which is convenient for usage in building and training ML models. A similar dataset from Kaggle was also included in our dataset to increase the data points in the dataset.

3.2	Preprocessing the dataset

Data set from Twitter has the following 19 attributes such as id, id_str, screen_name, location, description, url, followers_count, friends_counts, listed_count, created_at, favourites_count, verified, statuses_count, lang, status, default_profile, default_profile_image, has_extended_profile and name. Since there is negligible correlation between the attributes id, id_str, location, description, url, created_at, lang, status and label, we can remove
 
these nine columns from the dataset. We  need to replace ‘screen_name’   and ‘name’ with their lengths ’screen_name_length’ and ‘name_length’ respectively, as features can be only numeric values in sci-kit for machine learning algorithms. Incase of ‘location’ and ‘description’ more than length, the presence or absence of the feature for a particular row is a better judging criterion. So we replace them with a binary valued column indicating the presence or absence of the same. Since we had many missing values in our dataset, we replaced them with 0. The final attribute set used for building the model is verified, followers_count, listed_count, favourites_count, sta- tuses_count, default_count, default_profile_image, has_extended_profile, screen_name_length, name_length, location_binary, description_binary, url_binary. The attributes are listed in the descending order of their correla- tion with the output variable.


3.3	Base models

The classification models chosen for fake ID identification are Decision Tree Classifier,Random Forest Classifier and K-Neighbors Classifier. The three models are used as the base model for our ensemble model. Decision tree model splits the population into two or more homogeneous sub-populations based on the most significant differentiator in input variables until no further splits can be made. Random Forest Classifier selects random samples from the dataset and constructs a decision tree for each sample and gets a prediction result from each decision tree and performs a vote for each predicted result. In K Neighbour as the name suggests it considers K Nearest Neighbors (Data points) to predict the class value for the new Datapoint. For the given dataset, the best train-test splitting ratio between training and testing data was determined by looping from 0.5 to 0.85. For the Random Forest model the accuracy was obtained in the range 0.872 - 0.842 when the train-test ratio was varied from 0.50 to 0.85, the maximum accuracy(0.842) was obtained at 0.55. Similarly for the Decision tree model the accuracy obtained was in range 0.835 to 0.847 with maximum accuracy(0.847) at 0.55, for the K Neighbour the accuracy range was 0.804 to 0.832 with maximum accuracy(0.832) at 0.50. The split ratio at which maximum accuracy was obtained for the given dataset for each of the base models is shown in Table Ⅰ.
These base models produce best accuracy individually for this particular problem statement. The predictions of Random Forest Model, Decision Tree Model, K Nearest Neighbor Model are fed as input for the Max voting Ensemble Model. Based on this prediction Max Voting Ensemble model gives the final prediction.

Table 1 Split ratio at maximum Accuracy for Models
Model Name	          Split Ratio	  Training accuracy	  Test Accuracy 
Random Forest model	  0.55	         1.0	               0.842
Decision Tree model	  0.55	         1.0	               0.847
K Neighbour model	    0.50	         0.877               0.832	


3.4	Ensembling Technique

Supervised learning applications like Machine learning and statistical learning models suffer from generalization error. Generalization error is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data. The difference between prediction (for unseen data) and observed data is induced by sampling error, noise and model inaccuracy. Some of these errors are reducible by selecting the correct algorithm and by tuning the parameters precisely, but it’s seldom possible to eliminate generalization error.
 
In ensemble modeling multiple independent diverse models are used to predict a single outcome. Since the models are diverse and independent, the prediction error decreases and the model gives accurate predictions for unseen data, thereby decreasing generalization error. Even though the ensemble model has multiple base models within the model, it acts and performs as a single model. In ensemble models the individual models can be trained by different methods or algorithms. As long as the base models are diverse and independent, the prediction error of the model decreases and it outperforms the base models individually when an ensemble approach is used.

3.5	Ensemble Model

Since the accuracy of prediction of individual models is not satisfactory, we will use these models as base modes and build an ensemble model which uses the prediction of these base models to boost the overall accuracy of the model. Many styles of ensemble modelling exist, and here we are using a Hard Voting Ensemble model as shown in Fig. 1. Hard voting classifier gives prediction as binary value rather than probability. The overall Accuracy of the ensemble model is 0.881.

Fig. 1 Block Diagram of Machine Learning Model

4	Results and Discussions

The main aim of this work is to build a machine learning classifier model which will predict whether the given twitter account is real or fake accurately. When standard well defined machine learning algorithms such as Random Forest Classifier, Decision Tree Classifier, K-Neighbors Classifier were used to train the model and best possible accuracy that was obtained with the given dataset was 84%. After imposing ensembling techniques the accuracy rose to 88%, this may in fact appear to contradict the principle of Occam’s razor which states to maintain the simplest solution possible unless there is huge variation in result. When we take a deeper look into the magnitude of accounts in social media platforms, it becomes clearer that every single percentage increase in accuracy makes a greater contribution. As already stated there are 199 million daily active twitter users and 500 million tweets sent per day.

4.1	CONCLUSIONS

There are about 4.48 billion people using social media in the world as of July 2021, which is almost 57 percent of the global population. Social media is the easiest method of mass communication in the current technological generation and all information is accessible at the fingertips of anyone with a smart device and internet. Since social media touches the lives of more than half of the living
 
population, there is a need to maintain and regulate the information generated and delivered in their platforms and to protect the identity of everyone. In the proposed method to identify fake ID, we have used Machine Learning classifier models to identify fake IDs, and we have used Ensembling techniques to increase the accuracy of prediction. The final model had overall accuracy of
88.10 % and this can be used as a framework with automatic detection of fake profiles. It can be applied easily by online social networks which have millions of profiles whose profiles cannot be examined manually. This is very useful in places where there are a lot of user profiles, manual evaluation will be difficult in this case, So in such areas of social networking this technology can be used.


References

Alishahi MS, Mejri M, Tawbi N (2015) ‘Clustering Spam Emails Into Cam- paigns. Proc Int Conf Inf Syst Secur Privacy (ICISSP) pp 90–97
Aslam S (2021) URL https://www.omnicoreagency.com/twitter-statistics/ Beleites C, Geiger K, Kirsch M, Sobottka SB, Schackert G, Salzer R (2011)
‘Raman spectroscopic grading of astrocytoma tissues: Using soft reference information. Anal Bio anal Chem 400(9):2801–2801
Bellinger C, Sharma S, Japkowicz N (2012) ‘One-class versus binary classifi- cation: Which and when? Proc 11th Int Conf 2:102–106
Benevenuto F, Magno G, Rodrigues T, Almeida V (2010) ‘Detecting spammers on Twitter. Proc Collaboration, Electron Messaging, Anti Abuse Spam Conf (CEAS) 6:12–12
Budak C (2019)
Cresci S, Dipietro R, Petrocchi M, Spognardi A, Tesconi M (2015) ‘Fame for sale: Efficient detection of fake Twitter followers. Support Syst 80:56–71
Datareportal (2021) URL https://datareportal.com/social-media-users Ebrahimi M, Suen CY, Ormandjieva O, Krzyzak A (2016) ‘Recognizing preda-
tory chat documents using semi-supervised anomaly detection. Electron Imag 2016(17):1–9
Egele M, Stringhini G, Kruegel C, Vigna G (2013) ‘Compa: Detecting com- promised accounts on social networks. Proc NDSS pp 1–17
Gu G, Perdisci R, Zhang J, Lee W (2008) ‘BotMiner: Clustering analysis      of network traffic for protocol-and structure-independent botnet detection. Proc USENIX Secur Symp 5:139–154
Gurajala S, White JS, Hudson B, Voter BR, Matthews JN (2016) ‘Profile char- acteristics of fake Twitter accounts. Big Data Soc 3(2):2053951716674236– 2053951716674236
Hayati P, Potdar V, ‘ (2009) Toward spam 2.0: An evaluation of Web 2.0 anti- spam methods. Proc 7th IEEE Int Conf Ind Inform (INDIN) pp 875–880 Jiang M, Cui P, Faloutsos C (2016a) ‘Suspicious behavior detection: Current
trends and future directions. Intell Syst 31(1):31–39
Jiang M, Cui P, Faloutsos C (2016b) ‘Suspicious behavior detection:Current trends and future directions. Intell Syst 31(1):31–39
 
Kwak H, Lee C, Park H, Moon S (2010) ‘What is Twitter. Proc 19 th Int Conf World Wide Web pp 591–600
Mitchell A, Gottfried J, Matsa KE (2015) URL http://www.journalism.org/ 2015/06/01/facebooktop-source-for-politicalnews-among-millennials/
Radziwill NM, Benton MC (2016) URL https://arxiv.org/abs/1605.06555 Thomas K, Grier C, Song D, Paxson V (2011) ‘Suspended accounts in
retrospect: An analysis of Twitter spam. Proc ACM SIGCOMM Conf Internet Meas Conf pp 243–258
Tuteja SK (2016) ‘A survey on classification algorithms for email spam filtering. Int J Eng Sci 6(5):5937–5940
Venkatesan S, Albanese M, Shah A, Ganesan R, Jajodia S (2017) ‘Detecting stealthy botnets in a resource-constrained environment using reinforcement learning. Proc Workshop Moving Target Defense pp 75–85
Wu W, Alvarez J, Liu C, Sun HM (2018) Bot detection using unsupervised machine learning. Technol 24(1):209–217
Xiao C, Freeman DM, Hwa T (2015) ‘Detecting clusters of fake accounts in online social networks. Proc 8 th ACM Workshop Artif Intell Secur pp 91– 101
Yahyazadeh M, Abadi M (2012) ‘BotOnus: An online unsupervised method for botnet detection. Int J Inf Secur 4(1):51–62
 
