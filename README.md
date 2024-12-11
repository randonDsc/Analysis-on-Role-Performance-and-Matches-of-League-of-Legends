# Analysis On Role Performance And Matches of League of Legends

by Po-Chen Lin

This is a project for DSC 80 at UCSD.

# League of Legends Role Performance Analysis

Ths is a data science project includes a full life cycle of data science. It begins from data cleanning, exploratory data analysis, hypothesis testing, and creating the model for prediction. This project consist of two main part. First, exploring the data to see which role carries the team most often. Second, making a prediction model to predict the rolw of a player given the data after the matches.

## Introduction

### General Intro
League of Legends(LOL) is a MOBA video games which players compete in a map. The goal of the game is to destroy the "tower" of opponents team and eventually destroy the base. Within the map, there are three main paths: bot, mid, and top. It also have jungle(jng) for the player to make money by killing the mosters. 

The dataset we used in this project is from Oracle's Elixir. Specifically, the records of the match data from professional leagues in the 2022. The original dataset is a csv files containing 161 features of the matches such as economics, KDAs, damages to other players, and other in game statistics.

Within every match, players are usually separated into different position: bottom, middle, jungle, top, and support. Each position will have different in game dynamics. For instance, bottom is usually have high damage and low health, thus requires the help of support(sup). The jungle(jng) explores across the map, gain gold by killing monsters, and assassinate palyers. 

This is a key components of LOL. Thus, this project is center around the question: Which role “carries” (does the best) in their team more often? We will navigate the question by doing hypothesis testing, drawing graphs, and ultimately, make predictions to the positions of each players.

### Intro to Columns
Though there are 161 features in the dataset, we are only going to focus few of them, which includes:

-`gameid`: the unique id for each matches

-`datacompleteness`: record whether the data is complete. Some matches only have partial data

-`side`: the team of the player

-`position`: the role of players

-`gamelength`: records the time the match last

-`result`: record which team wins the match

-`kills`: number of players kill durign the match

-`assists`: number of player one helps to kill

-`deaths`: number of times one get killed

-`damagetochampions`: amount of damage to the opponents

-`dpm`: damage to the opponents per minutes

-`damagemitigatedperminute` amount of damage recieve but not decrease the health bar

-`totalgold`: total gold a player have throughout the match

-`minionkills`: number of minions killed


## Data Cleaning and Exploratory Data Analysis

### Data Cleanning
In the dataset, each 'gameid' corresponds to up to 12 rows – one for each of the 5 players on both teams and 2 containing summary data for the two teams. Therefore, I first separated the summary data and the rows of individual players. For the player df, I only keeps the following columss: ['gameid', 'datacompleteness', 'league', 'side', 'position', 'gamelength', 'playerid',
                            'result', 'kills','deaths', 'assists', 'damagetochampions', 'dpm', 'damageshare', 'damagetakenperminute', 
                            'damagemitigatedperminute', 'totalgold', 'earnedgold', 'earned gpm', 'earnedgoldshare', 'minionkills', 
                            'monsterkills', 'cspm']
Below is the first five rows of the resulting dataframe:

| gameid                | datacompleteness   | league   | side   | position   |   gamelength | playerid                                  | result   |   kills |   deaths |   assists |   damagetochampions |     dpm |   damageshare |   damagetakenperminute |   damagemitigatedperminute |   totalgold |   earnedgold |   earned gpm |   earnedgoldshare |   minionkills |   monsterkills |   cspm |   edit_length |
|:----------------------|:-------------------|:---------|:-------|:-----------|-------------:|:------------------------------------------|:---------|--------:|---------:|----------:|--------------------:|--------:|--------------:|-----------------------:|---------------------------:|------------:|-------------:|-------------:|------------------:|--------------:|---------------:|-------:|--------------:|
| ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   | top        |         1713 | oe:player:38e0af7278d6769d0c81d7c4b47ac1e | False    |       2 |        3 |         2 |               15768 | 552.294 |     0.278784  |               1072.4   |                    777.793 |       10934 |         7164 |      250.928 |          0.253859 |           220 |             11 | 8.0911 |           114 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   | jng        |         1713 | oe:player:637ed20b1e41be1c51bd1a4cb211357 | False    |       2 |        5 |         6 |               11765 | 412.084 |     0.208009  |                944.273 |                    650.158 |        9138 |         5368 |      188.021 |          0.19022  |            33 |            115 | 5.1839 |           114 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   | mid        |         1713 | oe:player:d1ae0e2f9f3ac1e0e0cdcb86504ca77 | False    |       2 |        2 |         3 |               14258 | 499.405 |     0.252086  |                581.646 |                    227.776 |        9715 |         5945 |      208.231 |          0.210665 |           177 |             16 | 6.7601 |           114 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   | bot        |         1713 | oe:player:998b3e49b01ecc41eacc392477a98cf | False    |       2 |        4 |         2 |               11106 | 389.002 |     0.196358  |                463.853 |                    218.879 |       10605 |         6835 |      239.405 |          0.242201 |           208 |             18 | 7.9159 |           114 |
| ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   | sup        |         1713 | oe:player:e9741b3a238723ea6380ef2113fae63 | False    |       1 |        5 |         6 |                3663 | 128.301 |     0.0647631 |                475.026 |                    490.123 |        6678 |         2908 |      101.856 |          0.103054 |            42 |              0 | 1.4711 |           114 |


After selecting the columns for the player dataframe, I found out `playerid`, `damagetochampions`, `dpm`, `damageshare`, `damagetakenperminute`, `damagemitigatedperminute`, and `minionkills` have missing values. damagetochampions has 10 missing values, and they are all from the same match, so does dpm. Therefore, I drop the rows of that match. Then I decide to do the conditional probabilistic imputation to the `damagemitigatedperminute`, so I put gamelength into groups by ground dividing them by 15. I then groupby the edit length, and found 51, 68, 213, 224 all have 10 missing values. Thus, I drop them and continue to conduct the imputation. After replacing the `result` column with Boolean values, I start to clean team dataframe.

For team df, I kept the following columns: ['gameid', 'datacompleteness', 'league', 'side', 'gamelength', 'result', 'teamkills', 'teamdeaths', 'assists',
                   'team kpm', 'dragons', 'opp_dragons', 'dpm', 'damagetakenperminute', 'damagemitigatedperminute',
                  'damagetochampions', 'totalgold', 'earnedgold', 'earned gpm', 'gspd', 'minionkills', 'monsterkills', 'cspm']

Below is the head of team dataframe:

| game_side                 | gameid                | datacompleteness   | league   | side   |   gamelength |   result |   teamkills |   teamdeaths |   assists |   team kpm |   dragons |   opp_dragons |     dpm |   damagetakenperminute |   damagemitigatedperminute |   damagetochampions |   totalgold |   earnedgold |   earned gpm |        gspd |   minionkills |   monsterkills |    cspm |
|:--------------------------|:----------------------|:-------------------|:---------|:-------|-------------:|---------:|------------:|-------------:|----------:|-----------:|----------:|--------------:|--------:|-----------------------:|---------------------------:|--------------------:|------------:|-------------:|-------------:|------------:|--------------:|---------------:|--------:|
| ESPORTSTMNT01_2690210Blue | ESPORTSTMNT01_2690210 | complete           | LCKC     | Blue   |         1713 |        0 |           9 |           19 |        19 |     0.3152 |         1 |             3 | 1981.09 |                3537.2  |                    2364.73 |               56560 |       47070 |        28222 |      988.511 | -0.0283123  |           680 |            160 | 29.4221 |
| ESPORTSTMNT01_2690210Red  | ESPORTSTMNT01_2690210 | complete           | LCKC     | Red    |         1713 |        1 |          19 |            9 |        62 |     0.6655 |         3 |             1 | 2799.02 |                3009.67 |                    2872.33 |               79912 |       52617 |        33769 |     1182.8   |  0.0283123  |           792 |            184 | 34.1856 |
| ESPORTSTMNT01_2690219Blue | ESPORTSTMNT01_2690219 | complete           | LCKC     | Blue   |         2114 |        0 |           3 |           16 |         7 |     0.0851 |         1 |             4 | 1690.98 |                2984.02 |                    3109.61 |               59579 |       57629 |        34688 |      984.522 | -0.207137   |           994 |            215 | 34.3141 |
| ESPORTSTMNT01_2690219Red  | ESPORTSTMNT01_2690219 | complete           | LCKC     | Red    |         2114 |        1 |          16 |            3 |        39 |     0.4541 |         4 |             1 | 2124.55 |                2745.72 |                    2868.42 |               74855 |       71004 |        48063 |     1364.13  |  0.207137   |          1013 |            244 | 35.6764 |
| 8401-8401_game_1Blue      | 8401-8401_game_1      | partial            | LPL      | Blue   |         1365 |        1 |          13 |            6 |        35 |     0.5714 |         2 |             1 | 1762.02 |                2263.25 |                    2596.82 |               40086 |       45468 |        30167 |     1326.02  | -0.00586225 |           578 |            172 | 32.967  |


I then remove the match I remove previously in the player dataframe, and found `damagemitigatedperminute` and `minionkills` have missing values, both missed 3776 values. Since these can be calculated from the player dataframe, I found these missing values by summing up the data in player dataframe and filled in the missing valies according to the team and gameid.


### Univariate Analysis

I performed the Univariate Analysis on total gold, damage to champions, and kills

Here is the distribution of total gold, with the mean of 11383.029248601119:

<iframe
  src="assets/total_gold.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

It appears that the distribution of total gold have a trends of skewed to the right. This might suggest that few players tends to have advantage in economic in the games.

Here is the distribution of damage to champions, with the mean of 13336.50582733813:

<iframe
  src="assets/damagechamp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of damage to champions have even mroe severe skewness. This might illustrate that certain role is better at causing damages.


Here is the distribution of kills, with the mean of 2.897857713828937 and also skewed to the right:
<iframe
  src="assets/kill.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Bivariate Analysis

I conduct Bivariate Analysis on relationship between economic and number of kills. There seems to have a positive relationship between them, indicating that amount of gold might be one of the cause of games result. Here is the graph:

<iframe
  src="assets/gold_kill.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates
One way to measure the performance of roles is see how well they utilize economy they earned. So I compute the (Kills + Assistance) / total gold, which measure how well they utilize gold they earned. I named this feature KA_gold_ratio. I create a pivot table showing the mean of KA_gold_ratio among different position and compare the value among winning team and losing team:

'result' |   ('mean', 'bot') |   ('mean', 'jng') |   ('mean', 'mid') |   ('mean', 'sup') |   ('mean', 'top') |
---------|------------------:|------------------:|------------------:|------------------:|------------------:|
'losing' |       0.000462342 |       0.000629245 |       0.000493052 |       0.000871795 |       0.000414605 |
'winning'|       0.00090964  |       0.00114483  |       0.000960187 |       0.00168347  |       0.000822086 |

It appears that across all position, winning team tends to have higher KA_gold_ratio. I will conduct hypothesis testing on this topic later. This might shows that KA_gold_ratio is a good indicator of a role's performance.


## Assessment of Missingness

### NMAR Analysis
In our dataset, I believe that the column 'playerid' is NMAR. The number of plarerid missing is very different with number of rows that have 'partial' for data completeness. Thus, it is unlike for it to be MD. Furthermore, majority of other columns are related to the statistics during the gameplay, so it is also unlikely for it to depends on those columns. It is likey that the playerid is missing simply because the value itself is missing.

### Missingness Dependency
Here I doubt whether the missingness of 'damagemitigatedperminute' is depends on League column since different league might have differnt ways of recording the matches. Therefore, I conduct the permutation test using the tvd as statistic. 

Null Hypothesis: Distribution of league when damagemitigatedperminute is missing is the same as the distribution of league when damagemitigatedperminute is not missing.

Alternative Hypothesis: Distribution of league when damagemitigatedperminute is missing is NOT same as the distribution of league when damagemitigatedperminute is not missing.

Here is the observed statistic:

|   damagemitigatedperminute_missing = False |   damagemitigatedperminute_missing = True |
|-------------------------------------------:|------------------------------------------:|
|                               nan          |                                0.0390914  |
|                                 0.022877   |                              nan          |
|                                 0.0203352  |                              nan          |
|                                 0.00715496 |                              nan          |
|                                 0.00244775 |                              nan          |
|                               nan          |                                0.0406762  |
|                                 0.0196761  |                              nan          |
|                                 0.0174167  |                              nan          |
|                                 0.0127095  |                              nan          |
|                                 0.0225946  |                              nan          |
|                                 0.0251365  |                              nan          |
|                                 0.0163811  |                              nan          |
|                                 0.0191113  |                              nan          |
|                                 0.0152514  |                              nan          |
|                                 0.0144041  |                              nan          |
|                                 0.00706082 |                              nan          |
|                                 0.0213707  |                              nan          |
|                                 0.0439654  |                              nan          |
|                                 0.037187   |                              nan          |
|                                 0.00150631 |                              nan          |
|                                 0.0199586  |                              nan          |
|                                 0.0288081  |                              nan          |
|                                 0.0508379  |                              nan          |
|                               nan          |                                0.497623   |
|                                 0.022877   |                              nan          |
|                                 0.0232536  |                              nan          |
|                                 0.0226888  |                              nan          |
|                                 0.0229712  |                              nan          |
|                                 0.020241   |                              nan          |
|                                 0.00357748 |                              nan          |
|                                 0.017605   |                              nan          |
|                                 0.0301262  |                              nan          |
|                               nan          |                                0.415214   |
|                                 0.0200527  |                              nan          |
|                                 0.0230653  |                              nan          |
|                                 0.00753154 |                              nan          |
|                                 0.018264   |                              nan          |
|                                 0.0361514  |                              nan          |
|                                 0.0257014  |                              nan          |
|                                 0.0530973  |                              nan          |
|                                 0.0141216  |                              nan          |
|                                 0.0350217  |                              nan          |
|                                 0.0155338  |                              nan          |
|                                 0.0192996  |                              nan          |
|                                 0.0209942  |                              nan          |
|                                 0.0229712  |                              nan          |
|                                 0.0387874  |                              nan          |
|                                 0.0305969  |                              nan          |
|                                 0.0160045  |                              nan          |
|                                 0.0132743  |                                0.00739567 |


Here is the result of the permutation test:
<iframe
  src="assets/tvdmiss.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Since p value is 1.0, which is larger that the significance level, thesefore, we fail reject the null hypothesis and missingness of 'damagemitigatedperminute' is not depends league.

Now I conduct another testing to test whether the missingness is dependent on length of the game, using the KS statistic:

Null Hypothesis: Distribution of gamelength when damagemitigatedperminute is missing is the same as the distribution of gamelength when damagemitigatedperminute is not missing.

Alternative Hypothesis: Distribution of gamelength when damagemitigatedperminute is missing is NOT same as the distribution of gamelength when damagemitigatedperminute is not missing.
The result is the following:
KstestResult(statistic=np.float64(0.06054070715893012), pvalue=np.float64(1.2377453593789456e-51), statistic_location=np.int64(1726), statistic_sign=np.int8(1))

Since p value is 1.2377453593789456e-51, which is smaller that the significance level 0.05, thesefore, we reject the null hypothesis and missingness of 'damagemitigatedperminute' is MAR depends gamelength.

## Hypothesis Testing
Here is the graph of difference in distribution of KA_gold_ratio among winning and losing players:

<iframe
  src="assets/hypo_ka.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

It appears that the mean of KA_gold_ratio of winning team is higher than the losing team. To confirm this, we conduct the permutation testing.
null: (kill + assist / gold) of players in the winning team is the same with the losing team.

### Testing 1
alternative: (kill + assist / gold) of players in the winning team is higher than player in the losing team
Result: the p value of the permutation testing is 0, thus, we reject the null hypothesis

Now we know that is is not likely to cause by probability, so we continue to explore the quesiont "Which role “carries” (does the best) in their team more often?" using this features.

Here we see the bar graph of KA_gold_ratio among different roles:

<iframe
  src="assets/mean_ka.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

It appears that support have higher KA_gold_ratio compares to other roles, so we coduct hypothesis testing:

<iframe
  src="assets/sup_ka.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### hypothesis testing 2:

null: The mean KA_gold_ratio in every role is the same.

alternative: The mean KA_gold_ratio of support role is higher than other role

Result: the p value of the permutation testing is 0, thus, we reject the null hypothesis

hypothesis testing 3: 

The testing above might impact our decision because the KA_gold_ratio jungle might be impact by other roles' KA_gold_ratio. So we conduct the third hypothesis testing

null: The mean KA_gold_ratio of jungle and support is the same

alternative: The mean KA_gold_ratio of support role is higher than jungle

Here is a graph showing the comparison between jungle and support:

<iframe
  src="assets/hypo3.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Result: the p value of the permutation testing is 0, thus, we reject the null hypothesis


### Conclusion:

In the first permutation test, we shows that KA_gold_ratio of player in the winning team is likely to be higher than the losing team. By doing so, we shows that KA_gold_ratio is a ok statistic to measure the performance of roles. In the second permutation testing, we examing whether the KA_gold_ratio of support role is likely to be higher than the other roles, the result of the test is rejecting the null hypothesis. Thus, the answer to the topic question Which role “carries” (does the best) in their team more often? is support, which is in the Bot lanes most of times


## Framing a Prediction Problem

In this part of the project, we are going to create a random forest model predicting which role a player played given their post game data. This is a classification problem. We will use accuracy to measure the performance of the model. Because we have multiple outcomes, we cannot use precision or recall to measure the model.

## Baseline Model

For the baseline model, I use KA_gold_ratio and damagemitigatedperminute as my features. As illustrated above, KA_gold_ratio is a features that is good at measureing the performance of the roles and it is also not impacted by the length of the match. I choose damagemitigatedperminute as my another features because the top roles usually have higher defense, so I hope this features can help differentiate the role. I use n_estimators=30, max_depth=10 for my random forest 


The accuracy of the training set is 0.48710031974420465
The accuracy of the testing set is 0.461550759392486


## Final Model

For the final model, I add the kills per minutes, minions kill per minutes, and assist per minutes to the set. However, my model might have some bug.

## Fairness Analysis

