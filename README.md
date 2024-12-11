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



## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis

