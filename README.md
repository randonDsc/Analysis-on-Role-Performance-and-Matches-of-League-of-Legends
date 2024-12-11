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

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis

