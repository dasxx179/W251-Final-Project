# Predicting NFL Rushing Yardage

> Based off the NFL Big Data Bowl Kaggle [Challenge](https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview)

## Table of Contents

- [Purpose](#Purpose)
- [Installation](#Installation)
- [Features](#Features)
  - [Datasets](#Datasets)
  - [EDA](#EDA)
  - [Data Modification](#Data-Modification)
- [Model](#Model)
  - [Model Architecture](#Model-Architecture)
  - [Model Development](#Model-Development)
- [Cloud Deployment](#Cloud-Deployment)
  - [Cloud Architecture](#Cloud-Architecture)
- [Results](#Results)
- [Visualization](#Visualization)
- [White Paper](#White-Paper)
- [FAQ](#FAQ)
- [Team](#Team)
- [References](#References)

## Purpose

Sports can be quite complex.  While many sports have an extremely simple goal, the actual mechanics and characteristics of sports can be incredibly arduous to comprehend.  In particular, American football is a sport comprised of a myriad of different plays and actions that can impact the objective of gaining yardage to accomplish the ultimate goal of scoring (ideally a touchdown).  Of course, the offense wants to score by run (rush) or throw (pass) plays with a ball to gain yards, moving towards the opposing teams' side of the field to score.  The defense wants to prevent the offensive teams' objective of scoring.

In the National Football League (NFL), about a third of teams' offensive yardage comes from run plays.  These run plays are thought to be attributed primarily to the talent of the ball carrier; however, the ball carrier's teammates, coach, and opposing defense have an enormous impact in the success of the ball carrier.  

Ultimately, this project explores game, play, and player-level data provided by NFL's Next Gen Stats.  Various neural network architectures and model types are experimented with and explored, and ultimately, deeper insights can help any lover of football to understand the various skills and strategies of players and coaches.   

## Installation

- Clone this repository to local machine using: 

  ```git
  git clone https://github.com/dasxx179/W251-Final-Project.git
  ```

- Use the **parsedData.csv** file (modified from the original **train.csv** file) to run the below scripts:
  - Idk
  - Idk
  - LOL idk

## Features

### Datasets

> Sayan TODO

### EDA

> Sayan TODO

### Data Modification

> Jake plz talk about what ya did.  

## Model

### Model Architecture

> Peter plz help.

### Model Development

> Peter plz help.

## Cloud Deployment

In order to deal with the large amount of data, we decided to deploy our model in the cloud in order to take avantage of higher powered v100 GPUs. We used the GPUs in the IBM Cloud, and ran Pytorch with CUDA.  We ran our model itsself inside the Docker container (w251 docker bases).

### Cloud Architecture

> We used a singular GPU. It took us XX hours to finish the model training, and XX hours for inference. 

## Results

We will compare our results to those from the Kaggle competition, which uses a Continuous Ranked Probability Score. There, competitors must predict a cimulative probability of yards achieved, strarting from -99 to +99 yards. 

However, we choose not to restrict ourselves. The first approach is to take the yards gained by the rusher, and use that as a numeric value that we want to predict. In this case, the data is available in the dataset given by Kaggle.

The second option is to compute a Continuous Ranked Probability Score. It is essentially the MSE of the cumulative density function. We can use the Properscoring library in order to caculate CRPS. For an arbitrary cumulative distribution function, we can use the crps_quadrature command. This package implements CRPS - https://github.com/raybellwaves/xskillscore.

Can also use (https://github.com/aguschin/kaggle/blob/master/rain_crps.py). However, we need the actual cumulative distribution function, which I don' think we have. 

## Visualization

## White Paper

## FAQ

- How did you guys do on the Kaggle leaderboard?

> We actually didn't submit our model to Kaggle, for in order to compete in the Kaggle competition, it is necessary to use their builtin kaggle module.  For the purposes of this project, we instead chose to tweak the challenge to our own needs.

- How large was your dataset?

> It was X gigs.

- What inspired you to choose this project topic?

> In the evergrowing field of AI, ML and DL, algorithms are getting more and more complex.  Often, it is difficult to comprehend what the changes are in the field and what impact is ultimately made.  Especially for individuals that are less technology versed, it can be truly difficult to comprehend the meaningful effect of these fantastic technologies.  Thus, our group decided to choose a project topic that is meaningful and relatable to anyone: sports.  The NFL Big Data Bowl challenge gave us a headstart on the actual framework for our project, but ultimately, we just want to show that these remarkable emerging technologies are relatable to everyone, regardless of their background.

### Team

* Sayan Das
* Hyunchul (Peter) Kim
* Hersh Solanki
* Jake Tosh

## References

- https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview
- https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400
- https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
