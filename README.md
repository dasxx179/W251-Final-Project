### Team

* Sayan Das
* Hyunchul (Peter) Kim
* Hersh Solanki
* Jake Tosh

# Predicting NFL Rushing Yardage

> Based off the NFL Big Data Bowl Kaggle [Challenge](https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview)

## Table of Contents

- [Purpose](#Purpose)
- [Features](#Features)
  - [Datasets](#Datasets)
  - [EDA](#EDA)
    - [Feature Engineering](#Feature-Engineering)
  - [Data Modification](#Data-Modification)
- [CNN Model](#CNN-Model)
  - [CNN Model Architecture](#CNN-Model-Architecture)
  - [CNN Model Development](#CNN-Model-Development)
- [Experimentation](#Experimentation)
  - [Hyperparameters](#Hyperparameters)
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

## Features

### Datasets

> The dataset used for this project contains Next Gen Stats tracking data for various running plays.  The head of the original dataset is shown below:
>
> ![Df Image](/src/images/originalDfhead.png "Df Head")

> The original dataset shown above is modified in two different ways to establish two models.  In the EDA section below, the dataset is modified using feature engineering and other methods to establish a baseline model based off different kaggle references.  This model is used as a reference for comparison to our group's own model, which is described in the Model section.  The alteration of the original dataset for our model is described in the below Data Modification section.

### EDA

> In order to establish the baseline model, it is necessary to clean up the original dataset and add new features along the way.
>
> At first, some features of the data are visualized such as the yards gained and lost for a random game.  The yard feature in particular is examined because ultimately, this feature is what will be predicted from our models.  The graph is shown below:
>
> ![Yard Image](/src/images/yardImage.png "Yards")
>
> Based on the plot above, there are many run plays that are occurring in this game, which implies that the teams most likely had two powerful runningbacks.  The majority of the plays resulted in yardage gain, and one of the runs resulted in a large gain of over 25 yards.  This is an example of a good play to analyze to determine the various defensive formations that are used against running plays.

> Another powerful visualization that was conducted during the EDA was examining the number of yards gained based on defensive schema.  The average yards gained based on each defensive schema is shown below:
>
> ![Defensive Schema](/src/images/defensiveSchema.png)
>
> By analyzing the different defensive schemas that are used by the teams, it is easy to see that the most common schema used is a 4-2-5, which is typically used against passing plays.  This is most likely attributed to the fact that the majority of the plays in the NFL are passing plays and not rushing plays.  However, a 4-2-5 is also effective at covering a run play, for the median yardage gain is 4 yards.  Additionally, it is shown that 75% of the plays with this schema are held to 6 yards or less.  This type of analysis helps determine which specific defensive schemas are more effective than others in the NFL.
>
> #### Feature Engineering
>
> First, it is important to gain an understanding of what categorial features are included in the dataset.  The following features are included:
>
> ![Categorical Features](/src/images/categoricalFeatures.png)
>
> Let's take a closer look at some of these categorical features.
>
> ##### Stadium Type
>
> The following image shows the counts of the different stadium types:
>
> ![Stadium Type](/src/images/stadiumType.png)
>
> As apparent from above, there are numerous variations and even misspellings of Indoor and Outdoor stadiums.  Let's clean it up and fix some of these typos.
>
> ```python
> def clean_StadiumType(txt):
>     if pd.isna(txt):
>         return np.nan
>     txt = txt.lower()
>     txt = ''.join([c for c in txt if c not in punctuation])
>     txt = re.sub(' +', ' ', txt)
>     txt = txt.strip()
>     txt = txt.replace('outside', 'outdoor')
>     txt = txt.replace('outdor', 'outdoor')
>     txt = txt.replace('outddors', 'outdoor')
>     txt = txt.replace('outdoors', 'outdoor')
>     txt = txt.replace('oudoor', 'outdoor')
>     txt = txt.replace('indoors', 'indoor')
>     txt = txt.replace('ourdoor', 'outdoor')
>     txt = txt.replace('retractable', 'rtr.')
>     return txt
> train['StadiumType'] = train['StadiumType'].apply(clean_StadiumType)
> ```
>
> Also, let's convert all outdoor or open stadium types to 1 and indoor or closed stadium types to 0:
>
> ```python
> def transform_StadiumType(txt):
>     if pd.isna(txt):
>         return np.nan
>     if 'outdoor' in txt or 'open' in txt:
>         return 1
>     if 'indoor' in txt or 'closed' in txt:
>         return 0
>
>     return np.nan
> train['StadiumType'] = train['StadiumType'].apply(transform_StadiumType)
> ```
>
> Now, our counts for the stadium type feature look like:
>
> ![Stadium Counts](/src/images/stadiumCounts.png)
>
> This categorical feature is now ready for input to our baseline model.
>
> Similarly, we process the other categorical features that were shown above.  In addition, we create  other features useful for our baseline model such as the numbers of Defenders in the Box vs Distance.  These features that are not categorical are shown below:
>
> ![Train Features](/src/images/trainFeatures.png)
>
> Ultimately, these are the features that provide an exploration of the dataset and gave us an understanding of what features to keep in mind for our model development.

### Data Modification - see https://github.com/dasxx179/W251-Final-Project/blob/master/src/tensorCreation.ipynb

> To get this data into image format for our CNN, there were several data modifications that needed to be accomplished. First we needed to standardize the field such that the offense was always moving from left to right. To do this, we start by adjusting the players' X,Y, and direction values. We also need to adjust the line of scrimmage to reflect the changes to the field. To do this we not only change the X position of the line of scrimmage, but we also need to change the range of the line of scrimmage from 0 to 50 to 0 to 100 so that we don't need to specify own yard lines versus opponent's yard line.
>
> Now that the field is standardized, we want to put our information into a pixel-based 3 dimensional matrix. This is tricky however since our X and Y values for our players have decimals and without the decimals, many of our players overlap. To overcome this challenge, we multiplied the X and Y values of the players and the field by 3 to get the dimensions in units of 1/3 yards (or a foot). This reduced our chances of overlapping once we did our rounding for putting players in pixels. At the end of this process we only lost about 500 plays due to overlaps out of 31000, which is about 1.5%. Once putting our data into image format, we realized that our images took up too much disk space and that our images were more sparse than they needed to be.
>
> To fix this, we standardized our field again, such that all plays occured at the 50 yard line and cropped the field such that we would fit our largest play, but not have any additional field. While this regularization could potentially affect the information given to our CNN since field position determines the maximum number of yards that can be gained on a play, we maintain the original field position in our line of scrimmage value.
>
> Now that our data has been modified appropriately to create our images, we iterate through each player in a play and place them and their information in the appropriate pixel. The information we include in the pixel are: whether the player is on offense or defense, whether the player is the ball carrier or not, the player's speed, their acceleration, their direction, the distance they have traveled since the start of the play, and their football position (Quarterback, linebacker, etc.). We also include in the pixel whether or not the line of scrimmage is included on that pixel and where the line of scrimmage would actually be if the play were not regularized.
>
> The final dimensions of our images end up being 176 by 192 by 9 where 176 is how wide our field is, 192 is how long our field is, and 9 is how many features we include in our image.

## CNN Model
The existing data structure is not suitable for the CNN model, which is widely used in many of the computer vision area. In order to use CNN, we constructed following data structure. The main idea is to construct pixel-like tensor data structure using the player's x and y coordinate in the training data set. On each player's position, we appended player-specific information such as player's speed, acceleration, angle and etc.

> ![tensor definition](/src/images/tensor_definition.png)


### CNN Model Architecture

Once we successfully convert original dataframe to tensor format, we can define CNN model. The baseline CNN model is simple 4 layer model. The first 3 layers are Convolutional network and last layer is fully-connected network. There are two significant difference between our model and the commonly used CNN model such as AlexNet or VGGNet. First the last layer use linear activation function instead of softmax. This is because our model is for regression rather than classification. Secondly, the kernel size we used is somewhat large compared to the conventional model The rationale behind this is that unlike image data with rich information in all the pixels, the density of meaningful data in our tensor structure is very low. Thus to capture certain player formation leading to yardage, kernel size should be large enough to cover multiple play'er data. Based on the histogram analysis, the player's distribution in Y is somewhere between 1 and 59. So we chose 64 as the kernel size. The further optimization can be made.


> ![CNN architecture](/src/images/CNN_architecture.png)


<img src="/src/images/convnet_kernel_size.png" width="400">



The original data has different dynamic range and prevent the network from converging. Thus it is important to standarize the data. The following figure shows the data distribution before and after data standarization.

> ![CNN architecture](/src/images/data_normalization.png)

### Toy model experiment

After model standarization and model selection, we have tested small 10x play data sets in Google Colab and confirmed model converges as shown below. This small dataset was split into 6 training and 4 test sets for experiment.

#### Toy model performance

Unlike the reasonable validation loss and no sign of overfitting based on the small data set, validation loss from the large training data set showed huge gap between training and validation loss. This means toy model is not sufficient enough to predict large dataset.
<img src="/src/images/initial_hyper.png" width="800">


#### Improved model with additional normalization
In order to improve the performance, firstly, we have shifted our yardage to the positive direction to make all the yardage values positive. In addition, we also applied log to make the skewed data closer to the normal distribution (See picture below for reference). By doing this, all the intermediate layer output can be witihin activation function output range, which makes us fully utilize out network model. Secondly, we added more layers as well as L2 regularization at each layer to prevent overfitting, which frequently occurs to the high dimension neural network model.

<img src="/src/images/updated_hyper.png" width="800">


## Cloud Deployment

In order to deal with the large amount of data, we decided to deploy our model in the cloud in order to take avantage of higher powered v100 GPUs. We used the GPUs in the IBM Cloud, and ran Keras/Tensorflow with CUDA.  We ran our model inside the Docker container (w251 hw6 base docker), inside of a Jupyter notebook.

We used a singular GPU. It took us 2.5 hours to finish the model training, with inference time being trivial. Our architecture is as follows:

![Cloud Architecture](/src/images/Cloud.png)

## Results

The Kaggle competitionuses a Continuous Ranked Probability Score. There, competitors must predict a cimulative probability of yards achieved, strarting from -99 to +99 yards.

However, we choose not to restrict ourselves. We choose to take the yards gained by the rusher, and use that as a numeric value that we want to predict. In this case, the data is available in the dataset given by Kaggle.

We then use MSE as our loss function to evaluate our results. Ultimately, our RMSE which is a proxy for standard deviation was 3.5 yards when our mean was about 70. That comes out to about a 5% SD on our predicted values.

We saw little to no overfitting, as we can see in our graph of predicted vs true data. The general pattern is captured, which the heavy deviations are not predicted. For train v val loss, it is a fantastic sign that both are decreasing steadily.

![Cloud Architecture](/src/images/results.png)

One caveat here is that due to the memory overflow issue, we split entire data into two parts and ran training in two steps. In other words, we generated weights from the first batch of the data and created new session. In a new session, we loaded weight from the previous session and continued training with second batch of the data. However we have witnessed that continued training with second batch of the data didn't actually improve the training and validation loss as shown below.

<img src="/src/images/2nd_batch_loss.png" width="500">


Since we couldn't improve the model performance based on the second batch of the data, we just used first batch based model weight and applied that to the new validation data set in the second batch of the data. As you see below, the prediction result for the test data based on the second data set follows the similar trend as we saw in the first batch prediction result above.

<img src="/src/images/2nd_batch_prediction.png" width="500">

## White Paper

## FAQ

- How did you guys do on the Kaggle leaderboard?

> We actually didn't submit our model to Kaggle, for in order to compete in the Kaggle competition, it is necessary to use their builtin kaggle module.  For the purposes of this project, we instead chose to tweak the challenge to our own needs.

- How large was your dataset?

> It was 36 gigs.

- What inspired you to choose this project topic?

> In the evergrowing field of AI, ML and DL, algorithms are getting more and more complex.  Often, it is difficult to comprehend what the changes are in the field and what impact is ultimately made.  Especially for individuals that are less technology versed, it can be truly difficult to comprehend the meaningful effect of these fantastic technologies.  Thus, our group decided to choose a project topic that is meaningful and relatable to anyone: sports.  The NFL Big Data Bowl challenge gave us a headstart on the actual framework for our project, but ultimately, we just want to show that these remarkable emerging technologies are relatable to everyone, regardless of their background.


## References

- https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview
- https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400
- https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
