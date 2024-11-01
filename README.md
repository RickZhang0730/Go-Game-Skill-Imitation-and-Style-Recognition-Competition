https://1drv.ms/p/c/a3d918eaa1794158/EbAnHlgjT_VHvw2UeygtHy4Bor-aha7UpsbxEAyllBCc-w?e=kony0e
This project endeavors to replicate and understand human behavior in Go through comprehensive training and prediction models on a 19x19 Go board. Two levels of playing strength, represented by Dan and Kyu players, are simulated, with predictions evaluated for accuracy against actual moves. Further analysis categorizes playing styles into aggressive, balanced, and territorial in specific game situations.

Convolutional Neural Network (DCNN Model) is chosen for its capability to capture intricate board features, accommodating both local and global considerations, thereby enhancing prediction accuracy in the complex game of Go. Model training specifics are outlined for both Dan and Kyu players and the recognition of playing styles, utilizing various optimization techniques.

In summary, this project contributes to the comprehensive understanding of Go by leveraging advanced computational techniques. By unraveling the intricacies of move prediction and playing style recognition, the project not only delves into the essence of this ancient game but also opens avenues for future applications in artificial intelligence and game theory.

# Introduction
Go game have been invented thousands of years ago, and it is originated in China. While the exact date of its invention is uncertain, it is confirmed that Go was widely popular as early as the 10th century BC. Currently, Go is prevalent in many East Asian countries and has gradually spread worldwide. In fact, in the hearts of many, Go is not merely a form of entertainment, due to its inherent qualities, it has long been regarded as an art form.

The allure of Go stems not only from its simple rules and intricate variations, providing ample space for creative imagination, but also from its millennia of study. Numerous tactical concepts and thinking methods have been developed and explored over the years, allowing individuals to swiftly immerse themselves in the world of Go through learning these principles.

Go game integrates various intelligent behaviors and thoughts of humans on the go board. This project aims to mimic and recognize human behavior through Go game records. The goal is to conduct training and predictions on a 19x19 Go board.

It simulates two levels of Go playing strength, namely Dan (higher-level) players with strong skills and Kyu (lower-level) players with weak skills, predictions are made for the next move on the current Go board, with accuracy indicating whether the predicted move matches the actual move on the given board.

After obtaining the moves, further analysis is conducted to identify the playing style in specific game situations, categorized as aggressive (enjoys challenges and fears no battle), balanced (takes a comprehensive view of the overall situation), and territorial (practical and steady). 

# Related works
We have talked about ‘Batch Normalization Layers’, it can help Accelerate model convergence and prevent gradient vanishing. In this project, I have added batch normalization after each convolutional layer.

And about ‘Data Augmentation’ in class, I have added activation function ‘ReLU’, which function is be like max(0, v), generating more samples to enhance model generalization.

# File Structure
  - **CSVs**: Dataset folder for training and testing
  - **Dan Training Tutorial.ipynb**: Baseline model training of Dan dataset
  - **Kyu Training Tutorial.ipynb**: Baseline model training of Kyu dataset
  - **PlayStyle Training Tutorial.ipynb**: Baseline model training of PlayStyle dataset
  - **Create Public Upload CSV.ipynb**: Predictions by using the baseline models
  - **model_dan_tutorial.h5**: Baseline model of Dan dataset
  - **model_kyu_tutorial.h5**: Baseline model of Kyu dataset
  - **model_playstyle_tutorial.h5**: Baseline model of PlayStyle dataset
  - **public_submission_template**: Predictions for public testing datasets

# Method
## Data Processing:
**Firstly, feature extraction is performed for predicting the next move of a player in the game of Go. Six features are utilized and transformed into one-hot encoded.**
- The 1st feature designates positions with black stones as 1 and others as 0.
- The 2nd feature designates positions with white stones as 1 and others as 0.
- The 3rd feature marks positions with stones as 1 and empty positions as 0.
- The 4th feature indicates the position of the last move as 1 and others as 0.
- The 5th feature marks positions where a black stone may be surrounded in the next move as 1 and others as 0.
- The 6th feature marks positions where a white stone may be surrounded in the next move as 1 and others as 0.
<div align = left><img width="800" height="120" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E7%89%B9%E5%BE%B51.JPG"/></div>
<div align = left><img width="400" height="120" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E7%89%B9%E5%BE%B52.JPG"/></div>

**For predicting player's playing style, thirteen features are employed and transformed into one-hot encoded. The first three features are similar to the move prediction case.**
- The 4th feature marks positions where a black stone may be surrounded in the next move as 1 and others as 0.
- The 5th feature does the same for white stones.
- The 6th feature designates positions with a horizontal connection of stones as 1 and others as 0.
- The 7th feature designates positions with a vertical connection of stones as 1 and others as 0.
- The 8th feature designates positions with a stone in the upper-right diagonal as 1 and others as 0.
- The 9th feature designates positions with a stone in the lower-left diagonal as 1 and others as 0.
- The 10th feature identifies positions where the opponent may place a stone in the next move for an offensive strategy, assuming the player is controlling black stones.
- The 11th feature serves a similar purpose but assumes the player is controlling white stones.
- The 12th feature identifies positions where the player may place a stone in the next move for a defensive strategy, assuming the player is controlling black stones.
- The 13th feature serves a similar purpose but assumes the player is controlling white stones.
<div align = left><img width="800" height="120" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E7%89%B9%E5%BE%B53.JPG"/></div>
<div align = left><img width="200" height="120" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E7%89%B9%E5%BE%B54.JPG"/><img width="400" height="120" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E7%89%B9%E5%BE%B55.JPG"/></div>

We choose a Convolutional Neural Network model (DCNN Model) as it can learn intricate board features, aiding in predicting optimal moves. The DCNN can capture both local and global board features, effectively handling the image data of the board and establishing appropriate balances in the model to enhance prediction and training accuracy for the complex game, Go.

## Model Training:
**We have both an old and a new version in model training. The old version is used during competitions, while the new version represents the improvements we've made thereafter.**

**For training the model for Dan (higher-level) players:**
- The model consists of multiple convolutional layers using the Conv2D function with filter sizes of 3x3. Each convolutional layer with BatchNormalization for improved training stability.

- MaxPooling2D is applied for pooling operations, and the Flatten layer is used to convert the output into a one-dimensional tensor. The model includes fully connected layers with ReLU activation functions and Dropout layers to reduce overfitting.

- The output layer uses the Softmax activation function to output a 19x19 tensor representing the probabilities of each position for the next move. The model is compiled using the RMSprop optimizer.

**For the model trained on Kyu (lower-level) players:**
- The model consists of multiple convolutional layers using the Conv2D function with filter sizes of 3x3. Each convolutional layer with BatchNormalization for improved training stability.

- MaxPooling2D is applied for pooling operations, and the Flatten layer is used to convert the output into a one-dimensional tensor. The model includes fully connected layers with ReLU activation functions and Dropout layers to reduce overfitting.

- The output layer uses the Softmax activation function to output a 19x19 tensor representing the probabilities of each position for the next move. The model is compiled using the RMSprop optimizer.

**For training the model to recognize playing styles:**
- The model includes convolutional layers with Dropout regularization, and Dense layers with L2 regularization.

- The output layer uses Softmax activation for a 3-dimensional vector representing the three playing styles (aggressive, balanced, territorial). The model is compiled using the RMSprop optimizer.

### Training models:
**Dan model:**
- old version vs new version
<table style="border: none;">
  <tr>
    <td align="center" style="border: none;">
      <p>old version</p>
      <img width="300" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/Dan%E6%A8%A1%E5%9E%8Bold.JPG" alt="Dan Old Version">
    </td>
    <td align="center" style="border: none;">
      <p>new version</p>
      <img width="300" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/Dan%E6%A8%A1%E5%9E%8B.JPG" alt="Dan New Version">
    </td>
  </tr>
</table>

**Kyu model:**
- old version vs new version
<table style="border: none;">
  <tr>
    <td align="center" style="border: none;">
      <p>old version</p>
      <img width="300" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/Kyu%E6%A8%A1%E5%9E%8Bold.JPG" alt="Kyu Old Version">
    </td>
    <td align="center" style="border: none;">
      <p>new version</p>
      <img width="300" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/Kyu%E6%A8%A1%E5%9E%8B.JPG" alt="Kyu New Version">
    </td>
  </tr>
</table>

**Playing-style model:**
<table style="border: none;">
  <tr>
    <td align="center" style="border: none;">
      <img width="300" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/Playing-style%E6%A8%A1%E5%9E%8B.JPG" alt="Playing-style">
    </td>
  </tr>
</table>

# Experiment
The experiment is divided into two main parts: The Imitation Go Skill Competition and The Recognition of Go Playing Styles. In the Imitation Go Skill Competition, we aim to replicate the playstyles of 1-dan and 10-kyu Go players. For each player level, we provide predictions for the most likely move and predictions for five possible moves, considering a prediction as correct if it includes the actual move. The accuracy is assessed separately for the 1st and 5th predictions, denoted as One_Dan_1, One_Dan_5, Ten_Kyu_1, and Ten_Kyu_5.

In the Recognition of Go Playing Styles, the task involves identifying the playing style from a given game record and outputting the predicted playing style of the final move. We assume that the predicted playing style is accurate if it matches the actual playing style, denoted as PSA.The final composite score is calculated by taking the accuracy of each task to four decimal places, multiplying it by the predefined weight, and summing them up. 

This study aims to comprehensively assess the model's performance by simulating and recognizing various skill levels and playing styles in the game of Go, providing valuable insights for the advancement of artificial intelligence in the field of Go.

# Results
**Public_test & Leaderboard [46/508]:**
<div align = left><img width="420" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E5%90%8D%E6%AC%A11.JPG"></div><img width="420" height="150" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C1.JPG"/></div>

**Private_test & Leaderboard [48/508]:**
<div align = left><img width="420" height="400" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E5%90%8D%E6%AC%A12.JPG"></div><img width="420" height="150" src="https://github.com/RickZhang0730/Go-Game-Skill-Imitation-and-Style-Recognition-Competition/blob/main/Images/%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C2.JPG"/></div>


# Conclusion
The game of Go itself is incredibly complex, making this project quite challenging for us. The comprehensive assessment across various skill levels and styles, the emulation based on analyzing others' playing styles, predicting the next move's position, and identifying different playing styles have deepened our understanding and application of deep learning. 

Although the model trained on the training set exhibits high accuracy, there's substantial room for improvement when tested on formal datasets. Aspects such as more precise feature extraction, multidimensional additions, experimenting with different activation functions, increased regularization or alterations in convolutional layers, alongside variations in training volume and adjustments in learning, aim to prevent overfitting.

# Reference
[1] Yuandong Tian and Yan Zhu, ”Better Computer Go Player with Neural Network and Long-term Prediction”, ICLR, 2016.

[2] Chang-Shing Lee, Mei-Hui Wang, Shi-Jim Yen, Ting-Han Wei, I-Chen Wu, Ping-Chiang Chou, Chun-Hsun Chou, Ming-Wan Wang, and Tai-Hsiung Yang, "Human vs. Computer Go: Review and Prospect", IEEE Computational Intelligence Magazine (IEEE CIM), Vol. 11, No. 3, pp. :67- 72, August 2016.

[3] David Silver, Aja Huang, Chris J. Maddison, "Mastering the game of Go with deep neural networks and tree search"Nature, vol. 529, no. 7587, pp. 484-489, 2016.

[4] David Silver, Thomas Hubert, Julian Schrittwieser,"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",Science, vol. 362, no. 6419, pp. 1140-1144, 2018.

[5] Christopher Clark, Amos Storkey,"Training deep convolutional neural networks to play Go",arXiv preprint arXiv:1412.3409, 2014.

[6] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton,"ImageNet Classification with Deep Convolutional Neural Networks",Advances in Neural Information Processing Systems (NIPS), 2012.

