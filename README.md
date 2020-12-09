# Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data
## Overview

Most flight-related fatalities stem from a loss of “airplane state awareness.” That is, ineffective attention management on the part of pilots who may be distracted, sleepy or in other dangerous cognitive states.Our challenge is to build a model to detect troubling events from aircrew’s physiological data.

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/00.webp)
### Some plots..
![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/4.png)

## Data Description
In this dataset, you are provided with real physiological data from eighteen pilots who were subjected to various distracting events. The benchmark training set is comprised of a set of controlled experiments collected in a non-flight environment, outside of a flight simulator. The test set (abbreviated LOFT = Line Oriented Flight Training) consists of a full flight (take off, flight, and landing) in a flight simulator.

The pilots experienced distractions intended to induce one of the following three cognitive states:

**Channelized Attention (CA)** is, roughly speaking, the state of being focused on one task to the exclusion of all others. This is induced in benchmarking by having the subjects play an engaging puzzle-based video game.

**Diverted Attention (DA)** is the state of having one’s attention diverted by actions or thought processes associated with a decision. This is induced by having the subjects perform a display monitoring task. Periodically, a math problem showed up which had to be solved before returning to the monitoring task.

**Startle/Surprise (SS)** is induced by having the subjects watch movie clips with jump scares.

For each experiment, a pair of pilots (each with its own crew id) was recorded over time and subjected to the CA, DA, or SS cognitive states. The training set contains three experiments (one for each state) in which the pilots experienced just one of the states. For example, in the experiment = CA, the pilots were either in a baseline state (no event) or the CA state. The test set contains a full flight simulation during which the pilots could experience any of the states (but never more than one at a time). The goal of this competition is to predict the probability of each state for each time in the test set.

Each sensor operated at a sample rate of 256 Hz. Please note that since this is physiological data from real people, there will be noise and artifacts in the data.

Data fields
Variables with the eeg prefix are electroencephalogram recordings.

id - (test.csv and sample_submission.csv only) A unique identifier for a crew + time combination. You must predict probabilities for each id.

crew - a unique id for a pair of pilots. There are 9 crews in the data.

experiment - One of CA, DA, SS or LOFT. The first 3 comprise the training set. The latter the test set.

time - seconds into the experiment

seat - is the pilot in the left (0) or right (1) seat

eeg_fp1 eeg_f7 eeg_f8 eeg_t4 eeg_t6 eeg_t5 eeg_t3 eeg_fp2 eeg_o1 eeg_p3 eeg_pz eeg_f3 eeg_fz eeg_f4 eeg_c4 eeg_p4 eeg_poz eeg_c3 eeg_cz eeg_o2

ecg - 3-point Electrocardiogram signal. The sensor had a resolution/bit of .012215 µV and a range of -100mV to +100mV. The data are provided in microvolts.

r - Respiration, a measure of the rise and fall of the chest. The sensor had a resolution/bit of .2384186 µV and a range of -2.0V to +2.0V. The data are provided in microvolts.

gsr - Galvanic Skin Response, a measure of electrodermal activity. The sensor had a resolution/bit of .2384186 µV and a range of -2.0V to +2.0V. The data are provided in microvolts.

event - The state of the pilot at the given time: one of A = baseline, B = SS, C = CA, D = DA

## EDA 

### Ploting the overall state of the pilots throughout the experiment

Here we plot the overall durations of Channelized Attention (CA),Diverted Attention (DA), Startle/Surprise (SS) and Baseline(A)
**The state of the pilot at the given time: one of A = baseline, B = SS, C = CA, D = DA**
![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/1.png)

As expected the pilot is aware most of the time. We can see that the concentration affects the most when doing CA(Channelized Attention) task.
DA(Diverted attention) and B(Startle/Surprise) have fairly minimum effect.

#### Let us now see how the indivvidual experiment have effect on the concentration
![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/2.png)

As expected , when doing CA(Channelized Attention) task the pilots concentration is fully diverted
In the other two tasks Pilots failry maintain the concentration on flying the plane.
Further on the pilots seat position has no signifiant impact on the result.

### Plotting all the EEG ECG R and GSR vaules

### EEG EXPLANATION
Now this is the interesting bit to me. EEG's role has been greatly overstated over the years, and it's definitely not a panacea of brain activity. Clinically, you can usefully tell if someone is awake, asleep, brain dead, having a seizure, and a handful of other things. EEG is a summation of all the electrical activity on the surface of the brain. This activity has to travel through layers of soft tissue, bone and skin, so it's no wonder that the data is quite noisy.

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/-1.jpg)

### Preparing EEG data
This data is prepared in a fairly typical arrangement of 20 electrodes across the scalp. The letter in each lead signifies the part of the brain that that lead is nearest to (Temporal, Frontal, Parietal etc), with odd numbers on the left, evens on the right. Usually in the clinic, we don't look at the electrical potentials at each electrode, but at the potential difference between pairs of electrodes. This gives us an idea of the electrical field in the brain region between these two points as a way to infer what the brain is doing in that region. Clearly you can choose any two electrodes and produce 20! different potential differences, but not all of those are going to be useful.
We talk about the layout of choosing the pairs of electrodes to compare potential differences as Montages. There's lots of different montage systems, but commonly there's the 10-20 system. This data has an additional 'poz' electrode to the diagram, but that doesn't cause us a problem.
![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/9.jpg)
Let us see if there are any deviations from the test and the training data

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/4.png)

The test test had a bit of a variance compared to the training set. The training set seems more concentrated and neetly dstributed but over all there are no major differences and we can develop a good model and expect high prediction rate

The ECG data shows a linear line making the pilots seen dead :P
Lets plot it and see how it varies

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/8.png)

They are similar Except foir the >20000-ish samples. I might say that they took less samples in the actual test. Rest looks fine.
### Challenge Duration and intensity through Violin plot

Seeing the effect of different experiments related to each other with respect to time

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/3.png)

''' 
The violin plot shows that all three experiments are equally spread through time except the B(Surprise) where we can make out that there are roughly two distinct surprises per experiment

## Gradient Boosting

Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference between AdaBoost and Gradient Boosting Algorithm is how the two algorithms identify the shortcomings of weak learners (eg. decision trees). While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function (y=ax+b+e , e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data. A logical understanding of loss function would depend on what we are trying to optimise. For example, if we are trying to predict the sales prices by using a regression, then the loss function would be based off the error between true and predicted house prices. Similarly, if our goal is to classify credit defaults, then the loss function would be a measure of how good our predictive model is at classifying bad loans. One of the biggest motivations of using gradient boosting is that it allows one to optimise a user specified cost function, instead of a loss function that usually offers less control and does not essentially correspond with real world applications.
params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
I did not change any parameters as we did get good results           
### Normalizing 

For comparision purposes we normalize the data which does help in getting a good insight

## Training the modle

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/6.png)

The loss had redused to .03213 which is better than the refrence model which I took which was .03830 making it 16 % better.
It is mainly because that was 2 years old and now the new documentation of gradient boost may have changed something. I will look into it.


## Feature Importance of LightBGM

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/5.png)

## Confusion Matrix

![](https://github.com/shubhamk8597/Project---Predicting-Cognetive-States-of-Pilots-By-Analysing-Physiological-Data/blob/main/Images/7.png)

## Conclusion
As seen in the confusion matrix our model is well and predicts the data almost with accuracy with accuracy as follows

A = **95%**

B = **97%**

C = **99%**

D = **94%**

Average Accuracy = **96.25 %**

The most important accuracy indicator is C with 99% which is very good as It was the most challenging task which took all the attention of the pilot. So we can alarm the pilots at a failry acurate amount when they are not cognitively aware. But we should not forget it is not practicle to have a measuring instrument as heavy as eeg to carry around. With deveopment of wareable gagets ike smart watches which can now even measure our oxygen level we can have devices that can measure eeg, ecg without discomforting the pilots and then we can make them aware when they are distracted with 99% accuracy.



