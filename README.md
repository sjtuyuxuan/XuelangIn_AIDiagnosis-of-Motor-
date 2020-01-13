# Xuelang_AI—Diagnosis-of-Motor
Xuelang Industrial Data Challenge
## Question background
#### Using big data and artificial intelligence technology to automate motor abnormal sound detection, solve the pain points that artificial detection cannot accurately and reliably identify abnormal sounds, and help upgrade lean manufacturing      and intelligent manufacturing
      
## Data exploration
### Training set:
#### There were 30 positive samples and 500 negative samples. The samples were severely uneven.
### Test set:
#### Sample 5738 sets
### Data length:
#### Single sample 4 columns of audio with a length of 79999 and a sampling rate of 51200.The duration is 1.5625s.

After visualization, it can be seen from the figure that the number of glitches in the positive sample is larger, and the glitches are larger.In the time-frequency plot of the data of the positive and negative samples, the dominant frequencies   are quite different.
     
In the original data (530 groups), there was a serious imbalance between positive and negative samples. To solve this problem, we enhanced the positive samples. We use Gaussian white noise for enhancement, on the one hand, it reduces the impact on the original spectrum and at the same time, it reduces the impact on statistical characteristics. Then we used the lightgbm model and the stft-cnn model to analyze the statistical and spectral characteristics, respectively.
     
    𝑓_𝑎𝑢𝑔 (𝑥)=𝜆∗max⁡( 𝑓(𝑥))∗𝑟𝑎𝑛𝑑𝑜𝑚()
    where:
     
      𝜆Proportion for data enhancement
      𝑓 (𝑥) is the feature column that needs to be enhanced, such as ai1, ai2
      max⁡ (𝑓 (𝑥)) The maximum value of the feature column
      𝑟𝑎𝑛𝑑𝑜𝑚 () is a random number generation function, a Gaussian distribution with a random number of 0-1
    
## Feature extraction
### Sliding window construction features:
#### Selection of sliding window size: The sound itself is not a stable random sequence. Set the length of the window to extract long and short audio information.
#### Sliding window 1: Window 2000, steps 1000 
#### Sliding window 2: Window 3000, steps 1000 
#### Sliding window 3: Window 10240, steps 10240 
#### Sliding window 4: Window 12800, steps 12800 
      
### Statistical characteristics: Maximum, minimum, range, mean, variance, skewness, kurtosis
### Glitch features: ai1 and ai2 glitch number, ratio, and period
### Frequency domain characteristics: STFT transform, using CNN to extract spectrogram features
### Timing characteristics: autocorrelation, binned entropy
### Tree model construction features: Use the values of the leaf nodes output by the tree model as features

## Model
### ResNet50
#### Idea: structure data volume overfitting
#### Do short-time Fourier transform to extract stft features
#### Splicing feature channels and inputting the resnet network structure model
    
### lightgbm model
#### Idea: Further tuning
#### No adjustment of parameters
#### 5-fold cross-validation output (fast and stable)
    
    
### Model advantages
#### Model advantages and program potential:
Feature-rich, including time domain and frequency domain features
The model runs fast, and 5 minutes from the feature extraction to the model run is enough
The results are stable, there is not much difference between the preliminary and live competition scores, the model has strong generalization performance and strong practicability
The model is simple and does not need to adjust parameters
         
         
## Additional promising        
#### Try more different data enhancement methods to increase the amount of data
#### To further improve the CNN model, in addition to using the cosine annealing method to train the model, you can try to use Zhejiang University's AdaBound optimization method in 2019
#### Further try to use the most recent octave convolution to speed up the calculation
#### Tuning the tree model for better results
#### try more methods
 
 
 ## Final result
 ### Recall rate 100% 
 ### Accuracy 89.11% 
 
