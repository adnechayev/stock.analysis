<h1>Tesla Stock Analysis</h1>

<h2>Description</h2>
This project analyzes the trend of the closing price of Tesla stock. After performing some initial data analysis, we utilize Linear Regression and a Recurrent Neural Network side by side to run predictions on future stock price.
<br />


<h2>Languages Used</h2>

- <b>Python</b> 

<h2>Environments Used </h2>

- <b>Visual Studio Code</b>

<h2>Project walk-through:</h2>

<p align="center">
To start the project off, we do some data exploration by looking at the first 5 rows of the dataset and also seeing if there are any existing null values:
<br/>
<br/>
<img src="https://i.imgur.com/oOcvFM2.png" height="50%" width="50%" alt="albums"/> <img src="https://i.imgur.com/9V5oALC.png" height="20%" width="20%" alt="albums"/>
<br/>
For the sake of running our future models, we only care about two columns: the adjusted closing price and the date. Using these two columns we can eventually construct a trend line that we can make predictions on. It's also good to note that we don't have any null values to clean up!
<br/>
<br/>
To get a better idea of the current stock price trend, we isolate some rows to encompass only the year of 2023 and find the percentage change in the stock.
<br/>
<br/>
<img src="https://i.imgur.com/wS23wz2.png" height="50%" width="60%" alt="albums"/>
<br/>
Upon first glance, we can see there to be positive linear trend in the stock prices. After performing a quick percentage change calculation, we can confirm this for ourselves:
<br/>
<br/>
<img src="https://i.imgur.com/DwDm1bs.png" height="40%" width="50%" alt="albums"/>
<br/>
<br/>
It's no easy task trying to accurately predict the movement of the market, but perhaps with the aresenal of machine learning and statistical methods on our side we can attempt to formulate some idea of the kind of prices we expect to see. To further this endeavor we'll be employing a technical indicator, namely the exponential moving average (EMA). Technical indicators are mathematical calculations commonly used by traders to predict future stock price based on factors such as historical price, volume, etc. The one we'll be using, EMA-12, places a greater emphasis and weight on more recent data points.
<br/>
<br/>
Let's take a look at the result of the EMA averages being added alongside our current stock price trend:
<br/>
<br/>
<img src="https://i.imgur.com/CwFULFZ.png" height="40%" width="50%" alt="albums"/>
<br/>
Our EMA technical indicator seems to track along our data points quite nicely, this is a good sign.
<br/>
<br/>
With our data prepared and an appropriate technical indicator attached, we can finally split our data into test and train sets and run our first model: good 'ol linear regression. We'll be looking at the relationship between our EMA values and our actual values to see if the EMA indicator is a good fit. Plotting out the line of best fit from our data points, we get this graph:
<br/>
<br/>
<img src="https://i.imgur.com/lTFoORN.png" height="40%" width="50%" alt="albums"/>
<br/>
Good news! We see a very clear positive trend here and our model coefficients, mean absolute error, and coeffecient of determination (R<sup>2</sup>) can confirm this:
<br/>
<br/>
<img src="https://i.imgur.com/ZMpGz3u.png" height="80%" width="80%" alt="albums"/>
<br/>
Having a high <b>model coefficient</b> tells us that we have a strong positive relationship between our actual values and EMA values. Our mean absolute error, or MAE, indicates the averaged absolute differences between our different variables. With data points on a magnitude scale in the 100s, a MAE score of around 3 tells us that there is very little error in our data points. Lastly, the coefficient of determination, or the R<sup>2</sup> score, shows how much percentage change in our dependent variable is explained by our independent variable. Once again, a very high value of 99% indicates a good fit of our EMA points!
<br/>
<br/>
We'll now be looking at a more involved method of for predicting future data points; namely a recurrent neural network (RNN). The reason we use an RNN is because of its unique disposition to retain information from previous inputs which is helpful for time series forcasting, or in our case, stock prediction.
<br/>
<br/>
After some quick data reshaping, we will begin to build our different layers:
<br/>
<br/>
<img src="https://i.imgur.com/jpz497u.png" height="40%" width="80%" alt="albums"/>
<br/>
We opt to use a specific type of RNN called a long short-term memory network, or LSTM. In short, LSTMs operate using a series of cell states and gates to control the flow and memory of the network. We choose LSTMs in particular because of their strong tendency to adapt to long-term dependencies. For each layer except for the dense (output) layer, we use 50 neurons and add an additional dropout layer to mitigate overfitting.
<br/>
<br/>
With our layers built, we apply the 'adam' optimizer and run the network over 50 epochs. To finalize, we graph out the prediction for 30 days past the last recorded price change and this is what we get:
<br/>
<br/>
<img src="https://i.imgur.com/8t5Gao9.png" height="40%" width="80%" alt="albums"/>
<br/>
<br/>
The last entry in our dataset was in mid June. Taking a look at the current price trend in TESLA stock, we can see that our model made a fairly accurate prediction:
<br/>
<br/>
<img src="https://i.imgur.com/yqFjvrz.png" height="40%" width="80%" alt="albums"/>


