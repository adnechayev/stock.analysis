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


