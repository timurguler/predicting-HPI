## How do economic and demographic features influence house price index across the US?
#### *I am actively updating this repository, so look out for further developments.*

My goal here is to use economic and demographic data from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) to predict House Price Index, taking advantage of classic and modern methods in time series analysis. I want to get great predictions, but given my background in economics it's equally (if not more) important for me to understand the relationships.

### Here's my overall plan:
1. Pull data on House Price Index and economic/demographic variables from several major US cities (also referred to as Metropolitain Statistical Areas or MSAs)
2. Prep the data
3. Assess baseline univariate performance (using past HPI to predict future HPI) using classic time series methods like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
4. Quantify the uncertainty of these projections
5. Do the same thing with more modern methods like Facebook's ``prophet``.
7. Repeat steps 3-5, incorporating the economic and demographic features
8. Quantify the improvement, and (hopefully) figure out what actually drives housing prices.

### And here's what I've done so far:
1. **Acquired the data** - this was easily the hardest step since FRED's API is not the most streamlined. I won't bore you with the details here but if you look through ``1-get-msa-and-county-series-list.py``, ``2-get-data.py`` and ``fred_msa.py`` in the ``code`` folder you can see how involved this really was.
2. **Cleaned up the data** - this was also a bit of a challenge when considering the different reporting levels of the different time series - some report annually, other report monthly, still others quarterly. Additionally, some features report at the county level, and these need to be aggregated to the MSA level in a variety of different ways. Check out ``3-data-prep.py`` in the ``code`` folder for the details of this process.
3. **Univariate modeling with classical time series models**:
    * Check out the ``Univariate Modeling.ipynb`` notebook for details
    * Looked at stationarity and partial and full autocorrelation to determine which model classes and hyperparameters might be successful, ultimately testing ARIMA models in the ``statsmodels`` package with <img src="https://render.githubusercontent.com/render/math?math=p \in [1,4], d \in [0,2], q \in [0,1]">
    
    ![image](https://user-images.githubusercontent.com/90712577/156183928-173f0a5b-d54e-4199-a925-2d512dc24e32.png)

    * Evaluated results across various cities using several metrics, including Akike's Information Criterior, percent error (derived from MSE), percent improvement over the naive assumption that <img src="https://render.githubusercontent.com/render/math?math=x_{t+k} = x_t \forall k">, and "win rate" versus the naive assumption, testing predictions at one and four quarters in the future
    ![image](https://user-images.githubusercontent.com/90712577/158294575-bada96df-7caf-4e24-98fe-dd92069fda32.png)

    * Selected the best hyperparameters based on the above metrics, and examined the results for each city:

      ![image](https://user-images.githubusercontent.com/90712577/158294777-c0fecc1e-2233-4e56-98de-42afed616d95.png)


### Analysis of Results:

In general, it is safe to say that using a univariate model is better than guessing for almost all cities. Looking at the distribution of error and improvement, we can see the following:

* **ARIMA predictions are typically within 2% of the real value one quarter out, and within 1-5% one year out**
* **ARIMA predictions typically beat the naive assumption by 25% one quarter out and almost 50% one year out**

![image](https://user-images.githubusercontent.com/90712577/158295012-a053c7d1-406d-4f8e-a3b8-dca84d9fc1d0.png)

![image](https://user-images.githubusercontent.com/90712577/158295083-488bb21f-62b8-4971-82e2-f64696382f72.png)

**Looking more in depth at different cities, we see that:**

* In some cities, like Ogden and Cincinnati, HPI shows clear trends and seasonality, and ARIMA models do a great job of accurately predicting HPI far in advance. 
  ![image](https://user-images.githubusercontent.com/90712577/158295264-0984acc4-399d-4716-9a90-dc7942edaf05.png)

* In other cities, like Los Angeles, short-term prediction is successful, but it is hard to attribute accuracy with underlying model rigor when the trend is essentially a straight line.
  ![image](https://user-images.githubusercontent.com/90712577/158295312-27ba7269-4982-4ec9-bcbe-f55d6d854198.png)

* In others still, like Trenton, HPI seems to deviate from trend in a sharp and erratic pattern, and the "model" roughly tracks with the naive assumption.
  ![image](https://user-images.githubusercontent.com/90712577/158295395-f3f9e1ad-3806-449a-b688-831b8a9988c5.png)

#### Other takeaways:

* The "best" cities for predictability depend on what metric you're using and how far out you want to predict. There can be a trade-off between short term and long term predictive power, with more complex (i.e. higher order of autoregression) models tending to better in the long term and simpler models tending to do better in the short term (e.g. order-1 AR in Los Angeles).
* There is a clear negative relationship between predicted uncertainty and the proportion of times the actual value is within the confidence interval. Simply put, good models tend to be overconfident. It also appears that models tend to overestimate certainty in the short term, and often get it wrong, tending to be more conservative farther away from the current date. Confidence intervals are either too good to be true or too wide to be helpful.
![image](https://user-images.githubusercontent.com/90712577/158296023-2db4b7cd-9ced-4ff5-b27e-c99338660e98.png)



