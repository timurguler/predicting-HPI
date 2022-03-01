## How do economic and demographic features influence house price index across the US?
#### *I am actively updating this repository, so look out for further developments.*

My goal here is to use economic and demographic data from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) to predict House Price Index, taking advantage of classic and modern methods in time series analysis. I want to get great predictions, but given my background in economics it's equally (if not more) important for me to understand the relationships.

### Here's my overall plan:
1. Pull data on House Price Index and economic/demographic variables from several major US cities (also referred to as Metropolitain Statistical Areas or MSAs)
2. Prep the data
3. Assess baseline univariate performance (using past HPI to predict future HPI) using classic time series methods like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
4. Do the same thing with more modern methods like Facebook's ``prophet``.
5. Quantify the uncertainty of these projections
6. Repeat steps 3-5, incorporating the economic and demographic features
7. Quantify the improvement, and (hopefully) figure out what actually drives housing prices.

### And here's what I've done so far:
1. **Acquired the data** - this was easily the hardest step since FRED's API is not the most streamlined. I won't bore you with the details here but if you look through ``1-get-msa-and-county-series-list.py``, ``2-get-data.py`` and ``fred_msa.py`` in the ``code`` folder you can see how involved this really was.
2. **Cleaned up the data** - this was also a bit of a challenge when considering the different reporting levels of the different time series - some report annually, other report monthly, still others quarterly. Additionally, some features report at the county level, and these need to be aggregated to the MSA level in a variety of different ways. Check out ``3-data-prep.py`` in the ``code`` folder for the details of this process.
3. **Univariate modeling with classical time series models**:
    * Check out the ``Univariate Modeling.ipynb`` notebook for details
    * Looked at stationarity and partial and full autocorrelation to determine which model classes and hyperparameters might be successful, ultimately testing ARIMA models in the ``statsmodels`` package with <img src="https://render.githubusercontent.com/render/math?math=p \in [1,4], d \in [0,2], q \in [0,1]">
    
    ![image](https://user-images.githubusercontent.com/90712577/156183928-173f0a5b-d54e-4199-a925-2d512dc24e32.png)

    * Evaluated results across various cities using several metrics, including Akike's Information Criterior, MSE, and MSE improvement over the naive assumption that <img src="https://render.githubusercontent.com/render/math?math=x_{t+k} = x_t \forall k">
    
    ![image](https://user-images.githubusercontent.com/90712577/156186316-5389e170-4add-4dd4-909b-ed7067998237.png)

    * Examined the results for each city:

![image](https://user-images.githubusercontent.com/90712577/156185895-99a8011b-3305-4136-ac29-ec56ecdd6250.png)

**My results showed that standard statistical methods provide some predictive power even in the univariate case, and I hope that expanding on these results with more complex models and additional features will provide further improvement.**
