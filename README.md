# Quantitative Analysis and Trading Project

## Contributors: Helena Fu, Rob Costa, Sindy Zhang

## Project Overview

## Book References

- For fundamental analysis, look at Murphy's [Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications](https://github.com/ShamaUgale/myBooks/blob/master/John%20J%20Murphy%20-%20Technical%20Analysis%20Of%20The%20Financial%20Markets.pdf).
- For technical analysis, check out Graham and Dodd's [Security Analysis: Principles and Technique](https://glenbradford.com/files/Stocks/security-analysis-benjamin-graham-6th-edition-pdf-february-24-2010-12-08-am-3-0-meg.pdf).
- For bonds and yields, the standard is Leibowitz and Homer's [Inside the Yield Book](https://archive.org/details/insideyieldbookn00home).
- The best book I know on machine learning is Bishop's [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).
- Some good books include "High Probability Trading," "Option Volatility and Pricing," Technical Analysis of the Financial Markets," "Options Trading: The Bible", "Mind over Markets."
- For more historical perspectives, I also enjoy reading the Market Wizards books, When Genius Failed, and Michael Lewis's writings from the last few decades such as Liar's Poker.
- Also for more general tips, "The Money Game," "Mastering the Trade" and "Trading in the Zone," or memoirs like "How I Made $2,000,000 in the Stock Market" from Nicolas Darvas and "Reminiscences of a Stock Operator" about Jesse Livermore.

## Mathematicians who have had trading success
- Leibowitz of Inside the Yield Book had a PhD in math before going into finance.
- Professor Edward Thorp started one of the first hedge funds based on statistical arbitrage. Check out his autobiography "A Man for all Markets," along with "Beat the Dealer" and "Beat the Market."
- Alongside Thorp as one of the first "quants" who applied mathematical and algorithmic principals to Wall Street is Jim Simons, known in math for the Chern-Simons form, and famous publicly for his hedge fund Rennaissance Technology and its Medallion Fund. His life is chronicled in the biography "The Man Who Solved the Market."

## Video References

### Videos about automated analysis and trading in python

- Full Courses
  - Algorithmic Trading Using Python - 3 hour course. [Video here.](https://www.youtube.com/watch?v=9Y3yaoi9rUQ)
  - Algorithmic Trading Using Python - 4 hour course. [Video here.](https://www.youtube.com/watch?v=xfzGZB4HhEE)
  - How To Build A Trading Bot In Python - 9 hour course. [Video here.](https://www.youtube.com/watch?v=jz3tEsCcie0)
- Demos
  - Stock Option Screener in Python - 16 min demo. [Video here.](https://www.youtube.com/watch?v=aeAoVTu66iU)
  - Introduction to Algorithmic Trading Using Python - 17 min demo. [Video here.](https://www.youtube.com/watch?v=fqltiq5EahU)
  - How To Build A Trading Bot In Python - 18 min demo. [Video here.](https://www.youtube.com/watch?v=WcfKaZL4vpA)
  - How to Code a Trading Bot with QuantConnect - 23 min demo. [Video here.](https://www.youtube.com/watch?v=s8uyLscRl-Q)
  - Coding an Options Trading Algorithm with QuantConnect - 26 min demo. [Video here.](https://www.youtube.com/watch?v=Lq-Ri7YU5fU)
  - Probability Distribution of Stock Returns - 35 min demo. [Video here.](https://youtu.be/a0rcZkJP4RQ?si=m9CgyBVBmF9W1DNS)
  - How to Code an AI Trading bot - 35 min demo. [Video here.](https://www.youtube.com/watch?v=c9OjEThuJjY)

### Videos about understanding options

- Pretty much all complex options strategies are made by combining different types of spreads. These all require level 3 options trading. For a basic overview of options, start [here](https://www.youtube.com/watch?v=4HMm6mBvGKE) and [here](https://www.youtube.com/watch?v=MiybniIIvx0). You may also want to start learning about [the Greeks](https://www.youtube.com/watch?v=kCJcEOYuuII).
- Here is a nice video on long vertical spreads, also called debit spreads, which is probably the most similar to buying calls or puts directly. The difference is you somewhat offset theta decay (though also limiting your maximum possible gain) by simultaneously selling an equal number of options. [Here is the video.](https://www.youtube.com/watch?v=1SVswX2V_vE)
- Here is another video, by the same channel, on short vertical spreads, also called credit spreads. This is where you are trying to profit from theta decay and taking on the role of the "option seller." The problem with selling options though, is that your losses can be huge if the options you sold end up in the money, so you can offset those risks by buying an equal number of options that are further out of the money. This limits your total possible losses. [Here is the video.](https://www.youtube.com/watch?v=6_0SbRaHv1U)
- If you want to go more in depth, there's a guy I used to watch who does pretty long detailed videos. [Here's his video on both types of vertical spreads.](https://www.youtube.com/watch?v=mwttDWfDQ9c)
- There is also the wheel, which requires owning the underlying stock. [Here's a video on it.](https://www.youtube.com/watch?v=EcsErh9Airs&t=395s)
- Once you start watching these videos, you'll find a lot of similar channels with different people's explanations and strategies, so I encourage you to explore and discover new channels that work for you!


## Research

This is primarily research I encountered at JMM 2025.

### Articles
- [Predicting stock market index using LSTM](https://www.sciencedirect.com/science/article/pii/S2666827022000378)
- [Predicting NEPSE index price using deep learning models](https://www.sciencedirect.com/science/article/pii/S2666827022000706)
- [Hedging via Perpetual Derivatives: Trinomial Option Pricing and Implied Parameter Surface Analysis](https://arxiv.org/abs/2410.04748)
- [The Distribution Builder: A Tool for Inferring Investor Preferences](https://web.stanford.edu/~wfsharpe/art/qpaper/qpaper.html)
- [Analysis of Investment Returns as Markov Chain Random Walk](https://onlinelibrary.wiley.com/doi/10.1155/2024/3966566)

### Unpublished AMS abstracts
- [Can We See the Next Recession Coming? Deep Learning in Economic Forecasting.](https://meetings.ams.org/math/jmm2025/meetingapp.cgi/Paper/41055)
- [FOREX Prediction Using Deep Learning](https://meetings.ams.org/math/jmm2025/meetingapp.cgi/Paper/43462)
- [Deep Learning Techniques for Equity Portfolio Construction, Optimization, and Performance Analysis](https://meetings.ams.org/math/jmm2025/meetingapp.cgi/Paper/45238)
- [Wavelet Based Reinforcement Learning for Pairs Trading Across Multiple Asset Classes](https://meetings.ams.org/math/jmm2025/meetingapp.cgi/Paper/45119)

## GitHub repositories
- https://github.com/thequantpy
- https://github.com/nickmccullum/algorithmic-trading-python
- https://github.com/Alejandro-Duenas/deep-learning-portfolio
- https://github.com/moondevonyt/moon-dev-ai-agents-for-trading

## Medium articles
- https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95
- https://kaabar-sofien.medium.com/how-to-create-a-sentiment-analysis-model-in-python-cf03cb9988e0
- https://medium.com/@kaveh.kamali/how-i-found-a-simple-way-to-use-machine-learning-in-stock-trading-6c80795cd95f
- https://medium.com/@zodiactrading/top-10-quantitative-trading-strategies-with-python-82b1eff67650
- https://medium.datadriveninvestor.com/creating-a-scalping-strategy-in-python-with-a-74-win-rate-53a17662ff03
- https://medium.com/@harshiljani2002/building-stock-market-engine-from-scratch-in-rust-i-9be7c110e137
- https://databento.medium.com/algorithmic-trading-guide-high-frequency-liquidity-taking-strategy-f1a04172d5d5
- https://levelup.gitconnected.com/backtesting-the-most-underrated-sma-trading-strategy-which-beats-the-market-b7a2f588a502
- https://medium.com/@crisvelasquez/algorithmically-identifying-stock-price-support-resistance-in-python-b9095f9aa279
- https://liamlincoln.medium.com/i-have-just-created-a-trading-ea-monster-c87a523afe7e
- https://medium.com/@amaltyagi/3-secure-trading-strategies-for-2024-2a922c5aee64
- https://medium.com/@austin-starks/i-built-an-algorithmic-trading-system-in-rust-heres-what-i-regret-a89f378b22c9
- https://wire.insiderfinance.io/architecting-a-trading-system-57ee3963e52a
- https://medium.com/@rgaveiga/python-for-options-trading-3-a-trade-with-100-probability-of-profit-886e934addbf
- https://medium.com/rustaceans/rust-in-finance-building-a-scalable-high-frequency-trading-platform-from-scratch-339cdf9d6f08
- https://levelup.gitconnected.com/an-algo-trading-strategy-which-made-8-371-a-python-case-study-58ed12a492dc
- https://medium.datadriveninvestor.com/an-in-depth-guide-on-mathematically-improving-your-trading-strategy-fa8ac2a0e5da
