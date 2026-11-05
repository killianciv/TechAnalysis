# Bollinger Band Opportunity Finder  
A collection of Python scripts using Yahoo Finance and MatPlotLib
All of these test the 30-day bollinger band theory 

<br>





## Setup
Working directory:  
```bash
cd TechAnalysis
```

Virtual environment:  
```bash
# Create a virtual environment named 'venv' 
python3 -m venv venv

# Activate it 
source venv/bin/activate 
```

Installing requirements:  
```bash
# Use the pip installer that lives directly inside the venv/ folder 
./venv/bin/python -m pip install -r requirements.txt

# If you believe your computer knows to use the venv/ installer, just run 
pip install -r requirements.txt
```

To run: 
```bash
python3 filename.py
```

To stop: 
```bash
control + c
```


<br>





# Running it
Each file describes its own exact inputs, but there are general rules: 
- You edit the simulation input variables at the beginning of main() 
- Create a list of ticker symbols and assign it to **tickers** 
- Create a list of conditional exit opportunities and assign it to **exit_tiers**. For each tier,  
  - 'days' is some number of calendar days 
  - 'return' is the percent return (decimal) that, if reached in ___ days, causes the position to be sold 
  - 'name' will be printed if the condition triggers the sale 
- Choose the number of calendar days long a slump must be, and assign to **required_slump_length** 
- Choose the amount of price history to be fetched and graphed, and assign it to **period** 
- Update any printing variables (booleans) to enable or disable certain printing outputs  



# Details that apply to all scripts 
- An "opportunity" is a record of a historical market signal. A buying opportunity 
  opens immediately after price has closed completely below the SMA for the 30th 
  consecutive CALENDAR day. If price touches the SMA, the counter is reset.   
- A "future" or "brewing" opportunity is recorded when the stock was still in a 
  slump the final day of the simulation, and the slump was fewer than 30 days 
  long at that time, so no real opportunity existed yet. Some scripts record these 
  so the user is aware of what stocks may soon provide an opportunity for them.  
- The simulation uses closing prices for its recommendations. If it recommends 
  a trade on a certain day, it's because that day's closing price met some 
  condition. Because a recommendation relies on days-end data, all recommended 
  trades should be executed right before market close (~3:59 pm EST) on that day.  
- The first day price closes below the SMA is "day 1" of the slump, even if the 
  stock was above the SMA for the majority of the trading day.  
- The 30th day of a slump is the first confirmed buying opportunity. If every day 
  was a trading day and the slump began on May 1st @3:59pm, the first opportunity
  to purchase is before the close (3:59 pm EST) on May 30th (29 days after day 1).  
- The downloaded data only has info for trading days, so if the 30th calendar 
  day is a Saturday or Sunday, the opportunity will be on the Monday.  
- There is only one purchase per slump. We purchase stock on the first available 
  day because waiting even one extra day could miss the opportunity.  





# Code notes
- SMA20 = 20-day simple moving average of the prior 20 trading days' closing prices  
- STD20 = sigma = standard deviation of the prior 20 trading days' closing prices. It
    is the standard deviation of price, not the standard deviation of returns.  
- HW = half-width = 2 * STD20 = the distance between the SMA and a Bollinger Band   
- CW = channel width = 2 * HW = the distance between Bollinger Bands   
- The dataframe stores the SMA as "SMA20" to be precise, but I write "SMA" everywhere else   
- All return/roi variables (unless suffixed with _as_percent) are decimals. 80% = 0.8   
- Price "hits" a structure (SMA or a Bollinger Band) when price gets close to 
    it (or exactly touches it). The equations use <= or >=  
- Price "crosses" a structure if price completely moves from one side to the 
    other. The equations use > or <  




