
# Bollinger Band Opportunity Finder  


<br>


## Setup
Working directory:  
```bash
cd TechAnalysis
```

Virtual environment:  
```bash
python3 -m venv .venv
source .venv/bin/activate 
```

Installing requirements:  
```bash
pip install -r requirements.txt

# If it is using the wrong pip installer, be more specific 
./.venv/bin/python -m pip install -r requirements.txt
```

To run: 
```bash
python3 band_dip_two.py
```


<br>


## Simulation inputs  
You can edit the simulation input variables at the beginning of main() 
- Create a list of ticker symbols and assign it to **tickers** 
- Create a list of conditional exit opportunities and assign it to **exit_tiers**. For each tier,  
  - 'days' is some number of calendar days 
  - 'return' is the percent return (decimal) that, if reached in ___ days, causes the position to be sold 
  - 'name' will be printed if the condition triggers the sale 
- Choose the number of calendar days long a slump must be, and assign to **required_slump_length** 
- Choose the amount of price history to be fetched and graphed, and assign it to **period** 
- Update any printing variables (booleans) to enable or disable certain printing outputs  


