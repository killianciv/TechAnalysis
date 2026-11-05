"""
Using Bollinger Bands to identify entry and exit opportunities  

**** DETAILS
A buying opportunity opens immediately after price has closed below the SMA for
  the 30th consecutive CALENDAR day. The downloaded data only has info for 
  trading days, so if the 30th calendar day is a Saturday or Sunday, the 
  simulation will recommend that you execute the trade on Monday. 
The first day price closes below the SMA is "day 1" of the slump, even if the 
  stock was above the SMA for the majority of the trading day. 
The simulation uses closing prices for its reommendations. If it recommends 
  a trade on a certain day, it's because that day's closing price met some 
  condition. Because a recommendation relies on days-end data, all recommended 
  trades should be executed right before market close (~3:59 pm EST) on that day. 
The 30th day of a slump is the first confirmed buying opportunity. If every day 
  was a trading day and the slump began on May 1st @3:59pm, the first opportunity
  to purchase is before the close (3:59 pm EST) on May 30th (29 days after day 1). 
We purchase stock on the first available day because waiting even one extra day 
  could result in loss of opportunity. There is only one purchase per slump. 
If during a slump the price rises to the SMA, the opportunity is cancelled. 

**** EXIT OPPORTUNITIES
These depend on your own criteria. 
You can have no exit strategy and just hold onto your positions. 
You can use a total return target, and you sell a position after it has been hit. 
You can use a tiered exit strategy, exiting a position when at least one of the
  tier's exit criteria has been met. Three-tier example: 
  - 30-day return exceeds 5%
  - 60-day return exceeds 10% 
  - Total return exceeds 15%

**** CODE NOTES
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
- Because "hit" margins are inclusive ("or equal to"), if the price is exactly on 
    the SMA or a Bollinger Band, both its hit variables (top & bottom) will be True. 
    
**** QUESTIONS
Band touches are good, but what do they count for? 
- Does each band touch reduce the minimum slump period by one day? 
- Does each band touch increase the amount of stock you will purchase if the indicator is confirmed? 
Should I only purchase once per slump? 
- Will I purchase more on days 35/40/50? If so, future_opportuntity say "yes" 
  if it's slump day 34 
Should "price hits upper band" be an exit tier (because of statistics)? 
"""



import yfinance as yf  # Yahoo finance 
import pandas as pd
from datetime import datetime, timedelta  # Formal dates and sorting
import matplotlib.pyplot as plt  # Graphing
from collections import deque  # Efficient FIFO, used for selling oldest shares 



# Helper function for main()
def print_section_header(title):
  # Each line is 70 total characters long
  # ================= ... 
  # {title} 
  # ================= ... 
  top = ("=" * 70)
  middle = title 
  bottom = top
  print("\n" + top + "\n" + middle + "\n" + bottom)

# Helper function for main()
def print_ticker_header(title):
  # Each line is 70 total characters long 
  # ***************** ... 
  # ***** {title} *** ...
  # ***************** ...
  suffix_count = max(0, 70 - 5 - 2 - len(title))
  top = ("*" * 70)
  middle = ("*" * 5)   + " " + title + " " +   ("*" * suffix_count)
  bottom = top
  print("\n" * 7   + top + "\n" + middle + "\n" + bottom)

# Helper function for main()
def print_ticker_header_two(title):
  # ======================== ...
  # AAPL----AAPL----AAPL---- ...
  # ----AAPL----AAPL----AAPL ...
  # AAPL----AAPL----AAPL---- ...
  # ======================== ...

  length = len(title)
  hyphens = "-" * length
  unit = title + hyphens  # 'AAPL----'
  reverse_unit = hyphens + title  # '----AAPL'
  
  # Calculate how many full repetitions are needed to reach at least 70 characters
  unit_repeats = (70 + len(unit) - 1) // len(unit)  # Round down

  top = "=" * (length * unit_repeats * 2)
  row1 = unit * unit_repeats 
  row2 = reverse_unit * unit_repeats 
  row3 = row1
  bottom = top
  
  print("\n" * 7   + top + "\n" + row1 + "\n" + row2 + "\n" + row3 + "\n" + bottom)



# Download data for that ticker 
# Save it as a pandas DataFrame where every row/record is a date
# Adds additional columns related to Bollinger Bands
# period = one of {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'}
# interval = one of {'1m', '2m', '15m', '30m', '1h', '1d', '1wk', '1mo', '3mo'}
def build_dataframe(ticker, period='1y', interval='1d'):
  df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, multi_level_index=False)
  # ^auto_adjust=False tolerates missing closing prices due to dividends or splits
  # ^multi_level_index=False chooses the dataframe structure 

  # Store the ticker symbol by adding a new column 
  df['Ticker'] = ticker  

  # Adding new columns to the dataframe. Every record gets a value 
  df['SMA20'] = df['Close'].rolling(window=20).mean()  # Average PRICE 
  df['STD20'] = df['Close'].rolling(window=20).std()  # Average volatility of PRICE
  df['Upper'] = df['SMA20'] + 2 * df['STD20']  # Price level
  df['Lower'] = df['SMA20'] - 2 * df['STD20']  # Price level

  # Return the complete dataframe 
  return df



# Plot a dataframe
def plot_dataframe(df):
  ticker = df["Ticker"].iloc[0]  # Was saved as a column. Grab first row's value 
  plt.figure(figsize=(12,6))

  # Draw the price, SMA, and the bands 
  plt.plot(df['Close'], label='Close Price', color='blue')
  plt.plot(df['SMA20'], label='SMA20', color='orange', linestyle='--')
  plt.plot(df['Upper'], label='Upper Band', color='green', linestyle=':')
  plt.plot(df['Lower'], label='Lower Band', color='red', linestyle=':')

  # Color the channel gray 
  plt.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.2)

  # Metadata and display options   
  plt.title(f'Bollinger Bands for {ticker}')
  plt.legend()
  plt.grid(True)

  # Show the plot 
  plt.show(block=False)  # So the next code can run before the window is closed 



# Plot the dataframe and every trade the simulation made 
def plot_dataframe_and_trades(df, trades):
  ##### Plot the basic chart like plot_dataframe()
  ticker = df["Ticker"].iloc[0] 
  plt.figure(figsize=(12, 6))
  plt.plot(df['Close'], label='Close Price', color='blue')
  plt.plot(df['SMA20'], label='SMA20', color='orange', linestyle='--')
  plt.plot(df['Upper'], label='Upper Band', color='green', linestyle=':')
  plt.plot(df['Lower'], label='Lower Band', color='red', linestyle=':')
  plt.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.2)

  ##### Add the trades to the plot 
  for trade in trades:
    color = 'green' if trade['action'] == 'buy' else 'red'
    marker = '^' if trade['action'] == 'buy' else 'v'
    price = trade['per_share_cost'] if trade['action'] == 'buy' else trade['per_share_proceeds']
    plt.scatter(trade['date'], price, color=color, marker=marker, s=100, label=trade['action'].capitalize())

  # Metadata and display options   
  plt.title(f'Bollinger Band Trading Simulation for {ticker}')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.grid(True)
  plt.tight_layout()

  # Avoid duplicate legend entries
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys())

  # Show the plot 
  plt.show(block=False)  # So the next code can run before the window is closed 



# Backtests the strategy one day at a time 
# The df must have been generated using interval='1d' 
def simulate_trades(df, required_slump_length, exit_tiers):
  ##############################################################################
  ##### Constants 
  ##############################################################################
  # The Bollinger Band hit margin (applied as an OFFSET, 0 ≤ s ≤ 1) 
  s = 0.2  # Price must be within this many HW's of a band for a "hit" 

  # The SMA hit margin (applied as an OFFSET, 0 ≤ r ≤ 1) 
  r = 0.1  # Price must be within this many HW's of the SMA for a "hit" 

  # The number of shares moved during a purchase or sale 
  LOT_SIZE = 100  # Not a function parameter because it doesn't change the strategy 



  ##############################################################################
  ##### State 
  ##############################################################################
  # How many shares I currently own 
  shares = 0  

  # Current stock ownership (a list of dictionaries) 
  # Each dictionary describes one purchase of shares 
  positions = deque()  # Left side = oldest purchase, right side = newest  

  # Trade history (a list of dictionaries)
  # Each dictionary represents one transaction (buy or sell)
  trades = []  

  # A list of opportunities that were identified during the simulation 
  # Each dictionary represents one time that you could have bought 
  opportunities = [] 

  # The next opportunity you have. If the simulation ends during a slump, 
  #   the slump's info is saved here 
  future_opportunity = {}

  # Last TRADING day's price levels (used to identify hits and crosses) 
  prev_date = None  # or a pandas.Timestamp
  prev_price = None  
  prev_sma = None 
  prev_lower = None 
  prev_upper = None  

  ##### Data about the current slump  
  # Have I already purchased in this slump? 
  opportunity_taken = False  # Prevents a 60-day slump from triggering 2 trades 
  # Consecutive CALENDAR days price has closed below the SMA 
  slump_length = 0  # 0 = not in a slump. On the first day, it is set to 1
  # How many times price has hit or crossed the lower band during the current slump
  lower_band_hits = 0 
  # The first day price closed below the SMA 
  date_slump_began = None  # or a pandas.Timestamp 



  ##############################################################################
  ##### Simulation: for each day, follow the strategy 
  ##############################################################################
  for date, row in df.iterrows():  # date is a pandas.Timestamp 
    # Read today's values 
    price = float(row['Close'])
    sma   = float(row['SMA20'])
    upper = float(row['Upper'])
    lower = float(row['Lower'])

    # "Was there a yesterday?". False for row 1
    # These 4 values are always updated together at the end of the loop, so if
    #   even one is missing, all values are untrustable
    valid_prev = ( 
      prev_price is not None 
      and prev_sma is not None
      and prev_lower is not None 
      and prev_upper is not None 
    )

    # Check if the dataframe has valid indicators 
    # False until row 21 because the SMA20 requires 20 prior days 
    indicators_valid = (
      not pd.isna(sma)
      and not pd.isna(upper)
      and not pd.isna(lower)
    )

    ############################################################################
    ##### Process the day 
    ##### Nested code does not need to check valid_prev 
    ##### It skips the first 20 rows because indicators_valid is False 
    ############################################################################  
    if valid_prev and indicators_valid:  
      # Half-width 
      hw = upper - sma 

      # Division safety 
      if hw != 0: 
        # How many HW's ABOVE the SMA is the price? 
        distance = (price - sma) / hw  

        # Today's daily return is from yesterday's close to today's close
        daily_return = 0.0
        if prev_price > 0.0:
          daily_return = ((price - prev_price) / prev_price)



        ##########################################################################
        ### Price location booleans (simplifies future code)
        ##########################################################################
        # Today's price location (Exclusive)
        above_sma = price > sma 
        below_sma = price < sma  
        above_upper = price > upper
        below_upper = price < upper
        above_lower = price > lower
        below_lower = price < lower
        # Previous-day price location (Inclusive)
        prev_at_or_above_sma = prev_price >= prev_sma 
        prev_at_or_below_sma = prev_price <= prev_sma
        prev_at_or_above_upper = prev_price >= prev_upper
        prev_at_or_below_upper = prev_price <= prev_upper
        prev_at_or_above_lower = prev_price >= prev_lower
        prev_at_or_below_lower = prev_price <= prev_lower



        ############################################################################
        ### Booleans to analyze the day's events. True if it happened TODAY
        ############################################################################  
        ### CROSSES ################################################################
        # Did price cross below the sma 
        crossed_below_sma = prev_at_or_above_sma and below_sma 
        # Did price rise above the sma? 
        crossed_above_sma = prev_at_or_below_sma and above_sma 
        # Did price cross the upper band, LEAVING the channel? 
        crossed_upper_out = prev_at_or_below_upper and above_upper 
        # Did price cross the upper band, ENTERING the channel 
        crossed_upper_in = prev_at_or_above_upper and below_upper 
        # Did price cross the lower band, LEAVING the channel? 
        crossed_lower_out = prev_at_or_above_lower and below_lower 
        # Did price cross the lower band, ENTERING the channel? 
        crossed_lower_in = prev_at_or_below_lower and above_lower 

        ### HITS ###################################################################
        # Did price DROP and hit the SMA? 
        hit_sma_from_above = (
          prev_at_or_above_sma
          and 0 <= distance <= r
        )
        # Did price RISE and hit the SMA? 
        hit_sma_from_below = (
          prev_at_or_below_sma
          and -r <= distance <= 0
        )
        # Did price DROP and hit the UPPER band? 
        hit_upper_from_outside = (
          prev_at_or_above_upper
          and 1 <= distance <= 1 + s
        )
        # Did price RISE and hit the UPPER band?
        hit_upper_from_inside = (
          prev_at_or_below_upper
          and 1 - s <= distance <= 1 
        )
        # Did price DROP and hit the LOWER band? 
        hit_lower_from_inside = (
          prev_at_or_above_lower
          and -1 <= distance <= -1 + s 
        )
        # Did price RISE and hit the LOWER band? 
        hit_lower_from_outside = (
          prev_at_or_below_lower
          and -1 - s <= distance <= -1 
        )

        ### LOWER BAND TOUCHES #####################################################
        # Was the lower Bollinger Band touched or broken? 
        if (hit_lower_from_inside or crossed_lower_out):  # Dropped to or past 
          lower_band_hits += 1



        ############################################################################
        ### Increment or reset the slump counters 
        ############################################################################
        if below_sma: 
          if slump_length == 0:
            # Yesterday was not a slump
            slump_length = 1
            date_slump_began = date
          else:
            # We were already in a slump. Works for weekdays, weekends, and holidays 
            slump_length += (date - prev_date).days 
        else: 
          # Not in a slump  
          slump_length = 0
          date_slump_began = None
          opportunity_taken = False    
          lower_band_hits = 0



        ############################################################################
        ### Purchase if there is a confirmed opportunity  
        ############################################################################
        # "There is a buying opportunity today" 
        confirmed_opportunity = (
          (slump_length >= required_slump_length)  
          and not opportunity_taken  # Only one purchase per slump 
        )

        if confirmed_opportunity:
          # Record the opportunity 
          opportunities.append(
            {
              'date': date, 
              'date_slump_began': date_slump_began,
              'length_of_slump': slump_length, 
              'price': price, 
              'band_hits': lower_band_hits
            }
          )

          # Acquire the shares 
          position_size = LOT_SIZE  # Could scale depending on band_hits 
          shares += position_size
          positions.append(
            {
              'date': date,
              'price': price,
              'shares': position_size
            }
          )

          # Record the purchase 
          trades.append(
            {
              'action': 'buy',
              'date': date,
              'per_share_proceeds': None, 
              'per_share_cost': price,  
              'shares': position_size,
              'shares_still_owned': shares,
              'per_share_realized_gain': None,
              'total_realized_gain': None, 
              'days_held': None, 
              'reason': '30_day_slump'  
            }
          )

          # Do not purchase multiple times during this slump 
          opportunity_taken = True 



        ############################################################################
        ### If any positions have achieved a tiered return requirement, sell them 
        ############################################################################
        for position in list(positions):  # << A COPY of the deque's contents  
          # Read data 
          position_size = position['shares']
          roi = ((price - position['price']) / position['price']) 
          duration_in_days = (date - position['date']).days  # int 

          # If and why we will sell this position  
          sell_position = False 
          sell_reasons = []

          ##### Tier checks 
          for tier in exit_tiers:
            required_days = tier['days']
            required_return = tier['return']
            sell_reason = tier['name'] 

            # Total return tier 
            if required_days is None:
              if roi >= required_return:
                sell_position = True
                sell_reasons.append(sell_reason)

            # Time-based return tier
            elif ((duration_in_days >= required_days) and (roi >= required_return)):
              sell_position = True
              sell_reasons.append(sell_reason)
                    
          ##### Possible other exits 
          use_more = False
          if use_more:
            # A) It hit the SMA (mean reversion, limits upside)
            if crossed_above_sma:
              sell_position = True
              sell_reasons.append('sma')

            # B) It hit the upper bollinger band (mean reversion + excessive move)
            if (hit_upper_from_inside or crossed_upper_out):
              sell_position = True
              sell_reasons.append('total_return')

          ##### Sell if any target was met  
          if sell_position: 
            per_share_realized_gain = price - position['price']
            total_realized_gain = per_share_realized_gain * position_size
            positions.remove(position)
            shares -= position_size
            trades.append(
              {
                'action': 'sell',
                'date': date,
                'per_share_proceeds': price,  # None if action=buy 
                'per_share_cost': position['price'], 
                'shares': position_size,
                'shares_still_owned': shares,
                'per_share_realized_gain': per_share_realized_gain,  # None if action=buy
                'total_realized_gain': total_realized_gain,  # None if action=buy
                'days_held': duration_in_days,  # None if action=buy 
                'reason': ', '.join(sell_reasons)  # One long string 
              }
            )



        ############################################################################
        ### If it's the final day and it's in a slump, tell the user about the 
        ###   current slump so they can keep an eye on this stock
        ############################################################################
        # "We were in a slump, but the simulation is over" 
        opportunity_cut_short = (
          (0 < slump_length < required_slump_length)  # Was not a real opportunity yet
          and date == df.index[-1]  # It's the final day 
        )

        if (opportunity_cut_short):
          future_opportunity = {
            'final_date_checked': date, 
            'date_slump_began': date_slump_began,
            'length_of_slump': slump_length,  # If it's 29 today, you will purchase tomorrow evening right before it turns to 30 
            'opportunity_date': date_slump_began + timedelta(days=required_slump_length),  # If slump continues, purchase on this day 
            'band_hits': lower_band_hits
          } 

      
    
    ##############################################################################
    ##### Save today's row for the next iteration 
    ##### This runs after every iteration, even if valid_prev is false
    ##############################################################################
    prev_date = date
    prev_price = price
    prev_sma = sma
    prev_lower = lower
    prev_upper = upper

  ##### End of loop 



  ##############################################################################
  ##### Return findings (as of the last day's close) 
  ##############################################################################
  return trades, positions, opportunities, future_opportunity 





# Main 
def main():
  ##############################################################################
  ##### Simulation inputs 
  ##############################################################################

  ##### How many calendar days of data you want to fetch from Yahoo 
  period = '2y'  # One of {'3mo', '6mo', '1y', '2y', '5y', '10y', 'max'}

  ##### How many consecutive calendar days price has to close below the SMA 
  required_slump_length = 30  
  
  ##### Create as many watchlists as you want 
  favorites = ['AAPL', 'HD', 'KMB', 'MCD', 'CMG']
  backup_list = ['AAPL']
  tickers_3 = ['HD', 'KMB']
  tickers_bad = ['NVDA', 'ORCL', 'HOOD', 'AMZN', 'RGTI', 'RKLB', 'TMUS']  # Poor results 

  ##### Choose one for the simulation (the 'tickers' variable is used) 
  tickers = favorites 

  ##### Create as many exit-requirement lists as you want 
  # Each exit tier is a date+return combo. If a position hits that return before
  #   or on that number of days, it gets sold. 
  # A position is sold when at least one of the tiers is met 
  # Any tier set to 10 or above will practically be ignored (a 1,000%+ return)
  exit_tiers_1 = [  # Four tiers 
    {'days': 14, 'return': 0.06, 'name': '14d_return'},
    {'days': 30, 'return': 0.05, 'name': '30d_return'},
    {'days': 60, 'return': 0.10, 'name': '60d_return'},
    {'days': None, 'return': 0.15, 'name': 'total_return'}
  ]  
  exit_tiers_2 = [  # Three "real" tiers (14d = 1,006%)
    {'days': 14, 'return': 10.06, 'name': '14d_return'},
    {'days': 30, 'return': 0.05, 'name': '30d_return'},
    {'days': 60, 'return': 0.10, 'name': '60d_return'},
    {'days': None, 'return': 0.15, 'name': 'total_return'}
  ]  
  exit_tiers_3 = [  # Two tiers  
    {'days': 60, 'return': 0.10, 'name': '60d_return'},
    {'days': None, 'return': 0.15, 'name': 'total_return'}
  ]  
  exit_tiers_4 = [  # Total return only 
    {'days': None, 'return': 0.15, 'name': 'total_return'}
  ] 

  ##### Choose one for the simulation (the 'exit_tiers' variable is used) 
  exit_tiers = exit_tiers_1  

  ##### Printing settings 
  print_trades = True
  print_future_opportunities = True
  print_performance = False
  print_summary = (False and print_performance)  # Performance calculations must exist 



  ##############################################################################
  ##### Run the simulation 
  ##############################################################################

  ##### Create each dataframe and store them by ticker 
  dataframes = {} 
  for ticker in tickers:
    df = build_dataframe(ticker, period=period, interval='1d') 
    dataframes[ticker] = df

  ##### Run the simulations and store results by ticker 
  sim_results = {}
  for ticker, df in dataframes.items():
    ticker = df["Ticker"].iloc[0] 
    trades, final_positions, opportunities, future_opportunity = simulate_trades(df, required_slump_length, exit_tiers)
    sim_results[ticker] = (trades, final_positions, opportunities, future_opportunity)

  ##### Plot every ticker's results   
  for ticker in tickers:
    plot_dataframe_and_trades(dataframes[ticker], sim_results[ticker][0])



  ##############################################################################
  ##### Results 
  ##############################################################################
  for ticker in tickers:
    print_ticker_header_two(ticker)
    ##### Show trade history (each slump's first opportunity, and any sales)
    if print_trades: 
      print_section_header("TRADE HISTORY")
      for trade in sim_results[ticker][0]:
        # Extract values (more condensed print statements) and reformat if needed 
        date = trade['date'].date()  # A date.datetime object, no seconds
        action = trade['action'].upper()
        shares = trade['shares']
        purchase_price = trade['per_share_cost']
        sale_price = trade['per_share_proceeds']
        shares_still_owned = trade['shares_still_owned']
        days_held = trade['days_held']
        reason = trade['reason']

        if action == 'BUY':
          print(f"{date} - BUY {shares} shares @ ${purchase_price:,.2f}   |   Shares owned now: {shares_still_owned}")

        elif action == 'SELL':
          per_share_gain = trade['per_share_realized_gain']
          total_gain = trade['total_realized_gain']
          return_percent = (per_share_gain / purchase_price) * 100
          print(f"{date} - SELL {shares} shares @ ${sale_price:,.2f}  |   Shares owned now: {shares_still_owned}")
          print(f"~~~ Gain: {shares} * ${per_share_gain:,.2f} = ${total_gain:,.2f}")
          print(f"~~~ Held for {days_held} days and sold for a {return_percent:.2f}% return")   
          print(f"~~~ Reason for selling: {reason}")



    ##### Show future opportunities
    if print_future_opportunities:
      print_section_header("FUTURE OPPORTUNITY")
      opp = sim_results[ticker][3]
      if opp:
        # Extract data for better printing 
        start = opp['date_slump_began'].date() 
        final = opp['final_date_checked'].date()
        length = opp['length_of_slump']
        date = opp['opportunity_date'].date()
        # Show everything  
        print(f"The slump began on {start}")
        print(f"It lasted for {length} days, all the way to the end of the simulation")
        print(f"The last date of the simulation was {final}")
        print(f"The 30-day mark is {date}, so if the slump continues until then,")
        print(f"  you should purchase shares at the close on {date}")
      else:  # Empty dictionary 
        print(f"No future opportunities")


    
    ##### Check the performance 
    if print_performance:
      df = dataframes[ticker]
      starting_price = float(df['Close'].iloc[0]) 
      current_price = float(df['Close'].iloc[-1]) 

      ### Exited positions 
      total_realized_gain = 0.0
      capital_spent_on_sold_shares = 0.0
      for trade in trades:
        if trade['action'].upper() == 'SELL':
          capital_spent_on_sold_shares += trade['per_share_cost'] * trade['shares']
          total_realized_gain += trade['total_realized_gain']

      ### Remaining positions 
      total_unrealized_gain = 0.0
      capital_still_deployed = 0.0 
      shares_still_owned = 0  
      for position in final_positions:
        # Calculate 
        position_cost_basis = position['shares'] * position['price']
        shares = position['shares']
        unrealized_gain = (shares * current_price) - position_cost_basis
        # Update totals
        capital_still_deployed += position_cost_basis
        shares_still_owned += shares 
        total_unrealized_gain += unrealized_gain 

      ### Final account stats  
      net_profit = total_realized_gain + total_unrealized_gain
      total_positions_value = shares_still_owned * current_price
      total_capital_deployed = capital_spent_on_sold_shares + capital_still_deployed
      # Average return on positions you bought and later sold 
      if capital_spent_on_sold_shares > 0:
        percent_return_on_past_positions = (total_realized_gain / capital_spent_on_sold_shares) * 100
      else:
        percent_return_on_past_positions = 0
      # Average return on positions you bought and still own 
      if (shares_still_owned > 0):
        percent_return_on_active_positions = (total_unrealized_gain / capital_still_deployed) * 100
        average_cost_basis = capital_still_deployed / shares_still_owned
      else:
        percent_return_on_active_positions = 0
        average_cost_basis = 0
    


    ##### Display a summary 
    if print_summary:  
      print_section_header("SUMMARY")
      # Overview 
      print(f"Stock price {period} ago:          ${starting_price:,.2f}")
      print(f"Most recent closing price:   ${current_price:,.2f}")
      # Current position 
      print(f"Shares currently owned:     {shares_still_owned}")
      print(f"Current value of shares:    ${total_positions_value:,.2f}")
      print(f"Average cost basis:         ${average_cost_basis:,.2f}")
      # Key data 
      print(f"Unrealized gain:            ${total_unrealized_gain:,.2f}")
      print(f"Realized gain:              ${total_realized_gain:,.2f}")



  # Keep all graph windows open
  plt.show()  # Without this, main() ending causes all plots to close 

##### End of main





# "When someone runs this file, run the main() function" 
if __name__ == '__main__':
  main()



