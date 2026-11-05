"""
Does the signal help with DCA returns? 
Ignores interest and dividends. 
The aggregate returns of the watchlist are the equal-weighted values because 
  we purchased an equal dollar amount of each stock ($100 per month). 
"""


from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import exchange_calendars as xcals 

import csv
from pathlib import Path



################################################################################
### CONSTANTS 
################################################################################

# NYSE calendar includes holidays and early closing sessions
XNYS = xcals.get_calendar("XNYS")  # Exchange New York Stock 
EASTERN = ZoneInfo("America/New_York")





################################################################################
### Data preparation
################################################################################

# Download data for that ticker 
# Save it as a pandas DataFrame where every row/record is a date
# Adds additional columns related to Bollinger Bands
# period = one of {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'}
# interval = one of {'1m', '2m', '15m', '30m', '1h', '1d', '1wk', '1mo', '3mo'}
def build_dataframe(ticker, period="2y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,  # tolerate missing closing prices due to dividends or splits
        multi_level_index=False,  # chooses the dataframe structure
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    # For daily data, ignore today's unfinished candle before market close. 
    # This prevents the scenario where the stock is in a slump at the close, but 
    #   we downloaded data during the midday rally, so we mistakenly cancel the  
    #   opportunity counter. 
    if interval == "1d":
        now = datetime.now(EASTERN)
        today = pd.Timestamp(now.date())

        # Is today an actual NYSE trading session?
        if XNYS.is_session(today):
            # Get the actual close time, including early closes.
            market_close = XNYS.session_close(today)
            # Compare timestamps in UTC.
            now_utc = pd.Timestamp(datetime.now(timezone.utc))

            if now_utc < market_close:
                # Daily Yahoo Finance indexes usually represent session dates 
                last_date = pd.Timestamp(df.index[-1]).date()
                # Remove the last row 
                if last_date == now.date():
                    df = df.iloc[:-1]


    # If today was the only day in the dataframe and we removed it. 
    if df.empty: 
        raise ValueError(f"No completed price data available for {ticker}.")

    # Add new columns to the dataframe. Every row gets some value. 
    df['Ticker'] = ticker  # Store the ticker symbol in every row 
    df['SMA20'] = df['Close'].rolling(window=20).mean()  # Average price ($)
    df['STD20'] = df['Close'].rolling(window=20).std()  # Average volatility of price ($)
    df['Upper'] = df['SMA20'] + 2 * df['STD20']  # Price level ($)
    df['Lower'] = df['SMA20'] - 2 * df['STD20']  # Price level ($)
    return df 



# Loads a large watchlist from a CSV file into one string 
DATA_PATH = Path(__file__).parent / "data"
def load_ticker_string(filename: str) -> str:
    tickers_file = DATA_PATH / filename
    return ", ".join(
        ticker.strip()
        for ticker in tickers_file.read_text().split(",")
        if ticker.strip()  # If line had extra commma at end, ignore it
    )
sp500 = load_ticker_string("sp500_tickers.csv")  # Only 453 tickers. Also excludes brk.b because of formatting. 
dow30 = load_ticker_string("dow30_tickers.csv")





################################################################################
### Opportunity detection
#################################################################################

# Identifies past opportunities within a dataframe. If the program ends while 
#   the stock is in a slump, it also tells you about a future opportunity. 
def identify_opportunities(df, required_slump_length=30):
    opportunities = []

    # Current slump 
    slump_length = 0
    date_slump_began = None
    opportunity_recorded = False  # One opportunity per slump 
    lower_band_hits = 0

    # Loop variable 
    previous_date = None  

    for date, row in df.iterrows():
        price = row["Close"]
        sma = row["SMA20"]
        upper = row["Upper"]
        lower = row["Lower"]

        # Skip dates before the 20-day indicators are available.
        if pd.isna(price) or pd.isna(sma) or pd.isna(upper) or pd.isna(lower):
            continue

        price = float(price)
        sma = float(sma)
        upper = float(upper)
        lower = float(lower)

        # Count lower band touches/crosses using today's close 
        if price <= lower:
            lower_band_hits += 1

        if price < sma:
            if slump_length == 0:
                # First day below SMA is calendar day 1.
                slump_length = 1
                date_slump_began = date
                opportunity_recorded = False
                lower_band_hits = 1 if price <= lower else 0
            else:
                # Include weekends and holidays between trading observations.
                slump_length += (date - previous_date).days

            if slump_length >= required_slump_length and not opportunity_recorded:
                opportunities.append({
                    "date": date,
                    "date_slump_began": date_slump_began,
                    "length_of_slump": slump_length,
                    "price": price,
                    "band_hits": lower_band_hits,
                    "reason": "required_slump_length_reached",
                })
                opportunity_recorded = True

        else:
            # A close at or above the SMA ends the slump.
            slump_length = 0
            date_slump_began = None
            opportunity_recorded = False
            lower_band_hits = 0

        previous_date = date

    return opportunities





################################################################################
### Plotting
################################################################################

# Plot the dataframe, indicators, and detected opportunities 
def plot_dataframe(df, opportunities=None):
    ticker = str(df["Ticker"].iloc[0])
    plt.figure(figsize=(13, 7))

    plt.plot(df.index, df["Close"], label="Close Price")
    plt.plot(df.index, df["SMA20"], label="SMA20", linestyle="--")
    plt.plot(df.index, df["Upper"], label="Upper Band", linestyle=":")
    plt.plot(df.index, df["Lower"], label="Lower Band", linestyle=":")
    plt.fill_between(
        df.index,
        df["Upper"].to_numpy(dtype=float),
        df["Lower"].to_numpy(dtype=float),
        alpha=0.15,
        label="Bollinger Channel",
    )

    if opportunities:
        dates = [opp["date"] for opp in opportunities]
        prices = [opp["price"] for opp in opportunities]
        plt.scatter(
            dates, prices, marker="^", s=100,
            label="Opportunity detected", zorder=5,
        )

    plt.title(f"Bollinger Band Opportunities for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)





################################################################################
##### Output helpers
################################################################################

# Printing the ticker symbol at the top of its data area  
def print_ticker_header(title):
  # AAPL----AAPL----AAPL---- ...
  # ----AAPL----AAPL----AAPL ...
  # AAPL----AAPL----AAPL---- ...

  length = len(title)
  hyphens = "-" * length
  unit = title + hyphens  # 'AAPL----'
  reverse_unit = hyphens + title  # '----AAPL'
  
  # Calculate how many full repetitions are needed to reach at least 70 characters
  unit_repeats = (70 + len(unit) - 1) // len(unit)  # Round down

  row1 = unit * unit_repeats 
  row2 = reverse_unit * unit_repeats 
  row3 = row1
  
  print("\n" * 3  + row1 + "\n" + row2 + "\n" + row3 )


# Printing a header for a section inside a ticker's data 
def print_section_header(title):
    # print("\n" + "=" * 70)
    # print(title)
    # print("=" * 70)
    print(f"\n{title}:") 

# Calculate performance of a strategy 
def calculate_performance(invested, ending_balance):
    """Return profit and return percentage."""
    profit = ending_balance - invested
    return_pct = (profit / invested * 100) if invested else 0.0
    return profit, return_pct

# Show performance of a DCA strategy  
def print_dca_result(label, invested, ending_balance):
    profit, return_pct = calculate_performance(
        invested, ending_balance
    )

    print(
        f"{label:<16}"
        f" Invested: ${invested:>9,.2f}"
        f" | Ending: ${ending_balance:>10,.2f}"
        f" | Profit: ${profit:>9,.2f}"
        f" | Return: {return_pct:>7.2f}%"
    )





################################################################################
##### DCA Mode 
################################################################################
MONTHLY_DCA_AMOUNT = 100.0

"""
Invest $100 on the first trading day of every month. At the closing price. 
"""
# Return (total_amount_invested, ending_market_value) 
def simulate_dca(df):
    total_invested = 0.0
    shares_owned = 0.0

    # The first available trading day in each month.
    monthly_dates = (
        pd.Series(df.index, index=df.index)
        .groupby(df.index.to_period("M"))
        .first()
    )

    for date in monthly_dates:
        price = float(df.loc[date, "Close"])

        if pd.isna(price) or price <= 0:
            continue

        shares_owned += MONTHLY_DCA_AMOUNT / price
        total_invested += MONTHLY_DCA_AMOUNT

    ending_price = float(df["Close"].iloc[-1])
    ending_balance = shares_owned * ending_price

    return total_invested, ending_balance



"""
Invest $100 on the first trading day of every month. At the closing price.
An advanced investment is a loan against future monthly contributions, 
  not extra money added to the strategy. If you choose make an advanced 
  payment, you become "in debt" to that source of capital. 
If you are debt-free and an opportunity opens up, you create an advanced 
  investment: you invest the next X months' contributions on that day, at 
  close, and you are now in debt. 
For the next X months, your typical contribution amount is spent, but it does
  not go toward new shares. Instead, it is used to pay down the debt from the 
  accelerated investment.  
Once you are debt-free, you go back to normal monthly contributions, and you 
  once again have the ability to make an advanced investment when an opportunity
  arrives. 
Example, with months=3: on January 1 you are debt-free so you invest the normal 
  $100 into new shares. Then an opportunity opens on Jan 15, so you invest $200 
  at market close. Now you have "borrowed" $200, so when you get more money to 
  inveset (from your Feb 1 paycheck), you must spend it to pay back the debt. The
  same thing happens on March 1. Then during March you can make an advanced 
  investment, or if no opportunies arise, you can invest normally on April 1.  
This excludes interest and dividends. 
"""
# Return (total_amount_invested, final_account_balance) 
def simulate_dca_with_opps(df, opportunities, months=3):
    if months < 1:
        raise ValueError("months must be at least 1")

    monthly_amount = MONTHLY_DCA_AMOUNT

    # All trading days, grouped by month
    monthly_dates = (
        pd.Series(df.index, index=df.index)
        .groupby(df.index.to_period("M"))
        .first()
    )

    # First available trading day in each calendar month.
    first_trading_days = {
        pd.Timestamp(date).normalize()
        for date in monthly_dates.values
    }

    # Every opportunity date  
    opportunity_dates = {
        pd.Timestamp(opp["date"]).normalize()
        for opp in opportunities
    }

    # Starting values 
    shares_owned = 0.0
    debt_remaining = 0.0
    total_contributions = 0.0

    for date, row in df.iterrows():
        date = pd.Timestamp(date).normalize()
        month = date.to_period("M")
        price = float(row["Close"])

        if pd.isna(price) or price <= 0:
            continue

        # If it's the first trading day in a month (after we get our paycheck) 
        if date in first_trading_days:
            total_contributions += monthly_amount

            if debt_remaining > 0:
                # The paycheck repays debt instead of buying shares.
                debt_remaining = max(0.0, debt_remaining - monthly_amount)
            else:
                # Make the normal monthly contribution.
                shares_owned += monthly_amount / price 

        # Check if you can make an advanced contribution today
        if (date in opportunity_dates and debt_remaining == 0):
            advance_amount = months * monthly_amount
            shares_owned += advance_amount / price
            debt_remaining = advance_amount

    # Calculate the value of the accumulated portfolio  
    ending_price = float(df["Close"].iloc[-1])
    ending_balance = shares_owned * ending_price

    return total_contributions, ending_balance





################################################################################
##### Main
################################################################################
def main():
    ############################################################################
    ##### WHAT YOU CHANGE 
    ############################################################################

    ### SETTINGS  
    backtest_mode_period = "5y" 
    required_slump_length = 30
    show_plots = False
    months = 6

    ### WATCHLISTS
    # Create multiple watchlists as CSV strings 
    smaller1 = "WM, HOOD, CRCL, NVDA, PEP, KO, KMB, CVX, MCD, CMG" 
    dad_1 = "WM, HOOD, CRCL, NVDA, PEP, KO, KMB, CVX, MCD, CMG" 

    # Select a watchlist 
    chosenWatchlist = dow30



    ############################################################################
    ##### DO NOT CHANGE ANYTHING BELOW THIS 
    ############################################################################
    interval = "1d" 
    tickers = [ticker.strip() for ticker in chosenWatchlist.upper().split(',')]  # Real list 



    ########################################################################
    ##### Calculate results for each stock, display results individually 
    ########################################################################
    results_by_ticker = {}

    for ticker in tickers:
        try:
            # Build the dataframe
            df = build_dataframe(ticker, period=backtest_mode_period, interval=interval)

            # Identify opportunities 
            opportunities = identify_opportunities(df, required_slump_length=required_slump_length)

            # Basic 
            base_invested, base_ending = simulate_dca(df)

            # Strategy. Try with several months. 
            month_results = []
            for i in range(1,7):  # 1,2,3,4,5,6
                strat_invested, strat_ending = simulate_dca_with_opps(df, opportunities, months=i)
                month_results.append((i, strat_invested, strat_ending ))

            # Store the results of all the analysis functions 
            results_by_ticker[ticker] = {
                "dataframe": df,
                "opportunities": opportunities, 
                "base_invested": base_invested, 
                "base_ending": base_ending, 
                "month_results": month_results
            }

            if show_plots:
                plot_dataframe(df, opportunities)

        except Exception as exc:
            print(f"Could not analyze {ticker}: {exc}")
            # pass 



    ########################################################################
    ##### Ensure the charts stay printed until main() ends  
    ########################################################################
    if show_plots:
        plt.show()



    
    ########################################################################
    ##### Display a condensed summary: one line per stock
    ########################################################################
    print_section_header(f"PER-STOCK WATCHLIST SUMMARY   |   months = {months}")

    for ticker in tickers:
        if ticker not in results_by_ticker:
            continue

        result = results_by_ticker[ticker]

        base_invested = result["base_invested"]
        base_ending = result["base_ending"]
        base_profit, base_return = calculate_performance(base_invested, base_ending)
        print(f"\n{ticker}")
        print_dca_result("Baseline DCA", base_invested, base_ending)

        for months, strat_invested, strat_ending in result["month_results"]:
            strat_profit, strat_return = calculate_performance(strat_invested, strat_ending)
            print(
                f"  {months:>2}-month strategy"
                f" | Invested: ${strat_invested:>9,.2f}"
                f" | Ending: ${strat_ending:>10,.2f}"
                f" | Profit: ${strat_profit:>9,.2f}"
                f" | Return: {strat_return:>7.2f}%"
                f" | vs Base: {strat_return - base_return:+.2f} pp"
            )





    ########################################################################
    ##### Display a summary for the entire watchlist
    ########################################################################
    print_section_header("OVERALL WATCHLIST RESULTS")

    total_base_invested = sum(
        result["base_invested"]
        for result in results_by_ticker.values()
    )
    total_base_ending = sum(
        result["base_ending"]
        for result in results_by_ticker.values()
    )

    base_profit, base_return = calculate_performance(total_base_invested, total_base_ending)

    print()
    print_dca_result("Baseline DCA", total_base_invested, total_base_ending)

    # Aggregate each strategy size across the watchlist.
    for months in range(1, 7):
        total_invested = 0.0
        total_ending = 0.0

        for result in results_by_ticker.values():
            for (
                result_months,
                strat_invested,
                strat_ending
            ) in result["month_results"]:

                if result_months == months:
                    total_invested += strat_invested
                    total_ending += strat_ending
                    break

        strat_profit, strat_return = calculate_performance(total_invested, total_ending)

        print_dca_result(f"{months}-month DCA", total_invested, total_ending)

        return_diff_percent = strat_return - base_return  # Whole number percent 
        return_alpha = ((1 + (return_diff_percent/100))**(1 / 5) - 1) * 100  # 5 years of backtesting 
        print(
            f"  vs Baseline:"
            f" Ending diff: ${total_ending - total_base_ending:+,.2f}"
            f" | Profit diff: ${strat_profit - base_profit:+,.2f}"
            f" | Return diff: {return_diff_percent:+.2f} pp"
            f" | Return alpha: {return_alpha:+.2f} pp"
        )



##### End of main


if __name__ == "__main__":
    main()

















