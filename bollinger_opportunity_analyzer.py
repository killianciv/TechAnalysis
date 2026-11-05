"""
This file has two modes that you adjust using main()'s backtesting_mode variable 

"Backtest" mode tests the historical success rate of the trading strategy by 
  identifying historical opportunities, then measuring what happened to the 
  stock price after each opportunity. For each company in the list, over the 
  chosen period of time, it calculates: 
 - Total opportunities 
 - The percent of opportunities that achieve a 10, 15, or 20 percent return 
 - The AVERAGE time it took SUCCEEDERS to reach a 10, 15, or 20 percent return

"Future" mode identifies half-formed opportunities so you can act in the future. It
  figures out which stocks are currently in a slump, and for each stock in a slump, 
  it tells you when its 30-day mark is, so you can check up on it. 

**** SCRIPT-SPECIFIC DETAILS
In backtest mode, an opportunity with under 90 days of follow-up is excluded  
  from the success-rate calculation, rather than being counted as a failure. 
The opportunities.csv file is a hardcoded file path that it rewritten each time 
  the program runs, so if you want to save its contents after a run, make a copy
  of it and move it into a folder named after the date of the final finished 
  candle it used. If you download the data in the middle of a trading day, the 
  code excludes today's candle, so use the date of the previous trading day. 
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
    future_opportunity = {}

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

    # If the loop's final iteration left slump_length > 0 but there was not a 
    #   confirmed opportunity yet, report it. 
    if slump_length > 0 and not opportunity_recorded:
        opportunity_date = date_slump_began + timedelta(days=required_slump_length - 1)
        final_date_checked = df.index[-1]
        future_opportunity = {
            "final_date_checked": final_date_checked,
            "date_slump_began": date_slump_began,
            "length_of_slump": slump_length,
            "opportunity_date": opportunity_date, 
            "days_until_opportunity": max(0, (opportunity_date - final_date_checked).days),
            "band_hits": lower_band_hits,
        }

    return opportunities, future_opportunity





################################################################################
### Evaluate What Happened to Price After Each Opportunity 
################################################################################ 

# For each PAST opportunity, check whether it eventually the return targets 
# For each opportunity, record whether any later available closing price reached
#   any of the targets, the first date it did, and how long it took. 
# An opportunity that never reaches a target within the downloaded data is
#   recorded as not having reached it within the observation window. 
def evaluate_opportunities(df, opportunities, return_targets=(0.10, 0.15, 0.20), min_followup_days=90):
    # Becomes 'opportunities', with extra information in each one 
    evaluated = []

    for opportunity in opportunities:
        result = opportunity.copy()
        opportunity_date = opportunity["date"]
        entry_price = opportunity["price"]

        # Exclude recent opportunities from success-rate calculations unless
        #   at least min_followup_days of calendar time has elapsed since signal.
        final_date = df.index[-1]
        result["followup_days"] = (final_date.date() - opportunity_date.date()).days
        result["eligible_for_success_rate"] = (
            result["followup_days"] >= min_followup_days
        )

        # For this opportunity, don't look at days before it
        future_prices = df.loc[df.index > opportunity_date, "Close"].dropna()

        for target in return_targets:
            target_pct = int(round(target * 100))
            target_price = entry_price * (1 + target)

            # An array of booleans. On each day, did we meet this target? 
            reached = future_prices[future_prices >= target_price]

            # It was reached it on the first day where it's true. 
            result[f"reached_{target_pct}pct"] = not reached.empty
            if reached.empty:
                result[f"date_reached_{target_pct}pct"] = None
                result[f"days_to_{target_pct}pct"] = None
            else:
                first_date = reached.index[0]
                result[f"date_reached_{target_pct}pct"] = first_date
                result[f"days_to_{target_pct}pct"] = (
                    first_date - opportunity_date
                ).days

        evaluated.append(result)

    return evaluated


# For each PAST opportunity evaluated in evaluate_opportunities(), print a summary. 
def summarize_opportunities(opportunities, return_targets=(0.10, 0.15, 0.20), min_followup_days=90):
    """Print target success rates using only sufficiently observed opportunities."""
    total = len(opportunities)
    eligible = [
        opp for opp in opportunities
        if opp["eligible_for_success_rate"]
    ]
    eligible_total = len(eligible)

    print(f"Total opportunities: {total}")
    print(
        f"Opportunities with at least {min_followup_days} calendar days "
        f"of follow-up: {eligible_total}"
    )

    if total == 0:
        print("No opportunities were identified.")
        return

    print("\nForward-return results (eligible opportunities only):")
    for target in return_targets:
        pct = int(round(target * 100))
        successful = [
            opp for opp in eligible
            if opp[f"reached_{pct}pct"]
        ]
        success_count = len(successful)
        success_rate = 100 * success_count / eligible_total if eligible_total else None

        durations = [
            opp[f"days_to_{pct}pct"]
            for opp in successful
            if opp[f"days_to_{pct}pct"] is not None
        ]
        average_days = sum(durations) / len(durations) if durations else None

        rate_text = f"{success_rate:.1f}%" if success_rate is not None else "N/A"
        print(
            f"  {pct}% return: {success_count}/{eligible_total} eligible opportunities "
            f"({rate_text}) reached the target"
        )
        if durations:
            sorted_durations = sorted(durations)
            days_list = ", ".join(str(days) for days in sorted_durations)
            print(f"    Calendar days to target (ascending): {days_list}")
            print(f"    Average calendar days to target: {average_days:.1f}")
        else:
            print("    Calendar days to target (ascending): None")
            print("    Average calendar days to target: N/A (no successes)")





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


# Print data about each past opportunity 
def print_opportunities(opportunities):
    print_section_header("OPPORTUNITIES")
    if not opportunities:
        print("No completed opportunities in this period.")
        return

    for index, opp in enumerate(opportunities, start=1):
        print(
            f"{index}. {opp['date'].date()} | "
            f"close=${opp['price']:.2f} | "
            f"slump began={opp['date_slump_began'].date()} | "
            f"slump length={opp['length_of_slump']} calendar days | "
            f"lower-band touches={opp['band_hits']}"
        )

        for target_pct in (10, 15, 20):
            if opp[f"reached_{target_pct}pct"]:
                date_reached = opp[f"date_reached_{target_pct}pct"].date()
                days = opp[f"days_to_{target_pct}pct"]
                print(
                    f"   {target_pct}% target: reached {date_reached} "
                    f"({days} calendar days later)"
                )
            else:
                print(
                    f"   {target_pct}% target: not reached within "
                    "the available data"
                )


# Print data about the future opportunity 
def print_future_opportunity(future_opportunity):
    print_section_header("DEVELOPING SLUMP")
    if not future_opportunity:
        print("No unfinished slump below the required duration.")
        return

    opp = future_opportunity
    print(f"Slump began: {opp['date_slump_began'].date()}")
    print(f"Last date checked: {opp['final_date_checked'].date()}")
    print(f"Current slump length: {opp['length_of_slump']} calendar days")
    print(f"Required opportunity date: {opp['opportunity_date'].date()}")
    if opp["days_until_opportunity"] > 0:
        print(f"Calendar days remaining: {opp['days_until_opportunity']}")
    else:
        print("The date threshold has passed; check the next available close.")





################################################################################
##### Main
##### Backtesting mode: show all the summaries
##### Future mode: show just the future opportunities  
################################################################################
def main():
    ############################################################################
    ##### WHAT YOU CHANGE 
    ############################################################################

    ### SETTINGS  
    backtesting_mode = False 
    backtest_mode_period = "2y" 
    required_slump_length = 30

    ### BACKTESTING MODE SETTINGS
    return_targets = (0.10, 0.15, 0.20)
    min_followup_days = 90
    show_plots = False
    summarize_past_opportunities = False
    show_each_opportunitys_details = False
    print_the_future_opportunity = True

    ### FUTURE MODE SETTINGS
    future_mode_file_ouput = True  # Write the output to a file instead of the terminal 
    sorted_and_cleaned = True  # If you already ran it with False and want it sorted and cleaned, run ```./.venv/bin/python data/cleaner_script.py```

    ### WATCHLISTS
    # Create multiple watchlists as CSV strings 
    smaller1 = "WM, HOOD, CRCL, NVDA, PEP, KO, KMB, CVX, MCD, CMG" 
    dad_1 = "WM, HOOD, CRCL, NVDA, PEP, KO, KMB, CVX, MCD, CMG" 

    # Select a watchlist 
    chosenWatchlist = sp500  



    ############################################################################
    ##### DO NOT CHANGE ANYTHING BELOW THIS 
    ############################################################################
    future_mode = not backtesting_mode
    future_mode_terminal_ouput = not future_mode_file_ouput
    interval = "1d" 
    future_mode_period = "3mo"  # First month of data is useless because we need 20TD for indicators
    tickers = [ticker.strip() for ticker in chosenWatchlist.upper().split(',')]  # Real list 


    # If the user wants to see missed opportunities and the success rate  
    if backtesting_mode:
        results_by_ticker = {}

        ########################################################################
        ##### Calculate results for each stock, display results individually 
        ########################################################################
        for ticker in tickers:
            print_ticker_header(ticker)

            try:
                # Build the dataframe
                df = build_dataframe(ticker, period=backtest_mode_period, interval=interval)

                # Identify opportunities 
                opportunities, future_opportunity = identify_opportunities(
                    df, required_slump_length=required_slump_length
                )

                # See what price does after each opportunity 
                opportunities = evaluate_opportunities(
                    df,
                    opportunities,
                    return_targets=return_targets,
                    min_followup_days=min_followup_days,
                )

                # Store the results of all the analysis functions 
                results_by_ticker[ticker] = {
                    "dataframe": df,
                    "opportunities": opportunities,
                    "future_opportunity": future_opportunity,
                }

                # Print whatever sections the user wants 
                if summarize_past_opportunities:
                    print_section_header("OPPORTUNITY SUMMARY")
                    summarize_opportunities(
                        opportunities,
                        return_targets=return_targets,
                        min_followup_days=min_followup_days,
                    )
                if show_each_opportunitys_details:
                    print_opportunities(opportunities)
                if print_the_future_opportunity:
                    print_future_opportunity(future_opportunity)
                if show_plots:
                    plot_dataframe(df, opportunities)

            except Exception as exc:
                # print(f"Could not analyze {ticker}: {exc}")
                pass 



        ########################################################################
        ##### Ensure the charts stay printed until main() ends  
        ########################################################################
        if show_plots:
            plt.show()



        ########################################################################
        ##### Display a condensed summary: one line for each stock's success
        ########################################################################
        print_section_header("PER-STOCK WATCHLIST SUMMARY")
        print(f"Success rates use only opportunities with {min_followup_days}+ calendar days of follow-up.")
        for ticker, result in results_by_ticker.items():
            opportunities = result["opportunities"]
            total = len(opportunities)
            eligible = [
                opp for opp in opportunities
                if opp["eligible_for_success_rate"]
            ]
            eligible_total = len(eligible)
            target_summaries = []

            for target_pct in (10, 15, 20):
                successful = sum(
                    opp[f"reached_{target_pct}pct"] for opp in eligible
                )
                rate = (100 * successful / eligible_total) if eligible_total else None
                rate_text = f"{rate:.1f}%" if rate is not None else "N/A"
                target_summaries.append(f"{target_pct}%: {successful}/{eligible_total} ({rate_text})")

            print(
                f"{ticker}: {total} opportunities, {eligible_total} eligible | "
                + " | ".join(target_summaries)
            )



        ########################################################################
        ##### Display a summary for the strategy's success over this watchlist 
        ########################################################################
        print_section_header("OVERALL WATCHLIST RESULTS")
        all_opportunities = [
            opportunity
            for result in results_by_ticker.values()
            for opportunity in result["opportunities"]
        ]

        # Doesn't just average the percentages of individual stocks
        total_opportunities = len(all_opportunities)
        eligible_opportunities = [
            opp for opp in all_opportunities
            if opp["eligible_for_success_rate"]
        ]
        eligible_total = len(eligible_opportunities)
        
        print(f"Stocks successfully analyzed: {len(results_by_ticker)}/{len(tickers)}")
        print(f"Total opportunities across watchlist: {total_opportunities}")
        print(
            f"Eligible opportunities with at least {min_followup_days} calendar days "
            f"of follow-up: {eligible_total}"
        )

        if total_opportunities == 0:
            print("No opportunities were identified across the watchlist.")
        else:
            for target_pct in (10, 15, 20):
                successful = [
                    opp for opp in eligible_opportunities
                    if opp[f"reached_{target_pct}pct"]
                ]
                success_count = len(successful)
                success_rate = 100 * success_count / eligible_total if eligible_total else None
                durations = sorted(
                    opp[f"days_to_{target_pct}pct"]
                    for opp in successful
                    if opp[f"days_to_{target_pct}pct"] is not None
                )

                rate_text = f"{success_rate:.1f}%" if success_rate is not None else "N/A"
                print(
                    f"\n{target_pct}% return: {success_count}/"
                    f"{eligible_total} eligible opportunities "
                    f"({rate_text}) reached the target"
                )
                if durations:
                    duration_list = ", ".join(str(days) for days in durations)
                    average_days = sum(durations) / len(durations)
                    print(f"    Calendar days to target (ascending): {duration_list}")
                    print(f"    Average calendar days to target: {average_days:.1f}")
                else:
                    print("    Calendar days to target (ascending): None")
                    print("    Average calendar days to target: N/A (no successes)")



    # If the user wants to see brewing opportunities in the terminal 
    elif future_mode and future_mode_terminal_ouput:
        results_by_ticker = {}
        for ticker in tickers:
            try:
                # Build the dataframe 
                df = build_dataframe(ticker, period=future_mode_period, interval=interval)
                # Identify opportunities 
                opportunities, future_opportunity = identify_opportunities(
                    df, required_slump_length=required_slump_length
                )
                # Store the results of all the analysis functions 
                results_by_ticker[ticker] = {
                    "dataframe": df,
                    "opportunities": opportunities,
                    "future_opportunity": future_opportunity,
                }
                # Print successes 
                if future_opportunity:  # If it's not still {}
                    print_ticker_header(ticker)
                    print_future_opportunity(future_opportunity)
 
            except Exception as exc:
                # print(f"Could not analyze {ticker}: {exc}")
                pass 
        # If the user wants to see developing opportunities



    # If the user wants to see brewing opportunities written to a file  
    elif future_mode and future_mode_file_ouput:
        output_file = DATA_PATH / "future_opportunities.csv"

        ##### Write everything to the file. One line per stock. 
        with output_file.open("w", newline="") as file:
            writer = csv.writer(file)
            # Write the CSV header
            writer.writerow(["ticker", "slump began", "opportunity date"])

            for ticker in tickers:
                try:
                    # Build the dataframe
                    df = build_dataframe(
                        ticker,
                        period=future_mode_period,
                        interval=interval,
                    )
                    # Identify opportunities
                    opportunities, future_opportunity = identify_opportunities(
                        df,
                        required_slump_length=required_slump_length,
                    )
                    if future_opportunity:
                        # An unfinished slump exists; record both dates
                        slump_began = future_opportunity["date_slump_began"]
                        opportunity_date = future_opportunity["opportunity_date"]

                        slump_began = slump_began.date().isoformat()
                        opportunity_date = opportunity_date.date().isoformat()
                    else:
                        # No unfinished slump
                        slump_began = "None"
                        opportunity_date = "None"
                    # Write one row per successfully analyzed ticker
                    writer.writerow([
                        ticker,
                        slump_began,
                        opportunity_date,
                    ])

                except Exception as exc:
                    print(f"Could not analyze {ticker}: {exc}")

        print(f"Future opportunities saved to: {output_file}") 

        ##### Optionally, produce a second, cleaned and sorted file
        if sorted_and_cleaned:
            sorted_output_file = DATA_PATH / "opportunities_sorted.csv"
            # Read the complete file, treating "None" as missing
            opportunities_df = pd.read_csv(
                output_file,
                na_values=["None", "none", ""],
            )
            # Remove rows where both dates are missing
            opportunities_df = opportunities_df.dropna(
                subset=["slump began", "opportunity date"],
                how="all",
            )
            # Parse dates and sort earliest slump first
            opportunities_df["slump began"] = pd.to_datetime(
                opportunities_df["slump began"],
                errors="coerce",
            )
            opportunities_df["opportunity date"] = pd.to_datetime(
                opportunities_df["opportunity date"],
                errors="coerce",
            )
            opportunities_df = opportunities_df.sort_values(
                by="slump began",
                ascending=True,
            )
            # Save the cleaned file
            opportunities_df.to_csv(
                sorted_output_file,
                index=False,
                date_format="%Y-%m-%d",
            )
            print(f"Sorted opportunities saved to: {sorted_output_file}")
            print(f"Remaining opportunities: {len(opportunities_df)}")

        

##### End of main


if __name__ == "__main__":
    main()
