from datetime import datetime, timedelta

def Is_balanced(current_portfolio_df):
    largest_pct_off_by = 0
    for cluster, row in current_portfolio_df.iterrows():
        this_cluster_off_by = current_portfolio_df.loc[cluster, "Pct Off Op"]
        if abs(this_cluster_off_by) > abs(largest_pct_off_by):
            largest_pct_off_by = this_cluster_off_by
    return abs(largest_pct_off_by) < 0.03 #changed threshold


def calculate_seconds_till_next_reallocation(timezone, hour_to_trade, minute_to_trade):
                now = datetime.now(timezone)
                target_time = now.replace(hour=hour_to_trade, minute=minute_to_trade, second=0, microsecond=0)
                if now > target_time:
                    target_time += timedelta(days=1)
                return int((target_time - now).total_seconds())


def main():
    pass
if __name__ == "__main__":
    main()
