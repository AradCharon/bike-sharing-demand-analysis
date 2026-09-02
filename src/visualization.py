from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of total hourly bike rentals.
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="cnt",
        bins=50,
        kde=True,
    )

    plt.title("Distribution of Hourly Bike Rental Demand")
    plt.xlabel("Total Rentals (`cnt`)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_hourly_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by hour of day.
    """
    hourly_demand = (
        df.groupby("hr", as_index=False)["cnt"]
        .mean()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=hourly_demand,
        x="hr",
        y="cnt",
        marker="o",
    )

    plt.title("Average Bike Rental Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rentals")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.show()


def plot_weekday_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by weekday.
    """
    weekday_demand = (
        df.groupby("weekday", as_index=False)["cnt"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=weekday_demand,
        x="weekday",
        y="cnt",
    )

    plt.title("Average Bike Rental Demand by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Rentals")
    plt.tight_layout()
    plt.show()


def plot_monthly_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by month.
    """
    monthly_demand = (
        df.groupby("mnth", as_index=False)["cnt"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=monthly_demand,
        x="mnth",
        y="cnt",
        marker="o",
    )

    plt.title("Average Bike Rental Demand by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Rentals")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.show()


def plot_yearly_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by year.
    """
    yearly_demand = (
        df.groupby("yr", as_index=False)["cnt"]
        .mean()
    )

    yearly_demand["year"] = yearly_demand["yr"].map({
        0: 2011,
        1: 2012,
    })

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=yearly_demand,
        x="year",
        y="cnt",
    )

    plt.title("Average Bike Rental Demand by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Rentals")
    plt.tight_layout()
    plt.show()


def plot_workingday_hourly_demand(df: pd.DataFrame) -> None:
    """
    Compare average hourly demand between working and non-working days.
    """
    workingday_hourly = (
        df.groupby(
            ["hr", "workingday"],
            as_index=False,
        )["cnt"]
        .mean()
    )

    workingday_hourly["day_type"] = (
        workingday_hourly["workingday"].map({
            0: "Non-working day",
            1: "Working day",
        })
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=workingday_hourly,
        x="hr",
        y="cnt",
        hue="day_type",
        marker="o",
    )

    plt.title(
        "Hourly Bike Rental Demand: "
        "Working vs. Non-Working Days"
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rentals")
    plt.xticks(range(24))
    plt.legend(title="Day Type")
    plt.tight_layout()
    plt.show()


def plot_seasonal_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by season.
    """
    season_demand = (
        df.groupby("season", as_index=False)["cnt"]
        .mean()
    )

    season_demand["season_name"] = (
        season_demand["season"].map({
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter",
        })
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=season_demand,
        x="season_name",
        y="cnt",
    )

    plt.title("Average Bike Rental Demand by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Rentals")
    plt.tight_layout()
    plt.show()


def plot_weather_demand(df: pd.DataFrame) -> None:
    """
    Plot average bike rental demand by weather situation.
    """
    weather_demand = (
        df.groupby("weathersit", as_index=False)["cnt"]
        .mean()
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=weather_demand,
        x="weathersit",
        y="cnt",
    )

    plt.title("Average Bike Rental Demand by Weather Situation")
    plt.xlabel("Weather Situation")
    plt.ylabel("Average Rentals")
    plt.tight_layout()
    plt.show()


def plot_temperature_vs_demand(df: pd.DataFrame) -> None:
    """
    Plot the relationship between temperature and demand.
    """
    sample = df.sample(
        min(5000, len(df)),
        random_state=42,
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=sample,
        x="temp",
        y="cnt",
        alpha=0.4,
    )

    plt.title("Temperature vs. Bike Rental Demand")
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Total Rentals")
    plt.tight_layout()
    plt.show()


def plot_humidity_vs_demand(df: pd.DataFrame) -> None:
    """
    Plot the relationship between humidity and demand.
    """
    sample = df.sample(
        min(5000, len(df)),
        random_state=42,
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=sample,
        x="hum",
        y="cnt",
        alpha=0.4,
    )

    plt.title("Humidity vs. Bike Rental Demand")
    plt.xlabel("Normalized Humidity")
    plt.ylabel("Total Rentals")
    plt.tight_layout()
    plt.show()


def plot_user_type_demand(df: pd.DataFrame) -> None:
    """
    Compare average hourly demand for casual and registered users.
    """
    user_hourly = (
        df.groupby("hr")[["casual", "registered"]]
        .mean()
        .reset_index()
    )

    user_hourly_long = user_hourly.melt(
        id_vars="hr",
        value_vars=["casual", "registered"],
        var_name="user_type",
        value_name="average_rentals",
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=user_hourly_long,
        x="hr",
        y="average_rentals",
        hue="user_type",
        marker="o",
    )

    plt.title("Average Hourly Demand by User Type")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rentals")
    plt.xticks(range(24))
    plt.legend(title="User Type")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap for selected numerical variables.
    """
    correlation_columns = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
    ]

    correlation_matrix = df[correlation_columns].corr()

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()