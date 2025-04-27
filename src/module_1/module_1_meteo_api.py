import requests
import pandas as pd
import matplotlib.pyplot as plt


API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def call_api(params: dict):
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching meteo data: {e}")
        return None


def get_data_meteo_api(city: str):
    cords = COORDINATES[city]

    # API doc: dict "params"  in a defined order to assign data correctly
    params = {
        "latitude": cords["latitude"],
        "longitude": cords["longitude"],
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "daily": VARIABLES,
    }

    raw_data = call_api(params=params)
    # Once we have the data -> dataframe
    daily_data = raw_data["daily"]

    df = pd.DataFrame(daily_data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.index.name = "date"

    return df


# Son demasiadas muestras, por lo que decido hacer la media mensual
def get_mean_data_monthly(df: pd.DataFrame, freq="MS"):
    # Index has to be datetime type to sample date
    df.index = pd.to_datetime(df.index)
    df_sampled = pd.DataFrame()

    # Calculate mean by period of all measures
    df_sampled = df.resample(freq).mean(numeric_only=True)

    return df_sampled


def plot_meteo_data(all_data: dict):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

    # Plot temp
    for city, city_df in all_data.items():
        axs[0].plot(city_df.index, city_df["temperature_2m_mean"], label=city)
        axs[0].set_title("Max Monthly Temperature")
        axs[0].set_ylabel("°C")
        axs[0].legend()

    # Plot precip
    for city, city_df in all_data.items():
        axs[1].plot(city_df.index, city_df["precipitation_sum"], label=city)
        axs[1].set_title("Monthly Precipitation")
        axs[1].set_ylabel("mm")
        axs[1].legend()

    # Plot windspeed
    for city, city_df in all_data.items():
        axs[2].plot(city_df.index, city_df["wind_speed_10m_max"], label=city)
        axs[2].set_title("Max Monthly WindSpeed")
        axs[2].set_ylabel("km/h")
        axs[2].legend()

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def main():
    all_data = {}

    # 1. Call API for each city
    for city in COORDINATES.keys():
        print(f"Obteniendo datos para {city}...")
        daily_df = get_data_meteo_api(city)
        if daily_df is not None:
            # 2. Monthly mean
            monthly_df = get_mean_data_monthly(daily_df)
            all_data[city] = monthly_df

    # 3. Plotting all data
    plot_meteo_data(all_data)


if __name__ == "__main__":
    main()
