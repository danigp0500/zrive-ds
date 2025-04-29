import requests
import pandas as pd
import matplotlib.pyplot as plt
import time


API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def call_api(URL: str, params: dict):
    for attempt in range(1, 4):
        try:
            response = requests.get(url=URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Attempt {attempt}: Received status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}:Error fetching data from API: {e}")
            return

        time.sleep(2)

    raise RuntimeError("Failed to fetch data after 3 attempts.")


def get_data_meteo_api(URL: str, city: str):
    params = {
        **COORDINATES[city],
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "daily": VARIABLES,
    }
    print(params)

    data_json = call_api(URL, params)

    daily_data = data_json["daily"]
    df = pd.DataFrame(daily_data)

    # Needed to convert to datetime type to be able to do mean
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    # print(df)
    return df


# Son demasiadas muestras, por lo que decido hacer la media mensual
def get_mean_data_monthly(df: pd.DataFrame):
    df_monthly = pd.DataFrame(df.resample("MS").mean())
    # print(df_monthly)

    return df_monthly


def plot_meteo_data(data: dict):

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

    # Plot temp
    for city, df in data.items():
        axs[0].plot(df.index, df["temperature_2m_mean"], label=city)
        axs[0].set_title("Max Monthly Temperature")
        axs[0].set_ylabel("°C")
        axs[0].legend()
    # Plot precip
    for city, df in data.items():
        axs[1].plot(df.index, df["precipitation_sum"], label=city)
        axs[1].set_title("Monthly Precipitation")
        axs[1].set_ylabel("mm")
        axs[1].legend()
    # Plot windspeed
    for city, df in data.items():
        axs[2].plot(df.index, df["wind_speed_10m_max"], label=city)
        axs[2].set_title("Max Monthly WindSpeed")
        axs[2].set_ylabel("km/h")
        axs[2].legend()
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def main():
    data = {}
    # 1. Call API for each city
    for city in COORDINATES.keys():
        d = get_data_meteo_api(API_URL, city)
        if d is not None:
            m = get_mean_data_monthly(d)
            data[city] = m

    plot_meteo_data(data)


if __name__ == "__main__":
    main()
