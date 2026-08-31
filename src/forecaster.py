import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def prepare_time_series(data):

    prepared_data = data.copy()


    prepared_data["Date"] = pd.to_datetime(

        prepared_data["Date"]
    )


    prepared_data = prepared_data.sort_values(

        "Date"
    )


    prepared_data["Day_Number"] = np.arange(

        len(
            prepared_data
        )
    )


    return prepared_data


def train_forecasting_model(data):

    model = LinearRegression()


    model.fit(

        data[
            ["Day_Number"]
        ],

        data[
            "Sales"
        ]
    )


    return model


def forecast_sales(

    data,

    model,

    forecast_days
):

    future_days = np.arange(

        len(
            data
        ),

        len(
            data
        )
        +
        forecast_days
    )


    predictions = model.predict(

        future_days.reshape(
            -1,
            1
        )
    )


    future_dates = pd.date_range(

        start=data[
            "Date"
        ].iloc[-1]
        +
        pd.Timedelta(
            days=1
        ),

        periods=forecast_days,

        freq="D"
    )


    forecast = pd.DataFrame({

        "Date": future_dates,

        "Predicted_Sales": predictions
    })


    return forecast
