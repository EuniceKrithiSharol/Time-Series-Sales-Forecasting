import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Time Series Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# -------------------------------------------------
# CREATE SAMPLE SALES DATA
# -------------------------------------------------

@st.cache_data
def create_sales_dataset():

    np.random.seed(42)

    dates = pd.date_range(
        start="2024-01-01",
        periods=365,
        freq="D"
    )


    trend = np.linspace(
        1000,
        3000,
        365
    )


    seasonality = (

        400
        *
        np.sin(
            np.linspace(
                0,
                12 * np.pi,
                365
            )
        )
    )


    noise = np.random.normal(
        0,
        180,
        365
    )


    sales = trend + seasonality + noise


    sales = np.maximum(
        sales,
        100
    )


    data = pd.DataFrame({

        "Date": dates,

        "Sales": sales.round(2)
    })


    return data


sales_data = create_sales_dataset()


# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------

def prepare_data(data):

    prepared_data = data.copy()


    prepared_data["Day_Number"] = np.arange(
        len(
            prepared_data
        )
    )


    return prepared_data


prepared_sales_data = prepare_data(
    sales_data
)


# -------------------------------------------------
# TRAIN FORECASTING MODEL
# -------------------------------------------------

@st.cache_resource
def train_forecasting_model(data):

    X = data[

        ["Day_Number"]

    ]


    y = data[

        "Sales"

    ]


    model = LinearRegression()


    model.fit(

        X,

        y
    )


    return model


model = train_forecasting_model(
    prepared_sales_data
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title(
    "📈 Time Series Sales Forecasting System"
)


st.markdown(
    "Analyze historical sales data and use Machine Learning "
    "to predict future sales trends."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header(
    "🤖 How It Works"
)


st.sidebar.info(
    """
    1. Historical sales data is collected.

    2. Dates are converted into numerical time features.

    3. A Machine Learning model learns the sales trend.

    4. Future time periods are generated.

    5. The system predicts future sales.
    """
)


# -------------------------------------------------
# DASHBOARD METRICS
# -------------------------------------------------

st.subheader(
    "📊 Sales Overview"
)


total_sales = sales_data[
    "Sales"
].sum()


average_sales = sales_data[
    "Sales"
].mean()


highest_sales = sales_data[
    "Sales"
].max()


latest_sales = sales_data[
    "Sales"
].iloc[-1]


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)


col2.metric(
    "Average Daily Sales",
    f"${average_sales:,.0f}"
)


col3.metric(
    "Highest Daily Sales",
    f"${highest_sales:,.0f}"
)


col4.metric(
    "Latest Sales",
    f"${latest_sales:,.0f}"
)


# -------------------------------------------------
# SALES DATA
# -------------------------------------------------

st.subheader(
    "📁 Historical Sales Data"
)


st.dataframe(
    sales_data.tail(30),
    use_container_width=True
)


# -------------------------------------------------
# HISTORICAL SALES VISUALIZATION
# -------------------------------------------------

st.subheader(
    "📈 Historical Sales Trend"
)


fig_sales = px.line(

    sales_data,

    x="Date",

    y="Sales",

    title="Historical Daily Sales"
)


st.plotly_chart(
    fig_sales,
    use_container_width=True
)


# -------------------------------------------------
# MONTHLY SALES ANALYSIS
# -------------------------------------------------

st.subheader(
    "📊 Monthly Sales Analysis"
)


monthly_sales = (

    sales_data
    .set_index(
        "Date"
    )
    .resample(
        "ME"
    )
    .sum()
    .reset_index()
)


fig_monthly = px.bar(

    monthly_sales,

    x="Date",

    y="Sales",

    title="Monthly Sales Performance"
)


st.plotly_chart(
    fig_monthly,
    use_container_width=True
)


# -------------------------------------------------
# FORECAST SETTINGS
# -------------------------------------------------

st.divider()


st.header(
    "🔮 Forecast Future Sales"
)


forecast_days = st.slider(

    "Select Forecast Period (Days)",

    min_value=7,

    max_value=180,

    value=30
)


# -------------------------------------------------
# GENERATE FORECAST
# -------------------------------------------------

if st.button(
    "📈 Generate Sales Forecast"
):

    future_day_numbers = np.arange(

        len(
            prepared_sales_data
        ),

        len(
            prepared_sales_data
        )
        +
        forecast_days
    )


    future_sales = model.predict(

        future_day_numbers.reshape(
            -1,
            1
        )
    )


    future_dates = pd.date_range(

        start=sales_data[
            "Date"
        ].iloc[-1]
        +
        pd.Timedelta(
            days=1
        ),

        periods=forecast_days,

        freq="D"
    )


    forecast_data = pd.DataFrame({

        "Date": future_dates,

        "Predicted_Sales": future_sales.round(2)
    })


    st.subheader(
        "📊 Sales Forecast Results"
    )


    total_forecast = forecast_data[
        "Predicted_Sales"
    ].sum()


    average_forecast = forecast_data[
        "Predicted_Sales"
    ].mean()


    col1, col2 = st.columns(2)


    col1.metric(

        "Predicted Total Sales",

        f"${total_forecast:,.0f}"
    )


    col2.metric(

        "Predicted Average Daily Sales",

        f"${average_forecast:,.0f}"
    )


    st.dataframe(

        forecast_data,

        use_container_width=True
    )


    # ---------------------------------------------
    # COMBINED FORECAST GRAPH
    # ---------------------------------------------

    st.subheader(
        "📈 Historical vs Forecasted Sales"
    )


    fig_forecast = go.Figure()


    fig_forecast.add_trace(

        go.Scatter(

            x=sales_data[
                "Date"
            ],

            y=sales_data[
                "Sales"
            ],

            mode="lines",

            name="Historical Sales"
        )
    )


    fig_forecast.add_trace(

        go.Scatter(

            x=forecast_data[
                "Date"
            ],

            y=forecast_data[
                "Predicted_Sales"
            ],

            mode="lines+markers",

            name="Forecasted Sales"
        )
    )


    fig_forecast.update_layout(

        title="Sales Forecast",

        xaxis_title="Date",

        yaxis_title="Sales"
    )


    st.plotly_chart(

        fig_forecast,

        use_container_width=True
    )


# -------------------------------------------------
# CUSTOM CSV UPLOAD
# -------------------------------------------------

st.divider()


st.header(
    "📤 Upload Historical Sales Data"
)


uploaded_file = st.file_uploader(

    "Upload a CSV containing Date and Sales columns",

    type=["csv"]
)


if uploaded_file is not None:

    uploaded_data = pd.read_csv(
        uploaded_file
    )


    if (

        "Date" not in uploaded_data.columns

        or

        "Sales" not in uploaded_data.columns

    ):

        st.error(
            "The CSV must contain Date and Sales columns."
        )


    else:

        uploaded_data[
            "Date"
        ] = pd.to_datetime(

            uploaded_data[
                "Date"
            ]
        )


        uploaded_data = uploaded_data.sort_values(

            "Date"
        )


        uploaded_data[
            "Day_Number"
        ] = np.arange(

            len(
                uploaded_data
            )
        )


        uploaded_model = LinearRegression()


        uploaded_model.fit(

            uploaded_data[
                ["Day_Number"]
            ],

            uploaded_data[
                "Sales"
            ]
        )


        st.success(
            "Historical sales data successfully loaded."
        )


        st.dataframe(

            uploaded_data,

            use_container_width=True
        )


        uploaded_forecast_days = st.slider(

            "Forecast Days for Uploaded Dataset",

            min_value=7,

            max_value=180,

            value=30
        )


        if st.button(
            "Generate Forecast for Uploaded Data"
        ):

            future_days = np.arange(

                len(
                    uploaded_data
                ),

                len(
                    uploaded_data
                )
                +
                uploaded_forecast_days
            )


            future_predictions = uploaded_model.predict(

                future_days.reshape(
                    -1,
                    1
                )
            )


            future_dates = pd.date_range(

                start=uploaded_data[
                    "Date"
                ].iloc[-1]
                +
                pd.Timedelta(
                    days=1
                ),

                periods=uploaded_forecast_days,

                freq="D"
            )


            uploaded_forecast = pd.DataFrame({

                "Date": future_dates,

                "Predicted_Sales": future_predictions.round(2)
            })


            st.subheader(
                "Uploaded Dataset Forecast"
            )


            st.dataframe(

                uploaded_forecast,

                use_container_width=True
            )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()


st.caption(
    "Time Series Sales Forecasting System | "
    "Python • Machine Learning • "
    "Time Series • Predictive Analytics • "
    "Linear Regression • Streamlit"
)
