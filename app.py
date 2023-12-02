import dash
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np


def compound_interest(
        initial_amount: float,
        interest_rate: float,
        periods : int,
        contributions : float
    ) -> (np.ndarray, np.ndarray):
    """Calculate compound interest with contributions."""
    periods = np.arange(periods)
    
    p = initial_amount
    r = interest_rate
    c = contributions

    f = (p + c / r) * (1 + r) ** periods - c / r

    return periods, f


INITIAL_AMOUNT_DEFAULT = 1000      # default principal
INTEREST_RATE_PERCENT_DEFAULT = 6          # default annual interest, percent
INTEREST_RATE_MONTHLY_DEFAULT = INTEREST_RATE_PERCENT_DEFAULT * 0.01 / 12
PERIODS_YEARS_DEFAULT = 10              # default investment period, years
PERIODS_MONTHS_DEFAULT = PERIODS_YEARS_DEFAULT * 12
CONTRIBUTIONS_DEFAULT = 0          # default additional contributions per period

T_DEFAULT, AMOUNT_DEFAULT = compound_interest(
    INITIAL_AMOUNT_DEFAULT,
    INTEREST_RATE_MONTHLY_DEFAULT,
    PERIODS_MONTHS_DEFAULT,
    CONTRIBUTIONS_DEFAULT
)
T_DEFAULT_YEARS = T_DEFAULT / 12


# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ]
)


# App components
# --------------

header = html.H4(
    "Investment return calculator",
    className="bg-primary text-white p-3 mb-2 text-center"
)

# Principal
initial_amount_component = html.Div(
    [
        dcc.Input(
            id="input-initial-amount",
            type="number",
            placeholder="Initial amount",
            value=INITIAL_AMOUNT_DEFAULT
        )
    ]
)

# Rate of return
rate_of_return_min = 0
rate_of_return_max = 100
rate_of_return_step = 0.1

rate_of_return_input_component = html.Div(
    [
        dcc.Input(
            id="input-rate-of-return",
            type="number",
            placeholder="Rate of return",
            value=INTEREST_RATE_PERCENT_DEFAULT,
            min=rate_of_return_min,
            max=rate_of_return_max,
            step=rate_of_return_step,
        )
    ]
)

# Period
investment_period_min = 1
investment_period_max = 100
investment_period_step = 1

investment_period_input_component = html.Div(
    [
        dcc.Input(
            id="input-investment-period",
            type="number",
            placeholder="Investment period",
            value=PERIODS_YEARS_DEFAULT,
            min=investment_period_min,
            max=investment_period_max,
            step=investment_period_step
        )
    ]
)

contributions_component = html.Div(
    [
        dcc.Input(
            id="input-contributions",
            type="number",
            placeholder="Contributions",
            value=CONTRIBUTIONS_DEFAULT
        )
    ]
)

graph_component = html.Div(
    [
        dcc.Graph(
            id="graph",
            figure=px.line(
                x=T_DEFAULT_YEARS,
                y=AMOUNT_DEFAULT,
                # template="bootstrap"
            )
        )
    ]
)



# Cards
# -----

controls_card = dbc.Card(
    [
        html.H4("Settings", className="card-title"),
        dbc.Container([
            dbc.Row([
                dbc.Col([dbc.Label("Principal")]),
                dbc.Col([initial_amount_component])
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Rate of return")]),
                dbc.Col([rate_of_return_input_component]),
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Period (years)")]),
                dbc.Col([investment_period_input_component]),
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Contributions")]),
                dbc.Col([contributions_component])
            ])
        ])
    ]
)

graph_card = dbc.Card(
    [
        dbc.Container([
            dbc.Row([graph_component])
        ])
    ]
)


# App
# ---

app.layout = dbc.Container(
    [
        header,
        controls_card,
        graph_card
    ],
    fluid=True,
    className="dbc"
)


# Callbacks
# ---------

@app.callback(
    Output("graph", "figure"),
    Input("input-initial-amount", "value"),
    Input("input-rate-of-return", "value"),
    Input("input-investment-period", "value"),
    Input("input-contributions", "value")
)
def update(
    initial_amount,
    rate_of_return,
    investment_period,
    contributions
):

    # Update line plot
    t, amount = compound_interest(
        initial_amount,
        rate_of_return*0.01/12, # convert from percentage
        investment_period*12,   # convert to months
        contributions
    )
    t_years = t / 12
    fig = px.line(
        x=t_years,
        y=amount
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)







