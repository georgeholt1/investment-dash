import dash
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def compound_interest(
        initial_amount: float,
        interest_rate: float,
        periods: int,
        contributions: float
    ) -> (np.ndarray, np.ndarray):
    """Calculate compound interest with contributions.

    Assumes additional contributions are made at the start of each period.

    Formula from:
    https://math.stackexchange.com/questions/1698578/compound-interest-formula-adding-annual-contributions
    
    Parameters
    ----------
    initial_amount : float
        Principal.
    interest_rate : float
        Interest rate expressed as a decimal per `periods`.
    periods : int
        Number of periods over which to accumulate interest.
    contributions : float
        Additional contributions made each investment period.
    """
    periods = np.arange(periods)
    
    p = initial_amount
    r = interest_rate
    c = contributions

    d = r / (1 + r)  # discount rate

    f = (p + c / d) * (1 + r) ** periods - c / d

    return periods, f


def investment_evolution_breakdown(
        initial_amount: float,
        interest_rate: float,
        periods: int,
        contributions: float
    ) -> pd.DataFrame:
    """Evolution of investment over time by contribution.
    
    Calculates the total value of investment over a given period broken down 
    into the amount due to additional contributions and the amount due to
    interest.

    Parameters
    ----------
    initial_amount : float
        Principal.
    interest_rate : float
        Interest rate expressed as a decimal per `periods`.
    periods : int
        Number of periods over which to accumulate interest.
    contributions : float
        Additional contributions made each investment period.

    Returns
    -------
    pandas DataFrame
        With columns:
            - `period` : investment period index
            - `value` : total investment value
            - `value_from_contributions` : portion of total investment value
              due to additional contributions
            - `value_from_interest` : portion of total investment value due
              to interest
    """
    periods_list, investment_value = compound_interest(
        initial_amount,
        interest_rate,
        periods,
        contributions
    )

    investment_contributions = contributions * np.arange(periods) + \
        initial_amount

    amount_from_interest = investment_value - investment_contributions

    df = pd.DataFrame({
        'period': periods_list,
        'value': investment_value,
        'value_from_contributions': investment_contributions,
        'value_from_interest': amount_from_interest
    })

    return df


def plot_line_graph(df: pd.DataFrame, show_breakdown: bool = False):
    """Plotly line graph of investment over time.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing investment value over time.
    show_breakdown : bool, optional
        Whether to show breakdown of total investment amount due to 
        contributions and interest. Defaults to False.
    """
    df['year'] = df['period'] / 12

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['value'],
        name='Balance',
        showlegend=True
    ))

    if show_breakdown:
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['value_from_contributions'],
            name='Principal'
        ))
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['value_from_interest'],
            name='Interest'
        ))
    
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Investment value')
    
    return fig


# Defaults
INITIAL_AMOUNT_DEFAULT = 1000  # default principal
INTEREST_RATE_PERCENT_DEFAULT = 6  # default annual interest, percent
INTEREST_RATE_MONTHLY_DEFAULT = INTEREST_RATE_PERCENT_DEFAULT * 0.01 / 12
PERIODS_YEARS_DEFAULT = 10  # default investment period, years
PERIODS_MONTHS_DEFAULT = PERIODS_YEARS_DEFAULT * 12
CONTRIBUTIONS_DEFAULT = 100  # default additional contributions per period

DF_DEFAULT = investment_evolution_breakdown(
    INITIAL_AMOUNT_DEFAULT,
    INTEREST_RATE_MONTHLY_DEFAULT,
    PERIODS_MONTHS_DEFAULT,
    CONTRIBUTIONS_DEFAULT
)
DF_DEFAULT['years']  = DF_DEFAULT['period'] / 12


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
principal_min = 0

initial_amount_component = html.Div(
    [
        dcc.Input(
            id="input-initial-amount",
            type="number",
            placeholder="Initial amount",
            value=INITIAL_AMOUNT_DEFAULT,
            min=principal_min
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

# Additional contributions
contributions_min = 0
contributions_component = html.Div(
    [
        dcc.Input(
            id="input-contributions",
            type="number",
            placeholder="Contributions",
            value=CONTRIBUTIONS_DEFAULT,
            min=contributions_min
        )
    ]
)

graph_component = html.Div(
    [
        dcc.Graph(
            id="graph",
            figure=plot_line_graph(DF_DEFAULT)
        )
    ]
)

breakdown_checklist_component = html.Div(
    [
        dcc.Checklist(
            options=[{'label': 'Show breakdown', 'value': 'show-breakdown'}],
            id='checklist-breakdown',
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
            dbc.Row([breakdown_checklist_component]),
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
    Input("input-contributions", "value"),
    Input("checklist-breakdown", "value")
)
def update(
    initial_amount,
    rate_of_return,
    investment_period,
    contributions,
    breakdown
):
    show_breakdown = 'show-breakdown' in breakdown if breakdown is not None else False
    # print(show_breakdown)

    # Update line plot
    df = investment_evolution_breakdown(
        initial_amount,
        rate_of_return*0.01/12, # convert from percentage
        investment_period*12,   # convert to months
        contributions
    )
    fig = plot_line_graph(df, show_breakdown)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)