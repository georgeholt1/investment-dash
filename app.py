import dash
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from dash.dependencies import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# Global colors for plots
#TODO make sure these are used everywhere needed
COLOR_CONTRIBUTIONS = '#636EFA'
COLOR_INTEREST = '#EF553B'
COLOR_INITIAL = '#00CC96'


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
    periods = np.arange(periods+1)
    
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

    investment_contributions = contributions * periods_list + \
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
    # Adding a 'year' column for plotting
    df['year'] = df['period'] / 12

    # Adding a 'year_month' column for tooltip display, assuming 'period' starts from 1 for the first month.
    df['year_month'] = df['period'].apply(lambda x: f"Year {x // 12}, Month {x % 12 if x % 12 != 0 else 12}")

    marker_sizes = [0] * (len(df) - 1) + [10]
    
    fig = go.Figure()

    # Main balancewith customized hover text
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['value'],
        mode='lines+markers',
        marker=dict(size=marker_sizes, color=COLOR_CONTRIBUTIONS, symbol='circle'),
        line=dict(color=COLOR_CONTRIBUTIONS),
        showlegend=False,
        hoverinfo='text',
        text=df.apply(lambda x: f"{x['year_month']}: {x['value']:,.2f}", axis=1),
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        name='Balance',
        mode='lines+markers',
        marker=dict(color=COLOR_CONTRIBUTIONS, symbol='circle'),
        line=dict(color=COLOR_CONTRIBUTIONS),
        showlegend=True,
    ))

    if show_breakdown:
        # Principal
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['value_from_contributions'],
            mode='lines+markers',
            marker=dict(size=marker_sizes, symbol='square', color=COLOR_INITIAL),
            line=dict(color=COLOR_INITIAL),
            showlegend=False,
            hoverinfo='text',
            text=df.apply(lambda x: f"{x['year_month']}: {x['value_from_contributions']:,.2f}", axis=1)
        ))
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            name='Principal',
            mode='lines+markers',
            marker=dict(color=COLOR_INITIAL, symbol='square'),
            line=dict(color=COLOR_INITIAL),
            showlegend=True
        ))

        # Interest
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['value_from_interest'],
            mode='lines+markers',
            marker=dict(size=marker_sizes, symbol='diamond', color=COLOR_INTEREST),
            showlegend=False,
            hoverinfo='text',
            text=df.apply(lambda x: f"{x['year_month']}: {x['value_from_interest']:,.2f}", axis=1)
        ))
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            name='Interest',
            mode='lines+markers',
            marker=dict(color=COLOR_INTEREST, symbol='diamond'),
            line=dict(color=COLOR_INTEREST),
            showlegend=True
        ))
    
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Investment value')

    fig.update_layout(
        hovermode='x unified',
        xaxis_range=(0, df['year'].max()*1.05)
    )
    
    return fig


def plot_pie_chart(df: pd.DataFrame, percentage: bool = False):
    """Plotly pie chart of contributions to final investment.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing investment value of time. The data in the last
        row is used for the amount due to contributions and interest and
        the data in the first row is used for the initial principal.
    percentage : bool, optional
        Whether to display data as percentage. Defaults to False, which displays
        the actual values.
    """
    #TODO Format to show 2 decimal places
    labels = [
        'Initial principal',
        'Contributions',
        'Interest'
    ]
    values = [
        df.iloc[0]['value'],
        df.iloc[-1]['value_from_contributions'],
        df.iloc[-1]['value_from_interest'],
    ]

    fig = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values
        )
    )

    if percentage:
        fig.update_traces(
            hoverinfo='label+value',
            textinfo='percent'
        )

    else:
        fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value'
        )

    return fig


# Defaults
INITIAL_AMOUNT_DEFAULT = 10000  # default principal
INTEREST_RATE_PERCENT_DEFAULT = 6  # default annual interest, percent
INTEREST_RATE_MONTHLY_DEFAULT = INTEREST_RATE_PERCENT_DEFAULT * 0.01 / 12
PERIODS_YEARS_DEFAULT = 30  # default investment period, years
PERIODS_MONTHS_DEFAULT = PERIODS_YEARS_DEFAULT * 12
CONTRIBUTIONS_DEFAULT = 1000  # default additional contributions per period

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

# Line graph
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
            value=['show-breakdown'],
            id='checklist-breakdown',
        )
    ]
)

# Pie chart
pie_chart_component = html.Div(
    [
        dcc.Graph(
            id="pie-chart",
            figure=plot_pie_chart(DF_DEFAULT)
        )
    ]
)

pie_chart_percentage_component = html.Div(
    [
        dcc.Checklist(
            options=[{'label': 'Percentage', 'value': 'percentage'}],
            id='checklist-percentage'
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
                dbc.Col([dbc.Label("Contributions (monthly)")]),
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

pie_card = dbc.Card(
    [
        dbc.Container([
            dbc.Row([pie_chart_percentage_component]),
            dbc.Row([pie_chart_component]),
        ])
    ]
)


# App
# ---

app.layout = dbc.Container(
    [
        header,
        controls_card,
        graph_card,
        pie_card
    ],
    fluid=True,
    className="dbc"
)


# Callbacks
# ---------

@app.callback(
    Output("graph", "figure"),
    Output("pie-chart", "figure"),
    Input("input-initial-amount", "value"),
    Input("input-rate-of-return", "value"),
    Input("input-investment-period", "value"),
    Input("input-contributions", "value"),
    Input("checklist-breakdown", "value"),
    Input("checklist-percentage", "value"),
)
def update(
    initial_amount,
    rate_of_return,
    investment_period,
    contributions,
    breakdown,
    percentage,
):
    # True if box is ticked
    show_breakdown = 'show-breakdown' in breakdown if breakdown is not None else False
    percentage = 'percentage' in percentage if percentage is not None else False

    # Update line plot
    df = investment_evolution_breakdown(
        initial_amount,
        rate_of_return*0.01/12, # convert from percentage
        investment_period*12,   # convert to months
        contributions
    )
    fig_line_graph = plot_line_graph(df, show_breakdown)

    # Update pie chart
    fig_pie_chart = plot_pie_chart(df, percentage)

    return fig_line_graph, fig_pie_chart


if __name__ == "__main__":
    app.run_server(debug=True)