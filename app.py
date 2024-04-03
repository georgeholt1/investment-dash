import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url


def compound_interest(
    initial_amount: float, interest_rate: float, periods: int, contributions: float
):
    """Calculate compound interest with contributions.

    Assumes additional contributions are made at the start of each
    period.

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
    periods = np.arange(periods + 1)

    p = initial_amount
    r = interest_rate
    c = contributions

    d = r / (1 + r)  # discount rate

    f = (p + c / d) * (1 + r) ** periods - c / d

    return periods, f


def investment_evolution_breakdown(
    initial_amount: float, interest_rate: float, periods: int, contributions: float
) -> pd.DataFrame:
    """Evolution of investment over time by contribution.

    Calculates the total value of investment over a given period broken
    down into the amount due to additional contributions and the amount
    due to interest.

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
            - `value_from_contributions` : portion of total investment
              value due to additional contributions
            - `value_from_interest` : portion of total investment value
              due to interest
    """
    periods_list, investment_value = compound_interest(
        initial_amount, interest_rate, periods, contributions
    )

    investment_contributions = contributions * periods_list + initial_amount

    amount_from_interest = investment_value - investment_contributions

    df = pd.DataFrame(
        {
            "period": periods_list,
            "value": investment_value,
            "value_from_contributions": investment_contributions,
            "value_from_interest": amount_from_interest,
        }
    )

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
    df["year"] = df["period"] / 12

    # Adding a 'year_month' column for tooltip display, assuming
    # 'period' starts from 1 for the first month.
    df["year_month"] = df["period"].apply(
        lambda x: f"Year {x // 12}, Month {x % 12 if x % 12 != 0 else 12}"
    )

    marker_sizes = [0] * (len(df) - 1) + [10]

    fig = go.Figure()

    # Main balance
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["value"],
            mode="lines+markers",
            marker=dict(size=marker_sizes, color=COLOR_CONTRIBUTIONS, symbol="circle"),
            line=dict(color=COLOR_CONTRIBUTIONS),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            name="Balance",
            mode="lines+markers",
            marker=dict(color=COLOR_CONTRIBUTIONS, symbol="circle"),
            line=dict(color=COLOR_CONTRIBUTIONS),
            showlegend=True,
        )
    )

    # Dummy plot for hover text
    if not show_breakdown:
        hovertext = df.apply(
            lambda x: f"<b>{x['year_month']}</b><br>Balance: {x['value']:,.2f}", axis=1
        )
    else:
        hovertext = df.apply(
            lambda x: f"<b>{x['year_month']}</b><br>Balance: {x['value']:,.2f}<br>Total contributions: {x['value_from_contributions']:,.2f}<br>Total interest: {x['value_from_interest']:,.2f}",
            axis=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["value"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hovertext=hovertext,
            hoverinfo="text",
        )
    )

    if show_breakdown:
        # Principal
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["value_from_contributions"],
                mode="lines+markers",
                marker=dict(size=marker_sizes, symbol="square", color=COLOR_INITIAL),
                line=dict(color=COLOR_INITIAL),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name="Principal",
                mode="lines+markers",
                marker=dict(color=COLOR_INITIAL, symbol="square"),
                line=dict(color=COLOR_INITIAL),
                showlegend=True,
            )
        )

        # Interest
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["value_from_interest"],
                mode="lines+markers",
                marker=dict(size=marker_sizes, symbol="diamond", color=COLOR_INTEREST),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name="Interest",
                mode="lines+markers",
                marker=dict(color=COLOR_INTEREST, symbol="diamond"),
                line=dict(color=COLOR_INTEREST),
                showlegend=True,
            )
        )

    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Investment value")

    fig.update_layout(
        hovermode="x unified",
        xaxis_range=(0, df["year"].max() * 1.05),
        margin=dict(t=10, b=10),
    )

    return fig


def plot_pie_chart(df: pd.DataFrame, percentage: bool = False):
    """Plotly pie chart of contributions to final investment.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing investment value of time. The data in the
        last row is used for the amount due to contributions and
        interest and the data in the first row is used for the initial
        principal.
    percentage : bool, optional
        Whether to display data as percentage. Defaults to False, which
        displays the actual values.
    """
    labels = ["Initial principal", "Contributions", "Interest"]
    values = [
        df.iloc[0]["value"],
        df.iloc[-1]["value_from_contributions"],
        df.iloc[-1]["value_from_interest"],
    ]

    fig = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values,
            sort=False,
            marker=dict(
                colors=(
                    "#AB63FA",
                    COLOR_INITIAL,
                    COLOR_INTEREST,
                )
            ),
        )
    )

    if percentage:
        fig.update_traces(
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:,.2f}<extra></extra>",
        )

    else:
        fig.update_traces(texttemplate="%{value:,.2f}")
        fig.update_traces(
            textinfo="value",
            hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>",
        )

    fig.update_layout(margin=dict(t=10, b=10))

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
    CONTRIBUTIONS_DEFAULT,
)
DF_DEFAULT["years"] = DF_DEFAULT["period"] / 12

# Global colors for plots
COLOR_CONTRIBUTIONS = "#636EFA"
COLOR_INTEREST = "#EF553B"
COLOR_INITIAL = "#00CC96"


# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
)


# App components
# --------------

header = html.H4(
    "Investment return calculator",
    className="bg-primary text-white p-3 mb-2 text-center",
)

intro_text = """
Use this tool to calculate and visualise the future value of an
investment, including regular contributions, over a given period of
time. This can be used, for example, to calculate the monthly
contributions necessary to reach an investment goal over a number of
years with a known average rate of return. The calculations are updated
live as the input values are changed. Note that currency is not
specified so that the calculations are valid for all currencies.
"""

intro = html.P(
    intro_text,
    className="text-center mb-2",
)

# Principal
principal_min = 0
principal_step = 0.01

initial_amount_component = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.Input(
                    id="input-initial-amount",
                    type="number",
                    placeholder="Initial amount",
                    value=INITIAL_AMOUNT_DEFAULT,
                    min=principal_min,
                    step=principal_step,
                    style=dict(width="50%"),
                ),
            ]
        )
    ],
)

# Rate of return
rate_of_return_min = 0
rate_of_return_max = 100
rate_of_return_step = 0.01

rate_of_return_input_component = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.Input(
                    id="input-rate-of-return",
                    type="number",
                    placeholder="Rate of return",
                    value=INTEREST_RATE_PERCENT_DEFAULT,
                    min=rate_of_return_min,
                    max=rate_of_return_max,
                    step=rate_of_return_step,
                    style=dict(width="50%"),
                ),
                dbc.InputGroupText("%"),
            ]
        )
    ],
)

# Period
investment_period_min = 1
investment_period_max = 100
investment_period_step = 1

investment_period_input_component = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.Input(
                    id="input-investment-period",
                    type="number",
                    placeholder="Investment period",
                    value=PERIODS_YEARS_DEFAULT,
                    min=investment_period_min,
                    max=investment_period_max,
                    step=investment_period_step,
                    style=dict(width="50%"),
                ),
                dbc.InputGroupText("years"),
            ]
        )
    ],
)

# Additional contributions
contributions_min = 0
contributions_step = 0.01

contributions_component = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.Input(
                    id="input-contributions",
                    type="number",
                    placeholder="Contributions",
                    value=CONTRIBUTIONS_DEFAULT,
                    min=contributions_min,
                    step=contributions_step,
                    style=dict(width="50%"),
                ),
                dbc.InputGroupText("per month"),
            ]
        )
    ],
)

# Final value
results_component = html.P(id="results-text", className="card-text")

# Line graph
graph_component = html.Div(
    [
        dcc.Graph(
            id="graph", figure=plot_line_graph(DF_DEFAULT), style={"height": "40vh"}
        )
    ]
)

breakdown_checklist_component = html.Div(
    [
        dcc.Checklist(
            options=[{"label": "Show breakdown", "value": "show-breakdown"}],
            value=["show-breakdown"],
            id="checklist-breakdown",
        )
    ]
)

# Pie chart
pie_chart_component = html.Div(
    [
        dcc.Graph(
            id="pie-chart", figure=plot_pie_chart(DF_DEFAULT), style={"height": "40vh"}
        )
    ]
)

pie_chart_percentage_component = html.Div(
    [
        dcc.Checklist(
            options=[{"label": "Show percentages", "value": "percentage"}],
            id="checklist-percentage",
        )
    ]
)


# Cards
# -----

settings_info_text = """
Adjust the variables of the investment here. If an invalid value is
entered, hover over the resulting warning for a description of what
values are accepted.
"""
controls_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Settings", className="card-title"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(settings_info_text, className="card-text"),
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Principal")]),
                                                dbc.Col([initial_amount_component]),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Rate of return")]),
                                                dbc.Col(
                                                    [rate_of_return_input_component]
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Period")]),
                                                dbc.Col(
                                                    [investment_period_input_component]
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Contributions")]),
                                                dbc.Col([contributions_component]),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Results", className="card-title"
                                                ),
                                                results_component,
                                            ]
                                        )
                                    ],
                                    outline=True,
                                    color="primary",
                                    className="ms-5 me-5",
                                )
                            ],
                            width=5,
                        ),
                    ]
                ),
            ]
        )
    ],
    color="light",
    className="mb-1 mt-0",
)

controls_and_results_cards = controls_card

graph_info = """
The investment value over time and its components due to the initial
principal, additional contributions and interest earned are displayed
here. The line graph shows the values over time, while the pie chart
shows the breakdown of the final balance. Hover over the plots for
additional information.
"""

graph_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Graphs", className="card-title"),
                html.P(graph_info, className="card-text"),
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row([breakdown_checklist_component]),
                                        dbc.Row([graph_component]),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row([pie_chart_percentage_component]),
                                        dbc.Row([pie_chart_component]),
                                    ]
                                ),
                            ]
                        )
                    ]
                ),
            ]
        )
    ],
    className="mb-1",
)


# App
# ---

app.layout = html.Div(
    [
        dbc.Container(
            [header, intro, controls_and_results_cards, graph_card],
            fluid=True,
            className="dbc",
        )
    ],
    style={"marginLeft": "80px", "marginRight": "80px"},
)


# Callbacks
# ---------


@app.callback(
    Output("graph", "figure"),
    Output("pie-chart", "figure"),
    Output("results-text", "children"),
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
    show_breakdown = "show-breakdown" in breakdown if breakdown is not None else False
    percentage = "percentage" in percentage if percentage is not None else False

    # Calculate investment
    df = investment_evolution_breakdown(
        initial_amount,
        rate_of_return * 0.01 / 12,  # convert from percentage
        investment_period * 12,  # convert to months
        contributions,
    )
    final_value = df.iloc[-1]["value"]
    total_contributions = df.iloc[-1]["value_from_contributions"]
    total_interest = df.iloc[-1]["value_from_interest"]

    # Update line plot
    fig_line_graph = plot_line_graph(df, show_breakdown)

    # Update pie chart
    fig_pie_chart = plot_pie_chart(df, percentage)

    # Text element
    final_value_message = [
        html.Div("Values after {} years".format(investment_period)),
        html.Br(),
        html.Div(
            [
                html.Span("Final balance: ", style={"float": "left"}),
                html.Span(
                    "{:,.2f}".format(final_value),
                    style={"float": "right", "clear": "right"},
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                html.Span("Total contributions: ", style={"float": "left"}),
                html.Span(
                    "{:,.2f}".format(total_contributions),
                    style={"float": "right", "clear": "right"},
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                html.Span("Total interest: ", style={"float": "left"}),
                html.Span(
                    "{:,.2f}".format(total_interest),
                    style={"float": "right", "clear": "right"},
                ),
            ]
        ),
    ]

    return fig_line_graph, fig_pie_chart, final_value_message


if __name__ == "__main__":
    app.run_server(debug=True)
