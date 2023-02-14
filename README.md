# algostrats

algostrats is a package containing various utilities for training agents on financial data.
data is imported via the Yahoo finance utility (yfinance) chosen from a user-provided list
each day k stocks having greatest intraday gap are chosen from this list and tracked 

strategies are coded as agents operating in a custom AI agent architecture wherein world
supplied prices and technical indicators to the agent that needs to respond with an action
of buy, sell or do nothing, as well as stop-loss and target-profit percentages

once a position is taken a reward is returned to the agent once a stop or target has been reached
or the day ends.

included are RL agents and meta-RL agents that operated within the above archicture and train
using the stable baselines 3 library

documentation is available at https://algostrats.readthedocs.io/
