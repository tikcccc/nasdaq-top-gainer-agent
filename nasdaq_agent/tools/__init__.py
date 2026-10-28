"""The agent's tools. Each is a LangChain tool returning a uniform envelope (see base.py)."""

from nasdaq_agent.tools.code_exec import execute_python
from nasdaq_agent.tools.emailer import send_email
from nasdaq_agent.tools.gainers import get_top_nasdaq_gainers
from nasdaq_agent.tools.history import get_price_history
from nasdaq_agent.tools.news import get_news
from nasdaq_agent.tools.plan import record_plan
from nasdaq_agent.tools.sentiment import analyze_sentiment

ALL_TOOLS = [
    record_plan,
    get_top_nasdaq_gainers,
    get_price_history,
    execute_python,
    get_news,
    analyze_sentiment,
    send_email,
]
