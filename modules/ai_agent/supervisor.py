"""Supervisor: a manager agent that sends each question to the right worker agent."""
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from modules.ai_agent.graph import create_chat_model
from modules.ai_agent.tools import (
    fetch_menu,
    get_discounts,
    get_offer,
    log_complaint,
    validate_order_id,
)


def build_supervisor(llm="groq"):
    model = create_chat_model(llm)

    menu_agent = create_react_agent(
        model=model,
        tools=[fetch_menu],
        name="menu_agent",
        prompt="You answer menu and price questions. Prices are in PKR. Always use fetch_menu. Write prices like Rs. 850, never use the peso sign.",
    )
    complaint_agent = create_react_agent(
        model=model,
        tools=[validate_order_id, log_complaint],
        name="complaint_agent",
        prompt=("You handle complaints. Collect customer name, phone, order ID and complaint. "
                "Check the order ID with validate_order_id, then save with log_complaint. "
                "Ask for missing details. Never invent a complaint ID."),
    )

    customer_support_agent = create_react_agent(
            model=model,
            tools=[],
            name="customer_support_agent",
            prompt=("You have to collect customer feedback and complaints. "),
        )

    offer_agent = create_react_agent(
        model=model,
        tools=[get_offer, get_discounts],
        name="offer_agent",
        prompt=("You handle offers and discounts. Call exactly one tool: get_offer for "
                "offers, get_discounts for discounts. Reply with that tool's message in "
                "plain text only. Never print JSON, tool names, or raw structures. "
                "Never invent, guess, or summarise offers or discounts."),
    )

    supervisor = create_supervisor(
        agents=[menu_agent, complaint_agent, offer_agent],
        model=model,
        prompt=("You are the manager of Xander Restaurant support. "
                "Menu or price questions -> menu_agent. Complaints -> complaint_agent. "
                "Offer or discount questions -> offer_agent."
                "For greetings, answer yourself in one short line. "
                "After a worker agent answers, write nothing more: repeat the worker's answer exactly. "
                "Never invent offers, prices, or complaint IDs. If a worker gives no answer, "
                "say only: Sorry, I could not check that right now."),
    )
    return supervisor.compile()
