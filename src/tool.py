# encoding: utf-8

import logging
import re
import os
import sys
import openai
from llama_index.agent import OpenAIAgent
from llama_index.tools.function_tool import FunctionTool
from llama_index.tools.types import ToolMetadata

from typing import TypedDict

from pydantic.v1 import create_model, Field

from publicodes import get_rule, map_value, evaluate

from kali import convention_collective_query_engine

if os.getenv("OPENAI_URL"):
    openai.api_base = os.getenv("OPENAI_URL")
    openai.verify_ssl_certs = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


logger = logging.getLogger()

# handler = logging.StreamHandler(stream=sys.stdout)
# handler.setFormatter(LlamaIndexFormatter())
# logger.addHandler(handler)


ParametresCalcul = TypedDict(
    "ParametresCalcul",
    {},
)


def toPublicodeKey(key):
    return key.replace("______", " . ").replace("___", " ")


def toUnderscoreKey(key):
    return key.replace(" . ", "______").replace(" ", "___")


PROMPT_CONSEILLER_PREAVIS = """
Tu es un assistant en phase de test en charge d'estimer la durée de préavis à respecter en cas de départ à la retraite ou de mise à la retraite de ton interlocuteur.

Tu ne sais poser aucune question. Les questions à poser à l'utilisateur sont données par la fonction get_next_question ou par la fonction get_convention_collective

Tu ne dois PAS utiliser des connaissances générales ni sur le code du travail ni sur les conventions colectives ou aucune règle de droit et tu ne dois pas calculer toi-même le préavis de retraite mais toujours utiliser la fonction get_next_question

Tu dois poser toutes les questions à l'utilisateur.

La fonction get_next_question te renvoie les questions que tu reformules à l'utilisateur. Tu ne dois pas poser d'autres questions que celles fournies par la fonction get_next_question.

Respectes bien la définition de la fonction get_next_question pour choisir le nom des paramètres de calcul à lui envoyer.

Si la fonction get_next_question renvoie un nombre:
    - affiche la liste des args utilisés pour get_next_question puis :
    - affiche le résultat en nombre de jours arrondi au jour inférieur, avec une estimation en nombre de mois
    - indiques le site du code du travail numérique: https://code.travail.gouv.fr/outils/preavis-retraite
    - remercie chaleureusement l'utilisateur

Si une fonction t'envoie un message commencant par "Assistant:", tu dois prendre en compte ses instructions
"""


def get_convention_collective(query: str):
    """Pour choisir une convention collective. Si plusieurs choix possibles, il faut demander à l'utilisateur de choisir le bon ID dans la liste"""
    response = convention_collective_query_engine.query(
        f'Quelles sont les conventions collectives liées à "{query}" ? Renvoies une liste des IDCC et le titre simplifié'
    )
    results = re.findall(r"(\d+),(.*)", str(response))
    if re.match("IDCC\\d+", query):
        return query
    if len(results) == 1:
        return f"IDCC{results[0][0]}"
    elif len(results) > 1:
        message = (
            "Assistant: Pour choisir sa convention collective, l'utilisateur doit choisir  parmi:\n"
            + "\n".join(map(lambda a: f" - IDCC{a[0]}: {a[1]}", results))
        )
        print(message)
        return f"{message}"
    return None


def update_tool(fn, fields: dict):
    """create a tool with custom fields schema"""
    print("fields")
    print(fields)
    name = fn.__name__
    docstring = fn.__doc__ or name
    fn_schema = create_model(
        name,
        **fields,
    )
    tool_metadata = ToolMetadata(name=name, description=docstring, fn_schema=fn_schema)
    logging.debug(f"set schema for {name}")
    logging.debug(str(fn_schema.schema()))
    return FunctionTool(fn=fn, metadata=tool_metadata)


# agent = OpenAIAgent.from_tools(
#     [get_next_question_tool],
#     verbose=True,
#     system_prompt=PROMPT_CONSEILLER_PREAVIS,
#     # llm=OpenAI(
#     #     model="gpt-3.5-turbo-0613",
#     #     temperature=0.1,
#     #     # openai_proxy="http://127.0.0.1:8084",
#     # ),
# )


#
# todo: this should be a OpenAIAgent subclass
#
class PublicodeAgent:
    def __init__(
        self,
    ):
        self.init_agent()

    def init_agent(self):
        # initialize the agent with some optional inputs
        self.get_next_question_tool = update_tool(
            self.get_next_question,
            fields={
                "contrat___salarié______ancienneté": Field(
                    0,
                    description="Ancienneté du salarié en mois",
                ),
                "contrat___salarié______travailleur___handicapé": Field(
                    False,
                    description="Le salarié est-il reconnu en tant que travail handicapé",
                ),
            },
        )
        get_convention_collective_tool = FunctionTool.from_defaults(
            fn=get_convention_collective,
            description="A utiliser pour identifier une convention collective",
        )
        self.agent = OpenAIAgent.from_tools(
            [self.get_next_question_tool, get_convention_collective_tool],
            verbose=True,
            system_prompt=PROMPT_CONSEILLER_PREAVIS,
            # llm=OpenAI(
            #     model="gpt-3.5-turbo-0613",
            #     temperature=0.1,
            #     # openai_proxy="http://127.0.0.1:8084",
            # ),
            # )
        )

    def chat(self, *args, **kwargs):
        return self.agent.chat(*args, **kwargs)

    def stream_chat(self, *args, **kwargs):
        return self.agent.stream_chat(*args, **kwargs)

    def chat_repl(self, *args, **kwargs):
        return self.agent.chat_repl(*args, **kwargs)

    def get_next_question(self, **parametres_calcul: ParametresCalcul) -> str | None:
        """
        Pour calculer le préavis de retraite. renvoie une question à poser à l'utilisateur ou un résultat en jours.

        Si je renvoie "None", ne pas donner de réponse à l'utilisateur
        """

        logger.debug("get_next_question")
        logger.debug(parametres_calcul or {})

        situation_publicodes = {
            toPublicodeKey(key): map_value(value)
            for (key, value) in parametres_calcul.items()
        }

        logger.debug("⚙️ publicodes")
        logger.debug(situation_publicodes)

        next_question, next_key, result = evaluate(
            "contrat salarié . préavis de retraite en jours", situation_publicodes
        )

        #
        # here we "infer" parameters types from publicodes definitions at runtime
        # and update the LLM function call signature
        # this looks necessary to enforce return types and parameter names
        #
        if next_question and next_key:
            parameters = situation_publicodes.copy()
            parameters[next_key] = ""
            # key type ?
            typed_parameters = {}
            for key in parameters.keys():
                rule = get_rule(key)
                description = rule.get("rawNode").get("question")
                node_type = rule.get("rawNode").get("cdtn", {}).get("type")
                if node_type == "oui-non":
                    typed_parameters[toUnderscoreKey(key)] = (
                        bool,
                        Field(description=description),
                    )
                elif node_type == "entier":
                    typed_parameters[toUnderscoreKey(key)] = (
                        int,
                        Field(description=description),
                    )
                elif node_type == "liste":
                    values: list[str] = list(
                        map(
                            lambda a: a.strip("'"),
                            rule.get("rawNode")
                            .get("cdtn", {})
                            .get("valeurs", {})
                            .values(),
                        )
                    )
                    if "oui" in values and "non" in values and len(values) == 2:
                        typed_parameters[toUnderscoreKey(key)] = (
                            bool,
                            Field(
                                description=description,
                            ),
                        )
                    else:
                        typed_parameters[toUnderscoreKey(key)] = (
                            str,  # todo: use typing unions from strings
                            Field(
                                description=description
                                + " Un choix parmi: "
                                + " ou ".join(map(lambda a: f"'{a}'", values)),
                                # enum force casting answers
                                json_schema_extra={
                                    "enum": values,
                                },
                            ),
                        )
                        if key == next_key:
                            next_question += "\nUn choix parmi: " + ", ".join(
                                map(lambda a: f"'{a}'", values)
                            )
                else:
                    typed_parameters[toUnderscoreKey(key)] = (
                        str,
                        Field(description=description),
                    )
            # necessary to monkey patch the tool signatures at the moment
            for i, tool in enumerate(self.agent._tools):
                if tool.metadata.name == "get_next_question":
                    params = typed_parameters or {}
                    logger.debug(
                        f"update tool {tool.metadata.name} with params: {params}"
                    )
                    self.agent._tools[i] = update_tool(self.get_next_question, params)

            return f'Assistant: demande à l\'utilisateur : "{next_question}"'
        elif result:
            return result
        return None


if __name__ == "__main__":
    print("\nex: Calcules moi mon préavis de retraite pour 24 mois d'ancienneté\n")
    agent = PublicodeAgent()
    agent.chat_repl()

# message = input("Human: ").encode("utf-8")  # "Calcules moi mon préavis de retraite"

# while message != "exit":
#     response = agent.chat(message)
#     print(f"Assistant: {response}\n")
#     message = input("Human: ").encode("utf-8")
