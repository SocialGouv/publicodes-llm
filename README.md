# publicodes-llm

⚠️ experimental

Utiliser un LLM (grand modèle de language) pour executer un modèle de calcul [publicodes](https://publi.codes).

Démo : https://publicodes-llm-preprod.dev.fabrique.social.gouv.fr

## A propos

Dans cette demo on utilise un LLM OpenAI pour réaliser le calcul d'un modèle [publicodes](https://publi.codes): la durée du préavis de retraite.

Le rôle du LLM est limité à reformuler les questions et interpretater les réponses de l'utilisateur.

Tout les calculs sont effectués par le moteur [publicodes](https://publi.codes).

## Stack:

- [llama_index](https://www.llamaindex.ai)
- OpenAI
- modèle [publicodes](https://publi.codes) du calcul de préavis de retraite
- [streamlit](https://streamlit.io) pour l'UI de démo

## Todo:

- detection/validation de CC
- recap parameters before results
- improve parameters matching/validation
- gestion initialisation des parametres
- lister les choix possibles quand il y des enums