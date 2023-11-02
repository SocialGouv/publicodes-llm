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

## Dev:

Les règles publicodes sont définies dans [./publicodes-api/rules.json](./publicodes-api/rules.json)

```sh
# démarrer l'API publicodes
cd publicodes-api
yarn && yarn start

# démarrer l'app streamlit
OPENAI_TOKEN=xxx streamlit run src/run.py
```

Le proxy NGINX proposé sert à juste à améliorer les perfs :

```sh
docker run -p 8083:8080 -v $PWD/nginx.conf:/etc/nginx/conf.d/default.conf nginx:alpine3.18
```

La variable d'environnement `OPENAI_URL` permet de forcer le passage par votre proxy, ex: `OPENAI_URL=http://127.0.0.1:8083`.

## Todo:

- detection/validation de CC
- improve parameters matching/validation
