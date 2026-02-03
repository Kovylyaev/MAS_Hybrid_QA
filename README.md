# Многоагентная система для ответа на вопросы HybridQA

Прототип многоагентной системы отвечает на вопросы из датасета **HybridQA** (≈ 70 k Q‑A, 13 k таблиц + связанные passages Википедии) и выполняет **мульти‑хоп рассуждение по табличным и текстовым данным**.  
Система построена на базе [LangGraph](https://github.com/langchain-ai/langgraph) и развёртывается в Docker.

---

## Архитектура

Основная логика графа находится в `src/agent/graph.py` и использует три узла:

- `planner` → **Planner‑LLM (GPT‑4o)**  
  Разбивает задачу на шаги в стиле ReAct и решает, кого вызывать дальше


- `table_agent` → **Table‑Tool Agent**  
  Отвечает за **извлечение данных из таблиц** (HybridQA / WikiTables-WithLinks):

- `analysis_agent` → **Analysis‑Agent**  
  Задачи:
  - решает, **достаточно ли** текущих данных для ответа;
  - если **недостаточно**, вызывает:
    - `retrieve_tables(query)` — векторный поиск релевантных `table_uid` по естественному запросу;
    - `retrieve_wiki_passages(query)` — поиск релевантных текстов из Википедии;
  - формирует запросы к `table_agent` (какие таблицы и какие строки/ячейки извлекать);
  - считает метрики;
  - возвращает итоговый JSON:
    - `reasoning` — пошаговый ход рассуждений;
    - `functions_called` — `{function, arguments, result}` для проверки фактов;
    - `metrics` — рассчитанные величины;
    - `answer` — финальный текстовый ответ;
    - `sources` — использованные `table_uid`, ссылки на строки/ячейки и wiki‑passages.

Хранение данных и RAG‑поиск реализован в `src/agent/utils.py`:

- `Storage`:
  - при первом запуске клонирует `WikiTables-WithLinks` в `./data/hybrid_qa`;
  - предоставляет возможнность векторного поиска по таблицам и вики-пассажам

---

## Доступ к системе

Реализуется в интерфейсе LangGraph Studio по ссылке https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

---

## Локальный запуск

### Зависимости

- Python ≥ 3.11
- `uv`  
- Действующий `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY`

Установка:

```bash
cd MAS_Hybrid_QA
uv sync
source .venv/bin/activate
```

API ключи:

Скопируйте файл .env.example, переименуйте в .env и заполните поля: OPENAI_API_KEY, LANGSMITH_API_KEY, LANGCHAIN_API_KEY

### Запуск сервера в Docker

```bash
langgraph up
```

После этого API будет доступен по адресу `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123`.

---

## Поток выполнения запроса

1. Пользователь отправляет вопрос (и, при наличии, `table_uid`)
2. `planner` анализирует запрос и вызывает следующего агента
3. `analysis_agent` и `table_agent` работая по очереди собирают данные и анализируют их
4. После одного или нескольких циклов `analysis_agent` формирует финальный JSON‑ответ

## Пример запроса

- Who were the builders of the mosque in Herat with fire temples?


* Первый запрос может обрабатываться слегка долго, т.к. скачиваются все необходимые данные
