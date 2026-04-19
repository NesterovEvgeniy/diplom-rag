# Diplom RAG

Telegram-бот с AI-ассистентом на основе **RAG** для поиска и генерации ответов по закрытому корпусу текстовых источников.

Проект реализует полный пайплайн подготовки корпуса и ответа по документам:

**PDF → preprocessing → page texts → chunking → ingestion → retrieval → context → LLM → citations → source links → bot → evaluation**

В проекте реализованы три варианта RAG:

- **Naive RAG**
- **Hybrid RAG** (dense + sparse / BM25 + RRF)
- **Graph RAG** (расширение retrieval через граф связей между чанками)

---

## Возможности

- загрузка и хранение исходных PDF-источников;
- предобработка PDF с очисткой шумов, page numbers, headers/footers;
- page-local chunking с сохранением привязки к физическим страницам PDF;
- dense retrieval через Qdrant;
- sparse retrieval через BM25;
- fusion dense/sparse через RRF;
- graph-based retrieval через соседние чанки;
- генерация ответа с citations вида `[1]`, `[2]`, ...;
- построение source links на страницы PDF;
- Telegram-бот для запросов к корпусу;
- evaluation pipeline по gold-вопросам;
- ручной error analysis по результатам прогонов.

---

## Технологии

Основные зависимости проекта:

- Python 3.10+
- Qdrant
- MinIO / S3
- LM Studio / OpenAI-compatible API
- `httpx`
- `boto3`
- `qdrant-client`
- `pypdf`
- `aiogram`
- `rich`
- `pydantic`
- `pydantic-settings`

Зависимости и минимальная версия Python описаны в `pyproject.toml`.

---

## Схема структуры проекта

```text
.
├── configs/
│   └── prompts/
│       ├── rag_force_citations.txt
│       └── rag_system.txt
├── data/
│   ├── corpus/
│   │   └── sources.jsonl
│   ├── evaluation/
│   │   ├── questions_gold/
│   │   └── runs/
│   ├── manual_runs/
│   ├── processed/
│   │   ├── chunks/
│   │   │   ├── chunks.jsonl
│   │   │   └── chunk_stats.json
│   │   ├── graph/
│   │   │   ├── chunk_graph.jsonl
│   │   │   └── graph_stats.json
│   │   └── page_texts/
│   └── sources_raw/
├── docker/
│   ├── docker-compose.yml
│   └── init/
│       ├── minio/
│       └── qdrant/
├── reports/
│   └── pdf_checks/
├── scripts/
│   ├── analyze_baseline_run.py
│   ├── chunk_page_texts.py
│   ├── eval_baseline.py
│   ├── preprocess_pdf_pages.py
│   ├── graph/
│   │   ├── build_chunk_graph.py
│   │   └── eval_graph.py
│   ├── hybrid/
│   │   └── eval_hybrid.py
│   └── pdf_checks/
│       ├── check_pdf_quality.py
│       └── check_pdf_samples.py
├── src/
│   ├── settings.py
│   ├── apps/
│   │   ├── bot/
│   │   │   └── main.py
│   │   └── cli/
│   │       ├── main.py
│   │       └── commands/
│   │           ├── ask.py
│   │           ├── ask_log.py
│   │           ├── embed_test.py
│   │           ├── ingest_chunks.py
│   │           ├── llm_ask.py
│   │           ├── minio_init.py
│   │           ├── qdrant_init.py
│   │           ├── save_manifest.py
│   │           ├── search.py
│   │           ├── search_log.py
│   │           ├── sources_upload.py
│   │           ├── graph/
│   │           │   ├── ask_graph.py
│   │           │   └── search_graph.py
│   │           └── hybrid/
│   │               ├── ask_hybrid.py
│   │               └── search_hybrid.py
│   ├── rag/
│   │   ├── rag_settings.py
│   │   ├── common/
│   │   │   ├── citations.py
│   │   │   ├── embeddings.py
│   │   │   ├── llm.py
│   │   │   ├── logging.py
│   │   │   ├── prompts.py
│   │   │   ├── refusals.py
│   │   │   └── text_utils.py
│   │   ├── graph_rag/
│   │   │   ├── graph_store.py
│   │   │   ├── pipeline.py
│   │   │   └── retrieval.py
│   │   ├── hybrid_rag/
│   │   │   ├── bm25.py
│   │   │   ├── fusion.py
│   │   │   ├── pipeline.py
│   │   │   ├── retrieval.py
│   │   │   └── sparse_index.py
│   │   └── naive_rag/
│   │       ├── context.py
│   │       ├── generation.py
│   │       ├── ingestion.py
│   │       ├── pipeline.py
│   │       └── retrieval.py
│   └── storage/
│       └── source_links.py
├── .env
└── pyproject.toml
```

### Кратко по папкам

- `configs/prompts` — prompt-шаблоны для генерации ответов.
- `data/sources_raw` — исходные PDF-файлы корпуса.
- `data/processed/page_texts` — очищенные тексты страниц после preprocessing.
- `data/processed/chunks` — готовые чанки для ingestion и retrieval.
- `data/processed/graph` — граф связей между чанками для Graph RAG.
- `data/corpus/sources.jsonl` — реестр загруженных источников и их метаданных.
- `data/evaluation/questions_gold` — gold-вопросы для оценки качества.
- `data/evaluation/runs` — результаты evaluation-запусков.
- `data/manual_runs` — ручные логи search/ask запусков.
- `reports/pdf_checks` — отчёты по качеству PDF и preprocessing.
- `scripts` — отдельные сценарии подготовки корпуса, evaluation и анализа.
- `src/apps` — Telegram-бот и CLI-интерфейс.
- `src/rag/common` — общие модули: embeddings, llm, citations, prompts и т.д.
- `src/rag/naive_rag` — реализация Naive RAG.
- `src/rag/hybrid_rag` — реализация Hybrid RAG.
- `src/rag/graph_rag` — реализация Graph RAG.
- `src/storage` — построение ссылок на источники.

---

## Архитектура

### 1. Подготовка корпуса

Исходные PDF-файлы помещаются в:

```text
data/sources_raw/
```

Далее выполняются этапы:

1. **Проверка PDF**
   - `scripts/pdf_checks/check_pdf_quality.py`
   - `scripts/pdf_checks/check_pdf_samples.py`

2. **Предобработка PDF**
   - `scripts/preprocess_pdf_pages.py`
   - результат: `data/processed/page_texts/*.jsonl`

3. **Чанкирование**
   - `scripts/chunk_page_texts.py`
   - результат: `data/processed/chunks/chunks.jsonl`

4. **Построение графа чанков**
   - `scripts/graph/build_chunk_graph.py`
   - результат: `data/processed/graph/chunk_graph.jsonl`

### 2. Retrieval / Generation

#### Naive RAG
- dense retrieval из Qdrant;
- сбор контекста;
- генерация ответа через LLM;
- нормализация citations;
- возврат структурированного ответа с источниками.

#### Hybrid RAG
- dense retrieval;
- sparse retrieval через BM25;
- объединение результатов через RRF;
- генерация ответа и сбор источников.

#### Graph RAG
- dense seed retrieval;
- расширение результата соседями из графа чанков;
- генерация ответа по расширенному контексту.

### 3. Интерфейсы

#### CLI
Основная точка входа:

```bash
python -m src.apps.cli.main <command>
```

#### Telegram Bot
Точка входа:

```bash
python -m src.apps.cli.main bot
```

---

## Установка проекта

### Требования

Перед установкой должны быть доступны:

- Python **3.10+**
- работающий **Qdrant**
- работающий **MinIO / S3-compatible storage**
- LLM API, совместимый с OpenAI `/chat/completions`
- Embeddings API, совместимый с OpenAI `/embeddings`

### 1. Клонирование репозитория

```powershell
git clone <your-repo-url>
cd diplom-rag
```

### 2. Создание виртуального окружения

```powershell
python -m venv .venv
.\.venv\scripts\activate
```

### 3. Установка зависимостей

```powershell
pip install -e .
```

Если хочешь сначала обновить инструменты установки:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

### 4. Настройка `.env`

Создай файл `.env` в корне проекта.

Пример:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=chunks_ru

S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_REGION=us-east-1
S3_BUCKET_SOURCES=sources
S3_BUCKET_ARTIFACTS=artifacts
S3_PUBLIC_BASE_URL=

LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=
LLM_MODEL=local-model

EMBED_BASE_URL=http://localhost:1234/v1
EMBED_API_KEY=lm-studio
EMBED_MODEL=text-embedding-model
EMBED_DIM=768

APP_ENV=local
RAG_MODE=naive
TELEGRAM_BOT_TOKEN=
BOT_TOP_K=5

STRICT_CITATIONS=false
```

---

## Быстрый старт

### 1. Проверка доступности сервисов

```powershell
python -m src.apps.cli.main ping
```

### 2. Инициализация MinIO buckets

```powershell
python -m src.apps.cli.main minio-init
```

### 3. Инициализация коллекции Qdrant

```powershell
python -m src.apps.cli.main qdrant-init
```

Для пересоздания коллекции:

```powershell
python -m src.apps.cli.main qdrant-init --recreate
```

### 4. Проверка embeddings

```powershell
python -m src.apps.cli.main embed-test "тестовый текст"
```

### 5. Загрузка исходных PDF в S3/MinIO

```powershell
python -m src.apps.cli.main sources-upload data/sources_raw
```

### 6. Проверка качества PDF

```powershell
python scripts/pdf_checks/check_pdf_quality.py
python scripts/pdf_checks/check_pdf_samples.py
```

### 7. Preprocessing страниц

```powershell
python scripts/preprocess_pdf_pages.py
```

### 8. Чанкирование

```powershell
python scripts/chunk_page_texts.py
```

### 9. Построение графа чанков

```powershell
python scripts/graph/build_chunk_graph.py
```

### 10. Ingestion в Qdrant

```powershell
python -m src.apps.cli.main ingest-chunks data/processed/chunks/chunks.jsonl
```

### 11. Проверка поиска и ответов

#### Naive RAG

```powershell
python -m src.apps.cli.main search "что такое ароматерапия" --k 5
python -m src.apps.cli.main ask "что такое ароматерапия" --k 5
```

#### Hybrid RAG

```powershell
python -m src.apps.cli.main search_hybrid "что такое ароматерапия" --k 5
python -m src.apps.cli.main ask_hybrid "что такое ароматерапия" --k 5
```

#### Graph RAG

```powershell
python -m src.apps.cli.main search_graph "что такое ароматерапия" --k 5
python -m src.apps.cli.main ask_graph "что такое ароматерапия" --k 5
```

### 12. Запуск Telegram-бота

```powershell
python -m src.apps.cli.main bot
```

---

## Evaluation

Gold-вопросы лежат в:

```text
data/evaluation/questions_gold/
```

Результаты прогонов сохраняются в:

```text
data/evaluation/runs/
```

### Naive RAG

```powershell
python scripts/eval_baseline.py
```

### Hybrid RAG

```powershell
python scripts/hybrid/eval_hybrid.py
```

### Graph RAG

```powershell
python scripts/graph/eval_graph.py
```

### Постобработка run-файла

```powershell
python scripts/analyze_baseline_run.py --input data/evaluation/runs/<run>.jsonl
```

---

## Данные проекта

### Исходные документы
```text
data/sources_raw/
```

### Реестр источников
```text
data/corpus/sources.jsonl
```

### Очищенные тексты страниц
```text
data/processed/page_texts/
```

### Чанки
```text
data/processed/chunks/
```

### Граф чанков
```text
data/processed/graph/
```

### Отчёты по PDF
```text
reports/pdf_checks/
```

---

## Принципы проекта

- корпус **закрытый**: ответы строятся только по заранее загруженным источникам;
- retrieval и generation должны быть **source-aware**;
- важна **page-level трассировка** до PDF;
- citations должны быть проверяемыми и привязанными к retrieved chunks;
- evaluation проводится на gold-вопросах по нескольким режимам RAG.

---


## License

```text
MIT
```

или

```text
All rights reserved
```

---

## Автор

**Evgenii Nesterov**
