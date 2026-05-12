from __future__ import annotations

from pathlib import Path

import typer
from psycopg import OperationalError
from rich import print_json
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .lm_studio import LMStudioClient
from .prompts import SEED_CATEGORIES
from .service import CategorisationService
from .store import DEFAULT_DATABASE_URL, PostgresStore


app = typer.Typer(help="Prototype local LLM document categoriser.")
console = Console()


def database_url_option() -> str:
    return typer.Option(
        DEFAULT_DATABASE_URL,
        help="PostgreSQL URL. Can also be set with DOCUMIND_DATABASE_URL.",
    )


def make_service(database_url: str) -> CategorisationService:
    return CategorisationService(make_store(database_url), LMStudioClient())


def make_store(database_url: str) -> PostgresStore:
    try:
        return PostgresStore(database_url)
    except (OperationalError, RuntimeError) as exc:
        console.print(
            "[bold red]Could not connect to PostgreSQL.[/bold red]\n"
            f"Database URL: {database_url}\n"
            "Start PostgreSQL with pgvector enabled or set DOCUMIND_DATABASE_URL to the correct connection string.\n"
            f"Error: {exc}"
        )
        raise typer.Exit(code=1) from exc


@app.command()
def add_category(
    name: str,
    description: str,
    database_url: str = database_url_option(),
) -> None:
    service = make_service(database_url)
    category = service.add_category(name=name, description=description)
    console.print(f"Added category [bold]{category.name}[/bold] ({category.id})")


@app.command()
def seed_categories(
    database_url: str = database_url_option(),
) -> None:
    service = make_service(database_url)
    existing_names = {category.name.lower() for category in service.store.load_categories()}
    table = Table("Category", "Status", "ID")

    for seed_category in SEED_CATEGORIES:
        name = seed_category["name"]
        description = seed_category["description"]
        if name.lower() in existing_names:
            table.add_row(name, "already exists", "")
            continue
        category = service.add_category(name=name, description=description)
        existing_names.add(name.lower())
        table.add_row(category.name, "added", category.id)

    console.print(table)


@app.command()
def list_categories(
    database_url: str = database_url_option(),
) -> None:
    categories = make_store(database_url).load_categories()
    table = Table("ID", "Name", "Status", "Description")
    for category in categories:
        table.add_row(category.id, category.name, category.status, category.description)
    console.print(table)


@app.command()
def categorise(
    path: Path,
    database_url: str = database_url_option(),
) -> None:
    service = make_service(database_url)
    document = service.categorise_file(path)
    print_json(data=document.model_dump(mode="json"))


@app.command()
def categorise_folder(
    folder: Path,
    pattern: str = typer.Option("*.txt", help="File glob pattern to process."),
    recursive: bool = typer.Option(True, help="Search recursively."),
    database_url: str = database_url_option(),
) -> None:
    service = make_service(database_url)
    categories = {category.id: category.name for category in service.store.load_categories()}
    paths = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))

    if not paths:
        console.print(f"No files matched [bold]{pattern}[/bold] in {folder}")
        raise typer.Exit(code=1)

    table = Table("File", "Status", "Suggested Category", "Errors")
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Categorising documents", total=len(paths))
        for path in paths:
            progress.update(task, description=f"Categorising {path.name}")
            document = service.categorise_file(path)
            category_name = categories.get(
                document.primary_category_id,
                document.primary_category_id or "",
            )
            table.add_row(
                str(path),
                document.status,
                category_name,
                "; ".join(document.errors),
            )
            progress.advance(task)

    console.print(table)


@app.command()
def list_documents(
    database_url: str = database_url_option(),
) -> None:
    store = make_store(database_url)
    documents = store.load_documents()
    categories = {category.id: category.name for category in store.load_categories()}
    table = Table("ID", "Filename", "Status", "Category", "Category ID")
    for document in documents:
        category_name = categories.get(
            document.primary_category_id,
            "" if document.primary_category_id is None else "unknown",
        )
        table.add_row(
            document.id,
            document.filename,
            document.status,
            category_name,
            document.primary_category_id or "",
        )
    console.print(table)


if __name__ == "__main__":
    app()
